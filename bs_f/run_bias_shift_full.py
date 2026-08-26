#!/usr/bin/env python3
"""Bias-Shift Full (BS-F) implementation.

The method follows this sequence:

1. Transport source residuals and source SAE encoder weights into the target
   residual basis.
2. Compute direct source-to-target post-ReLU feature similarities.
3. Keep only confident source matches using the top-1 score, top-1/top-2
   margin, and sample-dead gate.
4. Estimate a feature-local pre-activation offset at the component-wise
   median transported residual.
5. Clip and add that offset to the source SAE bias.
6. Recompute calibrated similarities and activation-frequency-weighted
   bidirectional greedy Precision, Recall, and F1.

The implementation operates on prepared NumPy arrays, separating model and
data preparation from BS-F scoring.

Input directory
---------------
The command-line runner expects these files:

    H_a.npy                  source residuals, shape (T, d_a)
    alignment_matrix.npy     source-to-target map, shape (d_a, d_b)
    W_a.npy                  source SAE encoder, shape (d_a, n_a)
    b_a.npy                  source SAE bias, shape (n_a,)
    W_b.npy                  target SAE encoder, shape (d_b, n_b)
    b_b.npy                  target SAE bias, shape (n_b,)

Optional files:

    feature_weights_a.npy    source activation frequencies, shape (n_a,)
    feature_weights_b.npy    target activation frequencies, shape (n_b,)

Feature-frequency weights are optional and default to uniform values. The input
alignment matrix represents the requested direction. The
``--transpose-alignment`` option supports maps stored in reverse orientation.

Example
-------
    python run_bias_shift_full.py \
        --input-dir prepared/base_to_instruct \
        --output-dir outputs/bs_f/base_to_instruct

The default numerical settings define the BS-F Full configuration:

    median centroid, top-1 > 0.10, margin >= 0.05,
    clip = 4.0, sample-dead threshold = 1e-6,
    native-dead frequency threshold = 1e-7.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


LOGGER = logging.getLogger("bias_shift_full")


@dataclass(frozen=True)
class BiasShiftFullConfig:
    """Numerical controls for the stable BS-F Full method."""

    steer_threshold: float = 0.10
    margin_threshold: float = 0.05
    delta_clip: float = 4.0
    sample_dead_threshold: float = 1e-6
    native_dead_threshold: float = 1e-7
    feature_chunk_size: int = 256
    token_chunk_size: int = 1024

    def validate(self) -> None:
        if self.steer_threshold < 0:
            raise ValueError("steer_threshold must be non-negative")
        if self.margin_threshold < 0:
            raise ValueError("margin_threshold must be non-negative")
        if self.delta_clip <= 0:
            raise ValueError("delta_clip must be positive")
        if self.sample_dead_threshold < 0:
            raise ValueError("sample_dead_threshold must be non-negative")
        if self.native_dead_threshold < 0:
            raise ValueError("native_dead_threshold must be non-negative")
        if self.feature_chunk_size <= 0:
            raise ValueError("feature_chunk_size must be positive")
        if self.token_chunk_size <= 0:
            raise ValueError("token_chunk_size must be positive")


@dataclass
class PostBCache:
    """Disk-backed target post-activation cache."""

    path: Path
    shape: tuple[int, int]
    denom_b: np.ndarray


@dataclass
class SimilarityPass:
    """Row/column reductions of a feature-similarity matrix."""

    max_sim_a: np.ndarray
    second_sim_a: np.ndarray
    max_sim_b: np.ndarray
    argmax_a: np.ndarray
    dead_a: np.ndarray
    dead_b: np.ndarray


REQUIRED_INPUTS = {
    "H_a": "H_a.npy",
    "M": "alignment_matrix.npy",
    "W_a": "W_a.npy",
    "b_a": "b_a.npy",
    "W_b": "W_b.npy",
    "b_b": "b_b.npy",
}

OPTIONAL_INPUTS = {
    "weights_a": "feature_weights_a.npy",
    "weights_b": "feature_weights_b.npy",
}


def _as_float32(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must have a floating dtype, got {array.dtype}")
    return array.astype(np.float32, copy=False)


def _validate_inputs(
    H_a: np.ndarray,
    M: np.ndarray,
    W_a: np.ndarray,
    b_a: np.ndarray,
    W_b: np.ndarray,
    b_b: np.ndarray,
    weights_a: np.ndarray | None,
    weights_b: np.ndarray | None,
) -> None:
    if H_a.ndim != 2:
        raise ValueError(f"H_a must be 2-D, got shape {H_a.shape}")
    if M.ndim != 2:
        raise ValueError(f"M must be 2-D, got shape {M.shape}")
    if W_a.ndim != 2 or W_b.ndim != 2:
        raise ValueError("W_a and W_b must both be 2-D")
    if b_a.ndim != 1 or b_b.ndim != 1:
        raise ValueError("b_a and b_b must both be 1-D")
    if H_a.shape[0] == 0:
        raise ValueError("H_a must contain at least one scoring row")
    if W_a.shape[1] == 0 or W_b.shape[1] == 0:
        raise ValueError("both SAE dictionaries must contain at least one feature")
    if H_a.shape[1] != M.shape[0]:
        raise ValueError(
            "source residual dimension mismatch: "
            f"H_a.shape={H_a.shape}, M.shape={M.shape}"
        )
    if W_a.shape[0] != H_a.shape[1]:
        raise ValueError(
            "source encoder dimension mismatch: "
            f"H_a.shape={H_a.shape}, W_a.shape={W_a.shape}"
        )
    if W_b.shape[0] != M.shape[1]:
        raise ValueError(
            "target encoder dimension mismatch: "
            f"M.shape={M.shape}, W_b.shape={W_b.shape}"
        )
    if b_a.shape != (W_a.shape[1],):
        raise ValueError(
            f"b_a must have shape ({W_a.shape[1]},), got {b_a.shape}"
        )
    if b_b.shape != (W_b.shape[1],):
        raise ValueError(
            f"b_b must have shape ({W_b.shape[1]},), got {b_b.shape}"
        )

    for name, weights, expected in (
        ("weights_a", weights_a, W_a.shape[1]),
        ("weights_b", weights_b, W_b.shape[1]),
    ):
        if weights is None:
            continue
        if weights.shape != (expected,):
            raise ValueError(
                f"{name} must have shape ({expected},), got {weights.shape}"
            )
        if not np.isfinite(weights).all():
            raise ValueError(f"{name} contains non-finite values")
        if np.any(weights < 0):
            raise ValueError(f"{name} must be non-negative")


def load_prepared_inputs(input_dir: Path) -> dict[str, np.ndarray | None]:
    """Load BS-F arrays from an input directory."""

    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")

    arrays: dict[str, np.ndarray | None] = {}
    missing: list[str] = []
    for key, filename in REQUIRED_INPUTS.items():
        path = input_dir / filename
        if not path.is_file():
            missing.append(filename)
            continue
        arrays[key] = np.load(path, mmap_mode="r", allow_pickle=False)

    if missing:
        formatted = ", ".join(missing)
        raise FileNotFoundError(
            f"missing required input files in {input_dir}: {formatted}"
        )

    for key, filename in OPTIONAL_INPUTS.items():
        path = input_dir / filename
        arrays[key] = (
            np.load(path, mmap_mode="r", allow_pickle=False)
            if path.is_file()
            else None
        )
    return arrays


def build_post_b_cache(
    H_prime: np.ndarray,
    W_b: np.ndarray,
    b_b: np.ndarray,
    *,
    token_chunk_size: int,
    work_dir: Path,
) -> PostBCache:
    """Build target post-ReLU activations once in a disk-backed array."""

    n_tokens = H_prime.shape[0]
    n_b = W_b.shape[1]
    work_dir.mkdir(parents=True, exist_ok=True)
    file_descriptor, raw_path = tempfile.mkstemp(
        prefix="bsf_post_b_",
        suffix=".mmap",
        dir=work_dir,
    )
    os.close(file_descriptor)
    path = Path(raw_path)
    post_b_map = np.memmap(
        path,
        dtype=np.float32,
        mode="w+",
        shape=(n_tokens, n_b),
    )
    sumsq_b = np.zeros(n_b, dtype=np.float64)

    try:
        for start in range(0, n_tokens, token_chunk_size):
            end = min(start + token_chunk_size, n_tokens)
            post_b_batch = np.maximum(
                H_prime[start:end] @ W_b + b_b[None, :],
                0.0,
            ).astype(np.float32)
            post_b_map[start:end] = post_b_batch
            sumsq_b += np.sum(
                post_b_batch * post_b_batch,
                axis=0,
                dtype=np.float64,
            )
        post_b_map.flush()
    except Exception:
        del post_b_map
        path.unlink(missing_ok=True)
        raise
    else:
        del post_b_map

    denom_b = np.sqrt(np.maximum(sumsq_b / n_tokens, 1e-30))
    return PostBCache(path=path, shape=(n_tokens, n_b), denom_b=denom_b)


def remove_post_b_cache(cache: PostBCache) -> None:
    cache.path.unlink(missing_ok=True)


def compute_similarity_pass(
    H_prime: np.ndarray,
    W_a_prime: np.ndarray,
    b_a: np.ndarray,
    *,
    post_b_cache: PostBCache,
    feature_chunk_size: int,
    token_chunk_size: int,
    dead_threshold: float,
) -> SimilarityPass:
    """Compute memory-safe feature cosine maxima and source top-2 scores.

    The full n_a by n_b matrix is never retained. Each source-feature chunk is
    reduced immediately to source maxima, source second-best values, source
    argmax indices, and running target maxima.
    """

    n_tokens = H_prime.shape[0]
    n_a = W_a_prime.shape[1]
    n_b = post_b_cache.shape[1]
    denom_b = post_b_cache.denom_b
    dead_b = denom_b < dead_threshold
    post_b_map = np.memmap(
        post_b_cache.path,
        dtype=np.float32,
        mode="r",
        shape=post_b_cache.shape,
    )

    max_sim_a = np.full(n_a, -np.inf, dtype=np.float64)
    second_sim_a = np.full(n_a, -np.inf, dtype=np.float64)
    max_sim_b = np.full(n_b, -np.inf, dtype=np.float64)
    argmax_a = np.zeros(n_a, dtype=np.int64)
    dead_a = np.zeros(n_a, dtype=bool)

    try:
        n_chunks = (n_a + feature_chunk_size - 1) // feature_chunk_size
        for chunk_index, start in enumerate(
            range(0, n_a, feature_chunk_size),
            start=1,
        ):
            end = min(start + feature_chunk_size, n_a)
            width = end - start
            sumsq_a = np.zeros(width, dtype=np.float64)
            cross = np.zeros((width, n_b), dtype=np.float64)

            for token_start in range(0, n_tokens, token_chunk_size):
                token_end = min(token_start + token_chunk_size, n_tokens)
                post_a_batch = np.maximum(
                    H_prime[token_start:token_end]
                    @ W_a_prime[:, start:end]
                    + b_a[None, start:end],
                    0.0,
                ).astype(np.float32)
                post_b_batch = np.asarray(
                    post_b_map[token_start:token_end],
                    dtype=np.float32,
                )
                sumsq_a += np.sum(
                    post_a_batch * post_a_batch,
                    axis=0,
                    dtype=np.float64,
                )
                cross += (post_a_batch.T @ post_b_batch).astype(np.float64)

            denom_a = np.sqrt(np.maximum(sumsq_a / n_tokens, 1e-30))
            dead_a_chunk = denom_a < dead_threshold
            dead_a[start:end] = dead_a_chunk

            similarity = (cross / n_tokens) / np.maximum(
                denom_a[:, None] * denom_b[None, :],
                1e-30,
            )
            similarity[
                dead_a_chunk[:, None] | dead_b[None, :]
            ] = 0.0
            similarity = np.clip(similarity, -1.0, 1.0).astype(np.float32)

            top1 = np.max(similarity, axis=1)
            best_index = np.argmax(similarity, axis=1)
            if n_b >= 2:
                top2_unsorted = np.partition(
                    similarity,
                    -2,
                    axis=1,
                )[:, -2:]
                top2 = np.minimum(
                    top2_unsorted[:, 0],
                    top2_unsorted[:, 1],
                )
            else:
                top2 = np.zeros(width, dtype=np.float32)

            max_sim_a[start:end] = top1
            second_sim_a[start:end] = top2
            argmax_a[start:end] = best_index
            max_sim_b = np.maximum(
                max_sim_b,
                np.max(similarity, axis=0),
            )

            LOGGER.info(
                "similarity source chunks: %d/%d",
                chunk_index,
                n_chunks,
            )

        return SimilarityPass(
            max_sim_a=max_sim_a,
            second_sim_a=second_sim_a,
            max_sim_b=max_sim_b,
            argmax_a=argmax_a,
            dead_a=dead_a,
            dead_b=dead_b,
        )
    finally:
        del post_b_map


def compute_full_bias_correction(
    H_prime: np.ndarray,
    W_a_prime: np.ndarray,
    b_a: np.ndarray,
    W_b: np.ndarray,
    b_b: np.ndarray,
    baseline: SimilarityPass,
    config: BiasShiftFullConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the stable median, margin-gated, clipped BS-F correction."""

    centroid = np.median(H_prime, axis=0).astype(np.float64)
    pre_a = centroid @ W_a_prime + b_a
    pre_b = centroid @ W_b + b_b
    margins = baseline.max_sim_a - baseline.second_sim_a
    steer_mask = (
        (baseline.max_sim_a > config.steer_threshold)
        & (margins >= config.margin_threshold)
        & (~baseline.dead_a)
    )
    raw_delta = pre_b[baseline.argmax_a] - pre_a
    delta = np.where(steer_mask, raw_delta, 0.0)
    delta = np.where(
        steer_mask,
        np.clip(delta, -config.delta_clip, config.delta_clip),
        0.0,
    )
    return steer_mask, delta, margins


def _weights_or_uniform(
    weights: np.ndarray | None,
    n_features: int,
) -> np.ndarray:
    if weights is None:
        return np.ones(n_features, dtype=np.float64)
    return np.asarray(weights, dtype=np.float64).copy()


def compute_weighted_f1(
    max_sim_a: np.ndarray,
    max_sim_b: np.ndarray,
    weights_a: np.ndarray,
    weights_b: np.ndarray,
) -> dict[str, float]:
    """Activation-frequency-weighted bidirectional greedy aggregation."""

    weight_sum_a = float(weights_a.sum())
    weight_sum_b = float(weights_b.sum())
    if weight_sum_a < 1e-30 or weight_sum_b < 1e-30:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = float(np.sum(max_sim_a * weights_a) / weight_sum_a)
    recall = float(np.sum(max_sim_b * weights_b) / weight_sum_b)
    denominator = precision + recall
    f1 = (
        float(2.0 * precision * recall / denominator)
        if denominator > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_v4_metrics(
    similarity: SimilarityPass,
    *,
    weights_a: np.ndarray | None,
    weights_b: np.ndarray | None,
    native_dead_threshold: float,
) -> dict[str, float | int]:
    """Apply the v4 native-dead scoring rule."""

    wa = _weights_or_uniform(weights_a, len(similarity.max_sim_a))
    wb = _weights_or_uniform(weights_b, len(similarity.max_sim_b))
    native_dead_a = wa < native_dead_threshold
    native_dead_b = wb < native_dead_threshold
    m_killed_a = similarity.dead_a & (~native_dead_a)
    m_killed_b = similarity.dead_b & (~native_dead_b)
    wa[native_dead_a] = 0.0
    wb[native_dead_b] = 0.0

    metrics: dict[str, float | int] = compute_weighted_f1(
        similarity.max_sim_a,
        similarity.max_sim_b,
        wa,
        wb,
    )
    metrics.update(
        {
            "sample_dead_a": int(similarity.dead_a.sum()),
            "sample_dead_b": int(similarity.dead_b.sum()),
            "native_dead_a": int(native_dead_a.sum()),
            "native_dead_b": int(native_dead_b.sum()),
            "m_killed_a": int(m_killed_a.sum()),
            "m_killed_b": int(m_killed_b.sum()),
            "m_killed_a_ratio": float(
                m_killed_a.sum() / max((~native_dead_a).sum(), 1)
            ),
            "m_killed_b_ratio": float(
                m_killed_b.sum() / max((~native_dead_b).sum(), 1)
            ),
        }
    )
    return metrics


def run_bias_shift_full(
    H_a: np.ndarray,
    M: np.ndarray,
    W_a: np.ndarray,
    b_a: np.ndarray,
    W_b: np.ndarray,
    b_b: np.ndarray,
    *,
    weights_a: np.ndarray | None = None,
    weights_b: np.ndarray | None = None,
    config: BiasShiftFullConfig | None = None,
    work_dir: Path | str = Path("."),
) -> dict[str, Any]:
    """Run the complete stable BS-F pipeline on prepared arrays."""

    config = config or BiasShiftFullConfig()
    config.validate()
    H_a = _as_float32("H_a", H_a)
    M = _as_float32("M", M)
    W_a = _as_float32("W_a", W_a)
    b_a = _as_float32("b_a", b_a)
    W_b = _as_float32("W_b", W_b)
    b_b = _as_float32("b_b", b_b)
    weights_a = (
        None
        if weights_a is None
        else np.asarray(weights_a, dtype=np.float64)
    )
    weights_b = (
        None
        if weights_b is None
        else np.asarray(weights_b, dtype=np.float64)
    )
    _validate_inputs(
        H_a,
        M,
        W_a,
        b_a,
        W_b,
        b_b,
        weights_a,
        weights_b,
    )

    if weights_a is None or weights_b is None:
        LOGGER.info(
            "using uniform feature-frequency weights for each unspecified side"
        )

    start_time = time.time()
    LOGGER.info("transporting residuals and source encoder")
    H_prime = np.asarray(H_a @ M, dtype=np.float32)
    W_a_prime = np.asarray(M.T @ W_a, dtype=np.float32)
    cache = build_post_b_cache(
        H_prime,
        W_b,
        b_b,
        token_chunk_size=config.token_chunk_size,
        work_dir=Path(work_dir),
    )

    try:
        LOGGER.info("computing uncalibrated BS-F similarities")
        baseline = compute_similarity_pass(
            H_prime,
            W_a_prime,
            b_a,
            post_b_cache=cache,
            feature_chunk_size=config.feature_chunk_size,
            token_chunk_size=config.token_chunk_size,
            dead_threshold=config.sample_dead_threshold,
        )
        steer_mask, delta, margins = compute_full_bias_correction(
            H_prime,
            W_a_prime,
            b_a,
            W_b,
            b_b,
            baseline,
            config,
        )
        b_a_shifted = b_a + delta
        LOGGER.info(
            "recomputing calibrated BS-F similarities for %d shifted features",
            int(steer_mask.sum()),
        )
        calibrated = compute_similarity_pass(
            H_prime,
            W_a_prime,
            b_a_shifted,
            post_b_cache=cache,
            feature_chunk_size=config.feature_chunk_size,
            token_chunk_size=config.token_chunk_size,
            dead_threshold=config.sample_dead_threshold,
        )
    finally:
        remove_post_b_cache(cache)

    baseline_metrics = compute_v4_metrics(
        baseline,
        weights_a=weights_a,
        weights_b=weights_b,
        native_dead_threshold=config.native_dead_threshold,
    )
    calibrated_metrics = compute_v4_metrics(
        calibrated,
        weights_a=weights_a,
        weights_b=weights_b,
        native_dead_threshold=config.native_dead_threshold,
    )
    shifted_delta = delta[steer_mask]
    summary = {
        "method": "Bias-Shift Full (BS-F)",
        "implementation": "Bias-Shift Full",
        "config": asdict(config),
        "shapes": {
            "n_tokens": int(H_a.shape[0]),
            "source_residual_dim": int(H_a.shape[1]),
            "target_residual_dim": int(M.shape[1]),
            "source_features": int(W_a.shape[1]),
            "target_features": int(W_b.shape[1]),
        },
        "weighting": {
            "source": (
                "activation_frequency"
                if weights_a is not None
                else "uniform"
            ),
            "target": (
                "activation_frequency"
                if weights_b is not None
                else "uniform"
            ),
        },
        "calibration": {
            "n_shifted": int(steer_mask.sum()),
            "shifted_fraction": float(steer_mask.mean()),
            "delta_mean": (
                float(shifted_delta.mean())
                if shifted_delta.size
                else 0.0
            ),
            "delta_std": (
                float(shifted_delta.std())
                if shifted_delta.size
                else 0.0
            ),
            "delta_min": (
                float(shifted_delta.min())
                if shifted_delta.size
                else 0.0
            ),
            "delta_max": (
                float(shifted_delta.max())
                if shifted_delta.size
                else 0.0
            ),
        },
        "baseline": baseline_metrics,
        "calibrated": calibrated_metrics,
        "delta_f1": float(
            calibrated_metrics["f1"] - baseline_metrics["f1"]
        ),
        "elapsed_seconds": float(time.time() - start_time),
    }
    arrays = {
        "max_sim_a_emp": baseline.max_sim_a,
        "max_sim_b_emp": baseline.max_sim_b,
        "argmax_a_emp": baseline.argmax_a,
        "top2_sim_a_emp": baseline.second_sim_a,
        "dead_a_emp": baseline.dead_a,
        "dead_b_emp": baseline.dead_b,
        "margins": margins,
        "steer_mask": steer_mask,
        "delta": delta,
        "b_a_shifted": b_a_shifted,
        "max_sim_a_calibrated": calibrated.max_sim_a,
        "max_sim_b_calibrated": calibrated.max_sim_b,
        "argmax_a_calibrated": calibrated.argmax_a,
        "dead_a_calibrated": calibrated.dead_a,
        "dead_b_calibrated": calibrated.dead_b,
    }
    return {"summary": summary, "arrays": arrays}


def save_outputs(result: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "bias_shift_full_summary.json"
    arrays_path = output_dir / "bias_shift_full_arrays.npz"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(result["summary"], handle, indent=2)
        handle.write("\n")
    np.savez_compressed(arrays_path, **result["arrays"])
    return summary_path, arrays_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Bias-Shift Full implementation"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the required prepared .npy arrays",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/bias_shift_full"),
    )
    parser.add_argument(
        "--transpose-alignment",
        action="store_true",
        help="Use the transpose of alignment_matrix.npy",
    )
    parser.add_argument(
        "--steer-threshold",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=0.05,
    )
    parser.add_argument("--delta-clip", type=float, default=4.0)
    parser.add_argument(
        "--sample-dead-threshold",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--native-dead-threshold",
        type=float,
        default=1e-7,
    )
    parser.add_argument(
        "--feature-chunk-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--token-chunk-size",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    inputs = load_prepared_inputs(args.input_dir)
    if args.transpose_alignment:
        inputs["M"] = np.asarray(inputs["M"]).T
    config = BiasShiftFullConfig(
        steer_threshold=args.steer_threshold,
        margin_threshold=args.margin_threshold,
        delta_clip=args.delta_clip,
        sample_dead_threshold=args.sample_dead_threshold,
        native_dead_threshold=args.native_dead_threshold,
        feature_chunk_size=args.feature_chunk_size,
        token_chunk_size=args.token_chunk_size,
    )
    result = run_bias_shift_full(
        inputs["H_a"],
        inputs["M"],
        inputs["W_a"],
        inputs["b_a"],
        inputs["W_b"],
        inputs["b_b"],
        weights_a=inputs["weights_a"],
        weights_b=inputs["weights_b"],
        config=config,
        work_dir=args.output_dir,
    )
    summary_path, arrays_path = save_outputs(result, args.output_dir)
    summary = result["summary"]
    print(
        "BS-F complete: "
        f"baseline_f1={summary['baseline']['f1']:.6f}, "
        f"calibrated_f1={summary['calibrated']['f1']:.6f}, "
        f"delta_f1={summary['delta_f1']:+.6f}"
    )
    print(f"summary: {summary_path.resolve()}")
    print(f"arrays:  {arrays_path.resolve()}")


if __name__ == "__main__":
    main()
