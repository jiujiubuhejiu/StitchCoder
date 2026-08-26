#!/usr/bin/env python3
"""Bias-Shift Ridge (BS-R) implementation.

The input directory contains native post-ReLU source/target SAE activations
at aligned token or word positions. A sequence-level split assigns complete
documents to either ridge fitting or evaluation.

Required files
--------------
``post_a.npy``
    Source SAE activations, shape ``(T, n_a)``.
``post_b.npy``
    Target SAE activations, shape ``(T, n_b)``.
``sequence_ids.npy``
    Document index for every aligned row, shape ``(T,)``.

The default configuration uses an unregularized intercept, ridge ``lambda=100``,
an 80/20 sequence split, ReLU after reconstruction, target training-frequency
weights, and native/sample dead thresholds of ``1e-7``/``1e-6``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from common.io_utils import write_json
from common.metrics import (
    activation_frequencies,
    compute_v4_metrics,
    cosine_similarity_reductions,
    self_slot_recovery,
)


LOGGER = logging.getLogger("bias_shift_ridge")


@dataclass(frozen=True)
class BiasShiftRidgeConfig:
    ridge_lambda: float = 100.0
    train_fraction: float = 0.8
    seed: int = 0
    sample_dead_threshold: float = 1e-6
    native_dead_threshold: float = 1e-7
    fit_row_chunk_size: int = 4096
    solve_column_chunk_size: int = 4096
    apply_row_chunk_size: int = 4096
    similarity_feature_chunk_size: int = 512
    similarity_row_chunk_size: int = 8192
    backend: str = "auto"
    device: str = "cuda"
    row_shuffle: bool = False
    source_feature_limit: int | None = None

    def validate(self) -> None:
        if self.ridge_lambda < 0:
            raise ValueError("ridge_lambda must be non-negative")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1")
        if self.sample_dead_threshold < 0 or self.native_dead_threshold < 0:
            raise ValueError("dead thresholds must be non-negative")
        if self.backend not in {"auto", "numpy", "cuda"}:
            raise ValueError("backend must be auto, numpy, or cuda")
        for value in (
            self.fit_row_chunk_size,
            self.solve_column_chunk_size,
            self.apply_row_chunk_size,
            self.similarity_feature_chunk_size,
            self.similarity_row_chunk_size,
        ):
            if value <= 0:
                raise ValueError("chunk sizes must be positive")
        if self.source_feature_limit is not None and self.source_feature_limit <= 0:
            raise ValueError("source_feature_limit must be positive")


def validate_inputs(
    post_a: np.ndarray,
    post_b: np.ndarray,
    sequence_ids: np.ndarray,
) -> None:
    if post_a.ndim != 2 or post_b.ndim != 2:
        raise ValueError("post_a and post_b must be 2-D")
    if post_a.shape[0] != post_b.shape[0]:
        raise ValueError(f"row mismatch: {post_a.shape} vs {post_b.shape}")
    if sequence_ids.shape != (post_a.shape[0],):
        raise ValueError(
            f"sequence_ids must have shape ({post_a.shape[0]},), "
            f"got {sequence_ids.shape}"
        )
    if post_a.shape[0] == 0 or post_a.shape[1] == 0 or post_b.shape[1] == 0:
        raise ValueError("activation arrays must be nonempty")
    if not np.issubdtype(post_a.dtype, np.floating):
        raise TypeError("post_a must have a floating dtype")
    if not np.issubdtype(post_b.dtype, np.floating):
        raise TypeError("post_b must have a floating dtype")
    if len(np.unique(sequence_ids)) < 2:
        raise ValueError("at least two sequences are required for a held-out split")


def sequence_train_test_masks(
    sequence_ids: np.ndarray,
    *,
    train_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split unique sequence IDs, never individual token/word rows."""

    sequence_ids = np.asarray(sequence_ids)
    unique = np.unique(sequence_ids)
    generator = np.random.default_rng(seed)
    shuffled = generator.permutation(unique)
    n_train = min(max(int(train_fraction * len(unique)), 1), len(unique) - 1)
    train_sequences = np.sort(shuffled[:n_train])
    test_sequences = np.sort(shuffled[n_train:])
    train_mask = np.isin(sequence_ids, train_sequences)
    test_mask = ~train_mask
    return train_mask, test_mask, train_sequences, test_sequences


def accumulate_gram_and_rhs_numpy(
    source: np.ndarray,
    target: np.ndarray,
    *,
    row_chunk_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate ``[X,1]^T[X,1]`` and ``[X,1]^T Y`` in float64."""

    n_rows, n_source = source.shape
    n_target = target.shape[1]
    gram = np.zeros((n_source + 1, n_source + 1), dtype=np.float64)
    rhs = np.zeros((n_source + 1, n_target), dtype=np.float64)
    for start in range(0, n_rows, row_chunk_size):
        end = min(start + row_chunk_size, n_rows)
        source_batch = np.asarray(source[start:end], dtype=np.float64)
        design = np.empty((end - start, n_source + 1), dtype=np.float64)
        design[:, :n_source] = source_batch
        design[:, n_source] = 1.0
        target_batch = np.asarray(target[start:end], dtype=np.float64)
        gram += design.T @ design
        rhs += design.T @ target_batch
    return gram, rhs


def accumulate_gram_and_rhs_cuda(
    source: np.ndarray,
    target: np.ndarray,
    *,
    device: str,
    row_chunk_size: int = 4096,
):
    """Accumulate streamed fp32 Gram/RHS matrices on a GPU."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("the CUDA ridge backend requires PyTorch") from exc

    n_rows, n_source = source.shape
    n_target = target.shape[1]
    design_dim = n_source + 1
    gram = torch.zeros(
        (design_dim, design_dim), dtype=torch.float32, device=device
    )
    rhs = torch.zeros((design_dim, n_target), dtype=torch.float32, device=device)
    for start in range(0, n_rows, row_chunk_size):
        end = min(start + row_chunk_size, n_rows)
        source_batch = torch.from_numpy(np.asarray(source[start:end])).to(device).float()
        design = torch.empty(
            (end - start, design_dim), dtype=torch.float32, device=device
        )
        design[:, :n_source] = source_batch
        design[:, n_source] = 1.0
        target_batch = torch.from_numpy(np.asarray(target[start:end])).to(device).float()
        gram.addmm_(design.T, design)
        rhs.addmm_(design.T, target_batch)
        del source_batch, design, target_batch
    return gram, rhs


def ridge_solve_numpy(
    gram: np.ndarray,
    rhs: np.ndarray,
    *,
    ridge_lambda: float,
) -> np.ndarray:
    """Solve ridge in float64, leaving the final intercept row unpenalized."""

    penalized = np.asarray(gram, dtype=np.float64).copy()
    penalized_indices = np.arange(penalized.shape[0] - 1)
    penalized[penalized_indices, penalized_indices] += ridge_lambda
    return np.linalg.solve(penalized, np.asarray(rhs, dtype=np.float64)).astype(
        np.float32
    )


def ridge_solve_torch(
    gram,
    rhs,
    *,
    ridge_lambda: float,
    column_chunk_size: int,
) -> np.ndarray:
    """Solve ridge in fp64 on CPU with chunked RHS output."""

    import torch

    gram.diagonal()[:-1] += ridge_lambda
    gram_cpu = gram.cpu().double()
    del gram
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gram_cpu = (gram_cpu + gram_cpu.T) * 0.5
    rhs_cpu = rhs.cpu().double()
    del rhs
    lu, pivots = torch.linalg.lu_factor(gram_cpu)
    del gram_cpu
    coefficients = np.empty(rhs_cpu.shape, dtype=np.float32)
    for start in range(0, rhs_cpu.shape[1], column_chunk_size):
        end = min(start + column_chunk_size, rhs_cpu.shape[1])
        solved = torch.linalg.lu_solve(lu, pivots, rhs_cpu[:, start:end])
        coefficients[:, start:end] = solved.float().numpy()
    return coefficients


def _select_backend(config: BiasShiftRidgeConfig) -> str:
    if config.backend != "auto":
        return config.backend
    try:
        import torch

        if torch.cuda.is_available() and config.device.startswith("cuda"):
            return "cuda"
    except ImportError:
        pass
    return "numpy"


def fit_ridge(
    source_train: np.ndarray,
    target_train: np.ndarray,
    config: BiasShiftRidgeConfig,
) -> tuple[np.ndarray, str]:
    backend = _select_backend(config)
    if backend == "cuda":
        gram, rhs = accumulate_gram_and_rhs_cuda(
            source_train,
            target_train,
            device=config.device,
            row_chunk_size=config.fit_row_chunk_size,
        )
        coefficients = ridge_solve_torch(
            gram,
            rhs,
            ridge_lambda=config.ridge_lambda,
            column_chunk_size=config.solve_column_chunk_size,
        )
    else:
        gram, rhs = accumulate_gram_and_rhs_numpy(
            source_train,
            target_train,
            row_chunk_size=config.fit_row_chunk_size,
        )
        coefficients = ridge_solve_numpy(
            gram, rhs, ridge_lambda=config.ridge_lambda
        )
    return coefficients, backend


def apply_remix(
    source_eval: np.ndarray,
    coefficients: np.ndarray,
    *,
    backend: str,
    device: str,
    row_chunk_size: int,
    output_path: Path | None = None,
) -> np.ndarray:
    """Evaluate ``ReLU([source,1] @ coefficients)`` in row chunks."""

    n_rows, n_source = source_eval.shape
    if coefficients.shape[0] != n_source + 1:
        raise ValueError("coefficient shape does not match source feature count")
    shape = (n_rows, coefficients.shape[1])
    if output_path is None:
        output: np.ndarray = np.empty(shape, dtype=np.float16)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=np.float16, shape=shape
        )

    if backend == "cuda":
        import torch

        coefficient_device = torch.from_numpy(coefficients).to(device)
        weights = coefficient_device[:-1]
        intercept = coefficient_device[-1]
        for start in range(0, n_rows, row_chunk_size):
            end = min(start + row_chunk_size, n_rows)
            source_batch = torch.from_numpy(
                np.asarray(source_eval[start:end])
            ).to(device).float()
            prediction = torch.relu(source_batch @ weights + intercept[None, :])
            output[start:end] = prediction.half().cpu().numpy()
            del source_batch, prediction
        del coefficient_device, weights, intercept
        torch.cuda.empty_cache()
    else:
        weights = np.asarray(coefficients[:-1], dtype=np.float32)
        intercept = np.asarray(coefficients[-1], dtype=np.float32)
        for start in range(0, n_rows, row_chunk_size):
            end = min(start + row_chunk_size, n_rows)
            source_batch = np.asarray(source_eval[start:end], dtype=np.float32)
            prediction = source_batch @ weights + intercept[None, :]
            np.maximum(prediction, 0.0, out=prediction)
            output[start:end] = prediction.astype(np.float16)
    if isinstance(output, np.memmap):
        output.flush()
    return output


def run_bias_shift_ridge(
    post_a: np.ndarray,
    post_b: np.ndarray,
    sequence_ids: np.ndarray,
    *,
    config: BiasShiftRidgeConfig | None = None,
    work_dir: Path | str = Path("."),
    keep_prediction: bool = False,
) -> dict[str, Any]:
    """Run the complete BS-R fit, held-out score, and SSR diagnostic."""

    config = config or BiasShiftRidgeConfig()
    config.validate()
    post_a = np.asarray(post_a)
    post_b = np.asarray(post_b)
    sequence_ids = np.asarray(sequence_ids)
    validate_inputs(post_a, post_b, sequence_ids)
    started = time.time()

    train_mask, test_mask, train_sequences, test_sequences = sequence_train_test_masks(
        sequence_ids,
        train_fraction=config.train_fraction,
        seed=config.seed,
    )
    source_indices = np.arange(post_a.shape[1])
    if config.source_feature_limit is not None and config.source_feature_limit < len(source_indices):
        generator = np.random.default_rng(config.seed)
        source_indices = np.sort(
            generator.choice(
                source_indices,
                size=config.source_feature_limit,
                replace=False,
            )
        )

    source_train = np.asarray(post_a[train_mask][:, source_indices])
    source_eval = np.asarray(post_a[test_mask][:, source_indices])
    target_train = np.asarray(post_b[train_mask])
    target_eval = np.asarray(post_b[test_mask])

    if config.row_shuffle:
        generator = np.random.default_rng(config.seed)
        source_train = source_train[generator.permutation(len(source_train))]
        source_eval = source_eval[generator.permutation(len(source_eval))]

    LOGGER.info(
        "fitting ridge on %d rows, %d source features, %d target features",
        len(source_train),
        source_train.shape[1],
        target_train.shape[1],
    )
    coefficients, fit_backend = fit_ridge(source_train, target_train, config)
    prediction_path = Path(work_dir) / "bsr_remixed_eval.npy"
    remixed = apply_remix(
        source_eval,
        coefficients,
        backend=fit_backend,
        device=config.device,
        row_chunk_size=config.apply_row_chunk_size,
        output_path=prediction_path,
    )
    invalid_by_column = np.zeros(remixed.shape[1], dtype=bool)
    for start in range(0, len(remixed), config.apply_row_chunk_size):
        end = min(start + config.apply_row_chunk_size, len(remixed))
        batch = np.asarray(remixed[start:end])
        invalid = ~np.isfinite(batch)
        invalid_by_column |= np.any(invalid, axis=0)
        if invalid.any():
            remixed[start:end] = np.nan_to_num(
                batch, nan=0.0, posinf=0.0, neginf=0.0
            )
    invalid_columns = int(invalid_by_column.sum())
    if invalid_columns:
        LOGGER.warning("zero-filled non-finite predictions in %d columns", invalid_columns)

    target_weights = activation_frequencies(
        target_train,
        threshold=config.native_dead_threshold,
    )
    reductions = cosine_similarity_reductions(
        remixed,
        target_eval,
        backend=fit_backend,
        device=config.device,
        feature_chunk_size=config.similarity_feature_chunk_size,
        row_chunk_size=config.similarity_row_chunk_size,
        dead_threshold=config.sample_dead_threshold,
    )
    metrics = compute_v4_metrics(
        reductions,
        weights_a=target_weights,
        weights_b=target_weights,
        native_dead_threshold=config.native_dead_threshold,
    )
    ssr = self_slot_recovery(
        reductions,
        target_weights,
        native_dead_threshold=config.native_dead_threshold,
    )

    summary = {
        "method": "Bias-Shift Ridge (BS-R)",
        "config": asdict(config),
        "backend_used": fit_backend,
        "shapes": {
            "aligned_rows": int(len(sequence_ids)),
            "source_features_original": int(post_a.shape[1]),
            "source_features_used": int(len(source_indices)),
            "target_features": int(post_b.shape[1]),
            "train_rows": int(train_mask.sum()),
            "eval_rows": int(test_mask.sum()),
            "train_sequences": int(len(train_sequences)),
            "eval_sequences": int(len(test_sequences)),
        },
        "metrics": metrics,
        "self_slot": ssr,
        "invalid_prediction_columns": invalid_columns,
        "elapsed_seconds": float(time.time() - started),
    }
    arrays = {
        "max_sim_a": reductions.max_sim_a,
        "max_sim_b": reductions.max_sim_b,
        "argmax_a": reductions.argmax_a,
        "dead_a": reductions.dead_a,
        "dead_b": reductions.dead_b,
        "diagonal": reductions.diagonal,
        "target_weights": target_weights,
        "source_feature_indices": source_indices,
        "train_sequences": train_sequences,
        "eval_sequences": test_sequences,
    }
    result: dict[str, Any] = {"summary": summary, "arrays": arrays}
    if keep_prediction:
        result["prediction_path"] = prediction_path
    else:
        del remixed
        prediction_path.unlink(missing_ok=True)
    return result


def load_prepared_inputs(input_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = {
        "post_a": input_dir / "post_a.npy",
        "post_b": input_dir / "post_b.npy",
        "sequence_ids": input_dir / "sequence_ids.npy",
    }
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing prepared BS-R inputs in {input_dir}: {', '.join(missing)}"
        )
    return (
        np.load(paths["post_a"], mmap_mode="r", allow_pickle=False),
        np.load(paths["post_b"], mmap_mode="r", allow_pickle=False),
        np.load(paths["sequence_ids"], mmap_mode="r", allow_pickle=False),
    )


def save_outputs(result: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = write_json(output_dir / "bias_shift_ridge_summary.json", result["summary"])
    arrays_path = output_dir / "bias_shift_ridge_arrays.npz"
    arrays = {
        key: value for key, value in result["arrays"].items() if value is not None
    }
    np.savez_compressed(arrays_path, **arrays)
    return summary_path, arrays_path



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bias_shift_ridge"))
    parser.add_argument("--ridge-lambda", type=float, default=100.0)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-dead-threshold", type=float, default=1e-6)
    parser.add_argument("--native-dead-threshold", type=float, default=1e-7)
    parser.add_argument("--backend", choices=["auto", "numpy", "cuda"], default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fit-row-chunk-size", type=int, default=4096)
    parser.add_argument("--solve-column-chunk-size", type=int, default=4096)
    parser.add_argument("--apply-row-chunk-size", type=int, default=4096)
    parser.add_argument("--similarity-feature-chunk-size", type=int, default=512)
    parser.add_argument("--similarity-row-chunk-size", type=int, default=8192)
    parser.add_argument("--row-shuffle", action="store_true")
    parser.add_argument("--source-feature-limit", type=int)
    parser.add_argument("--keep-prediction", action="store_true")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    post_a, post_b, sequence_ids = load_prepared_inputs(args.input_dir)
    config = BiasShiftRidgeConfig(
        ridge_lambda=args.ridge_lambda,
        train_fraction=args.train_fraction,
        seed=args.seed,
        sample_dead_threshold=args.sample_dead_threshold,
        native_dead_threshold=args.native_dead_threshold,
        fit_row_chunk_size=args.fit_row_chunk_size,
        solve_column_chunk_size=args.solve_column_chunk_size,
        apply_row_chunk_size=args.apply_row_chunk_size,
        similarity_feature_chunk_size=args.similarity_feature_chunk_size,
        similarity_row_chunk_size=args.similarity_row_chunk_size,
        backend=args.backend,
        device=args.device,
        row_shuffle=args.row_shuffle,
        source_feature_limit=args.source_feature_limit,
    )
    result = run_bias_shift_ridge(
        post_a,
        post_b,
        sequence_ids,
        config=config,
        work_dir=args.output_dir,
        keep_prediction=args.keep_prediction,
    )
    summary_path, arrays_path = save_outputs(result, args.output_dir)
    metrics = result["summary"]["metrics"]
    ssr = result["summary"]["self_slot"]["self_slot_recovery"]
    print(
        f"BS-R complete: F1={metrics['f1']:.6f}, "
        f"P={metrics['precision']:.6f}, R={metrics['recall']:.6f}, SSR={ssr:.6f}"
    )
    print(f"summary: {summary_path.resolve()}")
    print(f"arrays:  {arrays_path.resolve()}")


if __name__ == "__main__":
    main()
