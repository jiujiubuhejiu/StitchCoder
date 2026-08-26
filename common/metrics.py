"""Shared feature-similarity, dead-feature, and weighted-F1 routines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimilarityReductions:
    """Reductions needed by the weighted greedy aggregator."""

    max_sim_a: np.ndarray
    max_sim_b: np.ndarray
    argmax_a: np.ndarray
    dead_a: np.ndarray
    dead_b: np.ndarray
    diagonal: np.ndarray | None = None


def _weights_or_uniform(
    weights: np.ndarray | None,
    n_features: int,
) -> np.ndarray:
    if weights is None:
        return np.ones(n_features, dtype=np.float64)
    values = np.asarray(weights, dtype=np.float64).copy()
    if values.shape != (n_features,):
        raise ValueError(
            f"weights must have shape ({n_features},), got {values.shape}"
        )
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("weights must be finite and non-negative")
    return values


def compute_weighted_f1(
    max_sim_a: np.ndarray,
    max_sim_b: np.ndarray,
    weights_a: np.ndarray | None = None,
    weights_b: np.ndarray | None = None,
) -> dict[str, float]:
    """Activation-frequency-weighted bidirectional greedy P/R/F1."""

    sim_a = np.asarray(max_sim_a, dtype=np.float64)
    sim_b = np.asarray(max_sim_b, dtype=np.float64)
    wa = _weights_or_uniform(weights_a, len(sim_a))
    wb = _weights_or_uniform(weights_b, len(sim_b))
    sum_a = float(wa.sum())
    sum_b = float(wb.sum())
    if sum_a < 1e-30 or sum_b < 1e-30:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = float(np.sum(sim_a * wa) / sum_a)
    recall = float(np.sum(sim_b * wb) / sum_b)
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if precision + recall > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_v4_metrics(
    reductions: SimilarityReductions,
    *,
    weights_a: np.ndarray | None = None,
    weights_b: np.ndarray | None = None,
    native_dead_threshold: float = 1e-7,
) -> dict[str, float | int]:
    """Exclude native-dead features while retaining the M-killed penalty."""

    wa = _weights_or_uniform(weights_a, len(reductions.max_sim_a))
    wb = _weights_or_uniform(weights_b, len(reductions.max_sim_b))
    native_dead_a = wa < native_dead_threshold
    native_dead_b = wb < native_dead_threshold
    killed_a = reductions.dead_a & ~native_dead_a
    killed_b = reductions.dead_b & ~native_dead_b
    wa[native_dead_a] = 0.0
    wb[native_dead_b] = 0.0
    metrics: dict[str, float | int] = compute_weighted_f1(
        reductions.max_sim_a,
        reductions.max_sim_b,
        wa,
        wb,
    )
    metrics.update(
        {
            "sample_dead_a": int(reductions.dead_a.sum()),
            "sample_dead_b": int(reductions.dead_b.sum()),
            "native_dead_a": int(native_dead_a.sum()),
            "native_dead_b": int(native_dead_b.sum()),
            "m_killed_a": int(killed_a.sum()),
            "m_killed_b": int(killed_b.sum()),
            "m_killed_a_ratio": float(
                killed_a.sum() / max(int((~native_dead_a).sum()), 1)
            ),
            "m_killed_b_ratio": float(
                killed_b.sum() / max(int((~native_dead_b).sum()), 1)
            ),
        }
    )
    return metrics


def activation_frequencies(
    post: np.ndarray,
    *,
    threshold: float = 1e-7,
    row_chunk_size: int = 32768,
) -> np.ndarray:
    """Compute the fraction of rows on which each feature is active."""

    values = np.asarray(post)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("post must be a nonempty 2-D array")
    counts = np.zeros(values.shape[1], dtype=np.float64)
    for start in range(0, values.shape[0], row_chunk_size):
        batch = np.asarray(values[start : start + row_chunk_size], dtype=np.float32)
        counts += np.sum(batch > threshold, axis=0, dtype=np.float64)
    return counts / values.shape[0]


def _feature_denominators(
    post: np.ndarray,
    row_chunk_size: int,
) -> np.ndarray:
    values = np.asarray(post)
    sums = np.zeros(values.shape[1], dtype=np.float64)
    for start in range(0, values.shape[0], row_chunk_size):
        batch = np.asarray(values[start : start + row_chunk_size], dtype=np.float32)
        sums += np.sum(batch * batch, axis=0, dtype=np.float64)
    return np.sqrt(np.maximum(sums / values.shape[0], 1e-30))


def cosine_similarity_reductions_numpy(
    post_a: np.ndarray,
    post_b: np.ndarray,
    *,
    feature_chunk_size: int = 256,
    row_chunk_size: int = 4096,
    dead_threshold: float = 1e-6,
) -> SimilarityReductions:
    """Compute cosine maxima without retaining the full feature matrix."""

    a = np.asarray(post_a)
    b = np.asarray(post_b)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("post_a and post_b must be 2-D")
    if a.shape[0] != b.shape[0] or a.shape[0] == 0:
        raise ValueError(
            f"activation rows must be paired and nonempty: {a.shape} vs {b.shape}"
        )
    if feature_chunk_size <= 0 or row_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")

    n_rows, n_a = a.shape
    n_b = b.shape[1]
    denom_a = _feature_denominators(a, row_chunk_size)
    denom_b = _feature_denominators(b, row_chunk_size)
    dead_a = denom_a < dead_threshold
    dead_b = denom_b < dead_threshold
    max_a = np.full(n_a, -np.inf, dtype=np.float64)
    max_b = np.full(n_b, -np.inf, dtype=np.float64)
    argmax_a = np.zeros(n_a, dtype=np.int64)
    diagonal = (
        np.zeros(n_a, dtype=np.float64) if n_a == n_b else None
    )

    for feature_start in range(0, n_a, feature_chunk_size):
        feature_end = min(feature_start + feature_chunk_size, n_a)
        cross = np.zeros((feature_end - feature_start, n_b), dtype=np.float64)
        for row_start in range(0, n_rows, row_chunk_size):
            row_end = min(row_start + row_chunk_size, n_rows)
            a_batch = np.asarray(
                a[row_start:row_end, feature_start:feature_end],
                dtype=np.float32,
            )
            b_batch = np.asarray(b[row_start:row_end], dtype=np.float32)
            cross += np.asarray(a_batch.T @ b_batch, dtype=np.float64)
        similarity = (cross / n_rows) / np.maximum(
            denom_a[feature_start:feature_end, None] * denom_b[None, :],
            1e-30,
        )
        similarity[
            dead_a[feature_start:feature_end, None] | dead_b[None, :]
        ] = 0.0
        np.clip(similarity, -1.0, 1.0, out=similarity)
        local_argmax = np.argmax(similarity, axis=1)
        max_a[feature_start:feature_end] = similarity[
            np.arange(feature_end - feature_start), local_argmax
        ]
        argmax_a[feature_start:feature_end] = local_argmax
        max_b = np.maximum(max_b, np.max(similarity, axis=0))
        if diagonal is not None:
            local = np.arange(feature_end - feature_start)
            global_index = np.arange(feature_start, feature_end)
            diagonal[feature_start:feature_end] = similarity[local, global_index]

    return SimilarityReductions(
        max_sim_a=max_a,
        max_sim_b=max_b,
        argmax_a=argmax_a,
        dead_a=dead_a,
        dead_b=dead_b,
        diagonal=diagonal,
    )


def cosine_similarity_reductions_cuda(
    post_a: np.ndarray,
    post_b: np.ndarray,
    *,
    device: str = "cuda",
    feature_chunk_size: int = 512,
    row_chunk_size: int = 8192,
    dead_threshold: float = 1e-6,
) -> SimilarityReductions:
    """GPU implementation of BS-R cosine-similarity reductions."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("the CUDA similarity backend requires PyTorch") from exc

    a = np.asarray(post_a)
    b = np.asarray(post_b)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
        raise ValueError(f"invalid paired activation shapes: {a.shape}, {b.shape}")
    n_rows, n_a = a.shape
    n_b = b.shape[1]
    if n_rows == 0:
        raise ValueError("activation arrays must not be empty")
    torch.cuda.empty_cache()

    def denominators(values: np.ndarray):
        accumulator = torch.zeros(values.shape[1], dtype=torch.float64, device=device)
        for start in range(0, len(values), row_chunk_size):
            batch = torch.from_numpy(
                np.asarray(values[start : start + row_chunk_size])
            ).to(device).float()
            accumulator += (batch.double() ** 2).sum(dim=0)
            del batch
        return torch.sqrt(torch.clamp(accumulator / len(values), min=1e-30))

    denom_a = denominators(a)
    denom_b = denominators(b)
    dead_a = denom_a < dead_threshold
    dead_b = denom_b < dead_threshold
    b_device = torch.from_numpy(np.asarray(b)).to(device).float()
    max_a = torch.full((n_a,), -1.0, dtype=torch.float32, device=device)
    max_b = torch.full((n_b,), -1.0, dtype=torch.float32, device=device)
    argmax_a = torch.zeros(n_a, dtype=torch.int64, device=device)
    diagonal = (
        torch.zeros(n_a, dtype=torch.float32, device=device)
        if n_a == n_b
        else None
    )

    for start in range(0, n_a, feature_chunk_size):
        end = min(start + feature_chunk_size, n_a)
        a_chunk = torch.from_numpy(np.asarray(a[:, start:end])).to(device).float()
        similarity = (a_chunk.T @ b_device) / n_rows
        similarity /= (
            denom_a[start:end].float()[:, None]
            * denom_b.float()[None, :]
            + 1e-30
        )
        similarity[
            dead_a[start:end, None] | dead_b[None, :]
        ] = 0.0
        similarity.clamp_(-1.0, 1.0)
        row_max, row_argmax = similarity.max(dim=1)
        max_a[start:end] = row_max
        argmax_a[start:end] = row_argmax
        max_b = torch.maximum(max_b, similarity.max(dim=0).values)
        if diagonal is not None:
            local = torch.arange(end - start, device=device)
            global_index = torch.arange(start, end, device=device)
            diagonal[start:end] = similarity[local, global_index]
        del a_chunk, similarity

    result = SimilarityReductions(
        max_sim_a=max_a.double().cpu().numpy(),
        max_sim_b=max_b.double().cpu().numpy(),
        argmax_a=argmax_a.cpu().numpy(),
        dead_a=dead_a.cpu().numpy(),
        dead_b=dead_b.cpu().numpy(),
        diagonal=None if diagonal is None else diagonal.double().cpu().numpy(),
    )
    del b_device
    torch.cuda.empty_cache()
    return result


def cosine_similarity_reductions(
    post_a: np.ndarray,
    post_b: np.ndarray,
    *,
    backend: str = "auto",
    device: str = "cuda",
    feature_chunk_size: int = 512,
    row_chunk_size: int = 8192,
    dead_threshold: float = 1e-6,
) -> SimilarityReductions:
    """Dispatch to the NumPy or CUDA reduction implementation."""

    selected = backend
    if selected == "auto":
        try:
            import torch

            selected = "cuda" if torch.cuda.is_available() and device.startswith("cuda") else "numpy"
        except ImportError:
            selected = "numpy"
    if selected == "cuda":
        return cosine_similarity_reductions_cuda(
            post_a,
            post_b,
            device=device,
            feature_chunk_size=feature_chunk_size,
            row_chunk_size=row_chunk_size,
            dead_threshold=dead_threshold,
        )
    if selected == "numpy":
        return cosine_similarity_reductions_numpy(
            post_a,
            post_b,
            feature_chunk_size=feature_chunk_size,
            row_chunk_size=row_chunk_size,
            dead_threshold=dead_threshold,
        )
    raise ValueError(f"unknown similarity backend: {backend}")


def self_slot_recovery(
    reductions: SimilarityReductions,
    weights: np.ndarray,
    *,
    native_dead_threshold: float = 1e-7,
) -> dict[str, float | int]:
    """Fraction of alive target slots whose reconstructed best match is itself."""

    if len(reductions.max_sim_a) != len(reductions.max_sim_b):
        raise ValueError("SSR requires both similarity axes to index target slots")
    target_weights = _weights_or_uniform(weights, len(reductions.max_sim_b))
    alive = (
        (target_weights >= native_dead_threshold)
        & ~reductions.dead_a
        & ~reductions.dead_b
    )
    wins = (reductions.argmax_a == np.arange(len(alive))) & alive
    count = int(alive.sum())
    return {
        "self_slot_recovery": float(wins.sum() / max(count, 1)),
        "alive_slots": count,
        "self_slot_wins": int(wins.sum()),
    }
