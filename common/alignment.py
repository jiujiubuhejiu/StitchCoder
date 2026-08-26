"""Semi-orthogonal Procrustes utilities for residual-space alignment."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class AlignmentDiagnostics:
    n_samples: int
    source_dim: int
    target_dim: int
    singular_value_sum: float
    max_semiorthogonality_error: float
    preprocessing: str = "l2_row_normalize"


def l2_normalize_rows(array: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return a float64 copy whose nonzero rows have unit Euclidean norm."""

    values = np.asarray(array, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("alignment activations contain NaN or infinity")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)


def semiorthogonality_error(matrix: np.ndarray) -> float:
    """Maximum absolute deviation from the relevant Stiefel constraint."""

    matrix64 = np.asarray(matrix, dtype=np.float64)
    if matrix64.ndim != 2:
        raise ValueError("alignment matrix must be 2-D")
    if matrix64.shape[0] >= matrix64.shape[1]:
        gram = matrix64.T @ matrix64
    else:
        gram = matrix64 @ matrix64.T
    return float(np.max(np.abs(gram - np.eye(gram.shape[0]))))


def fit_semiorthogonal_procrustes(
    source: np.ndarray,
    target: np.ndarray,
    *,
    normalize_rows: bool = True,
) -> tuple[np.ndarray, dict[str, int | float | str]]:
    """Fit ``argmin_M ||source @ M - target||_F`` by thin SVD.

    The result has shape ``(source_dim, target_dim)``.  It has orthonormal
    columns when ``source_dim >= target_dim`` and orthonormal rows otherwise.
    This is the square/rectangular map used by StitchCoder.
    """

    source_values = np.asarray(source)
    target_values = np.asarray(target)
    if source_values.ndim != 2 or target_values.ndim != 2:
        raise ValueError("source and target alignment arrays must be 2-D")
    if source_values.shape[0] != target_values.shape[0]:
        raise ValueError(
            "paired alignment arrays must have the same row count: "
            f"{source_values.shape} vs {target_values.shape}"
        )
    if source_values.shape[0] == 0:
        raise ValueError("at least one paired alignment sample is required")

    if normalize_rows:
        source_fit = l2_normalize_rows(source_values)
        target_fit = l2_normalize_rows(target_values)
        preprocessing = "l2_row_normalize"
    else:
        source_fit = np.asarray(source_values, dtype=np.float64)
        target_fit = np.asarray(target_values, dtype=np.float64)
        preprocessing = "none"

    cross = source_fit.T @ target_fit
    left, singular, right_t = np.linalg.svd(cross, full_matrices=False)
    matrix = (left @ right_t).astype(np.float32)
    diagnostics = AlignmentDiagnostics(
        n_samples=int(source_values.shape[0]),
        source_dim=int(source_values.shape[1]),
        target_dim=int(target_values.shape[1]),
        singular_value_sum=float(singular.sum()),
        max_semiorthogonality_error=semiorthogonality_error(matrix),
        preprocessing=preprocessing,
    )
    return matrix, asdict(diagnostics)


def transport_source(
    hidden_source: np.ndarray,
    encoder_source: np.ndarray,
    alignment_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Express source residual rows and encoder columns in target coordinates."""

    hidden_source = np.asarray(hidden_source, dtype=np.float32)
    encoder_source = np.asarray(encoder_source, dtype=np.float32)
    alignment_matrix = np.asarray(alignment_matrix, dtype=np.float32)
    if hidden_source.shape[1] != alignment_matrix.shape[0]:
        raise ValueError("source hidden dimension does not match alignment map")
    if encoder_source.shape[0] != alignment_matrix.shape[0]:
        raise ValueError("source encoder dimension does not match alignment map")
    return (
        np.asarray(hidden_source @ alignment_matrix, dtype=np.float32),
        np.asarray(alignment_matrix.T @ encoder_source, dtype=np.float32),
    )
