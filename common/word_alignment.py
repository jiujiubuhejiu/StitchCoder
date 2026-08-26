"""Whitespace-word pooling for pairs that use different tokenizers."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np


_WORD_PATTERN = re.compile(r"\S+")


def _word_groups(
    sequence_ids: np.ndarray,
    offsets: np.ndarray,
    texts: list[str],
) -> list[tuple[int, int, np.ndarray]]:
    sequence_ids = np.asarray(sequence_ids)
    offsets = np.asarray(offsets)
    if sequence_ids.ndim != 1 or offsets.shape != (len(sequence_ids), 2):
        raise ValueError("invalid sequence-id or offset array")
    rows_by_sequence: dict[int, list[int]] = defaultdict(list)
    for row, sequence_id in enumerate(sequence_ids):
        rows_by_sequence[int(sequence_id)].append(row)

    groups: list[tuple[int, int, np.ndarray]] = []
    for sequence_id, text in enumerate(texts):
        token_rows = rows_by_sequence.get(sequence_id, [])
        spans = [(match.start(), match.end()) for match in _WORD_PATTERN.finditer(text)]
        if not token_rows or not spans:
            continue
        starts = np.asarray([start for start, _ in spans], dtype=np.int64)
        ends = np.asarray([end for _, end in spans], dtype=np.int64)
        buckets: dict[int, list[int]] = defaultdict(list)
        for row in token_rows:
            start, end = (int(offsets[row, 0]), int(offsets[row, 1]))
            if end <= start:
                continue
            midpoint = (start + end - 1) // 2
            word_index = int(np.searchsorted(starts, midpoint, side="right")) - 1
            if (
                0 <= word_index < len(spans)
                and starts[word_index] <= midpoint < ends[word_index]
            ):
                buckets[word_index].append(row)
        for word_index in sorted(buckets):
            groups.append(
                (
                    sequence_id,
                    word_index,
                    np.asarray(buckets[word_index], dtype=np.int64),
                )
            )
    return groups


def pool_by_whitespace_words(
    post: np.ndarray,
    sequence_ids: np.ndarray,
    offsets: np.ndarray,
    texts: list[str],
    *,
    output_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean-pool token activations by whitespace-delimited word spans."""

    values = np.asarray(post)
    if values.ndim != 2 or values.shape[0] != len(sequence_ids):
        raise ValueError("post rows must match sequence_ids")
    groups = _word_groups(sequence_ids, offsets, texts)
    shape = (len(groups), values.shape[1])
    if output_path is None:
        pooled: np.ndarray = np.empty(shape, dtype=np.float16)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pooled = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=np.float16, shape=shape
        )
    word_sequence_ids = np.empty(len(groups), dtype=np.int32)
    word_indices = np.empty(len(groups), dtype=np.int32)
    for output_row, (sequence_id, word_index, token_rows) in enumerate(groups):
        pooled[output_row] = np.asarray(
            values[token_rows], dtype=np.float32
        ).mean(axis=0).astype(np.float16)
        word_sequence_ids[output_row] = sequence_id
        word_indices[output_row] = word_index
    if isinstance(pooled, np.memmap):
        pooled.flush()
    return pooled, word_sequence_ids, word_indices


def align_word_rows(
    sequence_ids_a: np.ndarray,
    word_indices_a: np.ndarray,
    sequence_ids_b: np.ndarray,
    word_indices_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return row indices sharing the same ``(sequence, word)`` key."""

    def packed(sequence_ids: np.ndarray, word_indices: np.ndarray) -> np.ndarray:
        sequence_ids = np.asarray(sequence_ids, dtype=np.int64)
        word_indices = np.asarray(word_indices, dtype=np.int64)
        if sequence_ids.shape != word_indices.shape:
            raise ValueError("sequence IDs and word indices must have equal shape")
        return (sequence_ids << 32) | (word_indices & 0xFFFFFFFF)

    keys_a = packed(sequence_ids_a, word_indices_a)
    keys_b = packed(sequence_ids_b, word_indices_b)
    if len(np.unique(keys_a)) != len(keys_a) or len(np.unique(keys_b)) != len(keys_b):
        raise ValueError("word keys must be unique on each side")
    positions_a = {int(key): index for index, key in enumerate(keys_a)}
    positions_b = {int(key): index for index, key in enumerate(keys_b)}
    common = sorted(positions_a.keys() & positions_b.keys())
    indices_a = np.fromiter(
        (positions_a[key] for key in common), dtype=np.int64, count=len(common)
    )
    indices_b = np.fromiter(
        (positions_b[key] for key in common), dtype=np.int64, count=len(common)
    )
    sequence_ids = (np.asarray(common, dtype=np.int64) >> 32).astype(np.int32)
    return indices_a, indices_b, sequence_ids
