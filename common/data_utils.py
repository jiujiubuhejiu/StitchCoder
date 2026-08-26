"""Deterministic text selection for the Pile alignment and C4 scoring pools."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from common.io_utils import sha256_text


def _load_stream(dataset_config: dict[str, Any]):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("dataset loading requires the `datasets` package") from exc

    name = dataset_config["name"]
    subset = dataset_config.get("subset")
    split = dataset_config.get("split", "train")
    arguments: list[str] = [name]
    if subset is not None:
        arguments.append(subset)
    return load_dataset(
        *arguments,
        split=split,
        streaming=True,
        revision=dataset_config.get("revision"),
    )


def _example_text(example: dict[str, Any], field: str) -> str:
    value = example.get(field, example.get("content", ""))
    return value if isinstance(value, str) else str(value)


def load_texts(
    dataset_config: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Load and select texts while recording sufficient sampling metadata."""

    stream = _load_stream(dataset_config)
    count = int(dataset_config["num_samples"])
    field = dataset_config.get("text_field", "text")
    strategy = dataset_config.get("sampling", "first")
    seed = int(dataset_config.get("seed", 0))
    pool_size = int(dataset_config.get("pool_size", count))
    char_limit = dataset_config.get("char_limit")
    min_chars = int(dataset_config.get("min_chars", 0))

    candidates: list[tuple[int, str, str]] = []
    categories: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for stream_index, example in enumerate(stream):
        if stream_index >= pool_size:
            break
        text = _example_text(example, field)
        if char_limit is not None:
            text = text[: int(char_limit)]
        if len(text.strip()) < min_chars:
            continue
        metadata = example.get("meta", {})
        category = (
            metadata.get("pile_set_name", "unknown")
            if isinstance(metadata, dict)
            else "unknown"
        )
        record = (stream_index, text, str(category))
        candidates.append(record)
        categories[str(category)].append(record)
        if strategy == "first" and len(candidates) >= count:
            break

    if strategy == "first":
        selected = candidates[:count]
    elif strategy == "random":
        if len(candidates) < count:
            raise RuntimeError(
                f"dataset pool yielded {len(candidates)} usable rows, need {count}"
            )
        selected = random.Random(seed).sample(candidates, count)
    elif strategy == "stratified":
        if not categories:
            raise RuntimeError("no dataset categories were available for stratification")
        generator = random.Random(seed)
        selected = []
        category_items = list(categories.items())
        base = count // len(category_items)
        remainder = count % len(category_items)
        for index, (_, records) in enumerate(category_items):
            requested = base + (1 if index < remainder else 0)
            selected.extend(
                generator.sample(records, min(requested, len(records)))
            )
        generator.shuffle(selected)
        selected = selected[:count]
        if len(selected) < count:
            used = {record[0] for record in selected}
            remaining = [record for record in candidates if record[0] not in used]
            selected.extend(
                generator.sample(remaining, min(count - len(selected), len(remaining)))
            )
    else:
        raise ValueError(f"unknown dataset sampling strategy: {strategy}")

    if len(selected) != count:
        raise RuntimeError(f"selected {len(selected)} texts, expected exactly {count}")
    texts = [record[1] for record in selected]
    manifest = {
        "dataset": dataset_config["name"],
        "subset": dataset_config.get("subset"),
        "split": dataset_config.get("split", "train"),
        "revision": dataset_config.get("revision"),
        "text_field": field,
        "sampling": strategy,
        "seed": seed,
        "pool_size_requested": pool_size,
        "usable_pool_rows": len(candidates),
        "num_samples": len(texts),
        "char_limit": char_limit,
        "min_chars": min_chars,
        "selected_stream_indices": [record[0] for record in selected],
        "selected_text_sha256": [sha256_text(text) for text in texts],
        "selected_categories": [record[2] for record in selected],
    }
    return texts, manifest
