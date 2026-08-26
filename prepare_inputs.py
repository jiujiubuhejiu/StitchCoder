#!/usr/bin/env python3
"""Prepare the array artifacts consumed by BS-F and BS-R.

This stage loads the configured models, SAEs, and text datasets, then
materializes the aligned arrays used by the two methods.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from common.activation_extraction import (
    EXPERIMENT_ATTENTION_IMPLEMENTATION,
    extract_native_hidden_and_post,
    extract_pooled_hidden,
)
from common.alignment import fit_semiorthogonal_procrustes
from common.data_utils import load_texts
from common.io_utils import array_record, read_json, write_json
from common.metrics import activation_frequencies
from common.sae_loading import load_sae_encoder
from common.word_alignment import align_word_rows, pool_by_whitespace_words


LOGGER = logging.getLogger("prepare_bias_shift_inputs")



def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_cell_config(config_path: Path, cell_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    document = read_json(config_path)
    cells = document.get("cells", {})
    if cell_name not in cells:
        raise KeyError(
            f"unknown cell {cell_name!r}; available: {', '.join(sorted(cells))}"
        )
    config = deep_merge(document.get("defaults", {}), cells[cell_name])
    config["cell"] = cell_name
    for model_key in ("model_a", "model_b"):
        config[model_key].setdefault(
            "attn_implementation", EXPERIMENT_ATTENTION_IMPLEMENTATION
        )
    return config, document


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _encode_hidden_to_post(
    hidden_path: Path,
    weight: np.ndarray,
    bias: np.ndarray,
    output_path: Path,
    *,
    row_chunk_size: int = 4096,
) -> Path:
    hidden = np.load(hidden_path, mmap_mode="r", allow_pickle=False)
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float16,
        shape=(hidden.shape[0], weight.shape[1]),
    )
    for start in range(0, len(hidden), row_chunk_size):
        end = min(start + row_chunk_size, len(hidden))
        pre = np.asarray(hidden[start:end], dtype=np.float32) @ weight
        pre += bias[None, :]
        np.maximum(pre, 0.0, out=pre)
        output[start:end] = pre.astype(np.float16)
    output.flush()
    return output_path


def _save_selected_rows(
    source: np.ndarray,
    indices: np.ndarray,
    output_path: Path,
    *,
    row_chunk_size: int = 4096,
) -> Path:
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=source.dtype,
        shape=(len(indices), source.shape[1]),
    )
    for start in range(0, len(indices), row_chunk_size):
        end = min(start + row_chunk_size, len(indices))
        output[start:end] = source[np.asarray(indices[start:end])]
    output.flush()
    return output_path


def _same_model_hook(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = ("name", "revision", "layer", "dtype")
    return all(left.get(key) == right.get(key) for key in keys)


def _same_sae(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = ("loader", "release", "sae_id", "revision")
    return all(left.get(key) == right.get(key) for key in keys)



def _check_token_alignment(side_a: dict[str, Any], side_b: dict[str, Any]) -> None:
    for key in ("sequence_ids", "token_positions", "token_ids"):
        left = np.load(side_a[key], mmap_mode="r", allow_pickle=False)
        right = np.load(side_b[key], mmap_mode="r", allow_pickle=False)
        if left.shape != right.shape or not np.array_equal(left, right):
            raise RuntimeError(
                f"same-tokenizer BS-R alignment failed for {key}; "
                "use scoring.token_pooling=whitespace_word for different tokenizers"
            )


def _write_text_manifest(path: Path, metadata: dict[str, Any]) -> None:
    write_json(path, metadata)


def prepare_cell(
    config_path: Path,
    cell_name: str,
    output_root: Path,
    *,
    device_a: str,
    device_b: str,
    checksums: bool,
    keep_native_cache: bool,
) -> Path:
    config, protocol_document = load_cell_config(config_path, cell_name)
    cell_dir = output_root / cell_name
    completion = cell_dir / "preparation_complete.json"
    if completion.exists():
        LOGGER.info("%s already has a completion manifest; skipping", cell_name)
        return cell_dir
    if cell_dir.exists() and any(cell_dir.iterdir()):
        raise RuntimeError(
            f"output directory must be empty before preparation: {cell_dir}"
        )
    cell_dir.mkdir(parents=True, exist_ok=True)
    alignment_dir = cell_dir / "alignment"
    native_dir = cell_dir / "native_cache"
    bsf_dir = cell_dir / "bs_f"
    bsr_dir = cell_dir / "bs_r"
    for directory in (alignment_dir, native_dir, bsf_dir, bsr_dir):
        directory.mkdir(parents=True, exist_ok=True)

    LOGGER.info("[%s] loading alignment texts", cell_name)
    alignment_texts, alignment_text_manifest = load_texts(config["alignment_dataset"])
    _write_text_manifest(alignment_dir / "text_manifest.json", alignment_text_manifest)
    alignment_options = config["alignment"]
    LOGGER.info("[%s] extracting source alignment residuals", cell_name)
    pooled_a = extract_pooled_hidden(
        config["model_a"],
        alignment_texts,
        device=device_a,
        batch_size=int(alignment_options["batch_size"]),
        max_length=int(alignment_options["max_length"]),
    )
    same_model_hook = _same_model_hook(config["model_a"], config["model_b"])
    if same_model_hook:
        pooled_b = pooled_a.copy()
    else:
        LOGGER.info("[%s] extracting target alignment residuals", cell_name)
        pooled_b = extract_pooled_hidden(
            config["model_b"],
            alignment_texts,
            device=device_b,
            batch_size=int(alignment_options["batch_size"]),
            max_length=int(alignment_options["max_length"]),
        )
    matrix, alignment_diagnostics = fit_semiorthogonal_procrustes(
        pooled_a,
        pooled_b,
        normalize_rows=alignment_options["preprocessing"] == "l2_row_normalize",
    )
    np.save(alignment_dir / "pooled_a.npy", pooled_a)
    np.save(alignment_dir / "pooled_b.npy", pooled_b)
    np.save(alignment_dir / "alignment_matrix.npy", matrix)
    write_json(alignment_dir / "diagnostics.json", alignment_diagnostics)
    del pooled_a, pooled_b

    LOGGER.info("[%s] loading SAE encoders", cell_name)
    same_sae = _same_sae(config["sae_a"], config["sae_b"])
    sae_a = load_sae_encoder(config["sae_a"])
    if same_sae:
        sae_b = sae_a
    else:
        sae_b = load_sae_encoder(config["sae_b"])
    np.save(bsf_dir / "W_a.npy", sae_a.weight)
    np.save(bsf_dir / "b_a.npy", sae_a.bias)
    np.save(bsf_dir / "W_b.npy", sae_b.weight)
    np.save(bsf_dir / "b_b.npy", sae_b.bias)
    np.save(bsf_dir / "alignment_matrix.npy", matrix)

    LOGGER.info("[%s] loading C4 scoring texts", cell_name)
    scoring_texts, scoring_text_manifest = load_texts(config["scoring_dataset"])
    _write_text_manifest(cell_dir / "scoring_text_manifest.json", scoring_text_manifest)
    scoring = config["scoring"]
    word_pooling = scoring["token_pooling"] == "whitespace_word"
    side_a = extract_native_hidden_and_post(
        config["model_a"],
        scoring_texts,
        sae_a.weight,
        sae_a.bias,
        native_dir,
        device=device_a,
        batch_size=int(scoring["batch_size"]),
        max_length=int(scoring["max_length"]),
        prefix="a",
        include_offsets=word_pooling,
    )
    if _same_model_hook(config["model_a"], config["model_b"]):
        side_b = {
            "hidden": native_dir / "b_hidden.npy",
            "post": native_dir / "b_post.npy",
            "sequence_ids": native_dir / "b_sequence_ids.npy",
            "token_positions": native_dir / "b_token_positions.npy",
            "token_ids": native_dir / "b_token_ids.npy",
        }
        if word_pooling:
            side_b["offsets"] = native_dir / "b_offsets.npy"
        for key in ("hidden", "sequence_ids", "token_positions", "token_ids"):
            _link_or_copy(Path(side_a[key]), Path(side_b[key]))
        if word_pooling:
            _link_or_copy(Path(side_a["offsets"]), Path(side_b["offsets"]))
        if _same_sae(config["sae_a"], config["sae_b"]):
            _link_or_copy(Path(side_a["post"]), Path(side_b["post"]))
        else:
            _encode_hidden_to_post(
                Path(side_b["hidden"]), sae_b.weight, sae_b.bias, Path(side_b["post"])
            )
    else:
        side_b = extract_native_hidden_and_post(
            config["model_b"],
            scoring_texts,
            sae_b.weight,
            sae_b.bias,
            native_dir,
            device=device_b,
            batch_size=int(scoring["batch_size"]),
            max_length=int(scoring["max_length"]),
            prefix="b",
            include_offsets=word_pooling,
        )

    post_a_native = np.load(side_a["post"], mmap_mode="r", allow_pickle=False)
    post_b_native = np.load(side_b["post"], mmap_mode="r", allow_pickle=False)
    weights_a = activation_frequencies(
        post_a_native, threshold=float(config["bs_f"]["native_dead_threshold"])
    )
    weights_b = activation_frequencies(
        post_b_native, threshold=float(config["bs_f"]["native_dead_threshold"])
    )
    np.save(bsf_dir / "feature_weights_a.npy", weights_a)
    np.save(bsf_dir / "feature_weights_b.npy", weights_b)
    _link_or_copy(Path(side_a["hidden"]), bsf_dir / "H_a.npy")
    _link_or_copy(Path(side_a["sequence_ids"]), bsf_dir / "sequence_ids.npy")

    if not word_pooling:
        _check_token_alignment(side_a, side_b)
        _link_or_copy(Path(side_a["post"]), bsr_dir / "post_a.npy")
        _link_or_copy(Path(side_b["post"]), bsr_dir / "post_b.npy")
        _link_or_copy(Path(side_a["sequence_ids"]), bsr_dir / "sequence_ids.npy")
        aligned_rows = int(len(post_a_native))
    else:
        pooled_a_path = native_dir / "a_word_post.npy"
        pooled_b_path = native_dir / "b_word_post.npy"
        pooled_a, word_sid_a, word_idx_a = pool_by_whitespace_words(
            post_a_native,
            np.load(side_a["sequence_ids"], mmap_mode="r"),
            np.load(side_a["offsets"], mmap_mode="r"),
            scoring_texts,
            output_path=pooled_a_path,
        )
        pooled_b, word_sid_b, word_idx_b = pool_by_whitespace_words(
            post_b_native,
            np.load(side_b["sequence_ids"], mmap_mode="r"),
            np.load(side_b["offsets"], mmap_mode="r"),
            scoring_texts,
            output_path=pooled_b_path,
        )
        indices_a, indices_b, aligned_sequence_ids = align_word_rows(
            word_sid_a, word_idx_a, word_sid_b, word_idx_b
        )
        _save_selected_rows(pooled_a, indices_a, bsr_dir / "post_a.npy")
        _save_selected_rows(pooled_b, indices_b, bsr_dir / "post_b.npy")
        np.save(bsr_dir / "sequence_ids.npy", aligned_sequence_ids)
        aligned_rows = int(len(aligned_sequence_ids))
        del pooled_a, pooled_b

    # Release native-cache memmap handles before removing the cache directory.
    del post_a_native, post_b_native

    artifact_paths = [
        bsf_dir / filename
        for filename in (
            "H_a.npy",
            "alignment_matrix.npy",
            "W_a.npy",
            "b_a.npy",
            "W_b.npy",
            "b_b.npy",
            "feature_weights_a.npy",
            "feature_weights_b.npy",
            "sequence_ids.npy",
        )
    ] + [
        bsr_dir / "post_a.npy",
        bsr_dir / "post_b.npy",
        bsr_dir / "sequence_ids.npy",
    ]
    artifacts = {}
    for path in artifact_paths:
        if checksums:
            artifacts[str(path.relative_to(cell_dir))] = array_record(path)
        else:
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            artifacts[str(path.relative_to(cell_dir))] = {
                "path": str(path), "shape": list(values.shape), "dtype": str(values.dtype)
            }
    completion_payload = {
        "protocol": protocol_document.get("protocol"),
        "protocol_version": protocol_document.get("protocol_version"),
        "cell": cell_name,
        "scenario": config.get("scenario"),
        "config": config,
        "sae_a": sae_a.metadata,
        "sae_b": sae_b.metadata,
        "alignment": alignment_diagnostics,
        "bs_r_aligned_rows": aligned_rows,
        "artifacts": artifacts,
    }
    if not keep_native_cache:
        for path in list(native_dir.iterdir()):
            path.unlink()
        native_dir.rmdir()
    write_json(completion, completion_payload)
    LOGGER.info("[%s] preparation complete: %s", cell_name, completion)
    return cell_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PACKAGE_ROOT / "configs" / "paper_experiments.json",
    )
    parser.add_argument("--cell", action="append", help="Repeat for multiple cells")
    parser.add_argument("--all", action="store_true", help="Prepare every configured cell")
    parser.add_argument("--output-root", type=Path, default=Path("prepared"))
    parser.add_argument("--device-a", default="cuda")
    parser.add_argument("--device-b", default="cuda")
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument("--keep-native-cache", action="store_true")
    parser.add_argument("--list-cells", action="store_true")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    document = read_json(args.config)
    available = sorted(document.get("cells", {}))
    if args.list_cells:
        print("\n".join(available))
        return
    if args.all:
        selected = available
    else:
        selected = args.cell or []
    if not selected:
        raise SystemExit("select at least one --cell or pass --all")
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise SystemExit(f"unknown cells: {', '.join(unknown)}")
    for cell_name in selected:
        prepare_cell(
            args.config,
            cell_name,
            args.output_root,
            device_a=args.device_a,
            device_b=args.device_b,
            checksums=not args.skip_checksums,
            keep_native_cache=args.keep_native_cache,
        )


if __name__ == "__main__":
    main()
