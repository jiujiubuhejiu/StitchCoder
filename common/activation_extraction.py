"""Model hooks and disk-backed residual/SAE activation extraction."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np

# Eager attention follows the experiment protocol and keeps activation extraction
# on a consistent numerical path across supported Transformers releases.
EXPERIMENT_ATTENTION_IMPLEMENTATION = "eager"


def _torch_dtype(name: str):
    import torch

    values = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto",
    }
    if name not in values:
        raise ValueError(f"unsupported model dtype: {name}")
    return values[name]


def _decoder_layers(model):
    candidates = [
        ("model", "layers"),
        ("model", "decoder", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for path in candidates:
        value = model
        try:
            for name in path:
                value = getattr(value, name)
        except AttributeError:
            continue
        return value
    raise RuntimeError(
        "could not locate decoder layers; add this architecture to "
        "common.activation_extraction._decoder_layers"
    )


def _load_tokenizer(model_spec: dict[str, Any], *, require_offsets: bool = False):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("activation extraction requires transformers") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["name"],
        revision=model_spec.get("revision"),
        trust_remote_code=bool(model_spec.get("trust_remote_code", True)),
        use_fast=True if require_offsets else model_spec.get("use_fast", True),
    )
    if require_offsets and not tokenizer.is_fast:
        raise RuntimeError("word-level pooling requires a fast tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _model_load_kwargs(model_spec: dict[str, Any], device: str) -> dict[str, Any]:
    """Build deterministic model-loading arguments for the experiment protocol."""

    kwargs: dict[str, Any] = {
        "revision": model_spec.get("revision"),
        "trust_remote_code": bool(model_spec.get("trust_remote_code", True)),
        "low_cpu_mem_usage": True,
        "torch_dtype": _torch_dtype(model_spec.get("dtype", "float16")),
        "device_map": model_spec.get("device_map", device),
    }
    attention_implementation = model_spec.get(
        "attn_implementation", EXPERIMENT_ATTENTION_IMPLEMENTATION
    )
    if attention_implementation:
        kwargs["attn_implementation"] = attention_implementation
    return kwargs


def _load_model(model_spec: dict[str, Any], device: str):
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("activation extraction requires transformers") from exc
    kwargs = _model_load_kwargs(model_spec, device)
    model = AutoModelForCausalLM.from_pretrained(model_spec["name"], **kwargs)
    model.eval()
    return model


def _input_device(model, fallback: str):
    try:
        return model.get_input_embeddings().weight.device
    except (AttributeError, StopIteration):
        try:
            return next(model.parameters()).device
        except StopIteration:
            return fallback


def _clear_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def extract_pooled_hidden(
    model_spec: dict[str, Any],
    texts: list[str],
    *,
    device: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """Mean-pool hooked residuals over valid token positions per text."""

    import torch

    tokenizer = _load_tokenizer(model_spec)
    model = _load_model(model_spec, device)
    layer = _decoder_layers(model)[int(model_spec["layer"])]
    captured: dict[str, Any] = {}

    def hook(_module, _inputs, output):
        captured["hidden"] = (
            output[0] if isinstance(output, tuple) else output
        ).detach()

    handle = layer.register_forward_hook(hook)
    pooled: list[np.ndarray] = []
    input_device = _input_device(model, device)
    try:
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            encoded = {key: value.to(input_device) for key, value in encoded.items()}
            with torch.inference_mode():
                model(**encoded, use_cache=False)
            hidden = captured.pop("hidden").float()
            mask = encoded["attention_mask"].to(hidden.device).float()
            denominator = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            batch_pooled = (hidden * mask[..., None]).sum(dim=1) / denominator
            pooled.append(batch_pooled.cpu().numpy())
    finally:
        handle.remove()
        captured.clear()
        del layer, model, tokenizer
        _clear_accelerator_memory()
    return np.concatenate(pooled, axis=0).astype(np.float32, copy=False)


def _token_row_count(
    tokenizer,
    texts: list[str],
    *,
    batch_size: int,
    max_length: int,
) -> int:
    total = 0
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            padding=False,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        )
        total += sum(sum(mask) for mask in encoded["attention_mask"])
    return int(total)


def extract_native_hidden_and_post(
    model_spec: dict[str, Any],
    texts: list[str],
    encoder_weight: np.ndarray,
    encoder_bias: np.ndarray,
    output_dir: Path,
    *,
    device: str,
    batch_size: int,
    max_length: int,
    prefix: str,
    include_offsets: bool,
) -> dict[str, Path | int | list[int]]:
    """Write exact-size `.npy` arrays for native residual and SAE activations."""

    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = _load_tokenizer(model_spec, require_offsets=include_offsets)
    n_rows = _token_row_count(
        tokenizer, texts, batch_size=batch_size, max_length=max_length
    )
    model = _load_model(model_spec, device)
    layer = _decoder_layers(model)[int(model_spec["layer"])]
    input_device = _input_device(model, device)
    layer_device = next(layer.parameters()).device
    model_dtype = next(parameter for parameter in model.parameters() if parameter.is_floating_point()).dtype
    encoder_device = torch.from_numpy(np.asarray(encoder_weight)).to(
        device=layer_device, dtype=model_dtype
    )
    bias_device = torch.from_numpy(np.asarray(encoder_bias)).to(
        device=layer_device, dtype=model_dtype
    )
    hidden_dim, feature_count = encoder_weight.shape
    paths = {
        "hidden": output_dir / f"{prefix}_hidden.npy",
        "post": output_dir / f"{prefix}_post.npy",
        "sequence_ids": output_dir / f"{prefix}_sequence_ids.npy",
        "token_positions": output_dir / f"{prefix}_token_positions.npy",
        "token_ids": output_dir / f"{prefix}_token_ids.npy",
    }
    if include_offsets:
        paths["offsets"] = output_dir / f"{prefix}_offsets.npy"
    hidden_out = np.lib.format.open_memmap(
        paths["hidden"], mode="w+", dtype=np.float32, shape=(n_rows, hidden_dim)
    )
    post_out = np.lib.format.open_memmap(
        paths["post"], mode="w+", dtype=np.float16, shape=(n_rows, feature_count)
    )
    sequence_out = np.lib.format.open_memmap(
        paths["sequence_ids"], mode="w+", dtype=np.int32, shape=(n_rows,)
    )
    position_out = np.lib.format.open_memmap(
        paths["token_positions"], mode="w+", dtype=np.int32, shape=(n_rows,)
    )
    token_out = np.lib.format.open_memmap(
        paths["token_ids"], mode="w+", dtype=np.int64, shape=(n_rows,)
    )
    offset_out = (
        np.lib.format.open_memmap(
            paths["offsets"], mode="w+", dtype=np.int32, shape=(n_rows, 2)
        )
        if include_offsets
        else None
    )
    captured: dict[str, Any] = {}

    def hook(_module, _inputs, output):
        captured["hidden"] = (
            output[0] if isinstance(output, tuple) else output
        ).detach()

    handle = layer.register_forward_hook(hook)
    cursor = 0
    try:
        for start in range(0, len(texts), batch_size):
            encoded_cpu = tokenizer(
                texts[start : start + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                return_offsets_mapping=include_offsets,
            )
            offsets_cpu = encoded_cpu.pop("offset_mapping", None)
            encoded = {key: value.to(input_device) for key, value in encoded_cpu.items()}
            with torch.inference_mode():
                model(**encoded, use_cache=False)
                hidden = captured.pop("hidden")
                if hidden.shape[-1] != hidden_dim:
                    raise ValueError(
                        f"model hidden size {hidden.shape[-1]} does not match SAE {hidden_dim}"
                    )
                post = torch.relu(hidden.to(model_dtype) @ encoder_device + bias_device)
            mask = encoded["attention_mask"].bool()
            row_count = int(mask.sum().item())
            end = cursor + row_count
            hidden_out[cursor:end] = hidden[mask].float().cpu().numpy()
            post_out[cursor:end] = post[mask].half().cpu().numpy()
            batch_size_actual, sequence_length = mask.shape
            sequence_grid = (
                torch.arange(batch_size_actual, device=mask.device)[:, None]
                .expand(-1, sequence_length)
                + start
            )
            position_grid = torch.arange(sequence_length, device=mask.device)[None, :].expand(
                batch_size_actual, -1
            )
            sequence_out[cursor:end] = sequence_grid[mask].cpu().numpy()
            position_out[cursor:end] = position_grid[mask].cpu().numpy()
            token_out[cursor:end] = encoded["input_ids"][mask].cpu().numpy()
            if offset_out is not None and offsets_cpu is not None:
                offset_out[cursor:end] = offsets_cpu[mask.cpu()].numpy()
            cursor = end
    finally:
        handle.remove()
        for array in (hidden_out, post_out, sequence_out, position_out, token_out, offset_out):
            if array is not None:
                array.flush()
        del encoder_device, bias_device
        captured.clear()
        del layer, model, tokenizer
        _clear_accelerator_memory()
    if cursor != n_rows:
        raise RuntimeError(f"precomputed {n_rows} token rows but wrote {cursor}")
    return {
        **paths,
        "rows": n_rows,
        "hidden_shape": [n_rows, hidden_dim],
        "post_shape": [n_rows, feature_count],
    }
