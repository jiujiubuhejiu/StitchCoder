"""Load SAE encoder matrices and biases from configured artifacts."""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SAEEncoder:
    weight: np.ndarray
    bias: np.ndarray
    metadata: dict[str, Any]


def clear_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _extract_encoder(sae: Any) -> tuple[np.ndarray, np.ndarray]:
    weight = sae.W_enc.detach().float().cpu().numpy()
    bias_value = getattr(sae, "b_enc", None)
    bias = (
        bias_value.detach().float().cpu().numpy()
        if bias_value is not None
        else np.zeros(weight.shape[1], dtype=np.float32)
    )
    return weight.astype(np.float32, copy=False), bias.astype(np.float32, copy=False)


def _expanded(value: str) -> str:
    expanded = os.path.expanduser(os.path.expandvars(value))
    if "$" in expanded:
        raise RuntimeError(
            f"unresolved environment variable in SAE path: {value}. "
            "Set the path named in configs/paper_experiments.json."
        )
    return expanded


def load_sae_encoder(spec: dict[str, Any]) -> SAEEncoder:
    """Load an SAE through the configured ``sae_lens``, ``npz``, or ``hf_ae_pt`` backend."""

    loader = spec.get("loader", "sae_lens")
    release = _expanded(spec.get("release", spec.get("path", "")))
    sae_id = spec.get("sae_id", "")
    revision = spec.get("revision")

    if loader == "npz":
        path = Path(release)
        with np.load(path, allow_pickle=False) as artifact:
            weight = np.asarray(artifact["W_enc"], dtype=np.float32)
            bias = (
                np.asarray(artifact["b_enc"], dtype=np.float32)
                if "b_enc" in artifact
                else np.zeros(weight.shape[1], dtype=np.float32)
            )
        source = str(path.resolve())
    elif loader == "hf_ae_pt":
        try:
            import torch
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError("hf_ae_pt loading requires torch and huggingface_hub") from exc
        filename = spec.get("filename", f"{sae_id}/ae.pt")
        path = hf_hub_download(repo_id=release, filename=filename, revision=revision)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if "encoder.weight" in state:
            weight = state["encoder.weight"].float().numpy().T
            bias = (
                state["encoder.bias"].float().numpy()
                if "encoder.bias" in state
                else np.zeros(weight.shape[1], dtype=np.float32)
            )
        elif "W_enc" in state:
            weight = state["W_enc"].float().numpy()
            bias = (
                state["b_enc"].float().numpy()
                if "b_enc" in state
                else np.zeros(weight.shape[1], dtype=np.float32)
            )
        else:
            raise RuntimeError(f"unknown SAE state keys: {sorted(state)}")
        source = path
    elif loader == "sae_lens":
        try:
            from sae_lens import SAE
            from sae_lens.loading import pretrained_sae_loaders
        except ImportError as exc:
            raise RuntimeError("SAE loading requires the `sae-lens` package") from exc
        local = Path(release)
        if local.is_dir():
            sae = SAE.load_from_disk(path=str(local), device="cpu")
            source = str(local.resolve())
        else:
            expected_repo = spec.get("repo_id")
            original_download = pretrained_sae_loaders.hf_hub_download
            pinned_downloads: list[str] = []

            def pinned_hf_hub_download(repo_id, filename, *args, **kwargs):
                if revision is not None and repo_id == expected_repo:
                    kwargs["revision"] = revision
                downloaded = original_download(repo_id, filename, *args, **kwargs)
                if repo_id == expected_repo:
                    pinned_downloads.append(str(downloaded))
                return downloaded

            pretrained_sae_loaders.hf_hub_download = pinned_hf_hub_download
            try:
                loaded = SAE.from_pretrained(
                    release=release, sae_id=sae_id, device="cpu"
                )
            finally:
                pretrained_sae_loaders.hf_hub_download = original_download
            if revision is not None and expected_repo and not pinned_downloads:
                raise RuntimeError(
                    "SAE revision pin could not be verified; no files were loaded "
                    f"from expected repository {expected_repo}"
                )
            sae = loaded[0] if isinstance(loaded, tuple) else loaded
            source = f"{release}/{sae_id}@{revision or 'default'}"
        weight, bias = _extract_encoder(sae)
        del sae
        clear_accelerator_memory()
    else:
        raise ValueError(f"unsupported SAE loader: {loader}")

    weight = np.asarray(weight, dtype=np.float32)
    bias = np.asarray(bias, dtype=np.float32)
    if weight.ndim != 2 or bias.shape != (weight.shape[1],):
        raise ValueError(f"invalid SAE encoder shapes: {weight.shape}, {bias.shape}")
    if not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError("SAE encoder contains NaN or infinity")
    expected_features = spec.get("features")
    if expected_features is not None and weight.shape[1] != int(expected_features):
        raise ValueError(
            f"SAE feature count {weight.shape[1]} does not match expected "
            f"{expected_features}"
        )
    return SAEEncoder(
        weight=weight,
        bias=bias,
        metadata={
            "loader": loader,
            "source": source,
            "sae_id": sae_id,
            "revision": revision,
            "weight_shape": list(weight.shape),
            "bias_shape": list(bias.shape),
        },
    )
