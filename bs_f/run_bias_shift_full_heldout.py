#!/usr/bin/env python3
"""Document-disjoint evaluation of a frozen Bias-Shift Full correction.

Feature matches, the confidence gate, and the bias delta are estimated on the
first N document IDs and applied unchanged to the remaining documents,
providing a 2,000/2,000 calibration/evaluation split by default.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_bias_shift_full import (
    BiasShiftFullConfig,
    build_post_b_cache,
    compute_full_bias_correction,
    compute_similarity_pass,
    compute_v4_metrics,
    load_prepared_inputs,
    remove_post_b_cache,
)


LOGGER = logging.getLogger("bias_shift_full_heldout")


def document_split(
    sequence_ids: np.ndarray,
    calibration_sequences: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sequence_ids = np.asarray(sequence_ids)
    if sequence_ids.ndim != 1:
        raise ValueError("sequence_ids must be 1-D")
    unique = np.unique(sequence_ids)
    if not 0 < calibration_sequences < len(unique):
        raise ValueError(
            "calibration_sequences must be between 1 and n_sequences-1"
        )
    calibration_ids = unique[:calibration_sequences]
    evaluation_ids = unique[calibration_sequences:]
    calibration_mask = np.isin(sequence_ids, calibration_ids)
    evaluation_mask = ~calibration_mask
    if np.intersect1d(calibration_ids, evaluation_ids).size:
        raise AssertionError("document leakage in held-out split")
    return calibration_mask, evaluation_mask, calibration_ids, evaluation_ids


def _similarities_for_hidden(
    hidden_source: np.ndarray,
    matrix: np.ndarray,
    weight_source: np.ndarray,
    weight_target: np.ndarray,
    bias_target: np.ndarray,
    config: BiasShiftFullConfig,
    work_dir: Path,
):
    hidden_prime = np.asarray(hidden_source @ matrix, dtype=np.float32)
    weight_source_prime = np.asarray(matrix.T @ weight_source, dtype=np.float32)
    cache = build_post_b_cache(
        hidden_prime,
        weight_target,
        bias_target,
        token_chunk_size=config.token_chunk_size,
        work_dir=work_dir,
    )
    return hidden_prime, weight_source_prime, cache


def run_bias_shift_full_heldout(
    hidden_source: np.ndarray,
    sequence_ids: np.ndarray,
    matrix: np.ndarray,
    weight_source: np.ndarray,
    bias_source: np.ndarray,
    weight_target: np.ndarray,
    bias_target: np.ndarray,
    *,
    weights_source: np.ndarray | None,
    weights_target: np.ndarray | None,
    calibration_sequences: int,
    config: BiasShiftFullConfig | None = None,
    work_dir: Path | str = Path("."),
) -> dict[str, Any]:
    config = config or BiasShiftFullConfig()
    config.validate()
    hidden_source = np.asarray(hidden_source, dtype=np.float32)
    sequence_ids = np.asarray(sequence_ids)
    if sequence_ids.shape != (len(hidden_source),):
        raise ValueError("sequence_ids must match hidden_source rows")
    calibration_mask, evaluation_mask, calibration_ids, evaluation_ids = (
        document_split(sequence_ids, calibration_sequences)
    )
    started = time.time()
    work_dir = Path(work_dir)

    hidden_cal, weight_source_prime, calibration_cache = _similarities_for_hidden(
        hidden_source[calibration_mask],
        matrix,
        weight_source,
        weight_target,
        bias_target,
        config,
        work_dir,
    )
    try:
        calibration_off = compute_similarity_pass(
            hidden_cal,
            weight_source_prime,
            bias_source,
            post_b_cache=calibration_cache,
            feature_chunk_size=config.feature_chunk_size,
            token_chunk_size=config.token_chunk_size,
            dead_threshold=config.sample_dead_threshold,
        )
        steer_mask, delta, margins = compute_full_bias_correction(
            hidden_cal,
            weight_source_prime,
            bias_source,
            weight_target,
            bias_target,
            calibration_off,
            config,
        )
        calibration_on = compute_similarity_pass(
            hidden_cal,
            weight_source_prime,
            bias_source + delta,
            post_b_cache=calibration_cache,
            feature_chunk_size=config.feature_chunk_size,
            token_chunk_size=config.token_chunk_size,
            dead_threshold=config.sample_dead_threshold,
        )
    finally:
        remove_post_b_cache(calibration_cache)

    hidden_eval, weight_source_prime_eval, evaluation_cache = (
        _similarities_for_hidden(
            hidden_source[evaluation_mask],
            matrix,
            weight_source,
            weight_target,
            bias_target,
            config,
            work_dir,
        )
    )
    if not np.allclose(weight_source_prime_eval, weight_source_prime):
        raise RuntimeError(
            "transported source weights must remain fixed across document partitions"
        )
    try:
        evaluation_off = compute_similarity_pass(
            hidden_eval,
            weight_source_prime,
            bias_source,
            post_b_cache=evaluation_cache,
            feature_chunk_size=config.feature_chunk_size,
            token_chunk_size=config.token_chunk_size,
            dead_threshold=config.sample_dead_threshold,
        )
        evaluation_on = compute_similarity_pass(
            hidden_eval,
            weight_source_prime,
            bias_source + delta,
            post_b_cache=evaluation_cache,
            feature_chunk_size=config.feature_chunk_size,
            token_chunk_size=config.token_chunk_size,
            dead_threshold=config.sample_dead_threshold,
        )
    finally:
        remove_post_b_cache(evaluation_cache)

    def metrics(similarity):
        return compute_v4_metrics(
            similarity,
            weights_a=weights_source,
            weights_b=weights_target,
            native_dead_threshold=config.native_dead_threshold,
        )

    calibration_off_metrics = metrics(calibration_off)
    calibration_on_metrics = metrics(calibration_on)
    evaluation_off_metrics = metrics(evaluation_off)
    evaluation_on_metrics = metrics(evaluation_on)
    summary = {
        "method": "Bias-Shift Full (BS-F), document-disjoint",
        "protocol": "frozen_match_gate_delta",
        "config": asdict(config),
        "split": {
            "calibration_sequences": int(len(calibration_ids)),
            "evaluation_sequences": int(len(evaluation_ids)),
            "calibration_rows": int(calibration_mask.sum()),
            "evaluation_rows": int(evaluation_mask.sum()),
            "document_overlap": 0,
        },
        "decision_freeze": {
            "match": "calibration argmax",
            "gate": "calibration top1/top2/nondead",
            "delta": "calibration median pre-activation mismatch",
            "evaluation_refit": False,
        },
        "calibration": {
            "baseline": calibration_off_metrics,
            "calibrated": calibration_on_metrics,
            "delta_f1": float(
                calibration_on_metrics["f1"] - calibration_off_metrics["f1"]
            ),
        },
        "heldout": {
            "baseline": evaluation_off_metrics,
            "calibrated": evaluation_on_metrics,
            "delta_f1": float(
                evaluation_on_metrics["f1"] - evaluation_off_metrics["f1"]
            ),
        },
        "n_shifted": int(steer_mask.sum()),
        "elapsed_seconds": float(time.time() - started),
    }
    arrays = {
        "calibration_max_sim_a_baseline": calibration_off.max_sim_a,
        "calibration_max_sim_b_baseline": calibration_off.max_sim_b,
        "calibration_argmax_a_baseline": calibration_off.argmax_a,
        "calibration_second_sim_a_baseline": calibration_off.second_sim_a,
        "calibration_margins": margins,
        "frozen_steer_mask": steer_mask,
        "frozen_delta": delta,
        "evaluation_max_sim_a_baseline": evaluation_off.max_sim_a,
        "evaluation_max_sim_b_baseline": evaluation_off.max_sim_b,
        "evaluation_max_sim_a_calibrated": evaluation_on.max_sim_a,
        "evaluation_max_sim_b_calibrated": evaluation_on.max_sim_b,
        "evaluation_argmax_a_baseline": evaluation_off.argmax_a,
        "evaluation_argmax_a_calibrated": evaluation_on.argmax_a,
        "calibration_sequence_ids": calibration_ids,
        "evaluation_sequence_ids": evaluation_ids,
    }
    return {"summary": summary, "arrays": arrays}


def save_outputs(result: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "bias_shift_full_heldout_summary.json"
    arrays_path = output_dir / "bias_shift_full_heldout_arrays.npz"
    summary_path.write_text(
        json.dumps(result["summary"], indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(arrays_path, **result["arrays"])
    return summary_path, arrays_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/bs_f_heldout")
    )
    parser.add_argument("--calibration-sequences", type=int, default=2000)
    parser.add_argument("--steer-threshold", type=float, default=0.10)
    parser.add_argument("--margin-threshold", type=float, default=0.05)
    parser.add_argument("--delta-clip", type=float, default=4.0)
    parser.add_argument("--sample-dead-threshold", type=float, default=1e-6)
    parser.add_argument("--native-dead-threshold", type=float, default=1e-7)
    parser.add_argument("--feature-chunk-size", type=int, default=256)
    parser.add_argument("--token-chunk-size", type=int, default=1024)
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
    values = load_prepared_inputs(args.input_dir)
    sequence_path = args.input_dir / "sequence_ids.npy"
    if not sequence_path.is_file():
        raise FileNotFoundError(f"missing held-out split input: {sequence_path}")
    sequence_ids = np.load(sequence_path, mmap_mode="r", allow_pickle=False)
    result = run_bias_shift_full_heldout(
        values["H_a"],
        sequence_ids,
        values["M"],
        values["W_a"],
        values["b_a"],
        values["W_b"],
        values["b_b"],
        weights_source=values["weights_a"],
        weights_target=values["weights_b"],
        calibration_sequences=args.calibration_sequences,
        config=BiasShiftFullConfig(
            steer_threshold=args.steer_threshold,
            margin_threshold=args.margin_threshold,
            delta_clip=args.delta_clip,
            sample_dead_threshold=args.sample_dead_threshold,
            native_dead_threshold=args.native_dead_threshold,
            feature_chunk_size=args.feature_chunk_size,
            token_chunk_size=args.token_chunk_size,
        ),
        work_dir=args.output_dir,
    )
    summary_path, arrays_path = save_outputs(result, args.output_dir)
    heldout = result["summary"]["heldout"]
    print(
        "BS-F held-out complete: "
        f"baseline={heldout['baseline']['f1']:.6f}, "
        f"calibrated={heldout['calibrated']['f1']:.6f}, "
        f"delta={heldout['delta_f1']:+.6f}"
    )
    print(f"summary: {summary_path.resolve()}")
    print(f"arrays:  {arrays_path.resolve()}")


if __name__ == "__main__":
    main()
