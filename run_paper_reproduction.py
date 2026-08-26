#!/usr/bin/env python3
"""Run configured model comparisons through BS-F, BS-R, or both methods."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from bs_f.run_bias_shift_full import (
    BiasShiftFullConfig,
    load_prepared_inputs as load_bsf_inputs,
    run_bias_shift_full,
    save_outputs as save_bsf_outputs,
)
from bs_r.run_bias_shift_ridge import (
    BiasShiftRidgeConfig,
    load_prepared_inputs as load_bsr_inputs,
    run_bias_shift_ridge,
    save_outputs as save_bsr_outputs,
)
from common.io_utils import read_json, write_json
from prepare_inputs import deep_merge


LOGGER = logging.getLogger("stitchcoder_model_comparisons")


def required_prepared_files(cell_dir: Path, method: str) -> list[Path]:
    files: list[Path] = [cell_dir / "preparation_complete.json"]
    if method in {"bs_f", "both"}:
        files.extend(
            cell_dir / "bs_f" / name
            for name in (
                "H_a.npy",
                "alignment_matrix.npy",
                "W_a.npy",
                "b_a.npy",
                "W_b.npy",
                "b_b.npy",
                "feature_weights_a.npy",
                "feature_weights_b.npy",
            )
        )
    if method in {"bs_r", "both"}:
        files.extend(
            cell_dir / "bs_r" / name
            for name in ("post_a.npy", "post_b.npy", "sequence_ids.npy")
        )
    return files


def validate_prepared_cell(cell_dir: Path, method: str) -> None:
    missing = [path for path in required_prepared_files(cell_dir, method) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "prepared cell is incomplete; missing: "
            + ", ".join(str(path) for path in missing)
        )


def run_cell(
    cell_name: str,
    config: dict[str, Any],
    *,
    prepared_root: Path,
    output_root: Path,
    method: str,
    backend: str,
    device: str,
    ridge_lambda: float | None,
    row_shuffle: bool,
    source_feature_limit: int | None,
) -> dict[str, Any]:
    prepared_dir = prepared_root / cell_name
    output_dir = output_root / cell_name
    validate_prepared_cell(prepared_dir, method)
    result: dict[str, Any] = {
        "cell": cell_name,
        "scenario": config.get("scenario"),
    }

    if method in {"bs_f", "both"}:
        values = load_bsf_inputs(prepared_dir / "bs_f")
        options = config["bs_f"]
        bsf_config = BiasShiftFullConfig(
            steer_threshold=float(options["steer_threshold"]),
            margin_threshold=float(options["margin_threshold"]),
            delta_clip=float(options["delta_clip"]),
            sample_dead_threshold=float(options["sample_dead_threshold"]),
            native_dead_threshold=float(options["native_dead_threshold"]),
        )
        bsf_result = run_bias_shift_full(
            values["H_a"],
            values["M"],
            values["W_a"],
            values["b_a"],
            values["W_b"],
            values["b_b"],
            weights_a=values["weights_a"],
            weights_b=values["weights_b"],
            config=bsf_config,
            work_dir=output_dir / "bs_f",
        )
        save_bsf_outputs(bsf_result, output_dir / "bs_f")
        result["bs_f"] = bsf_result["summary"]

    if method in {"bs_r", "both"}:
        post_a, post_b, sequence_ids = load_bsr_inputs(prepared_dir / "bs_r")
        options = config["bs_r"]
        bsr_config = BiasShiftRidgeConfig(
            ridge_lambda=(
                float(ridge_lambda)
                if ridge_lambda is not None
                else float(options["ridge_lambda"])
            ),
            train_fraction=float(options["train_fraction"]),
            seed=int(options["seed"]),
            sample_dead_threshold=float(options["sample_dead_threshold"]),
            native_dead_threshold=float(options["native_dead_threshold"]),
            backend=backend,
            device=device,
            row_shuffle=row_shuffle,
            source_feature_limit=source_feature_limit,
        )
        bsr_result = run_bias_shift_ridge(
            post_a,
            post_b,
            sequence_ids,
            config=bsr_config,
            work_dir=output_dir / "bs_r",
        )
        save_bsr_outputs(bsr_result, output_dir / "bs_r")
        result["bs_r"] = bsr_result["summary"]
    write_json(output_dir / "cell_summary.json", result)
    return result


def summary_row(result: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cell": result["cell"],
        "scenario": result.get("scenario", ""),
        "bs_f_base_f1": "",
        "bs_f_full_f1": "",
        "bs_f_precision": "",
        "bs_f_recall": "",
        "bs_r_f1": "",
        "bs_r_precision": "",
        "bs_r_recall": "",
        "bs_r_ssr": "",
    }
    if "bs_f" in result:
        row.update(
            {
                "bs_f_base_f1": result["bs_f"]["baseline"]["f1"],
                "bs_f_full_f1": result["bs_f"]["calibrated"]["f1"],
                "bs_f_precision": result["bs_f"]["calibrated"]["precision"],
                "bs_f_recall": result["bs_f"]["calibrated"]["recall"],
            }
        )
    if "bs_r" in result:
        row.update(
            {
                "bs_r_f1": result["bs_r"]["metrics"]["f1"],
                "bs_r_precision": result["bs_r"]["metrics"]["precision"],
                "bs_r_recall": result["bs_r"]["metrics"]["recall"],
                "bs_r_ssr": result["bs_r"]["self_slot"]["self_slot_recovery"],
            }
        )
    return row


def compare_with_golden(
    rows: list[dict[str, Any]],
    golden_path: Path,
) -> dict[str, Any]:
    golden = read_json(golden_path)
    tolerance = float(golden["tolerance"]["absolute_f1"])
    comparisons = []
    all_within = True
    for row in rows:
        expected = golden.get("cells", {}).get(row["cell"])
        if expected is None:
            continue
        for key in ("bs_f_base_f1", "bs_f_full_f1", "bs_r_f1"):
            if row[key] == "":
                continue
            absolute_error = abs(float(row[key]) - float(expected[key]))
            within = absolute_error <= tolerance
            all_within &= within
            comparisons.append(
                {
                    "cell": row["cell"],
                    "metric": key,
                    "observed": float(row[key]),
                    "expected": float(expected[key]),
                    "absolute_error": absolute_error,
                    "within_tolerance": within,
                }
            )
    return {
        "golden_path": str(golden_path),
        "absolute_tolerance": tolerance,
        "all_within_tolerance": all_within,
        "comparisons": comparisons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PACKAGE_ROOT / "configs" / "paper_experiments.json",
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=PACKAGE_ROOT / "configs" / "golden_main_results.json",
    )
    parser.add_argument("--prepared-root", type=Path, default=Path("prepared"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/paper_reproduction"))
    parser.add_argument("--cell", action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--method", choices=["bs_f", "bs_r", "both"], default="both")
    parser.add_argument("--backend", choices=["auto", "numpy", "cuda"], default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ridge-lambda", type=float)
    parser.add_argument("--row-shuffle", action="store_true")
    parser.add_argument("--source-feature-limit", type=int)
    parser.add_argument(
        "--skip-golden",
        action="store_true",
        help="Run without reference-metric comparison",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate prepared files and report readiness")
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
    selected = available if args.all else (args.cell or [])
    if not selected:
        raise SystemExit("select at least one --cell or pass --all")
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise SystemExit(f"unknown cells: {', '.join(unknown)}")

    if args.dry_run:
        for cell_name in selected:
            validate_prepared_cell(args.prepared_root / cell_name, args.method)
            print(f"OK {cell_name}")
        return

    results = []
    for cell_name in selected:
        config = deep_merge(document.get("defaults", {}), document["cells"][cell_name])
        results.append(
            run_cell(
                cell_name,
                config,
                prepared_root=args.prepared_root,
                output_root=args.output_root,
                method=args.method,
                backend=args.backend,
                device=args.device,
                ridge_lambda=args.ridge_lambda,
                row_shuffle=args.row_shuffle,
                source_feature_limit=args.source_feature_limit,
            )
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = [summary_row(result) for result in results]
    fieldnames = list(rows[0])
    with (args.output_root / "paper_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    comparison = (
        {
            "skipped": True,
            "reason": "reference comparison disabled by --skip-golden",
        }
        if args.skip_golden
        else compare_with_golden(rows, args.golden)
    )
    write_json(
        args.output_root / "paper_results.json",
        {"results": results, "golden_comparison": comparison},
    )
    print(f"results: {(args.output_root / 'paper_results.csv').resolve()}")
    if args.skip_golden:
        print("golden comparison: skipped")
    else:
        print(f"golden tolerance satisfied: {comparison['all_within_tolerance']}")


if __name__ == "__main__":
    main()
