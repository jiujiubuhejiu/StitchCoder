#!/usr/bin/env python3
"""Render the current TMLR paper figures from the bundled data."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SCRIPTS = (
    "render_alignment_ladder.py",
    "make_minder_pie.py",
    "plot_causal.py",
    "plot_lambda_sweep.py",
    "render_de_scan.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "rendered")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    # Reference assets are the manuscript's originals, not an output folder.
    if output == (ROOT / "reference").resolve():
        parser.error("Choose an output directory other than reference/.")
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PAPER_FIGURES_OUTPUT_DIR"] = str(output)
    for script in SCRIPTS:
        print(f"Rendering {script}", flush=True)
        subprocess.run([sys.executable, str(ROOT / "scripts" / script)], env=env, check=True)
    # The framework is an existing PDF drawing; no matching source was found.
    shutil.copy2(ROOT / "reference" / "coder_framework2.pdf", output / "coder_framework2.pdf")
    print(f"Done: {output}")


if __name__ == "__main__":
    main()
