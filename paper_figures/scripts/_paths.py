"""Package-local paths shared by the paper figure renderers."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = Path(os.environ.get("PAPER_FIGURES_OUTPUT_DIR", ROOT / "rendered")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
