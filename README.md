# StitchCoder: Cross-Model Sparse Feature Alignment

StitchCoder is an experimental toolkit for comparing how independently trained sparse autoencoders (SAEs) represent the same or related concepts across language models.

Each model's residual stream and SAE dictionary occupy a distinct coordinate system, so feature indices and directions are not directly comparable. Training a new joint dictionary for every model pair also scales poorly as the number of models grows. StitchCoder instead learns a geometric map between residual spaces with semi-orthogonal Procrustes alignment, then reuses existing SAEs to analyze feature correspondence. This design supports scalable comparisons across identical models, training stages, model sizes, and architectures.

The experiments provide two complementary views. Bias-Shift Forward (BS-F) measures direct correspondence between individual features, while Bias-Shift Ridge (BS-R) measures whether a target concept can be reconstructed jointly from a set of source features. Together, they distinguish weak one-to-one matching from distributed concept preservation and provide quantitative evidence for representation auditing, model-difference analysis, and candidate model-specific feature discovery.

## Multi-Model Comparison Suite

The default configuration contains nine directional experiments:

| Comparison | Model and SAE setup | Directions |
| --- | --- | ---: |
| Same-model reference | Gemma 2 2B paired with the same SAE | 1 |
| Same model, different SAEs | Canonical and Matryoshka SAEs for Gemma 2 2B | 2 |
| Base vs. instruction-tuned | Gemma 2 2B and Gemma 2 2B-IT | 2 |
| Cross-scale | Gemma 2 2B and Gemma 2 9B | 2 |
| Cross-architecture | Gemma 2 2B and Llama 3.1 8B | 2 |

Residual-space alignment uses deterministically sampled Pile text, and feature scoring uses C4 validation text. The default settings use 6,000 alignment documents and 4,000 scoring documents. Cross-scale and cross-architecture configurations use 20,000 stratified documents to fit the alignment matrix. Model identifiers, datasets, SAE revisions, random seeds, and primary hyperparameters are defined in `configs/paper_experiments.json`.

## Core Methods

BS-F projects source SAE decoder directions into the target residual space, measures feature-pair correspondence with cosine similarity, and applies bias correction. Activation-frequency-weighted bidirectional greedy precision, recall, and F1 summarize direct feature correspondence.

BS-R fits a ridge map with an unregularized intercept between aligned post-ReLU SAE activations. By default, it uses a document-level 80/20 train/evaluation split, `lambda=100`, and ReLU reconstruction, then evaluates distributed correspondence with the same weighted greedy metrics. Self-Slot Recovery (SSR) additionally measures whether reconstruction preserves target-feature identity. Cross-tokenizer Llama–Gemma experiments pool and pair activations over shared whitespace-delimited word spans.

Activation extraction consistently uses eager attention so that supported Transformers versions follow the same numerical path.

## Repository Layout

```text
StitchCoder/
├── configs/
│   ├── paper_experiments.json      # Models, SAEs, data, and hyperparameters for nine directional experiments
│   └── golden_main_results.json    # Reference metrics and comparison tolerances for complete configurations
├── common/
│   ├── activation_extraction.py    # Model hooks and residual/SAE activation extraction
│   ├── alignment.py                # L2 row normalization and semi-orthogonal Procrustes alignment
│   ├── data_utils.py               # Deterministic sampling from Pile and C4
│   ├── metrics.py                  # Chunked cosine scores, dead-feature handling, P/R/F1, and SSR
│   ├── sae_loading.py              # Hugging Face and custom SAE loading
│   └── word_alignment.py           # Whitespace-span pooling across tokenizers
├── bs_f/
│   ├── run_bias_shift_full.py      # Main BS-F implementation
│   └── run_bias_shift_full_heldout.py # Document-disjoint evaluation
├── bs_r/run_bias_shift_ridge.py    # Main BS-R implementation
├── prepare_inputs.py               # Convert models and datasets into standardized experiment arrays
├── run_paper_reproduction.py       # Run configurations, aggregate metrics, and compare reference results
├── requirements.txt
└── .gitignore
```

## Installation

Python 3.12 and a CUDA-capable GPU are recommended:

```bash
git clone https://github.com/jiujiubuhejiu/StitchCoder.git
cd StitchCoder
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Access to the Gemma and Llama weights requires the corresponding Hugging Face permissions. After accepting the model licenses, authenticate with:

```bash
huggingface-cli login
```

The base/instruction-tuned comparison uses its matching instruction-tuned SAE. Set the checkpoint path through an environment variable:

```bash
export STITCHCODER_IT_SAE_PATH=/path/to/final_topk
```

The checkpoint must contain `cfg.json` and `sae_weights.safetensors` in a format readable by SAE Lens.

## Running Experiments

### List available model comparisons

```bash
python prepare_inputs.py --list-cells
```

Each configuration name represents a directional source-to-target comparison. Model identifiers, layers, SAE widths, and data settings are available in `configs/paper_experiments.json`.

### Run one model comparison

First, extract activations and prepare the standardized arrays shared by BS-F and BS-R:

```bash
python prepare_inputs.py \
  --cell base_to_instruct_750m \
  --output-root prepared \
  --device-a cuda:0 \
  --device-b cuda:0
```

Then run both correspondence analyses:

```bash
python run_paper_reproduction.py \
  --cell base_to_instruct_750m \
  --prepared-root prepared \
  --output-root outputs/base_to_instruct_750m \
  --method both \
  --backend cuda \
  --device cuda:0
```

### Run the complete multi-model suite

```bash
python prepare_inputs.py --all --output-root prepared --device-a cuda:0 --device-b cuda:0
python run_paper_reproduction.py --all --prepared-root prepared --output-root outputs/all_comparisons --method both --backend cuda --device cuda:0
```

The aggregation entry point writes `paper_results.csv`, `paper_results.json`, and per-configuration metric arrays. For complete default configurations, it also checks the results against the reference metrics and tolerances in `configs/golden_main_results.json`.

### Run BS-F and BS-R separately

```bash
python bs_f/run_bias_shift_full.py \
  --input-dir prepared/base_to_instruct_750m/bs_f \
  --output-dir outputs/base_to_instruct_750m/bs_f

python bs_r/run_bias_shift_ridge.py \
  --input-dir prepared/base_to_instruct_750m/bs_r \
  --output-dir outputs/base_to_instruct_750m/bs_r \
  --backend cuda
```

## Document-Disjoint Evaluation

BS-F supports independent calibration and evaluation document splits. The following configuration selects feature matches, confidence gates, and bias correction on the first 2,000 C4 documents, freezes those decisions, and rescores them on the final 2,000 documents:

```bash
python bs_f/run_bias_shift_full_heldout.py \
  --input-dir prepared/base_to_instruct_750m/bs_f \
  --output-dir outputs/base_to_instruct_750m/bs_f_heldout \
  --calibration-sequences 2000
```

The output records both document partitions, their overlap count, and the `evaluation_refit` status. The Procrustes map fitted on Pile text remains fixed across both scoring partitions.

## BS-R Controls

The shared entry point exposes the ridge penalty, source-row shuffling, and source-feature capacity:

```bash
python run_paper_reproduction.py --cell base_to_instruct_750m --method bs_r --ridge-lambda 10 --output-root outputs/ridge_lambda_10
python run_paper_reproduction.py --cell llama_to_gemma_d50 --method bs_r --row-shuffle --output-root outputs/row_shuffle
python run_paper_reproduction.py --cell llama_to_gemma_d50 --method bs_r --source-feature-limit 16384 --output-root outputs/source_capacity_16k
```

Use a separate `--output-root` for each control setting to compare BS-R metrics and SSR directly across conditions.

## Generated Files

The repository-root `.gitignore` excludes input arrays, experiment outputs, model files, logs, caches, and environment directories. Version control retains the implementation, run configurations, dependency specification, and documentation, while runtime artifacts remain in their working directories.
