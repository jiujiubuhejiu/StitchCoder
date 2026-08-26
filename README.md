# StitchCoder：跨模型稀疏特征对齐实验

本仓库提供 StitchCoder 跨模型稀疏特征对齐实验代码，用于比较不同语言模型及其独立训练的稀疏自动编码器（Sparse Autoencoders, SAEs）如何表示相同或相近的概念。

不同模型的残差流和 SAE 字典位于各自的坐标空间，特征索引与方向无法直接对应。为每一对模型重新训练联合字典又会使计算成本随模型数量快速增长。StitchCoder 先通过半正交 Procrustes 建立残差空间之间的几何映射，再复用现有 SAE 分析特征对应关系，从而支持可扩展的同模型、跨训练阶段、跨规模和跨架构表征比较。

实验同时提供两种互补视角：Bias-Shift Forward（BS-F）衡量单个特征之间的直接对应，Bias-Shift Ridge（BS-R）衡量目标概念能否由一组源特征共同重构。二者结合可以区分“一对一特征匹配较弱”和“概念以分布式形式保留”这两类情况，为模型表征审计、模型差异分析和候选模型特有特征筛选提供量化依据。

## 多模型对比范围

默认配置包含 9 组方向性实验，覆盖以下比较：

| 实验类别 | 模型与 SAE 设置 | 方向数 |
| --- | --- | ---: |
| 同模型参考 | Gemma 2 2B 与自身的相同 SAE | 1 |
| 同模型、不同 SAE | Gemma 2 2B 的 canonical SAE 与 Matryoshka SAE | 2 |
| Base / Instruct | Gemma 2 2B 与 Gemma 2 2B-IT | 2 |
| 跨模型规模 | Gemma 2 2B 与 Gemma 2 9B | 2 |
| 跨模型架构 | Gemma 2 2B 与 Llama 3.1 8B | 2 |

残差空间对齐使用确定性抽样的 Pile 文本，特征评分使用 C4 validation 文本。默认设置使用 6,000 条对齐文本和 4,000 条评分文本；跨规模与跨架构配置使用 20,000 条分层抽样文本拟合对齐矩阵。模型、数据集和 SAE revision、随机种子及主要超参数均集中记录在 `configs/paper_experiments.json` 中。

## 核心方法

BS-F 将源 SAE 解码方向通过残差空间映射投影到目标空间，计算特征对的余弦对应关系，并使用 median centroid、`top1 > 0.10`、`top1-top2 >= 0.05` 和 `[-4,4]` 偏置修正。最终结果采用激活频率加权的双向 greedy precision、recall 和 F1 汇总直接特征对应程度。

BS-R 在对齐的 post-ReLU SAE 激活上拟合带非正则化截距的 ridge map。默认使用按文档划分的 80/20 train/evaluation split、`lambda=100` 和 ReLU 重构，再以相同的加权 greedy 指标评估分布式对应关系。Self-Slot Recovery（SSR）进一步衡量重构是否保持目标特征身份。跨 tokenizer 的 Llama–Gemma 实验通过共享空白词跨度完成池化与配对。

模型激活提取统一使用 eager attention，使支持的 Transformers 版本遵循一致的实验数值路径。

## 项目结构

```text
StitchCoder/
├── configs/
│   ├── paper_experiments.json      # 9 组方向性实验的模型、SAE、数据与超参数
│   └── golden_main_results.json    # 完整配置的参考指标与比较容差
├── common/
│   ├── activation_extraction.py    # 模型 hook、残差和 SAE 激活提取
│   ├── alignment.py                # L2 行归一化、半正交 Procrustes
│   ├── data_utils.py               # Pile/C4 的确定性抽样
│   ├── metrics.py                  # 分块余弦、dead-feature、P/R/F1、SSR
│   ├── sae_loading.py              # Hugging Face 与自定义 SAE 加载
│   └── word_alignment.py           # 跨 tokenizer 的空白词跨度池化
├── bs_f/
│   ├── run_bias_shift_full.py      # BS-F 主实现
│   └── run_bias_shift_full_heldout.py # 文档不相交评估
├── bs_r/run_bias_shift_ridge.py    # BS-R 主实现
├── prepare_inputs.py               # 模型与数据到标准实验数组
├── run_paper_reproduction.py       # 多配置运行、聚合与参考指标比较
├── requirements.txt
└── .gitignore
```

## 环境配置

建议使用 Python 3.12 和 CUDA GPU：

```bash
git clone https://github.com/jiujiubuhejiu/StitchCoder.git
cd StitchCoder
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Gemma 和 Llama 权重需要对应的 Hugging Face 访问权限。完成模型许可确认后，执行：

```bash
huggingface-cli login
```

Base / Instruct 对比使用配套的 instruct SAE。通过环境变量配置 checkpoint 路径：

```bash
export STITCHCODER_IT_SAE_PATH=/path/to/final_topk
```

该 checkpoint 包含 SAE Lens 可读取的 `cfg.json` 和 `sae_weights.safetensors`。

## 运行实验

### 查看多模型对比配置

```bash
python prepare_inputs.py --list-cells
```

每个配置名称对应一个有方向的 source-to-target 比较；模型、层、SAE 宽度和数据设置可在 `configs/paper_experiments.json` 中查看。

### 运行单个模型对比

先提取激活并准备 BS-F 与 BS-R 共用的标准数组：

```bash
python prepare_inputs.py \
  --cell base_to_instruct_750m \
  --output-root prepared \
  --device-a cuda:0 \
  --device-b cuda:0
```

再运行两种对应关系分析：

```bash
python run_paper_reproduction.py \
  --cell base_to_instruct_750m \
  --prepared-root prepared \
  --output-root outputs/base_to_instruct_750m \
  --method both \
  --backend cuda \
  --device cuda:0
```

### 运行全部多模型对比

```bash
python prepare_inputs.py --all --output-root prepared --device-a cuda:0 --device-b cuda:0
python run_paper_reproduction.py --all --prepared-root prepared --output-root outputs/all_comparisons --method both --backend cuda --device cuda:0
```

聚合入口会生成 `paper_results.csv`、`paper_results.json` 和各配置的指标数组。对于完整默认配置，它还会使用 `configs/golden_main_results.json` 中的参考指标进行容差检查。

### 分别运行 BS-F 与 BS-R

```bash
python bs_f/run_bias_shift_full.py \
  --input-dir prepared/base_to_instruct_750m/bs_f \
  --output-dir outputs/base_to_instruct_750m/bs_f

python bs_r/run_bias_shift_ridge.py \
  --input-dir prepared/base_to_instruct_750m/bs_r \
  --output-dir outputs/base_to_instruct_750m/bs_r \
  --backend cuda
```

## 文档不相交评估

BS-F 支持独立的 calibration/evaluation 文档划分。下面的配置在前 2,000 个 C4 文档上确定特征匹配、置信门和偏置修正，并在后 2,000 个文档上冻结这些决定后重新评分：

```bash
python bs_f/run_bias_shift_full_heldout.py \
  --input-dir prepared/base_to_instruct_750m/bs_f \
  --output-dir outputs/base_to_instruct_750m/bs_f_heldout \
  --calibration-sequences 2000
```

输出记录两个文档分区、文档交集数量和 `evaluation_refit` 状态。Pile 文本拟合的 Procrustes 映射在两个评分分区中保持固定。

## BS-R 控制变量

Ridge penalty、source-row shuffle 和 source capacity 可通过统一入口配置：

```bash
python run_paper_reproduction.py --cell base_to_instruct_750m --method bs_r --ridge-lambda 10 --output-root outputs/ridge_lambda_10
python run_paper_reproduction.py --cell llama_to_gemma_d50 --method bs_r --row-shuffle --output-root outputs/row_shuffle
python run_paper_reproduction.py --cell llama_to_gemma_d50 --method bs_r --source-feature-limit 16384 --output-root outputs/source_capacity_16k
```

为每组控制变量指定独立的 `--output-root`，可以直接比较不同条件下的 BS-R 指标与 SSR。

## 生成文件管理

输入数组、实验输出、模型文件、日志、缓存和环境目录均由仓库根目录下的 `.gitignore` 管理。Git 提交保留实现源码、运行配置、依赖说明和使用文档，实验过程文件保存在工作目录中。
