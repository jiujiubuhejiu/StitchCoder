# 论文绘图代码与数据

对应 `outputs/paper/TMLR/TMLR/paper.tex` 实际引用的 **6 张图、8 个图件**。按论文引用和输出内容选择脚本，修改时间作为辅助依据；重复配色、旧布局及其他稿件的版本不收录。

| 论文图 | 绘图代码（scripts/） | 绘图数据（data/） |
| --- | --- | --- |
| Figure 1 框架图 | 原始 PDF，未找到对应绘图源代码 | `reference/coder_framework2.pdf` |
| Figure 2(a)(b) BS-F / BS-R | `render_alignment_ladder.py` | `alignment_ladder_data.json`、`alignment_self.json` |
| Figure 3 类别分布 | `make_minder_pie.py` | `minder_categories.json` |
| Figure 4 因果干预 | `plot_causal.py` | `causal_optionC.json`；`causal/` 内为实验汇总 |
| Figure 5 ridge λ 扫描 | `plot_lambda_sweep.py` | `lambda_sweep/` 下 5 份原始结果 |
| Figure 6(a)(b) DE 扫描 | `render_de_scan.py` | `de_scan_bias_shift_full.json`，31 个扫描点 |

`reference/` 保存论文原图；`manifest.json` 记录原始路径、时间、哈希和重复版本的取舍。脚本已改为读取本文件夹的数据，可独立运行。

已在项目外的独立目录运行验证。重绘的 7 个图件中，6 个与原图逐像素一致；饼图在 120 dpi 对比中仅有 1 个像素的微小差异。框架 PDF 与原文件完全相同。详见 `validation.json`。

## 运行

在本文件夹下执行：

```bash
python -m pip install -r requirements.txt
python render_all.py
```

输出位于 `rendered/`，也可用 `--output-dir /path/to/figures` 指定位置。无需 GPU。框架图直接复制原始 PDF，其余图按现有论文样式重新绘制。

**因果图的数据说明：** 原脚本使用固定的近似误差条，并非从逐样本数据重新计算的 bootstrap 区间。本目录保留该绘图输入以对应现有论文；真实实验的 `ci95` 另外保存在 `data/causal/` 中。
