# 第三章插图（RST 多模态融合框架图）

三张图均由 matplotlib 脚本按 **14.65 cm 定宽**（A4 21.0 − 3.17×2）绘制，
遵循 `scipilot-figure-skill` 的出版规范（按最终尺寸出图、矢量优先、
Okabe-Ito 色盲安全配色、最小字号 ≥6 pt、线条图 600 DPI、灰度冗余编码）。
该 skill 本身只覆盖数据图，明确「不做示意图、流程图、架构图」，
因此框架图由本目录脚本绘制，仅复用其 `setup_style` / `export_figure` /
`visual_qa` / `check_figure` 工具链。

## 重新生成

```bash
cd "E:/dmt/multimodal-dataset/Aircraft/code/figures"
python fig3_1_framework.py           # 图3-1 整体框架（14.65 × 12.45 cm）
python fig3_2_rst_fusion.py          # 图3-2 融合与识别细节（14.65 × 8.60 cm）
python fig3_3_training_inference.py  # 图3-3 分阶段训练与推理（14.65 × 8.60 cm）
```

每张图输出 4 个文件：`.pdf`（投稿/打印）、`.svg`（文字可编辑，Word 2016+ 可直接插入）、
`.png`（600 DPI，论文粘贴用）、`_grayscale.png`（灰度自检产物，不必插入论文）。

## 合规审计

```bash
python "C:/Users/WL/.qoder-cn/skills/scipilot-figure-skill/scripts/check_figure.py" \
  fig3_1_framework.pdf fig3_1_framework.png \
  --min-dpi 600 --width-in 5.7677 --height-in 4.9016
```

每个脚本运行时会打印 `[PASS]` / `[FAIL]`，检查项包括：PNG 像素尺寸与 DPI 是否等于
声明的版面尺寸、文字是否越界、**两两文字包围盒是否重叠**、以及 `visual_qa.audit_layout`
的缺字/裁切检查。

### 两个已知坑

1. `export_figure(..., tight=True)` 是其默认值，会用 `bbox_inches='tight'` 按内容反算尺寸，
   破坏「按最终尺寸出图」。本目录一律传 `tight=False, pad_inches=0.0`。
2. `export_figure` 的 `grayscale_preview=True` 内部会用 `bbox_inches='tight'` **重存同名 PNG**，
   覆盖掉定宽版本。因此灰度预览由 `figstyle.export()` 用 Pillow 自行生成。

`check_figure.py` 对 PDF 报的 `[WARN] 字体未嵌入` 是**误报**：matplotlib 输出的是
`/Type0` 组合字体，`/FontDescriptor` 位于 `/DescendantFonts` 之下，而该检查只在页面
资源层查找。实测三份 PDF 均含 5–6 个 `/FontFile2`（TrueType 已嵌入子集），
`/Type3` 计数为 0，符合期刊要求。

## 图注建议（中文，供正文使用）

- **图3-1** RST 多模态融合框架图。框架由数据预处理与实体构建、区域—场景视觉特征提取、
  MLLM 语义知识构建、RST 多模态融合与实体识别、模型训练与推理五个部分组成。
  实线箭头表示在线前向数据流，虚线表示离线构建或缓存复用，点线表示损失项的监督关系。
- **图3-2** RST 多模态融合与实体识别细节。区域特征作为 Query、场景特征作为 Key/Value
  构成交叉注意力，经残差得到场景增强实体表示 $F_{RS}$；类别语义以多 Token 形式参与
  逐 Token 匹配，匹配分数 $S_c$ 经 $\beta$ 注入型号分类头，属性头与型号头并行输出。
- **图3-3** 分阶段训练与推理流程。三个阶段依次引入视觉基础目标、视觉—文本对比约束与
  困难负样本判别约束；推理阶段由检测器提供候选框、缓存类别原型提供语义，MLLM 不参与推理。

图内标注为英文（面向投稿），若学位论文需要中文版，把 `figstyle.apply_style()` 的
`lang` 改为 `'zh'` 并将各脚本中的标签字符串替换为论文原术语即可（系统已装
SimHei / Microsoft YaHei / Noto Sans SC，`setup_style.py --list-fonts` 可验证）。
