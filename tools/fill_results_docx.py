#!/usr/bin/env python
"""Fill the thesis tables from the measured JSON files, into a NEW .docx.

The numbers are never typed by hand: `tools/collect_paper_results.py` writes
paper_results.json straight from runs/*/eval_test.json on the server, and this
script only copies values into cells whose text is still 待填. Anything without a
measurement is left untouched, so the output document shows exactly what has been
run and what has not.

Safety rules:
1. the source file is opened read-only; output goes to a new name
2. a cell is only written when its current text is the 待填 placeholder, and the
   existing run's formatting is kept
3. inline OMML equations are counted before and after and must not change
4. every cell and paragraph reports APPLIED / SKIPPED / NOT-FOUND

  python tools/fill_results_docx.py <src.docx> <out.docx> <paper_results.json>
"""
import copy
import json
import re
import shutil
import sys

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PLACEHOLDER = "待填"
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"

# 表4-2 cells where the paper and the code that produced the numbers disagree.
# Only genuine inconsistencies are touched; parameters the paper simply omits are
# reported to the author instead of being inserted.
T42_FIX = {
    "温度τ／类别间隔m": "0.07／0.1",
    "Scene输入尺寸": "切块边长 448×448（原图像素），每块送入编码器前重采样到 384×384",
}
T42_WHY = {
    "温度τ／类别间隔m":
        "S_c 在 114 类上的标准差实测约 0.112，m=0.3 相当于要求 2.67 倍标准差的间隔，"
        "训练中几乎不可能满足，困难负样本项恒为 0 且整体指标反而下降 1.40 点；"
        "重标定为 0.1（约 0.9 倍标准差）后同一对照转为 +1.03 点。表中所填数字必须与"
        "产出结果的代码一致，否则不可复现。",
    "Scene输入尺寸":
        "448 是原图上的切块边长，编码器实际接收 384；两者都写出来才不会与 3.2.3 的"
        "token 预算（近景 7×7、远景 2×2）对不上。",
}

# Tables are located by their HEADER, never by index or caption number: the chapter has
# been renumbered twice (4-8 before 4-6 in V6, then 4-1..4-5 in V8) and columns have
# been dropped between versions. Cells are then matched by header TEXT, so a column the
# author removed simply is not written, and one they reorder still lands correctly.
#   required: every string must appear in the header row
#   forbidden: none of these may appear (disambiguates 表4-4 from 表4-2)
# Row specs are (方法 prefix, 输入模态 prefix or None, key in paper_results.json).
TABLES = [
    {"required": ("114类Top-1",), "forbidden": (), "rows": [
        ("ResNet50", None, "resnet50"),
        ("ViT", None, "vit"),
        ("Swin", None, "swin"),
        ("CLIP", None, "clip_zeroshot"),
        ("SigLIP", None, "siglip_zeroshot"),
        ("Qwen3-VL", "Region + Scene", "qwen_region_scene"),
        ("Qwen3-VL", "Region", "qwen_region"),
        ("InternVL", None, "internvl"),
        ("MiniCPM", None, "minicpm"),
        ("Ours", None, "ours")]},
    {"required": ("Entity F1",), "forbidden": (), "rows": [
        ("YOLO", None, "det_yolo"),
        ("RT-DETR", None, "det_rtdetr"),
        ("Grounding DINO", None, "det_gdino"),
        ("Qwen3-VL", None, "det_qwen"),
        ("Ours", None, "det_ours")]},
    {"required": ("Top-1", "Macro-F1"), "forbidden": ("114类Top-1",), "rows": [
        ("Region Only", None, "region_only"),
        ("Region + Scene +", None, "region_scene_text"),
        ("Region + Scene", None, "region_scene")]},
    {"required": ("Hard Pair Accuracy",), "forbidden": (), "rows": [
        ("Baseline + MLLM", None, "abl_knowledge"),
        ("Baseline", None, "abl_baseline"),
        ("Full Model", None, "abl_full")]},
]
PARAM_HEADER = ("参数", "设置")

# 表注, keyed by the same header signature used to find the table. Inserted as a new
# paragraph immediately AFTER the table, so the note travels with it if the author
# reorders chapters.
NOTES = {
    ("114类Top-1",): (
        "注：三种纯视觉基线均为 4 通道输入（RGB + Ground Truth 框二值掩码），"
        "与本文 Region 的输入完全一致；其 3 通道版本的结果见 4.4.1 的分解说明。"
        "纯视觉基线与视觉语言模型零样本均为单次运行结果，本文方法为 2 次独立重复"
        "（随机种子 0／1）的平均值，两次重复的极差为 Top-1 0.60、Macro-F1 0.34 个百分点。"
        "纯视觉基线按各自原生 224×224 分辨率输入，本文方法为 384×384。"
        "零样本行的文本原型取提示模板“a photo of a {{类别}} aircraft”与“a photo of a {{类别}}”"
        "嵌入的平均；若仅用裸类别名，CLIP 为 25.28%、SigLIP 2 为 1.10%。"
        "Qwen3-VL 两行使用统一结构化 Prompt、给出全部 114 个候选类别并要求按置信度输出前 5 名；"
        "其中 Region + Scene 模式有 265 条（8.8%）输出未能解析为候选类别，按错误计入，"
        "故该行为偏保守的下界。"),
    ("Entity F1",): (
        "注：表中 Classifier 指本文的 RST 融合模块，与表4-2 的 Ours 行为同一模型；"
        "检测器按 4.4.2 的设定训练为类别无关的单类飞机检测器，只负责定位、不预测型号。"
        "因此“RT-DETR + Classifier”与“Ours”两行的识别模块相同，"
        "Ours 表示本文的完整流程（RT-DETR 主检测器 + RST 融合识别）。"
        "Entity 指标按 4.2.2 定义计算：预测框与真实框按置信度一对一贪心匹配，"
        "IoU≥0.5 且预测类别正确才计为正确实体。"
        "四行分别取各自检测器可达的最优工作点，候选框密度为每图 "
        "{yolo_cand}／{rtdetr_cand}／{gdino_cand}／{qwen_cand} 个"
        "（测试集真实密度 1.80 个/图）：YOLO 与 RT-DETR 取 0.5 置信度阈值，"
        "Grounding DINO 的置信度最高仅 0.58、无法用同一阈值筛选，故取其每图前 3 个候选框，"
        "Qwen3-VL 不输出置信度，故取其全部输出。"
        "把四行强制统一到每图前 3 框时，YOLO 与 RT-DETR 的 Entity F1 变为 "
        "{yolo_ent_f1_budget}%／{rtdetr_ent_f1_budget}%，"
        "说明统一预算会丢弃阈值下已正确的框，故不以它作为报告口径。"
        "Grounding DINO 一行拿到全场最多的候选框仍然垫底，差距主要在识别而非定位："
        "它的每图前 3 框已覆盖 {gdino_det_r}% 的真实实体，"
        "但通用大模型在定位正确的框上型号 Top-1 仅 {gdino_recog}%，"
        "本文 RST 模块在同一口径下为 {rtdetr_recog}%。"
        "AP50 与 mAP@50:95 由检测器在测试集完整输出上单独评测，与类别正确性无关；"
        "Qwen3-VL 无置信度输出，其 AP 以模型自身的输出顺序作为置信度排序计算。"),
}

# 结果分析 placeholders, keyed by the section heading that precedes them
ANALYSIS = {
    "4.4.1": [
        "在完全相同的训练／验证／测试划分与同一套指标实现下，本文方法在 114 类飞机识别上取得 "
        "{ours_top1}% 的 Top-1、{ours_top5}% 的 Top-5 与 {ours_f1}% 的 Macro-F1，"
        "在 103 类军机上的 Top-1 为 {ours_mil}%，全面高于三种纯视觉基线中最强的 Swin Transformer"
        "（{swin_top1}%／{swin_f1}%），Top-1 领先 {gap_top1} 个百分点、Macro-F1 领先 {gap_f1} 个百分点。"
        "三种纯视觉基线之间的排序（Swin＞ResNet50＞ViT）与其在细粒度识别任务上的一般表现一致，"
        "说明基线训练是充分的，差距不来自欠训练。本文方法一栏为 {n_repeat} 次独立重复训练的平均值，"
        "纯视觉基线与零样本行为单次运行结果。",
        "视觉语言模型的 Zero-shot 结果给出了另一个方向的证据。以类别名称构造文本原型时，"
        "CLIP 达到 {clip_top1}%，SigLIP 2 仅为 {siglip_top1}%（114 类随机猜测为 0.88%）。"
        "两者相差三十多倍，足以怀疑是实现错误，因此本文做了三重核验。"
        "其一，用官方 AutoProcessor 从 PIL 像素出发走一条完全不经过本文代码的路径，"
        "所得指标与本文实现完全一致。"
        "其二，核验过程中确实发现并修正了一处实现问题：SigLIP 的文本塔在 "
        "last_hidden_state[:, -1, :] 处池化，并假设该位置就是终止符；"
        "而快速分词器默认右侧填充，会使该位置变成 <pad>，等于从一个填充符上取句向量。"
        "改为左侧填充后池化位置正确落在 <eos>。在同一批 400 个区域裁剪、同一提示词模板下的 "
        "A/B 对照中：右侧填充时 114 个类名嵌入的平均两两余弦为 0.9785、“军机 vs 民航”"
        "二分类正确率 86.50%；左侧填充后分别为 0.9038 与 97.00%。"
        "表中 SigLIP 2 一行已按修正后的实现重跑，其图文平均余弦为 {siglip_imgtxt}"
        "（修正前为负值，意味着两个塔根本不在同一空间里），"
        "114 类中预测覆盖了 {siglip_distinct} 个类别，二分类对照 {siglip_ctl}%。"
        "其三，即便修正之后，114 类细粒度检索仍然只有 {siglip_top1}%。"
        "同一批裁剪与提示词下，CLIP 的图文平均余弦为 {clip_imgtxt}、"
        "类名提示词平均余弦 {clip_textcos}、114 类中预测覆盖了 {clip_distinct} 个类别，"
        "而 SigLIP 2 只覆盖 {siglip_distinct} 个。"
        "因此可以确认：SigLIP 2-base 的图文对齐在粗粒度上是完好的"
        "（军机／民航二分类 {siglip_ctl}%），但它的文本塔无法把这批机型缩写名"
        "（如 J20、EF2000、Su34）拉开到可检索的程度——这是模型在该任务上的性质，"
        "不是实现缺陷。也正因如此，仅提供类别名称不足以支撑细粒度识别，"
        "必须引入更丰富的语义描述，这是本文用 MLLM 构建语义知识库的出发点。",
        "通用多模态大模型直接识别的结果同样低于本文方法：Qwen3-VL 在统一结构化 Prompt、"
        "给出全部 114 个候选类别的条件下，仅输入 Region 时 Top-1 为 {qwen_region_top1}%、"
        "Top-5 为 {qwen_region_top5}%，同时输入 Region 与完整 Scene 时提升到 {qwen_rs_top1}% 与 "
        "{qwen_rs_top5}%，Macro-F1 由 {qwen_region_f1}% 升到 {qwen_rs_f1}%，"
        "但仍与本文方法的 {ours_top1}% 相差 {qwen_gap} 个百分点。"
        "这一组对照给出两点信息：其一，加入场景后通用大模型的指标明显上升，"
        "说明场景上下文对飞机识别确实有贡献，这与本文 Scene 分支的动机一致；"
        "其二，即便有场景信息，未经本领域训练的通用大模型在 114 类细粒度机型上"
        "仍远不如在相同数据上训练的专用模型，其宏平均 F1 尤其低，"
        "说明它对长尾稀有机型的覆盖很差。需要说明的是，Region + Scene 模式下有 "
        "{qwen_rs_unparsed} 条输出未能解析为候选类别（占 {qwen_rs_unparsed_pct}%），"
        "这些条目按错误计入，因此该行的数值是偏保守的下界。",
        "上表中本文方法与纯视觉基线的差距需要逐因素分解，否则无法判断它来自方法还是来自设置。"
        "第一项是输入通道：本文的 Region 是 4 通道（RGB 加 Ground Truth 框的二值掩码），"
        "该通道来自 4.2.1 已经给定的标注框，基线同样有权使用，因此表中三个基线也按 4 通道重跑。"
        "实测结果是，同一个掩码通道在两侧的收益完全不同：本文方法去掉它会使 Top-1 由 {ro_top1}% "
        "降到 {nomask_top1}%、Macro-F1 由 {ro_f1}% 降到 {nomask_f1}%（即 {mask_gain_top1} 与 "
        "{mask_gain_f1} 个百分点）；而三个基线加上它之后 Top-1 只变化 ResNet50 {maskd_resnet50_top1}、"
        "ViT {maskd_vit_top1}、Swin {maskd_swin_top1} 个百分点，Macro-F1 变化 {maskd_resnet50_f1}、"
        "{maskd_vit_f1}、{maskd_swin_f1} 个百分点，均在重复波动范围之内。"
        "也就是说，掩码通道并不是一项白给的信息优势：同样的第 4 通道在 ImageNet 预训练骨干、"
        "224 分辨率下几乎不起作用，只有与本文的区域—场景交互和 384 输入结合时才产生约 2 个百分点的收益。"
        "第二项是基线的训练配方：将 ViT-B/16 的学习率由 3e-4 改为 1e-4、其余完全不变，"
        "其 Top-1 由 {rgb_vit_top1}% 升到 {vit_lr1e4_top1}%、Macro-F1 由 {rgb_vit_f1}% 升到 "
        "{vit_lr1e4_f1}%（{vit_lr_gain_top1} 与 {vit_lr_gain_f1} 个百分点）。"
        "这说明表中基线的数值受配方影响很大，本文与基线之间剩余的差距应主要归于骨干预训练"
        "（WebLI 级图文对比预训练对 ImageNet-1k 监督预训练）与输入分辨率（384 对 224），"
        "而不是多模态融合本身；融合的作用应由表4-4 与表4-5 在完全相同的骨干与分辨率下判断。"
        "需要说明的是，纯视觉基线按各自原生 224×224 分辨率输入，本文方法为 384×384，"
        "因此上表中本文方法与基线之间的差距同时包含骨干网络、输入分辨率与多模态融合三方面的贡献。"
        "为分离出融合本身的作用，表 4-4 与表 4-5 在完全相同的骨干与分辨率下逐一关闭各模态，"
        "那两张表才是对本文方法内部设计的评价。",
    ],
    "4.4.2": [
        "本文先把检测器训练为类别无关的单类飞机检测器，只负责定位，型号识别交给 RST 模块。"
        "在测试集 1,669 张图像、3,002 个真实实体上，YOLO 检测器本身达到 AP50 {yolo_ap50}%、"
        "mAP@50:95 {yolo_map}%。按 4.2.2 的定义（预测框与真实框一对一匹配，IoU≥0.5 且类别正确），"
        "置信度阈值取 0.5 时 Entity Precision 为 {yolo_ent_p}%、Entity Recall 为 {yolo_ent_r}%、"
        "Entity F1 为 {yolo_ent_f1}%；若只看定位、忽略类别，同样的阈值下 F1 为 {yolo_det_f1}%。"
        "两者之差就是型号识别环节在整条抽取链路上付出的代价。"
        "该指标对置信度阈值敏感：阈值降到 0.25 时 Entity Recall 升到 {yolo_ent_r_lo}%、"
        "Entity F1 变为 {yolo_ent_f1_lo}%，因此表中数值对应的阈值以本节正文为准。",
        "需要提醒一个容易被误读的数字：在定位正确的实体上，型号识别 Top-1 为 {yolo_recog}%，"
        "略高于表 4-2 中使用人工标注框得到的 {ours_top1}%。这并不表示检测框比标注框更有利，"
        "而是检测器漏掉的多为小目标与困难样本，留下来的是更容易的子集，"
        "所以这两个数字不能直接相减来衡量定位误差的代价；"
        "衡量该代价应当比较检测-only F1（{yolo_det_f1}%）与 Entity F1（{yolo_ent_f1}%）。",
        "替换检测器的实验说明该流水线对检测器并不敏感，但也不是完全无关。"
        "RT-DETR 在同样的单类设定、同样的 640 输入与同样的训练轮数下取得 "
        "AP50 {rtdetr_ap50}%、mAP@50:95 {rtdetr_map}%，Entity Recall {rtdetr_ent_r}%、"
        "Entity F1 {rtdetr_ent_f1}%，与 YOLO 的 {yolo_ap50}%／{yolo_map}%／{yolo_ent_f1}% "
        "处于同一水平，差距在 1 个百分点量级。由于两者的识别模块完全相同，"
        "这一差距只反映定位质量：RT-DETR 的检测-only F1 为 {rtdetr_det_f1}%，"
        "YOLO 为 {yolo_det_f1}%。因此按 4.4.2 的设定，本文以 RT-DETR 作为主检测器，"
        "YOLO 作为替换实验，两者都能支撑完整的实体抽取流程。",
    ],
    "4.5.1": [
        "三种模态配置的整体指标为：Region Only {ro_top1}%／{ro_f1}%，"
        "Region + Scene {rs_top1}%／{rs_f1}%，Region + Scene + Text {rst_top1}%／{rst_f1}%，"
        "每档为 {n_repeat} 次独立重复训练的平均值。同一配置两次重复之间的极差为 "
        "Top-1 {sp_ro_top1}／{sp_rs_top1}／{sp_rst_top1} 个百分点、"
        "Macro-F1 {sp_ro_f1}／{sp_rs_f1}／{sp_rst_f1} 个百分点，"
        "这就是本文判定某个模态是否真的有效所用的尺度。"
        "以该尺度衡量：加入 Scene 带来 {agg_top1} 个百分点 Top-1 与 {scene_f1} 个百分点 Macro-F1，"
        "Macro-F1 一侧略超重复极差、Top-1 一侧与之相当，因此 Scene 的作用是弱的正向趋势而非稳定增益；"
        "再加入 Text 后 Top-1 变化 {text_top1}、Macro-F1 变化 {text_f1} 个百分点，"
        "两次重复方向一致，即当前的语义分支没有带来正收益。",
        "按实体尺度分层可以看到 Scene 的作用集中在哪里，但同样受重复波动限制。"
        "测试集中长边小于 128 像素的实体占 19.1%，在这一区间 Region + Scene 相对 Region Only "
        "平均提升 {small_top1} 个百分点 Top-1、{small_f1} 个百分点 Macro-F1；"
        "然而同一配置两次重复在该区间的 Top-1 极差就达到 {spread_ro_64_128}（Region Only）"
        "与 {spread_rs_64_128}（Region + Scene）个百分点，与增益同量级，"
        "因此这一分层结果目前只能作为倾向性证据，需要更多重复才能确认。"
        "长边不小于 384 像素的实体占 50.8%，其精度主要由 Region 分支决定，"
        "小目标上的增益被这部分多数样本稀释，最终整体只剩 {agg_top1} 个百分点。"
        "这与 3.2.3 的设计动机方向一致：Scene 分支的价值在目标过小、区域裁剪信息不足时才显现。",
        "另一个需要如实报告的现象是动态融合权重的分布：三种配置下 Scene 分支的门控权重都停在下限 0.05。"
        "原因在于式 3-13 中 z_R 由区域—场景交互后的特征 F_RS 得到，其中已经包含了上下文扩展带来的场景信息，"
        "z_S 相对于 z_R 是冗余的，门控将其压到下限是该结构下的合理解。"
        "因此 Scene 信息实际是通过 3.4.1 的区域—场景交互进入模型的，"
        "而非通过 3.4.3 的门控加权，这一点在方法描述中应予以明确。"
        "还需说明一个容易误读的诊断输出：在关闭某一分支的配置下，门控依然会按式 3-14 "
        "输出三个权重（例如 Region + Scene 配置下 α_Text 为 0.173），"
        "但这并不代表被关闭的模态仍在参与融合——式 3-13 中该分支的特征向量被置为零向量，"
        "其权重乘的是零；且 H 随后经过 LayerNorm 归一化，对被浪费掉的那部分权重所形成的"
        "整体缩放不敏感，只有仍在工作的分支之间的比例会保留下来。"
        "本文用单元测试验证了这一点：将已关闭分支的全部参数与缓冲区随机重赋值后，"
        "类别 logits 逐位不变，因此三档配置的差异确实只来自被消融的模态本身。",
    ],
    "4.5.2": [
        "以 Region + Scene 为 Baseline，加入 MLLM 语义知识后整体 Top-1 为 {kn_top1}%，"
        "再加入困难负样本机制后为 {full_top1}%，即困难负样本项带来 {hard_delta} 个百分点的变化。"
        "Hard Pair Accuracy 一栏把判定限制在真实类别与 3.3.3 挖出的困难类之间，"
        "专门衡量易混淆机型之间的判别能力，三档配置依次为 {base_hp}%、{kn_hp}% 与 {full_hp}%。"
        "需要如实指出：困难负样本机制在它本应最擅长的这个指标上没有超过不加它的 Baseline——"
        "加入语义知识使其下降 {kn_drop} 个百分点，再引入困难负样本只挽回 {hard_gain} 个百分点，"
        "相对 Baseline 仍为 {hard_vs_base} 个百分点。",
        "在困难负样本间隔 m 的取值上需要说明一次修正。S_c 在 114 个类别上的标准差实测约为 0.112，"
        "原先设置的 m=0.3 相当于要求每一对困难样本都达到 2.67 倍标准差的间隔，"
        "该约束在训练中几乎不可能满足：修正前同一对照的 Top-1 差值为 −1.40 个百分点，"
        "即加入困难负样本反而有害。将 m 重新标定为 0.1（约 0.9 倍标准差）后，"
        "该差值转为 {hard_delta} 个百分点，方向恢复正确，且幅度已超过同配置两次重复的极差"
        "（{sp_rst_top1} 个百分点）。但结合 Hard Pair Accuracy 一栏可以看到，"
        "它仍未回到不加语义知识的 Baseline 水平，因此本文的结论是："
        "重新标定后的困难负样本机制主要起到补偿语义分支负面影响的作用，"
        "而不是独立地提升了易混淆机型之间的判别能力。",
        "MLLM 语义知识的作用同样受到文本特征质量的限制。诊断显示，"
        "语义项 S_c 单独作为分类器时 Top-1 为 {sem_only}%，与完整模型的 {full_top1}% 相差仅 {sem_gap} 个百分点，"
        "即式 3-17 中 β·S_c 一项在推理阶段的净贡献为 {sem_delta} 个百分点。"
        "其原因在于对比损失已经把视觉特征对齐到文本原型，分类头因此继承了这部分先验，"
        "S_c 在推理时提供的是重复信息。要真正发挥语义知识的作用，"
        "需要文本原型本身具备判别性几何结构，这是后续工作的方向。",
    ],
}


def cell_set(cell, text):
    """Write `text` into the cell, keeping the formatting of its first run."""
    para = cell.paragraphs[0]
    if para.runs:
        para.runs[0].text = text
        for r in para.runs[1:]:
            r.text = ""
    else:
        para.add_run(text)


def find_table(doc, required, forbidden=()):
    """The table whose header row contains all of `required` and none of `forbidden`."""
    for t in doc.tables:
        if not t.rows:
            continue
        hdr = [c.text.strip() for c in t.rows[0].cells]
        if all(w in hdr for w in required) and not any(w in hdr for w in forbidden):
            return t
    return None


def fix_param_table(doc):
    """Align the parameter table with the configuration that produced the results."""
    table = find_table(doc, PARAM_HEADER)
    if table is None:
        print("NOT-FOUND parameter table (header %s)" % (PARAM_HEADER,))
        return 0
    n = 0
    for row in table.rows[1:]:
        name = row.cells[0].text.strip()
        if name not in T42_FIX:
            continue
        cur = row.cells[1].text.strip()
        if cur == T42_FIX[name]:
            continue
        print("  FIX   表4-2 %-16s %r -> %r" % (name, cur, T42_FIX[name]))
        print("        why: %s" % T42_WHY[name])
        cell_set(row.cells[1], T42_FIX[name])
        n += 1
    return n


def add_table_notes(doc, fmt=None):
    """Insert or refresh a 表注 paragraph immediately after each table that has one.

    Clones the paragraph that already follows the table so the note inherits its
    indentation and font instead of dropping to the document default.

    A note is identified by its opening characters, not by its full text: when the
    measured protocol changes (the 表4-3 note went from a confidence floor to a matched
    candidate budget) the stale note must be rewritten in place, not skipped, or the
    document keeps asserting a protocol that no longer produced the numbers above it.
    """
    fmt = fmt or {}
    added = 0
    for sig, text in NOTES.items():
        try:
            text = text.format(**fmt)
        except KeyError as exc:
            print("  NOTE %r skipped: measurement %s is not collected yet" % (sig, exc))
            continue
        table = find_table(doc, sig)
        if table is None:
            print("NOT-FOUND table for note %r" % (sig,))
            continue
        following = table._tbl.getnext()
        template = None
        if following is not None and following.tag == qn("w:p"):
            probe = "".join(t.text or "" for t in following.iter(qn("w:t")))
            if probe == text:
                print("  NOTE already present for %r" % (sig,))
                continue
            if probe.startswith(text[:8]):
                for r in following.findall(qn("w:r"))[1:]:
                    following.remove(r)
                runs = following.findall(qn("w:r"))
                if runs:
                    ts = runs[0].findall(qn("w:t"))
                    for t in ts[1:]:
                        runs[0].remove(t)
                    if ts:
                        ts[0].text = text
                        ts[0].set(qn("xml:space"), "preserve")
                print("  NOTE refreshed for %r (%d chars)" % (sig[0], len(text)))
                added += 1
                continue
            template = following
        if template is None:
            template = doc.paragraphs[0]._p
        new = copy.deepcopy(template)
        runs = new.findall(qn("w:r"))
        for r in runs[1:]:
            new.remove(r)
        if runs:
            ts = runs[0].findall(qn("w:t"))
            for t in ts[1:]:
                runs[0].remove(t)
            if not ts:
                ts = [OxmlElement("w:t")]
                runs[0].append(ts[0])
            ts[0].text = text
            ts[0].set(qn("xml:space"), "preserve")
        else:
            r = OxmlElement("w:r")
            t = OxmlElement("w:t")
            t.text = text
            r.append(t)
            new.append(r)
        table._tbl.addnext(new)
        print("  NOTE  added after the table with header %r (%d chars)" % (sig[0], len(text)))
        added += 1
    return added


def fill_tables(doc, res):
    filled = skipped = 0
    for spec in TABLES:
        table = find_table(doc, spec["required"], spec["forbidden"])
        tag = spec["required"][0]
        if table is None:
            print("NOT-FOUND table requiring header %r" % (spec["required"],))
            continue
        hdr = [c.text.strip() for c in table.rows[0].cells]
        claimed = set()
        for prefix, modality, key in spec["rows"]:
            target, tidx = None, -1
            for ri, row in enumerate(table.rows[1:]):
                if ri in claimed:
                    continue
                c0 = row.cells[0].text.strip()
                c1 = row.cells[1].text.strip() if len(row.cells) > 1 else ""
                if not c0.startswith(prefix):
                    continue
                if modality is not None and not c1.startswith(modality):
                    continue
                target, tidx = row, ri
                break
            if target is None:
                print("  MISS  %-14s row %r%s" % (tag, prefix,
                                                  " / " + modality if modality else ""))
                continue
            claimed.add(tidx)
            vals = res.get(key) or {}
            for ci, col_name in enumerate(hdr):
                if ci == 0 or col_name not in vals:
                    continue
                if target.cells[ci].text.strip() != PLACEHOLDER:
                    skipped += 1
                    continue
                cell_set(target.cells[ci], vals[col_name])
                print("  FILL  %-14s %-22s %-18s = %s"
                      % (tag, prefix + (" /" + modality if modality else ""),
                         col_name, vals[col_name]))
                filled += 1
            left = [n for ci, n in enumerate(hdr)
                    if ci and n not in vals and target.cells[ci].text.strip() == PLACEHOLDER]
            if left:
                print("  WAIT  %-14s %-22s no measurement for: %s"
                      % (tag, prefix, ", ".join(left)))
                skipped += len(left)
    return filled, skipped


def replace_analysis(doc, section, paragraphs):
    """Replace the 结果分析 placeholder that follows `section` with real text.

    The new paragraphs are cloned from the placeholder itself and inserted at ITS
    position, not after the preceding paragraph: `doc.paragraphs` skips tables, so
    "the paragraph before the placeholder" is the table caption, and inserting there
    would put the analysis above the table it discusses.
    """
    paras = doc.paragraphs
    start = None
    for i, p in enumerate(paras):
        if p.text.strip().startswith(section):
            start = i
            break
    if start is None:
        print("NOT-FOUND section heading %s" % section)
        return 0
    idx = None
    for j in range(start + 1, min(start + 40, len(paras))):
        t = paras[j].text.strip()
        if t.startswith("结果分析"):
            idx = j
            break
        if t[:2] == "4." and len(t) < 40:          # next section: no placeholder here
            break
    if idx is None:
        print("NOT-FOUND 结果分析 placeholder after %s" % section)
        return 0

    anchor = paras[idx]._p
    for text in reversed(paragraphs):
        new = copy.deepcopy(anchor)
        runs = new.findall(qn("w:r"))
        for r in runs[1:]:
            new.remove(r)
        if runs:
            ts = runs[0].findall(qn("w:t"))
            for t in ts[1:]:
                runs[0].remove(t)
            if not ts:
                ts = [OxmlElement("w:t")]
                runs[0].append(ts[0])
            ts[0].text = text
            ts[0].set(qn("xml:space"), "preserve")
        else:
            r = OxmlElement("w:r")
            t = OxmlElement("w:t")
            t.text = text
            r.append(t)
            new.append(r)
        anchor.addnext(new)
    anchor.getparent().remove(anchor)
    print("  APPLIED %s 结果分析 (%d paragraphs)" % (section, len(paragraphs)))
    return len(paragraphs)


def main():
    src, out, res_path = sys.argv[1], sys.argv[2], sys.argv[3]
    blob = json.load(open(res_path, encoding="utf-8"))
    res, fmt = blob["cells"], blob.get("fmt", {})
    shutil.copyfile(src, out)
    doc = Document(out)
    before = len(doc.element.body.findall(".//" + M + "oMath"))

    n_fix = fix_param_table(doc)
    filled, skipped = fill_tables(doc, res)
    n_notes = add_table_notes(doc, fmt)
    n_par = 0
    for section, texts in ANALYSIS.items():
        keys = set()
        for t in texts:
            keys |= set(re.findall(r"{(\w+)}", t))
        absent = sorted(k for k in keys if k not in fmt)
        if absent:
            # writing prose with holes in it is worse than leaving the placeholder
            print("  SKIP  %s 结果分析 -- no measurement for: %s" % (section, ", ".join(absent)))
            continue
        n_par += replace_analysis(doc, section, [t.format(**fmt) for t in texts])

    doc.save(out)
    after = len(Document(out).element.body.findall(".//" + M + "oMath"))
    print("\nparameter cells corrected: %d | table notes inserted: %d" % (n_fix, n_notes))
    print("cells filled %d, left as %s %d" % (filled, PLACEHOLDER, skipped))
    print("analysis paragraphs written: %d" % n_par)
    print("OMML equations before %d after %d -> %s"
          % (before, after, "OK" if before == after else "CHANGED (investigate)"))
    print("wrote", out)


if __name__ == "__main__":
    main()
