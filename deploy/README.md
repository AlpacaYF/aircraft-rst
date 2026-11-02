# RST-DEDF 端到端飞机实体抽取 Demo

输入一张图片，输出图中每架飞机的**型号、军/民域、置信度、边界框**，并生成标注图。

```
图片 → YOLO11x（单类 aircraft 检测）→ 逐实体保护性裁剪 + 场景切块
     → 三个 RST 专家（R+S / R+T / R+S+T）→ 等权 logits 平均（DEDF）→ 实体列表
```

默认使用论文最终采用的**统一监督配方**（仅域判别 + 分类损失）：

| 专家 | 检查点 | 模态 |
|---|---|---|
| 场景专家 | `runs/v2_rs_s0/best.pt` | Region + Scene |
| 语义专家 | `runs/v2_rt_nosup_s0/best.pt` | Region + Text（置信度自适应融合） |
| 全模态专家 | `runs/v2_rst_nosup_s0/best.pt` | Region + Scene + Text |

## 1. 服务器上直接用

```bash
cd ~/dmt/aircraft_rst
source env.sh
.venv/bin/python deploy/demo.py --image dataset/f026ed5a5e43e706cc3c56a371e6ab8f.jpg \
    --outdir deploy/output/demo --recipe uniform --gpu auto
```

可选参数：`--recipe full`（完整配方）或 `--recipe single`（只用全模态专家）；
`--conf 0.4` 调整检测阈值；`--boxes-json` 复用已有检测框跳过检测。

## 2. Windows 本机一键调用（推荐）

```bat
python C:\Users\WL\Desktop\RST_demo\run_demo.py C:\path\to\photo.jpg
python C:\Users\WL\Desktop\RST_demo\run_demo.py C:\path\to\folder --recipe uniform --conf 0.4
```

脚本会把图片 scp 到服务器 `demo_input/<hash>.<ext>`，调用 `deploy/demo.py`，
再把 `result.json` / `result.png` 拉回 `RST_demo\results\<图片名>\`。

## 3. 输出

`result.json`：

```json
{
  "system": "RST-DEDF", "recipe": "uniform",
  "experts": ["v2_rs_s0", "v2_rt_nosup_s0", "v2_rst_nosup_s0"],
  "image_size": [1200, 800], "n_entities": 2,
  "timings_s": {"detect": 0.04, "load_experts": 9.3, "recognize": 0.58, "total": 14.2},
  "entities": [{
    "bbox": [9.5, 185.4, 1184.7, 524.1], "det_score": 0.68,
    "class": "Tornado", "class_id": 92, "domain": "military",
    "confidence": 1.0, "expert_agreement": 1.0,
    "expert_top1": ["Tornado", "Tornado", "Tornado"]
  }]
}
```

`result.png` 为标注图（军机红框、民机蓝框，标签为 型号 + 域 + 置信度）。

## 4. 指标（test 3002 实体，seed0）

| 配方 | GT 框 Top-1 | Macro-F1 | 端到端 Entity F1（YOLO，conf .5） |
|---|---|---|---|
| uniform + DEDF（默认） | **93.50** | **92.14** | **87.69**（识别率 94.99） |
| uniform 单模型（RST） | 91.77 | 90.50 | 86.22（识别率 93.39） |
| full + DEDF | 93.64 | 92.15 | 87.80（识别率 95.10） |

- Entity F1 的分母是"有检测图的 GT"（2899）；若按全部 3002 GT，uniform DEDF
  为 **86.10**（P 91.36 / R 81.41）。
- 速度：1669 张测试图基准，检测 0.044 s/图，DEDF 识别 0.418 s/图，
  端到端 **0.462 s/图**。单图首次调用约 14 s（其中 9.3 s 是加载三专家）。
- 检测器：YOLO11x，30 epoch，单类 `aircraft`，imgsz 640，conf 0.5；
  AP50 93.73 / mAP@50:95 79.67。

## 5. 文件与路径

```
deploy/
  demo.py              # 服务端单图入口
  eval_uniform.sh      # 统一监督配方的实体级评测（复现表 4-7 用）
  export_models.sh     # 打包离线模型（约 3.7 GB）
  README.md
  manifest.json        # 模型/阈值/指标清单
runs/detect/det_runs/yolo11x_640_b16/weights/best.pt   # 检测器
runs/v2_rs_s0/best.pt, runs/v2_rt_nosup_s0/best.pt,
runs/v2_rst_nosup_s0/best.pt                            # 三专家
knowledge/text_features_clipL_en_diff.npz               # 语义原型库
index/class_meta.json                                   # 类别/域映射
```

## 6. 环境与实验纪律

- 识别端用项目 `.venv`（torch 2.4.1 + cu124）；检测端用 `.venv-det`
  （ultralytics），由 `demo.py` 通过子进程调用，互不污染。
- 不修改任何环境变量、不安装包；所有输出只写入项目目录或本机
  `RST_demo\results`。
- GPU：`--gpu auto` 优先选无计算进程且空闲显存最大的卡；没有空闲卡时会
  打印警告并选最空闲的一张。单图演示约占用 5–6 GB。
- 只使用自己的 `wangyangfan` 目录；不触碰他人的进程与文件。

## 7. 已知限制

- 闭集 114 类（103 军机 + 11 民机），未验证开放集/未见型号；
- 置信度为未校准的 softmax（三个专家均偏自信，ECE 约 6–7%），
  低置信样本建议结合 `expert_agreement` 与 `det_score` 一起看；
- 检测器 conf=0.5，可能漏检小目标或产生重叠框（同一实体多框）；
- 三专家为 seed0 单模型；论文中的三种子统计见 `补充实验结果与结论分析`。
