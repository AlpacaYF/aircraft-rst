# RST 训练包 —— 服务器执行说明

面向军用飞机实体抽取的**区域—场景—文本（Region-Scene-Text）多模态融合**模型训练代码。
本目录自包含，在实验室共享服务器上运行时**不污染任何公共环境**。

---

## 0. 隔离承诺（先读这段）

`setup.sh` 只做四件事，全部发生在本项目目录内：

| 项目 | 位置 | 说明 |
|---|---|---|
| Python 环境 | `<项目>/.venv` | `python3 -m venv` 创建，**不使用 conda、不激活 base** |
| pip 缓存 | `<项目>/.cache/pip` | 不写 `~/.cache/pip` |
| HuggingFace 缓存 | `<项目>/.cache/hf` | 由 `HF_HOME` 指定，不写 `~/.cache/huggingface` |
| torch / 临时目录 | `<项目>/.cache/{torch,tmp}` | 不写 `/tmp` 之外的共享位置 |

- 全程**不执行** `pip install --user`，**不修改** `~/.bashrc`、`~/.profile`、conda 任何环境。
- 所有环境变量只写进项目内的 `env.sh`，需要你**主动 `source env.sh`** 才生效，且仅对当前 shell 有效。
- `run_train.sh` 用 `nice -n 10` 启动，并**自动挑选剩余显存最多的卡**；若所有卡剩余不足 6GB 会直接拒绝启动，不会抢别人的卡。
- 卸载：`rm -rf .venv .cache` 即可，系统不留痕迹。

---

## 1. 上传什么

服务器上有两种做法。

### 做法 A（推荐）：只传代码 + 索引 + 知识库，数据用软链

数据集本体 12.5GB，如果服务器上已有副本就别重传：

```
Aircraft/
├── rst/                    # 模型与训练代码（必传）
├── scripts/                # 本机已跑通的索引/知识库构建脚本（必传，可复现）
├── index/                  # 44,700 条实体索引 + 25,015 条切块计划（必传，~80MB）
├── knowledge/              # 114 类语义知识库 + 归一化混淆对（必传，~1MB）
├── setup.sh  requirements.txt  env.sh 模板
├── run_pipeline.sh  run_train.sh  README_SERVER.md
└── dataset/  crop/         # 数据：若服务器已有，改成软链接
```

```bash
# 服务器上，假设数据已在 /data/aircraft
ln -s /data/aircraft/dataset  Aircraft/dataset
ln -s /data/aircraft/crop     Aircraft/crop
```

> **`crop/` 其实不需要**。Region 是在训练时从 `dataset/` 原图按 `index/instances.jsonl` 里的扩展框实时裁剪的（`crop/` 是无上下文硬裁剪，不符合方案 3.1 的要求）。只软链 `dataset/` 就够。

### 做法 B：全量打包上传

在本机执行（约 12.5GB，很慢，只在服务器没有数据副本时用）：

```bash
cd /e/dmt/multimodal-dataset
tar --exclude='Aircraft/.venv' --exclude='Aircraft/.cache' \
    --exclude='Aircraft/runs' --exclude='Aircraft/__pycache__' \
    --exclude='Aircraft/*/~$*' -czf aircraft_rst.tar.gz Aircraft
```

只传代码和元数据（约 85MB，**建议始终先传这个**）：

```bash
cd /e/dmt/multimodal-dataset/Aircraft
tar --exclude='.venv' --exclude='.cache' --exclude='runs' --exclude='__pycache__' \
    --exclude='dataset' --exclude='crop' --exclude='~$*' \
    -czf ../aircraft_rst_code.tar.gz .
```

上传后：`tar xzf aircraft_rst_code.tar.gz -C ~/your/project/Aircraft`

---

## 2. 安装环境

```bash
cd Aircraft
bash setup.sh                    # 自动按 nvidia-smi 驱动版本选 CUDA wheel
# 或指定：
bash setup.sh --torch cu121      # cpu | cu118 | cu121 | cu124
bash setup.sh --dry-run          # 先看要做什么
source env.sh
```

驱动版本与 wheel 的对应（脚本自动判断）：`>=525 → cu124`，`>=450 → cu121`，`>=418 → cu118`，否则 CPU。

**关于 sentencepiece：不需要，也不要装。** SigLIP2 仓库只提供 `tokenizer.json`，没有 `spiece.model`，所以 `AutoProcessor` 会走慢版 `SiglipTokenizer` 并报 `vocab_file is None`。`rst/encoders.py` 直接用 `AutoTokenizer`（GemmaTokenizerFast）绕开，实测可用。

---

## 3. 自检（务必先跑这一步）

```bash
python -m rst.check_env --download-model
```

它会依次检查并打印：

1. python / torch / CUDA 版本，每张卡的**剩余**显存、算力代际
2. `huggingface.co` 与 `hf-mirror.com` 可达性；校园网通不了 huggingface 时提示设 `HF_ENDPOINT`
3. 编码器权重能否真正加载
4. `index/`、`knowledge/`、`dataset/` 是否齐全（缺关键文件直接 FAIL）
5. **真实走一遍数据加载 → 前向 → 反向**，报告梯度覆盖率与单步峰值显存

退出码非 0 表示有阻塞项，先解决再往下走。

首次会下载 `google/siglip2-base-patch16-384`（1.5GB）。若镜像的 xet CDN 不可达，手动预下载：

```bash
huggingface-cli download google/siglip2-base-patch16-384 --local-dir models/siglip2-base-384
python -m rst.check_env --model models/siglip2-base-384
```
之后所有命令都加 `--model models/siglip2-base-384`。

---

## 4. 离线链路（知识库 → 文本特征 → 视觉验证 → 困难对）

一条命令跑完：

```bash
bash run_pipeline.sh --skip-train
```

或分步执行，每步都有进度条、可重复运行：

```bash
python -m rst.build_text_features          # 3.3.4  114×19×768 语义 token
python -m rst.extract_embeddings --split all   # 44,700 个 Region 的 SigLIP2 向量
python -m rst.verify_knowledge             # 3.3.2  视觉一致性验证，产出 keep/drop
python -m rst.build_text_features          # 用 3.3.2 的结论重建文本特征
python -m rst.mine_hard_pairs --topk 8     # 3.3.3  按真实视觉相似度挖困难对
```

**为什么 `build_text_features` 要跑两次**：第一次用未验证的知识库产出文本特征，才能拿去做 3.3.2 的图文一致性检验；检验结论（`knowledge/kb_verification.json`）再回填，第二次产出真正用于训练的特征。

产出物：
- `knowledge/text_features.npz` — 每类的语义 token 矩阵 + 属性矩阵（供 `L_diff`）
- `index/region_embeddings.npz` — 44,700 个 Region 向量
- `knowledge/kb_verification.json` — 每条知识句的检索命中类别、margin、keep/drop
- `knowledge/hard_pairs.json` + `index/hard_table.npy` — 困难负样本表（供 `L_hard`）

`verify_knowledge` 会打印判据、keep 率、**随机基线**和 own-class 排名。

### 关于 3.3.2 的一个实测警告（重要）

本机用 900 个真实 Region 做过一次小样本测量，结论是：**冻结的 SigLIP2 在 114 类细粒度飞机上，零样本文→图检索基本等于随机**——own-class 排名中位数 57，而随机期望是 54；top-1 只有 1.0%。

因此最初按"文本能否检索回自己的类别"实现的判据**不可用**：它只会把约 99% 的知识句判为不合格，而那 1% 与随机无异。现已改为**与预挖困难类做配对对比**：

```
score(t) = sim(t, 原型_c) − max_{n ∈ hardneg(c)} sim(t, 原型_n)     保留条件 score > min_margin
```

小样本下该判据 keep 率 28.8%，而 1 对 3 比较的随机基线是 25.0% —— **只比随机高一点**。原因是每个类只有约 8 个 Region，原型噪声很大。

跑完 `--split all` 的全量 embedding 后（每类 17–1560 个 Region），原型会干净得多，keep 率与随机基线的差距应当拉开。**判读方法**：看输出里的 `chance_keep_rate` 字段，只有当 `overall_keep_rate` 明显高于它（建议 ≥1.5 倍）时，3.3.2 的过滤才有实际意义。

如果全量跑完仍然贴着随机基线，说明这条路本身不成立，有两个替代方案：

1. **把验证挪到对齐训练之后**：先用全部知识句训练 3.5.2 的对比对齐，再用**训练后的**编码器做 3.3.2，然后重新训练。文本—视觉对齐正是要学的东西，用未对齐的编码器去检验对齐质量，逻辑上站不住。
2. **改用教师模型做验证**（不需要 GPU）：把知识句和该类的真实图片一起发给 `http://localhost:8000/v1`，问"这段描述是否与图中飞机一致"。之前实测该 API 单请求 20s、并发 4 可用，1892 条约需 2.6 小时。

方案原文写的是"在同类别多个真实飞机样本上均保持较高的一致性"，配对对比这个实现更贴近原意；但请据此在论文里把 3.3.2 的判据写清楚，不要写成"检索回自身类别"。

---

## 5. 训练

```bash
bash run_train.sh --name full --epochs 30 --batch 16
```

### 进度显示

每个 stage 每个 epoch 一条 tqdm 进度条，实时刷新：

```
s2 e3/10:  42%|████▏     | 700/1680 [04:12<05:51,  2.79it/s, loss=3.214, top1=0.412, lr=8.3e-05, mem=11.4G]
```

- `s2 e3/10` = stage 2、该阶段第 3/10 个 epoch
- 每个 epoch 结束自动在验证集上评估并打印：
  ```
  val Top-1 48.31  Top-5 71.02  MacroF1 46.88 | mil Top-1 47.10 mil F1 45.22 (101 cls) | domain 96.33 milF1 97.10 civF1 62.44
  saved best (macro-F1 46.88) -> runs/full
  ```
- 全部数值同时写入 `runs/<name>/history.jsonl`，可直接读出来画图。

### 三阶段（方案 3.5.1–3.5.3）

| stage | 打开的分支 | 对应损失 |
|---|---|---|
| 1 | Region only | `L_domain + L_class` |
| 2 | + Scene + Text | `+ λ₁L_con` |
| 3 | + Hard Negative + Diff | `+ λ₂L_hard + λ₃L_diff` |

三阶段共用一个模型和一个优化器，只切换"哪些分支和损失生效"，所以前一阶段的权重直接带入下一阶段，不重训。`--epochs 30` 会均分给各阶段；要固定每阶段轮数用 `--stage-epochs 10`。

### 消融（直接对应表 4-6 / 4-7）

```bash
bash run_train.sh --ablation region_only   --name abl_region     # 表4-6 第1行
bash run_train.sh --ablation region_scene  --name abl_rs          # 表4-6 第2行
bash run_train.sh --name full                                     # 表4-6 第3行 = 完整模型
bash run_train.sh --ablation no_hard       --name abl_nohard      # 表4-7 第2行
bash run_train.sh --ablation no_diff       --name abl_nodiff
bash run_train.sh --ablation no_mask       --name abl_nomask       # 验证 mask 通道的作用
```

单个开关也能直接给：`--no-use-scene`、`--no-use-text`、`--no-use-hard-negative`、`--no-use-diff-loss`、`--no-balance-class-loss`。

> `--ablation region_only` 就是"只用 Region 输入、SigLIP2 微调"的基线，即表 4-3 的 `SigLIP 2 (Fine-tuning)` 行。

### 断点续训与单测

```bash
bash run_train.sh --resume runs/last.pt --name full
bash run_train.sh --max-batches 5 --epochs 1 --stage-epochs 1   # 5 个 batch 的连通性冒烟
```

---

## 6. 显存与时间预算

默认配置（Region 384²、每图最多 24 个 Scene 切块、batch 16）在 fp16/AMP 下大约需要 **16GB 显存**。不够时按这个顺序调，代价从小到大：

| CLI 参数 | 默认 | 建议降到 | 影响 |
|---|---|---|---|
| `--batch 8` | 16 | 8 / 4 | 最直接，但小 batch 下 `L_con` 的负样本变少 |
| `--region-pool 8` | 12 | 8 / 6 | Region token 144→64→36，注意力开销近乎线性下降，**几乎无损** |
| `--scene-max-tiles 12` | 24 | 12 / 8 | 远景上下文变粗，小目标场景受影响 |
| `--scene-token-cap 512` | 1024 | 512 | 强制每块池化更狠 |
| `--freeze-vision` | 关 | 打开 | 只训融合层，显存大降、精度会掉，适合先跑通链路 |

**不能靠缩小输入省显存**：SigLIP2 视觉塔带一个按固定分辨率学习的位置编码（384 对应 24×24=576），喂 336 会因 441≠576 直接报错。代码里已加校验并给出明确提示。省显存只能靠 `--scene-max-tiles`、`--region-pool`、`--batch` 和 `--freeze-vision`。

其它常用参数：`--epochs`、`--stage-epochs`、`--warmup-epochs`、`--backbone-lr`、`--head-lr`、`--weight-decay`、`--lambda-con/-hard/-diff`、`--eval-batches N`（只做 N 个 batch 的粗略验证，调试用）、`--max-batches N`（每 epoch 只跑 N 个 batch，链路自检用）。

**显存构成提醒**：训练时模型**不加载 SigLIP2 文本塔**（约 1.2GB），因为语义原型是 `build_text_features` 离线算好的，训练阶段从不调用它。所以实际显存只包含视觉塔 + 融合层。

单 epoch 估算：33,583 个训练实体，balanced 采样后约 44,000 个样本；A100/4090 级 batch 16 约 2,750 step，**8–15 分钟/epoch**（取决于 `--scene-max-tiles`），30 epoch 约 4–7 小时。

**检查点体积**：`best.pt` / `last.pt` 各约 **1.2GB**（模型 ~420MB + AdamW 动量方差 ~800MB，为支持 `--resume` 必须存）。30 epoch 只覆盖这两个文件，不会累积。跑完确认结果后可删 `last.pt` 省盘。

`check_env` 打印的峰值显存是 batch=4 的小样本值，**不能外推**成整训需求。

---

## 7. 已实现 / 未实现（请如实看待）

**已实现，并在本机用真实 SigLIP2 权重（非随机初始化）跑通验证**：

| 验证项 | 结果 |
|---|---|
| 数据管线 | 44,700 实体索引加载 2.9s；collate 每唯一图只解码一次；mask 通道严格 {0,1}；letterbox 后 GT 框落在 [0,384] 内 |
| 增广一致性 | 同一图同 epoch 翻转决策确定；epoch 间 49.2% 的图改变翻转；翻转率 50.1% |
| Scene 切块 | 多 tile / 少 tile 补零 / whole-image 三条路径均正确，pad 标记与 tile 数一致 |
| 前向 | 输出 `logits_class(3,114)`、`logits_domain(3,2)`、`logits_semantic(3,114)`、`gate(3,3)`、`logits_attr(3,466)` |
| 反向 | 梯度覆盖 243/256 个可训练张量（文本塔已卸载，不占梯度） |
| 优化器 | 参数组正确：backbone 1e-5（80 decay + 131 nodecay）、head 1e-4（18 + 27）；warmup + cosine 生效；grad clip 80.9→1.0 |
| 三阶段 CLI | `--stages 1 2 3` 依次输出 `scene=False text=False...` → `scene=True text=True hard=False` → 全开，退出码 0，`history.jsonl`/`best.pt`/`summary.json` 均落盘 |
| 六个消融 | 解析为 6 组互不相同的开关组合，loss 随分支叠加单调上升（4.75→6.29→7.36→7.97→8.94） |
| 指标正确性 | `macro_f1` 与 `binary_prf` 在 5 组用例上**与 sklearn 逐位一致**（含全对、全错、类别缺失等边界） |
| 文本编码 | 1,892 条知识句全部编码，最长 54 token < 64 上限，**零截断** |

**尚未实现**，需要下一轮：

- 表 4-3 的 ResNet50 / ViT / Swin / CLIP 基线（`region_only` 已覆盖 SigLIP2 微调那一行）
- 表 4-8 的端到端实体抽取：检测器（YOLO / RT-DETR / Grounding DINO）与 Entity F1 评价
- APPEAR 的注意力自动部件发现适配
- 4.4.4 通用 MLLM 直接识别基线（教师模型 API 已可用，脚本未写）
- 表 4-2 的 `Scene Token 上限 1024` 与切块参数不自洽（448px 块在 patch16 下是 784 token/块），本包按"每块池化到 49/4 token"实现，**论文正文需相应补充池化系数**

**一个已修正但值得知道的教训**：stage 开关原先直接覆写配置，导致 `--ablation region_only` 等消融被 stage 3 重新全开——表 4-6/4-7 会全部退化成完整模型而看不出问题。现在 stage 提供"可开集"、ablation 提供"允许集"，两者取交集，并有断言保证五个消融解析出五个不同组合。

---

## 8. 常见问题

**`vocab_file is None` / `SiglipTokenizer requires SentencePiece`**
不要装 sentencepiece，也不要调 `AutoProcessor`。代码已用 `AutoTokenizer`。若你自行改回了 `AutoProcessor`，改回来。

**`huggingface.co` 连不上**
`export HF_ENDPOINT=https://hf-mirror.com`（`env.sh` 已默认设置）。若镜像的 `cas-bridge.xethub.hf.co` 也超时，用第 3 节的 `--local-dir` 手动下载。

**`--cache-dir` 导致重新下载 1.5GB**
不要传 `--cache-dir`。缓存位置由 `env.sh` 里的 `HF_HOME` 决定；显式传 `--cache-dir` 会覆盖它并造成缓存未命中。

**中文训练日志乱码**
`export PYTHONIOENCODING=utf-8`（`env.sh` 未设，因为服务器一般已是 UTF-8；本机 Windows 需要）。

**所有卡都被占用**
`run_train.sh` 会拒绝启动并打印 `nvidia-smi --query-compute-apps`。等空闲或换节点，不要强行 `--gpu` 指定已满的卡。

**想确认知识库到底可信不可信**
看 `knowledge/kb_verification.json` 的 `keep_rate_by_kind` 和 `drop_detail`。被 drop 的句子就是模型看图后认为"不像这一类"的描述，可直接作为 4.8 节错误案例分析素材。
