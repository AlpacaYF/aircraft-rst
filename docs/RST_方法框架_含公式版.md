# RST 方法框架详细描述（含公式版）—— 2026-10-15

> 全部公式与超参数均与代码逐行核对（`rst/model.py` / `losses.py` / `train.py` /
> `config.py` / `scripts/build_dataset_index.py`），可直接落进论文第 3 章。
> 编号 (3-x) 为占位，按你论文现有编号体系替换。

---

## 0. 记号约定

| 符号 | 含义 | 取值 |
|---|---|---|
| $I$ | 原始图像 | — |
| $b_i,\ y_i,\ d_i$ | 第 $i$ 个实体的框、型号标签、域标签 | $y\in\{1..C\}$，$C=114$；$d\in\{\text{mil},\text{civ}\}$ |
| $E_v$ | 视觉编码塔（Region/Scene 共享） | SigLIP2-base-patch16-384，微调，宽 $d_v{=}768$ |
| $E_t$ | 文本编码器（冻结） | CLIP-ViT-L/14 |
| $h$ | 融合隐维度 | 768（注意力 12 头） |
| $\mathbf{F}^{R},\ \mathbf{F}^{S}$ | 区域 / 场景 Token 集合 | $\mathbf{F}^{R}\in\mathbb{R}^{144\times h}$，$\mathbf{F}^{S}\in\mathbb{R}^{\sum_t k_t \times h}$ |
| $\mathbf{P}_{kb},\ \Delta\mathbf{P}$ | 离线语义原型库 / 可学习残差 | $\mathbf{P}_{kb}\in\mathbb{R}^{C\times N\times d_t}$ |
| $\mathcal{H}_c$ | 类 $c$ 的困难负类集合 | $|\mathcal{H}_c| = K = 8$ |
| $\varepsilon$ | 门控下限 | 0.05 |

---

## 1. 问题定义

给定图像 $I$ 及其中的飞机实体框集合 $\{b_i\}$，对每个实体同时判别**型号**
$y\in\{1,\dots,114\}$（103 类军机、11 类民机）与**军/民域** $d$。实体级识别
要求模型在仅有一小块外观证据（目标常小于 64 px）时仍能判别 114 个细粒度
型号，因此需要引入目标自身之外的两类信息：整图场景上下文与类别语义知识。
端到端实体抽取则进一步级联类别无关检测器完成定位。

---

## 2. 实体级多模态样本构建

### 2.1 保护性扩展裁剪与掩码通道

对实体框 $b_i=(x,y,w,h)$，以比例 $e$ 等比外扩并裁剪至图像内、扣除邻居实体
的遮挡区域（保证裁剪窗内不含其他飞机），得到区域输入窗：

$$b_i^{*} \;=\; \mathrm{Clip}_{\,I\setminus\bigcup_{j\neq i} b_j}\Big(b_i \oplus e\cdot s_i\Big),\qquad s_i=\max(w_i,h_i) \tag{3-1}$$

训练时扩展率逐实体随机化（抑制"固定目标-背景面积比"捷径）：

$$e \sim \mathcal{U}(0.15,\ 0.35)\quad(\text{基准 } 0.25,\ \text{抖动 }\pm 0.10) \tag{3-2}$$

区域输入为 4 通道张量（RGB + 二值掩码），掩码通道在窗内为 1、窗外为 0，
显式告知编码器"目标在哪"：

$$X^{R}=\big[\,R;\,G;\,B;\,M\,\big]\in\mathbb{R}^{4\times 384\times 384} \tag{3-3}$$

### 2.2 跨通道随机擦除

以概率 $p=0.25$ 对区域输入施加一块面积比 $r\sim\mathcal{U}(0.02,0.15)$ 的
灰色矩形，**RGB 与掩码通道同步置零**（抑制"部件固定位置"捷径，并保持训练
Top-1 低于 1.0、使判别损失不饱和）：

$$X^{R}[\Omega]\leftarrow \bar{c},\quad M[\Omega]\leftarrow 0 \tag{3-4}$$

### 2.3 场景分块与两级 Token 预算

长边 $\le 896$ 的图像整图送入；否则按 448 px 图块、步长 336（约 25% 重叠）
切块，与实体相交或相邻的块记为**实体块**（near），其余为**背景块**（far）。
每块缩放至 384² 编码为 $24\times24$ patch token 后自适应池化：

$$\mathbf{f}^{S}_{t}=\begin{cases}\mathrm{Pool}_{7\times7}\big(E_v(x_t)\big) & t\in\text{near（49 token）}\\[2pt] \mathrm{Pool}_{2\times2}\big(E_v(x_t)\big) & t\in\text{far（4 token）}\end{cases}\qquad \sum_t k_t \le 1024 \tag{3-5}$$

预算花在目标所在处（$\le 20$ 块 $\times 49 = 980 \le 1024$），远超预算时按
优先级降级。块级位置编码同时保留 near/far 身份。

### 2.4 语义知识库与困难类表

- 用 MLLM 对 114 类两两生成**成对差异描述句**（每类若干句），冻结
  CLIP-ViT-L/14 编码为语义 Token 原型库 $\mathbf{P}_{kb}\in\mathbb{R}^{C\times N\times d_t}$；
  另生成类级属性多标签矩阵 $A\in\{0,1\}^{C\times n_a}$。
- 困难类表 $\mathcal{H}\in\mathbb{N}^{C\times K}$：先由冻结嵌入相似度挖掘每类
  $K{=}8$ 个最易混淆类；训练第 2→3 阶段边界用**当前模型自身**的验证混淆
  重挖掘（$\S$2.6），混淆对不上的行用旧邻居补满。

---

## 3. 三分支编码

**Region**（实体外观）：$E_v$ 输出 $24\times24{=}576$ 个 patch token，池化为
$12\times12$ 以约束注意力开销，投影至 $h$ 并加二维正弦位置编码：

$$\mathbf{F}^{R}=\mathrm{Proj}_{R}\big(\mathrm{Pool}_{12\times12}(E_v(X^{R}))\big)+\mathrm{PE}^{2D}\in\mathbb{R}^{144\times h} \tag{3-6}$$

**Scene**（全局上下文）：共享同一视觉塔 $E_v$（权重同 Region，投影头独立）：

$$\mathbf{F}^{S}=\Big[\mathrm{Proj}_{S}(\mathbf{f}^{S}_{t})+\mathrm{PE}^{\text{tile}}(m_t)\Big]_{t=1..T} \tag{3-7}$$

**Text**（类别语义）：原型库加可学习残差（KB 作初始化、空间可适配）：

$$\mathbf{P}=\mathbf{P}_{kb}+\Delta\mathbf{P},\qquad \Delta\mathbf{P}\ \text{初始化为 }0 \tag{3-8}$$

---

## 4. 模态交互与置信度自适应决策

### 4.1 区域-场景跨模态注意力

以区域 Token 为 Query 在场景 Token 上做交叉注意力（残差 + 层归一化）：

$$\mathbf{F}_{RS}=\mathrm{LN}\Big(\mathbf{F}^{R}+\mathrm{MHA}\big(Q{=}\mathbf{F}^{R},\ K{=}V{=}\mathbf{F}^{S}\big)\Big) \tag{3-9}$$

场景键集按 (3-5) 的有效掩码屏蔽填充槽位。

### 4.2 分支向量：区域、场景环境视图、语义记忆

区域向量池化自场景增强表征；**场景分支只池化环境视图**（扣除实体块——
$\mathbf{F}_{RS}$ 已看过这些像素，环境块承载它看不到的其他飞机、机场、背景，
避免门控两路输入同源导致场景权重塌缩；全实体块时回退全场景）：

$$\mathbf{z}_{R}=\mathrm{AttnPool}(\mathbf{F}_{RS}),\qquad \mathbf{z}_{S}=\mathrm{AttnPool}\big(\mathbf{F}^{S}\setminus\text{near}\big) \tag{3-10}$$

语义分支把全部类的语义 Token 作为**类无关记忆**供实体查询（标签未知时
即可安全融合）：

$$\mathbf{z}_{T}=\mathrm{AttnPool}\Big(\mathrm{LN}\big(\mathbf{F}_{RS}+\mathrm{MHA}(Q{=}\mathbf{F}_{RS},\ K{=}V{=}\mathrm{Proj}_{T}(\mathbf{P}))\big)\Big) \tag{3-11}$$

其中 $\mathrm{AttnPool}$ 为单可学习查询的注意力池化。

### 4.3 动态门控融合

三分支门控在**激活分支上**做 softmax（未启用分支精确置零、有效分支重归一，
避免死通道吞没概率质量；全开时退化为普通形式）：

$$\boldsymbol{\alpha}=\varepsilon+(1-k\varepsilon)\cdot\mathrm{softmax}_{\text{active}}\Big(\mathrm{MLP}_{g}\big([\mathbf{z}_{R};\mathbf{z}_{S};\mathbf{z}_{T}]\big)\Big) \tag{3-12}$$

$$\mathbf{H}=\mathrm{LN}\big(\alpha_{R}\mathbf{z}_{R}+\alpha_{S}\mathbf{z}_{S}+\alpha_{T}\mathbf{z}_{T}\big),\qquad \varepsilon=0.05 \tag{3-13}$$

### 4.4 视觉-语义匹配 logits

分类得分之外，实体与每类原型做**Token 级匹配**：类 $c$ 的每个语义 Token
寻找最优解释它的区域 Token（"尾喷结构"绑定尾部、"进气道布局"绑定进气道），
再对该类语义 Token 取平均——避免单个虚高属性主导判类。匹配的视觉侧取
Token 集合 $\mathbf{F}_{RS}$ 而非已含语义的 $\mathbf{H}$（否则语义自匹配）：

$$S_{c}=\gamma\cdot\frac{1}{N_{c}}\sum_{n=1}^{N_{c}}\max_{m}\ \cos\big(\tilde{\mathbf{f}}_{m},\ \tilde{\mathbf{p}}_{cn}\big),\qquad \gamma\ \text{可学习（初始 10）} \tag{3-14}$$

### 4.5 置信度自适应语义融合

最终分类 logits 是视觉头与语义匹配的**凸组合**，权重由视觉置信度实时产生
——视觉越不确定，知识先验话语权越大；门输入 detach，仅 48 个门参数受训
（末层零初始化 $\Rightarrow w_0=\sigma(0)=0.5$）：

$$w=\sigma\Big(\mathrm{MLP}_{w}\big([\ell_{1},\ \ell_{1}{-}\ell_{2}]\big)\Big),\qquad \hat{\ell}=(1-w)\,\ell_{\text{vis}}(\mathbf{H})+w\,S \tag{3-15}$$

其中 $\ell_1,\ell_2$ 为 $\ell_{\text{vis}}$ 的 Top-1/Top-2 logit（梯度截断），
$\mathrm{MLP}_w$：$2\to16\to1$。

---

## 5. 训练目标

$$\mathcal{L}=\mathcal{L}_{\text{dom}}+\mathcal{L}_{\text{cls}}+\lambda_{1}\mathcal{L}_{\text{con}}+\lambda_{2}\mathcal{L}_{\text{hard}}+\lambda_{3}\mathcal{L}_{\text{diff}},\qquad (\lambda_{1},\lambda_{2},\lambda_{3})=(1.0,\ 1.0,\ 0.5) \tag{3-16}$$

**域判别**：$\mathcal{L}_{\text{dom}}=\mathrm{CE}(\ell_d, d)$（军/民二分类）。

**型号分类**（长尾逆频率加权）：

$$\mathcal{L}_{\text{cls}}=\mathrm{CE}\big(\hat{\ell},\ y;\ w_c\propto 1/n_c\big) \tag{3-17}$$

**视觉-语义对比对齐**：以 (3-14) 的原始余弦得分 $s=S/\gamma$ 对全部类做
CE，困难负类在配分函数中加权 $w{=}2$（等价于每个困难类计两次，用
logsumexp 拼接实现、不重建类别轴）：

$$\mathcal{L}_{\text{con}}=\mathbb{E}\Big[\mathrm{logsumexp}\big(\{s_j/\tau\}_j\ \cup\ \{\log(w{-}1)+s_{h}/\tau\}_{h\in\mathcal{H}_y\setminus y}\big)-s_{y}/\tau\Big],\quad \tau=0.07 \tag{3-18}$$

**受限候选集加性间隔 CE**（替换恒零的 hinge：均衡采样使训练 Top-1 达 1.0
后真-难得分差全部越过旧 margin、梯度恒零；restricted-CE 只要混淆对手仍有
竞争力就有梯度）：

$$\mathcal{L}_{\text{hard}}=\mathbb{E}\Big[\mathrm{CE}\Big(\tfrac{1}{\tau_{m}}\big[s_{y}-m,\ \{s_{h}\}_{h\in\mathcal{H}_y}\big],\ y\Big)\Big],\quad m=0.1,\ \tau_{m}=0.05 \tag{3-19}$$

margin $m{=}0.1$ 约为实测 $S_c$ 类间散布（0.112）的 0.9σ；退化列（$\mathcal{H}_y$
含 $y$）置 $-\infty$。**属性辅助**：$\mathcal{L}_{\text{diff}}$ 为属性多标签
BCE（pos_weight 按类内正例频率封顶 50）。

---

## 6. 三阶段课程训练

同一模型、同一优化器、单一余弦调度跨三阶段（阶段只决定分支与损失的开关）：

| 阶段 | 预算（epochs） | 激活分支 / 损失 | 目的 |
|---|---|---|---|
| S1 | 10 | 仅 Region；$\mathcal{L}_{\text{dom}}{+}\mathcal{L}_{\text{cls}}$ | 视觉基座预热 |
| S2 | 15–20 | +Scene +Text；+$\mathcal{L}_{\text{con}}$ | 多模态融合与对齐 |
| S3 | 15–20 | +$\mathcal{L}_{\text{hard}},\mathcal{L}_{\text{diff}}$；重挖掘 $\mathcal{H}$ | 细粒度判别 |

阶段切换判据：验证集**军机 Macro-F1** 连续 3 epochs 无提升且达到下限预算
（避免两三个幸运 epoch 提前离场）；总预算上限 50 epochs。优化：AdamW 分组
学习率（视觉塔 $10^{-5}$，其余 $10^{-4}$；一维参数不做 weight decay，
wd $=0.05$）、warmup 3 epochs、batch 16 类均衡采样、AMP、梯度裁剪 1.0、
梯度检查点。

**公平性约定**：全部消融行保持完整三阶段课程与同预算，仅改变模态开关
（use_scene / use_text / use_hard / use_diff），因此消融度量的是模态而非
课程差异。few-shot 实验中训练集每类截取 $K$ 实体（种子 12345 固定、所有
配置共享同一子集），val/test 永不截断。

---

## 7. DEDF：解耦专家决策融合

将 RST 框架的三种模态组合实例化为**独立训练的异质专家**（同一训练配方）：

$$\mathcal{E}_{1}=R{+}S\ \text{（场景专家）},\qquad \mathcal{E}_{2}=R{+}T\ \text{（语义专家，含 (3-15)）},\qquad \mathcal{E}_{3}=R{+}S{+}T\ \text{（全模态专家）}$$

推理时在 **logits 域等权平均**、argmax 判类（域判别同法）：

$$\hat{y}=\arg\max_{c}\ \frac{1}{3}\sum_{e=1}^{3}\ell^{(e)}_{c} \tag{3-20}$$

无温度缩放、无概率域变换、无拟合参数；专家规模异质（单模态能力
$E_R{=}89.44$ / $E_T{=}84.41$ / $E_S{=}50.50$）但错误模式互补，聚合阶梯
91.67 → 92.90 → **93.64** 单调递增。

---

## 8. 端到端实体抽取

检测（YOLO11x 或 RT-DETR-L，类别无关、conf $\ge$ 0.5）→ 按 (3-1) 保护性
裁剪（推理 $e{=}0.25$ 固定）→ 4 通道构造 → DEDF 判别 → 输出
（型号、军/民域、置信度）。评测匹配：预测框与 GT 贪心一对一，IoU $\ge$ 0.5
且型号正确记为正确实体；Entity P/R/F1 以全部 GT 为分母（含零检测图的
漏检）。

**实验流程八步**（与上文一一对应）：
① 数据与实体级样本构建（$\S$2）→ ② 同配置编码器选型（Swin 75.92 →
SigLIP2 90.64，确立 $E_v$）→ ③ 语义知识库构建与文本塔塌缩诊断
（$\S$2.4，SigLIP2 文本塔类间余弦 0.97、零样本 1.23% ≈ 随机，故换 CLIP）→
④ 模态消融 R/RS/RT/RST ×3 种子 + McNemar（RS +1.03 p=.036、RT +1.04
p=.022、联合 91.14 < RS 91.67）→ ⑤ 融合层级分析（特征/交互/决策 ×
四组合矩阵；$E_R/E_S/E_T$ 独立读数）→ ⑥ DEDF 三专家训练与聚合阶梯消融 →
⑦ 端到端集成（87.80 / 86.06 Entity F1，0.46 s/图）→ ⑧ 附加分析
（few-shot 曲线、增广增益分布、错误分析）。

**评测协议**：test 3,002 实体 / 1,669 图；指标 Top-1 / Top-5 / Macro-F1；
同配置极差 2.05 Macro-F1 点为噪声地板，<1 点差异不下结论，关键对比均做
逐实体 McNemar 配对检验。
