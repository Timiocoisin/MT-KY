# 实验方案 v3.3（一区投稿定稿版 · 唯一正式版 · 代码全量落地）

> **当前项目执行的实验方案**，涵盖论文定位 → 理论设计 → 数据处理 → P2 主表 → P3 外推 → SOFormer 消融 → 组件通用性验证 → 鲁棒性 → 显著性检验 → 分析图。
> 默认项目根：`/mnt/proj/MT-KY-EXP` · 默认 GPU：**0** · 主协议：**protocol2** · **主模型：`soformer`**
> **v3.1 前提（v3.3 更新）**：项目从零开始——无已训练权重、无历史结果；🆕 标记的脚本/配置**已全部实现**（见文末竣工清单）。任何全量训练前仍必须先通过 **阶段 S 冒烟测试闸门（§1.5）**。
> **v3.2 要点**：baseline 补齐近年序数 SOTA（CORN 必加 + 近年方法三选一，见 §4 与确认项 ④），堵住"对比方法过时"这一一区常见拒稿理由。
> **v3.3 要点**：实现清单 **P0/P1/P2 全部落地并通过端到端验证**；修复两处影响结果的实现 bug（CORAL 双重 off-by-one、mono 损失低效与梯度截断），按 §9 规则 4(b) **历史 coral/soformer 结果一律作废**；P2/P3 划分规则按本方案在代码中固化；确认项 ④ 已按默认 **RNC** 实现（单阶段联合变体）。

---

## v3.3 修订说明（相对 v3.2，🆕 代码全量落地版）

| # | 修订 | 原因 |
|---|------|------|
| 1 | **实现清单 P0/P1/P2 全部完成**（文末清单逐项标 ✅），每项均经合成数据端到端冒烟验证（单元测试 + 训练→评估→汇总→显著性→出图） | 阶段 S 前置条件就绪；阶段 E 脚本提前完成，不再阻塞任何步骤 |
| 2 | **修复 CORAL 双重 off-by-one**：(a) `coral_loss` 中 wear%→level 的 bucketize 缺 +1，除 level 1 外训练目标整体低一个 bin（≈−5.3%）；(b) `coral_logits_to_wear` 缺 +1 且取 bin 左边缘而非中点（再低 ≈−2.6%）。两者叠加使 coral baseline 输出与 SOFormer 的 HOD 序数监督被系统性拉低约 8% | **触发 §9 规则 4(b)**：修复前产生的一切 coral_resnet50 / soformer（含消融）结果作废，需全量重测并登记 tuning_log |
| 3 | **mono 损失向量化**并改作用于未截断的 `wear_pct_reg`（原 O(n²) Python 循环在 batch=128 下慢约 1700×，且 clamp 在 0/100 处梯度归零） | 训练效率与梯度正确性 |
| 4 | **P2 val 同构规则在代码固化**：`val = folder %10 == (k+5)%10`（确定性，原实现为随机 15% 违反 §3.2）；`--p2-offset k` 支持 k∈{0,3,7} | 划分协议与方案一致是 test 解封的前提 |
| 5 | **P3 区间修正**：train 1–139，val 140–152（自 train 1–152 内划出），153–171 为隔离带不参与，test 172–190（原实现误用 153–171 作 val） | 与 §3.2 一致 |
| 6 | **确认项 ④ 按默认 RNC 实现**：忠实官方数学（L1 标签距离、负 L2 特征相似度、温度 2、anchor 条件负样本集）；统一训练预算下采用**单阶段联合变体**（L1 + λ·RnC），官方为两阶段协议——**论文 Implementation Details 必须如实注明**；若导师改选 GOL/Ord2Seq 需替换实现 | 主表 11 模型全部就绪；RNC 排序质量依赖 batch 多样性，严守有效 batch=128 |
| 7 | **冒烟子集分层抽样**：`--smoke-fraction` 按 folder%10 余数类分层抽样，保证 P2 三个 split 在子集上均非空 | 纯随机抽样可能抽空某余数类导致 val/test 为空 |
| 8 | **eval.py 自动导出逐样本预测**（`{split}_predictions.csv`），显著性检验与外推曲线图依赖该文件；`export_predictions.py` 保留作独立重导出用 | 简化 §6.4 与分析图的数据流 |
| 9 | **原图支持**：`data.auto_resize: true`（默认开启）——任意分辨率全景原图加载时双线性缩放到 256×1536 再切 6-strip；宽高比偏离 6:1 较大时建议先离线 letterbox/裁剪。分辨率改变属输入分布变化，全部模型须重训 | 原生分辨率数据可直接使用 |
| 10 | benchmark 数据集（UTKFace/AFAD）**须手动下载**（许可条款），`prepare_benchmark.py --help` 含下载地址；划分为 image-level 80/10/10（seed 记入论文） | §10 落地细节 |
| 11 | 继承 v3.2 全部内容 | — |

---

## v3.2 修订说明（相对 v3.1，🆕 baseline 时效性补强版）

| # | 修订 | 原因 |
|---|------|------|
| 1 | §4 baseline 新增 **corn_resnet50**（CORN，2022，CORAL 改进版，必加）与 **rnc_resnet50**（Rank-N-Contrast，NeurIPS 2023，近年序数代表，三选一见确认项 ④） | 原序数 baseline 止于 CORAL(2020)/LDL，"对比方法过时"是一区常见直接拒稿理由；补齐后每个范式均有近年代表 |
| 2 | baseline 选取原则成文（§4 开头）：**范式覆盖完整 + 含近年代表 + 公开代码可公平复现**，而非盲目追榜 | 写入论文 rebuttal 弹药；明确不需要也不应该搬 CV 榜单 SOTA |
| 3 | 参考文献时效性规则成文（第一部分开头）：近 3 年占比 ≥50%、近 5 年 ≥80%，经典奠基单列 | 回应文献综述时效性审查 |
| 4 | §11 新增确认项 ④：近年序数方法三选一（RNC / GOL / Ord2Seq，默认 RNC） | 决定实现成本与对标叙事 |
| 5 | 全部训练循环、冒烟清单、算力估算同步更新：全量训练 72 → **约 80 次**（+8） | CORN 与近年方法各 +P2 三 seed +P3 一次 |
| 6 | 继承 v3.1 全部内容：阶段 S 冒烟闸门、P0/P1/P2 实现优先级、梯度累积公平性条款 | — |

---

## v3.1 修订说明（相对 v3.0，🆕 从零启动可行性版）

| # | 修订 | 原因 |
|---|------|------|
| 1 | 新增 **§1.5 阶段 S 冒烟测试与可行性闸门**：全量训练（~72 次）前，用小子集 + 短训跑通"划分→训练→评估→汇总→显著性→出图"完整链路 | 项目从零开始，~14 个脚本/配置尚未实现；不经端到端验证直接全量跑，任何一处断裂都会浪费数天算力 |
| 2 | 实现清单按 **P0/P1/P2 三级优先级** 排序，明确"阶段 S 只需 P0" | 避免在增强实验脚本上提前投入，缩短首次可行性确认周期 |
| 3 | 新增 **算力/显存可行性预检**：阶段 S 实测单 epoch 耗时与峰值显存，按公式外推 72 次训练总墙钟时间，给出 go / 降级 / no-go 决策点 | 1536×256 输入 + batch 128 在单卡上有 OOM 风险；总时长未实测前不可承诺 |
| 4 | 公平性规则补充 **梯度累积条款**：任何模型 OOM 时降 micro-batch 并用梯度累积保持有效 batch=128 统一 | 原方案"OOM 则 batch 64/32"会破坏 batch 统一的公平性规则 |
| 5 | §2.1 执行阶段表新增 **阶段 S**（位于阶段 A 之前），§11 执行顺序表同步更新 | 阶段闸门正式纳入流程 |
| 6 | 继承 v3.0 全部内容：论文定位与 claim 边界、等量去除假设、组件通用性验证、待确认事项清单、v2 全部纪律 | — |

---

## v3.0 修订说明（相对 v2）

| # | 修订 | 原因 |
|---|------|------|
| 1 | 新增 **§1.4 论文定位与 claim 边界**：明确投工程一区，定义"能说什么/不能说什么"与措辞规范 | 单数据集论文成败在于 claim 范围是否与数据范围匹配 |
| 2 | §1.3 重写：引入 **等量去除假设**（待确认），wear% 定义为"研磨进程百分比"，剩余% 定义闭环到工程目标 | 无物理测量条件下最强的可辩护表述 |
| 3 | Limitation 改为两层 + **部署标定 future work** 方案 | 给审稿人"作者已想清楚部署路径"的信号 |
| 4 | 新增 **§10 组件通用性验证**：HOD/mono/LDL 在公开序数 benchmark（UTKFace 或 AFAD）上对照 CORAL/LDL | 正面回应"方法是否只对自有数据有效"，单数据集论文的标准补强 |
| 5 | 新增 **§11 待确认事项清单**（工艺参数统一性 / 数据集可否公开） | 两项确认直接决定论文叙事强度 |
| 6 | 数据集公开（若可行）列为论文第三贡献 | "任务+数据集+方法"三合一显著提升一区接受度 |
| 7 | 继承 v2 全部纪律：val-only 调参、test 解封、三 seed 消融、显著性检验、P3 必做、StripPool/Bi-AST 对照 | — |

---

# 第一部分：理论依据与实验设计

> 本部分回答四件事：**论文定位（投哪、claim 什么）→ 为什么做（动机）→ 理论在哪（文献）→ 实验怎么设计（可检验假设）**。
> 正文引用编号 **[R1]–[R28]**，主列表为 **2021–2026** 深度学习 / IEEE / SCI / 顶会；经典奠基工作单独列出。
> **时效性规则（🆕 v3.2）**：定稿时近 3 年（2024–2026）文献占比 ≥50%，近 5 年 ≥80%；CORAL、LDL、序数回归综述等奠基性工作必引但单独成段，不挤占近年配额。

## 1. 研究问题、动机与论文定位

### 1.1 工业背景

锅炉内壁磨损在实验室中通过 **190 次可控研磨** 模拟，每次研磨后沿轴向拍摄全景图（1536×256）。任务是从单张图像估计当前磨损阶段（wear%，0–100%）。

### 1.2 为何不是普通回归？

| 性质 | 现象 | 理论后果 | 文献支撑 |
|------|------|----------|----------|
| **细粒度序数** | 190 个 folder 严格单调，相邻阶段极难区分 | 应用序数回归而非无约束 MSE | [R1][R7][R8] |
| **标签歧义** | 相邻 folder SSIM 高（实验测量） | 硬 one-hot 不合理，需 LDL 软分布 | [R6][R9] |
| **轴向几何** | 磨损沿宽度推进，图幅 6:1 | 不能直接全局池化，需 strip 序列建模 | [R10][R11][R12] |
| **部署风险** | 需对未见档位与高磨损外推可靠 | 设计 P2 插值 + P3 外推双协议 | [R13][R14] |

### 1.3 任务形式化（FOWSE）与输出语义

**Fine-grained Ordinal Wear Stage Estimation（FOWSE）**：

- 输入：\(I \in \mathbb{R}^{3\times256\times1536}\)
- 输出：\(s\in\{1,\ldots,190\}\)，\(\text{wear\%}=s/190\times100\)，**剩余% = 100 − wear%**
- 学习范式：**深度视觉特征 + 序数结构约束 + 标签分布建模**

**等量去除假设（论文 Methodology 必写，前提见 §11 确认项 ①）**：
190 次研磨在统一工艺参数（相同压力、时长、磨料、接触方式）下进行，故单次研磨的材料去除量近似恒定，研磨次数近似正比于累计材料去除量。在此假设下，wear% 不是任意序数编号，而是 **累计材料去除量的近似线性代理**；剩余% 近似对应剩余材料比例。
英文表述模板：*"Each grinding session was performed under identical process parameters; the session index therefore serves as an approximately linear proxy for cumulative material removal."*

> **Limitation（论文必写，两层）**：
> (1) wear% 为研磨进程的代理量，其与实际材料损失的精确映射未经物理测厚标定；
> (2) 全部数据来自单一实验装置/试件。
> **Future work（同段给出路径）**：部署到在役设备时，结合定期检修测厚点对 stage→壁厚做一次性标定即可获得物理磨损量输出，模型本身无需改动；跨装置泛化经由 §8 鲁棒性实验部分验证、由域适应技术承接。

### 1.4 论文定位与 claim 边界（🆕）

**目标期刊**：工程/仪器一区——IEEE TIM、IEEE TII、Measurement、Wear、Tribology International、MSSP。**不投** CV 期刊（TIP/PR），单数据集在该赛道不成立。

**三项贡献的写法**：
1. **任务与协议**：形式化 FOWSE 任务，提出 folder 级防泄漏 + 插值（P2）/外推（P3）双协议评测范式；
2. **数据集**（视 §11 确认项 ② 而定）：发布首个细粒度序数工业磨损图像数据集（19,643 图 / 190 阶段）；
3. **方法**：SOFormer（Causal AST + MS-ADM + HOD 深度监督），并在公开序数 benchmark 上验证序数组件的通用性（§10）。

**措辞规范（全文统一，写作时对照检查）**：

| ✅ 使用 | ❌ 禁用 |
|--------|--------|
| wear stage estimation / 磨损阶段估计 | thickness loss measurement / 壁厚损失测量 |
| grinding-progress percentage / 研磨进程百分比 | material loss percentage（未标定前） |
| approximately linear proxy under identical process parameters | ground-truth wear |
| extrapolation degradation study（P3 叙事） | P3 SOTA |

**claim 边界**：claim 单装置内对未见阶段（P2）与未见范围（P3）的泛化、对采集扰动的鲁棒性、序数组件在公开任务上的有效性；**不 claim** 跨装置直接部署精度、物理测厚精度。

## 2. 理论—实验对照总表（含成功判据）

| 实验步骤 | 要验证的假设 | 成功判据 | 若失败说明什么 |
|----------|--------------|----------|----------------|
| folder 级划分 + 泄漏检查 | 同 stage 多图不能跨 split | leak_check 通过 | 之前指标可能虚高 |
| 相邻 SSIM 分析 | 标签有序且相邻模糊 | 相邻 SSIM 显著高于远距对 | 硬分类也许足够 |
| **P2** 主表 11 模型 | SOFormer 优于全部 baseline | **MAE、QWK 配对 Wilcoxon p<0.05 且 3-seed 均值占优**；其余指标多数占优 | 结构或训练策略需再改（仅允许在 val 上迭代） |
| **StripPool 对照** | 增益来自核心模块而非 strip 预处理 | SOFormer 显著优于 StripPool | 贡献需重新归因到任务 co-design |
| **Bi-AST 对照** | 因果注意力优于双向 | Causal ≥ Bi（至少不劣且参数更省） | "因果"卖点降级为"序列建模" |
| **P3** 外推（全模型） | 高磨损 OOD 下 SOFormer 退化最小且保持序关系 | P3 上 Spearman/MedAE 相对最优 | 方法仅内插有效，需调整论文叙事 |
| 消融 w/o AST/ADM/HOD…（3 seeds） | 各模块有独立贡献 | 移除后主指标显著退化 | 可裁剪结构 |
| HOD 融合 vs 仅 reg | 推理弃用 HOD 是合理选择 | reg-only 不劣于融合 | 改用融合推理并更新叙事 |
| **公开 benchmark 组件验证** 🆕 | HOD/mono/LDL 序数组件具通用性 | HOD-Head ≥ CORAL、LDL 单头（同 backbone 同协议） | 组件贡献限于本任务，需弱化通用性表述 |
| 不确定性校准 | Uncertainty Head 输出可用 | ECE 低、可靠性图近对角线 | 从论文移除该头 |
| 鲁棒性扰动 | 方法对采集扰动稳定 | 扰动下 MAE 退化幅度小于 baseline | 补数据增广或收敛叙事 |
| 多 seed | 结论非单次初始化偶然 | std 小、排序稳定 | 需报 mean±std 并弱化结论 |

## 3. 数据划分

### 3.1 folder 级划分（防泄漏）

同一 folder 多张图近乎重复。图像级随机划分会让 test 出现训练见过的 stage → **标签泄漏**。一切划分均在 folder 级进行。

### 3.2 协议

| 协议 | 规则 | 验证能力 |
|------|------|----------|
| **P1** | folder 随机 70/15/15 | 可学性上界（辅助，不进主表） |
| **P2** ★主协议 | test = folder%10==k，**主报 k=0**；val 从剩余 folder 中按 %10==5 取 | **未见 stage 的插值泛化** |
| **P2-multi** | k∈{0,3,7} 三次重复（增强实验） | 划分敏感性 |
| **P3** ★必做 | train 1–152（其中 140–152 作 val），test 172–190 | **高磨损范围外推** |

P2 的 val 必须与 test 同构（均为"未见 stage"），否则 val 上选出的超参对 test 无指导意义。

P3 预期：通用 CNN 的 MAE 大幅上升、Acc@5%→0；论文不 claim P3 SOTA，而是 claim **首次系统评测该任务的外推退化** 并展示 SOFormer 退化最小。

### 3.3 SSIM 分析 → 支撑 LDL

相邻高 SSIM → 视觉标签边界模糊 → 高斯 LDL 分布 + KL 训练 [R6][R9]；CORAL 19-bin 与 LDL 190-way 构成 **HOD 两级头** [R1][R7]。论文中给出 SSIM-距离曲线图作为 LDL σ 选择依据。

## 4. 基线模型（10 个 + SOFormer）

**选取原则（🆕 v3.2，论文 Related Work / rebuttal 可直接引用）**：baseline 的合规标准是 **范式覆盖完整 + 每个范式含近年代表 + 有公开代码、可在统一协议下公平复现**，而非追逐榜单最高分。覆盖五个范式：经典特征下界 → 主流 CNN 回归 → 近年代表 backbone（Swin/ConvNeXt）→ 深度序数方法（经典 CORAL/LDL + 近年 CORN/RNC）→ strip 结构对照。

| # | 模型 | 配置 | 范式 | 备注 |
|---|------|------|------|------|
| 1 | hog_lr | `configs/model/hog_lr.yaml` | 经典特征 | 下界参照 |
| 2 | resnet50 | `configs/model/resnet50.yaml` | CNN 回归 | |
| 3 | efficientnet_b4 | `configs/model/efficientnet_b4.yaml` | 高效 CNN | |
| 4 | **swin_t** | `configs/model/swin_t.yaml` | Transformer 回归 | 近年代表 backbone |
| 5 | **convnext_t** | `configs/model/convnext_t.yaml` | 现代 CNN 回归 | 近年代表 backbone |
| 6 | coral_resnet50 | `configs/model/coral_resnet50.yaml` | 深度序数 | |
| 7 | ldl_resnet50 | `configs/model/ldl_resnet50.yaml` | 深度 LDL | |
| 8 | **corn_resnet50** 🆕 | `configs/model/corn_resnet50.yaml` | 深度序数（2022） | CORAL 原作者改进版，仅换损失，实现成本极低，**必加** |
| 9 | **rnc_resnet50** 🆕 | `configs/model/rnc_resnet50.yaml` | 近年序数代表（2023） | Rank-N-Contrast（NeurIPS 2023）；已按默认实现（单阶段联合变体，见 v3.3 修订 #6）；改选 GOL/Ord2Seq 需替换 |
| 10 | **strip_pool** | `configs/model/strip_pool.yaml` | 6-strip + 共享 encoder + 平均池化 + reg | **剥离 strip 预处理贡献的关键对照** |
| 11 | **soformer** ★ | `configs/model/soformer.yaml` | 自研（见 §5） | |

**公平性规则**（论文 Implementation Details 原文照写）：

1. 统一输入分辨率与 6-strip 切分可用性：整图模型按其标准预处理输入整图；strip 系模型用相同切分。
2. Baseline #2–#9 允许 ImageNet 预训练（对 baseline 有利）；SOFormer 与 strip_pool 的 AxialStripCNN **从零训练**。
3. 统一 CSV 划分、统一 `eval.py`、统一 **有效 batch-size 128**、统一早停规则（val MAE）。若某模型（如 Swin-T）在目标卡上 OOM，降低 micro-batch 并启用 **梯度累积** 保持有效 batch=128 不变（`--batch-size 32 --accum-steps 4`），在论文 Implementation Details 中如实注明；**禁止**直接改用不同有效 batch。
4. 每个模型在 val 上做相同预算的学习率网格（3 点），其余超参用各自论文默认值。
5. **预训练交叉核查（附录）**：ResNet50 from-scratch 一行，回应"预训练差异"质疑。

## 5. SOFormer 结构与创新点

```
1536×256 → 6 strips → AxialStripCNN → Causal AST+APE → MS-ADM → HOD-Head → wear%
```

### 5.1 模块清单

| # | 模块 | 训练 | 推理 | 创新等级 |
|---|------|------|------|----------|
| ① | Strip 切分（6×256） | — | — | ★★ 任务 co-design |
| ② | StripEncoder（共享） | ✅ | ✅ | — |
| ③ | AxialStripCNN | ✅ | ✅ | ★★ 域定制 backbone |
| ④ | Causal AST + APE | ✅ | ✅ | ★★★ **核心创新** |
| ⑤ | MS-ADM | ✅ | ✅ | ★★★ **核心创新** |
| ⑥ | HOD-Head（CORAL+LDL） | ✅ | 仅监督 | ★★ 深度监督策略（通用性经 §10 验证） |
| ⑦ | Reg Head | ✅ | ✅ **主输出** | — |
| ⑧ | Uncertainty Head | ✅ | 可选 | 须通过校准评估，否则移除 |
| ⑨ | mono / cal loss | ✅ | — | — |

### 5.2 Causal AST

strip 间 **因果** self-attention：条带 `j` 仅 attend 自身及之前条带，与磨损沿宽度单向推进一致。

**论文必备支撑**：(a) 注意力图可视化（不同磨损阶段的 attention pattern）；(b) 与 **Bi-AST（双向）** 的定量对照（§7）。仅当 Causal ≥ Bi 时才在论文中使用"因果先验"叙事，否则改述为"轴向序列建模"。

### 5.3 MS-ADM

一阶差分 \(\Delta f\) 近似 \(\mathrm{d}w/\mathrm{d}x\)；二阶差分 \(\Delta^2 f\) 刻画退化加速；门控融合 mean 与多尺度差分上下文。论文给出差分特征沿条带的可视化曲线，对应物理解释。

### 5.4 HOD + 推理策略

训练：ord + LDL + reg + mono + cal。**推理默认仅用 reg**（`infer_alpha=infer_beta=0`）。HOD 作深度监督。

**论文必备支撑**：消融表中加 **soformer_fuse**（推理融合 ord/ldl）一行，证明 reg-only 不劣于融合，从而正面回答"为何不用 HOD 输出"。

### 5.5 定稿配置

| 项目 | 值 |
|------|-----|
| 代码 | `boilerwear/models/soformer.py` |
| 配置 | `configs/model/soformer.yaml` |
| Backbone | `base_channels=64`, `blocks_per_stage=3` |
| 消融配置 | `soformer_wo_{ast,adm,hod,mono,uncertainty}.yaml`、`soformer_bi_ast.yaml`、`soformer_fuse`（仅推理开关） |

**注意**：Causal AST / MS-ADM 改版后须 **重新训练**；旧 checkpoint 不可直接加载。

## 6. 评估指标

### 6.1 精度指标（图像级 + folder 级各报一套）

| 指标 | 含义 |
|------|------|
| MAE / RMSE | wear% 误差（**MAE 为第一主指标**） |
| MedAE | 中位误差（P3 更稳健） |
| QWK | 序数一致性（**第二主指标**） |
| Acc@5% / Acc@10% | 容差命中率 |
| Spearman | 秩相关 |
| R² | 方差解释（P3 可为负） |

folder 级 = 同 folder 多图预测取中值后再算指标，反映"对一个磨损阶段的判断"，更贴近部署语义；主表报图像级，folder 级进附录。

### 6.2 校准指标（仅含 Uncertainty Head 的模型）

ECE、NLL、可靠性图（`tools/eval_calibration.py`）。

### 6.3 效率指标（主表附列或单独 Table）

Params (M)、FLOPs (G)、单图推理延迟 ms（GPU 0，batch=1，warmup 后取中位数；`tools/benchmark_efficiency.py`）。

### 6.4 统计检验

主对比（SOFormer vs 每个 baseline）：folder 级绝对误差做 **配对 Wilcoxon signed-rank**，报 p 值与效应量；3-seed 均值±std（`tools/significance_test.py`）。

## 7. 消融与多 seed

| 变体 | 配置 | 检验 | 需重训 |
|------|------|------|--------|
| Full | `soformer` | 完整系统 | — |
| w/o Causal AST | `soformer_wo_ast` | 序列建模必要性 | ✅ |
| **Bi-AST** | `soformer_bi_ast` | 因果 vs 双向 | ✅ |
| w/o MS-ADM | `soformer_wo_adm` | 多尺度差分 | ✅ |
| w/o HOD | `soformer_wo_hod` | 深度监督 | ✅ |
| w/o Mono | `soformer_wo_mono` | 单调损失 | ✅ |
| w/o Uncertainty | `soformer_wo_uncertainty` | 异方差校准 | ✅ |
| **HOD 融合推理** | `soformer_fuse` | 推理策略 | ❌（复用 Full 权重） |

**seed 规则**：主模型与全部需重训消融变体均跑 **0 / 42 / 123** 三 seed；深度 baseline 同样三 seed；hog_lr 单次。所有 mean±std 进表。

## 8. 鲁棒性评测（eval-time，无需重训）

对 P2 test 图像施加受控扰动后评估全部已训模型（`tools/eval_robustness.py`）：

| 扰动 | 级别 |
|------|------|
| 亮度/对比度 | ±10%、±20% |
| 高斯噪声 | σ=5、10（8-bit 尺度） |
| 轻微透视/平移 | ≤2% 图幅 |
| JPEG 压缩 | Q=90、70 |

报告各模型 MAE 退化曲线；预期 SOFormer 退化斜率 ≤ baseline。该实验是对"单装置数据"局限的内部分布偏移验证，论文中需明确写出这层用意。

## 9. 调参纪律与 test 解封规则（最高优先级）

1. **一切超参迭代只允许看 val 指标**。包括 `base_channels`、mono/cal 权重、学习率、σ 等。
2. 每次 val 调参运行登记在 `outputs/tuning_log.md`（日期、改动、val MAE/QWK）。
3. **test 解封条件**：val 上完成模型选择并冻结配置后，每个 (model, protocol, seed) 的 test **只评估一次**，结果直接进论文。
4. 若 test 结果不理想，**不得**回头改超参再测 test。允许的动作只有两种：(a) 如实报告并调整论文叙事；(b) 若发现实现 bug，修复后将该轮全部 test 结果作废、全量重测并在 tuning_log 中记录原因。
5. 严禁以"指标未第一"为由重跑 test。论文成功标准见 §2（主指标显著占优，而非全指标必赢）。

## 10. 组件通用性验证：公开序数 benchmark（🆕，附录实验）

**目的**：单数据集论文的标准补强——证明 HOD 两级头、mono 损失、LDL 策略不是只对自有数据有效的序数组件。Strip/AST/ADM 为任务定制，不参与迁移。

**benchmark 选择**（按可获得性择一）：

| 选项 | 任务 | 序数粒度 | 备注 |
|------|------|----------|------|
| **UTKFace** ★首选 | 人脸年龄估计 | 0–116 岁，细粒度 | 公开免费，与 190 阶段粒度类比最佳 |
| AFAD | 人脸年龄估计 | 15–40 岁 | CORAL 原文 benchmark，可直接对标其报告数字 |
| APTOS 2019 | 糖网分级 | 5 级 | Kaggle 公开，粒度偏粗，备选 |

**实验设计**：统一 ResNet50 backbone + 官方/惯例划分，对照四行——
(a) 纯回归头；(b) CORAL 头；(c) LDL 头；(d) **HOD 两级头（+ mono）**。
成功判据：(d) 在 MAE/QWK 上 ≥ (b)(c)。报 3 seed mean±std。

**论文写法**：附录一节 + 正文一句话引用（"the proposed ordinal components generalize beyond the wear dataset, see Appendix X"）。

## 11. 待确认事项清单（🆕，启动写作前必须落实）

| # | 事项 | 影响 | 确认结果 |
|---|------|------|----------|
| ① | **190 次研磨工艺参数是否全程统一**（压力/时长/磨料/接触方式） | 是 → §1.3 等量去除假设成立，wear% 可表述为材料去除量的近似线性代理；否 → 退回纯序数表述，删除线性代理段落 | ☐ 待确认 |
| ② | **数据集可否公开**（含脱敏方案、单位/导师许可） | 是 → 数据集列为第三贡献，论文升级为"任务+数据集+方法"；否 → 删除贡献 2，加 "data available upon reasonable request" | ☐ 待确认 |
| ③ | 公开 benchmark 取 UTKFace 还是 AFAD | 决定 §10 对标对象与划分协议 | ☐ 待确认 |
| ④ 🆕 | **近年序数 baseline 三选一**：RNC（Rank-N-Contrast，NeurIPS 2023，默认）/ GOL（NeurIPS 2022）/ Ord2Seq（ICCV 2023） | RNC 仅加对比损失、与 ResNet50 兼容、实现成本最低；GOL 为度量学习路线；Ord2Seq 为序列解码路线、实现最重。确认后配置名相应替换 `rnc_resnet50` | ☑ 代码已按默认 RNC 实现；导师改选则替换 |

---

# 第二部分：实验操作流程

> 所有命令均在项目根目录执行。训练/评估统一使用 `tools/` 下脚本，**无额外 shell 封装**。
> 🆕 标记的脚本/配置为新增，执行前需先实现（见文末清单）。

## 0. 环境准备（一次）

```bash
cd /mnt/proj/MT-KY-EXP
pip install -r requirements.txt
pip install -e .

python -c "import torch; print('CUDA:', torch.cuda.is_available())"
nvidia-smi

find datasets -name "*.bmp" | wc -l   # 期望 ~19643

# 可复现性：冻结环境
pip freeze > outputs/env_freeze.txt
git rev-parse HEAD > outputs/code_version.txt
```

## 1. 数据划分与 SSIM 分析

```bash
# k=0 主划分（含与 test 同构的 val：%10==5）
python tools/prepare_splits.py --data-root datasets --seed 42 --p2-offset 0
# 增强实验用的重复划分
python tools/prepare_splits.py --data-root datasets --seed 42 --p2-offset 3
python tools/prepare_splits.py --data-root datasets --seed 42 --p2-offset 7

python tools/analyze_label_noise.py --data-root datasets

cat outputs/reports/leak_check.json
head outputs/splits/protocol2_k0.csv
```

## 1.5 阶段 S：从零冒烟测试与可行性闸门（🆕 v3.1，全量训练前必过）

> **目的**：项目从零开始，在投入 ~72 次全量训练前，用最小代价验证三件事——
> ① 完整链路可跑通（实现无断裂）；② 显存可行（batch 128 / 梯度累积方案确定）；③ 总耗时可承受（实测外推，给出 go / 降级 / no-go）。
> 阶段 S 产生的一切指标 **仅用于工程验证，不进论文、不算 test 解封**（在 ≤10% 子集上训练的模型不构成模型选择）。

### 1.5.1 前置：仅需实现 P0 项

见文末实现清单的优先级列。阶段 S 只依赖 **P0**（swin_t/convnext_t/strip_pool/corn_resnet50 配置接入、`--smoke` 开关、`--accum-steps`），不依赖 benchmark / 鲁棒性 / 校准等 P1/P2 脚本。**例外（v3.3 已解除）**：`rnc_resnet50` 已实现并通过单模型冒烟，阶段 S 可直接将其纳入冒烟循环（共 11 个深度模型 + hog_lr）。

### 1.5.2 冒烟训练（子集 + 短训）

```bash
# 生成 10% folder 子集划分（保持 folder 级防泄漏与 P2 同构 val）
python tools/prepare_splits.py --data-root datasets --seed 42 --p2-offset 0 --smoke-fraction 0.1

# 每个深度模型跑 2 epoch（验证 forward/backward/ckpt 保存/早停逻辑均可执行）
for model in resnet50 efficientnet_b4 swin_t convnext_t coral_resnet50 ldl_resnet50 corn_resnet50 rnc_resnet50 strip_pool soformer; do
  python tools/train.py --model "$model" --protocol protocol2 --seed 42 \
    --batch-size 128 --gpu 0 --smoke --max-epochs 2 --no-progress
done
python tools/train.py --model hog_lr --protocol protocol2 --seed 42 --smoke

# 链路下游全部走一遍（用冒烟权重）
for model in resnet50 soformer; do
  python tools/eval.py --model "$model" --protocol protocol2 --split val --seed 42 --gpu 0 --tag smoke
  python tools/eval.py --model "$model" --protocol protocol2 --split test --seed 42 --gpu 0 --tag smoke
done
python tools/summarize_results.py --protocol protocol2 --tag smoke
python tools/significance_test.py --protocol protocol2 --target soformer --level folder --test wilcoxon --tag smoke
python tools/export_predictions.py --model soformer --protocol protocol2 --split test --seed 42 --gpu 0 --tag smoke
python tools/plot_scatter.py --predictions outputs/results/protocol2/soformer/seed42_smoke/test_predictions.csv \
  --out outputs/figures/smoke_scatter.png
```

> 注：冒烟阶段允许评 test，因为 (a) 仅 10% 子集 2 epoch 的权重不具备模型选择意义；(b) 目的是验证 `eval.py --split test` 与汇总/显著性脚本本身可运行。所有冒烟产物带 `--tag smoke` 隔离目录，正式阶段 C 解封规则不受影响。

### 1.5.3 显存与耗时实测 → 外推

```bash
# 峰值显存（每模型记录 nvidia-smi 峰值；OOM 则按公平性规则 3 改 micro-batch + 梯度累积）
# 单 epoch 耗时：从冒烟日志读取，按下式外推
```

外推公式（记入 `outputs/reports/feasibility_smoke.md`）：

- `T_full_epoch ≈ T_smoke_epoch × (1 / smoke_fraction)`（数据量近似线性）
- `T_per_run ≈ T_full_epoch × E_expect`（E_expect 取早停经验值，先按 60–80 epoch 估）
- `T_total ≈ Σ_models T_per_run × 训练次数`（P2 三 seed 31 次 + 消融 18 + P3 11 + P2-multi 6 + benchmark 12 + 杂项，见 §11）

### 1.5.4 闸门判据（全部 ✅ 才进入阶段 A）

| # | 判据 | 通过标准 |
|---|------|----------|
| S1 | 全部深度模型 + hog_lr 冒烟训练正常结束（rnc_resnet50 允许后补，见 §1.5.1） | 无报错、ckpt 与日志按 §9 路径落盘 |
| S2 | eval → summarize → significance → 出图链路通 | `smoke` 标签下各 CSV/图生成且字段完整 |
| S3 | leak_check 通过 | `leak_check.json` 无跨 split folder |
| S4 | 显存方案确定 | 每模型记录 batch/accum 配置，全部有效 batch=128 |
| S5 | 总耗时外推可承受 | `T_total` ≤ 可用机时；否则按 §11 降级顺序裁剪后重估 |
| S6 | 损失项数值健康 | soformer 各 loss（ord/ldl/reg/mono/cal）均下降、无 NaN/Inf |

任一不过 → 修复后重跑阶段 S；S5 不过且降级后仍超 → 与导师重议范围（no-go 信号），不得"先跑起来再说"。

## 2. P2 主实验（Table 2）

### 2.1 执行阶段

| 阶段 | 内容 | 何时 |
|------|------|------|
| **S** 🆕 | 冒烟测试与可行性闸门（§1.5）：链路 / 显存 / 耗时三项确认 | **现在（最先）** |
| **A** | `soformer` seed42 训练 + **val** eval，必要时 val 上调参迭代 | S 全部判据通过后 |
| **B** | 配置冻结后：soformer 三 seed + 全部 baseline 三 seed 训练 | A 在 val 达标后 |
| **C** | 全部模型 test 一次性评估 + 显著性检验（test 解封，见第一部分 §9） | B 完成后 |
| **D** | 消融三 seed（§4） | C 后 |
| **E** | P3 全模型 + 鲁棒性 + 效率 + 校准 + 公开 benchmark | 并行/随后 |

### 2.2 阶段 A：训练 + val 调参（前台，带进度条）

```bash
python tools/train.py \
  --model soformer --protocol protocol2 --seed 42 \
  --batch-size 128 --gpu 0 --progress

# 只看 val！
python tools/eval.py --model soformer --protocol protocol2 --split val --seed 42 --gpu 0
echo "$(date) | 改动说明 | val MAE=… QWK=…" >> outputs/tuning_log.md
```

### 2.3 后台训练模板（nohup）

```bash
mkdir -p logs/train

nohup python -u tools/train.py \
  --model soformer --protocol protocol2 --seed 42 \
  --batch-size 128 --gpu 0 --no-progress \
  > logs/train/soformer_protocol2_seed42_gpu0.log 2>&1 &

tail -f logs/train/soformer_protocol2_seed42_gpu0.log
```

### 2.4 阶段 B：配置冻结后全量训练（三 seed）

```bash
for model in resnet50 efficientnet_b4 swin_t convnext_t coral_resnet50 ldl_resnet50 corn_resnet50 rnc_resnet50 strip_pool soformer; do
  for seed in 0 42 123; do
    python tools/train.py --model "$model" --protocol protocol2 --seed "$seed" \
      --batch-size 128 --gpu 0 --no-progress
  done
done
python tools/train.py --model hog_lr --protocol protocol2 --seed 42 --batch-size 128 --gpu 0 --no-progress
```

### 2.5 阶段 C：test 解封（一次性）

```bash
for model in hog_lr resnet50 efficientnet_b4 swin_t convnext_t coral_resnet50 ldl_resnet50 corn_resnet50 rnc_resnet50 strip_pool soformer; do
  for seed in 0 42 123; do
    python tools/eval.py --model "$model" --protocol protocol2 --split test --seed "$seed" --gpu 0
  done
done

python tools/summarize_results.py --protocol protocol2          # 自动聚合 mean±std
python tools/significance_test.py --protocol protocol2 \
  --target soformer --level folder --test wilcoxon \
  --out outputs/results/significance_p2.csv                      # 🆕
```

### 2.6 P2-multi 划分敏感性（增强，仅 3 个代表模型）

```bash
for k in 3 7; do
  for model in soformer coral_resnet50 swin_t; do
    python tools/train.py --model "$model" --protocol protocol2 --p2-offset "$k" \
      --seed 42 --batch-size 128 --gpu 0 --no-progress
    python tools/eval.py --model "$model" --protocol protocol2 --p2-offset "$k" \
      --split test --seed 42 --gpu 0
  done
done
```

## 3. P3 外推（Table 3，★必做，全模型）

```bash
for model in hog_lr resnet50 efficientnet_b4 swin_t convnext_t coral_resnet50 ldl_resnet50 corn_resnet50 rnc_resnet50 strip_pool soformer; do
  python tools/train.py --model "$model" --protocol protocol3 --seed 42 --batch-size 128 --gpu 0 --no-progress
  python tools/eval.py --model "$model" --protocol protocol3 --split test --seed 42 --gpu 0
done

python tools/summarize_results.py --protocol protocol3 --out outputs/results/main_results_p3.csv
```

## 4. SOFormer 消融（Table 4，三 seed）

```bash
for model in soformer_wo_ast soformer_bi_ast soformer_wo_adm soformer_wo_hod soformer_wo_mono soformer_wo_uncertainty; do
  for seed in 0 42 123; do
    python tools/train.py --model "$model" --protocol protocol2 --seed "$seed" --batch-size 128 --gpu 0 --no-progress
    python tools/eval.py --model "$model" --protocol protocol2 --split test --seed "$seed" --gpu 0
  done
done

# HOD 融合推理对照（复用 Full 权重，仅评估）
for seed in 0 42 123; do
  python tools/eval.py --model soformer --protocol protocol2 --split test --seed "$seed" --gpu 0 \
    --infer-alpha 0.5 --infer-beta 0.5 --tag fuse        # 🆕 推理开关
done

python tools/summarize_results.py --protocol protocol2 --out outputs/results/ablation_p2.csv
```

## 5. 公开 benchmark 组件验证（🆕，附录 Table）

```bash
# 数据准备（以 UTKFace 为例；下载与划分脚本需新增）
python tools/prepare_benchmark.py --dataset utkface --data-root datasets_public/utkface

for head in reg coral ldl hod; do
  for seed in 0 42 123; do
    python tools/train_benchmark.py --dataset utkface --backbone resnet50 --head "$head" \
      --seed "$seed" --batch-size 128 --gpu 0 --no-progress
    python tools/eval_benchmark.py --dataset utkface --head "$head" --seed "$seed" --gpu 0
  done
done

python tools/summarize_results.py --benchmark utkface --out outputs/results/benchmark_utkface.csv
```

## 6. 校准 / 效率 / 鲁棒性

```bash
# 不确定性校准（仅含 uncertainty head 的模型）
python tools/eval_calibration.py --model soformer --protocol protocol2 --split test --seed 42 --gpu 0 \
  --out outputs/figures/reliability_soformer.png

# 效率基准
python tools/benchmark_efficiency.py \
  --models hog_lr resnet50 efficientnet_b4 swin_t convnext_t coral_resnet50 ldl_resnet50 corn_resnet50 rnc_resnet50 strip_pool soformer \
  --gpu 0 --out outputs/results/efficiency.csv

# 鲁棒性扰动评测（全部已训模型，seed 42）
python tools/eval_robustness.py --protocol protocol2 --split test --seed 42 --gpu 0 \
  --perturb brightness contrast gaussian_noise perspective jpeg \
  --out outputs/results/robustness_p2.csv
```

## 7. 预训练交叉核查（附录）

```bash
python tools/train.py --model resnet50 --protocol protocol2 --seed 42 --batch-size 128 --gpu 0 \
  --no-pretrain --no-progress
python tools/eval.py --model resnet50 --protocol protocol2 --split test --seed 42 --gpu 0 --tag scratch
```

## 8. 分析图

```bash
python tools/export_predictions.py --model soformer --protocol protocol2 --split test --seed 42 --gpu 0

python tools/plot_scatter.py \
  --predictions outputs/results/protocol2/soformer/seed42/test_predictions.csv \
  --title "SOFormer P2 test" \
  --out outputs/figures/p2_soformer_scatter.png

# 注意力可视化（支撑 Causal AST 叙事）
python tools/plot_attention.py --model soformer --protocol protocol2 --seed 42 --gpu 0 \
  --samples 6 --out outputs/figures/ast_attention.png     # 🆕

# P3 外推退化曲线（各模型 MAE vs 磨损区间）
python tools/plot_extrapolation.py --protocol protocol3 \
  --out outputs/figures/p3_degradation.png                # 🆕
```

## 9. 产出路径

| 类型 | 路径 |
|------|------|
| 权重 | `outputs/checkpoints/{protocol}/{model}/seed{N}/best.pt` |
| 训练日志 | `outputs/results/{protocol}/{model}/seed{N}/train.log` |
| 测试指标 | `outputs/results/{protocol}/{model}/seed{N}/test_metrics.json` |
| 主表 | `outputs/results/main_results.csv` |
| 消融表 | `outputs/results/ablation_p2.csv` |
| 显著性 | `outputs/results/significance_p2.csv` |
| benchmark 表 | `outputs/results/benchmark_utkface.csv` |
| 效率表 | `outputs/results/efficiency.csv` |
| 鲁棒性 | `outputs/results/robustness_p2.csv` |
| 调参日志 | `outputs/tuning_log.md` |
| nohup 日志 | `logs/train/{model}_{protocol}_seed{N}_gpu{G}.log` |

## 10. 常见问题

| 问题 | 处理 |
|------|------|
| `device=cpu` | 检查 `nvidia-smi`，确认 `--gpu 0` |
| CUDA OOM | 降 micro-batch 并加 `--accum-steps` 保持有效 batch=128（如 `--batch-size 32 --accum-steps 4`，Swin-T 可能需要）；配置记入 feasibility_smoke.md |
| nohup 无输出 | 使用 `python -u` 与 `--no-progress` |
| 训练中断 | 重跑同命令（暂不支持 resume） |
| **val 指标不达标** | 在 **val** 上调 `base_channels` / mono 权重 / lr，登记 tuning_log，重跑阶段 A；**不得动 test** |
| **test 指标不理想** | 见第一部分 §9：如实报告或修 bug 后全量重测；**严禁针对 test 调参** |
| 旧结果无效 | Causal AST+MS-ADM 改版后须重训 soformer 及全部消融；**v3.3 CORAL off-by-one 修复同样触发**：修复前的 coral_resnet50 / soformer（含消融）结果一律作废重测 |

## 11. 推荐执行顺序与算力估算

| 步骤 | 说明 | 训练次数 |
|------|------|----------|
| 0 | **落实 §11 三个待确认事项** | 0 |
| 0.5 🆕 | **实现 P0 清单项** + 环境准备（§0） | 0 |
| 1 | splits（k=0/3/7）+ SSIM 分析 | 0 |
| 1.5 🆕 | **阶段 S 冒烟测试与可行性闸门**（§1.5，10% 子集 2 epoch × 11 深度模型 + hog_lr） | 11（小，~数小时级） |
| 2 | 阶段 A：soformer val 调参迭代（期间并行实现 P1 清单项） | 1–3 |
| 3 | 阶段 B：10 深度模型 ×3 seed + hog_lr | 31 |
| 4 | 阶段 C：test 解封 + 显著性检验 | 0 |
| 5 | 消融 6 变体 ×3 seed + fuse 推理 | 18 |
| 6 | P3 全模型 ×1 seed | 11 |
| 7 | 公开 benchmark 4 头 ×3 seed（依赖 P2 清单项） | 12（小图快训） |
| 8 | P2-multi（k=3/7 × 3 模型） | 6 |
| 9 | 校准/效率/鲁棒性/from-scratch 核查 | 1 |
| 10 | 图表 + 填表写论文 | 0 |

**合计约 80 次全量训练 + 10–11 次冒烟短训**（其中 12 次为公开 benchmark 小规模快训）。算力紧张时降级顺序：
P2-multi 砍掉（−6）→ efficientnet_b4 / convnext_t 降单 seed（−4）→ P3 仅跑 6 个代表模型（hog_lr / resnet50 / coral / corn 或 rnc / strip_pool / soformer，−5）→ benchmark 降单 seed（−8）。
**注意**：corn_resnet50 与 rnc_resnet50 **不在降级清单中**——砍掉它们等于重新打开"对比方法过时"的拒稿理由，宁可降其他项。
**不可降级项**：阶段 S 冒烟闸门、val-only 调参纪律、消融三 seed、StripPool 与 Bi-AST 对照、显著性检验、P3 必做、benchmark 实验本身（可降 seed 不可砍）。

---

## 需新增实现清单（按优先级，🆕 v3.1）

> **P0 = 阶段 S 冒烟测试前必须完成；P1 = 阶段 B/C/D 前完成；P2 = 对应增强实验前完成。**
> **🆕 v3.3：下表全部项目已实现并通过端到端冒烟验证，本清单转为竣工记录。**

| 优先级 | 项 | 类型 | 说明 |
|--------|----|------|------|
| **P0** | `configs/model/swin_t.yaml` + 接入 | 配置/代码 | timm `swin_tiny` + 回归头 ✅ 已实现 |
| **P0** | `configs/model/convnext_t.yaml` + 接入 | 配置/代码 | timm `convnext_tiny` + 回归头 ✅ 已实现 |
| **P0** | `configs/model/strip_pool.yaml` | 配置 | 复用 StripEncoder，平均池化 + reg head ✅ 已实现 |
| **P0** 🆕 | `configs/model/corn_resnet50.yaml` + CORN 损失 | 配置/代码 | 在 coral_resnet50 基础上换 CORN 条件训练损失，成本极低 ✅ 已实现 |
| **P0** 🆕 | `prepare_splits.py --smoke-fraction` | 代码 | folder 级抽 10% 子集，保持防泄漏与同构 val ✅ 已实现 |
| **P0** 🆕 | `train.py --smoke --max-epochs` | 代码 | 短训开关，产物落入 `*_smoke` 隔离目录 ✅ 已实现 |
| **P0** 🆕 | `train.py --accum-steps` | 代码 | 梯度累积，保障有效 batch=128 公平性 ✅ 已实现 |
| **P0** 🆕 | `eval.py / summarize / significance --tag` | 代码 | 冒烟产物与正式结果目录隔离（fuse 推理同用此开关） ✅ 已实现 |
| P1 | `configs/model/soformer_bi_ast.yaml` | 配置 | AST 去 causal mask ✅ 已实现 |
| P1 🆕 | `configs/model/rnc_resnet50.yaml` + RNC 损失接入 | 配置/代码 | Rank-N-Contrast 官方实现移植（确认项 ④ 可换 GOL/Ord2Seq）；完成后补单模型冒烟 ✅ 已实现 |
| P1 | `eval.py --infer-alpha/--infer-beta` | 代码 | fuse 推理开关 ✅ 已实现 |
| P1 | `prepare_splits.py --p2-offset` | 代码 | k 偏移划分 + 同构 val（k=0 为 P0 默认路径） ✅ 已实现 |
| P1 | `tools/significance_test.py` | 脚本 | folder 级配对 Wilcoxon ✅ 已实现 |
| P1 | `train.py --no-pretrain` | 代码 | from-scratch 开关 ✅ 已实现 |
| P2 | `tools/prepare_benchmark.py` | 脚本 | UTKFace/AFAD 下载与划分 ✅ 已实现 |
| P2 | `tools/train_benchmark.py` / `eval_benchmark.py` | 脚本 | 公开 benchmark 四头训练/评估 ✅ 已实现 |
| P2 | `tools/eval_calibration.py` | 脚本 | ECE/NLL/可靠性图 ✅ 已实现 |
| P2 | `tools/benchmark_efficiency.py` | 脚本 | Params/FLOPs/延迟 ✅ 已实现 |
| P2 | `tools/eval_robustness.py` | 脚本 | 扰动评测套件 ✅ 已实现 |
| P2 | `tools/plot_attention.py` / `plot_extrapolation.py` | 脚本 | 论文图 ✅ 已实现 |

---

*模型代码：`boilerwear/models/soformer.py` · 主配置：`configs/model/soformer.yaml` · 调参日志：`outputs/tuning_log.md`*
