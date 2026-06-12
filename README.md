# BoilerWear-190

**Fine-Grained Ordinal Wear Estimation from Axial Panoramic Boiler Wall Images**

输入锅炉内壁 1536×256 轴向全景图，输出磨损程度 wear%（0–100%）。  
**SOFormer**（Strip-Ordinal Former）方法，以及三套 folder 级划分协议。

## 目录结构

```
MT-KY-EXP/
├── boilerwear/          # 核心 Python 包
├── configs/             # 数据集 / 模型 / 实验 YAML
├── tools/               # train.py, eval.py, prepare_splits.py
├── scripts/             # 一键批量实验
├── docs/                # 文档
├── datasets/            # 原始 BMP（不上传 GitHub）
└── outputs/             # 划分、结果、权重
```

完整说明见 [docs/STRUCTURE.md](docs/STRUCTURE.md)。

## 快速开始

```bash
pip install -r requirements.txt
pip install -e .

# 1. 生成划分
python tools/prepare_splits.py

# 2. 训练 SOFormer（Protocol-II 主实验）
python tools/train.py --model soformer --protocol protocol2 --seed 42 --batch-size 4

# 3. 测试集评估
python tools/eval.py --model soformer --protocol protocol2 --split test --seed 42

# 4. 一键跑完全部 baseline
bash scripts/run_all_baselines.sh protocol2 42
```

## 实验方案

| 项目 | 说明 |
|------|------|
| 主表 baseline | HOG+LR, ResNet50, EfficientNet-B4, **Swin-T, ConvNeXt-T**, CORAL, LDL, **CORN, StripPool**, **RNC**, **SOFormer**(rnc_resnet50 按确认项 ④ 默认 RNC 路线实现; 若改选 GOL/Ord2Seq 需替换) |
| 主实验协议 | **protocol2**（每 10 档 hold-out） |
| 外推协议 | **protocol3**（172–190 高磨损） |
| 消融 | SOFormer w/o AST / ADM / HOD / Mono / Uncertainty + **Bi-AST** + fuse 推理(`eval.py --infer-alpha/--infer-beta --tag fuse`) |
| 方法说明 | [docs/METHOD_SOFormer.md](docs/METHOD_SOFormer.md) |

| 文档 | 说明 |
|------|------|
| [docs/ALL_COMMANDS.md](docs/ALL_COMMANDS.md) | **逐条命令清单**（数据处理→训练→测试） |
| [docs/ALL_NOHUP_COMMANDS.md](docs/ALL_NOHUP_COMMANDS.md) | **全部 nohup 后台命令**（含所有模型/协议/消融） |
| [docs/EXPERIMENT_WORKFLOW_v3.3.md](docs/EXPERIMENT_WORKFLOW_v3.3.md) | **实验方案 v3.3（唯一正式版）** |
| [docs/SUBMISSION.md](docs/SUBMISSION.md) | 投稿结构与 venue 建议 |
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | 逐步复现 |

```bash
# 完整实验流水线
bash scripts/run_full_experiments.sh 42
```

## 引用

```bibtex
@article{boilerwear190,
  title={BoilerWear-190: Fine-Grained Ordinal Wear Estimation from Axial Panoramic Boiler Wall Images},
  year={2026}
}
```

## License

MIT

## v3.2 对齐说明(代码审查后)

- **P2 val 同构**:`val = folder %10 == (k+5)%10`(确定性),`--p2-offset k` 支持 k=0/3/7。
- **P3 区间**:train 1–139, val 140–152(从 train 1–152 内划出), 153–171 隔离带, test 172–190。
- **阶段 S 冒烟**:`prepare_splits.py --smoke-fraction 0.1`(按 %10 余数类分层抽样)+ `train.py --smoke --max-epochs 2`,产物落入 `seed{N}_smoke` 隔离目录。
- **梯度累积**:`train.py --batch-size 32 --accum-steps 4` 保持有效 batch=128(公平性规则 3)。
- **显著性检验**:`tools/significance_test.py --target soformer --level folder --test wilcoxon`(folder 级配对 Wilcoxon,依赖 eval.py 自动导出的 `*_predictions.csv`)。
- **原图支持**:`data.auto_resize: true`(默认开启)——任何分辨率的全景原图会在加载时双线性缩放到 256×1536 再做 6-strip 切分;若原图宽高比与 6:1 差异大,建议先离线 letterbox/裁剪以免形变。
