# 中文入口

[README.md](README.md) 是完整的英文项目说明，包含环境、数据、预训练、linear 与 MSP end-to-end 两套实验、参数和指标。本页提供中文导航与常用命令。

- [全部验证、训练、评估和分析脚本的职责](docs/SCRIPT_CATALOG.md)
- [中文参数规划、损失与实验协议](docs/PARAMETERS_AND_PROTOCOL.md)
- [迁移变化、兼容语义与验证范围](docs/MIGRATION.md)
- [模型选层与特征接口](models/README.md)
- [上游项目说明与作者信用](docs/UPSTREAM_README.md)
- [保留远程图片链接的历史 GitHub 首页](docs/LEGACY_GITHUB_README.md)

## 三个独立设置

| 配置字段 | 含义 |
| --- | --- |
| `attack.representation=features/blocks` | 生成 poison 时使用选定多层/末层特征，或原 BP/CP block 目标 |
| `victim.training=linear/end2end` | 受害只训练分类头，或更新全部预训练参数 |
| `victim.model_mode=eval/train` | 受害优化时的 BN/Dropout 模式；linear 只允许 eval |

**MSP 支持 linear 和 end-to-end 全参数微调**，都使用 `features`。例如 `MSP-Flat` 的方法名不变，受害场景由 `victim.training` 记录。全参数更新时使用 eval 仍会计算梯度，只固定 BN 运行统计。这里不是随机初始化训练，也没有对受害训练过程进行可微展开。

## 从头准备一次

先按英文 README 安装兼容的 PyTorch/torchvision 和 requirements。已有可信 split、匹配权重时可跳过相应步骤：

~~~powershell
python prepare_data.py --download --seed 2026
python workflow.py --config configs/smoke.json --stage pretrain
python -m unittest discover -s tests -v
python verify_layerwise.py
~~~

数据准备生成新的研究划分，不保证等同上游原始 split。smoke 的预训练仍为每模型 300 epochs，只有后续攻击/受害微调缩为 2 次/2 轮。切换受害训练场景可以复用相同干净权重。

## Linear 冒烟

~~~powershell
python workflow.py --config configs/smoke.json --stage plan
python workflow.py --config configs/smoke.json --stage craft
python workflow.py --config configs/smoke.json --stage evaluate
python summarize_results.py runs/smoke-seed2026 --config configs/smoke.json
~~~

## MSP end-to-end 冒烟

~~~powershell
python workflow.py --config configs/smoke_end2end.json --stage plan
python workflow.py --config configs/smoke_end2end.json --stage craft
python workflow.py --config configs/smoke_end2end.json --stage evaluate
python summarize_results.py runs/smoke-end2end-seed2026 --config configs/smoke_end2end.json
~~~

该预设为 `features + end2end + eval`，受害 Adam lr=1e-4、weight decay=0。正式配置分别为 [benchmark.json](configs/benchmark.json) 与 [benchmark_end2end.json](configs/benchmark_end2end.json)，使用 50 目标、18 代理、800 次攻击、60 轮受害训练。先把正式配置复制为仅目标 0 的 pilot，再扩展目标与随机种子。

`workflow.py --stage all` 只串行执行预训练、生成、评估；数据准备与汇总分别运行。已有权重时直接使用 craft/evaluate。不同受害场景及参数必须使用独立输出根目录；历史 E2E/CIFAR-100 结果不能当作这套新流水线的复现实测。