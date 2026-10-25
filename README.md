# Multi-scale Polytope Poisoning

基于 Bullseye Polytope（BP）与 Convex Polytope（CP）的干净标签投毒研究项目。MSP 在生成毒样本时引入多个 stage 的特征，提供池化拼接、展平拼接与逐层归一化对照，并用统一的数据、模型和受害训练协议比较这些方法。

本项目是对原实现的扩展。**MSP 的主要区别在攻击特征和损失聚合，不在给分类网络增加一套参数，也不在另换受害训练超参数。** 相同的预训练与受害训练条件是比较 BP/CP/MSP 所需要的控制变量；是否提升成功率，需要配对实验验证。

- [完整脚本目录：每个验证、训练、评估和分析脚本的作用](docs/SCRIPT_CATALOG.md)
- [参数、数学目标、实验规划与结果解释](docs/PARAMETERS_AND_PROTOCOL.md)
- [迁移说明、行为变化及验证范围](docs/MIGRATION.md)
- [模型选层与特征接口](models/README.md)
- [保留的上游项目说明](docs/UPSTREAM_README.md)
- [原 GitHub 首页与历史可视化结果](docs/LEGACY_GITHUB_README.md)

## 方法与支持范围

| 方法名 | 目标函数 | 攻击特征 | 场景 |
| --- | --- | --- | --- |
| CP-Original | 求单纯形系数的凸包重建误差 | 原末层特征 | linear transfer |
| BP-Original | 毒样本质心到目标的归一化范数 | 原末层特征 | linear transfer |
| MSP-Pool | BP 质心目标 | 三个 stage 分别全局平均池化后拼接 | linear transfer |
| MSP-Flat | BP 质心目标 | 三个 stage 展平后拼接 | linear transfer |
| MSP-Flat-Layerwise | 每层归一化误差的等权平均 | 与 MSP-Flat 相同的三个 stage | linear transfer |
| BP-3Layer-Pool-Control | 每层归一化误差的等权平均 | 三层各自池化 | linear transfer，受控对照 |
| BP-E2E-Original / CP-E2E-Original | 原版 block 特征目标 | 原模型 block 接口 | 原版 end-to-end 微调 |

MSP 模式目前只实现 linear transfer。将 MSP 特征与 end-to-end 同时启用会明确报错。原版 end-to-end 的攻击 block 接口支持 ResNet、ResNeXt、DPN、SENet；DenseNet、GoogLeNet、MobileNetV2 不提供该接口。end-to-end 在此指更新预训练受害模型全部参数，**不等于随机初始化后从零训练**。

## 目录

~~~text
configs/                         冒烟与正式实验配置
prepare_data.py                  从官方 CIFAR-10 训练集构建新 split
workflow.py                      计划、预训练、生成、评估的统一入口
train_cifar10_models.py          干净模型预训练、续训、权重单独评估
craft_poisons_transfer.py        单目标 BP/CP/MSP 分阶段入口
trainer.py                      共用攻击优化与受害训练
models/                         分类网络和多层特征接口
summarize_results.py             汇总全样本/条件 ASR 与干净准确率
tests/                          配置、数据契约、结果统计、真实张量回归
verify_layerwise.py             七种真实模型的特征和梯度检查
launch/  analysis/              历史实验包装与结果分析
Multi-Target-Mode/              独立的历史多目标实验
runs/                           新实验产物，默认被 Git 忽略
~~~

## 1. 安装环境

以下命令从仓库根目录执行。先按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)选择匹配操作系统和 CPU/CUDA 的 torch、torchvision，再安装其余依赖。requirements 是依赖范围，不是已验证的锁定环境。

~~~powershell
conda create -n msp python=3.11 -y
conda activate msp
# 在这里执行 PyTorch 官方选择器给出的安装命令
python -m pip install -r requirements.txt
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
~~~

只查看计划和运行不依赖张量的测试，不需要安装 PyTorch：

~~~powershell
python workflow.py --config configs/smoke.json --stage plan
python -m unittest discover -s tests -v
~~~

正式 GPU 实验将配置中的 device 设为 cuda；CPU 可执行小规模检查。Windows 的预训练 workers 默认使用 0。

## 2. 准备数据

已有上游 split 时，将其放到 datasets/CIFAR10_TRAIN_Split.pth，并保留其来源；严格复现上游结果需要相同的实际 split，不能只使用同名文件。

没有 split 时可生成新的、固定种子的研究划分：

~~~powershell
python prepare_data.py --download --seed 2026
~~~

生成：

~~~text
datasets/CIFAR10_TRAIN_Split.pth
datasets/CIFAR10_TRAIN_Split.json
~~~

默认从官方 CIFAR-10 的 50000 张**训练图片**中，逐类分成：

| 数据 | 数量 | 用途 |
| --- | --- | --- |
| clean_train | 4800/类，共 48000 | 干净特征提取器预训练 |
| others | 200/类，共 2000 | 受害微调、基图像和独立攻击目标 |
| 官方 test | 1000/类，共 10000 | 仅用于干净测试准确率 |

JSON 记录 seed、官方训练集源索引与 split SHA-256。该新划分不宣称等同于论文作者的原 split。新脚本拒绝覆盖已有数据文件。

默认受害训练取 others 中每类第 0..49 张，共 500 张；5 张 poison **原位替换**同类基图像，总数仍为 500。目标 index=0 指目标类第 50 张，index=49 指第 99 张，均不进入受害训练。修改训练窗口时，必须同步检查目标偏移和 base 偏移。

## 3. 准备干净模型

所有方法共享干净预训练权重。已有权重时核对配置中的模型名、模板和实际文件，直接进入下一步。

小规模流程只需 ResNet18 代理和 ResNet50 受害权重：

~~~powershell
python workflow.py --config configs/smoke.json --stage pretrain
~~~

这一步仍是**每个模型 300 epochs 的真实预训练**；smoke 只缩短后面的攻击与受害微调，不把未训练网络当成有效预训练权重。

单模型训练和恢复示例：

~~~powershell
python train_cifar10_models.py --net ResNet18 --epochs 300 --device cuda --output-dir model-chks
python train_cifar10_models.py --net ResNet18 --epochs 300 --device cuda --resume model-chks/ckpt-ResNet18-4800-latest.t7
python train_cifar10_models.py --net ResNet18 --device cuda --evaluate-only --resume model-chks/ckpt-ResNet18-4800.t7
~~~

最佳权重默认保存为 model-chks/ckpt-ResNet18-4800.t7；latest 文件保存优化器、调度器和 RNG 状态。epochs 表示总轮数，resume 从已完成轮次的下一轮继续。模型和训练配置必须与原 checkpoint 匹配。

## 4. 先跑一个目标的完整流程

[smoke.json](configs/smoke.json) 包含 1 个目标、1 个代理、1 个受害、5 种方法，攻击 2 次迭代、受害微调 2 epochs。它用于验证路径和流程，不用于比较算法效果。

~~~powershell
python verify_layerwise.py
python workflow.py --config configs/smoke.json --stage craft
python workflow.py --config configs/smoke.json --stage evaluate
python summarize_results.py runs/smoke-seed2026 --config configs/smoke.json
~~~

craft 阶段只优化图片像素、固定代理网络，不进行受害微调。evaluate 阶段不再生成 poison；对每个受害重新加载同一个 checkpoint，分别训练 clean 和 poisoned 条件，并使用配对的随机种子与样本位置。

需要一次执行所有阶段时可用 --stage all；它包含预训练，耗时很长。已有权重时使用上述分阶段命令。

## 5. 正式实验

[benchmark.json](configs/benchmark.json) 配置：

- 目标 0..49，6 种方法。
- 6 种代理架构 × 3 个 checkpoint 模板 = 18 个代理实例。
- dropout 为 0.2、0.25、0.3；net_repeat=1，不代表只有 1 个代理。
- 5 poisons，800 次优化，Adam lr=0.04，像素预算 epsilon=0.1。
- DenseNet121、ResNet18 两个受害；分类头 Adam lr=0.1、60 epochs、30/45 衰减、batch=64。
- seed=2026；正式报告还应复制配置分别执行多个种子并保存各自输出。

先查看计划：

~~~powershell
python workflow.py --config configs/benchmark.json --stage plan
~~~

确认权重准备方式后执行：

~~~powershell
# 没有相应权重才执行：18个代理权重 + 2个受害权重
python workflow.py --config configs/benchmark.json --stage pretrain
python workflow.py --config configs/benchmark.json --stage craft
python workflow.py --config configs/benchmark.json --stage evaluate
python summarize_results.py runs/benchmark-seed2026 --config configs/benchmark.json
~~~

正式配置有 300 个攻击任务，以及 300 × 2受害 × 2条件 = 1200 次受害训练。先复制配置将 targets 改为 [0]、output_root 改为新的 pilot 路径，跑完后再扩大到 50 个目标。

如果改变 seed、epsilon、迭代数、权重或训练参数，也要改变 output_root。相同目录的 protocol.json 与当前设置或文件哈希不同会拒绝执行，避免旧结果混入新实验。生成完的最终 poison 会复用；evaluate 显式执行时会重新训练并更新评估。

## 6. 输出与指标

~~~text
runs/benchmark-seed2026/
  MSP-Flat/
    flat/mean/800/0/
      protocol.json
      log.txt
      craft.json
      poison_00000.pth
      poison_00050.pth
      ...
      poison_00799.pth
      evaluation-DenseNet121-seed2026.json
      evaluation-ResNet18-seed2026.json
      evaluation.json
  summary.json
  summary.csv
~~~

poison 文件编号从 0 开始，因此 800 次迭代的最后文件是 poison_00799.pth。快照每 50 次和最后一次保存。

protocol 记录实际算法、全部参数、数据与模型 SHA-256；craft 记录生成耗时与 GPU 峰值分配内存；evaluation 记录基图像索引、实际扰动、clean/poisoned 目标预测、概率和测试准确率。旧日志分析器不应读取新 JSON 日志；新结果使用 summarize_results.py。

| 指标 | 含义 |
| --- | --- |
| ASR | 全部目标中，被投毒受害模型预测为 poison 类的比例 |
| conditional ASR | 只在干净微调受害模型原本正确的目标中计算 ASR |
| eligible targets | 条件 ASR 的分母；为 0 时结果为 null/N/A |
| clean accuracy drop | 干净微调与投毒微调后的测试准确率差，单位百分点 |
| seconds / peak_allocated_bytes | 成本记录；CPU 的 GPU 显存字段为 null |

配置指定的任务未齐全或发现额外结果时，汇总默认非零退出，缺失列表写入 summary.json；查看进行中结果可显式加 --allow-incomplete。汇总还会拒绝重复目标、混合训练协议和不一致的 base 选择。

## 7. 测试和历史入口

~~~powershell
python -m unittest discover -s tests -v
python verify_layerwise.py
~~~

tests 中的数据选择/加载契约使用 mock，可在没有训练依赖时执行；真实张量测试缺少 torch 会明确跳过。verify_layerwise 需要真实 PyTorch，覆盖七种架构，不需要数据或 checkpoint；它不能替代一次真实权重加载、生成和受害训练。

历史脚本、可用性研究、从零训练评估与车辆多目标实验的每个入口及限制，见 [脚本目录](docs/SCRIPT_CATALOG.md)。新版统一流程以 CIFAR-10 单目标实验为主；不能把 availability 的准确率下降当成 targeted ASR，也不能用目录名推断真实数据 overlap。

**本次验证状态（2026-10-25）：** 当前工作环境缺少 torch/torchvision、数据与 checkpoint；已执行的测试和未执行的完整实验详见 [迁移与验证记录](docs/MIGRATION.md)。仓库中的既有结果不作为本次重构后的实测成绩。

## 来源与引用

原实现来自 [Bullseye Polytope](https://arxiv.org/abs/2005.00191)，并沿用了 [Convex Polytope](https://github.com/zhuchen03/ConvexPolytopePosioning/) 的方法与实现基础；模型来源见上游说明。MSP 是本仓库的多层攻击特征扩展，不冒充原论文的官方复现或已发表成果。

~~~bibtex
@article{aghakhani2020bullseye,
  title={Bullseye Polytope: A Scalable Clean-Label Poisoning Attack with Improved Transferability},
  author={Aghakhani, Hojjat and Meng, Dongyu and Wang, Yu-Xiang and Kruegel, Christopher and Vigna, Giovanni},
  journal={arXiv preprint arXiv:2005.00191},
  year={2020}
}
~~~
