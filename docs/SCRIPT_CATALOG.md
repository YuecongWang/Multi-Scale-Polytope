# 脚本与实验目录

本目录按源码说明每个 Python、Bash 和 PowerShell 文件的职责。命令均从仓库根目录执行；`Multi-Target-Mode/` 的历史脚本通常要求先切换到该目录。结果目录里的 JSON、日志和模型权重是实验产物，不是测试脚本，也不能反推其完整生成流程。

## 先分清三类工作

| 工作 | 实际更新什么 | 目的 |
| --- | --- | --- |
| 干净预训练 | 代理模型或受害模型的网络参数 | 获得用于生成毒样本、迁移学习的起始权重 |
| 毒样本生成 | 基图像的像素，网络权重固定 | 令目标特征接近毒样本特征构成的凸包或质心 |
| 受害模型训练与评估 | 冻结特征提取器后的分类头，或全部受害模型参数 | 测量加入毒样本后的目标预测、攻击成功率与干净准确率 |

BP、CP 和 MSP 可以共用同一套干净预训练及受害模型训练协议。它们的主要实验变量是毒样本生成时的特征与目标函数。为公平对照，不应仅为使两个项目“看起来不同”而同时更换数据划分、受害训练轮数或优化器。

脚本类别如下：

- **数值验证**：使用断言检查特征、损失和梯度；不会生成研究结论。
- **训练/研究实验**：会运行攻击优化或受害模型训练；耗时随代理数量、目标数量、迭代和训练轮数增加。
- **离线分析**：读取已存在的结果，汇总或绘图；不会训练模型。
- **支持模块**：供其他脚本导入，不是独立命令行入口。
- **历史入口**：保留既有实验组织方式；其中硬编码路径、结果格式和参数假设需要与实际实验逐项匹配。

## 建议阅读顺序

先读 [项目说明](../README.md) 中的完整工作流和参数规划，再按下列职责定位源码：

1. 数据准备：`prepare_data.py`，或使用能追溯来源的已有 split。
2. 查看统一计划：`python workflow.py --config configs/smoke.json --stage plan --dry-run`。这一步只校验配置并打印命令，不导入训练框架。
3. 干净预训练：`workflow.py --stage pretrain` → `train_cifar10_models.py`。已有合适权重时直接进入后续阶段；不要为跑 smoke 无意中重训全部网络。
4. 特征/损失验证：`verify_layerwise.py` 及 `tests/` 中的配置、协议回归测试。
5. 毒样本生成：`workflow.py --stage craft` → `craft_poisons_transfer.py --stage craft` → `trainer.py`。
6. 配对受害评估：`workflow.py --stage evaluate`，对 clean/poisoned 各自从相同权重重新训练；`summarize_results.py` 汇总新格式。
7. 历史 800 轮毒样本可用 `evaluate_saved_poisons.py` 复评；旧格式 JSON/Pickle 可用 `analysis/`。两者不能只改目录名就替代新协议。
8. 可用性实验与外部车辆多目标实验属于不同协议，分别参见相应章节，不与单目标迁移攻击的 ASR 混报。

`configs/smoke.json` 缩小了生成迭代、目标与受害训练轮数，用来验证执行链；其中 pretrain 仍保留正式的 300 epochs。`configs/benchmark.json` 给出 50 目标、18 代理、6 方法和 2 受害的受控比较。`--stage all` 会先执行完整预训练，再生成，再评估，只有准备承担全部计算成本时才使用。

`workflow.py` 仅负责 plan/pretrain/craft/evaluate/all；数据准备和结果汇总分别执行 `python prepare_data.py ...` 与 `python summarize_results.py <output_root> --config <config.json>`，不会被 `workflow.py --stage all` 自动调用。

## 根目录：验证、训练、生成与评估

| 脚本 | 类别与作用 | 输入 | 输出及适用范围 |
| --- | --- | --- | --- |
| [`workflow.py`](../workflow.py) | **推荐统一入口**。校验 JSON 配置，注册方法，计划并依次执行 pretrain/craft/evaluate；相对路径按仓库根解析 | `--config` 指定配置；`--stage plan/pretrain/craft/evaluate/all`；`--dry-run` 只打印命令；`--json` 输出机器可读计划 | 计划阶段无模型训练；执行阶段调用下表训练/攻击器并检查子进程退出状态。方法差异来自注册表，代理与受害协议来自同一份配置 |
| [`prepare_data.py`](../prepare_data.py) | **数据准备**。从官方 CIFAR-10 训练集按类别和种子确定性划分 clean_train/others，显式检查源样本 ID 不相交 | `--root/--output/--seed/--clean-per-class`；仅在给出 `--download` 时下载官方数据 | 默认生成每类 4800/200、合计 48000/2000 的 split `.pth`，以及同名 `.json` 来源清单、源样本 ID 和 SHA-256；已有输出拒绝覆盖。生成的是新研究划分，不能标为作者原始 split；对应权重应使用该划分重新训练 |
| [`verify_layerwise.py`](../verify_layerwise.py) | **CPU 数值验证**。从 `trainer.py` 提取特征函数和质心损失函数；用 Toy 网络校验归一化残差为 0.5、重复采样结果与输入梯度，再校验真实网络的逐层拼接一致性和梯度 | PyTorch、根目录 `trainer.py`、`models/`；不需要数据或预训练权重 | 控制台 `PASS`；断言失败则非零退出。覆盖 DPN92、SENet18、ResNet50、ResNeXt29_2x64d、GoogLeNet、MobileNetV2、DenseNet121 七种架构的 flat/pool 逐层接口；不覆盖真实权重加载或完整攻击 |
| [`train_cifar10_models.py`](../train_cifar10_models.py) | **干净预训练**。在 split 的 `clean_train` 中逐类选择 `[sidx,eidx)`，用裁剪/翻转增强和 SGD 训练模型；支持 Windows DataLoader、CPU/CUDA 和训练恢复 | `--data-path/--dset-path`、模型名、dropout、种子、每类索引范围、学习率、总 epochs、milestones；`--resume` 继续训练，`--evaluate-only --resume` 仅评估 | 默认 `model-chks/` 下最佳与 `-latest` `.t7`。可用 `--output-dir/--checkpoint-name` 对齐配置模板；保存模型、优化器、调度器、RNG、配置及指标。旧权重缺恢复状态时给出明确提示 |
| [`train_cifar10_models.sh`](../train_cifar10_models.sh) | **历史预训练包装器**。一次调用上述训练器 | 位置参数 `$1=GPU`、`$2=模型名` | 固定 seed=2020、dropout=0、每类 `[1,2400)` 的权重；不是训练所有代理/受害模型的总入口 |
| [`craft_poisons_transfer.py`](../craft_poisons_transfer.py) | **单目标主入口**。`--stage craft` 仅优化像素，`evaluate` 读取已有毒样本并配对训练 clean/poisoned 受害，`all` 两者串行 | split、权重模板、代理/受害列表、目标类/索引/offset、基图像类、`mode`、`feature-mode`、像素预算和训练参数；`--eval-poison-path` 可导入历史快照；支持 CPU/CUDA | `<chk-path>/<feature-mode>/<mode[-NRepeat]>/<iterations>/<target>/` 下的 `protocol.json`、毒样本快照、`evaluation-<victim>-seed<seed>.json`、`evaluation.json`、日志与摘要。记录数据/权重哈希，协议变更拒绝复用输出目录；`mean` 是 BP 质心目标，`convex` 是 CP，MSP 由 feature mode 选择 |
| [`craft_poisons_bp3pool.py`](../craft_poisons_bp3pool.py) | **兼容的三层池化逐层 BP 入口**。转发到主 `craft_poisons_transfer.py`，默认补入 `--mode mean --feature-mode pool-layerwise`，统一使用同一训练器 | 与主攻击器相同的 CLI 参数；未显式传入时才补默认 mode/feature mode | 与主流程相同的协议、毒样本与配对评估格式；该文件名里的“3”表示三层，不是 `--net-repeat 3` |
| [`eval_poisons_transfer.py`](../eval_poisons_transfer.py) | **独立受害重训练与评估**。读取毒样本快照后重新加载受害模型；支持分类头迁移和 end-to-end 微调 | `--eval-poisons-root` 指向包含目标编号子目录的迭代根目录；受害权重、split、目标范围和微调参数 | 每目标 `eval-retrained-for-<epochs>epochs.json`，含预测、恶意类别分数、干净准确率与攻击日志统计。目标终点为 Python `range` 的排除端点；不是只做一次前向 |
| [`eval_poisons_scratch.py`](../eval_poisons_scratch.py) | **历史从零训练评估**。随机初始化受害网络，读取固定列表的攻击快照分别训练 | split、既有毒样本、目标范围、受害模型名、训练参数 | `eval-scratch-for-<epochs>epochs.json`。源码快照列表固定为 1、51、101、201、401、801、1601、2801、4000 对应的计数；不是自动适配所有迭代数的通用入口；耗时大 |
| [`evaluate_saved_poisons.py`](../evaluate_saved_poisons.py) | **带记录的历史结果复评**。不生成毒样本；对每个方法/目标/受害网络重新训练分类头，并测量前后指标 | `experiments.json` 中的方法目录、数据及权重路径；默认配置现指向本项目 benchmark，`--start/--end` 为闭区间；默认 ResNet50 | `evaluation_recorded_v2/seed<seed>/<method>/<victim>/<target>/{result.json,log.txt}` 和 `per_target.csv`。固定读取 `poison_00799.pth`，固定 CIFAR-10 6→8、每类 50、Adam 0.1、60 epochs；要求 CUDA。记录 SHA-256、各类准确率、耗时和显存；已完成结果的来源哈希匹配才跳过 |
| [`summarize_results.py`](../summarize_results.py) | **推荐新格式汇总器**。递归读取 `evaluation.json`，按方法/受害/种子统计，拒绝重复目标身份 | 位置参数 `root` 为一个实验输出根；`--config` 校验配置要求的方法×目标×受害是否完整；`--allow-incomplete` 允许显式汇总不完整结果 | `summary.json`、`summary.csv` 和控制台 ASR、条件 ASR、clean/poisoned 平均干净准确率及下降百分点。条件 ASR 只在 clean 微调后正确分类的目标上计算；无符合目标时为 null/N/A；不读取旧日志，不训练模型 |
| [`clean_test_evaluation.py`](../clean_test_evaluation.py) | **纯权重干净测试评估**。逐个加载预训练模型，在官方 CIFAR-10 测试集执行前向；不生成毒样本、不训练或微调 | `--nets`（别名 `--target-net`）、`--model-resume-path`、`--test-chk-name`、`--dset-path`、`--device cpu/cuda`、GPU 和 batch size；仅 `--download` 显式允许下载 | `--output` 指定 JSON，默认 `runs/pretrained-clean-evaluation.json`；记录模板、设备、各模型正确数/样本数/准确率。该预训练权重指标与主流程 clean 微调后指标分别报告 |
| [`compute_coeffs.py`](../compute_coeffs.py) | **历史结果补充分析**。为已有毒样本重新求单纯形最小二乘系数，填补评估 JSON 中缺失的 `coeff_list` | 既有评估 JSON、毒样本快照、代理权重和 split | **原地写回** `eval-retrained-for-<epochs>epochs.json`。源码固定遍历 0..79，并断言代理数为 18；声明的 `--target-index` 不控制此循环。用于系数分布分析，不是攻击训练总入口 |
| [`defense.py`](../defense.py) | **防御研究实验**。比较无防御、kNN 特征过滤、类内 L2 离群过滤，并为各防御设置重训受害网络 | 既有毒样本、受害权重、split、k 值/删除比例网格 | `<eval-poisons-root>/defenses/{no-defense,knn,l2outlier}.pickle`；会大量重训。当前单目标分支针对 102flowers，CIFAR-10 分支显式断言；快照计数固定 801，即文件 `poison_00800.pth`，与 800 次攻击的最终 `poison_00799.pth` 不同 |
| [`defense.sh`](../defense.sh) | **历史防御包装器**。执行 102flowers、25 poisons 的防御配置 | `$1=GPU`；硬编码的攻击结果目录与权重 | 调用 `defense.py` 扫描目标索引 `[1,11)` 的八个受害网络；应先确认该专用数据与快照存在 |
| [`run_availability_controls.py`](../run_availability_controls.py) | **可用性对照研究**。生成多目标质心毒样本；再从零训练 clean、unperturbed、random、msp 四组受害模型 | CIFAR-10 split 必须为 48000/2000，单代理权重，生成种子、训练种子、受害架构、阶段和输出路径 | `protocol.json`、48 组 `.pth/.json`、每组受害 `result.json/final_model.pth`、`summary.csv`。固定每组 100 targets/10 poisons、800 次优化、epsilon=0.1；使用 400 张额外图像。衡量总体/各类干净准确率下降，不是单目标 ASR |
| [`run_availability_two_groups.py`](../run_availability_two_groups.py) | **可用性二组消融研究**。与上一个脚本共用数据组织，但生成目标是归一化残差平方的一半，受害对照只有 clean/msp | 与上行相同；应使用不同的 `--out` 保存不同协议 | 协议标识 `reconstructed_availability_squared_two_groups_v2`，输出结构同上。文件名的 two groups 指两组受害训练条件；毒样本仍要求 48 个生成组，不是只生成两组 |

### 可用性脚本的阶段边界

`--stage generate --max-groups 1` 只适合生成冒烟检查。`--stage train` 和 `--stage all` 的训练阶段会检查全部 48 个生成组；减小 `--max-groups` 不会自动改成一个完整的小规模训练协议。两种可用性脚本的目标函数、条件组数不同，必须分别保存，不能合并均值。

## `tests/`：自动回归验证

```bash
python -m unittest discover -s tests -v
```

| 脚本 | 输入与检查内容 | 输出及不能替代的验证 |
| --- | --- | --- |
| [`tests/test_workflow.py`](../tests/test_workflow.py) | **标准库回归测试**。读取 smoke/benchmark 配置；检查预训练/攻击命令能被真实 CLI 解析、方法路由、18 代理/300 方法目标任务、空调度、checkpoint 冲突、布尔解析、目标越界/重叠、配置拼写和协议混用；用临时 JSON 检查 ASR 分母与去重 | unittest 结果；不导入训练框架、不需要数据/权重，不执行真实训练 |
| [`tests/test_data_contracts.py`](../tests/test_data_contracts.py) | **标准库数据/加载契约测试**。用 AST 读取生产选择/加载函数和小型 mock，检查毒样本替换与每类预算、subset group、位置保留、零扰动与 clean 顺序一致、标签/索引拒绝、基图像 offset、checkpoint 格式/路径/DataParallel 前缀和 CPU/CUDA 路由 | unittest 结果；验证生产 Python 逻辑，但没有真实文件读取、模型前向、张量、自动微分或 GPU |
| [`tests/test_attack_tensors.py`](../tests/test_attack_tensors.py) | **真实 PyTorch 数值测试**。用小型 Toy 网络检查逐层等权损失区别于拼接隐含能量加权，并检查输入梯度、非默认 epsilon 像素投影、代理权重冻结 | unittest 结果；没有 PyTorch 时明确 `skipped`。Toy 网络通过仍不意味着真实 checkpoint/架构或完整攻击通过，应接着运行 `verify_layerwise.py` 和一个目标的完整实验 |

当前套件共 35 项，其中 2 项是真实张量测试，缺少 PyTorch 时显式跳过。`skipped` 表示相应验证未运行。应报告实际通过数与跳过数；标准库回归通过、真实张量测试通过、真实权重单目标冒烟通过和完整研究结果是四个不同层级。

## 根目录：批处理 PowerShell

这些包装器会启动真实攻击和受害训练，不能用来检查依赖是否安装。每个目标包含 6 架构 × 3 权重模板 = 18 个代理实例，通常评估 DenseNet121 和 ResNet18；默认设定为 seed=2026、5 poisons、800 次优化、epsilon=0.1、60 epochs。`net-repeat` 与代理实例数是两个独立维度。

| 脚本 | 实验条件及输入 | 输出/运行行为 |
| --- | --- | --- |
| [`run_layerwise_all.ps1`](../run_layerwise_all.ps1) | `-StartTarget/-EndTarget` 控制 0..49 闭区间；`mean + flat-layerwise` | `comparison_layerwise_seed2026_v2/flat-layerwise/mean/800/<target>/`。最终毒样本与两个受害摘要完整才跳过；失败立即停止 |
| [`run_bp3pool_all.ps1`](../run_bp3pool_all.ps1) | `mean + pool-layerwise`，调用兼容攻击入口；目标闭区间与 `-Root` 路径参数 | `comparison_bp3pool_seed2026_v2/pool-layerwise/mean/800/<target>/`；完整结果跳过、失败停止 |
| [`run_cp_all.ps1`](../run_cp_all.ps1) | `convex + original`，目标闭区间 | `comparison_cp_seed2026_v2/original/convex/800/<target>/`；这是本地受控 CP 条件，不应仅凭该文件名宣称复现全部论文协议 |
| [`run_multiagent_5h.ps1`](../run_multiagent_5h.ps1) | **本项目内受控比较**。轮流运行 BP-Original（`original`）与 MSP pool/flat，全部调用本项目同一攻击与受害训练实现 | `comparison_multiagent_seed2026_v2/<feature-mode>/mean/800/<target>/`；275 分钟后停止新增任务，已运行的任务不被中止；仅检查最终文件及 `SUMMARY`，完成判定比其他三个脚本宽松 |

四个脚本从 `$PSScriptRoot` 推导项目父目录，保留 `Multi-scalePolytope` 的目录名约定；BP3Pool 另可通过 `-Root` 覆盖。输出名统一加 `_v2`，避免把新协议写入历史结果。迁移时核对实际目录名和权重模板。新实验优先使用项目 README 指定的 JSON 统一入口。

## `launch/`：单目标历史实验包装器

下面的 Bash 参数顺序必须按表填写；`$N` 不是有默认值的命名参数。`50-overlap`、`0-overlap`、`100-overlap` 是历史目录命名，真实预训练集合交集需要由数据和权重元信息核实。

| 脚本 | 输入及作用 | 输出/适用实验 |
| --- | --- | --- |
| [`launch/attack-transfer-18.sh`](../launch/attack-transfer-18.sh) | `$1 GPU, $2 mode, $3 target, $4 repeat, $5 poison_num`；6 架构 × 3 checkpoint，线性迁移 | `attack-results-10poisons/100-overlap/linear-transfer-learning/...`；实际毒样本数取 `$5`，目录中的 10poisons 不会验证该数值；800 次优化 |
| [`launch/attack-end2end-12.sh`](../launch/attack-end2end-12.sh) | `$1 GPU, $2 mode, $3 target, $4 repeat`；4 架构 × 3 checkpoint，end-to-end | `attack-results/100-overlap/end2end-training/...`；1500 次优化，受害学习率 1e-4、weight decay 5e-4 |
| [`launch/attack-transfer-difftraining-50.sh`](../launch/attack-transfer-difftraining-50.sh) | `$1 GPU, $2 mode, $3 target, $4 repeat`；代理模板 `3600to1200`，受害模板 `2400to1` | 历史 50-overlap 线性迁移目录；18 个代理、1500 次优化 |
| [`launch/attack-transfer-difftraining-0.sh`](../launch/attack-transfer-difftraining-0.sh) | 同上；代理模板改为 `4800to2400` | 历史 0-overlap 线性迁移目录；18 个代理、1500 次优化 |
| [`launch/attack-end2end-difftraining-50.sh`](../launch/attack-end2end-difftraining-50.sh) | 同样四个参数；代理模板 `3600to1200`；end-to-end | 历史 50-overlap end-to-end 目录；12 个代理、1500 次优化 |
| [`launch/attack-end2end-difftraining-0.sh`](../launch/attack-end2end-difftraining-0.sh) | 同样四个参数；代理模板 `4800to2400`；end-to-end | 历史 0-overlap end-to-end 目录；12 个代理、1500 次优化 |
| [`launch/eval-transfer.sh`](../launch/eval-transfer.sh) | `$1 GPU, $2 target_start, $3 poison_root`；调用独立分类头评估 | 默认评估该目标和八个受害网络、60 epochs；输入根目录必须已包含当前 feature-mode 层级 |
| [`launch/eval-end2end.sh`](../launch/eval-end2end.sh) | 同上，调用历史 end-to-end 重训练评估 | 八个受害网络、学习率 1e-4、weight decay 5e-4、60 epochs；用于旧日志/快照协议，新结果使用 `workflow.py --stage evaluate` |
| [`launch/eval-transfer-difftraining.sh`](../launch/eval-transfer-difftraining.sh) | 同样三个位置参数；受害 checkpoint 模板 `2400to1` | 50/0-overlap 配置对应的分类头重评；路径由调用者提供，不自动推断 overlap |
| [`launch/eval-end2end-difftraining.sh`](../launch/eval-end2end-difftraining.sh) | 同上并启用 end-to-end | 50/0-overlap 配置对应的完整受害微调 |
| [`launch/run_attack_all.py`](../launch/run_attack_all.py) | `GPU mode poison_num`；依次调用 transfer Bash，目标 0..49，repeat 固定 1 | 批量生成并执行攻击器内置评估；独立 eval 调用在源码中注释；子进程是否失败应由统一入口负责检查 |
| [`launch/end2end_run_attack_all.py`](../launch/end2end_run_attack_all.py) | `GPU mode [repeat]`，repeat 默认 1；保留历史目标 0..36，按 `target % 4 == GPU` 分配，GPU=0..3 | end-to-end 批量包装；显式将 repeat 传给 Bash，目标范围与 benchmark 的 0..49 不同 |
| [`launch/eval_attack_all.py`](../launch/eval_attack_all.py) | `GPU poison_root`，GPU=0..3；目标 0..49 按 `target % 4 == GPU` 分片 | 调用历史 transfer evaluator；只用于与旧日志格式匹配的已有结果，新批量复评使用统一入口 |
| [`launch/run_attack_fixedcoeffs.py`](../launch/run_attack_fixedcoeffs.py) | `GPU type`，type=1..10；`coeffs_fixed_type_N` 固定凸组合权重对照，repeat=1、poison_num=5，目标 0..49 按 GPU 分片 | 调用 transfer 包装器，当前攻击器 `all` 阶段负责评估；保存到 `attack-results-10poisons/100-overlap/linear-transfer-learning/original/coeffs_fixed_type_N/800/<target>/`；不再调用路径不匹配的二次旧评估 |

上述历史 Bash 脚本通常没有显式指定 `--feature-mode`，因此使用主攻击器当前默认的 `original`。要做 MSP 对照，应使用明确记录 feature mode 的入口，不能把历史结果目录重命名成 MSP。

旧 `eval_poisons_transfer.py` 的 `get_stats()` 会按旧攻击日志的固定位置解析迭代耗时等字段，新攻击器首先写入 JSON 协议记录，两种日志并不兼容。即使 shell 命令与路径正确，也不应使用历史 `eval*.sh` 复评新主流程结果；应给同一配置执行 `workflow.py --stage evaluate`。

## `analysis/`：单目标离线结果分析

这组脚本主要读取 `eval-retrained-for-<epochs>epochs.json` 或防御 Pickle；新格式的 `result.json/per_target.csv` 不会自动转换成它们要求的旧格式。多个方法的比较通常使用目标编号交集，因此运行前后应检查共同目标数，避免把不同样本量当作同一对照。

| 脚本 | 输入与用途 | 输出/限制 |
| --- | --- | --- |
| [`analysis/attack_performance_over_ites.py`](../analysis/attack_performance_over_ites.py) | `epochs poison_root [poison_root ...]`；按攻击迭代汇总目标预测、恶意概率、干净准确率、耗时和系数 | 每个结果根的 `plots-retrained-for-<epochs>epochs/` 内 PDF；离线分析，不训练受害模型 |
| [`analysis/compare_with_baseline.py`](../analysis/compare_with_baseline.py) | `epochs plots_root CP_root BP_root [BP3_root BP5_root]`；比较 CP/BP 和重复采样条件 | ASR、恶意概率、干净准确率、干净准确率变化和耗时 PDF；方法名按位置指定，不能任意交换根目录次序 |
| [`analysis/compare_diff_fixedcoeffs.py`](../analysis/compare_diff_fixedcoeffs.py) | 从代码中定义的十种固定系数方案路径读取旧评估 JSON | 固定系数/熵对照 PDF；默认路径和 epochs=60 硬编码，需要针对实际历史结果修改 |
| [`analysis/plot_polytopes.py`](../analysis/plot_polytopes.py) | 支持函数 `plot_diff_combinations(types, coeffs, methods, plots_path)`；绘制五个二维点与凸组合中心 | 编号 PDF 示意图；没有 CLI 主入口；随机打乱系数排列，不是从训练特征计算真实凸包 |
| [`analysis/draw_poison.py`](../analysis/draw_poison.py) | 反归一化毒样本，导出图像；主入口调用固定路径 `draw_all()` | `poisons-target-example.png`；辅助函数还可导出单张毒样本和基图像/目标。默认历史结果路径必须存在；不是图像质量数值评估 |
| [`analysis/defense_analysis.py`](../analysis/defense_analysis.py) | `--res-dir`、`--method`；读取多方法防御 Pickle 并汇总 | 控制台 LaTeX 表格；比较 ASR、毒样本检出与误删等防御表现；包含历史 pandas API |
| [`analysis/plot_defense_performance.py`](../analysis/plot_defense_performance.py) | `--res-dir`、`--num-poisons`、`--plots-dir`；读取 5/10/25 poison 等防御结果 | kNN/L2 的攻击成功率和毒样本检测率 PDF；迭代假设需与结果吻合 |
| [`analysis/finetuning_analysis.py`](../analysis/finetuning_analysis.py) | `poison_root method`；收集不同 epoch、Adam/SGD 和 dropout 的旧评估 JSON | 计算汇总 DataFrame 后进入 `IPython.embed()`，不是自动落盘报表；要求预先完成该脚本列举的全部受害训练配置 |
| [`analysis/compare-attacks-100-overlap-linear-transfer-learning.sh`](../analysis/compare-attacks-100-overlap-linear-transfer-learning.sh) | 固定 60 epochs，读取 4000 次攻击的 CP/BP/BP3/BP5 | 100-overlap 线性迁移比较 PDF；固定旧路径 |
| [`analysis/compare-attacks-50-overlap-linear-transfer-learning.sh`](../analysis/compare-attacks-50-overlap-linear-transfer-learning.sh) | 固定 60 epochs，读取 1500 次攻击的四种条件 | 50-overlap 线性迁移比较 PDF |
| [`analysis/compare-attacks-0-overlap-linear-transfer-learning.sh`](../analysis/compare-attacks-0-overlap-linear-transfer-learning.sh) | 固定 60 epochs，读取 1500 次攻击的四种条件 | 0-overlap 线性迁移比较 PDF |
| [`analysis/compare-attacks-100-overlap-end2end-training.sh`](../analysis/compare-attacks-100-overlap-end2end-training.sh) | 固定 60 epochs，读取 1500 次攻击的 CP/BP/BP3 | 100-overlap end-to-end 比较 PDF |
| [`analysis/analysis-attack-100-overlap-linear-transfer-learning.sh`](../analysis/analysis-attack-100-overlap-linear-transfer-learning.sh) | 历史逐迭代分析与论文图拷贝包装器 | 源码含 `eval/` 旧目录和作者本地论文路径；迁移后应直接调用上表 Python 分析器并提供实际路径 |
| [`analysis/analysis-attack-100-overlap-end2end-training.sh`](../analysis/analysis-attack-100-overlap-end2end-training.sh) | 调用逐迭代分析器，固定 1500 次攻击与 60 epochs | 各条件的逐迭代统计 PDF |
| [`analysis/utils.py`](../analysis/utils.py) | **支持模块**。递归读取旧评估 JSON，提供绘图颜色、线型和基图像读取函数 | 不单独运行；其中 `utils` 是分析目录模块，区别于根目录训练工具 |
| [`analysis/__init__.py`](../analysis/__init__.py) | **包标记** | 无独立输入输出 |

## `Multi-Target-Mode/`：外部车辆多目标历史协议

该目录中的 “multi-target” 指同一车辆的多视角图片等外部目标，默认读取 EPFL Multi-View Car 路径，目标标签 1、毒样本标签 6；它与根目录中把多个 CIFAR-10 目标压成质心的可用性实验不同。此目录依赖父目录的模型、数据加载器和优化器，必须保留相对目录关系。它是独立历史实现，根目录的 MSP 改动不会自动作用于这里。

| 脚本 | 输入与职责 | 输出/注意事项 |
| --- | --- | --- |
| [`Multi-Target-Mode/craft_poisons_transfer.py`](../Multi-Target-Mode/craft_poisons_transfer.py) | 多目标攻击主入口；外部目标路径/车辆编号/目标数量、CIFAR-10 基图像、代理权重，支持 mean/convex 与 end-to-end | 按车辆编号和 `target-num-<N>` 组织毒样本、日志和受害摘要；受害模型参数名是 `--victim-net` |
| [`Multi-Target-Mode/eval_poisons_transfer.py`](../Multi-Target-Mode/eval_poisons_transfer.py) | 读取一组车辆目标对应的毒样本，重训受害网络，区分用于生成的视角与额外验证视角 | `<poison_root>/<subset>/target-num-<N>/eval-retrained-for-<epochs>epochs.json`；包含攻击目标与额外目标表现；会训练模型 |
| [`Multi-Target-Mode/defense.py`](../Multi-Target-Mode/defense.py) | 外部多目标防御：无防御、kNN、类内 L2 过滤后重训 | 防御 Pickle；目标路径与 `target-num` 必须匹配原攻击 |
| [`Multi-Target-Mode/defense.sh`](../Multi-Target-Mode/defense.sh) | `$1 GPU, $2 method, $3 targetnum`；固定 1000 次线性迁移结果根 | 调用多目标防御研究；八个受害网络、标签 1→6 |
| [`Multi-Target-Mode/launch/attack-transfer-18.sh`](../Multi-Target-Mode/launch/attack-transfer-18.sh) | `$1 GPU, $2 mode, $3 car_index, $4 repeat, $5 target_num`；18 个代理 | 多目标线性迁移攻击；目标数量常比较 1/5/10 |
| [`Multi-Target-Mode/launch/attack-end2end-12.sh`](../Multi-Target-Mode/launch/attack-end2end-12.sh) | 相同五个参数；12 个代理、end-to-end、lr=1e-4、wd=5e-4 | 多目标 end-to-end 攻击，内置评估 DPN92 |
| [`Multi-Target-Mode/launch/eval-transfer.sh`](../Multi-Target-Mode/launch/eval-transfer.sh) | `$1 GPU, $2 target_subset, $3 poison_root, $4 target_num` | 八个受害网络的多目标分类头重评 |
| [`Multi-Target-Mode/launch/eval-end2end.sh`](../Multi-Target-Mode/launch/eval-end2end.sh) | 同上；end-to-end、1000 次攻击快照设置 | 八个受害网络的完整微调重评 |
| [`Multi-Target-Mode/launch/run_attack_all.py`](../Multi-Target-Mode/launch/run_attack_all.py) | `GPU mode repeat target_num`；固定车辆 ID 列表，当前代码只选列表中 `index % 4 == 3` | 名称虽为 run_attack_all，实际调用的是 end-to-end Bash，不能据文件名判断训练方式 |
| [`Multi-Target-Mode/launch/end2end_run_attack_all.py`](../Multi-Target-Mode/launch/end2end_run_attack_all.py) | `GPU mode repeat target_num poison_root`；固定车辆列表并按 GPU 截取一个 ID | 对所选车辆先攻击再独立 end-to-end 重评 |
| [`Multi-Target-Mode/launch/eval_attack_all.py`](../Multi-Target-Mode/launch/eval_attack_all.py) | `GPU mode poison_root`；按步长 4 查找历史快照 | 历史失配入口：调用当前仓库不存在的 `eval-attack-transfer-18-nodp.sh`；不能作为新实验入口 |
| [`Multi-Target-Mode/analysis/compare_with_baseline.py`](../Multi-Target-Mode/analysis/compare_with_baseline.py) | `--methods/--paths/--target-num/--net-repeats/--end2end/--epochs` | 多方法在已攻击目标、额外验证目标、干净准确率和耗时上的对照 PDF |
| [`Multi-Target-Mode/analysis/diff_target_num.py`](../Multi-Target-Mode/analysis/diff_target_num.py) | `--method/--path/--target-nums/--end2end/--epochs` | 单个方法在 1/5/10 个已知视角下的性能比较 PDF |
| [`Multi-Target-Mode/analysis/attack_performance_over_ites.py`](../Multi-Target-Mode/analysis/attack_performance_over_ites.py) | 历史位置参数 `convex_root mean_root epochs`；按迭代画两种攻击结果 | 目标/额外验证目标 ASR、干净准确率和时间 PDF；当前调用 `read_attack_stats` 时未给必需 `target_num`，应改用带显式参数的比较分析器 |
| [`Multi-Target-Mode/analysis/draw_poison.py`](../Multi-Target-Mode/analysis/draw_poison.py) | 毒样本反归一化和车辆目标图导出函数；默认主入口含固定路径 | PNG 拼图及 `target_png/`；需按本地 EPFL 和结果路径配置 |
| [`Multi-Target-Mode/analysis/compare-attacks-100-overlap-linear-transfer-learning.sh`](../Multi-Target-Mode/analysis/compare-attacks-100-overlap-linear-transfer-learning.sh) | 依次分析 targetnum=1/5/10，mean/mean-3/mean-5/convex | 固定 2000 次攻击目录下的线性迁移多方法 PDF |
| [`Multi-Target-Mode/analysis/compare-attacks-100-overlap-end2end-training.sh`](../Multi-Target-Mode/analysis/compare-attacks-100-overlap-end2end-training.sh) | targetnum=1/5/10；mean、mean-3 和 convex | BP 路径 2000、CP 路径 1000 的历史 end-to-end 比较 PDF |
| [`Multi-Target-Mode/analysis/diff-targetnum-transfer.sh`](../Multi-Target-Mode/analysis/diff-targetnum-transfer.sh) | 遍历 mean/mean-3Repeat/mean-5Repeat/convex，比较 1/5/10 个目标 | 历史调用路径为不存在的 `eval/diff_target_num.py`；需改成实际 `analysis/` 路径后使用 |
| [`Multi-Target-Mode/analysis/diff-targetnum-end2end.sh`](../Multi-Target-Mode/analysis/diff-targetnum-end2end.sh) | 遍历 mean/mean-3Repeat/convex，比较 1/5 个目标 | 同上，有 `eval/` 旧路径；输入根固定为 `chk-black-end2end` |
| [`Multi-Target-Mode/trainer.py`](../Multi-Target-Mode/trainer.py) | **支持模块**。多目标 CP/BP 损失、像素优化和受害训练 | 由多目标攻击入口导入；没有根目录 `feature-mode` 选择流程 |
| [`Multi-Target-Mode/utils.py`](../Multi-Target-Mode/utils.py) | **支持模块**。父目录模型加载、外部目标裁剪与采样、目标特征和基图像选取 | 由多目标训练/评估导入；不是独立测试 |
| [`Multi-Target-Mode/analysis/utils.py`](../Multi-Target-Mode/analysis/utils.py) | **支持模块**。按 target-num 递归聚合旧评估 JSON，提供绘图样式和基图像 | 无 CLI；若 2000 路径未发现结果，会尝试替换为 1000 路径，阅读结果时需核对实际读取路径 |
| [`Multi-Target-Mode/analysis/__init__.py`](../Multi-Target-Mode/analysis/__init__.py) | **包标记** | 无独立输入输出 |

## 支持模块与模型文件

| 文件 | 职责 | 输入/输出与测试边界 |
| --- | --- | --- |
| [`trainer.py`](../trainer.py) | 单目标 CP/BP 目标函数、MSP 特征路由、投影像素优化、受害训练、准确率统计 | 输入网络/目标/基图像/参数，返回毒样本或预测并按调用参数保存快照；不直接作为 CLI 运行 |
| [`experiment_protocol.py`](../experiment_protocol.py) | 不依赖训练框架的方法注册表、布尔参数解析、攻击组合约束、checkpoint 路径解析及协议来源校验 | 定义 CP/BP/MSP 与 E2E 方法；拒绝不支持的 MSP+end-to-end、逐层+convex 和不匹配固定系数组合；`ensure_protocol` 防止同目录混用数据/权重/参数 |
| [`trainer_bp3pool.py`](../trainer_bp3pool.py) | **兼容导出模块**。重新导出共享 `trainer.py` 的实现 | 保留旧导入名称；CP/BP/MSP 目标函数与受害训练均由同一个主 trainer 维护 |
| [`dataloader.py`](../dataloader.py) | `PoisonedDataset` 将毒样本及其基图像索引注入指定 split；`FeatureSet` 缓存特征；`SubsetOfList` 按类别索引选子集 | 输出 Dataset 对象。常见每类 50 的 poisoned dataset 是用毒样本替换相应基图像位置，不是简单在 500 张干净图上额外加 5 张 |
| [`utils.py`](../utils.py) | 预训练模型读取、目标与基图像读取、最近邻基图像搜索、毒样本元组构造、外部目标处理 | 供攻击器/评估器导入；模型加载与像素变换是其关键职责 |
| [`SignedAdam.py`](../SignedAdam.py) | 自定义 SignedAdam 优化器 | 梯度/参数输入，按符号更新；由 trainer 的优化器选择逻辑调用 |
| [`experiment_recorder.py`](../experiment_recorder.py) | 不依赖训练框架的 SHA-256 文件摘要与 JSON 原子落盘 | 给结果复评记录来源，避免写入一半的 JSON 被当作完整结果 |
| [`models/__init__.py`](../models/__init__.py) | 汇总导出下列七个模型模块 | 供训练脚本按架构名称构造模型；不会自动运行各模块的测试函数 |
| [`models/resnet_msp.py`](../models/resnet_msp.py) | ResNet18/34/50/101/152；MSP 层为 layer2/3/4 | 输出 logits、原 penultimate 特征或三层 pool/flat 特征；文件尾 `test()` 是未自动执行的形状示例 |
| [`models/senet_msp.py`](../models/senet_msp.py) | SENet18；MSP 层为 layer2/3/4 | 同上；`test()` 不等价于整个攻击验证 |
| [`models/dpn_msp.py`](../models/dpn_msp.py) | DPN26/DPN92；MSP 层为 layer2/3/4 | 同上；包含手动形状示例 |
| [`models/resnext_msp.py`](../models/resnext_msp.py) | ResNeXt29 系列；MSP 层为 layer1/2/3 | 同上；尾部 `test_resnext()` 是形状示例 |
| [`models/googlenet_msp.py`](../models/googlenet_msp.py) | GoogLeNet；MSP 层为 e4/a5/b5 | 保留原分类与末层特征接口；不支持原版 E2E 的 block=True |
| [`models/mobilenetv2_msp.py`](../models/mobilenetv2_msp.py) | MobileNetV2；MSP 使用最后三个 cfg stage（96/160/320 通道） | 三层特征不含末尾 1280 通道投影；含手动测试函数 |
| [`models/densenet_msp.py`](../models/densenet_msp.py) | DenseNet105/121 等；MSP 使用 dense2/3/4 原始输出 | dense4 在最终 BN/ReLU 之前；DenseNet121 已纳入 `verify_layerwise.py` 的真实架构特征/梯度检查 |

三层特征方法不增加可训练参数；旧 checkpoint 的可加载性仍须实际运行验证，不能由“没有新增参数”直接推断数据来源或预训练 overlap。

## 运行依赖与资源成本

### 远端已有、发布时保留的历史文件

目标 GitHub 仓库原有文件随本次发布保留；它们与上面本地重构的92个脚本并存。

| 文件 | 作用与适用范围 |
| --- | --- |
| `visualize_msp_features_bp_fixed_v3.py` | 历史特征几何可视化：加载同一目标、代理权重及 BP/MSP poison，计算质心/凸包几何误差，绘制热图、特征图和PCA等；不训练受害模型，不直接测量ASR。实际参数与示例见[原GitHub首页](LEGACY_GITHUB_README.md)及该脚本help。 |
| `models/{densenet,dpn,googlenet,mobilenetv2,resnet,resnext,senet}.py` | GitHub原有模型实现仍可按模块直接导入；新版统一入口通过models包导出相应`*_msp.py`实现。旧文件不参与新版训练流程的默认模型路由。 |
| `attack-results/feature-analysis*` | 远端原有图表和CSV保留作为历史资料，不是本次重构后的新实验成绩。 |

| 用途 | 依赖及外部文件 | 资源建议 |
| --- | --- | --- |
| 统一计划、CLI 帮助、JSON 配置、协议与记录工具 | Python 标准库；新 `workflow.py`、主攻击器和预训练器均可先解析帮助/配置再导入训练框架 | `plan/--dry-run/--help` 不应启动训练 |
| 标准库回归和新结果汇总 | `unittest` 等标准库；`test_workflow.py` 读取 configs，`test_data_contracts.py` 提取生产 Python 定义，汇总器读取新 evaluation JSON | 不要求 GPU 或真实数据；张量测试缺 PyTorch 时跳过 |
| `verify_layerwise.py` | PyTorch、NumPy、`models/` 和 `trainer.py` | CPU 可运行；真实架构的 flat 特征前向/反传仍会消耗内存 |
| 单目标预训练/生成/受害评估 | 相互兼容的 PyTorch 与 torchvision、NumPy、Pillow、OpenCV；历史独立评估器按导入链还需要 SciPy；本地 split、测试集与权重 | 新统一主流程支持 CPU/CUDA；正式研究建议 CUDA。历史兼容入口中仍有硬编码 CUDA，不能仅因声明 `--device cpu` 就推断可运行 |
| 可用性实验 | 上述训练依赖、精确 48000/2000 split、代理权重 | 源码明确要求 CUDA；48 次生成加各 seed/架构/条件的从零训练，属于耗时实验 |
| 旧分析/防御 | pandas、Matplotlib、SciPy；图像导出还需 torchvision；finetuning_analysis 还需 IPython | PDF 汇总通常可用 CPU；防御入口会训练受害模型且开销大。旧分析包含 `DataFrame.append` 等历史 API，应使用兼容环境或迁移相应代码 |
| Bash/PowerShell 包装器 | 对应 shell、能被调用的 `python`、工作目录匹配；Windows 使用 Bash 脚本需提供 Bash 环境 | 优先先跑一个目标的完整生成与评估，确认依赖和快照路径后再批量 |

正式结果至少报告：数据/权重哈希、实际代理数、feature mode、目标编号与类别、毒样本数和占比、像素空间 epsilon、生成/受害训练种子、训练方式和优化器、迭代/epochs、共同评估目标数、ASR、干净及各类准确率、耗时与显存。训练损失下降、脚本语法通过和 CPU 梯度断言均不能替代这些结果。
