# 重构、兼容与验证记录

## 本轮支持范围

主 README 已改为英文 GitHub 项目说明，[README_zh.md](../README_zh.md) 保留中文导航。新版统一流程同时支持 CIFAR-10 的 linear transfer 与 MSP 全参数受害微调；两个场景使用不同配置和输出根目录。

攻击表征、受害参数更新范围与受害模块模式独立配置：

| 设置 | 选项 | 作用 |
| --- | --- | --- |
| `attack.representation` / `--attack-representation` | features / blocks | 原末层/MSP 特征目标，或原 BP/CP block 攻击 |
| `victim.training` / `--victim-training` | linear / end2end | 分类头训练，或全部预训练参数微调 |
| `victim.model_mode` / `--victim-model-mode` | eval / train | 优化时 BN/Dropout 行为；linear 只支持 eval |

`MSP-Flat` 等方法名描述攻击，受害训练场景单独记录。MSP 用 features，支持 linear 和全参数两场景，不需要 block 接口。block 能力仅限制原 blocks 攻击的代理；诊断受害使用末层特征，七个模型家族均可作为受害。

全参数+eval 仍更新骨干和分类头，固定 BN 运行统计；全参数+train 更新训练模式的 BN 状态，属于另一个需单独报告的协议。测试指标始终使用 eval。毒样本生成固定代理、优化像素，不包含可微展开的受害训练，也不是从零训练受害。

## 行为变化

| 原问题 | 当前行为 |
| --- | --- |
| 主说明与实际 MSP 扩展、脚本不一致 | 英文 README 给出两套可执行流程，中文参数/目录/迁移文档同步；上游与原 GitHub 首页保留 |
| BP 依赖相邻旧项目、比较入口分散 | 本项目注册 CP/BP/MSP，统一 workflow、JSON 配置和受害训练 |
| 一个 end2end 开关同时控制攻击 block 特征和受害全参数更新 | 三个独立设置；MSP features 可以搭配全参数受害，不再因 full 场景丢失多层特征 |
| 原 E2E 限制误用于诊断受害 | blocks 只检查代理；MSP 与原 block 基线均可评估不提供 block 的受害模型 |
| 全参数训练与 eval/train 模块模式混淆 | 单独记录更新范围、模块模式、可训练参数数及骨干参数变化探针 |
| 生成和受害训练混在一次调用里 | 主 craft 有 craft/evaluate/all；workflow 有 plan/pretrain/craft/evaluate/all |
| train --resume 实际只测试 | 恢复下一轮训练；测试使用 --evaluate-only |
| 预训练文件名与攻击模板不一致 | 默认 model-chks/ckpt-{net}-4800.t7，best/latest 分开 |
| Windows 数据加载缺少 main 隔离 | 预训练可导入，有 main guard；Windows 默认 workers=0 |
| CPU 参数仍调用 CUDA | 主训练/生成/评估按 device 执行，CPU 包装不自动选 GPU |
| type=bool 把字符串 False 当 True | 兼容开关显式解析 true/false；与新显式参数冲突时报错 |
| 衰减轮次为字符串、空表无法传递 | 整数参数；空列表关闭衰减 |
| epsilon 只能为 0.1 | 支持 0..1 像素预算，重新评估核查实际 L∞ 和像素范围 |
| 全量数据模式保留 base 又追加 poison | 原位替换，并检查标签、唯一索引、预算、分组与样本数 |
| poison 前置导致同 seed 的 clean/poisoned batch 不对应 | 主流程保留原样本位置；历史数据集默认前置顺序保持兼容 |
| CP block 攻击忽略 net-repeat | 各 block 的重复前向先平均 |
| 可信完整 checkpoint/split 无法按现代 weights-only 默认加载 | 共用加载器显式读取可信本地完整文件 |
| 绝对路径、模型字符串 eval、DP 前缀有问题 | Path 路径、模型白名单、普通/DP 权重兼容 |
| 重复 trainer 继续分叉 | BP3Pool 入口与 trainer 转发共享实现 |
| clean-test 引用不存在的参数/方法 | 独立权重测试 CLI，CPU/CUDA 与 JSON 输出 |
| PowerShell 依赖其他用户路径 | PSScriptRoot 推导路径，新结果带 _v2，BP 也由本项目执行 |
| launch 参数/分卡/目标范围错误 | 修复 E2E 空格与 repeat、评估分卡、固定系数 poison 数及范围 |
| 只看日志或一次目标预测就判断实验成功 | 配对 clean/poisoned、结构化指标、数据/模型哈希、完整性及训练模式检查 |

这些变化会影响部分历史运行的顺序、兼容判断和输出格式。新结果必须使用独立目录；保留旧产物不代表已经重新计算。

## 旧 end2end 选项和快照语义

新实验优先写三项显式设置。兼容 `--end2end true` 时，未显式覆盖攻击表征的默认映射为：

| feature-mode | 攻击表征 | 受害更新 |
| --- | --- | --- |
| original | blocks | 全参数 |
| pool / flat / flat-layerwise / pool-layerwise | features | 全参数 |

旧 `--end2end false` 对应 linear 受害；显式 `--victim-training` 与旧开关矛盾时报错。`linear + model_mode=train` 被拒绝；MSP+原 blocks 仍被拒绝，因为 MSP 使用自己的 features 目标。这与允许 MSP+全参数受害是两个不同判断。

历史 poison 快照中的 `end2end` 表示**生成时是否使用 block 攻击**，不能据此推断受害训练范围。新版 CLI 解析后的 `args.end2end` 为受害全参数更新兼容字段，另用 `attack_end2end` 控制原 block 攻击；新产物用明确的 `attack_representation`、`victim_training` 和 `victim_model_mode` 记录语义。请结合协议读取历史快照，不要把这几个字段混为一谈。

## 新旧配置与结果边界

JSON 配置的 `schema_version` 仍为 1；新运行协议的 `schema_version` 为 2。二者是不同文件格式。原协议目录不能直接写入新协议，必须选择新 `output_root/--chk-path`，或显式导入 poison。

推荐配置：

- `configs/smoke.json` / `configs/benchmark.json`：features、linear、eval，受害 Adam lr=0.1。
- `configs/smoke_end2end.json` / `configs/benchmark_end2end.json`：features、end2end、eval，受害 Adam lr=1e-4、weight decay=0，正式 60 epochs，smoke 2 epochs。
- 原 blocks 基线：另建配置，选择 BP-E2E-Original/CP-E2E-Original 和 block 代理，并显式填写受害训练参数。历史 shell 的 weight decay=5e-4 与新版 full 预设不同。

clean 预训练权重可在两个场景间共享。协议目录锁定完整参数与输入哈希；不能在同目录换受害训练方式、模块模式、架构、种子或 epsilon。复用 poison 做新评估时，使用主 craft 的 `--eval-poison-path` 与新的 `--chk-path`，完整填写真实生成来源、表征、标签、预算和受害参数。

新汇总器记录并校验受害训练方式，不能用 linear 结果满足 full 配置的完整性要求，也不能把同方法/受害/种子下不同更新方式的结果混为一组。骨干变化字段是一个参数样本的诊断，不证明每个骨干参数都发生变化。

旧 `eval_poisons_transfer.py`、`analysis/` 等读取旧 log/JSON，不能直接读取新协议日志。旧 PowerShell 的跳过规则仍基于最终文件和摘要；新统一入口核对协议和哈希。攻击 warm start 恢复像素并重启优化器，不是完整优化状态的严格续跑。

workflow 的 all 仅串行执行预训练、生成、评估；`prepare_data.py` 与 `summarize_results.py` 必须分别调用。

## 历史结果和当前可复跑流程

旧目录中保留了 E2E 结果，例如 `cifar10_e2e_bpprotocol_v1`、`cifar100_resnet18_six_end2end_v1`。它们是以前运行的产物证据；实际训练范围、初始化、数据与超参数需结合其日志和元信息核实。不能把这些数字直接标成当前 features+全参数流程的成绩。

当前统一流程以 CIFAR-10 单目标实验为主；历史 CIFAR-100 文件不构成新版严格复现。上游说明、原 GitHub 可视化与远程图片链接保留在 [UPSTREAM_README.md](UPSTREAM_README.md) 和 [LEGACY_GITHUB_README.md](LEGACY_GITHUB_README.md)。特征图或 PCA 展示不能代替受害训练后的 ASR。

## 验证记录

先前发布版本的 GitHub CI 已通过其对应提交的标准库/CPU 张量检查与七架构特征验证。这是历史 CI 记录，不是本轮新增 full 训练路径的验证结论。

本轮本地 Windows 环境没有 torch/torchvision、真实 split 或 checkpoint。标准库配置、数据契约和统计测试可运行；真实攻击/受害张量测试因缺少 torch 显式跳过。当前提交的具体通过/跳过数以该次测试输出为准，避免把旧计数沿用到扩展后的套件。本轮远程 CPU CI 结果需要与实际提交对应查看。

~~~powershell
python -m unittest discover -s tests -v
python workflow.py --config configs/smoke.json --stage plan
python workflow.py --config configs/smoke_end2end.json --stage plan
python workflow.py --config configs/benchmark.json --stage plan
python workflow.py --config configs/benchmark_end2end.json --stage plan
~~~

测试职责：

- `test_workflow.py` / `test_data_contracts.py` / `test_summary_training.py`：标准库契约，校验独立路由、数据替换、加载和训练场景统计。
- `test_attack_tensors.py`：真实 CPU 攻击梯度、投影、特征/block 接口。
- `test_victim_training.py`：真实优化器和 BN，检查分类头与全参数更新，并包含 MSP 生成→全参数受害训练的合成数据集成检查。
- `verify_layerwise.py`：七种真实架构的 flat/pool 特征一致性与输入梯度。

AST/mock 只验证相应生产 Python 逻辑；小型真实张量测试也不等于真实 checkpoint 或正式攻击通过。本轮没有执行真实数据下载、干净预训练、CUDA、正式投毒/受害训练或性能实验。

## 在完整环境验收

1. 安装兼容 torch/torchvision，运行全部 unittest；确认张量测试实际执行。
2. 执行 `verify_layerwise.py`，检查七架构特征与梯度。
3. 准备有来源的 split 和对应权重，检查官方 test 上的权重准确率。
4. 分别执行 linear/full 的 smoke craft、evaluate、summarize；检查 500 张原位替换预算、扰动、配对 RNG、训练模式和骨干变化。
5. 复制对应 benchmark 为目标 0 的新 pilot，完成 800 次攻击/60 轮受害训练；测定资源成本后扩大目标和种子。
6. 两种受害场景分别汇总、报告。train 模块模式、原 blocks 基线或其他优化器作为单独协议。

若 BP 与 MSP 成绩接近，分析 layerwise/concat、pool/flat、clean 目标基线、收敛和多种子统计。重构解决实现路径与实验控制问题，不预设算法性能提升。