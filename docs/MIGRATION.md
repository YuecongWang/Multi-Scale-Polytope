# 本次重构与验证记录

## 行为变化

| 原问题 | 当前行为 |
| --- | --- |
| 根README仍介绍原BP，中文README仅是一份补丁说明 | 根README介绍本项目、可运行配置和完整步骤；上游原文保存在UPSTREAM_README.md |
| BP需调用相邻旧项目，MSP比较入口分散 | 本项目内明确注册CP/BP/MSP，统一workflow及JSON配置 |
| 生成、受害训练和结果判断混在一次脚本里 | craft/evaluate/all可分别执行，保留主脚本默认all兼容 |
| train --resume 实际只测试并退出 | 恢复下一轮训练；测试单独使用--evaluate-only |
| 预训练输出名和攻击默认模板不一致 | 默认model-chks/ckpt-{net}-4800.t7，best/latest分开 |
| Windows多进程数据加载缺少main隔离 | 预训练可导入，main guard，Windows默认workers=0 |
| CPU参数仍调用CUDA | 主训练/生成/评估按device执行，CPU包装不自动选可见GPU |
| type=bool使字符串False变成True | 主CLI显式解析true/false，支持单独--end2end |
| 传入的衰减轮次是字符串 | 整数参数，空列表可以关闭衰减 |
| epsilon只能为0.1 | 支持0..1像素预算；重新评估检查实际L∞和像素范围 |
| 全量数据模式保留原base又加入poison | base被替换；校验唯一索引、标签、预算、分组和数据足量 |
| poison前置使clean与poisoned同seed仍对应不同batch图像 | 主流程原位替换；历史数据集默认前置顺序保留兼容 |
| 多层/E2E组合可忽略MSP特征或运行中才崩溃 | 拒绝MSP+E2E，检查原版E2E模型block能力 |
| CP原版E2E忽略net-repeat | 与BP一样先对各block的重复前向取平均 |
| 新权重含RNG/PIL split无法按现代默认weights-only加载 | 共用加载器对可信本地完整文件显式使用weights_only=False |
| 绝对权重路径、模型字符串eval、DP前缀处理存在问题 | Path拼接、模型白名单、普通/DP权重兼容 |
| 重复trainer副本继续维护旧缺陷 | BP3Pool兼容入口与trainer转发到共用实现 |
| 干净评估脚本引用不存在的参数/方法 | 重写为独立权重准确率检查和JSON输出 |
| 旧PowerShell依赖其他用户的绝对路径 | 使用PSScriptRoot，新增结果目录带_v2，BP也由本项目运行 |
| 若干launch参数缺失或目标范围/分卡错误 | 修复E2E空格与repeat、评估分卡、固定系数poison数及0..49范围 |
| 历史日志和一次预测被当作完整证据 | 新协议、生成成本、配对条件、完整性校验和结构化汇总 |

这些修复会改变部分历史运行的训练顺序、失败检查和记录方式，因此新结果使用独立目录。已有结果目录、权重和数据未被迁移或重新计算；不能将旧结果当作新版实验。

## 新旧格式边界

新实验优先使用workflow、主craft和summarize_results。旧eval_poisons_transfer、analysis等工具读取旧log/旧JSON格式，对新日志不保证兼容。旧PowerShell只要发现最终文件与完整摘要仍会跳过；新统一入口以protocol配置和哈希核对为准。

导入历史poison时，用主craft的--eval-poison-path并指定新的--chk-path，完整填写真实的目标、代理、预算与训练参数。路径中的方法名不能证明文件确由该算法生成；如果旧文件无方法元数据，仍需保留其外部来源记录。

每个新输出目录锁定整个实验参数和输入哈希。不能在原目录直接换受害架构、seed或epsilon；复用相同poison做新的评估协议时，需建立新的输出目录，并明确导入poison。攻击warm start并不恢复完整优化状态。

## 2026-10-25本次验证

初次重构验证环境为Windows PowerShell，Python 3.13.9。可用NumPy、Pillow和unittest；没有torch、torchvision、OpenCV，也没有datasets或model-chks。本地工作目录当时没有Git元数据；后续GitHub发布提交见仓库历史。

已执行：

~~~powershell
python -m unittest discover -s tests -v
python workflow.py --config configs/smoke.json --stage plan
python -m compileall -q craft_poisons_transfer.py trainer.py experiment_protocol.py workflow.py prepare_data.py summarize_results.py train_cifar10_models.py utils.py dataloader.py tests
~~~

最终本地一轮共35项：33通过，2项真实PyTorch测试因缺少依赖显式跳过。数据/加载测试是执行生产选择逻辑的AST/mock契约测试，不能替代真实模型张量测试。另新增GitHub Actions标准库/CPU张量检查配置；本机未实际运行远程CI。

此外检查了命令行help、预训练/攻击参数对齐、PowerShell语法、历史包装器目标范围与参数传递、文档文件链接。未执行真实模型前向/梯度、CUDA、数据下载、干净预训练、毒样本优化、受害训练或成功率实验。

## 在完整环境验收

1. 安装匹配的torch/torchvision后，重新跑unittest，真实张量测试应执行而非skip。
2. 跑verify_layerwise.py，确认七种架构的flat/pool特征与梯度。
3. 准备真实split和配置对应权重，检查权重在官方test上的准确率。
4. 按README跑smoke的craft/evaluate/summarize，确认实际像素预算、500张替换预算、clean/poison配对与输出完整。
5. 将正式配置缩为目标0另建pilot输出；完成一次800迭代/60轮，再扩大目标和种子。

如果BP和MSP成绩仍接近，先分析layerwise/concat、pool/flat、目标基线和多种子统计。这一重构解决代码路径、实验控制与说明混淆，未预设算法性能差异。
