# 参数与实验协议

Linear 参数见 [smoke.json](../configs/smoke.json)、[benchmark.json](../configs/benchmark.json)；MSP 全参数受害微调见 [smoke_end2end.json](../configs/smoke_end2end.json)、[benchmark_end2end.json](../configs/benchmark_end2end.json)。它们是可执行的实验起点，不代表已在当前机器调优或取得某种成功率。

## 为什么训练过程相似

本项目的分类网络仍是原有 ResNet、DPN 等；新增方法读取多层中间张量，不增加可训练参数。干净预训练负责获得特征提取器；攻击阶段固定模型、更新图像；受害训练负责检验图像是否改变分类结果。方法差异发生在第二步，因此第一步、第三步保持一致才便于归因。

若 BP 与 MSP 结果相似，应检查特征是否真的切换、浅层是否被拼接能量掩盖、目标是否本来就被干净受害误分类，以及攻击是否收敛。不能把“不同训练命令”当作新算法，也不能预先保证 MSP 一定更好。

## 攻击表征与受害训练分开配置

| 选择 | JSON | 主攻击器 CLI | 实际控制 |
| --- | --- | --- | --- |
| 攻击表征 | `attack.representation` | `--attack-representation features/blocks` | `features` 使用 feature-mode 选择的末层/MSP目标；`blocks` 使用原 BP/CP block 目标 |
| 受害更新范围 | `victim.training` | `--victim-training linear/end2end` | 只更新分类头，或更新受害模型全部预训练参数 |
| 受害模块模式 | `victim.model_mode` | `--victim-model-mode eval/train` | 优化时 BN/Dropout 行为；linear 缓存特征，只允许 eval |

MSP 的 pool、flat、flat-layerwise、pool-layerwise 都使用 `features`，可以搭配 linear 或 end2end 受害。方法名如 `MSP-Flat` 不随受害场景改名，场景单独记录。`eval` 不关闭 autograd；在 end2end+eval 下，骨干与分类头仍被优化，BN 运行统计固定。end2end+train 会让 BN/Dropout 使用训练模式，属于应单独记录的新协议；选择 train 不会自动加入图像增强。

原 `BP-E2E-Original/CP-E2E-Original` 名称保留为 block 攻击的兼容名，缺省使用 blocks 和全参数受害；新配置应显式写出上述三项。block 接口只要求**代理**具备（ResNet、ResNeXt、DPN、SENet）；诊断受害始终使用原末层特征。MSP 的全参数受害训练不需要 block 接口，七个模型家族都可作为受害，包括 DenseNet、GoogLeNet 和 MobileNetV2。

这里的 end2end 是从预训练 checkpoint 微调所有受害参数。毒样本生成仍冻结代理、仅更新像素；没有可微展开受害优化过程，也不表示随机初始化后从零训练。

旧 `--end2end true` 保留兼容：未显式覆盖表征时，original 特征映射为 blocks+全参数受害，MSP 特征映射为 features+全参数受害。若旧选项与显式 victim-training 冲突会报错。新实验优先使用三个独立选项。

## 特征与损失

设第 i 个代理的第 l 个 stage 特征为 f_il，目标为 x_t，K 个基图像的毒样本为 x_pj。目标特征只采样一次后 detach；毒样本可重复前向 R 次，先平均每个特征位置，再求毒样本质心。R 是 net_repeat，与代理个数不同。

- **BP-Original**：用原末层特征，最小化 `||mean_j f_i(x_pj) - f_i(x_t)||₂ / ||f_i(x_t)||₂`，再对代理取平均。
- **CP-Original**：每轮为每个代理求系数 a_j≥0、sum(a_j)=1，再最小化 `0.5 × ||sum_j a_j f_i(x_pj) - f_i(x_t)||₂² / ||f_i(x_t)||₂²`。
- **MSP-Pool**：对每层做全局平均池化，沿特征维拼接，再使用 BP 质心目标。
- **MSP-Flat**：每层展平后拼接，再使用 BP 质心目标。
- **MSP-Flat-Layerwise**：展平相同三层，但先计算每层的归一化范数比，最后按层、代理等权平均。
- **BP-3Layer-Pool-Control**：与上一项同样逐层等权，输入为每层池化特征。

拼接模式隐含各层能量权重。若 `d_l=||centroid_l-target_l||₂/||target_l||₂`，那么拼接损失为 `sqrt(sum_l w_l*d_l²)`，其中 `w_l=||target_l||₂²/sum_h||target_h||₂²`。它与逐层 `mean_l d_l` 不相同。Flat 保留空间维，特征量通常更大，也可能让高能量层主导优化。

逐层模式遇到零目标范数会报错；共用攻击器拒绝非有限 loss。模型选层见 [models/README.md](../models/README.md)。受害网络仍使用原分类特征；日志里的 victim CP loss 是诊断，不是 MSP 的代理优化目标，也不参与 poison 梯度更新。

## JSON 与命令行对应

### 运行与路径

| JSON 字段 | CLI / 含义 |
| --- | --- |
| schema_version | 配置固定为 1；新运行 protocol 为 2，是另一文件格式；未知配置字段报错 |
| name | 计划显示名称，不参与算法 |
| dataset.split | 预训练 `--data-path`；攻击 `--train-data-path`，包含 PIL 图片的可信本地 split |
| dataset.root | 两个训练入口的 `--dset-path`，官方 CIFAR-10 缓存目录 |
| model_directory | 攻击 `--model-resume-path`；预训练输出目录 |
| output_root | 每个方法再分独立输出目录；修改协议时使用新根目录 |
| device / gpu | `--device cpu/cuda`、`--gpu`；gpu 设置可见物理卡，主入口使用其逻辑设备 |
| seed | 毒样本生成和受害配对训练种子；预训练另有 pretrain.seed |

workflow 的相对路径一律相对仓库根目录解析；直接调用单脚本时相对路径相对当前工作目录。模板统一使用 `%s` 代表模型名，例如 `ckpt-%s-4800.t7`；固定文件名只适合一个模型，不能让不同模型写入同一个文件。单独预训练入口额外支持 `{net}` 等格式。

### 干净预训练 pretrain

| 字段 | 默认 / 正式起点 | CLI 与规划 |
| --- | --- | --- |
| epochs | 300 | `--epochs` 总轮数；恢复时包含已训练轮数 |
| lr | 0.1 | `--lr`；SGD，momentum=0.9、weight_decay=5e-4 |
| batch_size | 128 | `--batchsize`；减少它可降低显存，但会改变训练协议 |
| seed | 1226 | `--seed`；同一模板下不要混入不同随机初始化的权重 |
| sidx / eidx | 0 / 4800 | 逐类 `[sidx,eidx)`，来自 clean_train，不是 split 全局索引 |
| workers | 0 | `--workers`，Windows 先使用 0 |
| milestones | [150,225] | `--milestones`；gamma 默认 0.1；空列表表示不衰减 |

workflow 根据代理模板/架构/dropout 的组合和受害列表规划预训练任务，同一输出权重去重。代理的 train_dp 取 attack.dropouts 对应值；受害 train_dp=0。若某个模板来自不同的已训练来源，应直接保留其文件并执行 craft，不应假定重新训练就能逐位复现上游。

直接训练器的额外参数：`--train-dp`、`--droplayer`、`--output-dir`、`--checkpoint-name`、`--gamma`、`--no-download`、`--resume`、`--evaluate-only`。resume 恢复 latest 的优化器、调度器与 RNG；旧 checkpoint 若缺状态会明确提示，不能宣称严格续训。改变模型、数据范围或调度时另建实验。

### 毒样本生成 attack

| 字段 | 正式起点 | CLI 与规划 |
| --- | --- | --- |
| methods | 六种受控对照 | 映射 `--mode`、`--feature-mode` 及默认表征；受害训练由 victim.training 独立指定 |
| representation | features | `--attack-representation`；MSP 使用 features；blocks 只适用于 original 特征及 block 代理 |
| targets | 0..49 | 每个值变为 `--target-index`，为目标类内部的相对索引 |
| substitute_nets | 六个架构 | `--substitute-nets`；与 checkpoint 列表做笛卡尔积 |
| checkpoints | 三个模板 | `--subs-chk-name`；每个模板都应有每个代理架构的实际文件 |
| dropouts | [0.2,0.25,0.3] | `--subs-dp`；长度必须等于模板数，不是等于架构数 |
| net_repeat | 1 | `--net-repeat`；可消融 1/3/5，增加重复会增加成本 |
| target_label / poison_label | 6 / 8 | 不同的 CIFAR-10 类；毒样本保持原基图像标签 |
| target_start | 50 | `--target-start`；类内实际位置=此值+target_index，必须在受害训练窗外 |
| subset_group | 0 | `--subset-group`；训练与基图像窗口起点=group×num_per_class |
| poison_num | 5 | `--poison-num`；在总预算中替换，不追加；不超过每类训练数 |
| iterations | 800 | `--poison-ites`；匹配本地对照。若改为1500/4000，全部方法使用相同新预算并另建目录 |
| lr | 0.04 | `--poison-lr`；作用在归一化图像坐标上 |
| epsilon | 0.1 | `--poison-epsilon`；约25.5/255，约束原始 `[0,1]` 像素的 L∞ 偏移 |
| optimizer | adam | `--poison-opt`；也支持 sgd / signedadam；统一比较时固定 |
| decay_iterations | [] | `--poison-decay-ites`；0基迭代索引；默认不衰减 |
| decay_ratio | 0.1 | `--poison-decay-ratio`；在指定迭代乘学习率 |

epsilon 是原始像素单位，不是标准化张量单位。投影先约束原始像素偏移，再裁剪到 `[0,1]`；新评估器会检查保存的图像是否真的满足预算和标签约束。

直接入口还支持 `--nearest`（在训练类窗口选邻近 bases）、`--tol`（CP 内循环容差）、`--resume-poison-ite`、`--eval-poison-path`。机制对照默认使用相同的前 K 张 bases；nearest 会受代理特征影响，需要另列实验。

攻击 resume 是**warm start**：从指定快照的下一次迭代开始，恢复图像、重启优化器与 dropout 抽样，并采用相应衰减进度。它不等于保存全部攻击优化状态后的严格续跑。`poison_00050.pth` 已完成51次更新；完整800次的最终快照是 `poison_00799.pth`。

### 受害训练 victim

| 字段 | linear 正式起点 | 全参数正式起点 | CLI 与规划 |
| --- | --- | --- | --- |
| training | linear | end2end | `--victim-training`；与攻击 representation 独立 |
| model_mode | eval | eval | `--victim-model-mode`；全参数另可消融 train，linear 只允许 eval |
| nets | DenseNet121, ResNet18 | 相同 | `--target-net`；每个方法评估相同列表；七个模型家族均可做受害 |
| checkpoint | ckpt-%s-4800.t7 | 相同 | `--test-chk-name`；每次条件训练重新加载 |
| epochs | 60 | 60 | `--retrain-epochs`；两套 smoke 都为2 |
| optimizer / lr | adam / 0.1 | adam / 1e-4 | `--retrain-opt` / `--retrain-lr`；这是配置预设，直接CLI也应显式填写 |
| momentum | 0.9 | 0.9 | `--retrain-momentum`，只用于SGD |
| weight_decay | 0 | 0 | `--retrain-wd`；不要与旧 block 包装器的5e-4混同 |
| batch_size | 64 | 64 | `--retrain-bsize`，包括 linear 缓存特征后的批大小 |
| num_per_class | 50 | 50 | `--num-per-class`；默认10类×50=500张，含5毒样 |
| milestones | [30,45] | [30,45] | `--lr-decay-epoch`，整数轮次，倍率0.1；空列表表示不衰减 |

linear 模式缓存原末层特征、冻结骨干并只训练分类头。end2end 模式不缓存特征，每个 batch 经骨干和分类头重新前向并反传。两者的受害图像变换都采用 ToTensor+Normalize；预训练所用的随机裁剪/翻转不会因选择 end2end 或 train 自动启用。评估目标和干净测试集时均切换到 eval。

受害更新记录包括实际可训练参数数量及一个骨干参数样本的最大变化，帮助确认 linear 骨干不动、全参数训练确实走过骨干梯度。这个单参数探针不是对全部参数的逐项审计。

两种场景复用同一组干净 checkpoint，但必须写入独立 output_root，分别汇总；不能把同名方法在不同受害协议下的结果合并。需要研究原 block 基线时，另建 `representation=blocks` 的配置，选择 block 代理并完整记录受害训练设置；历史 shell 的 lr=1e-4、wd=5e-4 与新版 MSP full 预设不同。

### 两套分阶段命令

完成 `prepare_data.py` 与相应预训练后，执行：

~~~powershell
# linear：features + linear + eval
python workflow.py --config configs/smoke.json --stage craft
python workflow.py --config configs/smoke.json --stage evaluate
python summarize_results.py runs/smoke-seed2026 --config configs/smoke.json

# MSP全参数：features + end2end + eval
python workflow.py --config configs/smoke_end2end.json --stage craft
python workflow.py --config configs/smoke_end2end.json --stage evaluate
python summarize_results.py runs/smoke-end2end-seed2026 --config configs/smoke_end2end.json
~~~

正式流程对应更换为 benchmark.json / benchmark_end2end.json 及各自的 output_root。`workflow --stage all` 只包含 pretrain/craft/evaluate；数据准备与汇总是独立命令。已有权重时不应重复执行 pretrain。

## 分阶段规划实验

| 阶段 | 设置 | 能说明什么 |
| --- | --- | --- |
| 依赖/逻辑验证 | unittest、真实模型特征检查 | 参数传递、数据替换、数学接口是否符合设计 |
| 冒烟 | 1目标、1代理、1受害、2迭代/2轮 | 路径、权重、前向、保存和整个流程是否能跑通 |
| Pilot | 1目标、正式代理与800迭代/60轮 | 大配置显存、耗时、目标预测是否正常 |
| 主实验 | 50目标、相同方法预算、相同受害 | 固定种子下方法间差异 |
| 随机性复核 | 复制主配置，seed=2026/2027/2028，各自新output_root | 差异是否依赖单一种子；分别报告再汇总均值/标准差 |
| 机制消融 | 一次只变pool/flat、concat/layerwise、repeat或代理数量之一 | 哪个因素解释变化 |

benchmark 的三模板×六架构是18个**实例**，不是18种架构。50目标×6方法=300次生成；两个受害×两条件又产生1200次训练。先测目标0的真实时间，再按任务数估算，不能由脚本名字推断“五小时必定结束”。

如果只是 GPU/显存压力，先减少任务数量，保持单次实验的正式预算。若改变batch、代理数或优化器，应另记协议，不能与旧实验当作同一条件合并。

## 指标、归因与 overlap

条件 ASR 的 clean 判断来自**相同受害微调协议的无毒训练**，不是加载预训练模型做一次预测。新入口同时报告全样本ASR、eligible数、条件ASR、测试准确率与成本；不合格目标不暗中删除，eligible=0时用null。

主流程的数据集采用原位替换，相同 seed 下 clean/poisoned 的样本索引位置一致；epsilon=0时图像与顺序逐项相同。历史防御工具依赖毒样本前置，因此 PoisonedDataset 默认兼容旧顺序，主 trainer 显式使用 preserve_order=True。

split 中 clean_train/others 不相交，并不自动证明代理/受害预训练的100%/50%/0% overlap。只有明确掌握 checkpoint 所使用的实际数据文件与训练索引时，才可计算预训练集合交集；权重文件名不能代替这一证据。新 protocol 默认标记未知，保留数据与权重哈希供追溯。

历史 E2E 目录仅说明对应旧结果文件存在，其实际生成协议应结合日志与 checkpoint 核查，不能视作新 features+全参数流水线已实测。当前新流程以 CIFAR-10 为主；保留的 CIFAR-100 结果不构成新版严格复现声明。

可用性实验是另一套从零训练协议：2000干净样本中追加400张同类图像，类别分布改变，且总体poison占2400张的1/6。至少应比较 clean/unperturbed/random/MSP；二组脚本不能排除追加样本量和类不平衡造成的影响。其准确率下降不等于单目标ASR。
