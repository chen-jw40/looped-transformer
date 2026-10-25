# Ouro 八任务持续学习实验

项目只使用 PBS 和 Python。每个 PBS 都有完整的 Python `CONFIG` 配置区，提交实验时直接在对应 PBS 修改参数。唯一命令入口是 `scripts/experiment.py`，参数解析和校验集中在 `scripts/parameters.py`；其中的默认值用于兼容直接 CLI 调用。核心训练、推理和投影实现位于 `ouro_pipeline/`，辅助代码位于 `scripts/`。

2026-10-19 按用户要求保留当前按任务固定 epoch（3/3/5/5/5/7/5/7）的协议，清除了旧统一1 epoch实验的权重、预测、日志及派生统计。当前结果见 `runs/20261017_fixed_epochs_mb8_depth_study/`；rank128 的 first/last/joint 五seed补充实验和本轮 DRSP 记录各在 `runs/20261019_*` 的对应目录中。文中旧单epoch路径仅为实现历史说明，其结果已不可用；清理详情见 `reports/cleanup_one_epoch_20261019/completion.json`。

## 训练方法

| `--method` | 方法 | 默认学习率 |
|---|---|---:|
| `full` | 无约束全参数微调 | 1e-5 |
| `projected` | 全参数深度历史激活投影；每任务更新所有深度基 | 1e-5 |
| `osft` | 按任务阶段预算的权重 SVD 正交子空间微调 | 1e-5 |
| `olora` | 每任务新增 q/v LoRA，冻结历史 adapter，并惩罚输入侧子空间重叠 | 1e-5 |
| `lora` | 单个固定 rank 的 q/v LoRA，八任务连续更新同一套因子 | 1e-5 |

默认任务顺序：FOMC → ScienceQA → NumGLUE-cm → NumGLUE-ds → C-STANCE → MeetingBank → Py150 → 20Minuten。默认每任务最多 5000 个训练样本、K=4、micro batch=8、gradient accumulation=16（单卡有效 batch=128）、max_length=4096。

DRSP 使用 `--method projected --basis-statistic drsp`，支持 micro batch=8。每任务最多在 64 个均匀分布的训练 micro-step 中各选一条样本，按采样次数轮换 batch 内位置；从正常反传中还原该样本的答案 token 平均 CE 梯度，校正 batch token 数和梯度累积缩放，不额外反传。默认 lambda=0.5、sketch width=8。启用 `OURO_RECOVERY_ENABLED=1` 后，优化器断点同时保存采样统计和 probe RNG；`--resume` 恢复 DRSP 历史基及未完成任务的统计。

默认关闭 early stopping，训练至当前任务配置的最后一个 epoch，保存末次权重。epoch 按 TRACE 官方发布脚本 [train_seq_naive.sh](https://github.com/BeyonderXX/TRACE/blob/462e39f616134f4f819efeb3baea8638c03c7db4/scripts/train_seq_naive.sh) 和 train_seq_cl.sh 按任务名映射：

| 任务 | epoch |
|---|---:|
| FOMC | 3 |
| ScienceQA | 3 |
| NumGLUE-cm | 5 |
| NumGLUE-ds | 5 |
| C-STANCE | 5 |
| MeetingBank | 7 |
| Py150 | 5 |
| 20Minuten | 7 |

`--epochs N` 可统一覆盖所有任务；`--epochs-by-task FOMC=3 MeetingBank=7` 可覆盖指定任务，优先于统一值。解析后的 `epochs_by_task` 写入运行配置；每个任务报告记录 `epochs_requested`。每任务 5000 样本时，每 epoch 为 625 个 micro-steps、40 次 optimizer updates，3/5/7 epoch 分别为 120/200/280 次更新。模型、任务顺序和其他超参数仍使用本项目设置，不代表完整复现官方配置。早停实现保留为可选 `--early-stopping`，仅显式启用时才运行 eval CE 检查和最佳权重恢复。

训练完成后 `full` 默认仅在最终 checkpoint 上评估完整 `test.json` 的八个任务；`projected/osft/olora` 默认评估全部阶段已学任务，以计算初学得分、BWT 和遗忘。`--checkpoint-scope all` 可为 Full 启用同样评估，`--checkpoint-scope final` 可仅测最终 checkpoint，但无法估计遗忘。`--train-only` 只训练。

查看完整参数以及运行前的解析结果：

```bash
python3 scripts/experiment.py train --help
python3 scripts/experiment.py train --method projected --output-dir runs/new_projected --dry-run
python3 scripts/experiment.py evaluate --help
```

## 中断恢复

`--resume` 支持 full 及基于 activation 的 append/joint 投影，校验原训练配置与数据哈希，并恢复最近完成任务的 checkpoint 和历史基。启用 `OURO_RECOVERY_ENABLED=1` 后，会在第 1 次及此后每 20 次 optimizer update 后原子保存中途状态，包括模型、AdamW、随机数、DataLoader 顺序、loss 统计及投影基/目标。恢复时沿用已完成更新，不额外增加 epoch。此机制仅支持关闭 early stopping 的 full/activation 投影训练。

`OURO_RECOVERY_DEADLINE` 是 Unix 时间戳；到达后在下一次完整参数更新处保存并将训练标记为 `paused`。外部作业需要安排后续 `--resume`；恢复功能本身不提交作业。旧版未保存的任务中途进度无法恢复，只能从最近完成任务重跑。2026-10-18 的恢复作业使用独立代码快照，每段申请 12 小时，在约 11 小时 15 分后安全暂停；依赖作业负责续训与未完成评估，结果仍写入原 `runs/` 目录。

## PBS 作业

通用 PBS 文件及独立实验配置：

| 文件 | 用途 | `qsub -v` 参数 |
|---|---|---|
| `pbs/train.pbs` | full / projected / osft / olora / ewc 训练，再测试；默认全深度 Projected | `RUN_DIR`，可选 `METHOD`、`SEED` 或 `SEEDS` |
| `pbs/train_adaptive.pbs` | 可重选历史联合基；默认 joint_greedy | `RUN_DIR`，可选 `BASIS_STRATEGY`、`SEED` |
| `pbs/evaluate.pbs` | 已有 checkpoint 的独立评估 | `RUN_DIR`、`EVAL_DIR`，可选 `SPLIT`、`CHECKPOINT_SCOPE` |
| `pbs/test_seeds.pbs` | 已有直接微调 seed42–46 的最终 test | `TEST_SEED` |
| `pbs/base.pbs` | 未训练模型的八任务基线 | `RUN_DIR` |
| `pbs/validate.pbs` | GPU 和 Ouro 模型检查 | 无 |

从项目根目录提交；训练输出必须使用新的目录。

```bash
qsub -v METHOD=full,SEED=42,RUN_DIR=runs/new_full_seed42 pbs/train.pbs
qsub -v METHOD=projected,SEED=42,RUN_DIR=runs/new_projected_seed42 pbs/train.pbs
qsub -v METHOD=osft,SEED=42,RUN_DIR=runs/new_osft_seed42 pbs/train.pbs
qsub -v METHOD=olora,SEED=42,RUN_DIR=runs/new_olora_seed42 pbs/train.pbs
qsub -v METHOD=ewc,SEED=42,RUN_DIR=runs/new_ewc_seed42 pbs/train.pbs

qsub -v RUN_DIR=runs/new_full_seed42,EVAL_DIR=runs/new_full_seed42_test,SPLIT=test pbs/evaluate.pbs
qsub -v TEST_SEED=42 pbs/test_seeds.pbs
```

同一张 GPU 顺序跑多个 seed，只提交一个 job：

```bash
PBS_DRY_RUN=1 SEEDS=42:43:44:45:46 RUN_DIR=runs/new_projected_seeds bash pbs/train.pbs
qsub -l walltime=48:00:00 -v METHOD=projected,SEEDS=42:43:44:45:46,RUN_DIR=runs/new_projected_seeds pbs/train.pbs
```

`SEEDS` 用冒号分隔且优先于 `SEED`；此时 `RUN_DIR` 是输出根目录，每次独立写入 `seed42/` 等子目录，从原始模型初始化。每个 seed 的训练和评估全部完成、GPU 进程退出后才开始下一个；任一步失败则作业退出，已完成结果保留。正式启动前检查全部 seed 目录不存在；重提时使用新的根目录，或只在 `SEEDS` 中填写尚未启动的 seed。所有 seed 共用 PBS 日志，输出配置中的 seed 和独立 summary 可用于对应。

默认 PBS 仍是单 seed 的 12h 时限，多 seed 必须按总耗时覆盖 `walltime`。Miyabi regular-g 的单节点作业最长 48h（[官方队列说明](https://www.cc.u-tokyo.ac.jp/supercomputer/miyabi/service/job.php)）。历史 Projected 约 7.7–8.7h/seed，五次顺序运行约 39–44h，仍需为新配置留余量；该估计不适用于尚未测时的自适应基。OSFT 历史仅训练就约 17.4h/seed，五次无法放进一个 48h job；O-LoRA 历史训练约 9–10h/seed，还需加评估，应拆成较小 seed 组。顺序模式减少 job 数和同时占用的 GPU 数，总 GPU 小时基本不变，不会加速五次训练。

所有可调选项均已列在对应 PBS 的 `CONFIG` 中：训练文件包含学习率、epoch、batch、early stopping、投影、W&B 和训练后评测；评测文件包含模型/数据路径、任务、循环次数、上下文、生成长度、temperature 和显存设置。测试/检查作业不训练，因此不添加无效的学习率等选项。训练配置里 `None` 表示未指定可选参数，评测配置中的继承行为有逐项注释。五次测试的历史 source 路径也列在 `pbs/test_seeds.pbs` 中。

`scripts/pbs_launch.py` 将配置转换成统一 CLI 参数；布尔值、列表和逐任务生成上限无需手动拼接命令。资源申请仍在文件顶部的 `#PBS` 行修改。提交前可以在登录节点检查配置，以下命令不会提交作业或启动容器：

```bash
PBS_DRY_RUN=1 bash pbs/train.pbs
PBS_DRY_RUN=1 METHOD=full bash pbs/train.pbs
PBS_DRY_RUN=1 TEST_SEED=43 bash pbs/test_seeds.pbs
```

也可以把 PBS 中的 `dry_run` 改为 `True`。已经提交的作业使用提交时的 PBS 内容；修改文件只影响后续提交。GPU 主机上也可直接使用 Python：

```bash
python3 scripts/experiment.py train --method full --output-dir runs/new_full --epochs 3
python3 scripts/experiment.py evaluate --run-dir runs/new_full --checkpoint-scope all
python3 scripts/experiment.py base --output-dir runs/base_k4
```

Python 自动启动固定的 PyTorch/vLLM Apptainer 容器。训练进程退出释放 GPU 后再启动 vLLM；每个 checkpoint/深度独立运行。没有独立 shell 启动脚本。

### 五个 seed 在一个作业内并行

`pbs/train_seeds_parallel.pbs` 使用 `select=5:mpiprocs=1`，通过 `pbsdsh` 在五个独立 GPU 节点各运行一个
seed42–46。`SEED_STUDY_ROOT` 指向包含冻结源码、五份配置和 `worker.py` 的实验目录。
每个 seed 独立训练和评估，不同步梯度，有效 batch 仍为128；输出和状态各自保存，整个作业等待最慢的 seed。
启动器按节点名映射 seed，并核对五个 worker 的本次作业状态，避免远程命令返回零但部分 seed 已失败。
`lora` 默认 rank8、alpha32、dropout0.1、q/v 模块；跨任务不重置或扩展 adapter，优化器仍按任务重置。
OSFT、O-LoRA、EWC、LoRA 支持固定 epoch 的优化器断点与任务边界恢复；保存原始 adapter／EWC 历史和 RNG，
adapter 恢复时从原始模型重建，避免将合并权重与 adapter 重复叠加。所有新结果继续写入 `runs/`。

## 已有五次直接微调的 test

历史 source 路径列在 `pbs/test_seeds.pbs` 的 `SOURCES` 中，CLI 默认值仍登记在 `TEST_SOURCES`。只读取已训练的 `08_after_20Minuten`，不重新训练，不追溯执行 early stopping。每个 seed 只评估一次完整八任务 test，输出到 `runs/20261005_direct_finetune_k4_5seeds_test/seedXX/`。checkpoint 使用指向原权重的链接，不复制大模型。修改测试设置时使用新的输出根目录；自动汇总会校验五次评测协议和数据哈希一致。

```bash
python3 scripts/experiment.py test-seeds --seed 42 --prepare-only
python3 scripts/experiment.py test-seeds --seed 42 --dry-run
qsub -v TEST_SEED=43 pbs/test_seeds.pbs
qsub -v TEST_SEED=44 pbs/test_seeds.pbs
qsub -v TEST_SEED=45 pbs/test_seeds.pbs
qsub -v TEST_SEED=46 pbs/test_seeds.pbs
```

最后完成的 seed 自动汇总五次结果，生成 `test_scores.csv`、`statistics.json`、`summary.md`；也可手动执行 `python3 scripts/experiment.py summarize`。完成标记前会将每条实际预测的 prompt/reference/index 与 test.json 核对。

## 评估设置

启用 W&B 时，eval 只上报表格：accuracy、ROUGE-L、Py150 similarity、SARI 各有独立的得分矩阵，以及相对初学分数和历史最好分数的遗忘矩阵。矩阵以 checkpoint 为行、任务为列，附带 split、seed、完成状态和样本数；每个 checkpoint 评估完成后更新同名表格。得分统一为 0–100，遗忘为该尺度上的分差；相对初学分数的遗忘为负表示提升。综合得分、BWT、平均遗忘分别保存为独立表格，不上报 eval 标量曲线。缺失评估或初学 checkpoint 的格子留空；仅测试最终 checkpoint 时遗忘与 BWT 留空。原始尺度的结果仍保留在本地 `summary.json` 中。

完整 test、不抽样，temperature=0，默认 batch=64、GPU memory utilization=0.85。每个任务的生成上限在对应 PBS 的 `max_new_tokens_by_task` 中显式设置：FOMC/C-STANCE=8，ScienceQA/MeetingBank=1024，NumGLUE-cm/ds=32，Py150=256，20Minuten=512。设置整数 `max_new_tokens` 可覆盖逐任务上限。CLI 对应 `--max-new-tokens-by-task FOMC=8 ScienceQA=1024 ...`。

分类及数值任务使用 accuracy，MeetingBank 使用 ROUGE-L，Py150 使用 fuzzy similarity，20Minuten 使用 SARI，同时保留 BLEU/ROUGE 等辅助指标。宏平均先统一为 0–100 分，五次运行使用样本标准差（ddof=1）。历史汇总若没有声明 `primary_metrics`，其原有 ROUGE-L 主指标口径仍可读取。

输入保留最多 4095 个 prompt token 加 BOS。长文本保留开头，Py150 保留末尾；截断方向和数量记录在结果中。对历史运行改变评估 split 必须指定单独的 `--output-dir`，以免混合 eval/test 结果。

`result/` 是用户明确授权后才能写入的归档目录，规则见 `AGENTS.md`。所有自动输出默认位于 `runs/`。

## 投影与模型协议

EWC 使用 `ouro_pipeline/ewc.py` 的多任务对角 empirical Fisher 版本。任务结束并恢复最佳 eval-CE 权重后，
从当前实际训练子集中按训练 seed 选取最多 64 条样本；对每条样本的平均答案 token CE 分别求梯度平方后再平均，
不使用 eval/test，不包含 EWC 惩罚，不把 batch 平均梯度平方当作逐样本 Fisher。采集时关闭 dropout，
保留 activation checkpointing；恢复模型模式和 RNG，清空采集梯度，不更新模型或 optimizer。
下一任务使用 `CE + λ/2 * Σ_k Σ_i F_ki(θ_i-θ_ki)^2`，默认 λ=400（取自
[TRACE 参考实现](https://github.com/BeyonderXX/TRACE/blob/462e39f616134f4f819efeb3baea8638c03c7db4/model/Regular/EWC.py)）。
存储累计 Fisher 和 Fisher 加权历史中心，省略与当前参数无关的常数；该压缩保留全部历史任务惩罚的精确梯度，
区别于只保护最新权重的 online EWC。TRACE 发布代码在训练过程中累加梯度平方且采用最新任务权重作为中心，
因此本项目实现是 Ouro setting 中的多任务 EWC 算法复现，不是该发布代码的逐行复刻。
所有可训练物理参数共用 K=4 循环，FP32 Fisher/参数、BF16 autocast，任务间重置 AdamW；early stopping 仅依据 eval CE。
逐阶段保存普通 dense checkpoint 和 `ewc/*.pt` 历史状态；`--resume` 目前仍只支持 full。
PBS 可通过 `EWC_LAMBDA` 和 `EWC_FISHER_SAMPLES` 覆盖配置，默认评估所有阶段已学任务的完整 test。
五 seed 并行需提交五个独立 job，分别设置 `SEED=42` 至 46 和独立 `RUN_DIR`；`SEEDS` 参数仍表示顺序执行。
设置 `EWC_STUDY_ROOT=runs/<study>` 后，最后完成的 job 自动汇总 `<study>/ewc/seed42` 至 seed46，
或执行 `python3 scripts/experiment.py summarize-study --output-root runs/<study> --methods ewc`。
混合指标统一到 0–100、样本标准差 ddof=1；汇总保留来源路径、split、配置、样本数和完成状态，输出不进入 `result/`。

O-LoRA 复现使用 [原作者发布代码](https://github.com/cmnfriend/O-LoRA/tree/07117e1fc4a5f5ad9308a815a42cee8f46502dc8) 的算法版本。冻结原始权重和历史 adapter，每任务新增 rank=8、alpha=32、dropout=0.1 的 q/v adapter；历史因子沿 rank 维拼接，所有 Ouro 循环共用物理层 adapter。发布代码的正交项是输入侧 `sum(abs(A_old @ A_new.T))`，权重 0.5；可选范数项是当前 A/B 的 Frobenius 范数之和，默认权重 0。它与论文记号中的输出侧 A 和平方惩罚有差异，summary 明确记录实现版本。普通 LoRA 的权重增量为 `(alpha/rank) * B @ A`。

本地独立实现位于 `ouro_pipeline/olora.py`，无需 PEFT。答案 token CE 和正交项一起按 accumulation 缩放；仅显式开启 early stopping 时使用 eval CE 并恢复最佳 A/B；默认使用末次 A/B，再保存累积 adapter 因子与已合并的普通 dense Ouro checkpoint；后续训练继续使用冻结的原始权重和历史因子，推理无需 task ID。默认评估所有阶段完整 test，计算 BWT 和遗忘。学习率 1e-5、有效 batch=128、K=4、TRACE、按任务固定 epoch 且关闭 early stopping 沿用本项目，而原作者 LLaMA 脚本使用首任务 1e-3、后续 1e-4，故这是 Ouro setting 中的算法复现。可通过 PBS 中的 `olora_*` 参数修改 rank、alpha、dropout 和两个损失系数。

对比统计支持 `python3 scripts/experiment.py summarize-study --output-root runs/<study> --methods full olora`；每个方法的 seed42–46 必须完成所有阶段 test，数据哈希、样本数和公共训练/评估配置必须一致。统计保存 0–100 得分、BWT、平均最终遗忘、样本标准差（ddof=1）和来源。

OSFT 复现采用《Sculpting Subspaces》ICLR 2026 附录 A.10 的任务阶段预算方案：第 i 个任务保护前 (i-1)/T 比例的奇异方向。八任务比例为 0、1/8、…、7/8；只运行两个任务的 smoke test 则为 0、1/2。比例对应奇异方向数量，并非参数元素比例。该实验是在 Ouro setting 中复现方法，模型和 batch/weight decay/early stopping 与论文原始实验不同。

`ouro_pipeline/osft.py` 对物理层的 q/k/v/o_proj 和 gate/up/down_proj 做 FP32 SVD（CUDA 使用 gesvd），冻结高奇异值因子并训练剩余 U/S/V 因子。共享循环只使用一套因子，不读取 route_depths 或历史激活 basis。AdamW 前投影梯度、后投影参数，参考 mini_trainer commit `fd5b552177bde9a551d30e73660d6a96205c97c0` 的约束方式；采用 FP32 dense 重建 forward 以保留现有 BF16 linear、K=4 最后出口 loss。TF32 在 OSFT 的 FP32 运算中关闭，不升级当前依赖。

每任务默认使用最后一次更新的因子（显式启用早停时恢复 eval CE 选中的因子），再导出普通 dense Ouro checkpoint，下一任务重新 SVD 并重置 optimizer。embedding、lm_head、norm 和 early_exit_gate 不做 SVD，保留原有训练行为（最后出口 loss 不使用 gate logits）。summary 保存实际秩、重建误差、最终正交残差、训练参数量、耗时、显存、数据来源及哈希。OSFT 不保证旧任务输出不变。默认评估所有阶段已学任务和最终八任务的完整 test，以计算 BWT；`full` 匹配基线也需 `checkpoint_scope=all` 并使用相同 epoch 和 batch 配置。

五 seeds study 的目录为 `runs/<study>/full/seed42` 至 seed46 和 `runs/<study>/osft/seed42` 至 seed46。提交时设置 `OSFT_STUDY_ROOT=runs/<study>`，最后完成的作业自动生成 scores.csv、statistics.json 和 summary.md；也可手动运行：

```bash
python3 scripts/experiment.py summarize-study --output-root runs/your_osft_study
```

汇总要求两种方法均完成所有阶段完整 test、配置/样本数/数据哈希匹配。混合指标统一到 0–100，标准差使用 ddof=1，并报告最终宏平均、初学宏平均和 BWT。来源路径和完成状态保留在 statistics.json 中；输出不进入 result/。

全参数训练使用 FP32 参数和 AdamW 状态、BF16 autocast、gradient checkpointing。K=4 时物理运行四次共享层，并以最后一次输出计算答案 loss。Projected 默认 `--basis-depth-mode all`：每个任务恢复最佳权重后，用同一批最多 64 个训练样本进行 K 次循环的一次完整前向采集流程，更新每个物理线性模块在全部深度的历史基；不再把任务分配到某个深度。每组基跨任务累积，保留已有方向，仅追加激活残差的重要方向。

采样及基存储均复用 `FullDepthActivationCollector` 和 `BasisStore`。`stat_vectors_per_layer=256` 表示每模块每深度最多 256 个激活向量，`basis_max_rank=128` 表示每模块每深度的累计 rank 上限。K=4 时每模块最多采集 1024 个向量，合并 rank 最高为 `min(512, input_dim)`；这些不是匹配单基 rank=128 的预算。达到上限后不替换旧基，未覆盖的残差及饱和状态记录在 `basis_updates`。

训练下一任务时，沿用 `HistoricalFullWeightProjection` 将同一模块各深度的基用 rank-revealing SVD 合并，AdamW 后恢复 `W U = W_task_entry U`，所有循环共用受约束的权重。默认覆盖 Attention q/k/v/o、MLP gate/up/down 及 lm_head/early_exit_gate；`--projection-scope o-proj` 可仅保护注意力输出。推理仍固定 K 次循环，不使用任务对应的退出深度。summary 记录 `projection_protocol`、各阶段 `basis_source_depths`、每深度采样量/基更新和训练时实际 `history_union_ranks`。

复现旧的单任务指定深度模式可设置 `--basis-depth-mode task --route-depths 0 0 1 1 2 2 3 3`。仅传 `--route-depths` 的旧 CLI 也会选择 task 模式；显式 all 模式与 route_depths 同时使用会报错。PBS 复现旧模式时须把 `basis_depth_mode` 改为 `task` 并填写对应列表。新方案使用独立运行目录；历史 summary 和 basis 文件不迁移。

深度消融支持 `--basis-depth-mode last`（仅最后一次循环采集，每模块一组累计基）。它保持训练、推理 K 不变，使用相同的 AdamW 后投影。`--route-depths` 仅 task 模式接受。

检验深度来源时，last 保持 `basis_max_rank=128`、`stat_samples=64`、`stat_vectors_per_call=4`、`stat_vectors_per_layer=256`，使用 `pbs/train_last128.pbs` 在独立目录从原始模型重新训练。更直接的深度来源对照是固定第 1/2/3/4 次循环分别构基：task 模式的八个 route_depths 全设为同一深度索引 0/1/2/3，其他参数一致。all 的四组各 128，合并仍可能达到 512，因此 all 与 last128 的差异还包含覆盖深度数和实际约束秩，不能单独证明循环深度来源的影响。

历史 `runs/20261009_projected_depth_ablations/last` 使用 rank 上限 512、16 vectors/call、1024 vectors/layer，是总容量上限对齐的对照，不与新 last128 合并统计，也不把历史 summary 改写为 128。短序列及基饱和会影响实际采样数和约束秩，分析需结合 `sampled_vectors_by_depth`、`basis_updates`、`history_union_ranks`。

这些是局部 `W U` 更新约束，不能保证整个模型的旧任务输出不变；embedding 和 norm 等未投影参数仍可改变。每个阶段保留投影诊断、basis 和 checkpoint。

### 可重选的历史联合基

默认 `--basis-strategy append` 继续使用上述累计基。新策略 `joint_greedy` 和 `joint_pca` 使用
`ouro_pipeline/adaptive_subspace.py`，支持 `--basis-depth-mode all` 或 `last`，不接受任务路由。
任务结束并恢复最佳权重后，继续复用现有校准前向和模块输入 hook；当前校准 batch size 为 1，
各深度使用相同的位置采样。每任务、每深度的未中心化二阶统计归一化到 trace=1，再按历史任务数等权合并。
低秩历史摘要保存方向、特征值、压缩前 trace 和任务计数，截断后不重新归一化谱值。

`history_rank` 是每深度摘要的 rank 上限，`basis_max_rank` 是每物理模块最终联合保护基的 rank 上限，
都受输入维度和数值秩限制。旧方向允许被替换；历史已丢失的能量无法恢复，不保证旧约束永久保留。
`basis_residual_tolerance` 仅用于 append 策略。joint 策略按独立方向预算选择，深度权重固定为均匀权重。

- `joint_greedy`：每个候选先去掉已选空间，再以所有深度的新增覆盖收益评分；保存来源计数及各深度实际覆盖率。
  来源计数只是候选标签，不表示该深度独占的保护维度。
- `joint_local`：对 `joint_greedy` 的打分消融。每个候选只按它对来源深度的新增覆盖收益评分，
  再从四深度的全部候选中逐次选择最高分；不计入对其他深度覆盖率的收益。
  每次选择后仍对所有候选做全局残差更新与正交化，跨深度重复方向不重复占秩；不是给各深度固定分配配额，
  也不是一次性按初始特征值排序。历史摘要、深度权重、采样和最终总秩预算均沿用 `joint_greedy`。
  summary 的 `selection_gain_scope=source_depth_only` 标记此消融，四深度最终实际覆盖率仍按相同口径报告。
- `joint_pca`：对相同摘要加权拼接的因子求主成分。`--basis-svd-method exact` 给出该摘要覆盖目标的谱最优解；
  默认 `randomized` 是近似求解（8 维 oversampling、2 次迭代，小矩阵使用精确 SVD）。PCA 不赋予方向来源计数。

下一任务固定 Q，复用 AdamW 后的 `W Q = W_task_entry Q` 约束。训练仍为 FP32 主权重、BF16 autocast。
摘要常驻 CPU，在任务边界逐模块搬到模型设备计算；只把最终 Q 交给训练投影器。summary 记录覆盖率、
实际 rank、摘要保留能量、构基耗时及历史摘要/保护基的分别存储量。R=H=512、四深度时，主要方向矩阵
约占单组 rank512 基的 5 倍存储，不属于等总内存对照。

新状态保存为独立的 `adaptive_v1` 格式，含 history 和 Q；`AdaptiveBasisStore.load()` 可恢复后继续构基。
训练入口的 `--resume` 仍只支持 full，尚未提供 projected 自动断点续训。评估读取普通模型 checkpoint，
无需加载历史摘要。旧 basis 文件不迁移。

独立 PBS 默认最终 rank512、每深度 history rank512、64 条校准样本、每次 4 个向量、每深度最多 256 个向量，
保持现有 all 的采样预算；12 小时时限和实际 GPU 构基开销仍需正式实测。运行前可预览：

```bash
PBS_DRY_RUN=1 bash pbs/train_adaptive.pbs
PBS_DRY_RUN=1 BASIS_STRATEGY=joint_pca bash pbs/train_adaptive.pbs
# 以下命令会提交作业；使用独立的新输出目录。
qsub -v SEED=42,RUN_DIR=runs/joint_greedy_seed42 pbs/train_adaptive.pbs
qsub -v BASIS_STRATEGY=joint_pca,SEED=42,RUN_DIR=runs/joint_pca_seed42 pbs/train_adaptive.pbs
```

可把独立 PBS 的 `basis_depth_mode` 改为 last，比较只使用最后深度的历史重选，区分历史更新与四深度联合选择的收益。

与当前固定 epoch、rank128 Joint 五 seed 实验对照时，使用深度对照 PBS 保持每任务64条校准样本、
每次16个向量、每模块每深度最多1024个向量的采样预算。以下仅预览新消融配置，不提交作业：

```bash
PBS_DRY_RUN=1 VARIANT=joint_local RANK=128 SEED=42 RUN_DIR=runs/new_joint_local128_seed42 bash pbs/train_depth_controls.pbs
```

正式比较使用相同 seed42–46、原有训练与完整 test 协议、相同历史摘要秩及最终保护秩，并逐 seed
配对比较 `joint_local` 与 `joint_greedy`。单 seed 深度对照汇总也支持 `joint_local`，优先报告相对同秩 Joint 的差值。
两组选择开销可能不同，不能称为等构基计算量。若 Joint 稳定领先，支持跨深度覆盖收益打分有用；
若接近，则支持在当前设置下仅用来源深度收益也能选出相近效果的保护空间。

## 环境与输入

- Ouro-1.4B revision：`574fa66cb8bf5abdc979642d01cf2b79b16bfab1`。
- TRACE revision：`462e39f616134f4f819efeb3baea8638c03c7db4`。
- 数据协议：`LLM-CL-Benchmark_5000`，数据位于 `data/TRACE/TRACE-Benchmark/`。
- PyTorch 容器：`containers/nvidia-pytorch-25.01-py3.sif`，哈希见 `containers/SHA256SUMS`。
- vLLM 容器：`containers/nvidia-vllm-26.02-py3.sif`。
- Miyabi-G 使用 ARM64；公共 ARM64 torch wheel 用于 CPU 检查，训练使用容器内 CUDA torch。

```bash
python3 scripts/experiment.py setup cpu
python3 scripts/experiment.py setup assets
python3 scripts/experiment.py setup data
python3 scripts/experiment.py setup gpu
python3 scripts/experiment.py validate --kind data
qsub pbs/validate.pbs
```

Ouro 本地 remote-code 已针对 Transformers 的 DynamicLayer cache 做过兼容补丁，模型权重不变。重新下载模型后仍需保留该兼容实现；模型检查覆盖循环执行数、cache/no-cache logits 和共享轨迹。

评分可从保存的预测重算，无需推理：

```bash
python3 scripts/experiment.py recompute runs/your_run
```

目录职责：`ouro_pipeline/` 保留模型、训练、推理评测、指标；`scripts/experiment.py` 是命令入口，`scripts/` 中其他模块负责统一参数配置、容器启动、环境准备、数据及模型检查、五次运行测试调度、报告和统计汇总。辅助模块由统一入口调用，不单独维护参数解析。

### 末深度覆盖约束下的轨迹选基与固定首深度对照

`--basis-strategy last_trajectory --basis-depth-mode all` 只读取前向激活：相同 token 位置的
相邻循环输入差分 `D_k=X_(k+1)-X_k` 提供跨深度候选。每模块先按当前权重和当前历史基运行
原 last/append 更新作为参考，保留全部旧方向，并使用与该参考相同的新增秩；这不保证与另一个
独立训练的 last 实验逐阶段实际秩相同。最终仍只有一组累计基，满容量后不替换方向。

默认 `--trajectory-floor tolerance` 保留 last 原有的5%未覆盖能量门槛；若参考 last 在容量内无法
达到门槛，改用参考的实际覆盖作为可行底线。`reference` 可要求不低于参考实际覆盖，但同秩 SVD
本已最大化 last 覆盖，严格模式通常没有择优空间。轨迹候选为参考新增方向，加上与参考正交的
差分主方向（数量不超过新增秩）；在该空间做 Lagrange 特征空间搜索，并对边界候选做连续混合。
求解不保证全局最优；返回基的末深度覆盖和差分覆盖均重新检查，不满足条件则退回参考 last。
`lm_head`、`early_exit_gate` 沿用末深度激活统计。旧基与硬投影保持不变，不用梯度、敏感度或损失探针。

`pbs/train_depth_controls.pbs` 支持 `VARIANT=trajectory/first/last/joint`、`RANK=512/128/64` 和
`SEED`、`RUN_DIR`。first 使用 task 模式且所有任务 `route_depths=0`，即全体物理模块采集第一次循环，
不是只保护第一个物理 Transformer 层；训练和推理始终 K=4。joint 为现有 joint_greedy，历史摘要秩
与最终秩上限相同。修正后的配置对所有 rank 固定每任务64条校准样本、16 vectors/call、最多1024
vectors/module/depth；采样排除特殊 token 和 padding，在有效 prompt/answer token 上按等分区间中点
取样，各模块与深度使用相同位置。保护 rank 不再决定采样预算。历史实验使用随 rank 缩放的预算和
包含首尾的等距采样，不能与修正后的实验混作同协议对照；比较时需重跑 first 和 last。
采样规则版本记录在 summary 的 `projection_protocol.activation_sampling` 中。
轨迹和 joint 采集四个深度，因此瞬时激活
记录量为对应 first/last 的四倍；轨迹还增加分解和约束求解，不能称为等计算或等瞬时内存实验。

```bash
PBS_DRY_RUN=1 VARIANT=first RANK=64 bash pbs/train_depth_controls.pbs
qsub -v VARIANT=trajectory,RANK=128,SEED=42,RUN_DIR=runs/new_trajectory128 pbs/train_depth_controls.pbs
```

每作业只运行一个 seed，训练八任务并评估全部阶段完整 test。通过 `CODE_DIR` 使用冻结源码，
`DEPTH_STUDY_ROOT` 让作业结束后更新来源、完成状态和计分统一到0–100的描述性比较。
可用 `python3 scripts/experiment.py summarize-depth-controls --output-root runs/your_depth_study` 重新汇总。
每方法一次运行的样本标准差未定义；不把逐模块数量当作实验重复数。输出不进入 `result/`。

### 跨循环梯度耦合保护

`--basis-statistic depth_coupled` 在每任务恢复最佳 eval-CE 权重后，使用原有训练集校准样本采集
`z_i = sum_k (r_i^T b_ik) x_ik`。同一模块、同一 token 的四次调用共用一个 Rademacher 探针，
未中心化二阶矩在期望上保留跨深度项；`depth_diagonal` 使用独立探针，供去交互消融使用。
两者要求 `projected`、`basis_depth_mode=all`、`basis_strategy=append`，每物理模块只有一个历史基，
`basis_max_rank` 是该基的累计上限。容量满后不替换方向，AdamW 后仍恢复 `WQ=W_task_entry Q`。

此实现选择固定最佳权重上的统计，**每任务增加最多 stat_samples 次校准反向**，不宣称零额外模型计算。
校准使用每序列答案 token 平均 CE，禁用 dropout，保留非重入 activation checkpointing；
新统计版本的 FP32 投影禁用 TF32（同时关闭容器覆盖项），BF16 前后向保持原设置。
首次 GPU 短测发现 TF32 下的实际 checkpoint 约束残差大于日志残差，独立 CPU FP32 重算后修正；
与历史激活实验比较时明确记录这一精度差异，不将全部收益归因于深度耦合。
只保存采样输入，反向重计算不重复登记前向，结束后清空梯度并恢复模式和训练 RNG。
`lm_head` 仅最后出口被调用，退出门不参与当前 loss，因此两者沿用末深度激活保护。
summary 记录采样量、探针 seed、耦合/对角探针能量、校准反向数、实际 rank、构基耗时及峰值显存。

`pbs/train_depth_coupled.pbs` 默认 seed42。`RANK=512` 使用 64 样本、16 vectors/call、1024 records/module；
`RANK=128` 使用 64 样本、4 vectors/call、256 records/module，分别匹配现有 last512/last128 的统计预算。
`SMOKE=1` 先做两个小任务阶段和有限 test 验证；正式运行评估全部阶段完整 test。
实验输出使用新 `runs/` 目录，冻结源码后提交。`scripts/coupled_statistics.py` 核对配置、数据哈希及
评估样本数，将混合指标统一至 0–100，并报告相对匹配基线的分差、BWT 与逐任务结果。
每组一次仅支持描述性比较；综合分增加 2 分作为预设参考阈值，不代表统计显著性。
