# MORE：多阶响应能量图异常检测

MORE（Multi-Order Response Energy）利用多阶图响应能量相对于典型水平的偏离作为无标签异常证据，并通过图关系构造三元组，学习更有区分性的异常分数。

本仓库**一次运行一个数据集**，默认运行 `yelpchi`，固定训练 **30 个 epoch**，**不使用 early stopping**。训练集与测试集按 **8:2** 分层划分，不另设验证集，使用最后一个 epoch 的模型进行测试。

## 1. 环境与依赖

- Python 3.10 或更高版本。
- PyTorch 2.1 或更高版本，支持 CPU 或 CUDA。
- NumPy、SciPy、scikit-learn，见 `requirements.txt`。
- 读取 GADBench 的 DGL 二进制图时，另需安装与 PyTorch/CUDA 匹配的 DGL。
- 读取 MATLAB v7.3 文件时，另需安装 `h5py`。

先根据设备安装 [PyTorch](https://pytorch.org/get-started/locally/)，再安装其余依赖：

```bash
python -m pip install -r requirements.txt
```

使用 DGL 数据文件时，请根据 [DGL 安装说明](https://www.dgl.ai/pages/start.html)选择与本地环境匹配的版本。仅使用普通 MAT/NPZ 数据时不需要 DGL。MATLAB v7.3 支持可单独安装：

```bash
python -m pip install h5py
```

## 2. 数据下载与准备

数据文件不随代码上传。可从下述论文的官方项目获取数据：

> Tang J, Hua F, Gao Z, Zhao P, Li J. GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection. Advances in Neural Information Processing Systems, 2023, 36: 29628–29653.

- [GADBench 官方仓库与数据说明](https://github.com/squareRoot3/GADBench#dataset-preparation)
- [官方 README 提供的数据集下载链接](https://drive.google.com/file/d/1txzXrzwBBAOEATXmfKzMUUKaXh6PJeR1/view?usp=sharing)

只需下载并放入本次要运行的数据集，无需准备全部数据。GADBench 不包含本论文全部数据版本；Flickr 和论文中的小规模 YelpChi 需按论文原始来源另行获取。

### 数据版本对应关系

**GADBench 的 YelpChi 为 45,954 节点的完整版，对应本论文的 YelpChiFull；本论文的 YelpChi 为 23,831 节点的另一版本。不能通过改文件名将两者视为同一数据集。**

| 论文数据集 | 论文节点数 | 论文异常节点数 | 论文属性维数 | 数据准备说明 |
| --- | ---: | ---: | ---: | --- |
| Flickr | 7,575 | 450 | 12,047 | 按论文原始来源获取 |
| YelpChi | 23,831 | 1,217 | 32 | 默认运行目标，需准备这一版本 |
| Weibo | 8,405 | 868 | 400 | 可从 GADBench 获取，并核对统计 |
| Questions | 48,921 | 1,460 | 301 | 可从 GADBench 获取，并核对统计 |
| YelpChiFull | 45,954 | 6,677 | 32 | 对应 GADBench 的 YelpChi |

如使用 GADBench 下载的无扩展名 `yelpchi` 文件，建议将其命名为 `YelpChiFull`，然后运行 `--dataset YelpChiFull`。程序读取图内容，不依赖扩展名判断其真实格式；启动时打印实际节点数、异常数和预处理后的属性维数。

将数据放在项目下的 `datasets/`，例如小规模 YelpChi 使用 `datasets/yelpchi.mat`；下载的完整版使用 `datasets/YelpChiFull`。也可用 `--data-dir` 指定其他目录。

支持以下格式：

- `.mat`：属性字段支持 `Attributes`、`features`、`X` 等；邻接矩阵支持 `Network`、`homo`、`adj` 等；标签支持 `Label`、`labels`、`y` 等。
- `.npz` / `.npy`：图字典，支持属性、邻接矩阵或 `edge_index`，以及标签；只加载可信文件。
- DGL 二进制图：支持 `.bin`、无扩展名文件及带 DGL 文件头的其他命名；节点属性字段为 `feat`/`feature`/`features`/`x`，标签为 `label`/`labels`/`y`。应使用同构图版本。

标签约定：`0` 为正常，`1` 为异常，`-1` 为未知。训练损失、参考统计与三元组采样均不使用节点真实标签；标签只用于分层划分和最终评估。未知标签节点加入无监督训练集，不参与评估，因此 **8:2 比例针对有标签节点**。若没有未知标签，比例即针对全部节点，整数节点数按 sklearn 的取整规则处理。

## 3. 运行命令

在项目根目录运行。默认命令为：

```bash
python run_fast.py
```

等价的显式命令：

```bash
python run_fast.py --data-dir datasets --dataset yelpchi --epochs 30 --seeds 42 --output-dir runs/yelpchi
```

运行从 GADBench 下载的 YelpChi 完整版：

```bash
python run_fast.py --data-dir datasets --dataset YelpChiFull --epochs 30 --seeds 42 --output-dir runs/yelpchifull
```

切换其他数据集时，每条命令仅指定一个数据集：

```bash
python run_fast.py --dataset Weibo --epochs 30 --output-dir runs/weibo
python run_fast.py --dataset Flickr --epochs 10 --output-dir runs/flickr
python run_fast.py --dataset Questions --epochs 10 --output-dir runs/questions
```

同一数据集可运行多个随机种子，例如：

```bash
python run_fast.py --dataset yelpchi --epochs 30 --seeds 42 43 44 45 46 --output-dir runs/yelpchi_5seeds
```

以上五个种子是使用示例，不代表已核实论文原实验使用的种子列表。汇总文件报告指定种子下的均值和标准差。

默认自动选择可用的 CUDA，否则使用 CPU。也可指定设备：

```bash
python run_fast.py --dataset yelpchi --device cuda:0 --output-dir runs/yelpchi_cuda
python run_fast.py --dataset yelpchi --device cpu --output-dir runs/yelpchi_cpu
```

每次运行请指定新的输出目录，程序不覆盖已有实验目录。`--datasets` 保留为 `--dataset` 的别名，但同样只能接收一个数据集，不接受 `all` 或多个名称。使用 `python run_fast.py --help` 查看全部参数。

## 4. 默认设置与实现

| 设置 | 默认值 |
| --- | --- |
| 数据集 | yelpchi |
| 训练 epoch | 30 |
| Early stopping | 关闭，无验证集与最佳轮次选择 |
| 划分 | train:test = 8:2，分层随机划分 |
| 最终模型 | 最后一轮模型 |
| 模型版本 | `homo`，即完整 MORE |
| 图计算 | `full`，整图精确计算 |
| 隐藏维数 / 表示维数 | 64 / 32 |
| 最高阶数 K / 分数上限 c | 3 / 20 |
| 学习率 / weight decay | 0.001 / 0.0005 |
| q0 / δsame / δrank | 0.5 / 0.1 / 0.25 |
| 排序 margin / 关系 margin / 温度 | 0.1 / 0.1 / 0.1 |
| λrank / λhom / λvar | 1 / 1 / 0.05 |
| 候选池刷新 / 分数统计记录周期 | 每 5 轮 / 每 5 轮 |
| 每轮更新次数 | 整图模式下默认 1 次 |

主要实现对应论文式 (4)–(17)：MLP 编码；以 `L-I = -D^(-1/2) A D^(-1/2)` 计算 Chebyshev 响应；仅使用第 1 至 K 阶；对响应平方和作 `log1p`；使用训练节点的 median/MAD 归一化；对双向偏离取跨阶最大值并截断；构造相邻低偏离节点对和高偏离非邻居；使用排序损失、三元组关系损失与表示方差约束。

参考统计与候选池按周期刷新，刷新后统计量在当前周期内固定，不对 median/MAD 或采样过程求导。训练中记录损失与分数分布；测试标签只在训练完成后计算一次 AUROC、AUPRC（average precision），不用于选择模型。可加 `--no-evaluate-test` 仅保存模型与分数。

属性预处理默认移除常量列，再进行 StandardScaler 标准化；稀疏属性不减均值，稠密属性减均值。该步骤使用全图可见属性，不使用标签。删除常量列可能使实际输入维数小于论文原始统计。可用 `--preprocessing legacy` 保留常量列进行对照。

若显存不足，可显式启用有界子图计算：

```bash
python run_fast.py --dataset YelpChiFull --graph-mode bounded --gpu-budget-gb 8 --cache-gb 2 --core-nodes 2048 --halo-nodes 16384 --halo-edges 2000000 --output-dir runs/yelpchifull_bounded
```

**有界模式会截断邻域/边，并限制每个 core 内的三元组候选，与论文整图公式不严格等价。** `--graph-mode auto` 也可能自动选择该模式；正式与论文对照时应使用默认 `full`。`--gpu-budget-gb` 是内存规划估计，不是显存分配硬限制。

## 5. 输出文件

| 文件 | 内容 |
| --- | --- |
| `config.json` | 本次完整参数 |
| `results.csv` / `summary.csv` | 各 seed 结果与汇总 |
| `execution_plan.json` | 整图/子图计算方式和内存估算 |
| `feature_audit.json` | 预处理后的属性统计 |
| `split_counts.json` | train/val/test 的节点数与异常数，val 为 0 |
| `history.csv` | 各轮损失、约束数量和参数更新次数 |
| `score_history.csv` | 分数分布、参考统计与触顶比例，不含测试指标 |
| `final_scores.npz` | 最终分数、各阶有符号偏离、参考统计和划分索引；val 为空 |
| `model.pt` | 最后一轮参数和运行配置 |
| `result.json` | 单次运行结果，包含最终 epoch 和 early_stop=false |

## 6. 检查

```bash
python -m unittest -v test_fast.py
```

包含数据预处理、8:2 划分、三元组采样及权重、响应能量/归一化的 CPU 检查；安装 PyTorch 后还会检查数值、梯度、损失公式、单数据集接口和小图端到端运行。未安装 PyTorch 时，相应测试会显示跳过。

本次整理环境没有 PyTorch/DGL，也没有真实数据集，因此未验证真实数据训练、GPU 显存或论文表格数值；不能将本代码包视为已经完成论文全部结果复现。逐项核对结果见 `PAPER_ALIGNMENT.md`。
