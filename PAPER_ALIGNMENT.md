# 代码与当前论文核对

核对依据：本次工作区中的 `MORE (12).pdf`。这里只确认代码与文字/公式的对应关系，不声称已复现论文表格。

## 方法公式

| 论文 | 代码位置 | 核对结果 |
| --- | --- | --- |
| 式 (4)：MLP 编码 | `run_fast.Detector` | 两层 MLP，隐藏 64，表示 32，LeakyReLU(0.2) |
| 式 (5)：Chebyshev 响应 | `run_fast.forward_raw`、`fast_utils.normalized_operator` | 整图模式对应缩放归一化 Laplacian，0 阶仅初始化，评分使用 1…K 阶 |
| 式 (6)：log 能量 | `run_fast.forward_raw` | `log1p(sum(r**2))` |
| 式 (7)–(8)：median/MAD 和有符号偏离 | `bounded_graph.calibration`、`run_fast.online_score` | 参考集合仅为训练节点，MAD 下限默认 1e-8 |
| 式 (9)：异常得分 | 同上 | 双向绝对偏离、跨阶最大值、上限截断 |
| 式 (10)–(12)：三元组条件 | `fast_utils.sample_constraints`、`run_fast.build_pools` | 相邻低分训练节点对、对内 rank 接近、负例为 anchor 的非邻居且 rank 更高；排除自身和正例 |
| 式 (13)：权重 | `fast_utils.sample_constraints` | 显式写为 `q_n-max(q_a,q_p)` |
| 式 (14)：排序损失 | `run_fast.compute_losses` | 加权平滑 margin，负例高于 anchor/positive 中较大分数 |
| 式 (15)：关系损失 | 同上，默认 `--hom-loss triplet` | 正例内距离相对负例最小距离的平滑 margin |
| 式 (16)：方差损失 | 同上 | 训练表示的逐维总体方差，目标标准差 1，稳定常数 1e-4 |
| 式 (17)：总损失 | `run_fast.run_variant` | 三项加权求和，默认 1、1、0.05 |

原采样权重写为 `min(neg, 1-pos)`。有效 percentile 均在 [0,1]，且 `pos=abs(q_a-q_p) <= max(q_a,q_p)`，所以 `neg=q_n-max(q_a,q_p) <= 1-max(q_a,q_p) <= 1-pos`，原式在有效三元组上已等于 `neg`；本次改写仅让式 (13) 的对应关系更直接。

## 已修正的公开运行设置

1. 原脚本默认一次运行 Amazon、Reddit、Questions；现在每次只接收一个数据集，默认 yelpchi。
2. 原脚本没有 early-stopping 分支，却固定调用 `make_split(y, True, seed)`，实际保留 7:1:2 划分。现固定 `early_stop=False`，调用 8:2 分支，无验证节点。
3. 原始脚本和修改版均默认 30 轮；现明确记录最终 epoch，保留最后一轮模型，测试指标只在训练完成后计算一次。
4. 原脚本 `--evaluate-test` 默认 True 且不能关闭；现在支持 `--no-evaluate-test`。
5. 默认由 `auto` 改为 `full`，避免默认静默切换到论文未描述的近似子图算法。保留 `auto` / `bounded` 供显存受限时显式使用。
6. 新增对任意名称无扩展名 DGL 图的发现和读取支持，以适配 GADBench。
7. 移除无验证集情况下的验证指标输出，改为记录训练损失与分数统计；清理旧的 ISP/book 专用说明。

## 仍需区分的实现细节和复现边界

- **数据版本**：论文 YelpChi 是 23,831 节点、1,217 异常；GADBench YelpChi 是 45,954 节点，对应论文 YelpChiFull。数据下载地址不是所有论文数据版本的统一替代来源。
- **预处理**：沿用上传代码的去常量列与 StandardScaler；论文没有完整交代这些步骤。若原实验预处理不同，数值不能保证一致，建议论文补充说明。
- **周期刷新**：统计量、rank 和候选池默认每 5 轮刷新，统计量在周期内固定。论文复杂度段提到了周期刷新，但没有给出此周期的具体数值和停止梯度细节，可在实现说明中补充。
- **采样预算**：候选池默认最多 32,768 项，每步最多使用 8,192 项，有限拒绝采样可能不足额，也可能有重复候选；池为空时只剩方差正则与 weight decay。日志如实记录有效数量。
- **近似子图**：有界模式使用截断图响应、core 内候选和局部方差估计，不能视为整图公式的严格实现；默认 full 避免此偏差。
- **数据划分**：分层划分使用真实标签以控制评估比例；标签不进入损失、参考统计或候选选择。未知标签节点加入训练集，8:2 仅适用于已知标签节点。
- **训练轮数**：按用户要求默认固定 30；论文 Table II 中 Flickr/Questions 为 10，YelpChi/Weibo/YelpChiFull 为 30，README 给出各自命令。
- **随机种子**：默认单次 seed=42；论文报告五次运行。原始五个 seed 未从稿件中确定，README 的五种子命令仅为示例。
- **耗时**：`seconds` 是本次固定轮数训练循环的墙钟时间，另含周期刷新/记录等开销；不包括加载、建图与初始评估。各阶段 CUDA 时间未逐段同步，且不是“到最佳验证 epoch 的训练时间”。

结论：默认完整模型的主要数学表达与论文一致；原始运行协议不完全一致，8:2 等设置已修正。由于缺少真实数据和 PyTorch/DGL 训练环境，不能宣称论文指标、时间和显存已经全部对上。

## 本次实际验证

- Python 语法编译通过。
- `python -m unittest -v test_fast.py`：共 15 项，11 项 CPU 检查通过，4 项 PyTorch 检查因缺少依赖跳过。
- 独立解析命令行函数，核实默认 yelpchi / 30 / early_stop=False / full，`--datasets` 单名称别名及 `--no-evaluate-test` 行为。
- 未执行真实数据集训练、DGL 实际文件解码、CUDA 数值/显存或论文指标复现。
