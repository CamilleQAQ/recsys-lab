# 04｜loss 在降，交付哪一轮？

第 03 篇的消融里有一组的训练 loss 最低、指标也最高，看起来皆大欢喜。这一篇回到训练本身：一个 epoch 里必须做对哪几件事，训练预算怎么给，以及**为什么"最后一轮"不能当成默认的交付轮次**。

## 一、训练循环里必须做对的四件事

主体逻辑只有"前向 → loss → 反向 → 更新"，但有四处错了不会报错：

**1. epoch 开头 `model.train()`。** 评测代码会调 `model.eval()` 关掉 dropout。不切回来就等于用推理模式训练：loss 照样降，只是 dropout 这个正则项完全没生效，而且重启一个 notebook 跑出来的结果会和上一次不一致。这是最隐蔽的一类——单看 loss 曲线完全正常。

**2. 张量搬到 `device`。** CPU-only 时 `.to(device)` 是空操作，所以漏写不会报错；换到 GPU 上才炸。

**3. 平均 loss 除以"实际跑过的批次数"，不是 `len(dataloader)`。** 加 `max_batches` 之类的限制之后两者会不一致。除以 `len(dataloader)` 只会让 loss 偏小，不报错，loss 曲线就是假的。

**4. `optimizer.zero_grad()` 在 `backward()` 之前。** PyTorch 的梯度是累加的；漏了它，每一步都在用"上一批 + 这一批"的梯度更新。

## 二、这条曲线的形状

`src/recsys/` 里那份锁定配置的完整训练历史（30 轮，`best_epoch = 30`）：

```text
epoch  1  loss 1.2397   val R@20 0.0612
epoch  5  loss 0.8948   val R@20 0.0879
epoch 10  loss 0.7788   val R@20 0.1180
epoch 15  loss 0.6835   val R@20 0.1488
epoch 20  loss 0.6320   val R@20 0.1653
epoch 25  loss 0.5911   val R@20 0.1752
epoch 29  loss 0.5673   val R@20 0.1917
epoch 30  loss 0.5619   val R@20 0.1977     ← best_epoch
```

三个读法：

- **loss 在 30 轮里一直单调下降**（1.2397 → 0.5619），没有任何一轮反弹。所以"看 loss 决定什么时候停"在这条曲线上完全无效——它会让你一路跑到预算用完。
- **这条曲线上第 30 轮恰好是峰值**，所以最终交付的确实是最新一轮。但这是 validation 选出来的结果，不是"跑满就交"的规则：同一个仓库里，KuaiRec 的 SASRec 上限 60 轮，验证指标最好的是第 54 轮，最后 6 轮都没有再提升。换配置、换预算，峰位都可能提前。
- **第 10 轮时 validation 只有 0.1180，约为最终值的 60%。** 也就是说这个模型在第 10 轮时还远没有训练好——`labs/lab5` 用 10 个 epoch 做对照实验，比较的是同一个预算下的相对表现，不是收敛值。

为什么收敛这么慢？把两个模型的目标放在一起看：

| | Two-Tower | SASRec |
|---|---|---|
| 一个 batch 的正样本数 | B = 256 | 最多 B × L = 12800 个位置 |
| 每个正样本配几个负样本 | 2B - 2（in-batch） | 1 |
| 每步参与竞争的候选数 | ~6.5 万 | ~1.3 万（还要扣掉补齐位） |

SASRec 的监督信号**数量**看起来更多，但每个位置的负样本只有 1 个，梯度里关于"哪些物品不该排前面"的信息比 in-batch 目标弱得多。想加快有两条路：每个位置采 5~10 个负样本，或者对全部物品做 full softmax（等价于把整个物品表当候选，贵得多）。两条都会引入新的开销或超参，所以先确认值得再上。

还有一个附带的后果：**每个 epoch 都会重新采一轮负样本**，同一个正样本在不同 epoch 配的是不同的负样本。这相当于数据增强，但也意味着不同 epoch 的 loss 不是同一个任务的 loss，跨 epoch 比较时要留个心眼。

## 三、早停的两个参数

`train_sasrec.py` 用的是 `max_epochs = 30, patience = 5`，配合"按 validation Recall@20 选最优轮次"。三个细节：

**深拷贝最佳权重。** `model.state_dict()` 返回的是指向模型张量的引用；直接存下来，后续 epoch 会继续改动它们，最后拿到的不是最佳那一刻的权重。这个 bug 不报错，只让实际交付的模型比记录里的差：

```text
best_state = copy.deepcopy(model.state_dict())
```

**patience 要跟着收敛速度给。** 这条曲线上第 20 轮到第 30 轮还在涨；patience 给小了，训练会在平台期提前停掉，最终结果明显偏低。序列模型的监督信号弱，patience 通常要比 ID-only 的双塔大一档。

**早停看的是 validation，不是 test。** test 只跑一次，而且要在配置锁定之后——理由不是形式主义，而是看过的 test 会进入你的决策回路，之后再声称"泛化能力"就不再是无偏的。

## 四、参数量：序列模型比双塔小

一个反直觉的对比（本仓库的实际配置）：

| | 参数量 |
|---|---:|
| Two-Tower（用户表 6038×128 + 物品表 3533×128） | 约 1.22 M |
| SASRec（物品表 3534×64 + 位置表 50×64 + 2 个 block） | 约 0.24 M |
| MF（用户表 6038×64 + 物品表 3533×64） | 约 0.61 M |

SASRec 的参数少 5 倍，标准配置下的 Recall 反而更高。不能把差距全部归因于结构——两边的评测协议并不相同；但至少说明在这份数据上，参数量不是瓶颈。更值得看的是**监督信号的密度**：Two-Tower 里一个用户共享一个向量、每次更新只有一条配对监督；SASRec 里同一条序列的 50 个位置各自产生监督。参数被更密集地用上了。

这个对比也给"加容量"这件事定了调：在 ID-only 的设定里，参数量的边际收益远不如把目标写对（`docs/01-two-tower/05-capacity-and-discipline.md` 里那组 MLP 对照是同一个结论）。

## 五、这一篇的坑

1. **漏掉 `model.train()`**：dropout 静默失效，loss 正常。
2. **`total_loss / len(dataloader)`**：加了 batch 上限之后 loss 偏小。
3. **早停不深拷贝**，交付的权重不是最佳那一刻的。
4. **patience 给太小**：验证指标还在涨的时候就提前停掉，最终结果明显偏低。
5. **用 loss 或最后一轮决定交付**：前者一直降；后者不保证是峰值（KuaiRec 的 SASRec 在 54/60 就停了）。

## 文献

- **SASRec**：Kang & McAuley, *Self-Attentive Sequential Recommendation*, ICDM 2018（[arXiv](https://arxiv.org/abs/1808.09781)）——4.2 节讨论了负样本数量对效果的影响；论文里的配置也是"每个位置一个负样本"，这个选择在后续工作里被反复调整。
- **GRU4Rec**：Hidasi et al., *Session-based Recommendations with Recurrent Neural Networks*, ICLR 2016（[arXiv](https://arxiv.org/abs/1511.06939)）——较早系统讨论"排序损失 + 负采样"的工作，也报告了采样数量与收敛速度的关系。
- **NCE**：Gutmann & Hyvärinen, *Noise-Contrastive Estimation: A New Estimation Principle for Unnormalized Statistical Models*, AISTATS 2010（[JMLR](https://proceedings.mlr.press/v9/gutmann10a.html)）——"用少量负样本近似全量 softmax"的方差来源，采样越多方差越小、代价越高；这一节的取舍都能从这篇的角度解释。

## 动手做

- `labs/lab5-sasrec-data-and-train.ipynb` 概念 6 ~ 9
- 对应自测：`5-4 初始 loss 在 2*ln2 附近`、`5-4 max_batches 生效且平均值没被算错`
- 训练记录：`artifacts/movielens/sasrec.pt` 里的 `history`，以及 `artifacts/results.json` 里的 `best_epoch`
