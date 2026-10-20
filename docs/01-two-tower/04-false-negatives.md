# 04｜真凶在训练目标里

第 03 篇排除掉了 temperature：三个取值只差千分之几。既然超参层面解释不了 4.7% 这个数，那就回头检查目标本身。

第 02 篇的公式里留了一个没验证过的假设：**非对角线上的候选，都是这一行用户的负样本。** 这一篇把它量出来、修掉，并做一次单变量对照。

## 一、先把它变成一个可数的问题

"其他列可能不是负样本"这句话没法直接检验。要数，得先有一张能回答下面这个问题的表：

> `(user, item)` 这一对，在训练集里出现过吗？

做法是建一张 `[num_users, num_items]` 的布尔真值表，出现过的地方为 `True`：

```text
positive_matrix[u, i] = True    # 训练集里有 (u, i) 这条正反馈
```

按坐标批量赋值，PyTorch / NumPy 都是同一套写法，比双重循环快好几个数量级。

**这张表只用 train split 构建。** 这一点必须守住：诊断训练目标时只能使用训练阶段已知的信息。一旦把 validation target 也写进去，后面的"修复后指标提升"就变成了自证——你是用 test 信息修的模型。

规模上，`6038 × 3533` 的 bool 不到 22 MB，在这个数据量下很直观。它不是面向亿级 user/item 的通用方案——那时候要么按 batch 现查，要么换稀疏结构。

## 二、对一个 batch 取子矩阵

训练 batch 里有 `user_ids [B]` 和 `item_ids [B]`。用真值表切出这批用户与这批物品之间的所有已知关系：

```text
known = positive_matrix[user_ids][:, item_ids]      # [B, B]
```

它和 `logits` 逐位对齐：`known[r, c]` 表示"第 r 行的用户，和第 c 列的物品，在训练集里有正反馈"。

对角线本来就是当前配对的正样本，是我们想要的答案，不用动它。要找的是**非对角线上的 `True`**：

```text
false_negative_mask = known & ~diagonal
```

冲突有两个常见来源：

1. **同一个用户在 batch 里出现多次**。shuffle 之后，他的多条正交互可能落在不同行，于是 A 行的用户和 B 行的物品之间有正反馈，但 B 行把这个物品当成负类。
2. **即使用户只出现一次**，其他列的物品也可能是他在另一条 train 记录里喜欢过的。

## 三、真实数据上的比例

MovieLens、`batch_size = 256`、`seed = 42`，扫一整个 train epoch：

```text
batches with repeated users: 2200 / 2201
batches with false negatives: 2201 / 2201
false-negative cells: 30,275,048
off-diagonal cells: 143,616,110
false-negative ratio: 21.080538%
```

读法：

- **每个 batch 都有冲突**（2201/2201），其中 2200 个含重复用户。这不是几个偶然噪声点。
- 非对角候选位置一共 1.43 亿个，其中 **3,027 万个位置与训练集里的正反馈冲突**，占 **21.08%**。

也就是说，模型收到的"负样本监督"里，大约每五个就有一个方向是反的。

## 四、这个冲突为什么真的会伤模型

不是"看起来矛盾"，而是梯度方向确实被抵消。看一个具体的：

假设 `(user 96, item 2064)` 是训练集里的正反馈，而 item 2064 恰好出现在同 batch 的某个 column。回到第 02 篇的公式，对 user 96 所在那一行，交叉熵对 logit 的梯度是 `p - y`：

- 他自己对角线那个 item：`y = 1`，梯度为负 → 这个 item 的分数被**推高**；
- 第 2064 列：`y = 0`，梯度为正 → item 2064 的分数被**推低**。

同一行里，用户 96 的两个已知正反馈被要求朝相反方向移动。

放大到整个训练集看更清楚：同一对 `(user 96, item 2064)`，当它作为**对角线**出现时被推高，作为**非对角候选**出现时被推低。它收到的信号取决于"这一轮 batch 怎么洗的牌"，而不是数据本身。

模型仍然能让 loss 下降——它在努力满足这个有冲突的任务，只是学到的排序空间已经被扭曲了。

**这就是为什么"代码正确"和"目标有效"是两件不同的事：这类错误不会让 loss 报错，它照样平稳下降。**

## 五、修复：从分母里去掉

回看第 02 篇的式子：

```text
L = -(1/B) · Σ_i log( exp(s_ii) / Σ_j exp(s_ij) )
```

要做的只是把那些冲突位置从**分母的求和**里去掉：

```text
diagonal            = eye(B)
false_negative_mask = known_positive_mask & ~diagonal
logits              = logits.masked_fill(false_negative_mask, -inf)
```

把一个 logit 设成 `-inf`，softmax 之后该位置概率为 0，相当于告诉当前行：「这个候选不参加负类竞争。」

### 为什么对角线必须保住

`known_positive_mask` 的对角线也是 `True`。如果照上面那样整张屏蔽掉，每一行的正确类别就没了候选——softmax 的分母变成 `Σ exp(-inf) = 0`，概率是 `0/0`，直接得到 `NaN`。

后果比"效果变差"严重：loss 变成 NaN 之后，梯度把它传回去，参数也变成 NaN，最后只看到指标全是 0，而且**没有任何报错**。

所以加完 mask 第一件事就是确认 loss 还是有限值。这是这类改动最常见的翻车方式。

### 语义没有变

注意这不是"把冲突位置变成第二个正类"，只是让它**退出负类竞争**。每行仍然是标准的单标签交叉熵。

这个区别有实际后果：把假负样本"不当负类"是保守的修法，它没有利用"这两个物品用户都喜欢"这个信息。更彻底的做法是多正样本 loss——那属于另一条路线，这一篇不做。

## 六、严格对照的结果

两组都使用 MLP、D=64、temperature=0.1、lr=0.001、5 epochs 和相同随机种子。唯一变化是训练目标：

| Variant | Val Recall@10 | Val Recall@20 | Val NDCG@10 | Val NDCG@20 |
|---|---:|---:|---:|---:|
| 无 mask | 0.027506 | 0.048219 | 0.013288 | 0.018431 |
| 有 mask | 0.053024 | 0.085833 | 0.025977 | 0.034165 |

（这组对照是在 v1 切分上跑的；第 05 篇的最终 test 用的是 v2 重训的 checkpoint，两组数字不能直接拼成一条曲线。）

Recall@20 绝对提升 `0.037614`，相对约 **78.0%**；Recall@10 和 NDCG@10 也接近翻倍。

这个量级比第 03 篇调 temperature 带来的小数点后第三位变化大得多，所以可以确认：**false negative 是主要问题。**

## 七、两个方法论上的坑

### 1. 两组 loss 不能横向比较

mask 之后每行参加 softmax 的候选变少了，loss 天然会低一些——这是结构决定的，与"目标对不对"无关。**两个不同训练目标的 loss 数值没有可比性。**

判断改动有没有效，看固定其余变量之后的 Recall / NDCG，不要看 loss 谁更低。

### 2. 新逻辑要先做退化检查

把 mask 全设成 `False`，带 mask 的训练循环必须和不带的**完全等价**：不只是 loss 相同，最终参数也要逐位相同。

这是验证"新逻辑没有意外副作用"的标准手法，也是在没有对照组的情况下唯一能自己做的自检。它比"跑一遍看看指标涨没涨"快得多，定位也准得多。

## 八、这不是唯一解

mask 不是所有 in-batch 训练的唯一解法。多正样本 loss、按用户分组采样、更大的 batch 都能处理同一类问题，只是改动更大或者引入新的超参。

当前实现胜在改动小、行为清楚：**只动一处，并且能说清它为什么有效。**

## 文献

- **Yi et al.**, *Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations*, RecSys 2019（[Google Research](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/)）——工业级双塔召回里对 batch 内采样偏差的系统处理，和这一篇处理的是同一类问题，只是从"校正概率"的角度入手。
- **Chuang et al.**, *Debiased Contrastive Learning*, NeurIPS 2020（[NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/63c3ddcc7b23daa1e42dc41f9a44a873-Abstract.html)）——对比学习里对假负样本的估计与去偏，思路可以对照着看。

## 动手做

- `labs/lab3-false-negatives-and-mask.ipynb` 概念 1 ~ 5
- 对应自测：`✅ 跑一下：扫完整训练集`、`✅ 跑一下：单变量对照`
- 关键断言：`3-2 对角线保住了（loss 仍是有限值）`、`3-3 零 mask 时两个训练循环的 loss 相同`
