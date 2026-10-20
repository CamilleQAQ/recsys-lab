# 05｜复杂度、容量与实验纪律

第 04 篇把 false negative 修掉之后，validation Recall@20 从 0.048 提到 0.086（v1 切分）。接下来自然会想：再往上加什么？

这一篇不引入新算法，只做两件事：**先把"怎么选"定下来**，然后在这个规则下依次比较训练预算、向量维度和 tower 结构，最后锁定配置、跑一次 test。

## 一、先定规则，再开始调

三条规则，违反任何一条都会让后面的结论失去意义。

**1. 只用 validation 选。** temperature、学习率、维度、结构，全部在 validation 上比。

**2. test 只跑一次，而且在配置锁定之后。** 理由不是形式主义：test 一旦看过，你对它的印象就进入决策回路，之后的"泛化能力"就不再是无偏估计。候选配置越多、看得越多，这个问题越严重。

**3. 早停保存最佳权重时必须深拷贝。** `model.state_dict()` 返回的是指向模型张量的引用；直接存下来，后续 epoch 会继续改动它们，最后拿到的并不是最佳那一刻的权重。这个 bug 不报错，只让最终模型比记录里的差。

## 二、训练是不是不够

> 本节到第四节的 lr / epoch / dimension / tower 消融，是在早先的 v1 切分上跑的；当时还没做 v2 严格时间切分。它们回答的是"往哪个方向加容量"，不能和第五节的 v2 test 数字拼成同一条曲线。

先保持 MLP、D=64、temperature=0.1 和 mask 不变，只比较学习率与训练预算：

| 配置 | best epoch | Val Recall@20 |
|---|---|---:|
| lr=0.001, 5 epochs | 5 | 0.085833 |
| lr=0.001, 15 epochs | 6 | 0.089975 |
| lr=0.01, 15 epochs | 10 | 0.108865 |

两个发现：

- 5 epochs 确实偏短，但**单纯延长帮助有限**：0.085833 → 0.089975，+5%。
- 学习率的影响大得多：0.089975 → 0.108865，+21%。

后续固定 `lr=0.01`、最多 15 epochs、patience=3。这一步没有动模型结构。

## 三、只改 embedding dimension

切回最小结构（裸查表），其他设置保持一致：

| D | Val Recall@10 | Val Recall@20 | Val NDCG@10 | Val NDCG@20 | Best epoch |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.059652 | 0.102403 | 0.028182 | 0.038913 | 9 |
| 64 | 0.065949 | 0.117481 | 0.031372 | 0.044310 | 11 |
| 128 | 0.076056 | 0.128252 | 0.036500 | 0.049570 | 9 |

32 → 64 → 128 持续提升，但增幅在放缓（+14.7%，+9.2%）。

这个实验只支持一件事：**在当前数据、训练方式和候选维度里，128 最好。** 推不出"维度越大越好"——3,533 个物品的规模下，128 维已经接近能把物品彼此分开的量级；换成上亿候选，结论会变。

## 四、MLP 到底有没有用

第 03 篇提到过，第一版给两侧各加了两层 MLP。现在用同一套评估结账——固定 D=64、temperature=0.1、mask、lr=0.01 和相同早停规则，只切换 tower：

| Tower | Val Recall@10 | Val Recall@20 | Val NDCG@10 | Val NDCG@20 | Best epoch |
|---|---:|---:|---:|---:|---:|
| MLP | 0.061475 | 0.108865 | 0.029702 | 0.041556 | 10 |
| Embedding-only | 0.065949 | 0.117481 | 0.031372 | 0.044310 | 11 |

**embedding-only 胜出。**

原因可能是：每个 ID 已经有一个独立可学习的向量，MLP 没有别的特征可以组合，只是在中间加了一层非线性变换，反而增加了优化难度。

这里要特别小心结论的外推边界：

- 这**不能**推出"MLP 对所有双塔都没用"。
- 它只在**纯 ID 输入**下成立。一旦 tower 的输入里有类别特征、文本或上下文，MLP 就有东西可以组合，结论很可能反过来。

这也是第 01 篇那句"输出维度必须相同、内部结构可以不同"的实际含义：结构自由度是有的，只是在当前数据上它没换来收益。

## 五、锁定配置与 final test

v1 消融选出的方向：

```text
tower                = embedding-only
output_dim           = 128
L2 normalization     = true
temperature          = 0.1
false-negative mask  = true
learning_rate        = 0.01
```

（`embedding-only` 分支直接查表，表宽就是 `output_dim`；配置里的 `embedding_dim` 不参与这一分支。）

v1 那一次的最佳 validation 是 Recall@10 = 0.076056，Recall@20 = 0.128252，NDCG@10 = 0.036500，NDCG@20 = 0.049570（best epoch = 9）。在 v2 切分上重训后的最终 checkpoint 是 `best_epoch = 4`，validation Recall@20 = 0.118678（`artifacts/movielens/two_tower_config.json`）。两次切分不同，0.128252 不能当成 v2 的 validation。

配置锁定之后，才跑那一次 test：

| Model | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |
|---|---:|---:|---:|---:|
| Popularity | 0.050248 | 0.024535 | 0.083636 | 0.032910 |
| MF | 0.064463 | 0.031490 | 0.103140 | 0.041186 |
| Two-Tower | 0.055207 | 0.028994 | 0.095537 | 0.039122 |

几点读法上的注意：

- 单 target 协议下，HitRate@10/20 与 Recall@10/20 数值相同，不是两份独立证据。
- Popularity 是"按训练集热度排序"的非个性化基线。Two-Tower 比它高 14%（0.095537 对 0.083636）——超过了，但幅度不大，而且 MF 还比它高。**所以"模型学到了个性化"这句话在这份数据上要说得很克制。**
- validation（0.118678）比 test（0.095537）高，这是正常的：validation 上做过模型选择，天然带一点乐观偏差。**两者之差不是"过拟合"的证据**，也不该拿它当早停依据。

## 六、回顾整条链

```text
连续 ID + 时间切分
    → 最小 Two-Tower + B×B in-batch CrossEntropy
    → naive MLP 能训练，但 Val Recall@20 ≈ 0.047（早先协议下的数字）
    → 自己写 full-ranking 评估，确认问题真实存在
    → 排除 temperature（三个取值只差千分之几）
    → 量出 21.1% 的非对角候选与 train positives 冲突
    → positive mask 把 Recall@20 提到 0.085833（同上，早先协议）
    → validation 上依次比较 lr / epoch / dimension / tower
    → 锁定 embedding-only, D=128
    → final test Recall@20 = 0.095537
```

**最重要的一步是中间那条：先确认目标是对的，再谈容量。**

整条链上唯一"加复杂度"的动作（MLP）没有带来收益，而去掉目标里的自相矛盾带来了 +78%。这个顺序不是巧合，也是这五篇最想留下的东西。

## 七、这一篇的坑

1. **拿 test 调参**。看过一次之后，它就不再是无偏估计。
2. **早停不深拷贝**，存下来的权重不是最佳那一刻的。
3. **把 ID-only 的结论外推**到有 side features 的场景——那是另一个问题。
4. **用 validation 与 test 的差距当"过拟合"证据**。做过模型选择的 validation 本身就带乐观偏差。

## 文献

- **MF**：Koren, Bell & Volinsky, *Matrix Factorization Techniques for Recommender Systems*, IEEE Computer 2009（[IEEE](https://doi.org/10.1109/MC.2009.263)）——隐向量点积模型，本页对照表里的 MF 基线出自这一类工作。
- **SASRec**：Kang & McAuley, *Self-Attentive Sequential Recommendation*, ICDM 2018（[Semantic Scholar](https://www.semanticscholar.org/paper/Self-Attentive-Sequential-Recommendation-Kang-McAuley/97faeefa771e8cc8e55159e2bd03e6f5eef249a8)）——把序列建模引入推荐的代表作。仓库里的 SASRec 实现属于这一类，和 Two-Tower 是两条路线：这里建模的是"用户与物品的静态匹配"，它建模的是"用户行为序列"。

## 动手做

- `labs/lab3-false-negatives-and-mask.ipynb` 的单变量对照格与 locked test 格
- 学习率、维度、tower 三组消融没有搬进 labs——这三组用 `src/recsys/training/tune_two_tower.py` 复现更省事
