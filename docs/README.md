# 从 Two-Tower 到向量检索：四套原理笔记

> 主线交付数字用的是当前协议（严格时间边界 + 首次播放标签），都在 `artifacts/` 里。部分章节的**消融**数字来自早先协议、没有按新切分重跑，读的时候注意区分。

第一次来，先看 [开始之前](getting-started.md)——那篇只讲怎么把环境跑起来，不涉及原理。

`docs/` 里有四个系列，各自是一条连续的判断链。

```text
01-two-tower/   一个 batch 怎么变成训练目标 → 训练起来了怎么判断好坏 → 结果很差怎么找根因 → 修好之后怎么锁配置
02-baselines-and-retrieval/
                为什么必须先跑热门基线 → 换负采样换来了什么 → 检索层要负责什么 → 精确索引到底省了什么
03-sasrec/      静态画像差在哪 → 一条序列怎么变成一堆标签 → 注意力凭什么不能看未来 → 训练预算怎么给 → 同一份权重两个数
04-kuairec/     没有评分的数据怎么定标签 → 同一套模型换数据集之后数字为什么不能直接比
```

顺序是按**依赖**排的，不是按实现日期：`02` 只依赖 `01`；`03` 是另一条路线（序列建模），它引用了 `01`/`02` 的结论，但反过来不成立；`04` 换数据集，不依赖任何一条路线的具体实现，但要求 `01`~`03` 建立的评测纪律。

`labs/` 的编号不是这个顺序——它是一条由浅入深的动手路径（Two-Tower → SASRec → 基线 → 检索 → 标签定义）。两个目录的编号不一一对应，具体映射看每篇末尾的「动手做」。

读者默认已经会 Python、`nn.Module`、`Dataset/DataLoader` 和训练循环，也理解 Embedding。这里只补推荐系统里新出现的那部分知识。

还有一份 [术语表](术语表.md)：代码里为什么用英文标识符、`user_idx` 和 `original_user_id` 差在哪、`false_negative` 指哪个位置。**它是查的，不是读的**——碰到看不懂的英文名再翻，不需要从头背一遍。

## 三层分工

| | 讲什么 | 什么时候看 |
|---|---|---|
| `docs/` | 为什么这样做、结果说明什么、有哪些坑 | 想快速理解 |
| `labs/` | 亲手把同一套东西写出来，每步立刻验证 | 想自己实现 |
| `src/recsys/` | 统一配置、checkpoint 元数据、测试 | 想看工程代码 |

五组实验都真跑过，但**配置不同，数字不要混着比**：

| | 模型 | 学习率 | 结果 |
|---|---|---|---|
| `01-two-tower` | MLP tower | 0.001 | mask 前 → 后：validation Recall@20 0.048 → 0.086（v1） |
| `labs/lab1`~`lab3` | embedding-only | 0.01 | mask 前 → 后：validation Recall@20 0.079339 → 0.105785（v2） |
| `02-baselines-and-retrieval` | Popularity / MF / 检索层 | 0.01 | MF random validation Recall@20 0.110522，和 Two-Tower 打平 |
| `03-sasrec` | 2 层因果 block，D=64 | 0.001 | test Recall@20 0.193058（标准序列协议） |
| `04-kuairec` | 同一套 Popularity / MF / Two-Tower / SASRec | 0.01 | 换数据集：候选物品 3,533 → 10,669，训练历史中位数 57 → 451（人均 94 → 494），test Recall@20 SASRec 0.249895 |

`01-two-tower` 的 naive 基线用的是两层 MLP tower；`labs/` 从 Lab 1 手写出来的 embedding-only 结构接着往下做，所以两侧基线不在同一个水平上。两组都是真实结果，只是属于两套实验。

`02-baselines-and-retrieval` 里那个"两个模型差不多"容易被误读：两者都是两张表加内积，差别在训练目标和训练预算；in-batch 目标必须配 temperature 才不退化解，配上之后也就追平了最简单的随机负采样。

SASRec 这一组多一层注意：**它的评测协议和 ID-only 的三个基线不同**（历史用 train + validation），所以 0.193058 不能和 Two-Tower 的 0.095537 直接相减。要对齐口径看 train-only 的 0.160331，`03-sasrec/05` 把两种协议讲清楚了。

`04-kuairec` 是换数据集而不是换模型，所以它那行要单独看：KuaiRec 的物品数是 10,669（MovieLens 3,533），评测用户 7,175（3,025），评测目标还更集中在少数视频上。同一套四个模型里 **SASRec 在两个数据集上都是第一，但 ID-only 三个模型的名次并不一致**（MovieLens：Popularity < Two-Tower < MF；KuaiRec：Two-Tower < MF < Popularity），而且**绝对数字跨数据集不可相减**——`04-kuairec/02` 把这件事拆成了可核对的量。

## 篇目

### 01-two-tower

| 篇 | 回答的问题 | 动手做在 |
|---|---|---|
| [01](01-two-tower/01-data-and-shared-space.md) | 数据为什么这样准备？为什么两座塔、两张表？normalize 放在哪？ | `labs/lab1` 概念 1 ~ 2 |
| [02](01-two-tower/02-in-batch-objective.md) | 一个 batch 怎么变成可训练的目标？temperature 在做什么？ | `labs/lab1` 概念 3 ~ 5 |
| [03](01-two-tower/03-loss-vs-recall.md) | loss 降了为什么不能交付？评估怎么设计才可信？ | `labs/lab2` 概念 1 ~ 5 |
| [04](01-two-tower/04-false-negatives.md) | 那个假设成不成立？冲突怎么量化？修了有多少用？ | `labs/lab3` 概念 1 ~ 5 |
| [05](01-two-tower/05-capacity-and-discipline.md) | 往哪个方向加容量？test 该怎么用？ | `labs/lab3` 对照格与 locked test 格 |

第 05 篇的三组消融（学习率、维度、tower 类型）没有搬进 labs——那三组用 `src/recsys/training/tune_two_tower.py` 复现更省事。labs 里保留了其中最关键的一组：mask 前后的单变量对照。

### 02-baselines-and-retrieval

| 篇 | 回答的问题 | 动手做在 |
|---|---|---|
| [01](02-baselines-and-retrieval/01-popularity-and-mf.md) | 0.096 到底算好算坏？换负采样换来了什么？ | `labs/lab7` 概念 1 ~ 7 |
| [02](02-baselines-and-retrieval/02-vector-retrieval.md) | 检索层要负责什么？精确索引省了什么、没省什么？ | `labs/lab8` 概念 1 ~ 7 |

第 01 篇的结论和直觉相反：**in-batch 负采样没有赢过最简单的随机负采样**（0.061475 / 0.044242 对 0.110522），而它和 Two-Tower 的 in-batch 目标本质是同一件事。第 02 篇则要说明 `IndexFlatIP` 和暴力内积是同一个算法——所以两者结果必然完全一致，而耗时在小规模上不可外推。

这一系列是后写的，但依赖上排在序列模型之前。

### 03-sasrec
| 篇 | 回答的问题 | 动手做在 |
|---|---|---|
| [01](03-sasrec/01-why-sequences.md) | 静态画像装不下什么？0.096 和 0.193 为什么不能相减？ | `labs/lab4` 概念 1 ~ 2、`labs/lab5` 三组对照格 |
| [02](03-sasrec/02-alignment-and-sampling.md) | 一条序列怎么变成一堆标签？负样本从哪来？loss 在哪些位置算？ | `labs/lab5` 概念 1 ~ 5 |
| [03](03-sasrec/03-causality-and-position.md) | 因果 mask 和位置编码到底有没有用？单次实验和短预算能证明什么？ | `labs/lab4` 概念 2 ~ 6、`labs/lab5` 三组对照格 |
| [04](03-sasrec/04-training-loop-and-budget.md) | 训练循环里哪四件事容易漏？为什么交付的不是最后一轮？ | `labs/lab5` 概念 6 ~ 9 |
| [05](03-sasrec/05-evaluation-protocol-and-lock.md) | 同一份权重为什么有两个数？该用哪个协议？ | `labs/lab6` 概念 1 ~ 4 |

第 03 篇有两组实验是**多种子**（每个配置 3 个种子）：位置编码的效应量级和种子抖动相当，单次运行分不出来。而因果 mask 那组的结论**随训练预算翻转**（10 轮时去掉 mask 更好、30 轮时更差）——这两件事是正文的核心结论，不是附带说明。

这一系列是最后一个实现的，但它的两处结论要引用前面的系列：因果 mask 那组的"训练 loss 更低不等于指标更好"，和 `02` 里 in-batch 的退化解是同一类现象（loss 看着正常、模型没学到东西）；末尾"该用哪个协议"的交付表，就是 `01` 那张表加上序列模型的一行。

### 04-kuairec

| 篇 | 回答的问题 | 动手做在 |
|---|---|---|
| [01](04-kuairec/01-label-threshold-and-data.md) | 没有评分的数据，什么算"喜欢"？重复播放怎么办？ | `labs/lab9` 概念 1 ~ 4 |
| [02](04-kuairec/02-protocol-and-models.md) | 同一套模型换数据集，数字为什么不能直接比？ | `labs/lab9` 概念 5、`labs/lab7` 概念 1 ~ 3 |

这一系列不引入新模型，换的是数据和标签：KuaiRec 只有 `watch_ratio`，阈值必须由人定，而定下来的那一刻，用户/物品词表、embedding 表形状、以及"负样本"的含义就都被固定了。第 01 篇的四阈值审计和重复 `(user, video)` 对的合并规则是这一系列的核心；第 02 篇把同一套四个模型放到这份数据上，说明为什么它的 Recall 不能和 MovieLens 的相减，并完整记录了一次被推翻的诊断：SASRec 初看像过拟合（两轮到顶、之后下降），实际是早停窗口把它砍在了低谷里。

## 四个系列共用的方法论

- **只用 validation 选配置，test 只跑一次。**
- **每一组对照各自 `set_seed` + 重建 loader**，复用同一个 loader 会让结果随执行顺序漂移。
- **平均 loss 除以实际跑过的批次数。**
- **改动训练方式之后，先看 loss 再看指标。** 两者的方向不一定一致，甚至可能随预算翻转。
- **先确认效应大于噪声，再下结论。** 同一个配置换随机种子就能差 0.018，单次运行分不出 0.008 的差异。
- **评测代码复用，不每个模型重写一遍。** Lab 6 的评测流程只有"历史怎么拼"是新写的；Lab 7 的训练循环、负采样、评测全部来自 Lab 1 ~ 3。
- **先有正确的基线，再谈优化。** 模型侧的基线是 Popularity，检索侧的基线是暴力内积。
- **检查要能拦住错误，不能替错误打补丁。** 归一化该抛错就抛错，不要顺手修好。

## 读完这些之后

`src/recsys/` 是同一套算法的工程版本：统一配置、checkpoint 元数据、测试。核心算法和文档里讲的一致。

MovieLens 的最终数字和完整协议在 `artifacts/results.json`，KuaiRec 的在 `artifacts/kuairec_results.json`；早先协议的记录留在 `artifacts/legacy_protocol_v1/`。

还没覆盖的方向只剩两个：**加特征**（用户画像、物品类目、多模态内容）和**换目标**（多正样本、listwise、挖空预测）。数据侧的三条线（MovieLens、KuaiRec、ID-only 基线）都已经跑通。
