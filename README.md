# 一个周末，手搓出你的第一个推荐算法系统

还在水课上刷短视频，跟舍友开黑开掉一整个下午？

深度学习学了一堆八股，然后就不知道该拿它做什么？

晚上回宿舍，只有瓦罗兰特启动？

跟着这个项目：

水课上舍友在那边刷抖音，你悄悄把自己的第一个推荐系统写出来了。

舍友绕后被架、正红温的时候，你的推荐算法已经能给出像样的结果了。

少打三天瓦，先人一步去投最热的大厂算法岗实习。

---

## 项目介绍：这是什么？

这是一个**写给会写 PyTorch、但没真正做过推荐算法的人**的教程。

里面没有 RecBole，没有"clone 下来改个 yaml 就能跑"。第一天你就写 `nn.Embedding`——一个 batch 怎么变成训练目标、loss 从哪来、为什么这么设计，得自己敲出来才算懂。

**为什么不用现成的框架？** 因为用它学不会推荐系统，你会学会改配置。这个项目的目标是让你能独立判断一个推荐结果好不好、为什么不好。

它也**不假装一切顺利**。仓库里留着真实的失败：去掉 temperature 之后 in-batch 目标退化到 0.061475（validation Recall@20）、假负样本怎么把指标悄悄拖低、SASRec 的 0.193058 为什么不能和双塔的 0.095537 相减。每个结论旁边都放着它是怎么被逼出来的。

**适合谁**

- 会 Python，写过 `nn.Module` 和训练循环，知道 batch 是什么
- 想搞明白推荐系统到底在做什么，而不是再 clone 一个项目
- 准备投搜广推实习，需要一个能讲深、能真跑的项目

**不适合**：想直接上大模型排名的、只想要 SOTA 数字的、以及不想写代码只看结论的。这个项目的价值就在"你自己写一遍"。

不需要先读论文。你需要的 PyTorch 知识在这里刚好够用。

## 你将亲手实现什么？

一条完整的召回链路，六个环节：

```text
Two-Tower → 负样本处理 → Baselines → FAISS → SASRec → 跨数据集验证
```

每一步都在解决上一步暴露出来的问题：

| 环节 | 要解决的问题 | 在哪 |
|---|---|---|
| **Two-Tower** | 怎么把"用户"和"物品"放进同一个向量空间，一个 batch 怎么变成训练目标 | lab1、lab2 |
| **负样本处理** | 训练 batch 里混进来的"假负样本"会把指标悄悄拖低，怎么发现、怎么修 | lab3 |
| **Baselines** | 0.096 到底算好算坏？先跟 Popularity 和 MF 比 | lab7 |
| **FAISS** | 物品向量算好之后，怎么从全部候选里精确地取出 Top-K | lab8 |
| **SASRec** | 静态画像装不下行为顺序，换成因果注意力 | lab4 ~ lab6 |
| **跨数据集验证** | 同一套模型搬到 KuaiRec，指标从 0.096 掉到 0.019，是模型变差了吗 | lab9、`docs/04-kuairec` |

不是六个互不相干的 demo，是一条能讲通的链。走完之后你会得到两样东西：

**一个真的能跑的系统。** 不是 loss 数字好看，是能对真实用户给出排序、能用 FAISS 做精确检索、能在留出的测试集上拿到可信的 Recall@20。

**一个能讲清楚的理由。** 面试问"你这个双塔为什么有效"，你能答的是 in-batch 目标的负样本从哪来、temperature 在调什么、假负样本为什么必须屏蔽——这些问题在文档里都有对照实验，不是背的。


## 实验结果：它真的有效吗？

两个数据集、四个推荐模型，以及 FAISS 精确检索。所有指标均来自真实运行，完整配置与实验记录见 [`artifacts/`](artifacts/)。

### 1. ID-only 模型：换个数据集，排名就变了

首先比较不使用用户行为顺序的三种模型：

| 模型 | MovieLens-1M Recall@20 | KuaiRec Recall@20 |
|---|---:|---:|
| Popularity | 0.083636 | **0.023275** |
| MF | **0.103140** | 0.021882 |
| Two-Tower | 0.095537 | 0.019233 |

有意思的是，**两个数据集上的模型排名并不一致**：

- **MovieLens-1M**：MF > Two-Tower > Popularity
- **KuaiRec**：Popularity > MF > Two-Tower

更复杂的模型并不一定更好。至少在当前实验设置下，Two-Tower 没有稳定超过传统 MF，甚至在 KuaiRec 上不如最简单的 Popularity。

这不是需要掩盖的失败，而是值得研究的结果：模型效果依赖数据分布、训练方式和评测任务，不能把单次实验的排名当成普遍规律。

### 2. SASRec：序列信息能带来什么？

接下来引入 SASRec，让模型利用用户历史行为的先后顺序。

| 数据集 | SASRec Recall@20 |
|---|---:|
| MovieLens-1M | **0.193058** |
| KuaiRec | **0.249895** |

这两个数值都明显高于前面 ID-only 模型的对应结果，但这里有一个重要前提：

**SASRec 与 ID-only 模型的评测协议并不完全相同，不能直接相减或计算提升百分比，更不能据此断言 SASRec 一定优于其他模型。**

值得注意的是，KuaiRec 的用户历史长度中位数为 451 条，而 MovieLens-1M 的中位数是 57 条（平均 94 条）。更丰富的行为历史可能为序列建模提供更多信息，但这只是一个值得验证的假设，不能直接作为性能差异的因果解释。

### 3. 为什么一定要用两个数据集？

因为一个数据集上的数字很容易被误读。

例如，Two-Tower 在 MovieLens-1M 上的 Recall@20 是 `0.095537`，在 KuaiRec 上则是 `0.019233`。

这不代表模型换个数据集就退化了五倍。两个数据集的候选池规模、交互分布和评测任务都不同，**Recall@20 的绝对值不能直接跨数据集比较**。

引入第二个数据集，不是为了凑更多实验数字，而是为了检验：哪些观察能够重复出现，哪些结论只在特定数据和评测设置下成立。

**这个项目不只关注模型能跑出多高的指标，更关注这些指标究竟能说明什么。**


## 学习路线：一个周末怎么完成？

先认识三个入口。它们解决的是不同的问题：

| 你想做什么 | 去哪 | 说明 |
|---|---|---|
| 看懂"为什么这么做" | [`docs/`](docs/README.md) | 四套原理笔记，每篇独立，末尾的「动手做」会指向具体的 lab 格 |
| 亲手写一遍 | [`labs/`](labs/) | 9 个 notebook，每个格子写完立刻有 `PASS` 告诉你对不对 |
| 看完整工程实现 | [`src/recsys/`](src/recsys/) | 和 lab 里同一套算法，多了配置管理和 checkpoint 元数据 |

三层缺一不可：只有 lab 会变成"我跟着抄了一遍"，只有 src 会变成"我看不懂这段在干嘛"，只有 docs 则一直停在纸上。另外还有一个 [`docs/术语表.md`](docs/术语表.md)：代码里为什么用英文标识符、`user_idx` 和 `original_user_id` 有什么区别。**随查随用，不需要提前通读。**

排期按一个周末算：

| | 你会得到什么 | 对应 lab |
|---|---|---|
| **周五晚** | 用 `nn.Embedding` 搭出双塔，把一个 batch 变成 in-batch 训练目标 | lab1、lab2（21 + 17 项自测） |
| **周六** | 训练、评估，找到假负样本并把 Recall 修回来；再补上 Popularity 和 MF 基线做对照 | lab3、lab7（18 + 45 项自测） |
| **周日** | 用 FAISS 做精确检索，用 SASRec 做序列推荐，走完召回到检索的链路 | lab4 ~ lab6、lab8（41 + 61 + 45 + 36 项自测） |

每个 lab 都是同一个节奏：**脚手架 → 概念 → 你来写 → 断言 → 折叠答案**。写完立刻有 `PASS` 告诉你对不对，卡住了再翻答案。不用等别人批改。

做不完也没关系，可以分几次做；部分 lab 会导入前一节导出的模块或 checkpoint，首次请按编号顺序完成。

## 快速开始

装好依赖之后，十几秒就能看到第一个 `PASS`：

```bash
python -m venv .venv
. .venv/bin/activate                      # Windows: .venv\Scripts\activate
python -m pip install -e .
python labs/verify_labs.py lab1-two-tower-from-scratch.ipynb
```

第一次跑 lab1，十几秒就能看到 21 项自测全过（含 MovieLens 自动下载）。全量校验要二十多分钟，时间基本都花在 lab5 上——它真的在训练 3 组 SASRec。

数据会自动下载：MovieLens-1M 约 6 MB，KuaiRec 的压缩包 431 MB（解压后的矩阵 1.03 GB，只下一次）。第一次跑 lab 会等一会儿，之后走缓存。

环境、数据体积、Colab 方案、以及卡住了怎么办，都在 **[docs/getting-started.md](docs/getting-started.md)**。

## 项目目录

```text
labs/          9 个 notebook，从双塔到序列模型，每个都带自测
  lab1~lab3    双塔：模型 → 训练评估 → 假负样本
  lab4~lab6    SASRec：注意力主干 → 数据与训练 → 序列评测
  lab7         Popularity 与 MF 基线
  lab8         FAISS 精确检索
  lab9         KuaiRec 的标签阈值（进阶，需要 431 MB 数据）
docs/          四套原理笔记 + 术语表，讲"为什么"和"哪里会错"
  01-two-tower/                    双塔主线
  02-baselines-and-retrieval/      基线与检索
  03-sasrec/                       序列模型
  04-kuairec/                      换数据集后协议怎么变
src/recsys/    同一套算法的工程实现，测试数量以 `pytest` 的当前输出为准
artifacts/     实跑出来的指标、checkpoint 哈希、模型配置
```

## 数据集、许可证与注意事项

这个仓库**不包含任何数据集**——`data/` 在 `.gitignore` 里，代码只负责按需下载。这一点和下面的条款有关。

**代码**：[MIT](LICENSE)，Copyright (c) 2026 CamilleQAQ。

**数据集不适用 MIT**，它们各有各的条款：

**MovieLens-1M**（GroupLens Research, University of Minnesota）

- 仅供研究用途；**不得再分发**、**不得用于商业目的**、使用时需致谢
- 条款原文：[ml-1m-README.txt](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt)
- 引用：F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context*. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19.

**KuaiRec 2.0**（Gao et al.）

- 官方仓库的 [LICENSE](https://github.com/chongminggao/KuaiRec/blob/main/LICENSE) 为 **CC BY-SA 4.0**，不是 CC BY 4.0；使用、改编和再分发前请核对署名与相同方式共享要求。
- 来源：[Zenodo 10.5281/zenodo.18164998](https://zenodo.org/records/18164998)
- 引用：Chongming Gao, Shijun Li, Wenqiang Lei, Jiawei Chen, Biao Li, Peng Jiang, Xiangnan He, Jiaxin Mao, Tat-Seng Chua. 2022. *KuaiRec: A Fully-observed Dataset and Insights for Evaluating Recommender Systems*. CIKM '22.

**关于 `artifacts/movielens/` 里的 checkpoint**：那是用 MovieLens 训练出来的模型权重。按 GroupLens 的条款，这份数据及其衍生物只用于研究和学习，不要拿去做商业产品。

---

<sub>环境：Python 3.10.7 / torch 2.14.1+cpu / faiss 1.15.1。所有数字都来自真实运行，复现方式见 `artifacts/`。</sub>
