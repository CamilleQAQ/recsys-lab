# 02｜向量检索层：精确不等于更快

前面每个模型训练完，最后一步都是同一句话："给全部物品打分，取 Top-K"。在 3533 个物品上这是一次矩阵乘法；换成上亿候选，这句话就不成立了。

这一篇讲检索层。它不属于任何一个模型，却决定了模型能不能上线。

## 一、检索层要负责什么

每个模型的评测里都有这么一段：

```text
scores = user_vectors @ item_vectors.T          # [Q, I]
scores[row, seen_items] = -inf                  # 已看物品退出竞争
top_k = torch.topk(scores, k).indices           # [Q, K]
```

这三步和模型无关——只要模型能吐出两组向量。把它抽出来有三个理由：

- **换模型不用改检索。** Two-Tower、MF、SASRec 打分方式不同，"取 Top-K"是一样的。
- **可以换后端。** 小规模用暴力内积；上亿候选时换 FAISS，调用方不动。
- **两个坑集中在一处。** 归一化和"已看过滤"各自出错都不会报错，写在一个类里更容易守住。

## 二、归一化必须是"检查"，不能是"自动修复"

内积 `e_u · e_i = ||e_u|| · ||e_i|| · cos` 同时受方向和长度影响。做余弦检索时两边都要归一化，漏一侧排序就悄悄变成另一回事。

本项目的 Two-Tower 在塔的出口就归一化了（`docs/01-two-tower/01`），所以检索层拿到的一般是合法输入。但检索层**不能假设这件事**——它应该主动检查并拒绝未归一化的输入。

关键在于"拒绝"而不是"顺手归一化"。自动归一化的代码在"上游忘了归一化"这个 bug 下会静默地让结果变正确，于是你永远发现不了上游漏了哪一步。而线上召回用的是上游算好的向量，那时候没人替你归一化。

```python
norms = np.linalg.norm(vectors, axis=1)
if not np.allclose(norms, 1.0, atol=1e-4):
    raise ValueError(f'{name} must be L2 normalized before inner-product search')
```

`atol=1e-4` 是给 `float32` 留的余量：归一化之后再转成 numpy、再拷贝一次，长度会有 `1e-7` 量级的偏差。用 `np.allclose` 的默认 `rtol` 对"和 1 比较"没有意义（1 的相对误差等于绝对误差），所以显式写 `atol`。

**检查要能拦住错误，不能替错误打补丁。**

## 三、`IndexFlatIP` 到底是什么

FAISS 的索引分两大类：

| 类型 | 例子 | 找的是 |
|---|---|---|
| 精确（exhaustive） | `IndexFlatIP` / `IndexFlatL2` | 真正的最近邻，逐条算距离 |
| 近似（ANN） | `IndexIVFFlat` / `IndexHNSWFlat` | 大概率正确的近邻，靠剪枝换速度 |

`IndexFlatIP` 是**精确**索引：它对每个查询和全部向量做内积，和 `user_vectors @ item_vectors.T` 是同一个算法。两个直接推论：

- 它和暴力内积的 Top-K **必然完全一致**（浮点误差范围内）；
- 它**不会**因为"用了向量库"就变快——省掉的计算一点没少，只是把循环放进 C++、外加一些内存布局优化。

实测（`labs/lab8`，800 个查询 × 3533 个物品 × 64 维）：

```text
Top-20 重合率                       1.000000
顺序完全一致的用户比例             1.000000
```

这个 1.0 不是"FAISS 效果好"，而是**两者算的是同一个内积矩阵**。如果你的实现里两者不一致，那不是精度问题，是代码问题——先去查过滤和位置映射。

## 四、位置下标不是 item_idx

FAISS 的索引里只有"插入顺序"。你喂进去 3533 个向量，它返回 `0..3532` 这些**位置下标**。

本项目的 `item_idx` 恰好也是连续的 `0..3532`，但这只是巧合，不能依赖。真正要保证的是这条链：

```text
FAISS 位置  --(position_to_item_idx)-->  内部 item_idx  --(index_to_original_movie_id)-->  原始 movie_id
```

少第一张映射表的后果最危险：输出的"item_idx"其实是插入顺序，只要插入顺序不是 `0, 1, 2, ...` 就全错——而且**不会报错**，因为返回的都还是合法物品。指标只会"莫名其妙差一点"。

`labs/lab8` 专门做了个对照来证明这类错误能被抓住：

```text
故意写坏之后：重合率 1.000000，顺序一致率 0.000000
```

把 Top-1 沉到第 5 位——**Top-20 的重合率仍然是 1.0**（集合没变），只有"顺序完全一致率"掉到 0。也就是说：**只看重合率抓不住排序错误。** 两个指标都要看：

- 重合率回答"选出的集合对不对"；
- 顺序一致率回答"名次对不对"（NDCG 关心后者）。

## 五、FAISS 后端为什么要"多取候选再补齐"

FAISS 的 `search` 接口是"给我 k 条"，它不知道你要过滤已看物品。如果前 k 条里有已看物品，凑不够。

做法是先把 `k` 乘一个倍数（比如 10）去检索，过滤后不够再翻倍，直到凑够或者已经取了全部候选：

```text
search_k = min(max(k, candidate_pool * k), num_items)
while 还有用户没凑够:
    检索 search_k 条 -> 过滤 -> 够了的收下，不够的留到下一轮
    search_k = min(search_k * 2, num_items)
    search_k 已经到顶还不够 -> 抛错（这个用户看过的物品比 num_items - k 还多）
```

暴力后端不需要这一步：它一次就把全部 `[Q, I]` 的分数算出来了，过滤后直接在整行上 `topk`。**这是接口差异，不是性能差异。**

## 六、耗时：这一格最容易被误读

同一份数据、同一批用户（800 个查询、3533 个物品、64 维，取中位数）：

```text
暴力内积  92.50 ms
FAISS     40.47 ms
索引构建  0.12 ms（一次性）
```

但 `artifacts/results.json` 里记的那次完整评测（3,025 个用户、128 维）是**暴力 0.2715 s、FAISS 0.2778 s**——FAISS 略慢。

两次结论不同，正好说明这种小规模计时不可外推。影响因素：

- **线程和 BLAS 实现。** 暴力内积在 numpy 里是一次矩阵乘法，会走 BLAS 多线程；FAISS 内部也是矩阵乘法，但线程策略不同。同一台机器换个 `OMP_NUM_THREADS` 就能翻转结论。
- **批量大小。** 查询多的时候两边都摊薄了固定开销；只有一两个查询时，FAISS 的 C++/Python 边界开销占比变大。
- **机器本身。** 不同 CPU 的缓存和向量化能力差别很大。

**所以这一格要读的不是"谁更快"，而是"两者同级"。** 在 3533 个物品这个规模上，性能差被实现细节淹没；FAISS 的收益要在候选数大到单机内存/带宽成为瓶颈时才出现。

## 七、什么时候才值得上 ANN

`IndexFlatIP` 真正的价值在于**它把检索这件事的接口固定下来了**。今天用精确索引，明天数据涨到一亿条、`O(I)` 扫不动了，换成 `IndexIVFFlat` 或 `IndexHNSWFlat` 只需要改构造函数——调用方（`search` 的签名）不变。

这时候要接受三件事：

- **结果不再保证是真 Top-K。** 要用 recall 衡量："ANN 找回来的 Top-10 里有多少真的在精确 Top-10 里"。
- **多出要调的参数。** IVF 的 `nlist` / `nprobe`，HNSW 的 `M` / `efSearch`——它们直接决定"召回率 vs 延迟"的取舍。
- **建索引要时间和内存**，通常离线做，而且要随物品更新重建。

本项目的规模离这一步很远。先把精确路径做对、把接口留出来就够了。**先有正确的基线，再谈优化**——和前面几篇"先跑 Popularity 再跑复杂模型"是同一个道理。

## 八、这一篇的坑

1. **把未归一化的向量喂给 `IndexFlatIP`**：不报错，只是返回"内积最大"而不是"余弦最大"。
2. **把 FAISS 的位置下标当成 `item_idx`**：插入顺序一变就全错，返回的都是合法物品，看不出异常。
3. **只取 `k` 个候选就不管了**：已看物品占掉位置后凑不够，要么返回不足 k 个，要么混进已看物品。
4. **以为用了向量库指标会变好**：`IndexFlatIP` 和暴力内积是同一个算法，指标必然相同。
5. **只看重合率**：它抓不住排序错误（顺序一致率才会掉）。
6. **拿小规模耗时外推**：同一份数据、同一台机器，换个线程数就可能翻转结论。
7. **测速只跑一次**：第一次调用包含一次性内存分配，取中位数更稳。

## 文献

- **FAISS**：Johnson, Douze & Jégou, *Billion-scale Similarity Search with GPUs*, IEEE Trans. Big Data 2019（[arXiv](https://arxiv.org/abs/1702.08734)）——`IndexFlat` / `IndexIVF` 这些索引类型的定义和设计动机都在这里；本项目的 `IndexFlatIP` 属于其中的精确索引。
- **HNSW**：Malkov & Yashunin, *Efficient and Robust Approximate Nearest Neighbor Search using Hierarchical Navigable Small World Graphs*, IEEE TPAMI 2018（[arXiv](https://arxiv.org/abs/1603.09320)）——近似索引里最常用的一类，用来对照"精确 vs 近似"的取舍。
- **ScaNN**：Guo et al., *Accelerating Large-Scale Inference with Anisotropic Vector Quantization*, ICML 2020（[arXiv](https://arxiv.org/abs/1908.10396)）——把量化误差按方向加权，专门优化最大内积检索（MIPS）；可以看作"精确索引之后还能怎么省"的下一个方向。

## 动手做

- `labs/lab8-faiss-retrieval.ipynb` 概念 1 ~ 7
- 对应自测：`8-0 长度不为 1时抛 ValueError`、`8-1 未归一化的物品向量会被拒绝`、`8-3 Top-20 重合率是 1.0`、`8-4 写坏的检索层会被这条对照发现`
- 同一组对照与耗时：`artifacts/results.json` 的 `faiss_index_flat_ip`
