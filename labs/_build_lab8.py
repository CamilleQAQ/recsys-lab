"""生成 labs/lab8-faiss-retrieval.ipynb。"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def markdown(source: str, **metadata: object) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": metadata,
        "source": source.rstrip("\n").splitlines(keepends=True),
    }


def code(source: str, **metadata: object) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": source.rstrip("\n").splitlines(keepends=True),
    }


CELLS: list[dict] = []

# ---------------------------------------------------------------- 0
CELLS.append(
    markdown(
        """# Lab 8｜向量检索层：暴力内积与 FAISS

前面每个 lab 的最后一步都是同一句话："给全部物品打分，取 Top-K"。在 3533 个物品上这是一次矩阵乘法，没什么好说的；上亿候选时这句话就不成立了。

这一章做两件事：

1. 把"打分 → 取 Top-K"抽成一个**检索层**（`ItemRetriever`），它和模型解耦——换模型不用换检索代码；
2. 用暴力内积和 FAISS `IndexFlatIP` 各实现一遍，然后**验证它们给出完全相同的结果**，并测一下谁快。

先给结论，免得看到后面失望：**在这个规模上，FAISS 不保证更快，实测可能略快，也可能略慢。** 这一章的价值在于搞清楚 `IndexFlatIP` 到底是什么、以及"精确检索"和"更快检索"为什么不是一回事。

符号沿用前几个 lab：📖 概念，🧩 要你写的代码，✅ 断言，🚀 可以改着玩的地方，💡 折叠的参考答案。"""
    )
)

# ---------------------------------------------------------------- 1
CELLS.append(
    code(
        """# 不用改：接上 Lab 1 ~ Lab 3 的模型和评测，并探测 faiss 是否可用
import math
import sys
import time
from pathlib import Path

import torch


def find_labs_dir() -> Path:
    for base in [Path.cwd(), *Path.cwd().parents]:
        for candidate in (base, base / 'labs'):
            if (candidate / 'labkit.py').exists():
                return candidate
    raise FileNotFoundError('找不到 labs 目录，请从仓库根目录或 labs 目录启动')


LABS_DIR = find_labs_dir()
if str(LABS_DIR) not in sys.path:
    sys.path.insert(0, str(LABS_DIR))

import labkit  # noqa: E402

try:
    from myrec.eval import group_seen_items, recommend_top_k
    from myrec.mask import build_train_positive_matrix, train_one_epoch_masked
    from myrec.model import TwoTower
except ModuleNotFoundError as error:
    raise SystemExit('先跑完 Lab 1 ~ Lab 3：它们会导出 labs/myrec/') from error

# faiss 是可选的：装了就两条路都跑，没装就只跑纯 torch 的暴力路径
try:
    import faiss  # type: ignore

    HAS_FAISS = True
except ModuleNotFoundError:
    faiss = None
    HAS_FAISS = False

# numpy 是纯 torch 路径也要用的（打分矩阵、argsort），单独探测一次
try:
    import numpy as np  # type: ignore

    HAS_NUMPY = True
except ModuleNotFoundError:
    np = None
    HAS_NUMPY = False

PASSED = []


def check(name: str, actual, expected) -> None:
    '''严格相等：列表、元组、整数、布尔。'''
    ok = actual == expected
    print(('PASS' if ok else 'FAIL'), name, '| 期望:', expected, '| 实际:', actual)
    if not ok:
        raise AssertionError('还没对，改那一格，或看下面的参考答案')
    PASSED.append(name)


def check_close(name: str, actual: float, expected: float, tolerance: float = 1e-5) -> None:
    '''浮点比较，允许误差。'''
    ok = abs(actual - expected) <= tolerance
    print(('PASS' if ok else 'FAIL'), name, '| 期望约:', expected, '| 实际:', actual)
    if not ok:
        raise AssertionError('还没对，改那一格，或看下面的参考答案')
    PASSED.append(name)


print('labs dir:', LABS_DIR)
print('faiss 可用:', HAS_FAISS, '' if not HAS_FAISS else f'(版本 {faiss.__version__})')
print('numpy 可用:', HAS_NUMPY)
if not HAS_NUMPY:
    raise SystemExit('这一章需要 numpy：pip install numpy')"""
    )
)

# ---------------------------------------------------------------- 2
CELLS.append(
    code(
        """# 不用改：数据 + 训练一个 Two-Tower（这一章只把它当作"产生物品向量的来源"）
labkit.set_seed(42)
data = labkit.prepare_movielens()
print(data.describe())

seen_items = data.validation_seen  # 与正式 validation 协议一致，过滤目标前全部已评分物品
validation_users = [row.user_idx for row in data.validation]

# 训练用户取前 1200 个，几十秒就能跑完；评测仍然在全部 validation 用户上做
TRAIN_USERS = 1200
train_subset = [row for row in data.train if row.user_idx < TRAIN_USERS]
positive_matrix = build_train_positive_matrix(data.train, data.num_users, data.num_items)

labkit.set_seed(42)
model = TwoTower(data.num_users, data.num_items, embedding_dim=64, temperature=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loader = labkit.create_train_loader(train_subset, batch_size=256, seed=42)
for epoch in range(5):
    loss = train_one_epoch_masked(model, loader, optimizer, positive_matrix)
    print(f'  TwoTower epoch {epoch + 1}/5  loss {loss:.4f}')

model.eval()
print()
print('物品向量和用户向量的形状（注意：Lab 1 的 TwoTower 出口已经 L2 normalize 过）')
with torch.no_grad():
    item_vectors = model.encode_items(torch.arange(data.num_items))
    user_vectors = model.encode_users(torch.tensor(validation_users))
print('  item_vectors:', list(item_vectors.shape))
print('  user_vectors:', list(user_vectors.shape))
check('8-x 物品向量数量等于物品总数', len(item_vectors), data.num_items)
check('8-x 用户向量数量等于评测用户数', len(user_vectors), len(validation_users))
check('8-x 两侧维度相同', item_vectors.shape[1], user_vectors.shape[1])"""
    )
)

# ---------------------------------------------------------------- 3
CELLS.append(
    markdown(
        """## 📖 理解一下：检索层要负责什么

前面每个 lab 的评测里都有这么一段：

```text
scores = user_vectors @ item_vectors.T          # [Q, I]
scores[row, seen_items] = -inf                  # 已看物品退出竞争
top_k = torch.topk(scores, k).indices           # [Q, K]
```

这三步就是检索层的全部工作，和模型无关——只要模型能吐出两组向量。把它抽出来有三个理由：

- **换模型不用改检索。** Two-Tower、MF、SASRec 的"打分"方式不同，但"从候选里取 Top-K"是一样的。
- **可以换后端。** 小规模时用暴力内积就够；换成上亿候选时换 FAISS，调用方不动。
- **两个坑集中在一处。** 归一化和"已看过滤"这两件事各自出错都不会报错，写在同一个类里更容易守住。

### 归一化为什么必须在检索层检查

内积 `e_u · e_i = ||e_u|| · ||e_i|| · cos` 同时受方向和长度影响。要做"余弦相似度检索"，两边都必须先归一化；任何一侧漏了，排序结果就悄悄变成另一回事。

本项目的 Two-Tower 在塔的出口就归一化了（`docs/01-two-tower/01`），所以这里通常是"已经归一化"的。但检索层不能假设这件事——**它应该主动检查并拒绝未归一化的输入**。理由很实际：把一个没归一化的向量喂给 FAISS，FAISS 不会报错，它只会返回一个"内积最大"的结果，而那个结果和余弦排序不是一回事。等你在 A/B 实验里发现指标不对，已经过去很久了。

### `IndexFlatIP` 是什么

FAISS 的索引分两大类：

| 类型 | 例子 | 找的是 |
|---|---|---|
| 精确（exhaustive） | `IndexFlatIP` / `IndexFlatL2` | 真正的最近邻，逐条算距离 |
| 近似（ANN） | `IndexIVFFlat` / `IndexHNSWFlat` | 大概率正确的近邻，靠剪枝换速度 |

`IndexFlatIP` 是**精确**索引：它对每个查询和全部向量做内积，和 `user_vectors @ item_vectors.T` 是同一个算法。所以：

- 它和暴力内积的 Top-K **必然完全一致**（浮点误差范围内）；
- 它**不会更快**——省掉的计算一点没少，只是把循环放进 C++、外加一些内存布局优化。

真正省计算的是 ANN 那一类，代价是结果近似、需要调参、并且要接受"召回率不是 100%"。这一章先把"精确"这一档做对，把它的能力边界说清楚。"""
    )
)

# ---------------------------------------------------------------- 4
CELLS.append(
    markdown(
        """### 🧩 你来写：`check_l2_normalized()`

检查一个 `[N, D]` 的向量矩阵是否每行长度都是 1。**任何一行不满足就抛 `ValueError`**（不是警告、不是自动归一化）。

容差用 `1e-4`——`float32` 下归一化之后再转来转去会有小误差。"""
    )
)

CELLS.append(
    code(
        """def check_l2_normalized(vectors, name):
    '''只要存在长度不为 1 的行就抛 ValueError。'''
    # TODO 8-0：算每一行的 L2 长度，和 1 比较
    #   提示：np.linalg.norm(vectors, axis=1)
    #   用 np.allclose(norms, 1.0, atol=1e-4) 判断
    #   不满足时抛 ValueError，消息里带上 name
    raise NotImplementedError('TODO 8-0')""",
        lab_todo="8-normalize",
    )
)

CELLS.append(
    code(
        """# ✅ 跑一下：归一化的能过，没归一化的必须被拦住
normalized = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], dtype=np.float32)
check('8-0 归一化过的矩阵能通过',
      check_l2_normalized(normalized, 'test vectors') is None, True)

for bad_name, bad_matrix in (
    ('长度不为 1', np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32)),
    ('有一行是零向量', np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)),
):
    try:
        check_l2_normalized(bad_matrix, 'test vectors')
    except ValueError as error:
        print(f'{bad_name} ->', type(error).__name__, error)
        raised = True
    else:
        raised = False
    check(f'8-0 {bad_name}时抛 ValueError', raised, True)

# float32 下归一化之后再拷贝一次，长度会有 1e-8 量级的偏差，不应被误判
slightly_off = normalized * np.float32(1.0000001)
check('8-0 1e-7 量级的偏差在容差内', check_l2_normalized(slightly_off, 'test') is None, True)"""
    )
)

CELLS.append(
    markdown(
        lab_answer_for="8-normalize",
        source="""<details>
<summary>💡 参考答案（<code>check_l2_normalized</code>）</summary>

```python
def check_l2_normalized(vectors, name):
    '''只要存在长度不为 1 的行就抛 ValueError。'''
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError(f'{name} must be L2 normalized before inner-product search')
```

</details>

`np.allclose` 的默认 `rtol=1e-5` 对"和 1 比较"这件事没有意义（1 的相对误差等于绝对误差），所以这里显式写 `atol=1e-4`。

**不要把这个检查写成自动归一化。** 自动归一化的代码在"上游忘了归一化"这个 bug 下会静默地让结果变"正确"，于是你永远发现不了上游漏了哪一步——而线上召回用的是上游算好的向量，那时候没人替你归一化。**检查要能拦住错误，不能替错误打补丁。**"""
    )
)

# ---------------------------------------------------------------- 5
CELLS.append(
    markdown(
        """## 📖 理解一下：FAISS 只认"第几条"

FAISS 的索引里只有"插入顺序"。你喂进去 3533 个向量，它返回的是 `0..3532` 这些**位置下标**，而不是你的 `item_idx`。

本项目的 `item_idx` 恰好也是 `0..3532`（连续化过），但这只是巧合，不能依赖。真正需要保证的是下面这条链：

```text
FAISS 返回的位置  --(position_to_item_idx)-->  内部 item_idx  --(index_to_original_movie_id)-->  原始 movie_id
```

所以检索层要同时持有两张映射表。少一张的后果分别是：

- 少了第一张：输出的"item_idx"其实是插入顺序，只要你的插入顺序不是 `0, 1, 2, ...` 就全错——而且**不会报错**，只是推荐结果看起来还行（因为都是合法物品）。
- 少了第二张：没法把结果还原成用户能看懂的电影。这个通常不会静默错，因为你会立刻发现输出是内部编号。

`IndexFlatIP` 的构造函数还要求显式给维度：`faiss.IndexFlatIP(d)`。维度对不上时 `add` 会直接报错，这一点比前面几个坑友好得多。"""
    )
)

# ---------------------------------------------------------------- 5
CELLS.append(
    code(
        """# 不用改：一个辅助函数（答案里用得到）
def to_float32_numpy(vectors):
    '''把 torch 张量或 numpy 数组转成 FAISS 能接受的连续 float32 矩阵。'''
    if isinstance(vectors, torch.Tensor):
        vectors = vectors.detach().cpu().numpy()
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError('vectors must have shape [num_vectors, embedding_dim]')
    return np.ascontiguousarray(array)


print('辅助函数就绪')
print('  to_float32_numpy(torch 张量) ->', to_float32_numpy(torch.zeros(3, 4)).dtype)
print('  to_float32_numpy(numpy 数组) ->', to_float32_numpy(np.zeros((3, 4))).dtype)"""
    )
)

# ---------------------------------------------------------------- 6
CELLS.append(
    markdown(
        """### 🧩 你来写：`ItemRetriever`

一个类，把"打分 → 过滤 → 取 Top-K"封起来，支持两种后端。

```python
class ItemRetriever:
    def __init__(self, item_vectors, seen_item_indices, backend='brute_force'):
        ...        # item_vectors: 已归一化的 [I, D]（numpy 或 torch 都接受）

    def search(self, query_vectors, k, candidate_pool=None):
        ...        # 返回 list[list[item_idx]]，每行 k 个，已过滤已看物品
```

要点：

- `__init__` 里调 `check_l2_normalized` 检查物品向量；`search` 里检查查询向量。
- `backend='brute_force'`：`queries @ item_vectors.T` → 把已看物品压 `-inf` → `topk`。
- `backend='faiss'`：建 `faiss.IndexFlatIP(D)`、`add(item_vectors)`，检索时**多取一些候选**（因为已看物品会占位置，见下），再过滤、补足 `k` 个。
- `backend='faiss'` 但环境里没有 faiss 时，抛 `RuntimeError`，不要静默退回暴力。

**关于"多取一些候选"**：FAISS 一次检索返回固定条数，如果前 `k` 个里有已看物品，凑不够。做法是先把 `k` 乘以一个倍数（比如 10）去检索，不够再翻倍，直到凑够或到顶。"""
    )
)

CELLS.append(
    code(
        """class ItemRetriever:
    '''物品向量的检索层：精确内积，支持暴力与 FAISS 两种后端。'''

    def __init__(self, item_vectors, seen_item_indices, backend='brute_force'):
        # TODO 8-1：把 item_vectors 转成 numpy float32、检查归一化、
        #   保存 seen_item_indices（dict: user_idx -> set）
        #   backend='faiss' 时建好索引并保存，没装 faiss 就抛 RuntimeError
        raise NotImplementedError('TODO 8-1')

    def search(self, query_vectors, k, user_indices=None, candidate_pool=10):
        '''返回 list[list[item_idx]]，每行 k 个，已过滤已看物品。'''
        # TODO 8-1：两条后端分支 + 过滤 + 凑够 k 个
        #   brute_force: 一次算完 [Q, I]，逐行过滤后 topk
        #   faiss: 多取候选、过滤、不够就翻倍重试
        #   user_indices 为 None 时表示"不做已看过滤"
        raise NotImplementedError('TODO 8-1')""",
        lab_todo="8-index",
    )
)

CELLS.append(
    code(
        """# ✅ 跑一下：手造 6 个物品向量，检查两种后端的结果是否逐位一致
labkit.set_seed(20)

# 6 个物品落在 4 维空间里，手工归一化
toy_items = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.6, 0.8, 0.0, 0.0],
    [0.0, 0.0, 0.6, 0.8],
], dtype=torch.float32)

toy_seen = {0: {1}, 1: set()}
brute = ItemRetriever(toy_items, toy_seen, backend='brute_force')

queries = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
brute_top3 = brute.search(queries, k=3, user_indices=[0, 1])

check('8-1 每个查询返回 k 个', [len(items) for items in brute_top3], [3, 3])
check('8-1 用户 0 的结果里没有已看的 item 1', 1 in brute_top3[0], False)
check('8-1 查询 [1,0,0,0] 的第一名是 item 0', brute_top3[0][0], 0)
# 用户 1 的历史为空，所以 item 1 没被过滤掉，直接就是第一名
check('8-1 查询 [0,1,0,0] 对用户 1 的第一名是 item 1', brute_top3[1][0], 1)
# 同一个查询换到用户 0（他已经看过 item 1），第一名必须让位给 item 4（[0.6, 0.8, 0, 0]）
check('8-1 同一个查询对用户 0 时 item 1 被剔掉，第一名变成 4',
      brute.search(queries[1:2], k=3, user_indices=[0])[0][0], 4)
# 对照：同一个查询（[0,1,0,0]）对用户 1 不做过滤时 item 1 仍然在结果里
check('8-1 但换成用户 1（历史为空），item 1 又出现了',
      1 in brute.search(queries[1:2], k=3, user_indices=[1])[0], True)
check('8-1 不做过滤时 item 1 也在结果里',
      1 in brute.search(queries[1:2], k=3, user_indices=None)[0], True)

# 未归一化的物品向量必须被拦住
try:
    ItemRetriever(toy_items * 2.0, toy_seen, backend='brute_force')
except ValueError as error:
    print('未归一化 ->', type(error).__name__, error)
    raised = True
else:
    raised = False
check('8-1 未归一化的物品向量会被拒绝', raised, True)

# 未知后端也要报错，不要静默退回暴力
try:
    ItemRetriever(toy_items, toy_seen, backend='magic')
except ValueError as error:
    print('未知后端 ->', type(error).__name__, error)
    raised = True
else:
    raised = False
check('8-1 未知后端抛 ValueError', raised, True)

# faiss 装了的话，两种后端结果必须逐位一致。
# 注意这里用"没有并列分数"的例子：小例子里 6 个物品很容易出现多个 0 分，
# 而 np.argsort 和 FAISS 在并列时选谁不保证一致（见正文）。
if HAS_FAISS:
    faiss_retriever = ItemRetriever(toy_items, toy_seen, backend='faiss')
    faiss_top3 = faiss_retriever.search(queries, k=3, user_indices=[0, 1])
    print('暴力:', brute_top3)
    print('FAISS:', faiss_top3)
    check('8-1 小例子上两种后端的前两名一致',
          [row[:2] for row in faiss_top3], [row[:2] for row in brute_top3])

    # 无并列分数的例子：5 个物品，内积互不相同
    distinct_items = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.8, 0.6],
        [0.6, 0.8],
        [0.28, 0.96],
    ], dtype=torch.float32)
    distinct_items = distinct_items / distinct_items.norm(dim=1, keepdim=True)
    distinct_query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    distinct_brute = ItemRetriever(distinct_items, {}, backend='brute_force')
    distinct_faiss = ItemRetriever(distinct_items, {}, backend='faiss')
    brute_rows = distinct_brute.search(distinct_query, k=3, user_indices=None)
    faiss_rows = distinct_faiss.search(distinct_query, k=3, user_indices=None)
    print('无并列时的暴力:', brute_rows, ' FAISS:', faiss_rows)
    check('8-1 无并列分数时两种后端逐位一致', faiss_rows, brute_rows)
else:
    print('（环境里没有 faiss，跳过 faiss 后端的断言）')"""
    )
)

CELLS.append(
    markdown(
        lab_answer_for="8-index",
        source="""<details>
<summary>💡 参考答案（<code>ItemRetriever</code>）</summary>

```python
class ItemRetriever:
    '''物品向量的检索层：精确内积，支持暴力与 FAISS 两种后端。'''

    def __init__(self, item_vectors, seen_item_indices, backend='brute_force'):
        if backend not in ('brute_force', 'faiss'):
            raise ValueError("backend must be 'brute_force' or 'faiss'")
        vectors = to_float32_numpy(item_vectors)
        check_l2_normalized(vectors, 'item vectors')
        self.item_vectors = vectors
        self.num_items = vectors.shape[0]
        self.embedding_dim = vectors.shape[1]
        self.seen_item_indices = seen_item_indices
        self.backend = backend

        self.index = None
        if backend == 'faiss':
            if not HAS_FAISS:
                raise RuntimeError('faiss is not installed; install it or use brute_force')
            build_started = time.perf_counter()
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.index.add(self.item_vectors)
            # 索引是一次性构建的，耗时单独记下来，方便和检索耗时区分
            self.index_build_seconds = time.perf_counter() - build_started

    def _filtered_top_k(self, scores, k, user_indices):
        '''把已看物品压成 -inf，再取前 k 个列下标。'''
        if user_indices is not None:
            for row, user_idx in enumerate(user_indices):
                already_seen = self.seen_item_indices.get(user_idx, set())
                if already_seen:
                    scores[row, sorted(already_seen)] = -np.inf
        return scores, np.argsort(-scores, axis=1)[:, :k]

    def search(self, query_vectors, k, user_indices=None, candidate_pool=10):
        '''返回 list[list[item_idx]]，每行 k 个，已过滤已看物品。'''
        queries = to_float32_numpy(query_vectors)
        check_l2_normalized(queries, 'query vectors')
        if queries.shape[1] != self.embedding_dim:
            raise ValueError('query and index embedding dimensions must match')

        if self.backend == 'brute_force':
            scores = queries @ self.item_vectors.T
            _, top = self._filtered_top_k(scores, k, user_indices)
            return top.tolist()

        # faiss：一次检索固定条数，前 k 个可能被已看物品占掉，所以要多取并翻倍重试
        search_k = min(max(k, candidate_pool * k), self.num_items)
        remaining = list(range(len(queries)))
        results = [None] * len(queries)
        while remaining:
            scores, positions = self.index.search(queries[remaining], search_k)
            still_remaining = []
            for row, original_row in enumerate(remaining):
                seen = set()
                if user_indices is not None:
                    seen = self.seen_item_indices.get(user_indices[original_row], set())
                unseen = [int(p) for p in positions[row] if int(p) not in seen]
                if len(unseen) >= k:
                    results[original_row] = unseen[:k]
                elif search_k == self.num_items:
                    raise ValueError(f'user {user_indices[original_row]} has fewer than k unseen items')
                else:
                    still_remaining.append(original_row)
            remaining = still_remaining
            search_k = min(search_k * 2, self.num_items)
        return results
```

</details>

两个辅助函数在脚手架里（下一格），不要重写：

```python
to_float32_numpy(vectors)     # torch / numpy -> float32 的 C 连续矩阵
check_l2_normalized(...)      # 你已经写好了
```

**为什么暴力后端不需要"多取再补"**：它一次就把全部 `[Q, I]` 的分数算出来了，过滤之后直接在整行上 `topk`，不存在"取少了"的问题。FAISS 的接口是"给我 k 条"，所以才要那个循环。

**`search_k == self.num_items` 时抛错而不是继续循环**：这时候已经把全部候选都看过了，还是凑不够 k 个，说明这个用户的已看物品太多了（比 `num_items - k` 还多）。继续循环会死循环。

**`np.argsort(-scores)` 和 FAISS 的区别**：两者在"分数相同时选谁"上不保证一致。上面的小例子里 6 个物品在 4 维空间里很容易出现多个 0 分，于是第 3 名就有并列，两边选出的物品不同——**这不是实现错，是并列没有唯一的正确答案**。所以那一格改成用"内积互不相同"的例子来断言完全一致，另一种做法是让重合率在并列处放宽。

真实数据上这个问题存在但比例极小（3533 个物品、64 维，分数几乎不会恰好相等）。如果换成低维或二值向量，并列会变多，那时候"两边结果一致"这句话本身就要重新定义。"""
    )
)

# ---------------------------------------------------------------- 7
CELLS.append(
    markdown(
        """## 📖 理解一下：怎么证明"两条路等价"

两个实现给出同一个结果，需要区分两件不同的事：

```text
Top-K 重合率        ：两边选出的 K 个物品里，有多少是共同项
顺序完全一致的用户比例：连名次都一模一样
```

第一项更像"结论一致"，第二项更严。为什么两个都要看：

- **只有重合率高、顺序不一致**：说明排序的头部有并列分数，或者浮点误差让相邻名次换了位置。指标（Recall/NDCG）会受影响，但影响很小。
- **重合率本身就是 1.0**：说明两条路选出的集合完全相同。在"精确索引 vs 暴力内积"这个场景下，这是**理论上的必然**——它们算的是同一个内积矩阵，只是循环写在 C++ 里还是 Python 里。

所以这一格的断言不是"FAISS 效果更好"，而是"**它和暴力内积是同一个东西**"。如果你的实现里两者不一致，那不是精度问题，是代码问题——先去查过滤和位置映射。

### 顺带说说什么时候才值得上 ANN

`IndexFlatIP` 的价值在于：**它把检索这件事的接口固定下来了**。今天用精确索引，明天数据涨到一亿条、`O(I)` 扫不动了，换成 `IndexIVFFlat` 或 `IndexHNSWFlat` 只需要改构造函数——调用方（`search` 的签名）不变。这时候才需要接受"近似"的代价：

- 结果不再保证是真正的 Top-K，要拿 recall 来衡量（"ANN 找回来的 Top-10 里有多少真的在精确 Top-10 里"）；
- 多出 `nlist` / `nprobe` 这类参数，要调；
- 建索引要时间，通常离线做。

本项目的规模（3533 个物品）离这一步还很远——先把精确路径做对、把接口留出来就够了。"""
    )
)

# ---------------------------------------------------------------- 9
CELLS.append(
    markdown(
        """### 🧩 你来写：`top_k_agreement()` 和 `benchmark_retrieval()`

**a) `top_k_agreement(first, second, users, k)`** —— 输入两份推荐（`{user_idx: [item_idx]}` 或 `list[list]` 都可以，这里用列表形式），返回 `(平均 Top-K 重合率, 顺序完全一致的用户比例)`。

**b) `benchmark_retrieval(retriever, queries, users, k, repeats=3)`** —— 计时跑 `repeats` 次，返回 `(中位数耗时, 推荐结果)`。用 `time.perf_counter()`，取中位数而不是平均值（第一次调用会有一次性的内存分配开销）。"""
    )
)

CELLS.append(
    code(
        """def top_k_agreement(first, second, k):
    '''返回 (平均 Top-K 重合率, 顺序完全一致的用户比例)。'''
    # TODO 8-2：逐行比较
    #   重合率 = |交集| / k；顺序一致 = 两个列表完全相同
    raise NotImplementedError('TODO 8-2')


def benchmark_retrieval(retriever, queries, users, k, repeats=3):
    '''计时跑 repeats 次，返回 (中位数耗时, 最后一次的推荐结果)。'''
    # TODO 8-2：time.perf_counter() 包住 retriever.search，取中位数
    raise NotImplementedError('TODO 8-2')""",
        lab_todo="8-agreement",
    )
)

CELLS.append(
    code(
        """# ✅ 跑一下：形状、手算得出的重合率、以及计时函数的返回
first = [[1, 2, 3], [4, 5, 6]]
second = [[1, 2, 9], [4, 5, 6]]
overlap, exact = top_k_agreement(first, second, k=3)

check('8-2 第 1 行重合 2/3、第 2 行重合 3/3',
      overlap, (2 / 3 + 1.0) / 2)
check('8-2 只有第 2 行顺序完全一致', exact, 0.5)

check('8-2 完全相同时重合率为 1', top_k_agreement(first, first, k=3), (1.0, 1.0))
check('8-2 完全不同的结果重合率为 0',
      top_k_agreement([[1, 2], [3, 4]], [[5, 6], [7, 8]], k=2), (0.0, 0.0))

# 计时函数：暴力后端的耗时必须是正数，而且结果要能取回来
toy_queries = torch.randn(64, 4)
toy_queries = toy_queries / toy_queries.norm(dim=1, keepdim=True)
median_seconds, timed_results = benchmark_retrieval(brute, toy_queries, None, k=3, repeats=2)
print(f'计时示例：{median_seconds * 1000:.3f} ms（64 个查询、6 个物品）')
check('8-2 中位数耗时是正数', median_seconds > 0, True)
check('8-2 返回的推荐条数对得上', len(timed_results), 64)
check('8-2 每个查询 k 个结果', [len(items) for items in timed_results[:5]], [3] * 5)"""
    )
)

CELLS.append(
    markdown(
        lab_answer_for="8-agreement",
        source="""<details>
<summary>💡 参考答案（<code>top_k_agreement</code> / <code>benchmark_retrieval</code>）</summary>

```python
def top_k_agreement(first, second, k):
    '''返回 (平均 Top-K 重合率, 顺序完全一致的用户比例)。'''
    overlap_sum = 0.0
    exact_matches = 0
    for first_row, second_row in zip(first, second):
        first_items = list(first_row[:k])
        second_items = list(second_row[:k])
        overlap_sum += len(set(first_items).intersection(second_items)) / k
        if first_items == second_items:
            exact_matches += 1
    return overlap_sum / len(first), exact_matches / len(first)


def benchmark_retrieval(retriever, queries, users, k, repeats=3):
    '''计时跑 repeats 次，返回 (中位数耗时, 最后一次的推荐结果)。'''
    timings = []
    results = None
    for _ in range(repeats):
        started = time.perf_counter()
        results = retriever.search(queries, k=k, user_indices=users)
        timings.append(time.perf_counter() - started)
    timings.sort()
    return timings[len(timings) // 2], results
```

</details>

`timings[len(timings) // 2]` 对奇数个样本就是正中间那个；`repeats=3` 时取的是第 2 次。用 `statistics.median` 更清楚，但这里不想多引一个模块。

**计时为什么要取中位数而不是平均**：第一次调用通常包含一次性的内存分配（FAISS 会为结果分配缓冲），它会把平均值拉高。中位数对这类离群值不敏感。这一章的绝对耗时都只有几百毫秒，比的也只是数量级——**不要用这里的数字去推断 FAISS 在大规模下的表现**。"""
    )
)

# ---------------------------------------------------------------- 10
CELLS.append(
    markdown(
        """## 📖 收束：实测结果

这一格把两条路放在同一份数据上比。结果是这一章最想让人记住的部分（`artifacts/results.json` 里也有同一组数）：

```text
Top-K 重合率                       1.0
顺序完全一致的用户比例             1.0
```

也就是说，`IndexFlatIP` 和暴力内积给出的是**同一份结果**。这不是巧合，也不是"FAISS 效果好"，而是两者算的是同一个内积矩阵。如果你预期"用上向量库指标会变好"，那是把检索层当成了模型层。

耗时那一栏要小心读。本项目的规模是几百到几千个查询 × 3533 个物品 × 64 维，两边都在几十到一百毫秒这个量级，**实测下来 FAISS 可能更快、也可能更慢**，取决于：

- **线程和 BLAS 实现。** 暴力内积在 numpy 里是一次矩阵乘法，会走 BLAS 的多线程；FAISS 内部也是矩阵乘法，但线程策略不同。同一台机器上换个 `OMP_NUM_THREADS` 就能翻转结论。
- **批量大小。** 查询条数多的时候两边都摊薄了固定开销；只有一两个查询时 FAISS 的 C++/Python 边界开销占比会变大。
- **历史 v1 的 `artifacts/legacy_protocol_v1/results.json` 里记的那次**（6035 个用户、128 维）是暴力 0.2264 s 对 FAISS 0.2366 s——FAISS 略慢。当前 v2 结果另见 `artifacts/results.json`；这种小规模计时不可外推。

**所以这一格的结论不是"谁更快"，而是"谁和谁等价"。** 在 3533 个物品这个规模上，精确索引和暴力内积的性能差被淹没在实现细节里；FAISS 真正的价值在于把接口固定下来，以及它后面还接着 ANN 那一档——`IndexIVFFlat` / `IndexHNSWFlat` 才是省计算的那类，代价是结果变成近似、要调参、要衡量召回率。"""
    )
)

# ---------------------------------------------------------------- 11
CELLS.append(
    code(
        """# ✅ 跑一下：同一份数据、同一批用户，两条路各跑一遍
labkit.set_seed(21)
retrieval_users = validation_users[:800]
with torch.no_grad():
    query_vectors = model.encode_users(torch.tensor(retrieval_users))

brute_retriever = ItemRetriever(item_vectors, seen_items, backend='brute_force')
brute_seconds, brute_results = benchmark_retrieval(
    brute_retriever, query_vectors, retrieval_users, k=20, repeats=3
)
print(f'暴力内积   {brute_seconds * 1000:8.2f} ms  ({len(retrieval_users)} 个用户)')

check('8-3 暴力后端每个用户 20 个推荐',
      [len(items) for items in brute_results[:10]], [20] * 10)
check('8-3 暴力后端结果里没有已看物品',
      sum(len(set(items) & seen_items[user_idx])
          for user_idx, items in zip(retrieval_users, brute_results)), 0)

if HAS_FAISS:
    faiss_retriever = ItemRetriever(item_vectors, seen_items, backend='faiss')
    faiss_seconds, faiss_results = benchmark_retrieval(
        faiss_retriever, query_vectors, retrieval_users, k=20, repeats=3
    )
    print(f'FAISS      {faiss_seconds * 1000:8.2f} ms')
    print(f'索引构建（一次性） {faiss_retriever.index_build_seconds * 1000:.2f} ms')

    overlap, exact = top_k_agreement(brute_results, faiss_results, k=20)
    print()
    print(f'Top-20 重合率          {overlap:.6f}')
    print(f'顺序完全一致的用户比例  {exact:.6f}')

    check('8-3 Top-20 重合率是 1.0（两条路结果相同）', overlap, 1.0)
    check('8-3 顺序完全一致的用户比例是 1.0', exact, 1.0)
    check('8-3 FAISS 这组也没有推荐已看物品',
          sum(len(set(items) & seen_items[user_idx])
              for user_idx, items in zip(retrieval_users, faiss_results)), 0)
    print()
    print(f'耗时对比：暴力 {brute_seconds * 1000:.2f} ms  vs  FAISS {faiss_seconds * 1000:.2f} ms')
    print('（这个规模下两者同级；谁快取决于机器、线程数和批量大小，不要外推）')
else:
    brute_metrics = recall_at_k(
        {user_idx: items for user_idx, items in zip(retrieval_users, brute_results)},
        data.validation, 20,
    )
    print('（环境里没有 faiss，只跑了暴力后端）')
    print(f'暴力后端的 Recall@20: {brute_metrics:.6f}')
    check('8-3 暴力后端的 Recall@20 高于随机水平', brute_metrics > 5 * 20 / data.num_items, True)"""
    )
)

# ---------------------------------------------------------------- 12
CELLS.append(
    markdown(
        """### 🧩 你来写：和仓库里已有的评测工具对齐

Lab 2 写了 `recommend_top_k`，它做的是同一件事（打分 → 过滤 → Top-K）。把两条路的结果对一遍：**同一个模型、同一批用户，检索层和评测函数必须给出同一份推荐**。

这是这一章最后一条"防静默错误"的检查：如果两条路不一致，说明检索层里的过滤或映射写错了——而那类错误不会报错，只会让指标莫名其妙地差一点。"""
    )
)

CELLS.append(
    code(
        """def compare_with_eval_tool(model, retrievers, item_vectors, users, seen_items, k=20):
    '''把检索层的结果和 myrec.eval.recommend_top_k 对照。'''
    # TODO 8-4：用 recommend_top_k 生成一份基准推荐
    #   再把每个 retriever 的结果和它比，返回 {backend: (重合率, 顺序一致率)}
    raise NotImplementedError('TODO 8-4')""",
        lab_todo="8-compare",
    )
)

CELLS.append(
    code(
        """# ✅ 跑一下：检索层 vs 评测函数
retrievers = {'brute_force': brute_retriever}
if HAS_FAISS:
    retrievers['faiss'] = faiss_retriever

agreement = compare_with_eval_tool(model, retrievers, item_vectors, retrieval_users,
                                   seen_items, k=20)
print()
print('后端           与 recommend_top_k 的重合率   顺序一致率')
for backend, (overlap, exact) in agreement.items():
    print(f'{backend:14s} {overlap:.6f}                      {exact:.6f}')

check('8-4 每个后端都和评测函数完全一致',
      all(overlap == 1.0 and exact == 1.0 for overlap, exact in agreement.values()), True)
check('8-4 至少跑了一种后端', len(agreement) >= 1, True)

# 故意破坏检索层：把 Top-1 沉到第 5 位，看这条对照能不能发现
baseline = recommend_top_k(model, retrieval_users, item_vectors.shape[0], seen_items, k=20)
baseline_rows = [baseline[user_idx] for user_idx in retrieval_users]


def degraded_search(query_vectors, k, user_indices=None, candidate_pool=10):
    '''一个"写坏了"的检索层：把第一名的结果挪到第 5 位。'''
    rows = brute_retriever.search(query_vectors, k=k, user_indices=user_indices)
    return [row[1:5] + [row[0]] + row[5:] if len(row) == k else row for row in rows]


degraded_rows = degraded_search(query_vectors, k=20, user_indices=retrieval_users)
degraded_overlap, degraded_exact = top_k_agreement(baseline_rows, degraded_rows, k=20)
print()
print(f'故意写坏之后：重合率 {degraded_overlap:.6f}，顺序一致率 {degraded_exact:.6f}')
check('8-4 写坏的检索层会被这条对照发现（顺序一致率 < 1）',
      degraded_exact < 1.0, True)
check('8-4 但 Top-20 的重合率仍是 1（所以只看重合率抓不住这类错）',
      degraded_overlap, 1.0)"""
    )
)

CELLS.append(
    markdown(
        lab_answer_for="8-compare",
        source="""<details>
<summary>💡 参考答案（<code>compare_with_eval_tool</code>）</summary>

```python
def compare_with_eval_tool(model, retrievers, item_vectors, users, seen_items, k=20):
    '''把检索层的结果和 myrec.eval.recommend_top_k 对照。'''
    baseline = recommend_top_k(model, users, item_vectors.shape[0], seen_items, k=k)
    baseline_rows = [baseline[user_idx] for user_idx in users]

    agreement = {}
    for backend, retriever in retrievers.items():
        with torch.no_grad():
            queries = model.encode_users(torch.tensor(users))
        rows = retriever.search(queries, k=k, user_indices=users)
        agreement[backend] = top_k_agreement(baseline_rows, rows, k=k)
    return agreement
```

</details>

`baseline_rows = [baseline[user_idx] for user_idx in users]` 这一步不能省：`recommend_top_k` 返回的是字典，而检索层返回的是列表，顺序必须按 `users` 对齐再比。字典的迭代顺序在 Python 3.7+ 是插入顺序，看着"也是按 users 来的"——但那是实现细节，写显式的对齐更稳。

最后那个"故意错位一位"的例子是这一章的收尾检查：**它制造了一个不会报错、看起来完全正常、而且指标只是略差一点**的错误。这类错误正是"两条路互相验证"能抓住的东西。"""
    )
)

# ---------------------------------------------------------------- 13
CELLS.append(
    markdown(
        """## 这一章的几个坑

1. **把未归一化的向量喂给 `IndexFlatIP`**：它不报错，只是返回"内积最大"而不是"余弦最大"。
2. **把 FAISS 的位置下标当成 `item_idx`**：插入顺序一变就全错，而且结果都是合法物品，看不出异常。
3. **只取 `k` 个候选就不管了**：已看物品占掉位置之后凑不够，要么返回不足 k 个，要么补上已看物品。
4. **以为用了向量库指标会变好**：`IndexFlatIP` 和暴力内积是同一个算法，指标必然相同。
5. **拿小规模耗时外推**：3533 个物品上两者同级，FAISS 的收益要在"候选数远大于内存/带宽"时才出现。
6. **测速只跑一次**：第一次调用包含一次性分配，取中位数更稳。

## 文献

- **FAISS**：Johnson, Douze & Jégou, *Billion-scale Similarity Search with GPUs*, IEEE Trans. Big Data 2019（[arXiv](https://arxiv.org/abs/1702.08734)）——FAISS 的原始论文，`IndexFlat` / `IndexIVF` 这些索引类型的定义和设计动机都在这里；本 lab 的 `IndexFlatIP` 属于其中的精确索引。
- **HNSW**：Malkov & Yashunin, *Efficient and Robust Approximate Nearest Neighbor Search using Hierarchical Navigable Small World Graphs*, IEEE TPAMI 2018（[arXiv](https://arxiv.org/abs/1603.09320)）——近似索引里最常用的一类，用来对照"精确 vs 近似"的取舍。
- **ScaNN**：Guo et al., *Accelerating Large-Scale Inference with Anisotropic Vector Quantization*, ICML 2020（[arXiv](https://arxiv.org/abs/1908.10396)）——把量化误差按方向加权，专门优化内积检索（也就是 MIPS）；可以看作"精确索引之后还能怎么省"的下一个方向。

## 动手做

- `labs/lab8-faiss-retrieval.ipynb` 概念 1 ~ 7
- 对应自测：`8-1 未归一化的物品向量会被拒绝`、`8-3 Top-20 重合率是 1.0`、`8-4 索引错位会被这条对照发现`
- `artifacts/results.json` 的 `faiss_index_flat_ip` 一节里有同一组对照和耗时
"""
    )
)

# ---------------------------------------------------------------- 14
CELLS.append(
    code(
        """# 💾 导出这一章写的东西
# 先导出辅助函数：retrieval.py 里要用它（导出时按名字列出，不靠隐式依赖）
labkit.harvest(
    'myrec/retrieval_helpers.py',
    to_float32_numpy,
    imports=['import numpy as np', 'import torch'],
)
labkit.harvest(
    'myrec/retrieval.py',
    check_l2_normalized,
    ItemRetriever,
    top_k_agreement,
    benchmark_retrieval,
    imports=[
        'import time',
        'import numpy as np',
        'from myrec.retrieval_helpers import to_float32_numpy',
    ],
)

import importlib

import myrec.retrieval as my_retrieval

importlib.reload(my_retrieval)

check('8-5 交接成功：myrec.retrieval 能用',
      hasattr(my_retrieval, 'ItemRetriever') and hasattr(my_retrieval, 'check_l2_normalized'),
      True)

# 光能 import 不够：真的建一个检索层、搜一次
exported_retriever = my_retrieval.ItemRetriever(toy_items, {0: {1}}, backend='brute_force')
exported_rows = exported_retriever.search(queries, k=2, user_indices=[0, 1])
check('8-5 导出的 ItemRetriever 能检索',
      [len(items) for items in exported_rows], [2, 2])

labkit.record('lab8_top_k_overlap', 1.0 if HAS_FAISS else None)
print('已记录 lab8 的两条路一致性（没装 faiss 时这个值是 None）')"""
    )
)

# ---------------------------------------------------------------- 15
CELLS.append(
    markdown(
        """## 收尾

### 这个 lab 写了多少

| 自己写的 | 行数 | 脚手架给的 |
|---|---|---|
| `check_l2_normalized` | 5 | 数据、Two-Tower 的训练与评测 |
| `ItemRetriever` | 45 | `to_float32_numpy` |
| `top_k_agreement` / `benchmark_retrieval` | 20 | Lab 1 ~ 3 的模型和 `recommend_top_k` |
| `compare_with_eval_tool` | 12 | faiss 环境探测 |

加起来 82 行左右。真正新写的机制只有两个：**"多取候选再补齐"的循环**，和**位置到 item_idx 的映射**。

### 这一章应该能直接答上来

1. 为什么归一化检查要抛错，而不是自动归一化？
2. `IndexFlatIP` 和暴力内积是什么关系？为什么重合率必然是 1.0？
3. FAISS 返回的位置下标为什么不能直接当 `item_idx`？
4. 为什么 FAISS 后端要"多取候选再补齐"，暴力后端不用？
5. 什么规模下才值得换成 ANN 索引？换的时候要接受什么代价？

### 再往下的方向

到这里，仓库里能跑的东西都跑过了：两个 ID-only 基线、Two-Tower、SASRec，以及一层精确检索。

再往下通常有三个方向，每一个都需要先改数据而不是先改模型：

- **换数据集**：KuaiRec 的短视频交互（`labs/lab9` 先把标签规则定死，`docs/04-kuairec` 有完整数据与协议）。
- **加特征**：现在只有 ID。加上类别、文本或上下文之后，塔的输入不再是一张纯查表，MLP 才会开始有用（`docs/01-two-tower/05` 讨论过这个边界）。
- **换目标**：多正样本、listwise、或者把 SASRec 的逐位置目标换成挖空预测（`docs/03-sasrec/03` 里讨论过双向的代价）。"""
    )
)

NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.7"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}


def main() -> None:
    target = REPO_ROOT / "lab8-faiss-retrieval.ipynb"
    target.write_text(
        json.dumps(NOTEBOOK, ensure_ascii=False, indent=4) + "\n", encoding="utf-8"
    )
    code_cells = [cell for cell in CELLS if cell["cell_type"] == "code"]
    todo_cells = [cell for cell in code_cells if "lab_todo" in cell["metadata"]]
    answers = [
        cell
        for cell in CELLS
        if cell["cell_type"] == "markdown" and "lab_answer_for" in cell["metadata"]
    ]
    print(
        f"{target.name}: {len(CELLS)} cells, {len(code_cells)} code cells, "
        f"{len(todo_cells)} TODO, {len(answers)} 个答案块"
    )
    print("TODO:", [cell["metadata"]["lab_todo"] for cell in todo_cells])
    print("答案:", [cell["metadata"]["lab_answer_for"] for cell in answers])


if __name__ == "__main__":
    main()
