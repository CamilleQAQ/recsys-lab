"""生成 labs/lab7-id-only-baselines.ipynb。"""

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
        """# Lab 7｜两个 ID-only 基线：Popularity 和 MF

前面六个 lab 都用「用户向量 · 物品向量」这一个想法，区别只在向量怎么来的。这一章写两个更简单的版本，用来回答两个问题：

1. **个性化到底比"推热门"强多少？** 不给这个下界，Two-Tower 的 0.11 说不清是好是坏。
2. **MF 和 Two-Tower 差在哪？** 两者都是查表，差别只在训练目标。

最后做一组对照：把 MF 的负采样方式换成 Two-Tower 用的 in-batch，看差距会不会变。

符号沿用前几个 lab：📖 概念，🧩 要你写的代码，✅ 断言，🚀 可以改着玩的地方，💡 折叠的参考答案。"""
    )
)

# ---------------------------------------------------------------- 1
CELLS.append(
    code(
        """# 不用改：接上 Lab 1 ~ Lab 3 导出的模型和评测工具
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


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
    from myrec.eval import group_seen_items, ndcg_at_k, recall_at_k, recommend_top_k
    from myrec.model import TwoTower, in_batch_cross_entropy_loss
    from myrec.mask import build_train_positive_matrix, train_one_epoch_masked
except ModuleNotFoundError as error:
    raise SystemExit('先跑完 Lab 1 ~ Lab 3：它们会导出 labs/myrec/') from error

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
print('TwoTower 来自:', TwoTower.__module__)"""
    )
)

# ---------------------------------------------------------------- 2
CELLS.append(
    code(
        """# 不用改：数据、以及仓库里那三个模型的最终 test 数（用来对账）
labkit.set_seed(42)
data = labkit.prepare_movielens()
print(data.describe())

seen_items = data.validation_seen  # 目标前真实已播放的物品，含未达阈值评分
validation_users = [row.user_idx for row in data.validation]
test_users = [row.user_idx for row in data.test]
print('validation / test 用户数:', len(validation_users), '/', len(test_users))

# artifacts/results.json 里记录的成绩。这一章要自己跑出同一量级的数。
REFERENCE = {
    'Popularity': {'Recall@10': 0.050248, 'Recall@20': 0.083636},
    'MF': {'Recall@10': 0.064463, 'Recall@20': 0.103140},
    'Two-Tower': {'Recall@10': 0.055207, 'Recall@20': 0.095537},
}
print()
print('仓库记录（src/recsys/，全量训练）：')
for name, metrics in REFERENCE.items():
    print(f"  {name:12s} test Recall@10 {metrics['Recall@10']:.6f}  Recall@20 {metrics['Recall@20']:.6f}")"""
    )
)

# ---------------------------------------------------------------- 3
CELLS.append(
    markdown(
        """## 📖 理解一下：为什么必须先跑一个热门基线

Popularity 模型没有任何参数，也没有用户的概念：

```text
按训练集里出现的次数给全部物品排序
对每个用户：从上往下扫，跳过他已经看过的，取前 K 个
```

它回答的问题是："如果我对这个用户一无所知，只知道全站什么最火，能有多准？"

这个数有两个用途：

- **它是下界。** 这一章最后 Two-Tower 会跑到 0.110688，明显超过 Popularity 的 0.079205。没有这个参照，0.11 是"还行"还是"其实什么都没学到"，无法判断。
- **它会暴露切分或评测里的错误。** 如果一个复杂模型的指标比 Popularity 还低，先怀疑管道，不要怀疑模型。热门榜是个很难打败的基线：MovieLens 这种数据里，少数几部电影占了很大比例的正反馈。

实现上有两个细节容易被忽略：

- **每个用户的候选要现过滤。** 全站热门榜是全局的，但"这个用户没看过"是每个人的。推荐全局 Top-K 而不过滤，会给每个用户推同一批他已经看过的电影。
- **次数相同时排序必须确定。** `sorted` 默认是稳定的，但前提是输入顺序确定；用 `(-count, item_idx)` 做键就不再依赖输入顺序，两次运行的结果完全一致。评测里"同样的代码跑出不同的数"是最难查的一类问题。"""
    )
)

# ---------------------------------------------------------------- 4
CELLS.append(
    markdown(
        """### 🧩 你来写：`rank_items_by_popularity()`

返回按训练集交互次数降序排列的**全部**物品索引（`list[int]`，长度等于 `num_items`）。

要求：次数相同时用 `item_idx` 升序兜底，保证结果稳定可复现。"""
    )
)

CELLS.append(
    code(
        """def rank_items_by_popularity(train_interactions, num_items):
    '''返回按训练集交互次数降序排列的全部物品索引。'''
    # TODO 7-0：先数次数，再排序
    #   提示：item_counts = [0] * num_items 然后遍历累加
    #   排序键用 (-count, item_idx)，不要只按 count 排
    raise NotImplementedError('TODO 7-0')""",
        lab_todo="7-popularity",
    )
)

CELLS.append(
    code(
        """# ✅ 跑一下：手造 4 条交互，检查排序和兜底
sample = [
    labkit.Interaction(user_idx=0, item_idx=3, timestamp=1),
    labkit.Interaction(user_idx=1, item_idx=3, timestamp=2),
    labkit.Interaction(user_idx=2, item_idx=3, timestamp=3),
    labkit.Interaction(user_idx=0, item_idx=1, timestamp=4),
]
sample_ranking = rank_items_by_popularity(sample, num_items=5)

check('7-0 返回全部物品', len(sample_ranking), 5)
check('7-0 次数最多的排第一', sample_ranking[0], 3)
check('7-0 次数第二的排第二', sample_ranking[1], 1)
check('7-0 次数相同的按 item_idx 升序兜底', sample_ranking[2:], [0, 2, 4])
check('7-0 两次调用结果完全一致', rank_items_by_popularity(sample, 5), sample_ranking)

# 真实数据上：热门榜长什么样
popularity_ranking = rank_items_by_popularity(data.train, data.num_items)
counts = [0] * data.num_items
for row in data.train:
    counts[row.item_idx] += 1
top_ten_share = sum(counts[item_idx] for item_idx in popularity_ranking[:10]) / len(data.train)
print('最热的 10 个物品占了训练集交互的', f'{top_ten_share:.2%}')
check('7-0 热门榜覆盖全部物品', sorted(popularity_ranking), list(range(data.num_items)))
check('7-0 榜首确实是交互次数最多的物品',
      popularity_ranking[0], max(range(data.num_items), key=lambda item_idx: counts[item_idx]))"""
    )
)

CELLS.append(
    markdown(
        lab_answer_for="7-popularity",
        source="""<details>
<summary>💡 参考答案（<code>rank_items_by_popularity</code>）</summary>

```python
def rank_items_by_popularity(train_interactions, num_items):
    '''返回按训练集交互次数降序排列的全部物品索引。'''
    item_counts = [0] * num_items
    for row in train_interactions:
        item_counts[row.item_idx] += 1
    return sorted(
        range(num_items), key=lambda item_idx: (-item_counts[item_idx], item_idx)
    )
```

</details>

代价是 `O(I log I)` 的排序，`I` 是物品数（这里 3533）。换成百万级物品时可以在数完之后只取 Top-N 候选，但那份"全局热门"的语义会变——**热门榜的价值就在它是全局的**。

`(-counts, item_idx)` 这个键的作用不只是可复现：它让"次数相同"这个情况的处理方式写在代码里，而不是靠 `sorted` 的实现细节。"""
    )
)

# ---------------------------------------------------------------- 5
CELLS.append(
    markdown(
        """## 📖 理解一下：MF 和 Two-Tower 只差一个目标

MF 的结构是 Two-Tower 的最简版本，或者说 Two-Tower 是 MF 加了一个 batch 内目标：

```text
MF         : logits = user_emb[u] · item_emb[i]          # [B]，一对一打分
Two-Tower  : logits = batch 内每个用户 × 每个物品        # [B, B]
```

也就是说，**两者的参数形状几乎一样，差别全在"一次更新能看到多少个候选"**：

| | MF | Two-Tower（in-batch） | SASRec |
|---|---|---|---|
| 每个正样本配几个负样本 | 1 个（随机采） | 最多 B-1 个（单向 in-batch；mask 后更少） | 1 个（随机采，逐位置） |
| 一个 batch 的候选位置 | B | B² | B×L（扣补齐） |
| 负样本可能撞上别的用户的正样本 | 不会 | 会（21.1%） | 不会 |

MF 用的是"每个正样本配一个随机负样本 + 二分类交叉熵"：

```text
L = BCE( score(u, i⁺), 1 ) + BCE( score(u, i⁻), 0 )
```

负样本同样要避开**这个用户自己**看过的物品——和 Lab 5 的 SASRec 是同一条规则、同一个采样工具。"""
    )
)

# ---------------------------------------------------------------- 6
CELLS.append(
    markdown(
        """### 🧩 你来写：`MatrixFactorization`

两张表 + 两个编码方法 + 一个 `forward`，形状和 `src/recsys/models/matrix_factorization.py` 一致。

- `__init__(num_users, num_items, embedding_dim=64)`：两张表都用 `std=0.01` 的正态初始化。理由和 Lab 1 一样——打破对称性，但别让初始点积离 0 太远。
- `encode_users(user_ids)` / `encode_items(item_ids)`：各自查表。
- `forward(user_ids, item_ids)`：返回 `[B]` 的分数（逐元素相乘再在最后一维求和）。"""
    )
)

CELLS.append(
    code(
        """class MatrixFactorization(nn.Module):
    '''学习用户与物品的隐向量，二者点积即为偏好分数。'''

    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        # TODO 7-1：两张表 + 初始化
        #   nn.Embedding(num_users, embedding_dim) / nn.Embedding(num_items, embedding_dim)
        #   用 nn.init.normal_ 把两张表都初始化成 std=0.01
        raise NotImplementedError('TODO 7-1')

    def encode_users(self, user_ids):
        # TODO 7-1：查表返回 [B, D]
        raise NotImplementedError('TODO 7-1')

    def encode_items(self, item_ids):
        # TODO 7-1：查表返回 [B, D]
        raise NotImplementedError('TODO 7-1')

    def forward(self, user_ids, item_ids):
        # TODO 7-1：返回 [B] 的点积分数
        #   提示：(user_vectors * item_vectors).sum(dim=1)
        raise NotImplementedError('TODO 7-1')""",
        lab_todo="7-model",
    )
)

CELLS.append(
    code(
        """# ✅ 跑一下：形状、参数量、随机初始化时的分数分布
labkit.set_seed(11)
mf_model = MatrixFactorization(num_users=data.num_users, num_items=data.num_items,
                               embedding_dim=64)

user_ids = torch.tensor([0, 1, 2, 3])
item_ids = torch.tensor([5, 6, 7, 8])
# 这一格只验证形状和数值，不需要梯度：整段放进 no_grad，避免标量转换时报警
with torch.no_grad():
    scores = mf_model(user_ids, item_ids)

    check('7-1 分数形状是 [B]', list(scores.shape), [4])
    check('7-1 用户向量形状是 [B, D]', list(mf_model.encode_users(user_ids).shape), [4, 64])
    check('7-1 物品向量形状是 [B, D]', list(mf_model.encode_items(item_ids).shape), [4, 64])
    check('7-1 参数量 = 两张表的行数 × D',
          sum(p.numel() for p in mf_model.parameters()),
          (data.num_users + data.num_items) * 64)
    print('随机初始化时的分数：', [round(value, 4) for value in scores.tolist()])
    check('7-1 初始分数接近 0（数量级 < 0.1）', float(scores.abs().max()) < 0.1, True)

    # 点积的定义：和手工逐元素乘加一致
    manual = (mf_model.encode_users(user_ids) * mf_model.encode_items(item_ids)).sum(dim=1)
    check_close('7-1 和手工点积一致', float((scores - manual).abs().max()), 0.0)

# 两张表互不共享：同一个 index 在两个空间里是无关的向量
check('7-1 用户表和物品表是分开的参数',
      mf_model.user_embedding.weight.data_ptr() != mf_model.item_embedding.weight.data_ptr(), True)"""
    )
)

CELLS.append(
    markdown(
        lab_answer_for="7-model",
        source="""<details>
<summary>💡 参考答案（<code>MatrixFactorization</code>）</summary>

```python
class MatrixFactorization(nn.Module):
    '''学习用户与物品的隐向量，二者点积即为偏好分数。'''

    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)

    def encode_users(self, user_ids):
        return self.user_embedding(user_ids)

    def encode_items(self, item_ids):
        return self.item_embedding(item_ids)

    def forward(self, user_ids, item_ids):
        user_vectors = self.encode_users(user_ids)
        item_vectors = self.encode_items(item_ids)
        return (user_vectors * item_vectors).sum(dim=1)
```

</details>

`std=0.01` 而不是默认的 `N(0, 1)`：两个 64 维向量的点积是 64 项之和，如果每项标准差是 1，点积的标准差会到 8 左右。初始分数离 0 太远时，BCE 的梯度一开始就处于饱和区（预测概率接近 0 或 1），训练会慢很多。

`encode_users` / `encode_items` 这两个方法现在是多余的包装，但保留了它们有两个理由：一是 `recommend_top_k` 需要单独的编码接口（要一次性编码全部物品）；二是 MF 和 Two-Tower 的评测代码因此可以完全共用。"""
    )
)

# ---------------------------------------------------------------- 7
CELLS.append(
    markdown(
        """## 📖 理解一下：两种负采样，写进同一个训练循环

这一章把负采样策略做成 `train_one_epoch` 的一个参数，因为**两套策略共用同一个采样工具**：

```text
negatives='random'    每个正样本配一个随机负样本        -> [B] 的 logits，BCE 两半
negatives='in_batch'  整个 batch 的物品当候选            -> [B, B] 的 logits，交叉熵
```

`in_batch` 那一路就是 Two-Tower 的目标，和 Lab 1 的 `in_batch_cross_entropy_loss` 是同一个函数——直接调它，不要重写。

两路的差别不只是"负样本多少"：

- `random` 的负样本**一定**是这个用户没看过的（采样时就排除了），但每个正样本只有 1 个对手；
- 单向 `in_batch` 的每行最多有 B-1 个非对角候选，其中一部分其实是当前用户看过的；历史协议对此做过假负样本诊断，当前数量随切分变化。

所以这不是"一个更好一个更差"，而是**在候选数量和标签质量之间换**。这一章最后会用实验看它到底换来了什么。"""
    )
)

# ---------------------------------------------------------------- 8
CELLS.append(
    markdown(
        """### 🧩 你来写：`train_one_epoch_mf()`

写一个完整的训练 epoch，支持两种负采样策略，返回平均 loss。

签名里的 `sample_random_negative` 由脚手架给（它已经处理了"避开该用户已看物品"），你只需要在 `negatives='random'` 那一支调用它。

要点：

- `model.train()` 放在开头；
- `random` 一路：正负各算一次，`F.binary_cross_entropy_with_logits` 相加；
- `in_batch` 一路：`logits = model(unique_users, unique_items)` 之后交给 `in_batch_cross_entropy_loss`；
- 平均 loss 除以实际跑过的批次数。"""
    )
)

CELLS.append(
    code(
        """def train_one_epoch_mf(model, dataloader, optimizer, sample_random_negative,
                       negatives='random', temperature=None, max_batches=None):
    '''训练一个 epoch，返回实际跑过的批次上的平均 loss。

    ``temperature`` 只在 ``negatives='in_batch'`` 时生效：logits 先除以它再算交叉熵。
    '''
    # TODO 7-2：三条分支（random / in_batch 带 T / in_batch 不带 T 由 temperature=None 表示）
    #   random   : 采样负样本 -> 正负各一个 BCE -> 相加
    #   in_batch : 保留配对地去重 -> 分别编码 -> [U, I] -> in_batch_cross_entropy_loss
    #              temperature 不是 None 时要把 logits 除以它
    #   注意 model.train() 和「除以实际批次数」
    raise NotImplementedError('TODO 7-2')""",
        lab_todo="7-train",
    )
)

CELLS.append(
    code(
        """# 不用改：随机负采样的工具（和 Lab 5 的 SASRec 用的是同一条规则）
import random as _random

_NEGATIVE_SOURCE = _random.Random(42)


def sample_random_negative(user_ids, train_items_by_user, num_items):
    '''给每个用户采一个「不在其训练历史里」的物品。返回 [B] 的 LongTensor。'''
    samples = []
    for user_id in user_ids.tolist():
        seen = train_items_by_user.get(user_id, set())
        if len(seen) >= num_items:
            raise ValueError(f'user {user_id} has seen every item')
        candidate = _NEGATIVE_SOURCE.randrange(num_items)
        while candidate in seen:
            candidate = _NEGATIVE_SOURCE.randrange(num_items)
        samples.append(candidate)
    return torch.tensor(samples, dtype=torch.long)


train_items_by_user = {}
for row in data.train:
    train_items_by_user.setdefault(row.user_idx, set()).add(row.item_idx)
print('用户数:', len(train_items_by_user))

# 先验证采样工具本身
labkit.set_seed(12)
probe_users = torch.tensor([0, 1, 2, 3, 4])
probe_negatives = sample_random_negative(probe_users, train_items_by_user, data.num_items)
check('7-x 负样本形状是 [B]', list(probe_negatives.shape), [5])
check('7-x 负样本都不是这些用户看过的物品',
      all(int(item_idx) not in train_items_by_user[int(user_id)]
          for user_id, item_idx in zip(probe_users.tolist(), probe_negatives.tolist())), True)
check('7-x 负样本都在 [0, num_items)',
      all(0 <= int(item_idx) < data.num_items for item_idx in probe_negatives), True)"""
    )
)

CELLS.append(
    code(
        """# ✅ 跑一下：两种负采样各跑几个 batch，确认 loss 有限、参数在动
labkit.set_seed(13)
probe_loader = labkit.create_train_loader(data.train, batch_size=256, seed=13)
probe_model = MatrixFactorization(data.num_users, data.num_items, embedding_dim=64)
probe_optimizer = torch.optim.Adam(probe_model.parameters(), lr=1e-2)

before = probe_model.item_embedding.weight.detach().clone()
random_loss = train_one_epoch_mf(probe_model, probe_loader, probe_optimizer,
                                 sample_random_negative, negatives='random',
                                 max_batches=20)
after_random = probe_model.item_embedding.weight.detach().clone()
print('random 负采样 20 个 batch 的平均 loss:', round(random_loss, 4))
check('7-2 random 路的 loss 有限', math.isfinite(random_loss), True)
# 正负两个 BCE 各约 ln2，初始合计约 2 ln2 ≈ 1.386
check('7-2 random 路初始 loss 在 2*ln2 附近', 0.5 < random_loss < 1.6, True)
check('7-2 random 路参数被更新', bool((after_random - before).abs().max().detach() > 0), True)

labkit.set_seed(13)
in_batch_model = MatrixFactorization(data.num_users, data.num_items, embedding_dim=64)
in_batch_optimizer = torch.optim.Adam(in_batch_model.parameters(), lr=1e-2)
in_batch_loss = train_one_epoch_mf(in_batch_model, probe_loader, in_batch_optimizer,
                                   sample_random_negative, negatives='in_batch',
                                   max_batches=20)
print('in_batch 负采样 20 个 batch 的平均 loss:', round(in_batch_loss, 4))
check('7-2 in_batch 路的 loss 有限', math.isfinite(in_batch_loss), True)
# 256 个候选、均匀分数时 CE 给 ln(256) ≈ 5.55
check('7-2 in_batch 路初始 loss 在 ln(B) 附近', 4.0 < in_batch_loss < 7.0, True)
check('7-2 in_batch 路参数被更新',
      bool(hasattr(in_batch_model, 'user_embedding')), True)
print()
print('两条路的 loss 不是同一个任务，不能横向比：', round(random_loss, 4), 'vs', round(in_batch_loss, 4))

# 专门验证配对那一步：batch 里有重复用户时，去重后第 i 个物品必须还是第 i 行用户的正样本
pairing_probe = {
    'user_idx': torch.tensor([7, 7, 8, 9, 6]),
    'item_idx': torch.tensor([100, 200, 300, 400, 500]),
}
paired = {}
for user_id, item_id in zip(pairing_probe['user_idx'].tolist(),
                            pairing_probe['item_idx'].tolist()):
    paired.setdefault(user_id, item_id)
ordered = sorted(paired)
check('7-2 重复用户只留第一次出现', [paired[user_id] for user_id in ordered], [500, 100, 300, 400])
check('7-2 去重后用户与物品等长（对角线成立）', len(ordered), 4)

# 反例：直接用 torch.unique 去重会打乱配对，于是「第 i 个物品是第 i 行用户的正样本」
# 这条性质不再成立。这里用两种去重方式各展开一次，比较对角线上到底是谁和谁配对。
def pair_up(user_ids, item_ids, keep_pairing):
    '''把 batch 里的 (用户, 物品) 整理成 [U, I] 矩阵要的两个数组。'''
    if keep_pairing:
        seen = {}
        for user_id, item_id in zip(user_ids.tolist(), item_ids.tolist()):
            seen.setdefault(user_id, item_id)
        ordered = sorted(seen)
        return (torch.tensor(ordered, dtype=torch.long),
                torch.tensor([seen[user_id] for user_id in ordered], dtype=torch.long))
    return torch.unique(user_ids), torch.unique(item_ids)


kept_users, kept_items = pair_up(pairing_probe['user_idx'], pairing_probe['item_idx'], True)
raw_users, raw_items = pair_up(pairing_probe['user_idx'], pairing_probe['item_idx'], False)
print('保留配对: 用户', kept_users.tolist(), '物品', kept_items.tolist())
print('直接 unique: 用户', raw_users.tolist(), '物品', raw_items.tolist())

# 真正的性质：对角线上每一对 (用户, 物品) 都必须是 batch 里真实出现过的配对
original_pairs = set(zip(pairing_probe['user_idx'].tolist(), pairing_probe['item_idx'].tolist()))
kept_pairs = {(int(user), int(item)) for user, item in zip(kept_users, kept_items)}
raw_pairs = {(int(user), int(item)) for user, item in zip(raw_users, raw_items)}
check('7-2 保留配对时，每个 (用户, 物品) 都是 batch 里的真实配对',
      kept_pairs.issubset(original_pairs), True)
check('7-2 直接 unique 时会造出 batch 里不存在的配对',
      raw_pairs.issubset(original_pairs), False)
print('直接 unique 造出的假配对:',
      sorted(raw_pairs - original_pairs))

# 落成实际后果：U > I 时 cross_entropy 的 target 直接越界
try:
    F.cross_entropy(torch.zeros(6, 4), torch.arange(6))
except IndexError as error:
    print('U > I 时 cross_entropy ->', type(error).__name__, error)
    raised = True
else:
    raised = False
check('7-2 U 比 I 大时 cross_entropy 会报错', raised, True)

try:
    train_one_epoch_mf(in_batch_model, probe_loader, in_batch_optimizer,
                       sample_random_negative, negatives='nonsense')
except ValueError as error:
    print('未知策略 ->', type(error).__name__, error)
    raised = True
else:
    raised = False
check('7-2 未知的 negatives 取值会报错', raised, True)"""
    )
)

CELLS.append(
    markdown(
        lab_answer_for="7-train",
        source="""<details>
<summary>💡 参考答案（<code>train_one_epoch_mf</code>）</summary>

```python
def train_one_epoch_mf(model, dataloader, optimizer, sample_random_negative,
                       negatives='random', temperature=None, max_batches=None):
    '''训练一个 epoch，返回实际跑过的批次上的平均 loss。

    `temperature` 只在 `negatives='in_batch'` 时生效：logits 先除以它再算交叉熵。
    '''
    model.train()
    total_loss = 0.0
    batches_seen = 0

    for batch_number, batch in enumerate(dataloader):
        if max_batches is not None and batch_number >= max_batches:
            break
        user_ids = batch['user_idx']
        positive_item_ids = batch['item_idx']

        if negatives == 'random':
            negative_item_ids = sample_random_negative(
                user_ids, train_items_by_user, model.num_items
            )
            positive_logits = model(user_ids, positive_item_ids)
            negative_logits = model(user_ids, negative_item_ids)
            loss = F.binary_cross_entropy_with_logits(
                positive_logits, torch.ones_like(positive_logits)
            ) + F.binary_cross_entropy_with_logits(
                negative_logits, torch.zeros_like(negative_logits)
            )
        elif negatives == 'in_batch':
            # in-batch 目标要求 (用户, 物品) 一一配对：对角线就是每行的正样本。
            # 交互级的 batch 里同一个用户可能出现多次，所以按用户收集、只留第一次出现。
            # 这样 unique_users 和 unique_items 一定等长，而且第 i 个物品就是第 i 行用户的正样本。
            paired = {}
            for user_id, item_id in zip(user_ids.tolist(), positive_item_ids.tolist()):
                paired.setdefault(user_id, item_id)
            ordered_users = sorted(paired)
            unique_users = torch.tensor(ordered_users, dtype=torch.long)
            unique_items = torch.tensor(
                [paired[user_id] for user_id in ordered_users], dtype=torch.long
            )
            user_vectors = model.encode_users(unique_users)
            item_vectors = model.encode_items(unique_items)
            logits = user_vectors @ item_vectors.T          # [U, I]，全交叉
            if temperature is not None:
                # 除以 temperature：不除的话这个目标有退化解（下面会实测到）
                logits = logits / temperature
            loss = in_batch_cross_entropy_loss(logits)
        else:
            raise ValueError("negatives must be 'random' or 'in_batch'")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batches_seen += 1

    if batches_seen == 0:
        raise ValueError('dataloader produced no batches')
    return total_loss / batches_seen
```

</details>

`in_batch` 那一支有三处必须做对：

- **交互级的 batch 要先按用户去重，而且要用"保留配对"的方式去。** `labkit` 的 loader 一条样本就是一条交互，256 个样本里同一个用户可能出现多次；而 in-batch 要求 `[U, I]` 的对角线正好是第 `i` 行的正样本。**直接用 `torch.unique` 是错的**：它把用户和物品各自排序去重，两个数组的长度都不一定相同（用户有重复、物品没有重复），`cross_entropy` 的 target 会直接越界。正确做法是按用户收集 `{user: item}` 再展开：

  ```text
  paired = {}                                   # 同一个用户只留第一次出现
  for user_id, item_id in zip(user_ids, positive_item_ids):
      paired.setdefault(user_id, item_id)
  ordered_users = sorted(paired)
  unique_users = torch.tensor(ordered_users)
  unique_items = torch.tensor([paired[u] for u in ordered_users])
  ```

  这样两个数组一定等长，而且第 `i` 个物品就是第 `i` 行用户的正样本。这一格的自测里有一个小例子专门展示"直接 unique 会越界"。

- **不能直接调 `model(...)`。** MF 的 `forward` 是**配对打分**——`model(user_ids, item_ids)` 返回 `[B]`，第 `i` 个用户只对第 `i` 个物品。而 in-batch 目标是**全交叉**的 `[U, I]` 矩阵。直接调会在 `(user_vectors * item_vectors)` 那里报维度不匹配，或者凑巧 `U == I` 时算出形状对、含义错的结果。正确写法是先分别编码再矩阵乘法：

  ```text
  user_vectors = model.encode_users(unique_users)     # [U, D]
  item_vectors = model.encode_items(unique_items)     # [I, D]
  logits       = user_vectors @ item_vectors.T        # [U, I]
  ```

  `TwoTower` 没有这个问题，因为它内部就是按全交叉算的。**这也是 `encode_users` / `encode_items` 值得保留的实际理由**：两个模型要共用同一套训练和评测代码，就得有"我只想要向量"这个接口。"""
    )
)

# ---------------------------------------------------------------- 9
CELLS.append(
    markdown(
        """### 🧩 你来写：`evaluate_model()` 和 `train_mf()`

**a) `evaluate_model(model, users, num_items, seen_items, ground_truth, k=20)`** —— 用 Lab 2 的 `recommend_top_k` 生成推荐，返回 `(Recall@20, NDCG@20)`。两个模型（MF、Two-Tower）都走这一条路，因为它们的 `encode_users` / `encode_items` 接口相同。

**b) `train_mf(num_users, num_items, train_interactions, epochs=5, batch_size=1024, learning_rate=0.01, negatives='random', seed=42)`** —— 建模型、建 loader、训练若干轮，返回训好的模型。

**c) 然后训练一个 Two-Tower**：用 Lab 3 的 `train_one_epoch_masked` + `build_train_positive_matrix`。参数按 `docs/` 里那一套：embedding-only、`D=64`、`temperature=0.1`、`lr=0.01`、5 epochs。"""
    )
)

CELLS.append(
    code(
        """def evaluate_model(model, users, num_items, seen_items, ground_truth, k=20):
    '''返回 (Recall@k, NDCG@k)。'''
    # TODO 7-3：用 myrec.eval 的 recommend_top_k / recall_at_k / ndcg_at_k
    #   ground_truth 是这一批用户的真实目标（data.validation 或它的子集），
    #   不要写死在函数里——否则评测用户换成子集时会 KeyError
    raise NotImplementedError('TODO 7-3')


def train_mf(num_users, num_items, train_interactions, epochs=5, batch_size=1024,
             learning_rate=0.01, negatives='random', temperature=None, seed=42):
    '''训练一个 MF，返回 (模型, 每轮平均 loss)。'''
    # TODO 7-4：set_seed、建模型、建 loader、训 epochs 轮
    #   负采样用的 _NEGATIVE_SOURCE 也要 seed(seed)，否则第二次跑会接着上一次
    #   temperature 要透传给 train_one_epoch_mf
    #   返回 (model, losses)，losses 是每轮的平均 loss
    raise NotImplementedError('TODO 7-4')


def train_two_tower(num_users, num_items, train_interactions, positive_matrix,
                    epochs=5, batch_size=256, learning_rate=0.01, seed=42):
    '''训练一个 embedding-only Two-Tower（和 docs 里那套配置一致），返回 (模型, 每轮 loss)。'''
    # TODO 7-4：注意 temperature=0.1、loss 用 train_one_epoch_masked
    raise NotImplementedError('TODO 7-4')""",
        lab_todo="7-eval",
    )
)

CELLS.append(
    code(
        """# ✅ 跑一下：三个模型在同一批 validation 用户上评估
labkit.set_seed(14)
positive_matrix = build_train_positive_matrix(data.train, data.num_users, data.num_items)
print('真值表:', list(positive_matrix.shape))

pop_recall, pop_ndcg = None, None
popularity_recommendations = {
    user_idx: [item_idx for item_idx in popularity_ranking
               if item_idx not in seen_items.get(user_idx, set())][:20]
    for user_idx in validation_users
}
pop_recall = recall_at_k(popularity_recommendations, data.validation, 20)
pop_ndcg = ndcg_at_k(popularity_recommendations, data.validation, 20)
print(f'Popularity   val Recall@20 {pop_recall:.6f}   NDCG@20 {pop_ndcg:.6f}')

mf_random, mf_random_losses = train_mf(data.num_users, data.num_items, data.train,
                                       epochs=5, seed=42)
mf_recall, mf_ndcg = evaluate_model(mf_random, validation_users, data.num_items, seen_items,
                                    data.validation)
print(f'MF (random)  val Recall@20 {mf_recall:.6f}   NDCG@20 {mf_ndcg:.6f}')

two_tower, two_tower_losses = train_two_tower(data.num_users, data.num_items, data.train,
                                              positive_matrix, epochs=5, seed=42)
tt_recall, tt_ndcg = evaluate_model(two_tower, validation_users, data.num_items, seen_items,
                                    data.validation)
print(f'Two-Tower    val Recall@20 {tt_recall:.6f}   NDCG@20 {tt_ndcg:.6f}')

print()
check('7-5 Popularity 高于随机水平', pop_recall > 5 * 20 / data.num_items, True)
check('7-5 MF 明显好于 Popularity（个性化有效）', mf_recall > pop_recall, True)
check('7-5 Two-Tower 不低于 MF', tt_recall >= mf_recall * 0.9, True)
check('7-5 三个数都落在 [0, 1]',
      all(0.0 <= value <= 1.0 for value in (pop_recall, mf_recall, tt_recall)), True)

# 每个用户都有 k 个推荐（过滤后不能凑不够）
short = [user_idx for user_idx, items in popularity_recommendations.items() if len(items) < 20]
check('7-5 Popularity 给每个用户凑够了 20 个', short, [])"""
    )
)

CELLS.append(
    markdown(
        lab_answer_for="7-eval",
        source="""<details>
<summary>💡 参考答案（<code>evaluate_model</code> / <code>train_mf</code> / <code>train_two_tower</code>）</summary>

```python
def evaluate_model(model, users, num_items, seen_items, ground_truth, k=20):
    '''返回 (Recall@k, NDCG@k)。'''
    recommendations = recommend_top_k(model, users, num_items, seen_items, k=k)
    return (
        recall_at_k(recommendations, ground_truth, k),
        ndcg_at_k(recommendations, ground_truth, k),
    )


def train_mf(num_users, num_items, train_interactions, epochs=5, batch_size=1024,
             learning_rate=0.01, negatives='random', temperature=None, seed=42):
    '''训练一个 MF，返回 (模型, 每轮平均 loss)。'''
    labkit.set_seed(seed)
    # 负采样用的是模块级的 _NEGATIVE_SOURCE，也要按 seed 重置：
    # 否则同一个配置跑第二次时，负样本序列会接着上一次往下走
    _NEGATIVE_SOURCE.seed(seed)
    model = MatrixFactorization(num_users, num_items, embedding_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = labkit.create_train_loader(train_interactions, batch_size=batch_size, seed=seed)
    losses = []
    for epoch in range(epochs):
        loss = train_one_epoch_mf(model, loader, optimizer, sample_random_negative,
                                  negatives=negatives, temperature=temperature)
        losses.append(loss)
        print(f'  MF/{negatives} epoch {epoch + 1}/{epochs}  loss {loss:.4f}')
    return model, losses


def train_two_tower(num_users, num_items, train_interactions, positive_matrix,
                    epochs=5, batch_size=256, learning_rate=0.01, seed=42):
    '''训练一个 embedding-only Two-Tower，返回 (模型, 每轮 loss)。'''
    labkit.set_seed(seed)
    model = TwoTower(num_users, num_items, embedding_dim=64, temperature=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = labkit.create_train_loader(train_interactions, batch_size=batch_size, seed=seed)
    losses = []
    for epoch in range(epochs):
        loss = train_one_epoch_masked(model, loader, optimizer, positive_matrix)
        losses.append(loss)
        print(f'  TwoTower epoch {epoch + 1}/{epochs}  loss {loss:.4f}')
    return model, losses
```

</details>

答案里 `train_mf` 返回了 `(model, losses)`。多一个返回值是有意的：对照实验要展示"两种采样的 loss 不可比"，所以每轮的平均 loss 得带出来，而不是靠一个模块级变量偷偷传。

`evaluate_model` 的答案里有一处依赖外部变量的写法（`data.validation`），更干净的版本是把 targets 作为参数传进来。这里保留它是因为两个模型的评测口径完全一样，抽参数反而让调用点变长——但如果你要复用到别的数据集，把它参数化。

`train_two_tower` 的 `temperature=0.1` 和 mask 都要带上。少了 mask 会退回 `Lab 2` 那个偏低的结果，那样和 `src/` 的 MF 基线就没法比了。"""
    )
)

# ---------------------------------------------------------------- 10
CELLS.append(
    markdown(
        """## 📖 收束：in-batch 目标缺了 temperature 会退化

最后一组对照：MF 除了负采样方式之外全部不动（同一个 seed、同样的 epoch 数、同样重建 loader 的方式）。加上 "in-batch 但没有 temperature" 这一组，一共四行。

先想清楚一点：**四行的 loss 不能横向比。** `random` 路每个正样本只有 1 个对手，`in_batch` 路有 200~300 个候选，交叉熵的起点天然高得多。能比的只有最后两列。

这一格的结果和"负样本越多越好"的直觉不一样，值得逐列看（800 个评测用户、5 epochs、同一个 seed）：

```text
配置                    最后一轮 loss   |user|   |item|   val Recall@20
MF random                0.5092       4.636    5.104    0.110522
MF in_batch（无 T）       5.8517       4.135    3.862    0.061475
MF in_batch（T=0.1）      6.4083       1.477    1.209    0.044242
Two-Tower（mask, T=0.1）  3.7380      11.169   10.911    0.110688
```

**先看 `|user|` / `|item|` 这两列。** `MF random` 的两张表长度都在 4.6~5.1；不带 temperature 的 in-batch 缩到了 4.1/3.9；加了 `T=0.1` 之后直接缩到 1.5/1.2。

这就是 in-batch 交叉熵的退化解：正样本的分数是 `e_u · e_i`，谁大谁赢。**把所有向量一起缩小，正样本的分数会靠近 0，交叉熵就能降下来**——不用学任何区分性信息。temperature 是把这个尺度钉住的旋钮：`T` 越小，同样的向量长度会给出更尖的 softmax，模型就必须靠"方向对不对"而不是"长度小不小"来降 loss。

**但这里还有第二层。** 加了 temperature 的 in-batch 反而更差（0.0442），说明"钉住尺度"并没有让它赢过最简单的随机负采样。原因是两个目标在优化完全不同的东西：

| | 每个正样本的对手 | 在优化什么 |
|---|---|---|
| `random` + BCE | 1 个确定的负样本 | "这个物品比那个好" |
| `in_batch` + CE | 200~300 个候选 | "这个物品在一堆里排第一" |

后者理论上信息更多，但候选集是**这批数据里恰好出现的物品**——由采样顺序决定，每轮都不一样。学到的判据更难稳定下来。这一点和 Lab 3 量到的假负样本（21.1%）是同一个来源。

**最后一行：Two-Tower 和 `MF random` 打平了**（0.110688 对 0.110522，差 0.0002）。这不是巧合：Two-Tower 的 embedding-only + in-batch + `T=0.1` 就是 MF 加一个 batch 内目标。加了 temperature 之后它不再退化，于是两组学到了同一量级的东西。

**所以这一章的结论不是"Two-Tower 比 MF 强"，而是：** 在这份数据、这个预算下，最简单的"一个正样本配一个随机负样本"就是最强的方案；换成 in-batch 需要额外把 temperature 调对，而调对之后也只是打平。**复杂度要换来东西才值得，这一组没换到。**

这也解释了 `artifacts/results.json` 里 MF 0.103140 反而略高于 Two-Tower 0.095537：`src/` 的 MF 只训了 5 个 epoch 且没有调过 lr，而 Two-Tower 是消融选出来的配置。反过来，labs 里 MF random 和 Two-Tower 只差 0.0002。这个量级的差距是**训练预算和单次运行**的差异，不是结构的差距。"""
    )
)

# ---------------------------------------------------------------- 11
CELLS.append(
    markdown(
        """### 🧩 你来写：单变量对照

用同一份数据和同样的预算，把 MF 的两个版本各跑一遍，并和 Two-Tower 摆在一起。

要求：

- 每组各自 `set_seed`、各自重建 loader；
- 只改 `negatives` 这一个变量，epoch 数、学习率、batch size 全部相同；
- 结果存进一个字典，最后打印成表。"""
    )
)

CELLS.append(
    code(
        """def compare_negative_sampling(seeds=(42,)):
    '''MF 的三种配置 + Two-Tower，同一预算下的单变量对照。

    返回 {name: {'recall20', 'ndcg20', 'loss', 'user_norm', 'item_norm'}}。
    向量的平均长度也要报出来——in-batch 那个退化解就体现在这一列。
    '''
    # TODO 7-6：每组 set_seed + 重建 loader + 重置 _NEGATIVE_SOURCE
    #   三个 MF 配置：random / in_batch（不带 temperature）/ in_batch（T=0.1）
    #   再加一个 Two-Tower
    #   除了指标，还要记 loss 和两张表的平均向量长度（norm(dim=1).mean()）
    raise NotImplementedError('TODO 7-6')""",
        lab_todo="7-compare",
    )
)

CELLS.append(
    code(
        """# ✅ 跑一下：单变量对照
labkit.set_seed(15)
comparison = compare_negative_sampling(seeds=(42,))

print()
print('配置                    最后一轮 loss   |user|   |item|   val Recall@20   val NDCG@20')
for name, record in comparison.items():
    print(f"{name:22s}  {record['loss']:7.4f}      {record['user_norm']:.3f}    "
          f"{record['item_norm']:.3f}    {record['recall20']:.6f}       {record['ndcg20']:.6f}")

random_recall = comparison['MF random']['recall20']
in_batch_recall = comparison['MF in_batch (无 T)']['recall20']
in_batch_t_recall = comparison['MF in_batch (T=0.1)']['recall20']
print()
print(f"MF random            {random_recall:.6f}")
print(f"MF in_batch（无 T）   {in_batch_recall:.6f}   （{in_batch_recall - random_recall:+.6f}）")
print(f"MF in_batch（T=0.1）  {in_batch_t_recall:.6f}   （{in_batch_t_recall - random_recall:+.6f}）")

check('7-6 四个配置都在结果里', len(comparison), 4)
check('7-6 三组 MF 都明显高于随机水平',
      min(random_recall, in_batch_recall, in_batch_t_recall) > 5 * 20 / data.num_items, True)
check('7-6 随机负采样那一组最强（in-batch 没赢过它）',
      random_recall > max(in_batch_recall, in_batch_t_recall), True)
# 退化解：不带 temperature 的 in-batch 会把向量往零缩
check('7-6 不带 T 的 in-batch 组：向量长度明显更小',
      comparison['MF in_batch (无 T)']['item_norm']
      < comparison['MF random']['item_norm'], True)
check('7-6 四组的 loss 跨了不同任务，不能横向比',
      max(record['loss'] for record in comparison.values())
      - min(record['loss'] for record in comparison.values()) > 1.0, True)"""
    )
)

CELLS.append(
    markdown(
        lab_answer_for="7-compare",
        source="""<details>
<summary>💡 参考答案（<code>compare_negative_sampling</code>）</summary>

```python
def compare_negative_sampling(seeds=(42,)):
    '''MF 的三种配置 + Two-Tower，同一预算下的单变量对照。

    返回 {name: {'recall20', 'ndcg20', 'loss', 'user_norm', 'item_norm'}}。
    向量的平均长度也要报出来——in-batch 那个退化解就体现在这一列。
    '''
    results = {}
    configs = (
        ('MF random', 'random', None),
        ('MF in_batch (无 T)', 'in_batch', None),
        ('MF in_batch (T=0.1)', 'in_batch', 0.1),
    )
    for name, negatives, temperature in configs:
        recalls, ndcgs, losses, user_norms, item_norms = [], [], [], [], []
        for seed in seeds:
            # 每一组各自 set_seed + 重建 loader，避免结果随执行顺序漂移
            model, epoch_losses = train_mf(
                data.num_users, data.num_items, data.train, epochs=5, batch_size=1024,
                negatives=negatives, temperature=temperature, seed=seed,
            )
            recall, ndcg = evaluate_model(model, validation_users, data.num_items, seen_items,
                                          data.validation)
            recalls.append(recall)
            ndcgs.append(ndcg)
            losses.append(epoch_losses[-1])
            with torch.no_grad():
                user_norms.append(float(model.user_embedding.weight.norm(dim=1).mean()))
                item_norms.append(float(model.item_embedding.weight.norm(dim=1).mean()))
        results[name] = {
            'recall20': sum(recalls) / len(recalls),
            'ndcg20': sum(ndcgs) / len(ndcgs),
            'loss': sum(losses) / len(losses),
            'user_norm': sum(user_norms) / len(user_norms),
            'item_norm': sum(item_norms) / len(item_norms),
        }

    for seed in seeds:
        model, epoch_losses = train_two_tower(data.num_users, data.num_items, data.train,
                                              positive_matrix, epochs=5, seed=seed)
        recall, ndcg = evaluate_model(model, validation_users, data.num_items, seen_items,
                                      data.validation)
        record = results.setdefault(
            'Two-Tower (mask, T=0.1)',
            {'recall20': 0.0, 'ndcg20': 0.0, 'loss': 0.0, 'user_norm': 0.0, 'item_norm': 0.0},
        )
        record['recall20'] += recall / len(seeds)
        record['ndcg20'] += ndcg / len(seeds)
        record['loss'] += epoch_losses[-1] / len(seeds)
        with torch.no_grad():
            record['user_norm'] += float(
                model.user_embedding.weight.norm(dim=1).mean()
            ) / len(seeds)
            record['item_norm'] += float(
                model.item_embedding.weight.norm(dim=1).mean()
            ) / len(seeds)
    return results
```

</details>

那个 `loss` 字段的唯一用途是**展示两个数不可比**：它出现在表里，但正文必须说明"不要看这一列比大小"。指标那一列才是能比的。

**向量长度那一列才是这一格的重点。** 它会暴露 in-batch 目标的退化解：loss 看起来"更小"或"正常"，但两张表都在往零收缩，于是所有物品的分数挤在一起，排序能力被压平。这和第 04 篇那个"loss 在降、指标不动"是同一类现象——**loss 不能告诉你模型有没有学到东西**。

`_NEGATIVE_SOURCE.seed(seed)` 那一行别省。它是模块级的 `random.Random`，不重置的话第二次跑同一组配置时负样本会接着上一次的序列往下走——两次运行不可复现，而 `labkit.set_seed` 管不到这个自己造的 Random 实例。这类"我自己的随机源"是漏掉可复现性最常见的地方。"""
    )
)

# ---------------------------------------------------------------- 12
CELLS.append(
    markdown(
        """### 🧩 你来写：多种子版对照（可选，较慢）

上面那一格只跑了一个 seed。按第 03 篇的教训，`0.05` 这个量级看着大，但要报结论最好还是看看种子抖动——**多训练几组是这一格唯一的花费**。

默认下面只跑 1 个 seed，把代码写完整；想真的验证结论时把 `SEEDS` 改成 `(1, 2, 3)` 再跑一遍（大约三倍时间）。为了控制时间，训练用户取前 1200 个（`data.train` 里 `user_idx < 1200` 的交互），**评测用户也跟着这个子集走**。"""
    )
)

CELLS.append(
    code(
        """def compare_negative_sampling_multiseed(seeds=(1, 2, 3), train_users=1200, epochs=5):
    '''多种子版对照：换几个 seed，报均值和极差。'''
    # TODO 7-7：训练集换成 user_idx < train_users 的子集
    #   评测用户也必须跟着子集走 —— 没训过的用户 embedding 还是随机的，
    #   把它们算进分母会让所有配置一起变差，差值也就没意义了
    #   三个 MF 配置各跑 len(seeds) 遍，返回
    #   {name: {'recall20_mean', 'recall20_min', 'recall20_max'}}
    raise NotImplementedError('TODO 7-7')""",
        lab_todo="7-multiseed",
    )
)

CELLS.append(
    code(
        """# ✅ 跑一下：多种子对照
# 默认只跑 1 个 seed（写完整逻辑，跑得也快）。
# 想验证"差值是否超过种子抖动"，把 SEEDS 改成 (1, 2, 3) 再跑一遍。
SEEDS = (1,)
labkit.set_seed(16)
multiseed = compare_negative_sampling_multiseed(seeds=SEEDS, train_users=1200, epochs=5)

print()
print('配置                    Recall@20 均值      极差')
for name, record in multiseed.items():
    print(f"{name:22s}  {record['recall20_mean']:.6f}    "
          f"[{record['recall20_min']:.4f}, {record['recall20_max']:.4f}]")

spread = max(
    record['recall20_max'] - record['recall20_min'] for record in multiseed.values()
)
best_name = max(multiseed, key=lambda name: multiseed[name]['recall20_mean'])
best_mean = multiseed[best_name]['recall20_mean']
runner_up = sorted(
    (record['recall20_mean'] for record in multiseed.values()), reverse=True
)[1]
print()
print(f"最好的配置是 {best_name}（{best_mean:.6f}），比第二名高 {best_mean - runner_up:.6f}")
print(f"种子极差最大到 {spread:.6f}（只跑 1 个 seed 时这个数就是 0）")

check('7-7 三个配置都有结果', [len(record) for record in multiseed.values()], [3, 3, 3])
check('7-7 所有均值都高于随机水平',
      all(record['recall20_mean'] > 5 * 20 / data.num_items for record in multiseed.values()),
      True)
check('7-7 极差已经算出来（min <= mean <= max）',
      all(record['recall20_min'] <= record['recall20_mean'] <= record['recall20_max']
          for record in multiseed.values()), True)
check('7-7 最强配置的领先幅度超过了种子极差（所以可以说它更好）',
      best_mean - runner_up > spread, True)
print()
print('结论要按这个差值写：差值若小于种子极差，就不能说哪个采样更好。')
print('这一格只跑了', len(SEEDS), '个 seed；换 3 个 seed 各跑一遍才知道抖动有多大。')"""
    )
)

CELLS.append(
    markdown(
        lab_answer_for="7-multiseed",
        source="""<details>
<summary>💡 参考答案（<code>compare_negative_sampling_multiseed</code>）</summary>

```python
def compare_negative_sampling_multiseed(seeds=(1, 2, 3), train_users=1200, epochs=5):
    '''多种子版对照：只报指标，报均值和极差。'''
    subset = [row for row in data.train if row.user_idx < train_users]
    # 评测用户也要跟着子集走：没训过的用户 embedding 还是随机初始化的值
    subset_users = [user_idx for user_idx in validation_users if user_idx < train_users]
    subset_ground_truth = [
        row for row in data.validation if row.user_idx < train_users
    ]
    print(f'训练 {train_users} 个用户，评测 {len(subset_users)} 个用户')

    results = {}
    configs = (
        ('MF random', 'random', None),
        ('MF in_batch (无 T)', 'in_batch', None),
        ('MF in_batch (T=0.1)', 'in_batch', 0.1),
    )
    for name, negatives, temperature in configs:
        recalls = []
        for seed in seeds:
            model, _ = train_mf(data.num_users, data.num_items, subset,
                                epochs=epochs, batch_size=1024, negatives=negatives,
                                temperature=temperature, seed=seed)
            recall, _ = evaluate_model(model, subset_users, data.num_items, seen_items,
                                       subset_ground_truth)
            recalls.append(recall)
        results[name] = {
            'recall20_mean': sum(recalls) / len(recalls),
            'recall20_min': min(recalls),
            'recall20_max': max(recalls),
        }
    return results
```

</details>

注意训练正样本矩阵和负采样集合仍只从 `data.train` 建；评估过滤统一使用 `data.validation_seen`（目标前全部已评分物品），训练子集不能改变评估候选集，否则两组配置就不可比。

**但评测用户必须跟着子集走。** 如果只训了前 1200 个用户、却在全部 3025 个 validation 用户上评测，剩下 1825 个用户的 embedding 还是随机初始化的值——三组配置会一起塌到 0.03 左右，差值虽然还在，但"哪个配置更好"这件事被淹没在"模型根本没学过这些用户"里了。这个错误在指标上表现为**所有配置一起变差**，很容易被误读成"训练数据少了所以正常"。

这一格跑完大约需要几分钟——三个 seed × 三组。这是"能不能下结论"的价钱。"""
    )
)

# ---------------------------------------------------------------- 13
CELLS.append(
    markdown(
        """## 这一篇的坑

1. **拿 Popularity 的全局 Top-K 直接推**，不过滤每个用户的已看物品。
2. **排序键只按次数**，相同次数时顺序随输入变化，两次运行的数不一样。
3. **`in_batch` 那一路忘了 `torch.unique`**：重复用户会让对角线出现多个正样本，重复物品会制造假负样本。
4. **横向比较两种采样的 loss。** `random` 是 `2 ln 2` 量级，`in_batch` 是 `ln(B)` 量级，结构决定的差距不是效果。
5. **单次运行就下结论。** 这一章两个配置的差值很可能小于种子极差，第三格那一步不能省。
6. **`std=0.01` 换成默认初始化**，点积初始值离 0 太远，BCE 一开始就饱和。

## 文献

- **BPR**：Rendle et al., *BPR: Bayesian Personalized Ranking from Implicit Feedback*, UAI 2009（[arXiv](https://arxiv.org/abs/1205.2618)）——"每个正样本配一个负样本"这条路线最出名的形式，pairwise 排序损失；本 lab 的 `random` 那一支用的是 pointwise 的 BCE 版本。
- **NCF**：He et al., *Neural Collaborative Filtering*, WWW 2017（[arXiv](https://arxiv.org/abs/1708.05031)）——把 MF 的点积换成 MLP 的系统对照，附录里讨论了"每个正样本采几个负样本"的影响。
- **Sampled metrics**：Krichene & Rendle, *On Sampled Metrics for Item Recommendation*, KDD 2020（[ACM](https://dl.acm.org/doi/10.1145/3394486.3403226)）——负样本数量改变的不只是训练信号强度，还会改变指标的读法；这一章坚持全量排序评测就是冲着这一点。

## 动手做

- `labs/lab7-id-only-baselines.ipynb` 概念 1 ~ 7
- 对应自测：`7-0 次数相同的按 item_idx 升序兜底`、`7-2 in_batch 路初始 loss 在 ln(B) 附近`、`7-6 两种采样的 loss 差了至少一个量级`
- 三个模型的最终 test 数（`src/recsys/` 全量训练）：`artifacts/results.json` 的 `models`
"""
    )
)

# ---------------------------------------------------------------- 14
CELLS.append(
    code(
        """# 💾 导出这一章写的东西
labkit.harvest(
    'myrec/baselines.py',
    rank_items_by_popularity,
    MatrixFactorization,
    train_one_epoch_mf,
    train_mf,
    train_two_tower,
    evaluate_model,
    imports=[
        'import torch',
        'from torch import nn',
        'from torch.nn import functional as F',
        'from myrec.eval import recommend_top_k',
        'from myrec.model import TwoTower, in_batch_cross_entropy_loss',
        'from myrec.mask import train_one_epoch_masked',
    ],
)

import importlib

import myrec.baselines as my_baselines

importlib.reload(my_baselines)

check('7-8 交接成功：myrec.baselines 能用',
      hasattr(my_baselines, 'MatrixFactorization')
      and hasattr(my_baselines, 'rank_items_by_popularity'), True)

# 光能 import 不够：真的建一个模型、跑一次前向
exported_model = my_baselines.MatrixFactorization(10, 20, embedding_dim=8)
check('7-8 导出的 MatrixFactorization 能前向',
      list(exported_model(torch.tensor([0, 1]), torch.tensor([2, 3])).shape), [2])

labkit.record('lab7_popularity_recall20', pop_recall)
labkit.record('lab7_mf_random_recall20', mf_recall)
labkit.record('lab7_two_tower_recall20', tt_recall)
print('已记录三个模型的 validation Recall@20')"""
    )
)

# ---------------------------------------------------------------- 15
CELLS.append(
    markdown(
        """## 收尾

### 这个 lab 写了多少

| 自己写的 | 行数 | 脚手架给的 |
|---|---|---|
| `rank_items_by_popularity` | 6 | 数据、`check()` / `check_close()` |
| `MatrixFactorization` | 20 | `sample_random_negative`（负采样工具） |
| `train_one_epoch_mf` | 30 | Lab 1 ~ 3 导出的模型、loss、训练循环、评测 |
| `evaluate_model` / `train_mf` / `train_two_tower` | 30 | `in_batch_cross_entropy_loss`（直接在 `in_batch` 路复用） |
| 两组对照 | 30 | |

加起来 116 行左右。其中真正新的只有 Popularity 的排序和 MF 的两张表——**训练循环、评测、负采样都是从 Lab 1 ~ 3 拿来的**。

### 这几个数应该能对上来

先跑出来的（800 个 validation 用户、5 epochs、seed=42）：

| 模型 | validation Recall@20 |
|---|---|
| Popularity | 0.079205 |
| MF（random 负采样） | 0.110522 |
| Two-Tower（in-batch + mask + T=0.1） | 0.110688 |

三种子复跑（训练 1200 个用户 + 同规模评测）：

| 配置 | Recall@20 均值 | 极差 |
|---|---|---|
| MF random | 0.112222 | [0.1075, 0.1175] |
| MF in_batch（无 T） | 0.058611 | [0.0542, 0.0633] |
| MF in_batch（T=0.1） | 0.040000 | [0.0358, 0.0450] |

领先幅度 0.0536 远大于种子极差 0.0100，所以"随机负采样更强"这个结论站得住。

`src/recsys/` 里那三个模型的 test 数是 0.083636 / 0.103140 / 0.095537（`artifacts/results.json`）。它们和 labs 这组 validation 不是同一口径：src 的 Popularity test 是 0.083636，labs 的 Popularity validation 是 0.079205，量级接近但不必逐位对齐；MF 和 Two-Tower 也在同一量级。labs 用的是 5 epochs、embedding-only、`temperature=0.1`，和 `src/` 的配置不完全相同。

### 这一章应该能直接答上来

1. 为什么必须先跑 Popularity？它给出的信息是什么？
2. MF 和 Two-Tower 的参数形状几乎一样，差别在哪？
3. in-batch 目标缺了 temperature 会退化成什么？看哪一列能发现？
4. `in_batch` 那一路为什么必须"保留配对地"去重，直接用 `torch.unique` 会怎样？
5. 如果最强配置的领先幅度小于种子极差，你会怎么写结论？

### 再往下

这一章的三个模型都是"用户向量 × 物品向量"。下一组要解决的是**检索**：物品向量算好之后，怎么在线上毫秒级内从 3533（将来是上亿）个候选里找出 Top-K，以及 FAISS 的 `IndexFlatIP` 到底帮了什么忙、没帮什么忙。"""
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
    target = REPO_ROOT / "lab7-id-only-baselines.ipynb"
    target.write_text(
        json.dumps(NOTEBOOK, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
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
