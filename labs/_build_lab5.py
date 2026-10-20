"""生成 labs/lab5-sasrec-data-and-train.ipynb。"""

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
        """# Lab 5｜序列数据、负采样与训练循环

Lab 4 把模型写好了，但它现在只是个随机初始化的函数。这一章把它接到真实数据上：

1. 一个用户的整条历史怎么切成「输入 → 下一个物品」的监督；
2. 每个位置配的那个负样本从哪来；
3. loss 怎么写才只算真实位置；
4. 一个 epoch 怎么跑。

再做两组对照：**序列长度** 和 **去掉因果 mask**。这一章会把它们跑到 30 轮——因果 mask 那一组的结论会随预算翻转，那是这一章最值得记的一件事。

符号沿用前几个 lab：📖 概念，🧩 要你写的代码，✅ 断言，🚀 可以改着玩的地方，💡 折叠的参考答案。"""
    )
)

# ---------------------------------------------------------------- 1
CELLS.append(
    code(
        """# 不用改：接上 Lab 4 导出的模型
import math
import random
import sys
import time
from pathlib import Path

import torch
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
    from myrec.sasrec_model import SASRec, build_causal_attention_mask
except ModuleNotFoundError as error:
    raise SystemExit('先跑完 Lab 4：它会把模型导出成 labs/myrec/sasrec_model.py') from error

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
print('SASRec 来自:', SASRec.__module__)"""
    )
)

# ---------------------------------------------------------------- 2
CELLS.append(
    code(
        """# 不用改：真实数据
labkit.set_seed(42)
data = labkit.prepare_movielens()
print(data.describe())

# 按时间整理每个用户在训练集里的行为序列（和 Lab 4 里看的是同一份东西）
train_histories = {}
for row in data.train:
    train_histories.setdefault(row.user_idx, []).append((row.timestamp, row.item_idx))
train_histories = {
    user_idx: [item_idx for _, item_idx in sorted(rows)]
    for user_idx, rows in train_histories.items()
}

usable = [user_idx for user_idx, items in train_histories.items() if len(items) >= 2]
print('训练集里有历史的用户:', len(train_histories))
print('其中至少有 2 次行为（能构造出监督）的:', len(usable))
print('这些用户的序列长度 中位数/最长:',
      sorted(len(train_histories[u]) for u in usable)[len(usable) // 2],
      max(len(train_histories[u]) for u in usable))"""
    )
)

# ---------------------------------------------------------------- 3
CELLS.append(
    markdown(
        """## 📖 理解一下：一条序列变成一张错位的表

长度为 6 的序列 `[a, b, c, d, e, f]`，切法是整体错开一格：

```text
输入   a  b  c  d  e
目标   b  c  d  e  f
```

5 个位置，5 条监督。一次 forward 全部算完，这是 Lab 4 里那个因果 mask 换来的。

要留意两件事：

**每个用户只需要算一次。** 不需要把 5 组样本拆成 5 条数据——那样前 4 个位置会被反复计算。

**只取最近的 `L + 1` 个物品。** 历史长到 1433 的用户，最老的那些行为对「下一个买什么」几乎没有信息量，而注意力是 `O(L²)` 的。取最近 50 个，是「够用的上下文」和「算得起」之间的取舍。

对 MovieLens 来说这个截断是有损的：中位数长度 56，一半以上的用户会被砍掉一点。换成短会话的数据集（比如电商点击流），截断就几乎无损。"""
    )
)

# ---------------------------------------------------------------- 4
CELLS.append(
    markdown(
        """### 🧩 你来写：`pad_item_sequence()`

把一个 `item_idx` 列表变成定长 token 序列，三件事按顺序做：

1. 只保留**末尾** `max_seq_len` 个；
2. 每个元素 `+1`（token `0` 留给补齐位，见 Lab 4）；
3. 在**右侧**补 `0` 到 `max_seq_len`。

返回普通 Python `list` 就行，不用张量。"""
    )
)

CELLS.append(
    code(
        """def pad_item_sequence(item_indices, max_seq_len):
    '''把 item_idx 整体加 1、只保留最新的若干项，再在右侧用 0 补齐。'''
    # TODO 5-0：三步——切片、加 1、右边补 0
    #   提示：item_indices[-max_seq_len:] 取末尾；[0] * n 造补齐位
    raise NotImplementedError('TODO 5-0')""",
        lab_todo="5-pad",
    )
)

CELLS.append(
    code(
        """# ✅ 跑一下：短序列补 0、长序列只留末尾、编号整体加 1
check('5-0 短序列补到 4 位', pad_item_sequence([3, 5], 4), [4, 6, 0, 0])
check('5-0 刚好 4 位时不用补', pad_item_sequence([1, 2, 3, 4], 4), [2, 3, 4, 5])
check('5-0 超长时只留最后 3 个', pad_item_sequence([7, 8, 9, 10, 11], 3), [10, 11, 12])
check('5-0 item_idx=0 的物品变成 token 1，不会被当成补齐位',
      pad_item_sequence([0], 2), [1, 0])
check('5-0 空序列全是补齐位', pad_item_sequence([], 3), [0, 0, 0])
check('5-0 返回的是长度正好 max_seq_len 的列表',
      len(pad_item_sequence([1, 2, 3, 4, 5], 4)), 4)"""
    )
)

CELLS.append(
    markdown(
        lab_answer_for="5-pad",
        source="""<details>
<summary>💡 参考答案（<code>pad_item_sequence</code>）</summary>

```python
def pad_item_sequence(item_indices, max_seq_len):
    '''把 item_idx 整体加 1、只保留最新的若干项，再在右侧用 0 补齐。'''
    if max_seq_len <= 0:
        raise ValueError('max_seq_len must be positive')
    shifted_items = [item_idx + 1 for item_idx in item_indices[-max_seq_len:]]
    return shifted_items + [0] * (max_seq_len - len(shifted_items))
```

</details>

右边补 0 是硬约定，不是风格问题。Lab 4 里 `sequence_representation` 取的是「最后一个非补齐位置」，左补齐会让新用户取到一个全零向量。

`[0] * (max_seq_len - len(...))` 在序列已经够长时是空列表，正好不加东西。"""
    )
)

# ---------------------------------------------------------------- 5
CELLS.append(
    markdown(
        """## 📖 理解一下：每个位置配一个负样本

上一步切出来的表里只有正样本：模型被告知「这个位置的答案应该是 `d`」，但没有被告知「不是 `x`」。只做正样本的二分类，模型只要把所有物品的分数一起推高就够了。

所以每个位置再配一个**该用户没有交互过**的物品，label 为 0：

```text
输入    a     b     c     d     e
正样本  b     c     d     e     f
负样本  x1    x2    x3    x4    x5     # 每个都是这个用户没看过的
```

怎么取。最直接的做法是拒绝采样：从全部物品里随机挑一个，如果是这个用户已经交互过的就重新挑。

这里有一个**很容易写错、而且不会报错**的地方：

```text
seen_items 必须只来自训练集
```

如果 `seen_items` 里混进了 validation 或 test 的物品，负样本就会系统性地避开这些物品——模型在训练阶段就被告知「这些物品是特殊的」。评测时这些正好是要预测的目标，指标会虚高，而且看不出任何异常。

这份数据还有一个结构性的限制：这个数据集里**没有一个物品是所有人都交互过的**，所以拒绝采样不会卡死。物品数只有 3533、用户平均 93 次训练交互，撞上已交互物品的概率在百分之几的量级，循环几乎不会转第二次。换成物品很少或用户很活跃的场景，拒绝采样可能反复失败，那时候要换成按物品权重采样或者维护候选集。"""
    )
)

# ---------------------------------------------------------------- 6
CELLS.append(
    markdown(
        """### 🧩 你来写：`sample_negative_item()`

签名和 `src/recsys/training/samplers.py` 里那个一致：返回**已经加 1 的 token**，也就是可以直接送进 embedding 的值。这个约定是有意的——`Dataset.__getitem__` 里正样本、负样本、输入序列都走在同一个 token 空间里，负样本不需要在别处再翻译一次。

用传进来的 `random` 实例取数，保证可复现。如果全部物品都在 `seen_items` 里，抛 `ValueError`，不要死循环。"""
    )
)

CELLS.append(
    code(
        """def sample_negative_item(num_items, seen_items, random):
    '''随机取一个该用户从未交互过的物品，返回加 1 之后的 token。'''
    # TODO 5-1：拒绝采样：抽一个，撞上就重抽
    #   返回的是 token（item_idx + 1），调用方会直接送进 embedding
    #   物品全被看过时要能报错退出，不要 while True
    raise NotImplementedError('TODO 5-1')""",
        lab_todo="5-negative",
    )
)

CELLS.append(
    code(
        """# ✅ 跑一下：抽 200 次都不该撞上已看物品，同一 seed 结果一样
import random as _random

seen = {0, 1, 2, 3, 4}


def draw_two_hundred():
    '''每次都用全新的 Random(0) 抽 200 次，方便和另一个实例对照。'''
    sampler = _random.Random(0)
    return [sample_negative_item(100, seen, sampler) for _ in range(200)]


draws = draw_two_hundred()
# 返回的是 token，换回 item_idx 才和 seen 可比
drawn_items = {token - 1 for token in draws}

check('5-1 抽 200 次都没撞上已看物品', any(item_idx in seen for item_idx in drawn_items), False)
check('5-1 抽出来的都在 [1, 100]（token 空间，0 留给补齐）',
      all(1 <= token <= 100 for token in draws), True)
check('5-1 不是每次都返回同一个值（真的在抽）', len(set(draws)) > 10, True)
check('5-1 相同 seed 的两个实例抽出的序列完全一致', draws, draw_two_hundred())

# 只剩一个候选物品时，必须精确地选出它（item_idx=4 -> token 5）
check('5-1 只剩一个候选时选出那一个', sample_negative_item(5, {0, 1, 2, 3}, _random.Random(1)), 5)

# 全部物品都被看过：报错，不要卡住
try:
    sample_negative_item(3, {0, 1, 2}, _random.Random(1))
except ValueError as error:
    print('全部看过 ->', type(error).__name__, error)
    raised = True
else:
    raised = False
check('5-1 全部物品都看过时抛 ValueError', raised, True)"""
    )
)

CELLS.append(
    markdown(
        lab_answer_for="5-negative",
        source="""<details>
<summary>💡 参考答案（<code>sample_negative_item</code>）</summary>

```python
def sample_negative_item(num_items, seen_items, random):
    '''随机取一个该用户从未交互过的物品，返回加 1 之后的 token。'''
    if len(seen_items) >= num_items:
        raise ValueError('cannot sample a negative when every item is seen')
    negative_item_idx = random.randrange(num_items)
    while negative_item_idx in seen_items:
        negative_item_idx = random.randrange(num_items)
    return negative_item_idx + 1
```

</details>

结构就是源码里的两行：先抽一次，撞上了再循环重抽。`len(seen_items) >= num_items` 那个前置判断是**必须的**——没有它，物品全被看过时 `while` 永远转不出去。这是拒绝采样的固有边界：候选集非空才成立，工程上要么保证非空，要么像这里一样显式抛错。

`random.randrange(num_items)` 的取值范围是 `[0, num_items)`，右端开——和 `item_idx` 的取值范围一致。写成 `random.randint(0, num_items)` 会让 `num_items` 这个越界值偶尔出现，然后在 embedding 查表时报 index out of range。"""
    )
)

# ---------------------------------------------------------------- 7
CELLS.append(
    markdown(
        """## 📖 理解一下：数据集要给出三张对齐的表

`Dataset` 的每个样本是一条用户序列，返回三个张量：

```text
input_sequence     [L]   最近 L 个物品，最右端是「用来预测的那个时刻」
positive_sequence  [L]   每个位置的正确答案
negative_sequence  [L]   每个位置配的负样本
```

对齐关系是：

```text
input_sequence  = 序列整体加 1，取 [-L-1 : -1]        # 丢掉最后一个
positive_sequence = 序列整体加 1，取 [-L :]           # 丢掉第一个
```

也就是「输入的第 `t` 位，预测 `positive` 的第 `t` 位」，错开一格。

`positive_sequence` 里的 `0` 是补齐位，`input_sequence` 里的 `0` 也是。两者位置一一对应，所以训练时用 `positive_sequence != 0` 就能一次性筛出所有有效位置。

这里有一个容易搞混的约定，源码和这一章是一致的：

- **物品编号整体加 1，只加一次。** `sample_negative_item` 返回的**已经是 token**（`item_idx + 1`），`pad_item_sequence` 也负责给输入序列和正样本序列整体加 1。三张表最后都在同一个 token 空间里，谁也不用再翻译。
- 忘了加、或者加了两遍，都不会报错，只会让编号整体错位一格。

还有一条边界：**`__getitem__` 里不要用 `self.user_sequences` 之外的历史。** 负采样用的 `seen_items` 一旦包含 validation/test，就是上面说的泄漏。"""
    )
)

# ---------------------------------------------------------------- 8
CELLS.append(
    markdown(
        """### 🧩 你来写：`SASRecTrainDataset`

`__init__` 要做的事（脚手架里已经写好了）：按用户整理序列、**丢掉最近窗口内不足两个不同时间戳的用户**（同秒先后未知，不能作为「下一个」标签）、记住 `max_seq_len` 和一个固定 seed 的 `random.Random`。

你要写 `__getitem__(index)`，返回三个 `torch.long` 张量 `input_sequence` / `positive_sequence` / `negative_sequence`，长度都是 `max_seq_len`。

需要的变量脚手架已经备好：`full_sequence` 是这个用户的完整 `item_idx` 列表，`self._sample_negative(...)` 返回一个未交互物品（已按 `sample_negative_item` 实现）。"""
    )
)

CELLS.append(
    code(
        """class SASRecTrainDataset(torch.utils.data.Dataset):
    '''每个样本是一条用户序列，并且已经对齐好「下一个物品」的目标。'''

    def __init__(self, train_interactions, num_items, max_seq_len, seed=42):
        if num_items <= 0:
            raise ValueError('num_items must be positive')
        if max_seq_len <= 0:
            raise ValueError('max_seq_len must be positive')
        rows_by_user = {}
        for row in train_interactions:
            rows_by_user.setdefault(row.user_idx, []).append(row)
        self.user_sequences = {
            user_idx: [
                row.item_idx
                for row in sorted(rows, key=lambda row: (row.timestamp, row.item_idx))
            ]
            for user_idx, rows in rows_by_user.items()
        }
        self.user_rows = {
            user_idx: sorted(rows, key=lambda row: (row.timestamp, row.item_idx))
            for user_idx, rows in rows_by_user.items()
        }
        # 只有不同 timestamp 才能明确哪个行为是「下一个」。
        self.user_indices = sorted(
            user_idx
            for user_idx, rows in self.user_rows.items()
            if len({row.timestamp for row in rows[-(max_seq_len + 1):]}) >= 2
        )
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.random = random.Random(seed)

    def __len__(self):
        return len(self.user_indices)

    def _sample_negative(self, seen_items):
        return sample_negative_item(self.num_items, seen_items, self.random)"""
    )
)

CELLS.append(
    code(
        """def dataset_getitem(self, index):
    '''返回一条对齐好的样本：输入 / 正样本 / 负样本。'''
    user_idx = self.user_indices[index]
    full_sequence = self.user_sequences[user_idx]
    # TODO 5-2：切成错开一格的两段
    #   recent = 末尾 max_seq_len + 1 个；输入是 recent[:-1]，正样本是 recent[1:]
    #   seen_items 用这个用户的全部训练行为（不是 recent！）
    #   每个正样本位置采一个负样本，最后三个张量都要补齐到 max_seq_len
    raise NotImplementedError('TODO 5-2')


SASRecTrainDataset.__getitem__ = dataset_getitem""",
        lab_todo="5-dataset",
    )
)

CELLS.append(
    code(
        """# ✅ 跑一下：用真实数据建数据集，逐个检查对齐关系
labkit.set_seed(5)
dataset = SASRecTrainDataset(data.train, num_items=data.num_items, max_seq_len=8, seed=42)
print('样本数:', len(dataset), '| 用户数:', len(dataset.user_indices))

users_with_history = sum(
    len({row.timestamp for row in rows[-9:]}) >= 2
    for rows in dataset.user_rows.values()
)
check('5-2 样本数等于最近窗口有两种 timestamp 的用户数', len(dataset), users_with_history)
check('5-2 序列长度参数生效', dataset.max_seq_len, 8)

sample = dataset[0]
check('5-2 返回三个序列张量（外加 user_idx）', sorted(sample.keys()),
      ['input_sequence', 'negative_sequence', 'positive_sequence', 'user_idx'])
check('5-2 三个张量长度都是 max_seq_len',
      [len(sample[key]) for key in ('input_sequence', 'positive_sequence', 'negative_sequence')],
      [8, 8, 8])
check('5-2 都是 long 张量',
      {str(sample[key].dtype) for key in sample}, {'torch.int64'})
check('5-2 user_idx 只是顺带带出来的元信息', int(sample['user_idx']), dataset.user_indices[0])

# 逐个位置核对错位关系：正样本的第 t 位 = 输入的第 t+1 位
# 输入的最后一位应该是「倒数第二个物品 + 1」，不是目标本身
user_idx = dataset.user_indices[0]
items = dataset.user_sequences[user_idx]
recent = items[-(8 + 1):]

check('5-2 输入最后一位是序列的倒数第二个物品（+1）',
      int(sample['input_sequence'][-1]), recent[-2] + 1)
check('5-2 正样本最后一位才是要预测的那个物品（+1）',
      int(sample['positive_sequence'][-1]),
      recent[-1] + 1 if dataset.user_rows[user_idx][-1].timestamp > dataset.user_rows[user_idx][-2].timestamp else 0)
check('5-2 正样本与输入错开一格',
      [int(v) for v in sample['positive_sequence'][:-1]],
      [int(sample['input_sequence'][i + 1]) if dataset.user_rows[user_idx][-9:][i + 1].timestamp > dataset.user_rows[user_idx][-9:][i].timestamp else 0
       for i in range(7)])
print('末位：输入', int(sample['input_sequence'][-1]),
      '-> 正样本', int(sample['positive_sequence'][-1]),
      '| 负样本', int(sample['negative_sequence'][-1]))

# 负样本必须避开这个用户训练集里的全部物品
# （负样本序列里存的已经是 token，减 1 换回 item_idx 才和训练集可比）
seen_items = set(items)
negative_items = {token - 1 for token in sample['negative_sequence'].tolist() if token != 0}
overlap = negative_items & seen_items
print('这个用户训练集物品数:', len(seen_items), '| 采样出的负样本数:', len(negative_items))
check('5-2 负样本不含该用户训练集里的任何物品', len(overlap), 0)
check('5-2 负样本全部落在 [0, num_items)', all(0 <= item_idx < data.num_items for item_idx in negative_items), True)
check('5-2 负样本仅在有效 next-item 位置非零',
      [int(v != 0) for v in sample['negative_sequence']],
      [int(v != 0) for v in sample['positive_sequence']])

# 采样是可复现的：同一个 seed 建两次数据集，负样本必须逐位一致
again = SASRecTrainDataset(data.train, num_items=data.num_items, max_seq_len=8, seed=42)
check('5-2 同一个 seed 的负采样完全一致',
      [int(value) for value in again[0]['negative_sequence']],
      [int(value) for value in sample['negative_sequence']])

# 短序列会被右补齐：找一个历史不足 8 次的用户
short_users = [u for u in dataset.user_indices if len(dataset.user_sequences[u]) < 8
               and len({row.timestamp for row in dataset.user_rows[u]}) == len(dataset.user_rows[u])]
print('历史不足 8 次的用户数:', len(short_users))
short_user = short_users[0]
short_items = dataset.user_sequences[short_user]
short_sample = dataset[dataset.user_indices.index(short_user)]
with torch.no_grad():
    valid = int((short_sample['positive_sequence'] != 0).sum())
    # 逐步重建一遍期望值，和数据集给的对照
    expected_positive = [item_idx + 1 for item_idx in short_items[1:]]
    expected_positive += [0] * (8 - len(expected_positive))
    expected_input = pad_item_sequence(short_items[:-1], 8)
    check('5-2 短序列的有效位置数等于行为数 - 1', valid, len(short_items) - 1)
    check('5-2 短序列右侧确实是 0', int(short_sample['positive_sequence'][-1]), 0)
    check('5-2 短序列的输入用 pad_item_sequence 重建得出来',
          [int(v) for v in short_sample['input_sequence']], expected_input)
    check('5-2 短序列的正样本就是整体后移一位再加 1',
          [int(v) for v in short_sample['positive_sequence']], expected_positive)
    check('5-2 短序列的负样本也是右补齐的',
          int(short_sample['negative_sequence'][valid]), 0)"""
    )
)

CELLS.append(
    markdown(
        lab_answer_for="5-dataset",
        source="""<details>
<summary>💡 参考答案（<code>SASRecTrainDataset.__getitem__</code>）</summary>

```python
def dataset_getitem(self, index):
    '''返回一条对齐好的样本：输入 / 正样本 / 负样本。'''
    user_idx = self.user_indices[index]
    full_sequence = self.user_sequences[user_idx]
    recent_items = full_sequence[-(self.max_seq_len + 1):]
    input_items = recent_items[:-1]
    positive_items = recent_items[1:]

    recent_rows = self.user_rows[user_idx][-(self.max_seq_len + 1):]
    ordered_targets = [right.timestamp > left.timestamp
                       for left, right in zip(recent_rows[:-1], recent_rows[1:])]

    # 负样本要避开这个用户在训练集里交互过的**全部**物品，不只是 recent_items。
    # 这里只用了 train_interactions，validation / test 的物品不参与。
    # sample_negative_item 返回的已经是 token，直接补 0 就行。
    seen_items = set(full_sequence)
    negative_items = [self._sample_negative(seen_items) for _ in positive_items]

    input_sequence = pad_item_sequence(input_items, self.max_seq_len)
    positive_sequence = [item_idx + 1 if ordered else 0
                         for item_idx, ordered in zip(positive_items, ordered_targets)]
    positive_sequence += [0] * (self.max_seq_len - len(positive_sequence))
    negative_sequence = [item_idx if ordered else 0
                         for item_idx, ordered in zip(negative_items, ordered_targets)] + [0] * (
        self.max_seq_len - len(negative_items)
    )
    return {
        'user_idx': torch.tensor(user_idx, dtype=torch.long),
        'input_sequence': torch.tensor(input_sequence, dtype=torch.long),
        'positive_sequence': torch.tensor(positive_sequence, dtype=torch.long),
        'negative_sequence': torch.tensor(negative_sequence, dtype=torch.long),
    }


SASRecTrainDataset.__getitem__ = dataset_getitem
```

</details>

`seen_items = set(full_sequence)` 里的 `full_sequence` 是这个用户的**全部训练行为**，而不是切出来的 `recent_items`。用 `recent_items` 会让负样本有机会撞上更早交互过的物品——那些位置在序列里出现过，模型正被要求把它们当成上下文，这里又被要求压低。

`ordered_targets` 把同秒相邻位置的正、负标签都设为 0，loss 会略过；仅靠 item ID 排序无法证明同秒事件的先后。

负样本不需要 `pad_item_sequence`：它已经是一个个独立的 `item_idx`，只要 `+1` 之后补 0 就行。"""
    )
)

# ---------------------------------------------------------------- 9
CELLS.append(
    markdown(
        """## 📖 理解一下：loss 只在有效位置上算

每个位置有两条监督：

```text
正样本位置：让「隐状态 · 正样本向量」的分数变高
负样本位置：让「隐状态 · 负样本向量」的分数变低
```

两条都是二分类交叉熵（`binary_cross_entropy_with_logits`，输入是 logit 不是概率），加起来就是这一批的 loss。

关键在于**哪些位置要算**。补齐位置的 `positive_sequence` 是 0，如果照算不误，模型会被要求「让第 7 个位置的分数去匹配一个不存在的物品」——梯度方向完全是噪声。

做法是用正样本序列的 `!= 0` 当掩码，在算 loss 之前把 logits 压平再筛：

```text
valid_positions = positive_sequence != 0     # [B, L]
positive_logits[valid_positions]             # [有效位置数]
```

这样补齐位置根本不进 loss，梯度自然也不会回到它们。

和第 04 篇那个 false negative mask 是同一类手法：**先确定哪些位置该参与，再算 loss。**"""
    )
)

# ---------------------------------------------------------------- 10
CELLS.append(
    markdown(
        """### 🧩 你来写：`sasrec_loss()`

输入是模型输出的 `hidden_states [B, L, D]`，加上 `positive_sequence [B, L]` 和 `negative_sequence [B, L]`。

步骤：

1. 用 `model.item_embedding(...)` 把两个序列查成向量；
2. 和 `hidden_states` 逐元素相乘再在最后一维求和，得到两个 `[B, L]` 的 logits；
3. `valid_positions = positive_sequence != 0`，只在这上面算两个 BCE，再相加。

返回一个标量。注意 `negative_sequence` 里补齐位也是 `0`，但掩码用的是**正样本**序列，两边位置一一对应。"""
    )
)

CELLS.append(
    code(
        """def sasrec_loss(hidden_states, positive_sequence, negative_sequence, model):
    '''每个有效位置上的「正样本 BCE + 负样本 BCE」。'''
    # TODO 5-3：查表 -> 点积 -> 掩码 -> 两个 BCE 相加
    #   点积用 (hidden_states * embeddings).sum(dim=-1)
    #   掩码用 positive_sequence != 0，两个序列共用同一张掩码
    raise NotImplementedError('TODO 5-3')""",
        lab_todo="5-loss",
    )
)

CELLS.append(
    code(
        """# ✅ 跑一下：补齐位不参与、手算对得上、常量分数时等于 ln2
labkit.set_seed(6)
toy_model = SASRec(num_items=20, max_seq_len=4, embedding_dim=16, num_blocks=1,
                   num_heads=2, dropout=0.0).eval()

positive = torch.tensor([[3, 4, 5, 0], [6, 7, 0, 0]])
negative = torch.tensor([[8, 9, 10, 0], [11, 12, 0, 0]])
with torch.no_grad():
    toy_hidden = toy_model(torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]))
    toy_loss = sasrec_loss(toy_hidden, positive, negative, toy_model)

check('5-3 loss 是标量（没有维度）', list(toy_loss.shape), [])
check('5-3 loss 是有限值', bool(torch.isfinite(toy_loss)), True)

# 手算一遍：有效位置一共 5 个，正负各 5 条
valid = positive != 0
with torch.no_grad():
    positive_logits = (toy_hidden * toy_model.item_embedding(positive)).sum(dim=-1)
    negative_logits = (toy_hidden * toy_model.item_embedding(negative)).sum(dim=-1)
    manual = F.binary_cross_entropy_with_logits(
        positive_logits[valid], torch.ones(int(valid.sum()))
    ) + F.binary_cross_entropy_with_logits(
        negative_logits[valid], torch.zeros(int(valid.sum()))
    )
check_close('5-3 和手算的两个 BCE 一致', float(toy_loss), float(manual))

# 如果模型对所有物品给出同一个分数（logit=0），正负两个 BCE 各是 ln2，合计 2 ln2
with torch.no_grad():
    zeros_hidden = torch.zeros_like(toy_hidden)
    flat_loss = sasrec_loss(zeros_hidden, positive, negative, toy_model)
check_close('5-3 分数全相等时 loss = 2·ln2（正负各一份）', float(flat_loss), 2 * math.log(2))

# 补齐位置的 logits 换成任意值，loss 一点都不该变
with torch.no_grad():
    padded_hidden = toy_hidden.clone()
    padded_hidden[0, 3] = 123.0
    padded_hidden[1, 2:] = -45.0
    padded_loss = sasrec_loss(padded_hidden, positive, negative, toy_model)
check('5-3 改动补齐位置的隐状态，loss 完全不变', float(padded_loss), float(toy_loss))

# 有效位置的隐状态换掉，loss 必须变——否则上面那条是假通过
with torch.no_grad():
    real_hidden = toy_hidden.clone()
    real_hidden[0, 0] = 5.0
    real_loss = sasrec_loss(real_hidden, positive, negative, toy_model)
check('5-3 改动真实位置，loss 会变（说明掩码没把有效位置也滤掉）',
      float(real_loss) != float(toy_loss), True)"""
    )
)

CELLS.append(
    markdown(
        lab_answer_for="5-loss",
        source="""<details>
<summary>💡 参考答案（<code>sasrec_loss</code>）</summary>

```python
def sasrec_loss(hidden_states, positive_sequence, negative_sequence, model):
    '''每个有效位置上的「正样本 BCE + 负样本 BCE」。'''
    positive_embeddings = model.item_embedding(positive_sequence)
    negative_embeddings = model.item_embedding(negative_sequence)
    positive_logits = (hidden_states * positive_embeddings).sum(dim=-1)
    negative_logits = (hidden_states * negative_embeddings).sum(dim=-1)

    # 补齐位置不进 loss：正样本序列里 0 就是补齐，负样本序列的位置和它一一对应
    valid_positions = positive_sequence.ne(0)
    positive_loss = F.binary_cross_entropy_with_logits(
        positive_logits[valid_positions],
        torch.ones_like(positive_logits[valid_positions]),
    )
    negative_loss = F.binary_cross_entropy_with_logits(
        negative_logits[valid_positions],
        torch.zeros_like(negative_logits[valid_positions]),
    )
    return positive_loss + negative_loss
```

</details>

`torch.ones_like(...)` / `torch.zeros_like(...)` 而不是 `torch.ones(n)`：不用自己数有效位置有几个，形状自动对上，也不会因为数错而静默地只算一半。

正负两个 BCE 各取均值之后相加。这里没有权重超参——两个方向被同等对待。想让负样本更强可以调，但那会多一个需要 validation 上选的参数。"""
    )
)

# ---------------------------------------------------------------- 11
CELLS.append(
    markdown(
        """## 📖 理解一下：训练循环的四件事

一个 epoch 就是「遍历 dataloader，反复 前向 → 算 loss → 反向 → 更新」。除了主体逻辑，有四件事必须在：

**1. epoch 开头 `model.train()`。** 评估代码会调 `model.eval()`（关掉 dropout），不切回来就等于用推理模式训练——loss 会照降，只是 dropout 这个正则项完全没生效。

**2. 张量搬到 `device`。** 这部分在 Lab 4 里没有体现，因为那里的玩具模型直接在 CPU 上。CPU-only 的时候 `.to(device)` 是空操作，但代码要写对，换到 GPU 上才不用改。

**3. 平均 loss 除以「实际跑过的批次数」，不是 `len(dataloader)`。** 加了 `max_batches` 之类的限制之后，`len(dataloader)` 和真实跑过的批次数会不一致，除错了 loss 曲线就是假的。

**4. 把 loss 记下来。** 后面那两组对照要看的就是 loss 与 Recall 的关系。

另外 `optimizer.zero_grad()` 要在 `backward()` 之前：PyTorch 的梯度是累加的，不清零就会把上一批的梯度一起算进来。"""
    )
)

# ---------------------------------------------------------------- 12
CELLS.append(
    markdown(
        """### 🧩 你来写：`train_one_epoch()`

写一个完整的训练 epoch，返回这一轮的平均 loss。

签名里给了 `max_batches`：用来只跑前若干个 batch（先确认能跑通、或者做快速对照）。注意上面第 3 条——有它的时候平均值的分母就不一样了。"""
    )
)

CELLS.append(
    code(
        """def train_one_epoch(model, dataloader, optimizer, device, max_batches=None):
    '''训练一个 epoch，返回实际跑过的批次上的平均 loss。'''
    # TODO 5-4：model.train() / 搬设备 / forward / loss / zero_grad / backward / step
    #   平均 loss 除以实际跑过的批次数
    #   一个 batch 里要同时拿到 input_sequence、positive_sequence、negative_sequence
    raise NotImplementedError('TODO 5-4')""",
        lab_todo="5-train",
    )
)

CELLS.append(
    code(
        """# ✅ 跑一下：30 个 batch，确认 loss 有限、参数真的被更新了
labkit.set_seed(7)
train_dataset = SASRecTrainDataset(data.train, num_items=data.num_items,
                                   max_seq_len=20, seed=42)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True,
                                           generator=torch.Generator().manual_seed(7))
small_model = SASRec(num_items=data.num_items, max_seq_len=20, embedding_dim=64,
                     num_blocks=2, num_heads=2, dropout=0.2)
optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-3)

before = small_model.item_embedding.weight.detach().clone()
batch_loss = train_one_epoch(small_model, train_loader, optimizer, torch.device('cpu'),
                             max_batches=30)
after = small_model.item_embedding.weight.detach().clone()

print('30 个 batch 的平均 loss:', round(batch_loss, 4))
check('5-4 loss 是有限的', math.isfinite(batch_loss), True)
# 随机初始化时正负 logits 都接近 0，两个 BCE 各约 ln2，合计约 2 ln2 ≈ 1.386
check('5-4 初始 loss 在 2*ln2 附近（0.5 ~ 1.6）', 0.5 < batch_loss < 1.6, True)
check('5-4 参数确实被更新了', bool((after - before).abs().max().detach() > 0), True)
check('5-4 补齐 token 那一行仍然是 0（padding_idx 保护住了）',
      float(after[0].abs().max()), 0.0)
check('5-4 训练完模型在 train 模式', small_model.training, True)

# max_batches 的分母要对：跑 5 个 batch 和跑 30 个 batch，loss 都是「跑过的那些」的平均
optimizer2 = torch.optim.Adam(small_model.parameters(), lr=1e-3)
five_batch_loss = train_one_epoch(small_model, train_loader, optimizer2, torch.device('cpu'),
                                  max_batches=5)
check('5-4 max_batches 生效且平均值没被算错', math.isfinite(five_batch_loss), True)
check('5-4 不限制 batch 数时也能跑完一个 epoch',
      math.isfinite(train_one_epoch(
          small_model,
          torch.utils.data.DataLoader(train_dataset, batch_size=2048, shuffle=False),
          torch.optim.Adam(small_model.parameters(), lr=1e-3),
          torch.device('cpu'))), True)"""
    )
)

CELLS.append(
    markdown(
        lab_answer_for="5-train",
        source="""<details>
<summary>💡 参考答案（<code>train_one_epoch</code>）</summary>

```python
def train_one_epoch(model, dataloader, optimizer, device, max_batches=None):
    '''训练一个 epoch，返回实际跑过的批次上的平均 loss。'''
    model.train()
    total_loss = 0.0
    batches_seen = 0
    for batch_number, batch in enumerate(dataloader):
        if max_batches is not None and batch_number >= max_batches:
            break
        input_sequence = batch['input_sequence'].to(device)
        positive_sequence = batch['positive_sequence'].to(device)
        negative_sequence = batch['negative_sequence'].to(device)

        hidden_states = model(input_sequence)
        loss = sasrec_loss(hidden_states, positive_sequence, negative_sequence, model)

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

`batches_seen` 必须在循环里数。写成 `total_loss / len(dataloader)` 在 `max_batches=None` 时结果一样，一旦加上限制就错了，而且错得很隐蔽——loss 只是偏小，不会报错。

`loss.item()` 把标量取成 Python float。直接累加张量会让整个计算图一直挂着，几十个 batch 之后显存就满了。"""
    )
)

# ---------------------------------------------------------------- 13
CELLS.append(
    markdown(
        """## 📖 理解一下：训练完了怎么知道有没有学到

一个 epoch 跑完，loss 大概从 1.39 降到 1.3 上下。这个数字说明不了什么——正样本分数推高一点、负样本压低一点，loss 就会降，哪怕排序一点没变好。

要看的是**下一个物品有没有被排到前面去**。做法和 Lab 2 一样：给全部物品打分，过滤掉已经交互过的，取 Top-20，看真正的下一个物品在不在里面。

区别只在构造输入的方式：

```text
验证阶段：历史 = 这个用户在 train 里的全部行为   输入 = pad(history, max_seq_len)
                   目标 = validation 的那一个物品
```

注意**目标物品不在输入里**。如果把它放进输入，模型看到的就是答案本身，指标会高得离谱。这是序列模型评测里最容易犯的泄漏——Lab 6 会把这条检查写死。"""
    )
)

# ---------------------------------------------------------------- 14
CELLS.append(
    markdown(
        """### 🧩 你来写：三件评测要用的小工具

**a) `score_sequences(model, sequences)`** —— 给一批序列打分，返回 `[B, num_items]`。内部就一句 `model.score_all_items`，包一层是为了让它能在 `no_grad` 下批量跑。

**b) `recommend_top_k(model, examples, num_items, k)`** —— `examples` 是 `(input_sequence, seen_item_indices)` 的列表。逐批打分，把已看物品的分数设成 `-inf`，再取 top-k。返回 `dict[user_idx] -> [item_idx]`，这里为了简单先不做逐用户的映射，直接返回一个列表的列表。

**c) `build_validation_examples(data, max_seq_len, users)`** —— 对给定用户，用 train 历史构造输入，目标取 validation 里的那一个物品。返回 `(user_indices, input_sequences, targets, seen_items_list)`。"""
    )
)

CELLS.append(
    code(
        """def score_sequences(model, sequences):
    '''给一批序列打分，返回 [B, num_items]。'''
    # TODO 5-5a：model.eval() + no_grad，直接调 model.score_all_items
    raise NotImplementedError('TODO 5-5a')


def recommend_top_k(model, sequences, seen_items_list, k, batch_size=256):
    '''逐批打分、把已看物品压到 -inf、取 top-k。返回 list[list[item_idx]]。'''
    # TODO 5-5b：注意不要原地改 model.score_all_items 返回的张量之外的东西
    #   已看物品用 scores[row, seen_items] = -inf 压掉，再 torch.topk
    raise NotImplementedError('TODO 5-5b')


def build_validation_examples(data, max_seq_len, users):
    '''用 train 历史构造输入，目标是 validation 的物品。'''
    # TODO 5-5c：把 train 历史按用户整理好，再为每个用户造一条输入
    #   历史里不能出现 validation / test 的物品
    raise NotImplementedError('TODO 5-5c')""",
        lab_todo="5-eval",
    )
)

CELLS.append(
    code(
        """# ✅ 跑一下：先在 24 个用户上看形状和过滤
labkit.set_seed(8)
eval_users = sorted({row.user_idx for row in data.validation})[:24]
eval_user_indices, eval_sequences, eval_targets, eval_seen = build_validation_examples(
    data, max_seq_len=20, users=eval_users
)

print('评测用户数:', len(eval_user_indices))
check('5-5 每个评测用户一条输入', len(eval_sequences), len(eval_users))
check('5-5 输入形状和长度对得上',
      [len(sequence) for sequence in eval_sequences[:2]], [20, 20])
check('5-5 目标数与用户数一致', len(eval_targets), len(eval_users))

# 目标物品绝不能出现在输入里，也不在 seen 里
targets_by_user = {row.user_idx: row.item_idx for row in data.validation}
leaked = []
for position, user_idx in enumerate(eval_user_indices):
    target_token = targets_by_user[user_idx] + 1
    if target_token in eval_sequences[position]:
        leaked.append(user_idx)
    if targets_by_user[user_idx] in eval_seen[position]:
        leaked.append(user_idx)
check('5-5 目标物品既不在输入序列里，也不在 seen 里', leaked, [])

eval_model = SASRec(num_items=data.num_items, max_seq_len=20, embedding_dim=64,
                    num_blocks=2, num_heads=2, dropout=0.0).eval()
with torch.no_grad():
    scores = score_sequences(eval_model, eval_sequences[:4])
check('5-5 分数矩阵形状是 [B, num_items]', list(scores.shape), [4, data.num_items])

with torch.no_grad():
    recommendations = recommend_top_k(eval_model, eval_sequences[:4], eval_seen[:4], k=10)
check('5-5 每个用户 10 个推荐', [len(items) for items in recommendations], [10] * 4)
check('5-5 推荐里没有已看物品',
      [len(set(items) & eval_seen[row]) for row, items in enumerate(recommendations)], [0] * 4)
check('5-5 推荐物品不重复',
      [len(set(items)) for items in recommendations], [10] * 4)
check('5-5 推荐下标都在 [0, num_items)',
      all(0 <= item_idx < data.num_items for items in recommendations for item_idx in items), True)"""
    )
)

CELLS.append(
    markdown(
        lab_answer_for="5-eval",
        source="""<details>
<summary>💡 参考答案（<code>score_sequences</code> / <code>recommend_top_k</code> / <code>build_validation_examples</code>）</summary>

```python
def score_sequences(model, sequences):
    '''给一批序列打分，返回 [B, num_items]。'''
    model.eval()
    with torch.no_grad():
        batch = torch.tensor(sequences, dtype=torch.long)
        return model.score_all_items(batch)


def recommend_top_k(model, sequences, seen_items_list, k, batch_size=256):
    '''逐批打分、把已看物品压到 -inf、取 top-k。返回 list[list[item_idx]]。'''
    model.eval()
    recommendations = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch_sequences = sequences[start:start + batch_size]
            batch_seen = seen_items_list[start:start + batch_size]
            scores = model.score_all_items(
                torch.tensor(batch_sequences, dtype=torch.long)
            )
            for row, seen_items in enumerate(batch_seen):
                if seen_items:
                    scores[row, torch.tensor(sorted(seen_items), dtype=torch.long)] = -torch.inf
            top_items = torch.topk(scores, k=k, dim=1).indices.tolist()
            recommendations.extend(top_items)
    return recommendations


def build_validation_examples(data, max_seq_len, users):
    '''用 train 历史构造输入，目标是 validation 的物品。'''
    histories = {}
    for row in data.train:
        histories.setdefault(row.user_idx, []).append((row.timestamp, row.item_idx))
    histories = {
        user_idx: [item_idx for _, item_idx in sorted(rows)]
        for user_idx, rows in histories.items()
    }

    user_indices = []
    input_sequences = []
    targets = []
    seen_items_list = []
    for row in data.validation:
        if row.user_idx not in users:
            continue
        history = histories.get(row.user_idx, [])
        if not history:
            continue
        user_indices.append(row.user_idx)
        input_sequences.append(pad_item_sequence(history, max_seq_len))
        targets.append(row.item_idx)
        seen_items_list.append(data.validation_seen[row.user_idx])
    return user_indices, input_sequences, targets, seen_items_list
```

</details>

`build_validation_examples` 只从 `data.train` 取模型输入，`data.validation` 只用来拿目标。过滤候选时还要包含目标之前播放过但没达到正反馈阈值的电影，因此用 `data.validation_seen`，不能只用正反馈序列的 `set(history)`。

`scores[row, seen] = -inf` 是原地修改，改的是这一批自己的张量，不会影响别处。用 `0.0` 代替 `-inf` 是不行的：0 分可能仍然排在前面——分数本来就不保证是负的。"""
    )
)

# ---------------------------------------------------------------- 15
CELLS.append(
    code(
        """# 不用改：一组实验的完整流程（训练 + 在固定用户上评测）
TRAIN_USERS = 6037      # 固定训练用户 index 范围；有效序列用户由 Dataset 再筛选
TRAIN_EPOCHS = 30       # 每组训练的轮数，和 src/recsys/training/train_sasrec.py 的 max_epochs 一致
EVAL_USERS = sorted({row.user_idx for row in data.validation})[:800]

# 训练子集固定下来，三组用的是同一批用户
labkit.set_seed(42)
TRAIN_INTERACTIONS = [row for row in data.train if row.user_idx < TRAIN_USERS]
print(f'训练用户数: {len({row.user_idx for row in TRAIN_INTERACTIONS})}')
print(f'固定评测用户数: {len(EVAL_USERS)}')
print(f'每组 {TRAIN_EPOCHS} 个 epoch，batch_size=256')


def run_config(name, max_seq_len, epochs=TRAIN_EPOCHS, batch_size=256,
               learning_rate=1e-3, dropout=0.2, use_causal_mask=True, eval_every=5):
    '''一组完整实验：各自 set_seed、各自重建 loader、固定评测用户上算 Recall@20。'''
    # 受控实验的纪律：每一组都从同一个 seed 重新开始，并且重建 loader。
    # 复用同一个 loader 会让结果随执行顺序漂移。
    labkit.set_seed(42)
    model = SASRec(num_items=data.num_items, max_seq_len=max_seq_len, embedding_dim=64,
                   num_blocks=2, num_heads=2, dropout=dropout)
    if not use_causal_mask:
        # 把因果约束整条去掉：attn_mask 永远不挡任何位置。
        # 注意这里是挂在**实例**上，不是类上，所以要先用 __get__ 绑上 self
        for block in model.blocks:
            block.forward = _bidirectional_forward.__get__(block, type(block))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dataset = SASRecTrainDataset(TRAIN_INTERACTIONS, num_items=data.num_items,
                                 max_seq_len=max_seq_len, seed=42)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    _, sequences, targets, seen_items_list = build_validation_examples(
        data, max_seq_len=max_seq_len, users=EVAL_USERS
    )

    history = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, torch.device('cpu'))
        if epoch % eval_every == 0 or epoch == epochs:
            recommendations = recommend_top_k(model, sequences, seen_items_list, k=20)
            hits = sum(
                1
                for row, recommendation in enumerate(recommendations)
                if targets[row] in recommendation
            )
            recall = hits / len(recommendations)
            history.append((epoch, loss, recall))
            print(f'  epoch {epoch:02d}  loss {loss:.4f}  val Recall@20 {recall:.4f}'
                  f'   （累计 {time.time() - started:.0f}s）')
    print(f'  {name} 完成: {time.time() - started:.0f}s')
    return model, history


def _bidirectional_forward(self, hidden_states, causal_mask, padding_mask):
    '''去掉因果约束的 block：mask 全 False，其余和 Lab 4 一样。
    用 __get__ 绑到实例上，所以这里要带 self。'''
    normalized = self.attention_norm(hidden_states)
    attention_output, _ = self.attention(
        normalized, normalized, normalized,
        attn_mask=torch.zeros_like(causal_mask),
        key_padding_mask=padding_mask,
        need_weights=False,
    )
    hidden_states = hidden_states + self.attention_dropout(attention_output)
    hidden_states = hidden_states.masked_fill(padding_mask.unsqueeze(-1), 0.0)
    normalized = self.feed_forward_norm(hidden_states)
    feed_forward_output = self.feed_forward(normalized)
    hidden_states = hidden_states + self.feed_forward_dropout(feed_forward_output)
    return hidden_states.masked_fill(padding_mask.unsqueeze(-1), 0.0)


print('随机水平（Recall@20 的下界参考）:', 20 / data.num_items)"""
    )
)

# ---------------------------------------------------------------- 16
CELLS.append(
    code(
        """# ✅ 跑一下：A) 标准配置
print('A) max_seq_len=50，带因果 mask')
model_a, history_a = run_config('A', max_seq_len=50)
best_a = max(history_a, key=lambda row: row[2])
print(f'   -> best epoch {best_a[0]}, loss {best_a[1]:.4f}, val Recall@20 {best_a[2]:.4f}')"""
    )
)

# ---------------------------------------------------------------- 17
CELLS.append(
    code(
        """# ✅ 跑一下：B) 只改序列长度：10
print('B) max_seq_len=10，带因果 mask')
model_b, history_b = run_config('B', max_seq_len=10)
best_b = max(history_b, key=lambda row: row[2])
print(f'   -> best epoch {best_b[0]}, loss {best_b[1]:.4f}, val Recall@20 {best_b[2]:.4f}')"""
    )
)

# ---------------------------------------------------------------- 18
CELLS.append(
    code(
        """# ✅ 跑一下：C) 只改因果约束：去掉 mask，让每个位置直接看到未来
print('C) max_seq_len=50，去掉因果 mask')
model_c, history_c = run_config('C', max_seq_len=50, use_causal_mask=False)
best_c = max(history_c, key=lambda row: row[2])
print(f'   -> best epoch {best_c[0]}, loss {best_c[1]:.4f}, val Recall@20 {best_c[2]:.4f}')"""
    )
)

# ---------------------------------------------------------------- 19
CELLS.append(
    code(
        """# ✅ 跑一下：三组摆在一起看
print('配置              训练 loss         val Recall@20')
random_level = 20 / data.num_items
for name, history in (('A 标准 (L=50)', history_a), ('B 短序列 (L=10)', history_b),
                      ('C 去掉因果 mask', history_c)):
    first_loss, last_loss = history[0][1], history[-1][1]
    best = max(row[2] for row in history)
    print(f'{name:18s} {first_loss:.4f} -> {last_loss:.4f}   {best:.4f}')
print(f'随机水平           --                    {random_level:.4f}')

check('5-6 三组都明显高于随机水平',
      min(best_a[2], best_b[2], best_c[2]) > 5 * random_level, True)
check('5-6 序列更长的一组不会更差（L=50 >= L=10）', best_a[2] >= best_b[2], True)
check('5-6 去掉因果 mask 的那组训练 loss 更低', best_c[1] < best_a[1], True)

# 数字固定下来：换 torch / numpy 版本导致随机流不同时，这里会直接报出来，
# 而不是让正文里引用的数悄悄变掉。改动这一章之后要同步更新这两个基准。
check_close('5-6 A 组 val Recall@20 与基准一致（参考值 0.2300）', best_a[2], 0.2300, tolerance=1e-4)
check_close('5-6 C 组 val Recall@20 与基准一致（参考值 0.1938）', best_c[2], 0.1938, tolerance=1e-4)

print()
print('C 组（去掉因果 mask）: 训练 loss', f'{best_c[1]:.4f}',
      '，val Recall@20', f'{best_c[2]:.4f}')
print('A 组（标准）        : 训练 loss', f'{best_a[1]:.4f}',
      '，val Recall@20', f'{best_a[2]:.4f}')
print('两组的训练 loss 差', f'{best_c[1] - best_a[1]:+.4f}',
      '，val Recall@20 差', f'{best_c[2] - best_a[2]:+.4f}')"""
    )
)

# ---------------------------------------------------------------- 17
CELLS.append(
    markdown(
        """## 📖 收束：三组结果说明什么

上面的对照表要横着看：三组的模型结构、参数、优化器、batch 顺序完全一样，只差 `max_seq_len` 或者因果约束。三组都是 30 个 epoch；到最后一轮 validation 还在涨，所以表里的数是同一个预算下的横向比较，不是收敛值。

**关于 loss 和指标。** C 组的训练 loss 明显最低（0.4595 对 0.5608，差 0.10），拟合训练集最容易——它每个位置的隐状态能看到整条序列，而每个位置的答案就在被看的范围里。

**但它的 validation Recall 仍低于标准组**（0.1938 对 0.2300）。这一组的读法比上一版更有意思：

- 训练方式给了它一个捷径：位置 `t` 的目标就是输入的第 `t+1` 个 token，而它能直接看到那个 token。所以 loss 最低。
- 这个捷径在**训练预算小的时候**甚至能换到一点指标（10 轮时它是 0.1487，标准组 0.1263）。原因是预算小时模型还没学会靠历史推断，能抄就抄反而占便宜。
- 预算给足之后就反过来了（30 轮：0.1938 对 0.2300）。抄答案的那部分参数没有在学"从历史推断"，到后面就落后了。

**这个翻转本身是这一章最值得记的一件事：** `loss 更低`、`短预算下指标更高`，两个信号都指向"这个改动更好"，而它最终是错的。判断改动有没有效只能看固定协议下的指标，而且**要看足够长的预算**——短预算的结论可能反过来。

保留因果 mask 的真正理由和指标无关：**部署时拿不到未来。** 位置 `t` 的目标在线上是不存在的，训练时让它可见就是训练和服务不一致。

**关于序列长度。** L=50 比 L=10 多出 0.0525（0.2300 对 0.1775）。上下文更长，模型能看到更远的兴趣。但 50 也不是越大越好：注意力是 `O(L²)`，而且这份数据里用户序列长度中位数只有 56，L 加到 200 时多出来的位置大多是补齐位。

**这一章把配置定在 A。** 选择依据是 validation 上的 Recall@20，不是训练 loss。test 要到 Lab 6 配置锁定之后再跑一次。"""
    )
)

# ---------------------------------------------------------------- 18
CELLS.append(
    markdown(
        """## 📖 理解一下：为什么需要这么多轮

看 A 组的曲线，30 个 epoch 之后 validation 还在涨：

```text
epoch 05  loss 0.8915  val Recall@20 0.1025
epoch 15  loss 0.6904  val Recall@20 0.1762
epoch 30  loss 0.5608  val Recall@20 0.2300
```

和 Two-Tower 比一下就有感觉了：那边 5 个 epoch 就能在 validation 上选出配置，这边 30 轮还在稳步上升。原因是每个位置的监督太弱：

- Two-Tower 的 in-batch 目标，一个 `B=256` 的 batch 有 `256 × 255` 个候选位置；
- SASRec 每个位置只有**一个**负样本，一个 `[256, 50]` 的 batch 最多 12800 个位置，其中还有大量补齐位。

候选少、信号弱，收敛自然慢。想加快有两条路：每个位置采多个负样本（5~10 个），或者对全部物品做 full softmax 交叉熵（等价于把整个物品表当候选，贵得多）。两条都会引入新的开销或超参。

这也是为什么序列模型的训练轮数通常比 ID-only 的双塔多。`src/recsys/training/train_sasrec.py` 用的是 30 epochs + patience=5，而且最好的一轮出现在第 30 轮——也就是说 30 轮这个预算刚刚够，再少一点就会明显偏低。

顺带一个观察：A 组的 loss 一直在降，最后一轮是 0.5608，而 C 组（去掉因果 mask）的同轮 loss 只有 0.4595。**两个数不是同一个任务的 loss**，不能直接比——两组能看到的输入范围不同。

另外一件事这里看不出来但工程上重要：**每个 epoch 都会重新采一轮负样本**。同一个正样本在不同 epoch 配不同的负样本，这相当于数据增强，但也意味着不同 epoch 的 loss 不是同一个任务的 loss，跨 epoch 比较要小心。"""
    )
)

# ---------------------------------------------------------------- 19
CELLS.append(
    code(
        """# 💾 导出这一章写的东西，Lab 6 直接 import
labkit.harvest(
    'myrec/sasrec_train.py',
    pad_item_sequence,
    sample_negative_item,
    SASRecTrainDataset,
    sasrec_loss,
    train_one_epoch,
    score_sequences,
    recommend_top_k,
    build_validation_examples,
    imports=[
        'import random',
        'import torch',
        'from torch.nn import functional as F',
        'from myrec.sasrec_model import SASRec',
    ],
)

import importlib

import myrec.sasrec_train as my_train

importlib.reload(my_train)

check('5-7 交接成功：myrec.sasrec_train 能用',
      hasattr(my_train, 'SASRecTrainDataset') and hasattr(my_train, 'train_one_epoch'), True)

# 光能 import 不够：真的用它建一个数据集、跑一个 batch
exported_dataset = my_train.SASRecTrainDataset(data.train, num_items=data.num_items,
                                               max_seq_len=8, seed=42)
exported_sample = exported_dataset[0]
check('5-7 导出的数据集能取样本',
      [len(exported_sample[key]) for key in exported_sample if key != 'user_idx'],
      [8, 8, 8])
exported_model = SASRec(num_items=data.num_items, max_seq_len=8, embedding_dim=64,
                        num_blocks=2, num_heads=2, dropout=0.0)
exported_loader = torch.utils.data.DataLoader(exported_dataset, batch_size=8)
with torch.no_grad():
    exported_batch = next(iter(exported_loader))
    exported_hidden = exported_model(exported_batch['input_sequence'])
    exported_loss = my_train.sasrec_loss(
        exported_hidden, exported_batch['positive_sequence'],
        exported_batch['negative_sequence'], exported_model,
    )
check('5-7 导出的 loss 能算', bool(torch.isfinite(exported_loss)), True)

# 把 A 组这一次训练的结果存下来，Lab 6 直接读这个文件
CHECKPOINT_PATH = labkit.state_path('sasrec_lab_trained.pt')
torch.save(
    {
        'model_config': {
            'num_items': data.num_items,
            'max_seq_len': 50,
            'embedding_dim': 64,
            'num_blocks': 2,
            'num_heads': 2,
            'dropout': 0.2,
        },
        'model_state_dict': model_a.state_dict(),
        'max_seq_len': 50,
        'train_users': TRAIN_USERS,
        'train_epochs': TRAIN_EPOCHS,
        'eval_users': len(EVAL_USERS),
        'best_epoch': best_a[0],
        'best_train_loss': best_a[1],
        'validation_recall20': best_a[2],
    },
    CHECKPOINT_PATH,
)
print('已写出', CHECKPOINT_PATH.name, '| 训练用户', TRAIN_USERS,
      '| best epoch', best_a[0], '| val Recall@20', f'{best_a[2]:.4f}')
check('5-7 A 组的 checkpoint 已经存下来', CHECKPOINT_PATH.exists(), True)"""
    )
)

# ---------------------------------------------------------------- 20
CELLS.append(
    markdown(
        """## 收尾

### 这个 lab 写了多少

| 自己写的 | 行数 | 脚手架给的 |
|---|---|---|
| `pad_item_sequence` | 5 | `run_config`（对照实验的骨架） |
| `sample_negative_item` | 8 | `SASRecTrainDataset.__init__` |
| `SASRecTrainDataset.__getitem__` | 22 | 数据、`check()` / `check_close()` |
| `sasrec_loss` | 15 | Lab 4 导出的模型 |
| `train_one_epoch` | 22 | |
| 三个评测小工具 | 40 | |

加起来 112 行左右。比前几个 lab 长，但一半是「把 Lab 2 写过的东西改成按序列输入」。

### 这一章应该能直接答上来

1. 输入序列和正样本序列为什么要错开一格？错开的方向反了会怎样？
2. 负采样的 `seen_items` 为什么只能用训练集？混进 validation 会出现什么现象？
3. loss 的掩码为什么用正样本序列而不是负样本序列？
4. `total_loss / len(dataloader)` 在什么情况下是错的？
5. 去掉因果 mask 之后训练 loss 更低，为什么 validation 指标在 10 轮时更高、30 轮时更低？

### 再往下

模型和数据都齐了，A 组那条权重也存下来了（`myrec/sasrec_lab_trained.pt`，带训练元信息）。Lab 6 要做的是把它接到一次完整的评测上：两种历史协议的差别、test 只跑一次，以及"同一份权重为什么会有两个数"。"""
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
    target = REPO_ROOT / "lab5-sasrec-data-and-train.ipynb"
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
    todo_ids = [cell["metadata"]["lab_todo"] for cell in todo_cells]
    answer_ids = [cell["metadata"]["lab_answer_for"] for cell in answers]
    print("TODO:", todo_ids)
    print("答案:", answer_ids)


if __name__ == "__main__":
    main()
