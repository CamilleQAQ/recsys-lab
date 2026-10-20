"""生成 labs/lab4-sasrec-model.ipynb。

单独用一个脚本生成 notebook，是为了避免手写 JSON 的转义问题，
也方便后面 lab5 / lab6 复用同一套格子约定。
"""

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
        """# Lab 4｜从零写 SASRec 的注意力主干

前三个 lab 里，用户被压成一个 64 维向量，模型只看得到「这个人是谁」。

这一章换一条路线：把用户最近的行为按时间排成一条序列，让**每一个位置**都去预测它的下一个物品。这样同一个用户的第 1 次行为和第 50 次行为各自产生一条训练信号，而不是共用一条用户向量。

要处理的新问题只有三个：

1. 序列要按位置查表，位置信息得自己加进去；
2. 每个位置只能看到它左边的东西——**未来还没发生**，看不到也不许看；
3. 补齐位置（没有历史的新用户）不能参与计算。

这一章先把模型主干写出来，训练目标留到 Lab 5。

符号沿用前几个 lab：📖 概念，🧩 要你写的代码，✅ 断言，🚀 可以改着玩的地方，💡 折叠的参考答案。"""
    )
)

# ---------------------------------------------------------------- 1
CELLS.append(
    code(
        """# 不用改：准备数据和检查工具
import math
import sys
from pathlib import Path

import torch
from torch import nn


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

# 这一章不 import 前三个 lab 的任何代码：注意力主干是独立的一层。
# 如果 myrec/ 已经存在，说明前面的 lab 跑过，对照时可以直接翻回去看。
try:
    import myrec  # noqa: F401

    print('myrec/ 已存在，前三个 lab 的产物可以直接对照')
except ModuleNotFoundError:
    print('还没跑过前三个 lab（这不影响本章）')

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
print('torch:', torch.__version__)"""
    )
)

# ---------------------------------------------------------------- 2
CELLS.append(
    code(
        """# 不用改：真实数据。每个用户的正反馈已经按 (timestamp, item_idx) 排好序
labkit.set_seed(42)
data = labkit.prepare_movielens()
print(data.describe())

sequences = {}
for row in data.train:
    sequences.setdefault(row.user_idx, []).append(row.item_idx)
for items in sequences.values():
    items.sort(key=lambda item_idx: item_idx)  # 占位，下面按时间重排

# 重新按时间排：前面那份只留了 item_idx，这里用原始顺序重建一次
by_user = {}
for row in data.train:
    by_user.setdefault(row.user_idx, []).append((row.timestamp, row.item_idx))
sequences = {
    user_idx: [item_idx for _, item_idx in sorted(rows)]
    for user_idx, rows in by_user.items()
}

lengths = torch.tensor([len(items) for items in sequences.values()])
print('用户数:', len(sequences))
print('序列长度 中位数/均值/最长:', int(lengths.median()), round(float(lengths.float().mean()), 1), int(lengths.max()))
print()
print('取一个用户看看他的行为序列（前 12 个）:')
example_user = max(sequences, key=lambda user_idx: len(sequences[user_idx]))
print('user_idx =', example_user, '长度 =', len(sequences[example_user]))
print(sequences[example_user][:12], '...')"""
    )
)

# ---------------------------------------------------------------- 3
CELLS.append(
    markdown(
        """## 📖 理解一下：把序列切成「用前 t 个预测第 t+1 个」

对一个长度为 6 的序列 `[a, b, c, d, e, f]`，切出来的是 5 组监督：

```text
输入 [a]          预测 b
输入 [a, b]       预测 c
输入 [a, b, c]    预测 d
...
```

这是自回归的做法，展开成 5 条独立样本会很浪费：第 5 条的前 4 个位置和第 4 条完全重复。

SASRec 的做法是**一次算完**。给模型整条 `[a, b, c, d, e]`，它同时输出 5 个位置的隐状态，每个位置各自去预测它的下一个。所以一个长度为 `L` 的序列，在一个 forward 里产出 `L` 条训练信号。

要让这件事成立，第 3 个位置的隐状态就不能看到第 4、5 个位置——否则它等于偷看了答案。约束「每个位置只能看左边」的机制就是因果 mask。"""
    )
)

# ---------------------------------------------------------------- 4
CELLS.append(
    markdown(
        """### 🧩 你来写：`build_causal_attention_mask()`

返回 `[seq_len, seq_len]` 的布尔张量，`True` 表示**不允许看**。

把它当成一张「行 = 查询位置，列 = 被看的位置」的表：第 `i` 行只允许看 `0..i`，所以 `j > i` 的位置是 `True`，其余是 `False`。"""
    )
)

CELLS.append(
    code(
        """def build_causal_attention_mask(seq_len, device):
    '''返回 [seq_len, seq_len] 的布尔张量，True 表示「未来位置，不许看」。'''
    # TODO 4-0：要的是严格上三角（不含对角线）
    #   提示：torch.triu / torch.ones + dtype=torch.bool
    raise NotImplementedError('TODO 4-0')""",
        lab_todo="4-mask",
    )
)

CELLS.append(
    code(
        """# ✅ 跑一下：形状对不对、对角线保没保住
small_mask = build_causal_attention_mask(4, torch.device('cpu'))

check('4-0 形状是 [4, 4]', list(small_mask.shape), [4, 4])
check('4-0 dtype 是 bool', small_mask.dtype, torch.bool)
check('4-0 对角线可以看自己（都是 False）', [bool(small_mask[i, i]) for i in range(4)], [False] * 4)
check('4-0 位置 0 看不到任何右边的东西', [bool(small_mask[0, j]) for j in range(4)], [False, True, True, True])
check('4-0 位置 3 是最后一行，谁都不能拦', [bool(small_mask[3, j]) for j in range(4)], [False, False, False, False])
check('4-0 被挡住的位置共 6 个 (1+2+3)', int(small_mask.sum()), 6)
check('4-0 拿到 CPU 上之后还在 CPU 上', small_mask.device.type, 'cpu')"""
    )
)

CELLS.append(
    markdown(
        lab_answer_for="4-mask",
        source="""<details>
<summary>💡 参考答案（<code>build_causal_attention_mask</code>）</summary>

```python
def build_causal_attention_mask(seq_len, device):
    '''返回 [seq_len, seq_len] 的布尔张量，True 表示「未来位置，不许看」。'''
    return torch.triu(
        torch.ones((seq_len, seq_len), dtype=torch.bool, device=device),
        diagonal=1,
    )
```

</details>

`diagonal=1` 是这里的要点：`diagonal=0` 会把对角线也设成 `True`，那样每个位置连自己都看不到，softmax 在一整行全 `-inf` 上会直接给出 `NaN`。

`nn.MultiheadAttention` 的 `attn_mask` 语义是「**True 的位置不参与注意力**」——和 `key_padding_mask` 一样是布尔屏蔽。写反了这个 mask，模型不但不会报错，还会在训练集上显得学得很快，因为每个位置都直接看到了答案。"""
    )
)

# ---------------------------------------------------------------- 5
CELLS.append(
    markdown(
        """## 📖 理解一下：一个 block 里的两个子层

一个 SASRec block 就是 transformer 的标准结构，只是把「不许看未来」写死在注意力里：

```text
x = x + Attention(Norm(x))        # 子层一：看左右的上下文（这里被限制成只能看左边）
x = x + FeedForward(Norm(x))      # 子层二：每个位置各自过一次两层 MLP
```

三处细节值得留意：

- **先 Norm 再进子层**（pre-LN）。原版 transformer 是 post-LN，训练更深时更容易不稳，pre-LN 省事。
- **残差连接把原始输入原封不动带过去**。没有它，几层之后每个位置的表示会互相糊在一起，位置自己的信息反而丢了。
- **FFN 里先升维到 4 倍再降回来**。注意力负责在位置之间搬信息，FFN 负责在每个位置内部加工。

还有一个容易漏的点：注意力会把补齐位置的信息也混进来。`key_padding_mask` 只保证补齐位置不被当成 key，补齐位置自己那一行的输出仍然是别的物品的加权平均。所以每过一个子层，都要把补齐位置重新清零一次。"""
    )
)

# ---------------------------------------------------------------- 6
CELLS.append(
    markdown(
        """### 🧩 你来写：`SASRecBlock.forward()`

参数都已经在 `__init__` 里建好了（脚手架给的），你要写的是 `forward`：

```text
输入 hidden_states [B, L, D]，causal_mask [L, L]，padding_mask [B, L]
子层一：attention_norm -> attention(attn_mask=causal_mask, key_padding_mask=padding_mask)
        -> attention_dropout -> 残差 -> 补齐位置清零
子层二：feed_forward_norm -> feed_forward -> feed_forward_dropout -> 残差 -> 补齐位置清零
返回 [B, L, D]
```

`padding_mask` 里 `True` 是补齐位置。清零用 `hidden_states.masked_fill(padding_mask.unsqueeze(-1), 0.0)`。"""
    )
)

CELLS.append(
    code(
        """class SASRecBlock(nn.Module):
    '''先做 LayerNorm 的因果自注意力，再接一个前馈网络。'''

    def __init__(self, embedding_dim, num_heads, dropout):
        super().__init__()
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.feed_forward_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )
        self.feed_forward_dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, causal_mask, padding_mask):
        # TODO 4-1：按上面写的两个子层实现，每个子层末尾都要把补齐位置清零
        #   注意力那句的返回值有两个，第二个（注意力权重）这里用不上，写成 _
        #   注意力要传 attn_mask=causal_mask 和 key_padding_mask=padding_mask
        raise NotImplementedError('TODO 4-1')""",
        lab_todo="4-block",
    )
)

CELLS.append(
    code(
        """# ✅ 跑一下：形状、补齐位置、以及「看不到未来」
labkit.set_seed(0)
block = SASRecBlock(embedding_dim=16, num_heads=2, dropout=0.0)
block.eval()

length = 5
hidden = torch.randn(2, length, 16)
causal = build_causal_attention_mask(length, torch.device('cpu'))
padding = torch.zeros(2, length, dtype=torch.bool)
padding[1, 3:] = True  # 第 2 条序列只有前 3 个位置是真实历史

output = block(hidden, causal, padding)

check('4-1 输出形状不变', list(output.shape), [2, length, 16])
check('4-1 补齐位置的输出是 0', float(output[1, 3:].abs().max().detach()), 0.0)
check('4-1 没被补齐的位置不是 0', bool(output[0].abs().max().detach() > 0), True)
check('4-1 残差确实带过来了（输出不等于全零）', bool(output[1, :3].abs().max().detach() > 0), True)

# 把最后一个位置换掉，它左边的输出必须逐位不变——这就是因果性
changed = hidden.clone()
changed[:, -1] = torch.randn(2, 16)
with torch.no_grad():
    output_changed = block(changed, causal, padding)
    # float32 在 CPU 上重算同一行也会有几 e-7 的舍入差，所以阈值取「远小于真实信号」
    effect_of_future = float((output_changed[:, :-1] - output[:, :-1]).abs().max())
print('改动最后一个位置，前面位置的输出最大变化:', effect_of_future)
check('4-1 改未来位置，前面位置的输出不变', effect_of_future < 1e-5, True)

# 反例：把 mask 全设成 False，未来就漏进来了
with torch.no_grad():
    leaky = block(hidden, torch.zeros(length, length, dtype=torch.bool), padding)
    leaky_changed = block(changed, torch.zeros(length, length, dtype=torch.bool), padding)
    effect_without_mask = float((leaky_changed[:, :-1] - leaky[:, :-1]).abs().max())
print('同样的改动，去掉 mask 之后的最大变化:', effect_without_mask)
check('4-1 去掉 mask 之后未来真的会漏进来（说明这个检查有效）',
      effect_without_mask > 1e-5, True)"""
    )
)

CELLS.append(
    markdown(
        lab_answer_for="4-block",
        source="""<details>
<summary>💡 参考答案（<code>SASRecBlock.forward</code>）</summary>

```python
class SASRecBlock(nn.Module):
    '''先做 LayerNorm 的因果自注意力，再接一个前馈网络。'''

    def __init__(self, embedding_dim, num_heads, dropout):
        super().__init__()
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.feed_forward_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )
        self.feed_forward_dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, causal_mask, padding_mask):
        normalized = self.attention_norm(hidden_states)
        attention_output, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        hidden_states = hidden_states + self.attention_dropout(attention_output)
        hidden_states = hidden_states.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        normalized = self.feed_forward_norm(hidden_states)
        feed_forward_output = self.feed_forward(normalized)
        hidden_states = hidden_states + self.feed_forward_dropout(feed_forward_output)
        return hidden_states.masked_fill(padding_mask.unsqueeze(-1), 0.0)
```

</details>

`padding_mask.unsqueeze(-1)` 把 `[B, L]` 变成 `[B, L, 1]`，才能和 `[B, L, D]` 逐元素对齐——广播时最后一维复制 `D` 次。

清零那句放在残差之后。放前面等于让补齐位置参与计算：注意力那边虽然被 `key_padding_mask` 挡住了 key，但这些位置自己那一行仍然会算出别的东西，再顺着残差和后面几层流到有效位置上。"""
    )
)

# ---------------------------------------------------------------- 7
CELLS.append(
    markdown(
        """## 📖 理解一下：输入层的三件事

序列要先变成 `[B, L, D]` 的向量才能进 block。这里有三步：

**1. 物品查表，并且整体挪一位。** `nn.Embedding(num_items + 1, D)`，把 token `0` 让给补齐位（`padding_idx=0`）。内部 `item_idx = i` 的物品用 token `i + 1` 表示，所以查表时要 `+1`，给全部物品打分时要用 `item_embedding.weight[1:]`。

**2. 加位置编码。** 自注意力本身没有「第几个」这个概念——它只按相似度在向量之间搬信息。`nn.Embedding(max_seq_len, D)` 把位置也变成一个可学的向量加进去，模型才有办法区分「先看 A 再看 B」和「先看 B 再看 A」。

这里有个容易想错的地方值得先记下来：**加上因果 mask 之后，光靠 mask 本身已经能区分一部分顺序了**。位置 1 看到 `[a]`、位置 2 看到 `[a, b]`，下一层注意力的 key 就带上了这段前缀的信息。所以位置编码不是「唯一的顺序来源」，而是让顺序信息在**第一层**就可用、不必靠多层注意力一层层传下去。下一格的自测就检查它有没有被真的用上。

**3. 乘 `sqrt(D)`。** 物品向量和位置向量是直接相加的，如果两者量级差太多，小的那一半等于没加。乘 `sqrt(D)` 是原版 transformer 留下的习惯做法，把查表出来的量级抬到和位置向量可比。

进注意力之前还有一步：把补齐位置整体置零。

```text
token      = item_idx + 1                    # 0 留给补齐
hidden     = item_embedding(token) * sqrt(D) + position_embedding(0..L-1)
hidden     = dropout(hidden)
hidden     = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
```

> 完整推导与文献：[03｜注意力凭什么不能看未来](../docs/03-sasrec/03-causality-and-position.md)"""
    )
)

# ---------------------------------------------------------------- 8
CELLS.append(
    markdown(
        """### 🧩 你来写：`SASRec.forward()`

类的骨架在上一格：`__init__`（两张表、dropout、block 列表、output_norm）和 `_reset_parameters` 已经给好了，接口固定成：

```text
forward(item_sequence)          -> [B, L, D]
sequence_representation(seq)    -> [B, D]     （下一个概念再写）
score_all_items(seq)            -> [B, num_items]
```

`forward` 按上面那几行来，最后逐个 block 过一遍，再 `output_norm`、再清零补齐位置。"""
    )
)

CELLS.append(
    code(
        """class SASRec(nn.Module):
    '''根据因果编码后的物品序列，预测下一个物品。'''

    def __init__(self, num_items, max_seq_len=50, embedding_dim=64, num_blocks=2,
                 num_heads=2, dropout=0.2):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError('embedding_dim must be divisible by num_heads')
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [SASRecBlock(embedding_dim, num_heads, dropout) for _ in range(num_blocks)]
        )
        self.output_norm = nn.LayerNorm(embedding_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        '''用较小方差初始化 embedding，并把补齐位置的向量固定为零。'''
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()"""
    )
)

CELLS.append(
    code(
        """def sasrec_forward(self, item_sequence):
    # item_sequence: [batch_size, seq_len]，0 表示补齐位置
    batch_size, seq_len = item_sequence.shape
    if seq_len > self.max_seq_len:
        raise ValueError('sequence is longer than max_seq_len')
    padding_mask = item_sequence.eq(0)
    # TODO 4-2a：位置向量 + 查表乘 sqrt(D) + dropout + 补齐位置清零
    # TODO 4-2b：建因果 mask，过完全部 block，output_norm，再清零一次
    #   每个位置都要算，不只是最后一个
    raise NotImplementedError('TODO 4-2')


SASRec.forward = sasrec_forward""",
        lab_todo="4-embed",
    )
)

# ---------------------------------------------------------------- 9
CELLS.append(
    code(
        """# 不用改：先验证一下「位置编码到底有没有被用上」
# 把动作反过来做：先看置零之后会怎样，再加回去
labkit.set_seed(1)
toy = SASRec(num_items=20, max_seq_len=6, embedding_dim=16, num_blocks=1,
             num_heads=2, dropout=0.0).eval()
probe = torch.tensor([[1, 2, 3, 4, 5, 6]])

with torch.no_grad():
    before = toy(probe)
    # 原地改一个 requires_grad 的参数，必须放在 no_grad 里
    toy.position_embedding.weight.zero_()
    without_position = toy(probe)

    difference = float((before - without_position).abs().max())
    touched_positions = float(before[0].ne(without_position[0]).any(dim=-1).sum())
print('把位置表置零，输出变化:', difference)
check('4-x 位置编码确实参与了前向（置零后输出变了）', difference > 1e-5, True)
check('4-x 位置编码不是可有可无的点缀：它改动了每一个位置的输出',
      touched_positions, 6.0)

# 顺手确认一件事：mask 挡住的是未来，不是过去
changed_past = probe.clone()
changed_past[0, 0] = 9
with torch.no_grad():
    moved_past = toy(changed_past)
    history_effect = float((moved_past[0, 1:] - without_position[0, 1:]).abs().max())
check('4-x 改动过去的位置，后面的输出会跟着变（mask 没有挡住历史）',
      history_effect > 1e-5, True)"""
    )
)

CELLS.append(
    code(
        """# ✅ 跑一下：形状、补齐位、可复现性、参数量
labkit.set_seed(2)
toy = SASRec(num_items=20, max_seq_len=6, embedding_dim=16, num_blocks=2,
             num_heads=2, dropout=0.0).eval()

toy_sequence = torch.tensor([
    [1, 2, 3, 4, 5, 6],
    [7, 8, 0, 0, 0, 0],
])
with torch.no_grad():
    toy_output = toy(toy_sequence)

check('4-2 输出形状是 [B, L, D]', list(toy_output.shape), [2, 6, 16])
check('4-2 token 0（补齐位）的输出是 0', float(toy_output[1, 2:].abs().max().detach()), 0.0)
check('4-2 真实位置的输出不是 0', bool(toy_output[1, :2].abs().max().detach() > 0), True)
check('4-2 补齐位的物品向量是 0（padding_idx 生效）',
      float(toy.item_embedding.weight[0].abs().max().detach()), 0.0)

# 只给前几个位置喂数据，后面全是补齐位：这就是没有历史的新用户
with torch.no_grad():
    empty_history = toy(torch.zeros(1, 6, dtype=torch.long))
    single_history = toy(torch.tensor([[3, 0, 0, 0, 0, 0]]))
check('4-2 全零序列不会崩：左边没有东西可看',
      bool(torch.isfinite(empty_history).all()), True)
check('4-2 只有一个位置的序列也能算', list(single_history.shape), [1, 6, 16])

# 参数量：把两个 block 拆开算一遍。MultiheadAttention 的参数是
# in_proj_weight [3D, D] + in_proj_bias [3D] + out_proj_weight [D, D] + out_proj_bias [D]
D, H = 16, 2
just_items = (20 + 1) * D
attention_parameters = 3 * D * D + 3 * D + D * D + D
layer_norm_parameters = 2 * D
feed_forward_parameters = D * 4 * D + 4 * D + 4 * D * D + D
expected_parameters = (
    (20 + 1) * D                                  # item_embedding
    + 6 * D                                       # position_embedding
    + 2 * (attention_parameters + 2 * layer_norm_parameters + feed_forward_parameters)
    + 2 * D                                       # output_norm
)
actual_parameters = sum(parameter.numel() for parameter in toy.parameters())
block_parameters = 2 * (attention_parameters + 2 * layer_norm_parameters + feed_forward_parameters)
print('参数量:', actual_parameters, '| 物品表:', just_items, '| 两个 block:', block_parameters)
check('4-2 参数量对得上手工计算', actual_parameters, expected_parameters)
check('4-2 block 的参数是物品表之外的那部分',
      actual_parameters - just_items,
      block_parameters + 6 * D + 2 * D)

# dropout 的开关由「构造时的参数」和「train() / eval()」共同决定
toy.train()
with torch.no_grad():
    first_pass = toy(toy_sequence)
    second_pass = toy(toy_sequence)
    same_without_dropout = float((first_pass - second_pass).abs().max())
check('4-2 dropout=0.0 时，train 模式下两次前向也一样', same_without_dropout, 0.0)

labkit.set_seed(7)
noisy = SASRec(num_items=20, max_seq_len=6, embedding_dim=16, num_blocks=2,
               num_heads=2, dropout=0.5).train()
with torch.no_grad():
    first_pass = noisy(toy_sequence)
    second_pass = noisy(toy_sequence)
    noisiness = float((first_pass - second_pass).abs().max())
check('4-2 dropout=0.5 且 train 模式时，两次前向不同', noisiness > 0, True)

noisy.eval()
with torch.no_grad():
    first_pass = noisy(toy_sequence)
    second_pass = noisy(toy_sequence)
    stable = float((first_pass - second_pass).abs().max())
check('4-2 切回 eval 之后又稳定了（评估必须调 model.eval()）', stable, 0.0)"""
    )
)

CELLS.append(
    markdown(
        lab_answer_for="4-embed",
        source="""<details>
<summary>💡 参考答案（<code>SASRec.forward</code>）</summary>

```python
def sasrec_forward(self, item_sequence):
    batch_size, seq_len = item_sequence.shape
    if seq_len > self.max_seq_len:
        raise ValueError('sequence is longer than max_seq_len')
    padding_mask = item_sequence.eq(0)

    position_ids = torch.arange(seq_len, device=item_sequence.device)
    position_embeddings = self.position_embedding(position_ids)
    hidden_states = self.item_embedding(item_sequence) * math.sqrt(self.embedding_dim)
    hidden_states = hidden_states + position_embeddings.unsqueeze(0)
    hidden_states = self.input_dropout(hidden_states)
    hidden_states = hidden_states.masked_fill(padding_mask.unsqueeze(-1), 0.0)

    causal_mask = build_causal_attention_mask(seq_len, item_sequence.device)
    for block in self.blocks:
        hidden_states = block(hidden_states, causal_mask, padding_mask)
    hidden_states = self.output_norm(hidden_states)
    hidden_states = hidden_states.masked_fill(padding_mask.unsqueeze(-1), 0.0)
    return hidden_states


SASRec.forward = sasrec_forward
```

</details>

`position_embeddings.unsqueeze(0)` 把 `[L, D]` 变成 `[1, L, D]`，靠广播加给 `[B, L, D]` 里的每一条序列。

`padding_idx=0` 会把第 0 行的梯度一直保持为 0，但初始化时它仍然是个随机向量。所以 `_reset_parameters()` 里手动 `zero_()` 一次，让「补齐 = 全零向量」这个约定立刻成立——否则补齐位一开始会带着一个随机方向进网络。

注意 `forward` 里没有 `padding_mask` 之外的任何条件分支：有没有历史、历史有多长，都靠这一张 mask 表达。"""
    )
)

# ---------------------------------------------------------------- 10
CELLS.append(
    markdown(
        """## 📖 理解一下：整条序列只要一个向量

训练时每个位置都有自己的监督信号，但**预测时**要的是「这条序列代表什么」。做法是取最后一个有效位置的隐状态：

```text
valid_lengths = (item_sequence != 0).sum(dim=1)     # 每条序列真实有多少个物品
last_positions = valid_lengths - 1
representation = hidden_states[行, last_positions]  # [B, D]
```

为什么是最后一个有效位置，而不是所有位置取平均：因果编码下，第 `L` 个位置的隐状态是通过注意力**汇总了它左边全部位置**之后得到的，它本来就承载了整条序列的信息；取平均会把早年的兴趣和最近的兴趣等权混在一起，而推荐里「最近看了什么」的权重通常更高。

这里有一个和补齐方向绑死的约定：**补齐只能加在右边**。项目里的 `pad_item_sequence` 就是右侧补 0。如果改成左补齐，`valid_lengths - 1` 会落到一个补齐位置上，取出来的表示就是个全零向量——不报错，只是推荐质量莫名其妙地差。"""
    )
)

# ---------------------------------------------------------------- 11
CELLS.append(
    markdown(
        """### 🧩 你来写：`sequence_representation()`

取每条序列最后一个非补齐位置的隐状态，返回 `[B, D]`。要求：整条序列全是补齐位（`valid_length` 为 0）时抛 `ValueError`，不要静默返回零向量。

写法上和 `SASRec.forward` 一样：定义成函数，再挂到类上。"""
    )
)

CELLS.append(
    code(
        lab_todo="4-represent",
        source="""def sequence_representation(self, item_sequence):
    '''取出每条序列「最后一个非补齐位置」的隐状态，作为整条序列的表示。'''
    # TODO 4-3：先 forward，再算 valid_lengths，最后按行取
    #   取 [B, L, D] 里每行的第 last_positions 个：hidden_states[行下标, 列下标]
    #   整条序列全是补齐位时要抛 ValueError，不要静默返回零向量
    raise NotImplementedError('TODO 4-3')


SASRec.sequence_representation = sequence_representation""",
    )
)

CELLS.append(
    code(
        """# ✅ 跑一下：形状、取的是哪个位置、以及左补齐会怎样
labkit.set_seed(3)
toy = SASRec(num_items=20, max_seq_len=6, embedding_dim=16, num_blocks=2,
             num_heads=2, dropout=0.0).eval()
toy_sequence = torch.tensor([
    [1, 2, 3, 4, 5, 6],
    [7, 8, 0, 0, 0, 0],
])
with torch.no_grad():
    representation = toy.sequence_representation(toy_sequence)
    hidden_states = toy(toy_sequence)
    last_position_gap = float((representation[0] - hidden_states[0, 5]).abs().max())
    second_last_gap = float((representation[1] - hidden_states[1, 1]).abs().max())

check('4-3 形状是 [B, D]', list(representation.shape), [2, 16])
check('4-3 第 1 条序列取的是第 5 个位置（最后一个有效位）', last_position_gap, 0.0)
check('4-3 第 2 条序列取的是第 1 个位置，不是补齐位', second_last_gap, 0.0)
check('4-3 有历史时的表示不是零向量', bool(representation[1].abs().max().detach() > 0), True)

# 左补齐的危险：最后一个位置就是补齐位，取出来是全零
left_padded = torch.tensor([[0, 0, 0, 0, 3, 4]])
with torch.no_grad():
    left_representation = toy.sequence_representation(left_padded)
    left_norm = float(left_representation.norm())
print('左补齐序列的表示范数:', left_norm)
check('4-3 左补齐会取到补齐位，得到零向量（所以只能右补齐）', left_norm, 0.0)

# 全零序列：没有任何历史，应该报错而不是返回零向量
try:
    with torch.no_grad():
        toy.sequence_representation(torch.zeros(1, 6, dtype=torch.long))
except ValueError as error:
    print('全零序列 ->', type(error).__name__, error)
    raised = True
else:
    raised = False
check('4-3 全零序列抛 ValueError', raised, True)"""
    )
)

CELLS.append(
    markdown(
        lab_answer_for="4-represent",
        source="""<details>
<summary>💡 参考答案（<code>sequence_representation</code>）</summary>

```python
def sequence_representation(self, item_sequence):
    '''取出每条序列「最后一个非补齐位置」的隐状态，作为整条序列的表示。'''
    hidden_states = self.forward(item_sequence)
    valid_lengths = item_sequence.ne(0).sum(dim=1)
    if torch.any(valid_lengths == 0):
        raise ValueError('each sequence must contain at least one item')
    last_positions = valid_lengths - 1
    batch_indices = torch.arange(item_sequence.shape[0], device=item_sequence.device)
    return hidden_states[batch_indices, last_positions]


SASRec.sequence_representation = sequence_representation
```

</details>

`hidden_states[batch_indices, last_positions]` 是「按行取不同列」的写法：两个长度都是 `B` 的张量配对之后，取的是 `(0, last[0])`、`(1, last[1])`…… 结果形状 `[B, D]`。

为什么不用 `for` 循环：这条路径在评测时要对全部用户跑一遍，Python 循环会变成瓶颈；而且评测本身就是按 batch 走的，循环写法会让「逐用户」这件事看起来是必要的，其实不是。"""
    )
)

# ---------------------------------------------------------------- 12
CELLS.append(
    markdown(
        """## 📖 理解一下：给全部候选物品打分

训练完模型之后，评测要的是「这条序列会给每个物品打多少分」。做法很直接：

```text
representation = sequence_representation(seq)   # [B, D]
candidates     = item_embedding.weight[1:]      # [num_items, D]，去掉第 0 行
scores         = representation @ candidates.T  # [B, num_items]
```

`score_all_items` 是 Lab 6 的入口。这里不做 softmax——评测只关心名次；也不做 L2 归一化，和 Two-Tower 不一样，那边归一化是为了让 in-batch 交叉熵的分母可比。

`item_embedding.weight[1:]` 的第 `i` 行就是内部 `item_idx = i` 的物品向量，所以**列下标可以直接当 `item_idx` 用**。这条对应关系是查表时 `+1` 的另一半，两处必须成对出现。"""
    )
)

CELLS.append(
    markdown(
        """### 🧩 你来写：`score_all_items()`

和 `sequence_representation` 一样，定义成函数再挂到 `SASRec` 上。"""
    )
)

CELLS.append(
    code(
        lab_todo="4-score",
        source="""def score_all_items(self, item_sequence):
    '''给全部候选物品打分，分数是它们与序列末位表示的相似度。'''
    # TODO 4-4：取序列表示，和 item_embedding.weight[1:] 做矩阵乘法
    #   注意不要用第 0 行——那是补齐 token，不是任何真实物品
    raise NotImplementedError('TODO 4-4')


SASRec.score_all_items = score_all_items""",
    )
)

CELLS.append(
    code(
        """# ✅ 跑一下：形状、和手工矩阵乘法是否一致
labkit.set_seed(4)
toy = SASRec(num_items=20, max_seq_len=6, embedding_dim=16, num_blocks=2,
             num_heads=2, dropout=0.0).eval()
toy_sequence = torch.tensor([
    [1, 2, 3, 4, 5, 6],
    [7, 8, 0, 0, 0, 0],
])
with torch.no_grad():
    scores = toy.score_all_items(toy_sequence)
    representation = toy.sequence_representation(toy_sequence)
    manual_scores = representation @ toy.item_embedding.weight[1:].T
    scores_gap = float((scores - manual_scores).abs().max())

check('4-4 分数矩阵形状是 [B, num_items]', list(scores.shape), [2, 20])
check('4-4 和手工矩阵乘法一致', scores_gap, 0.0)
check('4-4 分数是有限的', bool(torch.isfinite(scores).all()), True)

# 列下标就是 item_idx：第 0 列对应 item_idx = 0，而不是补齐 token
with torch.no_grad():
    column_zero = (representation[0] * toy.item_embedding.weight[1]).sum()
    print('第 0 列 vs 手工算 item_idx=0:', float(scores[0, 0]), float(column_zero))
    column_matches = bool(torch.allclose(scores[0, 0], column_zero))
check('4-4 第 0 列是 item_idx = 0 的物品，不是补齐 token', column_matches, True)

top_k = torch.topk(scores[0], k=3).indices.tolist()
print('第 1 条序列的 Top-3 item_idx:', top_k)
check('4-4 Top-K 的下标落在 [0, num_items)', all(0 <= item_idx < 20 for item_idx in top_k), True)"""
    )
)

CELLS.append(
    markdown(
        lab_answer_for="4-score",
        source="""<details>
<summary>💡 参考答案（<code>score_all_items</code>）</summary>

```python
def score_all_items(self, item_sequence):
    '''给全部候选物品打分，分数是它们与序列末位表示的相似度。'''
    representation = self.sequence_representation(item_sequence)
    candidate_embeddings = self.item_embedding.weight[1:]
    scores = representation @ candidate_embeddings.T
    return scores


SASRec.score_all_items = score_all_items
```

</details>

这张分数矩阵是 `[B, num_items]`，评测时每一行都会被排序一次。`num_items = 3533` 的时候这没什么；换成上亿物品就得换成检索结构，那是另一条路线。

序列模型打分不做 L2 归一化，也不做 softmax——它只负责排序，分数只要单调就够了。这一点和 Two-Tower 不一样：那边归一化之后点积才等价于余弦相似度，是为了让 in-batch 的交叉熵分母可比。"""
    )
)

# ---------------------------------------------------------------- 13
CELLS.append(
    code(
        """# 💾 导出这一章写的东西，Lab 5 直接 import
labkit.harvest(
    'myrec/sasrec_model.py',
    build_causal_attention_mask,
    SASRecBlock,
    SASRec,
    imports=['import math', 'import torch', 'from torch import nn'],
)

import importlib

import myrec.sasrec_model as my_sasrec

importlib.reload(my_sasrec)

check('4-5 交接成功：myrec.sasrec_model 能用',
      hasattr(my_sasrec, 'SASRec') and hasattr(my_sasrec, 'build_causal_attention_mask'), True)

# 光能 import 不够：真正跑一遍，确认导出的文件是完整的
exported = my_sasrec.SASRec(num_items=20, max_seq_len=4, embedding_dim=16,
                            num_blocks=1, num_heads=2).eval()
with torch.no_grad():
    exported_states = exported(torch.tensor([[1, 2, 3, 0]]))
    exported_scores = exported.score_all_items(torch.tensor([[1, 2, 0, 0]]))
check('4-5 导出的模型能前向', list(exported_states.shape), [1, 4, 16])
check('4-5 导出的模型能打分', list(exported_scores.shape), [1, 20])"""
    )
)

# ---------------------------------------------------------------- 14
CELLS.append(
    markdown(
        """## 📖 收束：模型部分到这里就完了

对照 Two-Tower，这个模型多出来的东西只有三样：

| | Two-Tower | SASRec |
|---|---|---|
| 用户侧输入 | 一个 `user_idx` | 最近 50 个 `item_idx` |
| 用户表示 | 直接查表 | 因果注意力编码每个位置，取最后一个有效位 |
| 每个样本的监督 | 1 条（这个用户和这个物品） | 最多 50 条（每个位置各预测下一个） |
| 参数量（本仓库配置） | user 表 6038×128 + item 表 3533×128 ≈ 1.2M | 物品表 3534×64 + 位置表 50×64 + 2 个 block ≈ 0.24M |

参数更少，但每个 batch 产生的监督信号多几十倍，而且用的是行为顺序而不是用户 ID。这就是 Lab 5 要接的部分。

这一章里有两处细节会在 Lab 5 变成具体动作：

- **序列里的物品编号要 `+1`**。`item_idx = 0` 的物品是合法物品，但 token `0` 已经被补齐位占了。忘了加 1，所有编号整体错位一格，模型照样能训。
- **右侧补齐**。左补齐不会报错，只是新用户的表示会变成零向量。"""
    )
)

# ---------------------------------------------------------------- 15
CELLS.append(
    markdown(
        """## 收尾

### 这个 lab 写了多少

| 自己写的 | 行数 | 脚手架给的 |
|---|---|---|
| `build_causal_attention_mask` | 5 | `labkit.py` 的数据管道 |
| `SASRecBlock.forward` | 14 | `SASRecBlock.__init__`（四层结构已建好） |
| `SASRec.__init__` + `forward` | 26 | `SASRec._reset_parameters` |
| `sequence_representation` | 10 | `check()` / `check_close()` |
| `score_all_items` | 4 | Lab 1 导出的 `myrec.model` |

加起来 59 行左右。新写的机制只有两个：一个 `torch.triu` 的 mask，和一处「按行取不同列」的索引。

### 这一章应该能直接答上来

1. 因果 mask 的 `diagonal` 为什么必须是 1 而不是 0？设错了会看到什么？
2. 为什么每个子层末尾都要重新清一次补齐位置？
3. 位置编码去掉之后，指标会掉吗？上面那格自测是怎么验证的？
4. `item_idx` 为什么要 `+1`？在哪里把这个 1 减回来？
5. 为什么序列表示取最后一个有效位，而不是所有位置取平均？

### 再往下

模型有了，但它现在只是个随机初始化的函数：输入一条序列，输出一堆分数。Lab 5 要把它接到真实数据上——序列怎么切、负样本怎么采、loss 怎么写、一个 epoch 怎么跑，以及两组能直接看出差别的对照实验。
"""
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
    target = REPO_ROOT / "lab4-sasrec-model.ipynb"
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


if __name__ == "__main__":
    main()
