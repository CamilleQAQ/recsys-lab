"""生成 labs/lab6-sasrec-evaluate.ipynb。"""

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
        """# Lab 6｜两种历史协议与 locked test

Lab 5 训完了一个模型，也在 800 个用户上看过 validation 指标。这一章把它送进评测，并回答一个一直在绕的问题：

> 同一份权重，为什么会有两个 test 数？

做法是跑两种历史协议，看差在哪、差多少，最后按纪律只跑一次 test 收尾。

符号沿用前几个 lab：📖 概念，🧩 要你写的代码，✅ 断言，🚀 可以改着玩的地方，💡 折叠的参考答案。"""
    )
)

# ---------------------------------------------------------------- 1
CELLS.append(
    code(
        """# 不用改：接上 Lab 5 导出的训练代码和 checkpoint
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
    from myrec.sasrec_model import SASRec
    from myrec.sasrec_train import (
        build_validation_examples,
        pad_item_sequence,
        recommend_top_k,
        score_sequences,
    )
except ModuleNotFoundError as error:
    raise SystemExit('先跑完 Lab 4 和 Lab 5：它们会导出 labs/myrec/') from error

# 同一份权重来自 Lab 5 存下的 checkpoint
CHECKPOINT_PATH = labkit.state_path('sasrec_lab_trained.pt')
if not CHECKPOINT_PATH.exists():
    raise SystemExit(
        f'找不到 {CHECKPOINT_PATH.name}。先跑完 Lab 5（它会训练 A 组并保存）。'
    )

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
print('checkpoint:', CHECKPOINT_PATH.name)"""
    )
)

# ---------------------------------------------------------------- 2
CELLS.append(
    code(
        """# 不用改：数据 + 把 Lab 5 训好的权重装回来
labkit.set_seed(42)
data = labkit.prepare_movielens()
print(data.describe())

checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=True)
model = SASRec(**checkpoint['model_config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print()
print('这份 checkpoint 的训练信息：')
for key in ('train_users', 'train_epochs', 'eval_users', 'best_epoch', 'best_train_loss'):
    print(f'  {key:16s} {checkpoint[key]}')
print(f"  {'validation R@20':16s} {checkpoint['validation_recall20']:.6f}")

check('6-0 checkpoint 的结构配置和模型对得上',
      checkpoint['model_config']['num_items'], data.num_items)
check('6-0 模型进了评估模式', model.training, False)
check('6-0 参数总量没变', sum(p.numel() for p in model.parameters()) > 0, True)"""
    )
)

# ---------------------------------------------------------------- 3
CELLS.append(
    markdown(
        """## 📖 理解一下：评测要先把历史拼出来

序列模型的评测输入不是 `user_idx`，是一条序列。所以对每个待评测用户要做三件事：

```text
1. 取他「在那个时刻已经发生」的交互，按时间排好        -> 输入序列
2. 把目标物品本身排除掉                                -> 目标永不进输入
3. 记录他看过哪些物品                                  -> 从候选里剔掉
```

第 2 步和第 3 步是两件事，容易混：

- **目标物品不在输入里。** 这是防泄漏的底线：把答案喂给模型，指标就没有意义了。
- **已看物品要从候选里剔掉。** 这是防"推荐已经看过的东西"：不剔的话，模型只要把训练集里的高分行排前面就能刷分。

第 3 步的"已看"应该包含哪些，取决于第 1 步用了哪些历史——**两者必须一致**。这就是这一章两种协议的差别所在。"""
    )
)

# ---------------------------------------------------------------- 4
CELLS.append(
    markdown(
        """## 📖 理解一下：两种历史协议

test 目标发生的时刻，"用户看过什么"这个集合里到底有什么？

```text
协议 A（train-only 输入）
    模型输入 = train 正反馈
    候选过滤 = test 时刻之前实际播放过的全部物品

协议 B（标准序列协议）
    模型输入 = train + validation 正反馈
    候选过滤 = 同一份 test 时刻之前实际播放过的全部物品
```

协议 B 多出来的那条 validation 交互，在定义的时间顺序里**确实早于 test 目标**，所以它是"已经观测到的事实"，不是特权。这也是 `src/recsys/evaluation/sasrec_evaluator.py` 默认用的协议（`include_validation_in_test_history=True`）。

两处边界必须同时成立：

- **validation 交互可以进历史，test 交互不行。** 后者就是目标那一刻或之后的记录。
- **两组候选过滤必须一致。** 只改变 SASRec 的输入历史，不能同时改变候选集；测试时曾经播放过但未达正反馈阈值的物品也须过滤。

协议 A 是输入历史的受控实验。ID-only 模型的用户向量在训练时就固定了，它没有"加一条历史"这个入口；但所有模型在测试时都应过滤真实已看物品。两类模型训练目标也不同，指标差异不能全归因于序列结构。

validation 切分上没有这个选择：目标就是紧接着 train 的那一条，能用的历史只有 train。"""
    )
)

# ---------------------------------------------------------------- 5
CELLS.append(
    markdown(
        """### 🧩 你来写：`build_evaluation_inputs()`

按 `split` 和 `include_validation` 拼出评测输入。返回四个东西，顺序固定：

```text
user_indices, input_sequences, targets, seen_items_list
```

规则：

- `split='validation'`：历史 = train 交互。忽略 `include_validation`。
- `split='test'` 且 `include_validation=True`：历史 = train + validation 交互。
- `split='test'` 且 `include_validation=False`：历史 = train 交互。
- 每个用户的历史里**不能包含对应切分的交互**（目标物品尤其不能进输入序列）。
- 两组 test 的 `seen_items_list` 都使用 `data.test_seen`；validation 用 `data.validation_seen`。这是目标时刻之前的全部播放（含未达阈值），与模型输入是否纳入 validation 是两件事。
- `users=None` 表示"不筛用户"，全部目标都参与；否则只保留 `user_idx in users` 的。
- `split` 不合法时抛 `ValueError`。用户没有历史时跳过（不放进返回结果）。"""
    )
)

CELLS.append(
    code(
        """def build_evaluation_inputs(data, split, max_seq_len, users, include_validation=True):
    '''按指定协议拼评测输入，返回 (user_indices, input_sequences, targets, seen_items_list)。'''
    # TODO 6-2：先按 split 选出「允许看到的历史」，再逐个目标造输入
    #   - validation: 历史只有 data.train
    #   - test + include_validation: 历史是 train + validation
    #   - test + include_validation=False: 历史只有 train
    #   历史的整理方式和 Lab 5 的 build_validation_examples 一样：
    #   按 (timestamp, item_idx) 排序之后取 item_idx
    #   seen 集合从 data.validation_seen / data.test_seen 取；输入 = pad_item_sequence(历史, max_seq_len)
    raise NotImplementedError('TODO 6-2')""",
        lab_todo="6-history",
    )
)

CELLS.append(
    code(
        """# ✅ 跑一下：24 个用户，两种协议各拼一次，逐条检查边界
labkit.set_seed(9)
probe_users = sorted({row.user_idx for row in data.validation})[:24]

train_only_users, train_only_inputs, train_only_targets, train_only_seen = \\
    build_evaluation_inputs(data, 'test', max_seq_len=50, users=probe_users,
                            include_validation=False)
with_validation_users, with_validation_inputs, with_validation_targets, with_validation_seen = \\
    build_evaluation_inputs(data, 'test', max_seq_len=50, users=probe_users,
                            include_validation=True)

print('协议 A（train-only）用户数:', len(train_only_users))
print('协议 B（train+validation）用户数:', len(with_validation_users))

check('6-2 两种协议的评测用户集合相同', train_only_users, with_validation_users)
check('6-2 目标顺序也相同', train_only_targets, with_validation_targets)
check('6-2 输入长度等于 max_seq_len',
      [len(sequence) for sequence in train_only_inputs[:3]], [50, 50, 50])

# 目标物品绝不能出现在输入序列里，也不能在 seen 里
targets_by_user = {row.user_idx: row.item_idx for row in data.test}
leaked_input = []
leaked_seen = []
for position, user_idx in enumerate(with_validation_users):
    target_token = targets_by_user[user_idx] + 1
    if target_token in with_validation_inputs[position]:
        leaked_input.append(user_idx)
    if targets_by_user[user_idx] in with_validation_seen[position]:
        leaked_seen.append(user_idx)
check('6-2 目标物品不在输入序列里（两种协议都要成立）', leaked_input, [])
check('6-2 目标物品不在 seen 集合里', leaked_seen, [])

# 只改变模型输入，两组使用同一个 test 时刻的候选过滤集。
validation_items = {row.user_idx: row.item_idx for row in data.validation}
bad_diff = []
for position, user_idx in enumerate(train_only_users):
    if with_validation_seen[position] != train_only_seen[position]:
        bad_diff.append(user_idx)
check('6-2 两组 test 候选过滤完全一致', bad_diff, [])

# 协议 A 的输入里不能出现 validation / test 的物品
validation_pairs = {(row.user_idx, row.item_idx) for row in data.validation}
test_pairs = {(row.user_idx, row.item_idx) for row in data.test}
polluted = []
for position, user_idx in enumerate(train_only_users):
    for pair in (validation_items[user_idx], targets_by_user[user_idx]):
        if pair + 1 in train_only_inputs[position]:
            polluted.append((user_idx, pair))
check('6-2 协议 A 的模型输入没有 validation / test 的物品', polluted, [])

# 协议 B 里 validation 那件该出现（它是历史），test 那件不该出现
missing_validation = []
for position, user_idx in enumerate(with_validation_users):
    if validation_items[user_idx] not in with_validation_seen[position]:
        missing_validation.append(user_idx)
check('6-2 两组 test seen 都含已发生的 validation 物品', missing_validation, [])
polluted_b = []
for position, user_idx in enumerate(with_validation_users):
    if test_pairs and targets_by_user[user_idx] in with_validation_seen[position]:
        polluted_b.append(user_idx)
check('6-2 协议 B 的历史里仍然没有 test 目标', polluted_b, [])

# validation 切分：历史只能来自 train
val_users, val_inputs, val_targets, val_seen = build_evaluation_inputs(
    data, 'validation', max_seq_len=50, users=probe_users
)
val_polluted = []
for position, user_idx in enumerate(val_users):
    if validation_items[user_idx] in val_seen[position]:
        val_polluted.append(user_idx)
check('6-2 validation 协议的历史里没有 validation 目标本身', val_polluted, [])

try:
    build_evaluation_inputs(data, 'train', max_seq_len=50, users=probe_users)
except ValueError as error:
    print('split=train ->', type(error).__name__, error)
    raised = True
else:
    raised = False
check('6-2 非法 split 抛 ValueError', raised, True)"""
    )
)

CELLS.append(
    markdown(
        lab_answer_for="6-history",
        source="""<details>
<summary>💡 参考答案（<code>build_evaluation_inputs</code>）</summary>

```python
def build_evaluation_inputs(data, split, max_seq_len, users, include_validation=True):
    '''按指定协议拼评测输入，返回 (user_indices, input_sequences, targets, seen_items_list)。'''
    if split == 'validation':
        target_rows = data.validation
        history_rows = list(data.train)
    elif split == 'test':
        target_rows = data.test
        history_rows = list(data.train)
        if include_validation:
            # test 目标发生的时刻，validation 交互早已被观测到，属于测试阶段的历史
            history_rows += list(data.validation)
    else:
        raise ValueError("split must be 'validation' or 'test'")

    histories = {}
    for row in history_rows:
        histories.setdefault(row.user_idx, []).append((row.timestamp, row.item_idx))
    histories = {
        user_idx: [item_idx for _, item_idx in sorted(rows)]
        for user_idx, rows in histories.items()
    }

    user_indices = []
    input_sequences = []
    targets = []
    seen_items_list = []
    for row in target_rows:
        if users is not None and row.user_idx not in users:
            continue
        history = histories.get(row.user_idx, [])
        if not history:
            continue
        user_indices.append(row.user_idx)
        input_sequences.append(pad_item_sequence(history, max_seq_len))
        targets.append(row.item_idx)
        seen_items_list.append(
            data.validation_seen[row.user_idx] if split == 'validation'
            else data.test_seen[row.user_idx]
        )
    return user_indices, input_sequences, targets, seen_items_list
```

</details>

关键就在 `history_rows` 那一处分支：**它只决定模型输入**。`seen_items_list` 由目标时刻之前真实播放过的全部物品决定；两组 test 输入不同、过滤集相同，才只改变了一个主要因素。只用正反馈序列的 `set(history)` 会漏掉此前没看完的电影。

`users=None` 那一行是一个小接口选择：`None` 表示不筛，避免调用方为了"全部用户"去构造一个 3025 个元素的集合。代价是函数内部多一次判断，换来的是 `evaluate_locked_test` 那一格不用写集合推导。

用户顺序由 `target_rows` 的顺序决定，而 `data.validation` / `data.test` 在数据准备阶段就已经按 `(user_idx, timestamp, item_idx)` 排好序，所以同一批用户在不同协议下的顺序一致——两种协议的指标才能逐个对齐比较。"""
    )
)

# ---------------------------------------------------------------- 6
CELLS.append(
    markdown(
        """## 📖 理解一下：排序和四个指标

评测的剩下三步：

```text
scores = model.score_all_items(sequences)        # [B, num_items]，原始点积
scores[row, seen_items] = -inf                   # 已看物品退出竞争
top_k = torch.topk(scores, k=K).indices          # 每行取前 K
```

**已看物品必须压成 `-inf`，不能用 `0`。** 点积分数不保证是负的，用 0 代替时那些物品很可能仍然排在前列，而且不会报错。压成 `-inf` 之后 softmax 意义上的概率是 0，`topk` 永远不会取到它们。

指标四个：

```text
Recall@K = 命中目标 / 相关物品数        # 留一法下相关物品数 = 1
NDCG@K   = 命中时 1 / log2(rank + 2)    # rank 从 0 开始，命中越靠前分越高
```

留一法下每个用户只有一个目标，所以 **HitRate@K 和 Recall@K 恒等**，仓库的指标模块同时输出两者只是为了接口统一——它们不是两份独立证据。

和一个"零信息"的下界对照一下：候选有 3533 个，随机排序命中 Top-20 的概率是 `20 / 3533 = 0.0057`。所有报告出来的数都应该离这个值很远。"""
    )
)

# ---------------------------------------------------------------- 7
CELLS.append(
    markdown(
        """### 🧩 你来写：`ranking_metrics()` 和 `recommend_with_filter()`

**a) `ranking_metrics(recommendations, targets, ks=(10, 20))`** —— 返回一个字典，键是 `Recall@10` / `Recall@20` / `NDCG@10` / `NDCG@20`。

**b) `recommend_with_filter(model, sequences, seen_items_list, k, batch_size=256)`** —— 逐批打分、把 seen 压成 `-inf`、取 Top-K，返回 `list[list[item_idx]]`。

Lab 5 里已经有 `recommend_top_k` 可以做 b 这件事。这一格要你自己写一遍：重点是 `-inf` 的处理和"过滤后仍然凑够 K 个"。"""
    )
)

CELLS.append(
    code(
        """def ranking_metrics(recommendations, targets, ks=(10, 20)):
    '''返回 {'Recall@K': ..., 'NDCG@K': ...}。'''
    # TODO 6-2a：Recall 是命中率；NDCG 还要看命中位置
    #   NDCG 的折损用 1 / log2(rank + 2)，rank 从 0 开始
    raise NotImplementedError('TODO 6-2a')


def recommend_with_filter(model, sequences, seen_items_list, k, batch_size=256):
    '''逐批打分、把已看物品压到 -inf、取 Top-K。返回 list[list[item_idx]]。'''
    # TODO 6-2b：分批评分；对每一行把 seen 的分数设成 -inf；再 torch.topk
    #   注意张量要在 no_grad 下算，模型要先 eval()
    raise NotImplementedError('TODO 6-2b')""",
        lab_todo="6-score",
    )
)

CELLS.append(
    code(
        """# ✅ 跑一下：手造一个小例子，指标能手算出来
# 用户 0：目标在推荐里排第 0 位  -> Recall@2 = 1，NDCG@2 = 1/log2(2) = 1.0
# 用户 1：目标在推荐里排第 1 位  -> Recall@2 = 1，NDCG@2 = 1/log2(3) = 0.6309
# 用户 2：目标不在推荐里          -> 两个都是 0
toy_recommendations = [[7, 8, 9], [11, 12, 13], [14, 15, 16]]
toy_targets = [7, 12, 99]
toy_metrics = ranking_metrics(toy_recommendations, toy_targets, ks=(1, 2))

check('6-2a 三个用户的 Recall@2', toy_metrics['Recall@2'], 2 / 3)
check_close('6-2a NDCG@2 手算对得上',
            toy_metrics['NDCG@2'], (1.0 + 1.0 / math.log2(3)) / 3)
check('6-2a Recall@1 只有第一个用户命中', toy_metrics['Recall@1'], 1 / 3)
check_close('6-2a NDCG@1 只有第一个用户贡献满分', toy_metrics['NDCG@1'], 1 / 3)

# 留一法下 Recall 和 HitRate 是同一个数：命中数 / 用户数
hits = sum(1 for row, rec in enumerate(toy_recommendations) if toy_targets[row] in rec)
check('6-2a 留一法下 Recall 就等于命中率', toy_metrics['Recall@2'], hits / len(toy_targets))

# 过滤：已看物品必须一个都不出现在推荐里
labkit.set_seed(10)
probe_scores = score_sequences(model, train_only_inputs[:8])
check('6-2b 打分形状是 [B, num_items]', list(probe_scores.shape), [8, data.num_items])
filtered = recommend_with_filter(model, train_only_inputs[:8], train_only_seen[:8], k=10)
check('6-2b 每个用户 10 个推荐', [len(items) for items in filtered], [10] * 8)
check('6-2b 推荐里没有已看物品',
      [len(set(items) & train_only_seen[row]) for row, items in enumerate(filtered)], [0] * 8)
check('6-2b 推荐物品不重复', [len(set(items)) for items in filtered], [10] * 8)

# 和 Lab 5 的 recommend_top_k 对照：同一批输入应该给出同一批推荐
lab5_recommendations = recommend_top_k(model, train_only_inputs[:8], train_only_seen[:8], k=10)
check('6-2b 和 Lab 5 的 recommend_top_k 结果一致', filtered, lab5_recommendations)"""
    )
)

CELLS.append(
    markdown(
        lab_answer_for="6-score",
        source="""<details>
<summary>💡 参考答案（<code>ranking_metrics</code> / <code>recommend_with_filter</code>）</summary>

```python
def ranking_metrics(recommendations, targets, ks=(10, 20)):
    '''返回 {'Recall@K': ..., 'NDCG@K': ...}。'''
    metrics = {}
    for k in ks:
        hits = 0
        ndcg = 0.0
        for row, recommendation in enumerate(recommendations):
            top_k = recommendation[:k]
            if targets[row] in top_k:
                hits += 1
                rank = top_k.index(targets[row])
                ndcg += 1.0 / math.log2(rank + 2)
        metrics[f'Recall@{k}'] = hits / len(recommendations)
        metrics[f'NDCG@{k}'] = ndcg / len(recommendations)
    return metrics


def recommend_with_filter(model, sequences, seen_items_list, k, batch_size=256):
    '''逐批打分、把已看物品压到 -inf、取 Top-K。返回 list[list[item_idx]]。'''
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
```

</details>

`recommendations.extend(top_items)` 而不是逐个 append：`top_items` 已经是"这一批每一行的 Top-K"，直接展开成一串列表就是全局结果，顺序和 `sequences` 一致。

`ranking_metrics` 里没有归一化因子。留一法下每个用户只有一个相关物品，理想 DCG 是 `1 / log2(0 + 2) = 1`，所以折损值直接累加再除以用户数就是 NDCG。多目标场景下要除以 `IDCG`（把所有相关物品按最优顺序排列能拿到的分），那时候这个简化就不成立了。"""
    )
)

# ---------------------------------------------------------------- 8
CELLS.append(
    code(
        """# ✅ 跑一下：两种协议在同一批用户上各跑一次
# 用同一份权重、同一批用户，只换「历史能看到多少」。
# 评测用户取自 validation（Lab 5 也是在这批用户上看的 validation 指标），
# 目标换成本人的 test 交互。
EVAL_USERS = sorted({row.user_idx for row in data.validation})[:800]

started = time.time()
users_a, inputs_a, targets_a, seen_a = build_evaluation_inputs(
    data, 'test', max_seq_len=50, users=EVAL_USERS, include_validation=False
)
recommendations_a = recommend_with_filter(model, inputs_a, seen_a, k=20)
metrics_a = ranking_metrics(recommendations_a, targets_a)
print(f'协议 A（train-only）        {len(users_a)} 个用户   {time.time() - started:.0f}s')
for name, value in metrics_a.items():
    print(f'  {name:12s} {value:.6f}')

started = time.time()
users_b, inputs_b, targets_b, seen_b = build_evaluation_inputs(
    data, 'test', max_seq_len=50, users=EVAL_USERS, include_validation=True
)
recommendations_b = recommend_with_filter(model, inputs_b, seen_b, k=20)
metrics_b = ranking_metrics(recommendations_b, targets_b)
print(f'协议 B（train+validation）  {len(users_b)} 个用户   {time.time() - started:.0f}s')
for name, value in metrics_b.items():
    print(f'  {name:12s} {value:.6f}')

print()
print(f"Recall@20            {metrics_a['Recall@20']:.6f}  ->  {metrics_b['Recall@20']:.6f}"
      f"   （{metrics_b['Recall@20'] - metrics_a['Recall@20']:+.6f}）")
print(f"NDCG@20              {metrics_a['NDCG@20']:.6f}  ->  {metrics_b['NDCG@20']:.6f}"
      f"   （{metrics_b['NDCG@20'] - metrics_a['NDCG@20']:+.6f}）")

check('6-3 两种协议的评测用户集合一致', users_a, users_b)
check('6-3 两种协议的目标一致', targets_a, targets_b)
check('6-3 只有历史不同（协议 B 的 seen 更大）',
      all(len(seen_b[row]) >= len(seen_a[row]) for row in range(len(users_a))), True)
check('6-3 协议 B 的 Recall@20 更高（多一条历史值分）',
      metrics_b['Recall@20'] > metrics_a['Recall@20'], True)
check('6-3 两种协议都不止随机水平', min(metrics_a['Recall@20'], metrics_b['Recall@20']) > 5 * 20 / data.num_items, True)
check('6-3 两个数确实不同（这就是「同一份权重，两个数」）',
      abs(metrics_b['Recall@20'] - metrics_a['Recall@20']) > 1e-4, True)"""
    )
)

# ---------------------------------------------------------------- 9
CELLS.append(
    markdown(
        """## 📖 理解一下：同一份权重，两个数

上面那两行打印就是这个 lab 想说的全部：

```text
Recall@20   train-only 的历史  ->  train+validation 的历史
```

差值的来源不是模型变好了。权重一个字都没动，改的只是"评测时允许模型看到多少历史"。所以：

- **协议 B 的数不能拿去和 ID-only 的基线相减。** 那边没有"多一条历史"的入口，两者的信息量不同。
- **要对齐口径，只能用协议 A。** 这就是 `src/recsys/evaluation/sasrec_evaluator.py` 里 `--train-history-only` 那个开关存在的理由，它产出的是 `artifacts/results.json` 里 `supplementary_controlled_evaluation` 那一节。
- **两个数都该报，但要说清各自回答什么问题。** 一个是"这个序列模型能做到多好"，另一个是"序列建模相对静态画像值多少"。

顺带注意 `seen` 那一条断言：协议 B 的 `seen` 严格更大。**历史多了、过滤集也必须跟着变大**，两者是同一份东西的两个视角。只改一边就是上面说的那类静默错误。"""
    )
)

# ---------------------------------------------------------------- 10
CELLS.append(
    code(
        """# ✅ 跑一下：和 Lab 5 记下的 validation 指标对账
# 这份 checkpoint 是 Lab 5 训出来的，它里面存着当时的 validation Recall@20。
# 现在用同一个评测口径重算一遍：数字对得上，才说明两章用的是同一份数据、同一个权重。
labkit.set_seed(42)
validation_users = sorted({row.user_idx for row in data.validation})[:800]
users_v, inputs_v, targets_v, seen_v = build_evaluation_inputs(
    data, 'validation', max_seq_len=50, users=validation_users
)
recommendations_v = recommend_with_filter(model, inputs_v, seen_v, k=20)
metrics_v = ranking_metrics(recommendations_v, targets_v)

recorded = checkpoint['validation_recall20']
print(f"重新计算 validation Recall@20: {metrics_v['Recall@20']:.6f}")
print(f"checkpoint 里记着的          : {recorded:.6f}")
print(f"差异                         : {abs(metrics_v['Recall@20'] - recorded):.2e}")

check('6-4 重算的 validation Recall@20 和 checkpoint 记录一致',
      round(metrics_v['Recall@20'], 6), round(recorded, 6))
check_close('6-4 差异在浮点误差内',
            metrics_v['Recall@20'], recorded, tolerance=1e-9)
check('6-4 评测用户数和训练时一致', len(users_v), checkpoint['eval_users'])"""
    )
)

# ---------------------------------------------------------------- 11
CELLS.append(
    markdown(
        """## 📖 理解一下：test 只跑一次

前面所有比较都在 validation 上做（选序列长度、选因果 mask、选轮数）。现在配置已经锁定，可以对 test 跑那唯一的一次了。

纪律写在这里不是形式主义：

- **看过的 test 会进入决策回路。** 一旦知道"test 上 A 比 B 好 0.002"，你下一次就不自觉地偏向 A，之后再说"泛化能力"就不再是无偏的。
- **跑多次 test 会让"最好那一次"变成选择的结果。** 跑 10 次取最高，等价于在 test 上做了一次超参搜索。
- **validation 比 test 高是正常的**（做过模型选择，天然带乐观偏差），不要因此回头改配置。

固定评测集也一样：这一章从头到尾用的是同一批 800 个用户（`EVAL_USERS` 取 validation 用户排序后的前 800 个，和 Lab 5 完全一致）。如果两个协议用不同的用户集合，差值里就混进了"用户难度不同"这个变量。"""
    )
)

# ---------------------------------------------------------------- 12
CELLS.append(
    markdown(
        """### 🧩 你来写：`compare_history_protocols()`

把上面几格串成一个函数，返回两种协议的结果字典：

```python
{
    'train_only': {'Recall@10': ..., ...},
    'with_validation': {'Recall@10': ..., ...},
}
```

要求：

- 同一个 `users` 列表、同一份权重，只切 `include_validation`；
- 每次都用 `build_evaluation_inputs` + `recommend_with_filter` + `ranking_metrics`，不要复制粘贴那三步的实现；
- 额外把两种协议各自的"平均历史长度"也放进去（用 `input_sequences` 里非零 token 的个数算），用来解释差值——协议 B 之所以更好，是因为它看到的历史更长。"""
    )
)

CELLS.append(
    code(
        """def compare_history_protocols(model, data, users, max_seq_len=50, k=20):
    '''同一份权重、同一批用户，只换历史协议。返回两种协议的结果字典。'''
    # TODO 6-3：两种协议各跑一遍，返回指标 + 平均历史长度
    #   历史长度 = 输入序列里非零 token 的个数（补齐位是 0）
    #   结构：{'train_only': {...}, 'with_validation': {...}}
    raise NotImplementedError('TODO 6-3')""",
        lab_todo="6-compare",
    )
)

CELLS.append(
    code(
        """# ✅ 跑一下：用刚写好的函数把两种协议重跑一遍，看它和手工流程图是否一致
comparison = compare_history_protocols(model, data, EVAL_USERS, max_seq_len=50, k=20)

for name in ('train_only', 'with_validation'):
    block = comparison[name]
    print(f"{name:16s} 平均历史长度 {block['mean_history_length']:.1f}  "
          f"Recall@10 {block['Recall@10']:.6f}  Recall@20 {block['Recall@20']:.6f}  "
          f"NDCG@20 {block['NDCG@20']:.6f}")

check('6-3 两种协议都在返回值里', sorted(comparison.keys()), ['train_only', 'with_validation'])
check('6-3 协议 B 的平均历史更长',
      comparison['with_validation']['mean_history_length'] > comparison['train_only']['mean_history_length'], True)
check_close('6-3 和手工那两格算出的数完全一致',
            comparison['train_only']['Recall@20'], metrics_a['Recall@20'], tolerance=0.0)
check_close('6-3 协议 B 也对得上',
            comparison['with_validation']['Recall@20'], metrics_b['Recall@20'], tolerance=0.0)
check('6-3 两个协议给出不同的数',
      comparison['train_only']['Recall@20'] != comparison['with_validation']['Recall@20'], True)"""
    )
)

CELLS.append(
    markdown(
        lab_answer_for="6-compare",
        source="""<details>
<summary>💡 参考答案（<code>compare_history_protocols</code>）</summary>

```python
def compare_history_protocols(model, data, users, max_seq_len=50, k=20):
    '''同一份权重、同一批用户，只换历史协议。返回两种协议的结果字典。'''
    comparison = {}
    for name, include_validation in (('train_only', False), ('with_validation', True)):
        user_indices, input_sequences, targets, seen_items_list = build_evaluation_inputs(
            data, 'test', max_seq_len=max_seq_len, users=users,
            include_validation=include_validation,
        )
        recommendations = recommend_with_filter(model, input_sequences, seen_items_list, k=k)
        metrics = ranking_metrics(recommendations, targets, ks=(10, k))
        lengths = [sum(1 for token in sequence if token != 0) for sequence in input_sequences]
        comparison[name] = {
            **metrics,
            'mean_history_length': sum(lengths) / len(lengths),
            'num_users': len(user_indices),
        }
    return comparison
```

</details>

每次循环里都重新调 `build_evaluation_inputs`，而不是把两种输入都先算好——看着多了一次数据整理，但它保证了"指标、历史长度、用户数"这三样东西来自同一次构造，不会出现三者互相不对应的情况。

平均历史长度存在的意义是让差值可解释：如果协议 B 只是 `seen` 变大而历史长度没变，那就说明哪里写错了——但指标仍然可能更高，所以单看指标发现不了。"""
    )
)

# ---------------------------------------------------------------- 13
CELLS.append(
    markdown(
        """### 🧩 你来写：locked test

配置已经锁定，这是唯一一次 test。用**协议 A**（口径对齐那一版）跑完整评测集，把结果整理成可以直接贴进报告的一行。

完整评测集是 `data.test` 的全部用户（3,025 个），不是那 800 个。用 `users=None` 表示"不筛用户"——`build_evaluation_inputs` 里那一行判断对 `None` 会怎么走，你写的时候要处理掉。"""
    )
)

CELLS.append(
    code(
        """def evaluate_locked_test(model, data, max_seq_len=50, include_validation=False, k=20):
    '''配置锁定后的那一次 test。返回指标字典。'''
    # TODO 6-4：对全部 test 用户跑一遍
    #   注意 users=None 时要选中所有用户（提示：可以先用集合推导把评测用户收集出来）
    #   返回 ranking_metrics 的结果，外加 num_users
    raise NotImplementedError('TODO 6-4')""",
        lab_todo="6-locked",
    )
)

CELLS.append(
    code(
        """# ✅ 跑一下：这一次 test 只在配置锁定之后跑
started = time.time()
locked_test = evaluate_locked_test(model, data, max_seq_len=50, include_validation=False, k=20)
print(f"test 用户数: {locked_test['num_users']}   （{time.time() - started:.0f}s）")
for name in ('Recall@10', 'NDCG@10', 'Recall@20', 'NDCG@20'):
    print(f'  {name:12s} {locked_test[name]:.6f}')

validation_recall20 = metrics_v['Recall@20']
print()
print(f"validation Recall@20 {validation_recall20:.6f}  vs  test Recall@20 {locked_test['Recall@20']:.6f}")

check('6-4 test 覆盖全部评测用户', locked_test['num_users'], len(data.test))
check('6-4 指标都落在 [0, 1]',
      all(0.0 <= locked_test[name] <= 1.0 for name in ('Recall@10', 'NDCG@10', 'Recall@20', 'NDCG@20')), True)
check('6-4 test 明显高于随机水平', locked_test['Recall@20'] > 5 * 20 / data.num_items, True)
check('6-4 Recall@20 >= Recall@10（K 更大只会更容易命中）',
      locked_test['Recall@20'] >= locked_test['Recall@10'], True)
check('6-4 NDCG 不超过同档 Recall（折损只会更低）',
      locked_test['NDCG@20'] <= locked_test['Recall@20'], True)"""
    )
)

CELLS.append(
    markdown(
        lab_answer_for="6-locked",
        source="""<details>
<summary>💡 参考答案（<code>evaluate_locked_test</code>）</summary>

```python
def evaluate_locked_test(model, data, max_seq_len=50, include_validation=False, k=20):
    '''配置锁定后的那一次 test。返回指标字典。'''
    if include_validation:
        users = {row.user_idx for row in data.validation} | {row.user_idx for row in data.test}
    else:
        users = {row.user_idx for row in data.test}
    user_indices, input_sequences, targets, seen_items_list = build_evaluation_inputs(
        data, 'test', max_seq_len=max_seq_len, users=users,
        include_validation=include_validation,
    )
    recommendations = recommend_with_filter(model, input_sequences, seen_items_list, k=k)
    metrics = ranking_metrics(recommendations, targets, ks=(10, k))
    metrics['num_users'] = len(user_indices)
    return metrics
```

</details>

`users` 用测试集里的用户集合，而不是"全部用户再去筛"：`build_evaluation_inputs` 是按目标行遍历的，目标只来自 test，所以传全集也只是多判断几次，结果一样。写成目标集合更直白，也避免"某个用户没有 test 目标却被算进分母"这类问题。

`num_users` 就是评测分母。它必须显式返回并断言——指标除以了多少人是这个数说了算，混用会直接改变结论。"""
    )
)

# ---------------------------------------------------------------- 14
CELLS.append(
    markdown(
        """## 📖 收束：这一章的两个数

把这一章跑出来的结果放一起（同一份权重，Lab 5 训练，评测用户是 validation 排序后的前 800 个）：

```text
协议              平均历史长度   Recall@10   Recall@20   NDCG@20
train-only          40.0        0.103750    0.170000    0.074801
train+validation    40.4        0.140000    0.208750    0.092204
                                          +0.038750   +0.017403
```

三个读法：

**1. 差值全部来自"能看到什么"。** 权重在 Lab 5 就固定了，这一章没有训练任何东西。多一条 validation 历史，Recall@20 涨了 0.03875——比前面几章任何一个结构改动都大。

**2. 平均历史长度几乎没变（40.0 → 40.4），但指标涨了 0.03875。** 这个对比值得留意：多出来的历史只有**一件**物品，均值上几乎看不出来，可它是"最近发生的那一件"，信息量远大于平均。用平均值衡量"历史变长了多少"在这里是会骗人的——真正起作用的是序列末位多了什么。

**3. 协议 A 才是能和 ID-only 基线并列的那一行。** 跑完整 test 集（3,025 个用户）拿到：

```text
test（协议 A，train-only）  Recall@10 0.097521   Recall@20 0.154050   NDCG@10 0.051586   NDCG@20 0.065897
```

这个数来自 labs 自己的评测流程（800 个用户选轮、在全部评测用户上跑 test），和 `artifacts/results.json` 里 `src/` 的 train-history-only test（Recall@20 = 0.160331）不是同一口径：两边的训练预算、评测用户和过滤实现都不同，不能直接相减。协议本身的影响看 `docs/03-sasrec/05`。

**协议 B 的数更好看（0.208750），但它的信息量和那三个 ID-only 基线不一样**，不能横着比。

这一章也把前三章的方法论收了口：

1. Lab 5 选出来的配置在运行前就定死了，这一章没有回头改；
2. test 只跑一次；
3. 唯一的"调参"动作是协议选择，**而它不改变模型**，只改变你问的问题；
4. 重算 validation 得到 0.230000，和 checkpoint 里记的 `0.230000` 差异是 `0.00e+00`——这份权重、这份数据、这个评测口径，三章之间是接得上的。

顺带说一下候选集这件事：全部 3533 个物品都参与排序，其中 6 个在所有用户的训练集里都没出现过（3 个是 validation 目标、3 个是 test 目标）。它们的 embedding 拿不到任何训练信号，却仍然参与竞争。这是冷启动限制，不是泄漏——身份是已知的，只是模型没机会学会它们。"""
    )
)

# ---------------------------------------------------------------- 15
CELLS.append(
    code(
        """# 💾 导出这一章写的东西（评测工具），并留一份结果
labkit.harvest(
    'myrec/sasrec_eval.py',
    build_evaluation_inputs,
    ranking_metrics,
    recommend_with_filter,
    compare_history_protocols,
    evaluate_locked_test,
    imports=[
        'import math',
        'import torch',
        'from myrec.sasrec_train import pad_item_sequence',
    ],
)

import importlib

import myrec.sasrec_eval as my_eval

importlib.reload(my_eval)

check('6-5 交接成功：myrec.sasrec_eval 能用',
      hasattr(my_eval, 'ranking_metrics') and hasattr(my_eval, 'evaluate_locked_test'), True)

# 光能 import 不够：真的用它算一次
exported_metrics = my_eval.ranking_metrics([[7, 8], [11, 12]], [8, 12], ks=(2,))
check_close('6-5 导出的 ranking_metrics 能算', exported_metrics['Recall@2'], 1.0, tolerance=0.0)

labkit.record('lab6_train_only_test_recall20', comparison['train_only']['Recall@20'])
labkit.record('lab6_with_validation_test_recall20', comparison['with_validation']['Recall@20'])
print('已记录两种协议的 test Recall@20（800 用户子集）')"""
    )
)

# ---------------------------------------------------------------- 16
CELLS.append(
    markdown(
        """## 收尾

### 这个 lab 写了多少

| 自己写的 | 行数 | 脚手架给的 |
|---|---|---|
| `build_evaluation_inputs` | 30 | checkpoint 加载、`check()` / `check_close()` |
| `ranking_metrics` | 15 | Lab 4 的模型、Lab 5 的 `pad_item_sequence` / `recommend_top_k` |
| `recommend_with_filter` | 18 | |
| `compare_history_protocols` | 16 | |
| `evaluate_locked_test` | 12 | |

加起来 91 行左右。真正的新东西只有"历史怎么拼"，其余都是从 Lab 2 / Lab 5 搬过来的评测流程——这是有意的：**评测代码要复用，不要每个模型重写一遍。**

### 这一章应该能直接答上来

1. 为什么序列模型的 test 可以用 validation 交互当历史，而 ID-only 的模型不行？
2. "目标物品不在输入里"和"已看物品从候选里剔掉"分别防的是什么？
3. 历史集和过滤集为什么必须一致？只改一边会怎样？
4. 同一份权重为什么会得到两个 test 数？该拿哪一个和别人比？
5. 为什么 test 只跑一次？

### 三个 lab 走完之后

```text
myrec/sasrec_model.py    build_causal_attention_mask、SASRecBlock、SASRec
myrec/sasrec_train.py    pad_item_sequence、sample_negative_item、SASRecTrainDataset、
                         sasrec_loss、train_one_epoch、score_sequences、
                         recommend_top_k、build_validation_examples
myrec/sasrec_eval.py     build_evaluation_inputs、ranking_metrics、recommend_with_filter、
                         compare_history_protocols、evaluate_locked_test
myrec/sasrec_lab_trained.pt    Lab 5 训出来的权重 + 训练元信息
myrec/results.json             两章记下的关键数字
```

模型、训练、评测三件事都是自己写的，彼此是 import 关系。这套东西和 `src/recsys/` 里的工程版本是同一套算法，差别只在工程化程度（统一配置、checkpoint 哈希、pytest）。

### 再往下

仓库里还有两个 ID-only 的基线（Popularity、MF）和一层 FAISS 精确检索没有动手。前者能回答"个性化到底比推热门强多少"，后者能回答"向量检索层是怎么接上的"。那是下一组 lab 的事。"""
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
    target = REPO_ROOT / "lab6-sasrec-evaluate.ipynb"
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
