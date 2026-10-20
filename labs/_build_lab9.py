"""生成 labs/lab9-kuairec-label-threshold.ipynb。

和 _build_lab7 / _build_lab8 一样，用脚本生成 notebook，避免手写 JSON 转义问题。

这一章不训练模型：它只做「把标签规则在训练之前定死」这一件事，
所以格子比前面的 lab 少，自测集中在「数据事实」和「这个 bug 到底会不会发生」。
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
        """# Lab 9｜标签先定死：KuaiRec 的 watch_ratio 阈值

前八个 lab 用的都是 MovieLens，标签是现成的：评分 4 或 5 算「喜欢」。

KuaiRec 没有评分，只有 `watch_ratio = 播放时长 / 视频时长`。于是有一个必须由人回答的问题：**多少算喜欢？**

这一章不训练模型。它要做的是在训练之前把一个决定做完，并且把理由留成可核对的数字：

1. 四个候选阈值各切出多少正样本；
2. 阈值为什么定在 1.0；
3. 重复的 `(user, video)` 对怎么处理——不处理会坏在哪里。

第 3 条是这一章的重点：它不是一个"数据干净一点"的优化，而是一个**会让评测悄悄失效**的 bug。

符号：📖 概念，🧩 要你写的代码，✅ 断言，🚀 可以改着玩的地方，💡 折叠的参考答案。"""
    )
)

# ---------------------------------------------------------------- 1
CELLS.append(
    code(
        """# 不用改：准备数据和检查工具
import math
import sys
from collections import defaultdict
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


print('labs dir:', LABS_DIR)"""
    )
)

# ---------------------------------------------------------------- 2
CELLS.append(
    markdown(
        """## 📖 理解一下：为什么这一章要单独拿出来做

标签不是一个清洗选项，它是模型结构的一部分：

```text
阈值 → 正样本集合 → (user, video) 词表 → num_users / num_items →
      embedding 表的形状 → checkpoint
```

阈值动了，词表就变了，之前训好的权重全部作废，已经写进文档的指标也不可复现。所以顺序只能是**先定标签、再训模型**，而不是"先跑起来，阈值以后再调"。

这一章的样本是 KuaiRec `big_matrix` 里前 400 个用户（按原始 `user_id` 升序）的全部播放记录。取整用户而不是随机抽行，样本内部的时间顺序和重复播放结构才和真实数据一样。"""
    )
)

# ---------------------------------------------------------------- 3
CELLS.append(
    code(
        """# 不用改：真实数据。第一次调用会解出官方压缩包里的 big_matrix.csv（431 MB），
# 之后走缓存 data/kuairec/kuairec_sample_400.csv，不再扫大矩阵。
labkit.set_seed(42)
rows = labkit.load_kuairec_sample(num_users=400)

print('播放记录:', len(rows))
print('用户数:', len({row.original_user_id for row in rows}))
print('视频数:', len({row.original_item_id for row in rows}))
print()
print('前 3 行原始数据：')
for row in rows[:3]:
    print(
        f'  user={row.original_user_id} video={row.original_item_id} '
        f'timestamp={row.timestamp} watch_ratio={row.watch_ratio}'
    )"""
    )
)

# ---------------------------------------------------------------- 4
CELLS.append(
    markdown(
        """## 📖 理解一下：先看这份样本的事实

这一格的输出不用你写，先把三个量记住：样本有多少行、时间戳精度、以及它占完整数据的比例。

时间戳精度值得单独看一眼：KuaiRec 的时间戳带小数，而 MovieLens 只到秒。这个差别在下一节会变成一个真问题。"""
    )
)

CELLS.append(
    code(
        """# 不用改：样本的事实
print('播放记录:', len(rows))
print('用户数:', len({row.original_user_id for row in rows}))
print('视频数:', len({row.original_item_id for row in rows}))
print('时间戳样例:', [row.timestamp for row in rows[:3]])

ratios = [row.watch_ratio for row in rows]
print('watch_ratio 最小/最大:', round(min(ratios), 3), '/', round(max(ratios), 3))
print('  ！最大值远大于 1，因为同一个视频会被看很多遍，时长比会累加成几十上百')
print('  ！也就是说 watch_ratio 并没有「上界是 1」这种性质，别照着直觉写死区间')
print('占比:', round(len(rows) / 12530806, 4))"""
    )
)

# ---------------------------------------------------------------- 5
CELLS.append(
    markdown(
        """## 📖 理解一下：同一个视频被重复播放

先看一个数字：12,530,806 行原始记录里只有 10,300,969 个不同的 `(user, video)` 对。也就是说 **17.8% 的行是重复的**。

重复分两种，性质完全不同：

```text
时间戳不同   用户在不同时刻重复播放了同一个视频   → 真实行为，是兴趣信号
时间戳相同   一次播放被记成了多行               → 记录方式，没有额外信息
```

两种都会在评测里出问题，而且**不会报错**。先看评测是怎么做切的：

```text
只有 train < validation < test 的 timestamp 边界明确时才留出；同秒边界用户全部留在 train
排序时把 train 里出现过的物品从候选里剔掉，因为已经推荐过的不能再推荐
```

如果同一个 `(user, video)` 在 train 里留了一条、又被时间顺序选成了 test，那这个 test 目标会在排序前被自己剔掉——**永远不可能命中**。Recall 被系统性压低，而且没有任何报错。

所以这一章要显式定一条合并规则，而不是默认保留。"""
    )
)

# ---------------------------------------------------------------- 6
CELLS.append(
    markdown(
        """### 🧩 你来写：`positive_pair_counts()`

给定阈值，数三个量：命中的行数、命中的不同用户数、命中的不同视频数。

阈值判定是**闭区间**：`watch_ratio >= threshold` 算命中。"""
    )
)

CELLS.append(
    code(
        """def positive_pair_counts(rows, threshold):
    '''返回 (命中行数, 不同用户数, 不同视频数)。'''
    # TODO 9-1：
    #   1. 过滤出 watch_ratio >= threshold 的行
    #   2. 分别用 set 数出不同的 original_user_id 和 original_item_id
    raise NotImplementedError('TODO 9-1')""",
        lab_todo="9-counts",
    )
)

CELLS.append(
    code(
        """# ✅ 跑一下：阈值 0.5 的命中量
rows_05, users_05, items_05 = positive_pair_counts(rows, 0.5)
print(rows_05, users_05, items_05)

check('9-0 样本 724,024 行、400 个用户', (len(rows), len({row.original_user_id for row in rows})), (724024, 400))
check('9-0 只有 7,097 行 watch_ratio 超过 5',
      sum(1 for row in rows if row.watch_ratio > 5.0), 7097)
check('9-0 watch_ratio 最大值 161.7，说明它没有上界',
      round(max(row.watch_ratio for row in rows), 1), 161.7)
check_close('9-0 样本只占完整 big_matrix 的 5.8%',
            len(rows) / 12530806, 0.0578, tolerance=5e-4)
check('9-1 阈值 0.5 命中 465,304 行', rows_05, 465304)
check('9-1 阈值 0.5 时 400 个用户全部有命中', users_05, 400)
check('9-1 阈值 0.5 时命中 8,626 个视频', items_05, 8626)
check('9-1 阈值 1.0 的命中量少得多', positive_pair_counts(rows, 1.0)[0], 251724)
check('9-1 阈值 2.0 只命中 55,420 行', positive_pair_counts(rows, 2.0)[0], 55420)"""
    )
)

CELLS.append(
    markdown(
        lab_answer_for="9-counts",
        source="""<details>
<summary>💡 参考答案（<code>positive_pair_counts</code>）</summary>

```python
def positive_pair_counts(rows, threshold):
    '''返回 (命中行数, 不同用户数, 不同视频数)。'''
    positive = [row for row in rows if row.watch_ratio >= threshold]
    users = {row.original_user_id for row in positive}
    items = {row.original_item_id for row in positive}
    return len(positive), len(users), len(items)
```

</details>

注意这里数的是**行**，不是不同的 `(user, video)` 对。下一个格子会把这两个数的差暴露出来。"""
    )
)

# ---------------------------------------------------------------- 7
CELLS.append(
    markdown(
        """### 🧩 你来写：把四个阈值排成一张表

这张表就是「定阈值」要看的全部信息：命中多少行、占比多少、用户和视频覆盖还剩多少。

比上一格多看一个量：**不同的 `(user, video)` 对有多少**。它和命中行数的差，就是重复播放的量。"""
    )
)

CELLS.append(
    code(
        """def threshold_table(rows, thresholds=(0.5, 0.8, 1.0, 2.0)):
    '''返回 {threshold: {'rows':…, 'ratio':…, 'users':…, 'items':…, 'pairs':…}}。'''
    # TODO 9-2：
    #   ratio = 命中行数 / 全部行数
    #   pairs = 不同的 (original_user_id, original_item_id) 组合数
    raise NotImplementedError('TODO 9-2')""",
        lab_todo="9-table",
    )
)

CELLS.append(
    code(
        """# ✅ 跑一下：四个阈值一起看
table = threshold_table(rows)

print(f"{'阈值':>5} {'命中行':>10} {'占比':>8} {'用户':>6} {'视频':>6} {'不同对':>10}")
for threshold, stats in table.items():
    print(
        f"{threshold:>5} {stats['rows']:>10,} {stats['ratio']:>8.4f} "
        f"{stats['users']:>6} {stats['items']:>6} {stats['pairs']:>10,}"
    )

check_close('9-2 阈值 0.5 占比 0.6427', table[0.5]['ratio'], 0.6427, tolerance=1e-4)
check_close('9-2 阈值 1.0 占比 0.3477', table[1.0]['ratio'], 0.3477, tolerance=1e-4)
check_close('9-2 阈值 2.0 占比 0.0765', table[2.0]['ratio'], 0.0765, tolerance=1e-4)
check('9-2 阈值 1.0 时 400 个用户都还在', table[1.0]['users'], 400)
check('9-2 阈值 2.0 时视频覆盖掉到 5,956', table[2.0]['items'], 5956)
check('9-2 阈值 1.0 的不同对比命中行少 29,841',
      table[1.0]['rows'] - table[1.0]['pairs'], 29841)
check('9-2 阈值越紧，正样本越少（单调）',
      [stats['rows'] for stats in table.values()],
      sorted([stats['rows'] for stats in table.values()], reverse=True))"""
    )
)

CELLS.append(
    markdown(
        lab_answer_for="9-table",
        source="""<details>
<summary>💡 参考答案（<code>threshold_table</code>）</summary>

```python
def threshold_table(rows, thresholds=(0.5, 0.8, 1.0, 2.0)):
    '''返回 {threshold: {'rows':…, 'ratio':…, 'users':…, 'items':…, 'pairs':…}}。'''
    table = {}
    for threshold in thresholds:
        hit_rows, users, items = positive_pair_counts(rows, threshold)
        pairs = {
            (row.original_user_id, row.original_item_id)
            for row in rows
            if row.watch_ratio >= threshold
        }
        table[threshold] = {
            'rows': hit_rows,
            'ratio': hit_rows / len(rows),
            'users': users,
            'items': items,
            'pairs': len(pairs),
        }
    return table
```

</details>

最后两行是这个脚本最想让人看到的东西。阈值 1.0 时命中 251,724 行，但只有 221,883 个不同的对——**29,841 行是重复播放**（11.9%）。阈值越紧，重复占比越高，因为忠实用户看得最认真的那些视频最容易被看第二遍。"""
    )
)

# ---------------------------------------------------------------- 8
CELLS.append(
    markdown(
        """## 📖 理解一下：阈值定在 1.0，理由不是"指标更好"

四个阈值都能用，选哪个不是看哪个数好看，而是看哪个语义站得住：

- **1.0 是数据自己给出的边界。** `watch_ratio = 1` 就是"看完了整个视频"。0.5 和 0.8 是"看了一半左右"这种没有物理含义的位置。
- **样本量够。** 阈值 1.0 时 400 个用户全部保留、视频覆盖 7,927 个，没有任何实体掉出词表。
- **越松越像噪声。** 阈值 0.5 把 64.3% 的播放都记成"喜欢"——短视频信息流里被划过一半太常见了。
- **越紧开始付代价。** 阈值 2.0 只剩 7.7%，视频覆盖掉到 5,956，长尾直接被砍掉一截。

所以定案是 `watch_ratio >= 1.0`。这个值写在项目级配置里（`recsys.config.KUAIREC_WATCH_RATIO_THRESHOLD`），所有入口默认读它。

**阈值是这一章唯一"由人决定"的地方，所以它必须被写下来。** 剩下的部分——重复怎么合并、切分怎么排序——都应该是规则，而不是每次重跑临时决定。"""
    )
)

# ---------------------------------------------------------------- 9
CELLS.append(
    markdown(
        """### 🧩 你来写：`merge_duplicate_pairs()`

把每个 `(user, video)` 对压成「首次播放」事件，返回值是 `{(user, video): (首次时间, 当次 watch_ratio)}`，只保留首次事件达标的 pair。

合并规则（两条都要）：

```text
时间：取该对首次播放的 timestamp
watch_ratio：取这一次播放自身的值
```

如果最早同一时间出现互相矛盾的达标/未达标标签，真实先后无法确定，排除该 pair。

实现时先扫描**所有**原始行，找每对最早的事件，再按阈值筛选。不能先筛掉负事件，否则「先未达标、后达标」会被误当成首次正反馈。"""
    )
)

CELLS.append(
    code(
        """def merge_duplicate_pairs(rows, threshold):
    '''返回达标的 {(user, video): (first_timestamp, first_watch_ratio)}。'''
    # TODO 9-3：
    #   扫描所有行，找每对最早的 timestamp 及该事件的 watch_ratio
    #   同一最早时间出现冲突标签时排除；最后才应用 threshold
    raise NotImplementedError('TODO 9-3')""",
        lab_todo="9-merge",
    )
)

CELLS.append(
    code(
        """# ✅ 跑一下：合并之后剩下多少、以及一个"信号是混的"例子
merged = merge_duplicate_pairs(rows, 1.0)

check('9-3 首次播放达标后 211,023 对', len(merged), 211023)
check('9-3 比原始达标行数少 40,701',
      table[1.0]['rows'] - len(merged), 40701)
check('9-3 key 是 (user, video) 二元组',
      sorted({len(key) for key in merged}), [2])
check('9-3 每对都带着两个值',
      sorted({len(value) for value in merged.values()}), [2])

# 同一个对里"有一次达标、另一次没达标"的情况确实存在。
mixed = 0
per_pair_ratios = defaultdict(list)
for row in rows:
    per_pair_ratios[(row.original_user_id, row.original_item_id)].append(row.watch_ratio)
for key, values in per_pair_ratios.items():
    if key not in merged:
        continue
    if any(value >= 1.0 for value in values) and any(value < 1.0 for value in values):
        mixed += 1
check('9-3 首次达标且后来混信号的对有 8,810 个', mixed, 8810)
check('9-3 这类混信号占首次达标对的 4.17%',
      round(mixed / len(merged), 4), 0.0417)

# 同一对里时间次序不同的记录，合并结果必须与输入顺序无关。
key = (rows[0].original_user_id, rows[0].original_item_id)
same_pair = [
    row for row in rows if (row.original_user_id, row.original_item_id) == key
]
if len(same_pair) >= 2:
    forward = merge_duplicate_pairs(same_pair, 0.0)[key]
    backward = merge_duplicate_pairs(list(reversed(same_pair)), 0.0)[key]
    check('9-3 合并结果与输入顺序无关', forward == backward, True)
else:
    print('PASS 9-3 合并结果与输入顺序无关 | 该对只有一条记录，跳过')"""
    )
)

CELLS.append(
    markdown(
        lab_answer_for="9-merge",
        source="""<details>
<summary>💡 参考答案（<code>merge_duplicate_pairs</code>）</summary>

```python
def merge_duplicate_pairs(rows, threshold):
    '''返回首次播放达标的 {(user, video): (first_timestamp, first_watch_ratio)}。'''
    first = {}
    for row in rows:
        key = (row.original_user_id, row.original_item_id)
        previous = first.get(key)
        if previous is None:
            first[key] = (row.timestamp, row.watch_ratio, False)
            continue
        earliest, first_ratio, ambiguous = previous
        if row.timestamp < earliest:
            first[key] = (row.timestamp, row.watch_ratio, False)
        elif row.timestamp == earliest:
            conflict = (row.watch_ratio >= threshold) != (first_ratio >= threshold)
            first[key] = (earliest, min(first_ratio, row.watch_ratio),
                          ambiguous or conflict)
    return {
        key: (timestamp, ratio)
        for key, (timestamp, ratio, ambiguous) in first.items()
        if ratio >= threshold and not ambiguous
    }
```

</details>

最早播放时间和该次播放的标签来自**同一行**。先筛正反馈再找时间会把「先跳过、后来重看完」错标成新的未看过正反馈；取最大比例配最后时间则凭空合成一个从未发生的事件。相同最早时间若标签冲突，就不能借文件行序臆造先后。

这并不是说重播没有兴趣信号；只是本项目的任务是预测用户**下一次尚未观看过**且首次播放达标的视频。若研究「会不会重播」，需要另立任务与评估协议，不能改写首次曝光标签。"""
    )
)

# ---------------------------------------------------------------- 10
CELLS.append(
    markdown(
        """### 🧩 你来写：`temporal_split()` 和它的验收

这一格写的函数就是**这个 bug 的验收工具**：同一个切分规则跑两遍——一遍用原始行，一遍用合并后的对——数出「留出目标出现在该用户训练历史里」的条数。

切分规则：

```text
每个用户按真实 timestamp 排序
不足 3 条，或最后两道边界有同一时间戳：全部进 train，不参与评测
否则：最后一条 = test，倒数第二条 = validation，其余 = train
```

返回 `(train, held_out)`，两者都是 `{user_idx: {item_idx, ...}}`。`user_idx` 用原始 `user_id` 即可。"""
    )
)

CELLS.append(
    code(
        """def temporal_split(events):
    '''events 是 (user, item, timestamp) 三元组的列表。

    返回 (train, held_out)，都是 {user: {item, ...}}。
    '''
    # TODO 9-4：
    #   1. 按 user 收起 (timestamp, item)
    #   2. 每个用户按 timestamp 排序；时间边界不严格则全部进 train
    #   3. 仅在 train < validation < test 时留出最后两条
    raise NotImplementedError('TODO 9-4')


def count_leaked_targets(train, held_out):
    '''数出「留出目标也出现在该用户训练历史里」的条数。'''
    # TODO 9-4：对 held_out 里每个 (user, item)，看 item 是否在 train[user] 里
    raise NotImplementedError('TODO 9-4')""",
        lab_todo="9-split",
    )
)

CELLS.append(
    code(
        """# ✅ 跑一下：同一个切分规则，两种数据形态
unmerged_events = [
    (row.original_user_id, row.original_item_id, row.timestamp) for row in rows
]
merged_events = [
    (user_id, item_id, timestamp)
    for (user_id, item_id), (timestamp, _) in merged.items()
]

train_raw, held_raw = temporal_split(unmerged_events)
train_merged, held_merged = temporal_split(merged_events)

leaked_raw = count_leaked_targets(train_raw, held_raw)
leaked_merged = count_leaked_targets(train_merged, held_merged)

print('原始行    :', len(unmerged_events), '条 → 留出目标', len(held_raw), '个，其中混进训练历史', leaked_raw, '个')
print('合并后的对:', len(merged_events), '条 → 留出目标', len(held_merged), '个，其中混进训练历史', leaked_merged, '个')

check('9-4 原始行严格时间切分有 785 个不同留出目标',
      sum(len(items) for items in held_raw.values()), 785)
check('9-4 其中 9 个用户最后两条指向同一个视频',
      sum(
          1
          for user_id in held_raw
          if len(held_raw[user_id]) < 2
      ),
      9)
check('9-4 原始行状态下有 66 个留出目标混进了训练历史', leaked_raw, 66)
check('9-4 合并后 400 个用户全部参与评测',
      sum(len(items) for items in held_merged.values()), 800)
check('9-4 合并之后一个都不混', leaked_merged, 0)
check('9-4 合并后的数据比原始行少',
      len(merged_events) < len(unmerged_events), True)

labkit.record('lab9_leaked_targets_raw', leaked_raw)
labkit.record('lab9_leaked_targets_merged', leaked_merged)
labkit.record('lab9_merged_pairs_at_1_0', len(merged))"""
    )
)

CELLS.append(
    markdown(
        lab_answer_for="9-split",
        source="""<details>
<summary>💡 参考答案（<code>temporal_split</code> / <code>count_leaked_targets</code>）</summary>

```python
def temporal_split(events):
    '''events 是 (user, item, timestamp) 三元组的列表。

    返回 (train, held_out)，都是 {user: {item, ...}}。
    '''
    by_user = defaultdict(list)
    for user_id, item_id, timestamp in events:
        by_user[user_id].append((timestamp, item_id))

    train = {}
    held_out = {}
    for user_id, entries in by_user.items():
        entries.sort(key=lambda pair: pair[0])
        if (len(entries) < 3 or entries[-3][0] >= entries[-2][0]
                or entries[-2][0] >= entries[-1][0]):
            train[user_id] = {item_id for _, item_id in entries}
            continue
        train[user_id] = {item_id for _, item_id in entries[:-2]}
        held_out[user_id] = {item_id for _, item_id in entries[-2:]}
    return train, held_out


def count_leaked_targets(train, held_out):
    '''数出「留出目标也出现在该用户训练历史里」的条数。'''
    return sum(
        1
        for user_id, items in held_out.items()
        for item_id in items
        if item_id in train.get(user_id, set())
    )
```

</details>

这一格是整章的结论。两种数据形态用的是**同一个**切分规则、同一份原始数据，唯一的差别是合并：

```text
原始行：    785 个不同留出目标，其中 66 个混进了训练历史
合并后的对：800 个留出目标，0 个混进训练历史
```

为什么原始行不是 800：同秒边界用户不参与留出；在剩余用户里，9 个用户的最后两条指向**同一个视频**。首次事件规则之后，每个 `(user, video)` 只出现一次，400 个用户的 800 个留出位置各归不同的视频。

66 / 785 ≈ 8.4% 的留出目标会混进训练历史。这意味着什么，值得写清楚：评测时会把目标之前已经播放过的物品从候选里剔掉，所以这些目标在排序前就被自己剔掉了，**永远不可能命中**。Recall 被系统性压低，而且不会有任何报错。这不是"数据脏一点"，是评测在这个子集上失效了。

合并之后不只是"少了几行"：同样一条切分规则，在两种数据形态下切出的评测目标数和内容都不一样。**标签定义在数据层，评测口径跟着它走**——这就是"必须选在训练之前"的意思：不是模型在变，是评测在变。"""
    )
)

# ---------------------------------------------------------------- 11
CELLS.append(
    markdown(
        """## 🚀 试试看

1. 构造一个 `(user, video)`：第一次 `watch_ratio=0.2`，第二次 `1.5`。用你的函数看它是否作为首次正反馈留下；若留下，检查是不是偷偷先筛了负事件。
2. 把阈值从 1.0 改成 2.0，最后那个格子里的 `leaked_raw` 会变大还是变小？为什么？
3. 把 `rows` 换成前 50 个用户（`labkit.load_kuairec_sample(num_users=50)`），重复第 7 格的表。比例还稳吗？

### 这一章留下的东西

- **一个定义**：`watch_ratio >= 1.0`，写在 `recsys.config` 里，训练/评测/审计都读它。
- **一条规则**：`(user, video)` 保留真实首次播放事件，标签和时间来自同一行；同秒标签冲突则排除。正式实现见 `src/recsys/data/kuairec.py`。
- **一个验收**：合并之后，"留出目标混进训练历史"的条数必须是 0。

这里的 `count_leaked_targets` 只检查正样本切分自身。正式评估还会按目标时间过滤所有已播放物品（包括此前未达标的播放）；Lab 9 不在这一格重复实现完整 Top-K 评估。

### 还没做的

这一章决定的是标签，不是模型。`docs/04-kuairec` 的下一篇要把四个模型（Popularity / MF / Two-Tower / SASRec）放到这份数据上跑一遍，并说明为什么 KuaiRec 的 Recall 不能和 MovieLens 的相减——随机基线差 2.7 倍、评测目标集中度差一倍、人均历史差 5 倍以上，同样的模型在两边本来就不该拿到同一个数。"""
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
    target = REPO_ROOT / "lab9-kuairec-label-threshold.ipynb"
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
