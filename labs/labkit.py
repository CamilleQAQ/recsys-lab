"""Labs 的共享脚手架：数据 I/O 和随机种子，不含推荐算法。

分工：

    下载 / 解析 / 编码 / 时间切分 / DataLoader   -> 这里给好
    模型 / loss / 负采样 / 评估循环 / 指标        -> 在 notebook 里写

所以在 Colab 上 `!git clone` 一份仓库就能开跑，不用先写一套数据管道。
"""

from __future__ import annotations

import csv
import inspect
import json
import linecache
import random
import re
import shutil
import ssl
import urllib.request
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import certifi
import torch
from torch.utils.data import DataLoader, Dataset

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
MOVIELENS_MEMBER = "ml-1m/ratings.dat"
POSITIVE_RATING = 4

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"


@dataclass(frozen=True)
class Interaction:
    """一条正向交互：某个用户在某个时刻喜欢过某个物品。"""

    user_idx: int
    item_idx: int
    timestamp: int


@dataclass(frozen=True)
class PreparedData:
    train: list[Interaction]
    validation: list[Interaction]
    test: list[Interaction]
    num_users: int
    num_items: int
    index_to_user: list[int]
    index_to_item: list[int]
    validation_seen: dict[int, set[int]]
    test_seen: dict[int, set[int]]

    def describe(self) -> str:
        return (
            f"users={self.num_users:,} items={self.num_items:,} "
            f"train/val/test="
            f"{len(self.train):,}/{len(self.validation):,}/{len(self.test):,}"
        )


class InteractionDataset(Dataset):
    def __init__(self, interactions: Sequence[Interaction]) -> None:
        self.interactions = list(interactions)

    def __len__(self) -> int:
        return len(self.interactions)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.interactions[index]
        return {
            "user_idx": torch.tensor(row.user_idx, dtype=torch.long),
            "item_idx": torch.tensor(row.item_idx, dtype=torch.long),
            "timestamp": torch.tensor(row.timestamp, dtype=torch.long),
        }


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def download_movielens(data_dir: Path) -> Path:
    """ratings.dat 已存在就直接复用，否则下载并解压一次。"""
    ratings_path = data_dir / "raw" / MOVIELENS_MEMBER
    if ratings_path.exists():
        return ratings_path

    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "ml-1m.zip"
    if not zip_path.exists():
        print(f"downloading MovieLens-1M from {MOVIELENS_URL}")
        context = ssl.create_default_context(cafile=certifi.where())
        temporary_path = zip_path.with_suffix(".zip.part")
        with urllib.request.urlopen(MOVIELENS_URL, context=context, timeout=120) as response:
            with temporary_path.open("wb") as file:
                shutil.copyfileobj(response, file)
        temporary_path.replace(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extract(MOVIELENS_MEMBER, raw_dir)
    return ratings_path


def prepare_movielens(data_dir: Path | None = None) -> PreparedData:
    """筛正反馈 -> 编码成连续 index -> 每个用户按时间留最后两条做 validation/test。"""
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    ratings_path = download_movielens(data_dir)

    positives: list[tuple[int, int, int]] = []
    all_ratings: list[tuple[int, int, int]] = []
    with ratings_path.open("r", encoding="latin-1") as file:
        for line in file:
            user_id, item_id, rating, timestamp = map(int, line.rstrip().split("::"))
            all_ratings.append((user_id, item_id, timestamp))
            if rating >= POSITIVE_RATING:
                positives.append((user_id, item_id, timestamp))

    original_users = sorted({row[0] for row in positives})
    original_items = sorted({row[1] for row in positives})
    user_to_index = {value: index for index, value in enumerate(original_users)}
    item_to_index = {value: index for index, value in enumerate(original_items)}
    encoded = [
        Interaction(user_to_index[user], item_to_index[item], timestamp)
        for user, item, timestamp in positives
    ]

    by_user: dict[int, list[Interaction]] = defaultdict(list)
    for row in encoded:
        by_user[row.user_idx].append(row)

    train: list[Interaction] = []
    validation: list[Interaction] = []
    test: list[Interaction] = []
    for rows in by_user.values():
        rows.sort(key=lambda row: row.timestamp)
        if (len(rows) < 3 or rows[-3].timestamp >= rows[-2].timestamp
                or rows[-2].timestamp >= rows[-1].timestamp):
            train.extend(rows)
        else:
            train.extend(rows[:-2])
            validation.append(rows[-2])
            test.append(rows[-1])

    sort_key = lambda row: (row.user_idx, row.timestamp, row.item_idx)  # noqa: E731
    train.sort(key=sort_key)
    validation.sort(key=sort_key)
    test.sort(key=sort_key)
    validation_times = {row.user_idx: row.timestamp for row in validation}
    test_times = {row.user_idx: row.timestamp for row in test}
    validation_seen = {user_idx: set() for user_idx in validation_times}
    test_seen = {user_idx: set() for user_idx in test_times}
    for user_id, item_id, timestamp in all_ratings:
        user_idx = user_to_index.get(user_id)
        item_idx = item_to_index.get(item_id)
        if user_idx is None or item_idx is None:
            continue
        if timestamp < validation_times.get(user_idx, float("-inf")):
            validation_seen[user_idx].add(item_idx)
        if timestamp < test_times.get(user_idx, float("-inf")):
            test_seen[user_idx].add(item_idx)
    return PreparedData(
        train=train,
        validation=validation,
        test=test,
        num_users=len(original_users),
        num_items=len(original_items),
        index_to_user=original_users,
        index_to_item=original_items,
        validation_seen=validation_seen,
        test_seen=test_seen,
    )


def create_train_loader(
    interactions: Sequence[Interaction],
    batch_size: int = 256,
    seed: int = 42,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        InteractionDataset(interactions),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )


KUAIREC_ARCHIVE_URL = "https://zenodo.org/records/18164998/files/KuaiRec.zip"
KUAIREC_MATRIX_MEMBER = "KuaiRec 2.0/data/big_matrix.csv"
KUAIREC_SAMPLE_COLUMNS = ("user_id", "video_id", "timestamp", "watch_ratio")


@dataclass(frozen=True)
class RawKuaiRecRow:
    """KuaiRec 的一行原始播放记录；标签怎么定由 lab 自己决定。"""

    original_user_id: int
    original_item_id: int
    timestamp: float
    watch_ratio: float


def _kuairec_matrix_path(data_dir: Path) -> Path:
    """官方压缩包里的 ``big_matrix.csv``；已经解压过就直接复用。

    压缩包 431 MB，只有第一次需要下。这里只解出矩阵这一个成员，
    ``small_matrix.csv`` 和几张带特征的表不参与标签实验。
    ``data_dir`` 是仓库的 ``data/``，KuaiRec 自己住在 ``data/kuairec/`` 下——
    和 ``src/recsys/data/kuairec.py`` 用的是同一份文件，不会重复下载。
    """
    kuairec_dir = data_dir / "kuairec"
    matrix_path = kuairec_dir / "raw" / KUAIREC_MATRIX_MEMBER
    if matrix_path.exists():
        return matrix_path

    archive_path = kuairec_dir / "raw" / "KuaiRec.zip"
    if not archive_path.exists():
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading KuaiRec from {KUAIREC_ARCHIVE_URL}（约 431 MB，只下一次）")
        temporary_path = archive_path.with_suffix(".zip.part")
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(
            KUAIREC_ARCHIVE_URL, context=context, timeout=600
        ) as response:
            with temporary_path.open("wb") as file:
                shutil.copyfileobj(response, file)
        temporary_path.replace(archive_path)

    with zipfile.ZipFile(archive_path) as archive:
        archive.extract(KUAIREC_MATRIX_MEMBER, kuairec_dir / "raw")
    return matrix_path


def load_kuairec_sample(
    num_users: int = 400,
    data_dir: Path | None = None,
) -> list[RawKuaiRecRow]:
    """取前 ``num_users`` 个用户（按原始 user_id 升序）的全部播放记录作为样本。

    取整用户而不是随机行，样本内部的时间顺序和重复播放结构才是真的。
    结果缓存成 ``kuairec_sample_{num_users}.csv``，第二次调用不再扫大矩阵。
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    sample_path = data_dir / "kuairec" / f"kuairec_sample_{num_users}.csv"
    rows: list[RawKuaiRecRow] = []

    if sample_path.exists():
        with sample_path.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                rows.append(
                    RawKuaiRecRow(
                        original_user_id=int(row["user_id"]),
                        original_item_id=int(row["video_id"]),
                        timestamp=float(row["timestamp"]),
                        watch_ratio=float(row["watch_ratio"]),
                    )
                )
        return rows

    matrix_path = _kuairec_matrix_path(data_dir)
    wanted: set[int] = set()
    with matrix_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            user_id = int(row["user_id"])
            if len(wanted) < num_users:
                wanted.add(user_id)
            if user_id not in wanted:
                continue
            rows.append(
                RawKuaiRecRow(
                    original_user_id=user_id,
                    original_item_id=int(row["video_id"]),
                    timestamp=float(row["timestamp"]),
                    watch_ratio=float(row["watch_ratio"]),
                )
            )

    sample_path.parent.mkdir(parents=True, exist_ok=True)
    with sample_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=KUAIREC_SAMPLE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "user_id": row.original_user_id,
                    "video_id": row.original_item_id,
                    "timestamp": row.timestamp,
                    "watch_ratio": row.watch_ratio,
                }
            )
    return rows


def _defining_cells(obj: object) -> list[str]:
    """列出所有可能定义了这个对象的格（候选，不是最终结果）。

    一个类可能被拆在好几格里：``class`` 和 ``__init__`` 在一格，``forward`` 在另一格
    （读者的答案会整格替换 TODO 格，所以常常是这样）。所以这里收集**全部**候选，
    由 ``_cells_for`` 决定每一格分别贡献什么。只取第一格会让导出的模块少了方法——
    能 import，跑起来才报「没有 forward」。

    被装饰器包过的函数要先顺着 ``__wrapped__`` 拆开：``functools.wraps`` 会把外层
    wrapper 的 ``co_filename`` 指向装饰器所在的文件，``torch.no_grad()`` 就会踩这个坑
    （不加这一步，导出的是 torch 自己的 ``no_grad.py``）。
    """
    obj = inspect.unwrap(obj)
    code = getattr(obj, "__code__", None)
    if code is not None:
        return [code.co_filename]
    if isinstance(obj, type):
        candidates: list[str] = []
        for value in vars(obj).values():
            inner = getattr(value, "__code__", None)
            if inner is not None and inner.co_filename not in candidates:
                candidates.append(inner.co_filename)
        return candidates
    filename = getattr(obj, "__file__", None)
    return [filename] if filename else []


def _defines(text: str, marker: str) -> bool:
    """``marker`` 是不是这一格里的完整定义。

    ``def foo`` 不能匹配到 ``def foobar``，但 ``def __init__`` 后面跟的是 ``(``，
    所以不能用 ``\\b`` 收尾。
    """
    return (
        re.search(rf"^{marker}(?![0-9A-Za-z_])", text, flags=re.MULTILINE) is not None
        or re.search(rf"^\s+{marker}(?![0-9A-Za-z_])", text, flags=re.MULTILINE)
        is not None
    )


def _cells_for(obj: object) -> list[str]:
    """挑出定义这个对象要用的全部格，按 notebook 顺序排列。

    类是逐个方法看的：``forward`` 可能在另一格里写成 ``def sasrec_forward`` 再挂到类上，
    所以不能只找 ``def forward``。
    """
    name = getattr(obj, "__name__", None)
    if name is None:
        raise TypeError(f"找不到 {obj!r} 的定义位置，无法导出")

    # 每一格需要在里面找到哪些定义
    names_by_cell: dict[str, list[str]] = {}
    if isinstance(obj, type):
        class_marker = f"class {name}"
        class_cell = None
        for cell in _defining_cells(obj):
            if cell and _defines("".join(linecache.getlines(cell)), class_marker):
                class_cell = cell
                break
        if class_cell is None:
            raise RuntimeError(
                f"没有哪一格含 ``class {name}``，导出 {name} 会缺少类定义。"
            )
        names_by_cell.setdefault(class_cell, []).append(class_marker)
        for key, value in vars(obj).items():
            code = getattr(value, "__code__", None)
            if code is None:
                continue
            # 方法可能是「定义成别的名字再挂上来」的：SASRec.forward = sasrec_forward
            marker = f"def {getattr(value, '__name__', key)}"
            if not _defines("".join(linecache.getlines(code.co_filename)), marker):
                marker = f"def {key}"
            names_by_cell.setdefault(code.co_filename, []).append(marker)
    else:
        code = getattr(inspect.unwrap(obj), "__code__", None)
        if code is not None:
            names_by_cell.setdefault(code.co_filename, []).append(f"def {name}")

    selected: list[str] = []
    for cell in _defining_cells(obj):
        if cell is None or cell in selected:
            continue
        wanted = names_by_cell.get(cell)
        if not wanted:
            continue
        text = "".join(linecache.getlines(cell)).rstrip()
        missing = [marker for marker in wanted if not _defines(text, marker)]
        if missing:
            raise RuntimeError(
                f"{cell} 里找不到 {missing}，导出的 {name} 会缺东西。"
                "如果定义它的那一格还没运行，先运行一次。"
            )
        selected.append(cell)
    if not selected:
        raise TypeError(
            f"找不到 {obj!r} 的定义位置，无法导出（它所在的格读不到，或者格是空的）"
        )
    return selected


def harvest(
    path: str,
    *objects: object,
    imports: Iterable[str] = (),
) -> Path:
    """把 notebook 里写好的对象所在的格子，原样导出成一个模块文件。

    lab 之间的交接就靠它：Lab 1 导出 ``myrec/model.py``，Lab 2 直接 import。
    导出的是格子里的原文，不是参考实现。对象跨多格时（类骨架和方法分在格里），
    会把相关的格全部带上。

    >>> harvest('myrec/model.py', TwoTower, in_batch_cross_entropy_loss,
    ...         imports=['import torch', 'from torch import nn',
    ...                  'from torch.nn import functional as F'])
    """
    target = Path(path)
    if not target.is_absolute():
        target = Path(__file__).resolve().parent / target
    target.parent.mkdir(parents=True, exist_ok=True)
    (target.parent / "__init__.py").touch(exist_ok=True)

    cells: list[str] = []
    for obj in objects:
        for cell in _cells_for(obj):
            if cell not in cells:
                cells.append(cell)

    blocks: list[str] = []
    header = "\n".join(line for line in imports if line.strip())
    if header:
        blocks.append(header)
    for cell in cells:
        source = "".join(linecache.getlines(cell)).rstrip()
        if not source:
            raise RuntimeError(
                f"拿不到 {cell} 的源码。把定义它的那一格重新运行一次；"
                "还是不行就改用 %%writefile 手动保存。"
            )
        blocks.append(source)

    target.write_text("\n\n\n".join(blocks) + "\n", encoding="utf-8")
    line_count = len(target.read_text(encoding="utf-8").splitlines())
    print(f"已写出 {target}（{len(cells)} 格，{line_count} 行）")
    return target


RESULTS_PATH = Path(__file__).resolve().parent / "myrec" / "results.json"


def state_path(name: str) -> Path:
    """lab 之间传递的模型文件路径，统一放在 ``labs/myrec/`` 下。

    lab 之间要交接的不只是代码，还有训出来的模型。路径集中在这里定义，
    免得一个 notebook 写 ``myrec/sasrec_lab_trained.pt``、另一个抄错名字。
    """
    directory = Path(__file__).resolve().parent / "myrec"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def record(name: str, value: object) -> None:
    """把一次实验的关键数字记到 ``labs/myrec/results.json``。

    lab 之间用它做交叉校验：Lab 2 记下「不加 mask」的基线，Lab 3 用同一套配置再跑一遍后比对，
    确认两章用的确实是同一个 seed、同一个 loader 和同一组超参。数字对不上就说明某一边漂移了。
    """
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else {}
    results[name] = value
    RESULTS_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def recorded(name: str) -> object:
    """读回之前 ``record`` 的数字。没记录过就报错，提示先跑完记录它的那个 lab。"""
    if not RESULTS_PATH.exists():
        raise RuntimeError(f"还没有任何实验记录（{RESULTS_PATH}）。先跑完记录它的那个 lab。")
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    if name not in results:
        raise RuntimeError(f"没有记录过 {name}。先跑完记录它的那个 lab。")
    return results[name]
