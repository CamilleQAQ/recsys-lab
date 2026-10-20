"""把 lab4 / lab5 的脚手架格 + 参考答案格拼成一个可以直接用的命名空间。

用途：写 docs 时需要采集正文数字，又不想为了拿数字去跑整本校验。
它和 labs/verify_labs.py 用的是同一套「答案替换 TODO 格」的规则，
所以拿到的就是读者看到的参考实现。

只在采集脚本里用，notebook 本身不依赖它。
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
from pathlib import Path

import torch
from torch import nn

LABS_DIR = Path(__file__).resolve().parent
if str(LABS_DIR) not in sys.path:
    sys.path.insert(0, str(LABS_DIR))

import labkit  # noqa: E402

ANSWER_BLOCK = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
DEFAULT_NOTEBOOKS = ("lab4-sasrec-model.ipynb", "lab5-sasrec-data-and-train.ipynb")


def load_reference_namespace(notebooks=DEFAULT_NOTEBOOKS) -> dict:
    """执行脚手架格与答案格，返回一个带全部参考实现的命名空间。"""
    namespace = {
        "__name__": "__main__",
        "torch": torch,
        "nn": nn,
        "math": math,
        "random": random,
        "labkit": labkit,
        "F": torch.nn.functional,
    }
    for name in notebooks:
        notebook = json.loads((LABS_DIR / name).read_text(encoding="utf-8"))
        answers = {
            cell["metadata"]["lab_answer_for"]: ANSWER_BLOCK.search(
                "".join(cell["source"])
            ).group(1)
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
            and "lab_answer_for" in cell.get("metadata", {})
        }
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            todo = cell.get("metadata", {}).get("lab_todo")
            if todo:
                source = answers[todo]
            # 跳过「import 上一个 lab 的产物」和纯断言格
            if "myrec" in source or "check(" in source:
                continue
            try:
                exec(compile(source, f"<{name}:{index}>", "exec"), namespace)
            except Exception:  # noqa: BLE001
                pass
    return namespace
