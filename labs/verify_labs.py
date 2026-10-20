"""校验 lab notebook：参考答案确实能通过全部自测。

用法（在仓库根目录或 labs/ 下都可以）：

    python labs/verify_labs.py                      # 校验全部 lab
    python labs/verify_labs.py lab1-two-tower-from-scratch.ipynb

做三件事：

1. **未填写状态**：把 `raise NotImplementedError('TODO ...')` 留在原地跑一遍，
   确认它在第一个 TODO 处停下，也就是空白的 lab 确实跑不通。
2. **逐阶段填写状态**：从标记了 `lab_answer_for` 的 markdown cell 里取出 `<details>`
   中的 python 代码，替换掉对应 `lab_todo` 的代码 cell，然后把整本跑一遍。
3. **答案覆盖率**：每个 `lab_todo` 都要有答案，每个答案块都要被执行到。

这样读者看到的参考答案就是这里被测试的代码，不会出现文档与测试两套的情况。

两条约束：

- 每个 `lab_todo` 代码 cell 要自包含，因为答案会整格替换它。脚手架代码放在不带
  `lab_todo` 的独立 cell 里。
- 如果某个答案是在「回头改之前写的代码」（Lab 1 给 `TwoTower` 加 temperature 就是），
  用 `lab_stage` 分段：后一阶段的答案和对应 cell 都标上 `"lab_stage": 2`。校验器先只跑
  第 1 阶段（用第 1 版答案，执行到第一个 stage-2 cell 为止），再跑第 2 阶段，否则第 1 版
  答案永远不会被执行到。
"""

from __future__ import annotations

import io
import json
import linecache
import os
import re
import sys
import traceback
import warnings
from contextlib import redirect_stdout
from pathlib import Path

LABS_DIR = Path(__file__).resolve().parent
REPO_ROOT = LABS_DIR.parent

# 每个 lab 期望通过的自测条数；改了 lab 记得同步
EXPECTED_CHECKS = {
    "lab1-two-tower-from-scratch": 21,
    "lab2-train-and-evaluate": 17,
    "lab3-false-negatives-and-mask": 18,
    "lab4-sasrec-model": 41,
    "lab5-sasrec-data-and-train": 61,
    "lab6-sasrec-evaluate": 45,
    "lab7-id-only-baselines": 45,
    "lab8-faiss-retrieval": 36,
    "lab9-kuairec-label-threshold": 29,
}
ANSWER_BLOCK = re.compile(r"```python\n(.*?)\n```", re.DOTALL)

# --show-output：把填写后的运行输出原样打出来，用来核对 lab 正文里引用的数字
SHOW_OUTPUT = False

# --debug-warnings：打印每个格产生的警告和它的首行源码，定位「哪个格在报警」
DEBUG_WARNINGS = False


def cell_source(cell: dict) -> str:
    source = cell["source"]
    return source if isinstance(source, str) else "".join(source)


def cell_stage(cell: dict) -> int:
    return int(cell.get("metadata", {}).get("lab_stage", 1))


def extract_answer(cell: dict) -> str:
    """从 <details> 里抽出第一个 python 代码块。"""
    match = ANSWER_BLOCK.search(cell_source(cell))
    if match is None:
        raise ValueError(f"答案 cell 里找不到 ```python 代码块: {cell_source(cell)[:60]}...")
    return match.group(1)


def run_cells(cells: list[dict]):
    namespace = {"__name__": "__main__"}
    buffer = io.StringIO()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with redirect_stdout(buffer):
            for index, cell in enumerate(cells, start=1):
                source = cell_source(cell)
                # 像 Jupyter 一样把每格源码放进 linecache，这样 labkit.harvest
                # 才能按「格」取出读者自己写的代码
                filename = f"/tmp/lab_verify/{index}.py"
                linecache.cache[filename] = (
                    len(source),
                    None,
                    source.splitlines(keepends=True),
                    filename,
                )
                try:
                    exec(compile(source, filename, "exec"), namespace)
                except Exception as error:  # noqa: BLE001
                    return index, error, buffer.getvalue(), [str(w.message) for w in caught]
                if DEBUG_WARNINGS and caught:
                    print(f"!! 第 {index} 格产生警告: {caught[-1].message}", file=sys.stderr)
                    print(f"!! 源码首行: {source.splitlines()[0][:80]}", file=sys.stderr)
    return None, None, buffer.getvalue(), [str(w.message) for w in caught]


def verify(path: Path) -> None:
    name = path.stem
    nb = json.loads(path.read_text(encoding="utf-8"))
    for key in ("cells", "metadata", "nbformat", "nbformat_minor"):
        assert key in nb, f"缺少顶层字段 {key}"
    assert nb["nbformat"] == 4, f"nbformat 应为 4，实际 {nb['nbformat']}"

    code_cells = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
    todo_cells = {
        cell["metadata"]["lab_todo"]: cell
        for cell in code_cells
        if "lab_todo" in cell.get("metadata", {})
    }
    assert todo_cells, "这个 lab 里没有找到任何 lab_todo 代码 cell"

    stages = sorted({cell_stage(cell) for cell in nb["cells"]})
    print(
        f"{name}: {len(nb['cells'])} cells, {len(code_cells)} code cells, "
        f"{len(todo_cells)} 个 TODO, {len(stages)} 个阶段"
    )

    # ---- 1) 未填写 ----
    index, error, _, emitted = run_cells(code_cells)
    assert isinstance(error, NotImplementedError), (
        f"未填写状态期望 NotImplementedError，实际: {error!r}"
    )
    assert not emitted, f"未填写状态出现警告: {emitted}"
    print(f"  未填写   -> 停在 code cell {index}: {error}")

    # ---- 2) 逐阶段填写 ----
    expected = EXPECTED_CHECKS.get(name)
    checked_answers: set[str] = set()
    final_passes = 0

    for stage in stages:
        resolved: dict[str, str] = {}
        for cell in nb["cells"]:
            metadata = cell.get("metadata", {})
            if (
                cell["cell_type"] == "markdown"
                and "lab_answer_for" in metadata
                and cell_stage(cell) <= stage
            ):
                resolved[metadata["lab_answer_for"]] = extract_answer(cell)
        for target in todo_cells:
            assert target in resolved, f"阶段 {stage}: lab_todo={target} 还没有可用的答案"

        solved = []
        for cell in code_cells:
            if cell_stage(cell) > stage:
                break
            target = cell.get("metadata", {}).get("lab_todo")
            if target is not None:
                checked_answers.add(f"阶段{stage}:{target}")
                source = resolved[target]
            else:
                source = cell_source(cell)
            solved.append(
                {
                    "cell_type": "code",
                    "metadata": dict(cell["metadata"]),
                    "execution_count": None,
                    "outputs": [],
                    "source": source,
                }
            )

        index, error, output, emitted = run_cells(solved)
        if error is not None:
            print(output)
            print("---- 出错位置 ----")
            print(
                "".join(
                    traceback.format_exception(type(error), error, error.__traceback__)
                ).strip()
            )
            print("---- 以下是各 TODO 的答案来源 ----")
            for name, source in resolved.items():
                print(f"  {name}: {source.splitlines()[0][:70]}")
            print("---- 出错位置结束 ----")
            raise AssertionError(
                f"阶段 {stage} 在 code cell {index} 失败: {type(error).__name__}: {error}"
            )
        passes = [line for line in output.splitlines() if line.startswith("PASS")]
        fails = [line for line in output.splitlines() if line.startswith("FAIL")]
        assert not fails, f"阶段 {stage} 出现 FAIL: " + "; ".join(fails)
        assert not emitted, f"阶段 {stage} 出现警告，初学者会困惑: {emitted}"
        print(f"  阶段 {stage}   -> 跑到 code cell {len(solved)}，{len(passes)} 项自测 PASS，零警告")
        if SHOW_OUTPUT and stage == stages[-1]:
            print("  ---- 填写后的运行输出 ----")
            for line in output.splitlines():
                print("   ", line)
            print("  ---- 输出结束 ----")
        final_passes = len(passes)

        if stage == stages[-1]:
            for target in todo_cells:
                assert f"阶段{stage}:{target}" in checked_answers, f"{target} 的答案没被执行"

    if expected is not None:
        assert final_passes == expected, f"期望 {expected} 项自测，实际 {final_passes}"


def main(argv: list[str]) -> int:
    global SHOW_OUTPUT, DEBUG_WARNINGS
    if "--show-output" in argv:
        argv = [argument for argument in argv if argument != "--show-output"]
        SHOW_OUTPUT = True
    if "--debug-warnings" in argv:
        argv = [argument for argument in argv if argument != "--debug-warnings"]
        DEBUG_WARNINGS = True
    os.chdir(REPO_ROOT)
    paths = [Path(argument) for argument in argv] if argv else sorted(LABS_DIR.glob("lab*.ipynb"))
    # 允许写 labs/lab2-....ipynb，也允许只写文件名
    paths = [path if path.exists() else LABS_DIR / path.name for path in paths]
    if not paths:
        print("没有找到 lab notebook")
        return 1

    for path in paths:
        verify(path)
        print()
    print(f"全部通过：{len(paths)} 个 lab")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
