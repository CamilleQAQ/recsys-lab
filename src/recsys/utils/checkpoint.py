"""checkpoint 元信息相关的小工具。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any


def json_ready_config(config: Any) -> dict[str, Any]:
    """把 dataclass 配置转换成可写入 JSON 的普通值。

    ``Path`` 会被转成字符串，因为 checkpoint 的元信息最终以 JSON 保存，
    而 JSON 本身没有路径类型。
    """
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }
