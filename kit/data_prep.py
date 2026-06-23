"""把用户选定的数据集注册进 LLaMA-Factory 的 data/dataset_info.json。

支持本地或已下载目录中的 json / jsonl / csv / parquet 文件。
交互式选择格式（alpaca / sharegpt）并做列映射，统一注册成 config.DATASET_NAME。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from kit import config, ui

_DATA_SUFFIXES = (".json", ".jsonl", ".csv", ".parquet")


def _find_data_files(root: Path) -> List[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in _DATA_SUFFIXES else []
    files: List[Path] = []
    for suf in _DATA_SUFFIXES:
        files.extend(sorted(root.rglob(f"*{suf}")))
    return files


def _peek_columns(path: Path) -> List[str]:
    """读取首条样本，推断可用列名（仅对 json/jsonl 生效）。"""
    try:
        if path.suffix.lower() == ".jsonl":
            with path.open(encoding="utf-8") as f:
                first = json.loads(f.readline())
        elif path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            first = data[0] if isinstance(data, list) and data else data
        else:
            return []
        return list(first.keys()) if isinstance(first, dict) else []
    except Exception:
        return []


def _map_column(role: str, cols: List[str], default: str) -> str:
    """让用户把语义角色映射到实际列名。cols 为空时直接手填。"""
    if cols:
        choices = cols + ["（无 / 留空）", "✏️ 手动输入"]
        default_choice = default if default in cols else choices[0]
        sel = ui.ask_select(f"字段映射：{role} →", choices, default=default_choice)
        if sel == "（无 / 留空）":
            return ""
        if sel.startswith("✏️"):
            return ui.ask_text(f"输入 {role} 对应列名", default=default)
        return sel
    return ui.ask_text(f"输入 {role} 对应列名", default=default)


def register(choice: dict) -> bool:
    """choice 来自 sources.choose()。返回是否注册成功。"""
    root = Path(choice["local_path"])
    files = _find_data_files(root)
    if not files:
        ui.err(f"在 {root} 下未找到 json/jsonl/csv/parquet 数据文件。")
        return False

    if len(files) == 1:
        data_file = files[0]
    else:
        rel = [str(f.relative_to(root)) if root.is_dir() else str(f) for f in files]
        sel = ui.ask_select("选择要使用的数据文件：", rel)
        data_file = files[rel.index(sel)]

    cols = _peek_columns(data_file)
    if cols:
        ui.info(f"检测到字段: {', '.join(cols)}")

    fmt = ui.ask_select(
        "数据集格式：",
        ["alpaca（instruction/input/output 指令式）", "sharegpt（多轮对话 conversations）"],
    )

    entry: dict = {"file_name": str(data_file.resolve())}
    if data_file.suffix.lower() == ".parquet":
        entry["file_name"] = str(data_file.resolve())

    if fmt.startswith("alpaca"):
        entry["formatting"] = "alpaca"
        columns = {
            "prompt": _map_column("prompt(指令)", cols, "instruction"),
            "query": _map_column("query(输入,可空)", cols, "input"),
            "response": _map_column("response(回答)", cols, "output"),
            "system": _map_column("system(可空)", cols, "system"),
            "history": _map_column("history(可空)", cols, "history"),
        }
        entry["columns"] = {k: v for k, v in columns.items() if v}
    else:
        entry["formatting"] = "sharegpt"
        msg_col = _map_column("messages(对话列表)", cols, "conversations")
        entry["columns"] = {"messages": msg_col}
        entry["tags"] = {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
        }

    # 合并写入 dataset_info.json（保留用户自定义的其它数据集）
    info = {}
    if config.DATASET_INFO_FILE.exists():
        try:
            info = json.loads(config.DATASET_INFO_FILE.read_text(encoding="utf-8"))
        except Exception:
            info = {}
    info[config.DATASET_NAME] = entry
    config.DATASET_INFO_FILE.write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    config.update_state(
        dataset_source=choice.get("source"),
        dataset_repo=choice.get("repo_id"),
        dataset_file=str(data_file.resolve()),
        dataset_format=entry["formatting"],
    )
    ui.ok(f"已注册数据集 '{config.DATASET_NAME}' → {config.DATASET_INFO_FILE}")
    ui.info(f"映射: {json.dumps(entry.get('columns', {}), ensure_ascii=False)}")
    return True
