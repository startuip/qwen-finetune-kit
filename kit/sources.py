"""模型与数据集来源的统一封装：魔搭社区 / HuggingFace / 本地。

设计原则：
- 搜索尽力而为；任何一步失败都回退到「直接输入仓库 ID」，不阻塞流程。
- 下载统一落到项目内 models/ 或 data/，便于整体迁移。
- 返回值统一为本地绝对路径（字符串）。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from kit import config, ui


# ------------------------------------------------------------------
# 搜索
# ------------------------------------------------------------------
def _search_hf(keyword: str, kind: str, limit: int = 15) -> List[dict]:
    """kind: 'model' | 'dataset'"""
    from huggingface_hub import HfApi

    api = HfApi()
    rows = []
    if kind == "model":
        it = api.list_models(search=keyword, sort="downloads", direction=-1, limit=limit)
        for m in it:
            rows.append({"id": m.id, "downloads": getattr(m, "downloads", 0) or 0,
                         "likes": getattr(m, "likes", 0) or 0})
    else:
        it = api.list_datasets(search=keyword, sort="downloads", direction=-1, limit=limit)
        for d in it:
            rows.append({"id": d.id, "downloads": getattr(d, "downloads", 0) or 0,
                         "likes": getattr(d, "likes", 0) or 0})
    return rows


def _search_modelscope(keyword: str, kind: str, limit: int = 15) -> List[dict]:
    """魔搭关键词搜索，走官方 OpenAPI（GET /openapi/v1/models|datasets）。

    返回每项 {id, downloads, likes}；失败抛异常由调用方回退到手填 ID。
    """
    import requests

    endpoint = os.environ.get("MODELSCOPE_ENDPOINT", "https://www.modelscope.cn").rstrip("/")
    sub = "models" if kind == "model" else "datasets"
    url = f"{endpoint}/openapi/v1/{sub}"
    params = {"search": keyword, "page_size": limit, "page_number": 1, "sort": "downloads"}
    r = requests.get(url, params=params, timeout=20,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    payload = r.json()
    if isinstance(payload, dict) and payload.get("success") is False:
        raise RuntimeError(payload.get("message", "魔搭返回失败"))
    data = payload.get("data", payload) if isinstance(payload, dict) else {}
    items = data.get(sub) or data.get(sub.capitalize()) or []
    rows = []
    for it in items[:limit]:
        rid = it.get("id") or ""
        if rid:
            rows.append({"id": rid,
                         "downloads": it.get("downloads", 0) or 0,
                         "likes": it.get("likes", 0) or 0})
    return rows


def search(source: str, keyword: str, kind: str) -> List[dict]:
    try:
        if source == "huggingface":
            return _search_hf(keyword, kind)
        return _search_modelscope(keyword, kind)
    except Exception as e:
        ui.warn(f"搜索失败（{e}）。可直接输入完整仓库 ID。")
        return []


# ------------------------------------------------------------------
# 下载
# ------------------------------------------------------------------
def _safe_dirname(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def download_hf(repo_id: str, kind: str) -> str:
    from huggingface_hub import snapshot_download

    base = config.MODELS_DIR if kind == "model" else config.DATA_DIR
    local_dir = base / _safe_dirname(repo_id)
    ui.info(f"⬇️  从 HuggingFace 下载 {repo_id} → {local_dir}")
    path = snapshot_download(
        repo_id=repo_id,
        repo_type=("model" if kind == "model" else "dataset"),
        local_dir=str(local_dir),
    )
    return str(path)


def download_modelscope(repo_id: str, kind: str) -> str:
    from modelscope import snapshot_download

    base = config.MODELS_DIR if kind == "model" else config.DATA_DIR
    local_dir = base / _safe_dirname(repo_id)
    ui.info(f"⬇️  从魔搭社区下载 {repo_id} → {local_dir}")
    kwargs = dict(cache_dir=str(config.CACHE_DIR / "modelscope"))
    if kind == "dataset":
        kwargs["repo_type"] = "dataset"
    # 新版支持 local_dir；老版只有 cache_dir，二者都尝试。
    try:
        path = snapshot_download(repo_id, local_dir=str(local_dir), **kwargs)
    except TypeError:
        path = snapshot_download(repo_id, **kwargs)
    return str(path)


def download(source: str, repo_id: str, kind: str) -> str:
    if source == "huggingface":
        return download_hf(repo_id, kind)
    return download_modelscope(repo_id, kind)


# ------------------------------------------------------------------
# 交互式选择（搜索 / 输入 ID / 本地路径）
# ------------------------------------------------------------------
def choose(kind: str) -> Optional[dict]:
    """返回 {'source','repo_id','local_path'} 或 None（取消）。

    source 为 'local' 时仅有 local_path 有效。
    """
    label = "基础模型" if kind == "model" else "数据集"
    mode = ui.ask_select(
        f"选择{label}来源：",
        ["魔搭社区（搜索/下载）", "HuggingFace（搜索/下载）", "本地路径", "返回"],
    )
    if mode == "返回":
        return None
    if mode == "本地路径":
        p = ui.ask_text(f"输入本地{label}路径")
        if not p or not Path(p).exists():
            ui.err("路径不存在。")
            return None
        return {"source": "local", "repo_id": None, "local_path": str(Path(p).resolve())}

    source = "modelscope" if mode.startswith("魔搭") else "huggingface"
    repo_id = _pick_repo_id(source, kind)
    if not repo_id:
        return None
    if not ui.ask_confirm(f"确认下载 {repo_id} ?", default=True):
        return None
    try:
        local = download(source, repo_id, kind)
    except Exception as e:
        ui.err(f"下载失败: {e}")
        return None
    ui.ok(f"已下载到: {local}")
    return {"source": source, "repo_id": repo_id, "local_path": local}


def _pick_repo_id(source: str, kind: str) -> Optional[str]:
    site = "魔搭 modelscope.cn" if source == "modelscope" else "HuggingFace huggingface.co"
    do_search = ui.ask_confirm(f"在 {site} 关键词搜索？(否=直接输入完整 ID)", default=True)
    if do_search:
        kw = ui.ask_text("输入搜索关键词（如 Qwen3.5-4B）")
        rows = search(source, kw, kind) if kw else []
        if rows:
            ui.table(
                ["#", "仓库 ID", "下载量", "点赞"],
                [[i + 1, r["id"], r["downloads"], r["likes"]] for i, r in enumerate(rows)],
                caption="选择编号，或下一步直接输入 ID",
            )
            choices = [r["id"] for r in rows] + ["✏️  手动输入完整 ID", "返回"]
            sel = ui.ask_select("选择仓库：", choices)
            if sel == "返回":
                return None
            if not sel.startswith("✏️"):
                return sel
    rid = ui.ask_text(f"输入完整{('模型' if kind=='model' else '数据集')} ID（如 Qwen/Qwen3.5-4B）")
    return rid or None
