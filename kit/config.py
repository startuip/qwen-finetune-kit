"""路径常量、默认超参与项目状态读写。

状态保存在 configs/state.json，记住用户上一步选择（模型/数据集/最近一次训练），
让菜单各步骤之间可以衔接，也方便「一键全流程」。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# 加载 .env（若存在）
try:
    from dotenv import load_dotenv

    _ROOT_FOR_ENV = Path(__file__).resolve().parent.parent
    load_dotenv(_ROOT_FOR_ENV / ".env")
except Exception:
    pass

# ---- 路径 ----
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
CONFIGS_DIR = ROOT / "configs"
CACHE_DIR = ROOT / ".cache"
STATE_FILE = CONFIGS_DIR / "state.json"
DATASET_INFO_FILE = DATA_DIR / "dataset_info.json"

for _d in (MODELS_DIR, DATA_DIR, OUTPUT_DIR, CONFIGS_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- 默认超参（6GB 显存下的保守起点，可在交互中覆盖）----
DEFAULTS = {
    "epochs": 3.0,
    "learning_rate": 5e-5,
    "lora_rank": 8,
    "lora_alpha": 16,
    "cutoff_len": 1024,
    "per_device_batch": 1,
    "grad_accum": 8,
    "quantization_bit": 4,  # 4=QLoRA 兜底；设 0 关闭量化
    "template": "qwen3_5",   # Qwen3.5 专属模板（源码版 LLaMA-Factory 提供）
    "logging_steps": 5,
    "save_steps": 100,
    "warmup_ratio": 0.1,
}

# 本项目统一注册到 LLaMA-Factory 的数据集名称
DATASET_NAME = "user_ds"


def get_source_default() -> str:
    """默认下载来源：modelscope / huggingface。"""
    return os.environ.get("DEFAULT_SOURCE", "modelscope").lower()


def use_modelscope() -> bool:
    return os.environ.get("USE_MODELSCOPE_HUB", "0") == "1"


# ---- 状态读写 ----
def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def update_state(**kwargs) -> dict:
    state = load_state()
    state.update(kwargs)
    save_state(state)
    return state
