"""加载基础模型 + LoRA 适配器，进入终端交互对话验证微调效果。"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml

from kit import config, ui


def chat() -> bool:
    state = config.load_state()
    base = state.get("last_model") or state.get("model_path") or state.get("model_repo")
    adapter = state.get("last_run_dir")
    template = state.get("last_template", config.DEFAULTS["template"])
    if not base or not adapter or not Path(adapter).exists():
        ui.err("缺少基础模型或已训练的适配器，请先完成菜单【2】【4】。")
        return False

    cfg = {
        "model_name_or_path": str(base),
        "adapter_name_or_path": str(adapter),
        "template": template,
        "finetuning_type": "lora",
        "trust_remote_code": True,
        "infer_backend": "huggingface",
        # 4-bit 加载：4B 模型在 6GB 显存下若用 bf16 会被 accelerate 下放到 CPU，
        # 触发 peft.load_adapter → accelerate.get_balanced_memory 的
        # "unhashable type: 'set'" 兼容 bug。量化后整模放进单卡即可绕过，
        # 且与 QLoRA 训练设置一致。显存充足可在 infer.yaml 里把它删掉。
        "quantization_bit": 4,
    }
    cfg_path = config.CONFIGS_DIR / "infer.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                        encoding="utf-8")
    ui.ok(f"已生成推理配置: {cfg_path}")
    ui.info("进入对话（输入 exit 退出，clear 清空历史）...")

    env = os.environ.copy()
    if state.get("model_source") == "modelscope":
        env["USE_MODELSCOPE_HUB"] = "1"
    try:
        subprocess.run(["llamafactory-cli", "chat", str(cfg_path)], env=env, check=True)
    except subprocess.CalledProcessError as e:
        ui.err(f"对话进程退出（码 {e.returncode}）。")
        return False
    except FileNotFoundError:
        ui.err("未找到 llamafactory-cli，请确认已激活 .venv。")
        return False
    return True
