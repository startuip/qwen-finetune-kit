"""生成 LoRA SFT 配置并调用 llamafactory-cli 训练。"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import List, Optional

import yaml

from kit import config, ui


def _available_templates() -> List[str]:
    """尝试从 LLaMA-Factory 读取可用对话模板，失败返回空列表。"""
    try:
        from llamafactory.data.template import TEMPLATES  # type: ignore

        return sorted(TEMPLATES.keys())
    except Exception:
        return []


def _resolve_model() -> Optional[str]:
    """从状态取基础模型路径/ID。"""
    state = config.load_state()
    path = state.get("model_path") or state.get("model_repo")
    if not path:
        ui.err("尚未选择基础模型，请先在菜单【2】选择模型。")
        return None
    ui.info(f"基础模型: {path}")
    return path


def _ask_float(name: str, default: float) -> float:
    raw = ui.ask_text(name, default=str(default))
    try:
        return float(raw)
    except ValueError:
        return default


def _ask_int(name: str, default: int) -> int:
    raw = ui.ask_text(name, default=str(default))
    try:
        return int(raw)
    except ValueError:
        return default


def configure_and_train() -> bool:
    model = _resolve_model()
    if not model:
        return False
    if not (config.DATASET_INFO_FILE.exists()
            and config.DATASET_NAME in (
                __import__("json").loads(config.DATASET_INFO_FILE.read_text("utf-8")))):
        ui.err("尚未注册数据集，请先在菜单【3】选择数据集。")
        return False

    d = config.DEFAULTS
    ui.title("配置 LoRA 微调超参（回车用默认值）")
    epochs = _ask_float("训练轮数 epochs", d["epochs"])
    lr = _ask_float("学习率 learning_rate", d["learning_rate"])
    rank = _ask_int("LoRA rank", d["lora_rank"])
    cutoff = _ask_int("最大序列长度 cutoff_len", d["cutoff_len"])
    bs = _ask_int("单卡 batch size", d["per_device_batch"])
    accum = _ask_int("梯度累积 grad_accum", d["grad_accum"])
    qbit = _ask_int("量化位数 quantization_bit (4=QLoRA, 0=关闭)", d["quantization_bit"])

    # 模板选择
    templates = _available_templates()
    if templates:
        # Qwen3.5 优先用专属模板，其次 qwen3，再退到配置默认
        default_tpl = next(
            (t for t in ("qwen3_5", d["template"], "qwen3") if t in templates),
            templates[0],
        )
        template = ui.ask_select("对话模板 template：", templates, default=default_tpl)
    else:
        ui.warn("未能从 LLaMA-Factory 读取模板列表（可能依赖未就绪），改为手动输入。")
        template = ui.ask_text("对话模板 template", default="qwen3_5")

    run_name = f"lora-{time.strftime('%Y%m%d-%H%M%S')}"
    output_dir = config.OUTPUT_DIR / run_name

    cfg = {
        "model_name_or_path": str(model),
        "trust_remote_code": True,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": rank,
        "lora_alpha": rank * 2,
        "lora_target": "all",
        "dataset": config.DATASET_NAME,
        "dataset_dir": str(config.DATA_DIR),
        "template": template,
        "cutoff_len": cutoff,
        "overwrite_cache": True,
        "preprocessing_num_workers": 4,
        "output_dir": str(output_dir),
        "logging_steps": d["logging_steps"],
        "save_steps": d["save_steps"],
        "plot_loss": True,
        "overwrite_output_dir": True,
        "per_device_train_batch_size": bs,
        "gradient_accumulation_steps": accum,
        "learning_rate": lr,
        "num_train_epochs": epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": d["warmup_ratio"],
        "bf16": True,
    }
    if qbit in (4, 8):
        cfg["quantization_bit"] = qbit

    cfg_path = config.CONFIGS_DIR / "train_lora.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                        encoding="utf-8")
    ui.ok(f"已生成训练配置: {cfg_path}")
    ui.info(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))

    if not ui.ask_confirm("开始训练？", default=True):
        return False

    env = os.environ.copy()
    if config.load_state().get("model_source") == "modelscope":
        env["USE_MODELSCOPE_HUB"] = "1"

    ui.info("🚀 启动 llamafactory-cli train ...（日志直接输出，Ctrl+C 可中断）")
    try:
        subprocess.run(["llamafactory-cli", "train", str(cfg_path)], env=env, check=True)
    except subprocess.CalledProcessError as e:
        ui.err(f"训练失败（退出码 {e.returncode}）。OOM 时可调小 batch/cutoff_len 或启用 QLoRA。")
        return False
    except FileNotFoundError:
        ui.err("未找到 llamafactory-cli，请确认已运行 setup.sh 并激活 .venv。")
        return False

    config.update_state(
        last_run_dir=str(output_dir),
        last_template=template,
        last_model=str(model),
    )
    ui.ok(f"训练完成！LoRA 适配器位于: {output_dir}")
    return True
