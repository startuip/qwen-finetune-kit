"""环境体检：Python / PyTorch+CUDA / GPU 显存 / Ollama / 磁盘空间。

可独立运行： python -m kit.env_check
也被 setup.sh 与主菜单调用。
"""
from __future__ import annotations

import shutil
import subprocess
import sys

from kit import config, ui


def _check_python() -> bool:
    v = sys.version_info
    ok = v[:2] >= (3, 10)
    (ui.ok if ok else ui.err)(f"Python {v.major}.{v.minor}.{v.micro}")
    return ok


def _check_torch_cuda() -> bool:
    try:
        import torch
    except Exception:
        ui.err("未安装 PyTorch（请运行 setup.sh）")
        return False
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        ui.ok(f"PyTorch {torch.__version__} | CUDA 可用 | {name} | 显存 {total:.1f} GB")
        if total < 8:
            ui.warn("显存 < 8GB，已默认启用 QLoRA(4bit)；若仍 OOM 可调小 batch/cutoff_len。")
        return True
    ui.warn(f"PyTorch {torch.__version__} 已装，但 CUDA 不可用（将退化为 CPU，极慢）。")
    return False


def _check_ollama() -> bool:
    if shutil.which("ollama") is None:
        ui.warn("未找到 ollama 命令（导入 Ollama 步骤需要它）。安装见 https://ollama.com")
        return False
    try:
        out = subprocess.run(
            ["ollama", "--version"], capture_output=True, text=True, timeout=10
        )
        ui.ok(f"Ollama: {out.stdout.strip() or out.stderr.strip()}")
        return True
    except Exception as e:
        ui.warn(f"ollama 调用失败: {e}")
        return False


def _check_disk() -> bool:
    usage = shutil.disk_usage(config.ROOT)
    free_gb = usage.free / (1024**3)
    (ui.ok if free_gb > 30 else ui.warn)(f"磁盘可用空间: {free_gb:.0f} GB")
    if free_gb <= 30:
        ui.warn("建议至少预留 30GB（模型 + 合并产物 + GGUF 量化）。")
    return free_gb > 30


def run() -> dict:
    ui.title("环境检查")
    results = {
        "python": _check_python(),
        "torch_cuda": _check_torch_cuda(),
        "ollama": _check_ollama(),
        "disk": _check_disk(),
    }
    ui.info("")
    src = config.get_source_default()
    ui.info(f"默认下载来源: [bold]{src}[/bold]  (USE_MODELSCOPE_HUB={int(config.use_modelscope())})")
    return results


if __name__ == "__main__":
    run()
