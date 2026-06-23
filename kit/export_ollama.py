"""把微调结果合并并导入 Ollama。

主路径（推荐，无需编译 llama.cpp）：
  llamafactory-cli export 合并 LoRA → safetensors 目录
  → 写 Modelfile（FROM . + Qwen ChatML 模板）
  → ollama create <name> -f Modelfile --quantize q4_K_M

回退路径（当 Ollama 不支持该架构直导 safetensors 时）：
  clone+build llama.cpp → convert_hf_to_gguf.py → llama-quantize → Modelfile(FROM gguf)
  → ollama create
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import yaml

from kit import config, ui

# Qwen 系列通用 ChatML 模板（Qwen2.5 / Qwen3 / Qwen3.5 兼容）
QWEN_TEMPLATE = (
    '''TEMPLATE """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
"""
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.6
PARAMETER top_p 0.95
'''
)


def _merge_lora() -> Optional[Path]:
    state = config.load_state()
    base = state.get("last_model") or state.get("model_path") or state.get("model_repo")
    adapter = state.get("last_run_dir")
    template = state.get("last_template", config.DEFAULTS["template"])
    if not base or not adapter or not Path(adapter).exists():
        ui.err("缺少基础模型或适配器，请先完成训练（菜单【4】）。")
        return None

    merged_dir = Path(str(adapter) + "-merged")
    cfg = {
        "model_name_or_path": str(base),
        "adapter_name_or_path": str(adapter),
        "template": template,
        "finetuning_type": "lora",
        "export_dir": str(merged_dir),
        "export_size": 5,
        "export_legacy_format": False,
        "trust_remote_code": True,
    }
    cfg_path = config.CONFIGS_DIR / "merge.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                        encoding="utf-8")
    ui.info(f"🔀 合并 LoRA → {merged_dir}")
    env = os.environ.copy()
    if state.get("model_source") == "modelscope":
        env["USE_MODELSCOPE_HUB"] = "1"
    try:
        subprocess.run(["llamafactory-cli", "export", str(cfg_path)], env=env, check=True)
    except subprocess.CalledProcessError as e:
        ui.err(f"合并失败（码 {e.returncode}）。")
        return None
    except FileNotFoundError:
        ui.err("未找到 llamafactory-cli，请确认已激活 .venv。")
        return None
    ui.ok(f"合并完成: {merged_dir}")
    config.update_state(last_merged_dir=str(merged_dir))
    return merged_dir


def _write_modelfile(from_target: str, dest: Path) -> Path:
    content = f"FROM {from_target}\n{QWEN_TEMPLATE}"
    mf = dest / "Modelfile"
    mf.write_text(content, encoding="utf-8")
    return mf


def _ollama_create(name: str, modelfile: Path, quantize: Optional[str]) -> bool:
    cmd = ["ollama", "create", name, "-f", str(modelfile)]
    if quantize:
        cmd += ["--quantize", quantize]
    ui.info(f"🦙 {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        ui.warn(f"ollama create 失败（码 {e.returncode}）。")
        return False
    except FileNotFoundError:
        ui.err("未找到 ollama 命令，请先安装 Ollama (https://ollama.com)。")
        return False


def _smoke_test(name: str) -> None:
    if ui.ask_confirm(f"运行冒烟测试 `ollama run {name}` ?", default=True):
        prompt = ui.ask_text("测试提示词", default="你好，请简单自我介绍")
        try:
            subprocess.run(["ollama", "run", name, prompt], check=False)
        except FileNotFoundError:
            pass


# ------------------------------------------------------------------
# 回退路径：llama.cpp
# ------------------------------------------------------------------
def _fallback_llamacpp(merged_dir: Path, name: str) -> bool:
    ui.title("回退路径：通过 llama.cpp 转 GGUF")
    if not ui.ask_confirm("Ollama 直导失败，尝试用 llama.cpp 转换？（需 git/cmake/编译，耗时较长）",
                          default=True):
        return False
    for tool in ("git", "cmake"):
        if shutil.which(tool) is None:
            ui.err(f"缺少 {tool}，请先安装：sudo apt install -y git cmake build-essential")
            return False

    lc = config.ROOT / ".cache" / "llama.cpp"
    if not lc.exists():
        ui.info("克隆 llama.cpp ...")
        if subprocess.run(["git", "clone", "--depth", "1",
                           "https://github.com/ggml-org/llama.cpp", str(lc)]).returncode != 0:
            ui.err("克隆失败。")
            return False
    quant_bin = lc / "build" / "bin" / "llama-quantize"
    if not quant_bin.exists():
        ui.info("编译 llama.cpp（首次较慢）...")
        subprocess.run(["cmake", "-B", "build"], cwd=str(lc))
        subprocess.run(["cmake", "--build", "build", "--config", "Release", "-j"], cwd=str(lc))

    # 转换依赖
    subprocess.run(["pip", "install", "-r", str(lc / "requirements.txt")], check=False)

    f16 = merged_dir / "model-f16.gguf"
    ui.info("convert_hf_to_gguf.py → f16 GGUF ...")
    conv = subprocess.run(
        ["python", str(lc / "convert_hf_to_gguf.py"), str(merged_dir),
         "--outfile", str(f16), "--outtype", "f16"]
    )
    if conv.returncode != 0 or not f16.exists():
        ui.err("GGUF 转换失败：当前模型架构可能尚未被 llama.cpp 支持。")
        ui.warn("建议：改用已被支持的模型（如 Qwen3 系列），或等待 llama.cpp 适配。")
        return False

    q4 = merged_dir / "model-q4_k_m.gguf"
    if quant_bin.exists():
        subprocess.run([str(quant_bin), str(f16), str(q4), "Q4_K_M"], check=False)
    target_gguf = q4 if q4.exists() else f16
    mf = _write_modelfile(f"./{target_gguf.name}", merged_dir)
    return _ollama_create(name, mf, quantize=None)


# ------------------------------------------------------------------
# 入口
# ------------------------------------------------------------------
def export_to_ollama() -> bool:
    merged = _merge_lora()
    if not merged:
        return False

    default_name = "my-qwen-ft"
    name = ui.ask_text("Ollama 模型名称", default=default_name) or default_name
    quant = ui.ask_select(
        "导入时量化精度：",
        ["q4_K_M（推荐，体积小）", "q8_0（更高精度）", "不量化（f16，体积大）"],
    )
    quant_arg = {"q4_K_M（推荐，体积小）": "q4_K_M",
                 "q8_0（更高精度）": "q8_0",
                 "不量化（f16，体积大）": None}[quant]

    # 主路径：safetensors 直导
    mf = _write_modelfile(".", merged)
    ui.info("尝试主路径：Ollama 直接导入合并后的 safetensors ...")
    if _ollama_create(name, mf, quant_arg):
        ui.ok(f"✅ 已导入 Ollama：{name}")
        config.update_state(last_ollama_name=name)
        _smoke_test(name)
        return True

    # 回退路径
    ui.warn("主路径失败，转入 llama.cpp 回退路径。")
    if _fallback_llamacpp(merged, name):
        ui.ok(f"✅ 经 llama.cpp 已导入 Ollama：{name}")
        config.update_state(last_ollama_name=name)
        _smoke_test(name)
        return True

    ui.err("导入 Ollama 未成功。")
    ui.warn("可能原因：Qwen3.5 全新混合/多模态架构暂未被 Ollama 与 llama.cpp 支持转换。")
    ui.warn("替代方案：① 换用 Qwen3 系列等已支持模型；"
            "② 保留适配器走 `FROM <base> + ADAPTER`（需非量化适配器）。")
    return False
