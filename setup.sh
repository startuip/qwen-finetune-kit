#!/usr/bin/env bash
# ============================================================
# Qwen 微调工具箱 —— 一键环境引导（WSL / Linux）
# 在朋友的电脑 / 新机器上：bash setup.sh 即可重建环境
# 包内不含模型与数据集，运行后在交互菜单里自行下载/选择
# ============================================================
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
echo "==> 项目根目录: $ROOT"

# ---- 1. 选择 Python (>=3.11，LLaMA-Factory 源码版要求) ----
# 依次尝试：环境变量 PYTHON_BIN > python3.12 > python3.11 > python3。
PY_INSTALL_VERSION="${PY_INSTALL_VERSION:-3.12}"   # 缺失时自动安装的版本

pick_python() {
  for c in "${PYTHON_BIN:-}" "python${PY_INSTALL_VERSION}" python3.12 python3.11 python3; do
    [ -z "$c" ] && continue
    command -v "$c" >/dev/null 2>&1 || continue
    if "$c" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,11) else 1)' 2>/dev/null; then
      echo "$c"; return 0
    fi
  done
  return 1
}

# 通过 apt + deadsnakes 自动安装 Python（Ubuntu/Debian）。需要 root 或 sudo。
install_python_apt() {
  local v="$PY_INSTALL_VERSION"
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "❌ 非 apt 系统，无法自动安装 Python。请手动安装 Python>=3.11 后重试。"
    return 1
  fi
  local SUDO=""
  if [ "$(id -u)" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1; then SUDO="sudo"; else
      echo "❌ 需要 root 或 sudo 才能安装 Python。"; return 1
    fi
  fi
  echo "==> 未发现 Python>=3.11，自动安装 python${v}（deadsnakes，需要管理员权限）..."
  $SUDO apt-get update -y
  $SUDO apt-get install -y software-properties-common
  $SUDO add-apt-repository -y ppa:deadsnakes/ppa
  $SUDO apt-get update -y
  $SUDO apt-get install -y "python${v}" "python${v}-venv" "python${v}-dev"
}

PYBIN="$(pick_python || true)"
if [ -z "$PYBIN" ]; then
  if [ "${SKIP_PYTHON_INSTALL:-0}" = "1" ]; then
    echo "❌ 未找到 Python>=3.11，且 SKIP_PYTHON_INSTALL=1。请手动安装后重试。"; exit 1
  fi
  install_python_apt || {
    echo "❌ 自动安装 Python 失败。可手动执行："
    echo "   sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt update"
    echo "   sudo apt install -y python${PY_INSTALL_VERSION} python${PY_INSTALL_VERSION}-venv python${PY_INSTALL_VERSION}-dev"
    exit 1
  }
  PYBIN="$(pick_python || true)"
  [ -z "$PYBIN" ] && { echo "❌ 安装后仍未找到 Python>=3.11。"; exit 1; }
fi
echo "==> 使用 $("$PYBIN" --version) ($PYBIN)"

# ---- 2. 创建/重建虚拟环境 ----
# 若已存在但解释器 < 3.11，则重建，避免沿用旧的 3.10 环境。
if [ -d ".venv" ]; then
  if ! .venv/bin/python -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,11) else 1)' 2>/dev/null; then
    echo "==> 现有 .venv 的 Python 过旧，重建"
    rm -rf .venv
  fi
fi
if [ ! -d ".venv" ]; then
  echo "==> 创建虚拟环境 .venv"
  "$PYBIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# ---- 3. 安装 PyTorch（CUDA 版）----
# WSL 下用 NVIDIA 驱动自带的 CUDA，安装 cu121 wheel（向下兼容新驱动）。
# 如需指定其它 CUDA 版本，改 TORCH_INDEX 环境变量即可。
# 重点：torch / torchvision / torchaudio 必须同源同版本，否则 LLaMA-Factory 会
# 从默认 PyPI 拉到 CUDA 版本不一致的 torchaudio，导入时报 libcudart.so.* 缺失。
# torch 与 torchaudio 版本号一致（如 2.5.1 ↔ 2.5.1），据此校验并按需用 == 强制修正
#（不能用裸 `pip install torchaudio`：pip 会认为已安装版本已满足、不肯降级替换）。
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
if ! python -c "import torch" 2>/dev/null; then
  echo "==> 安装 PyTorch 全家桶 torch/torchvision/torchaudio（来源: $TORCH_INDEX）"
  pip install torch torchvision torchaudio --index-url "$TORCH_INDEX"
fi
# 到此 torch 已在；校验 torchaudio 能否导入且版本与 torch 一致，否则按 torch 版本强制对齐
TVER="$(python -c 'import torch;print(torch.__version__.split("+")[0])' 2>/dev/null || true)"
if [ -n "$TVER" ] && ! python -c "import torch,torchaudio,sys; sys.exit(0 if torchaudio.__version__.split('+')[0]==torch.__version__.split('+')[0] else 1)" 2>/dev/null; then
  echo "==> torchaudio 缺失/版本不一致，强制对齐到 torch $TVER（来源: $TORCH_INDEX）"
  pip install "torchaudio==$TVER" --index-url "$TORCH_INDEX"
fi
# 最终自检：三者都能 import 且 CUDA 可用
python -c "import torch,torchvision,torchaudio,sys; assert torch.cuda.is_available(); print('==> PyTorch 全家桶就绪:', torch.__version__, '| audio', torchaudio.__version__)" \
  || echo "⚠️  PyTorch 全家桶自检未通过，请检查 CUDA/torch 安装。"

# ---- 4. 安装 LLaMA-Factory（GitHub 源码版，支持 Qwen3.5）----
# PyPI 上的 0.9.3 仍把 transformers 限制在 <=4.52.4，无法加载 Qwen3.5(qwen3_5)。
# 源码版 pyproject 允许 transformers<=5.6.0 并内置 qwen3_5 对话模板。
# 可用 LLAMAFACTORY_REF 指定分支/标签/commit 以固定版本（默认 main）。
LF_REF="${LLAMAFACTORY_REF:-main}"
# 注意：新版 LLaMA-Factory(0.9.6.dev0+) 改用 hatchling，不再提供 metrics/bitsandbytes/qwen
# 等 extras（tiktoken/modelscope 已并入核心依赖）。bitsandbytes 与评测指标包改由
# requirements.txt 显式安装，故这里不带 extras。
if python -c "import importlib.metadata as m,sys; v=m.version('llamafactory'); sys.exit(0 if v>='0.9.4' or 'dev' in v else 1)" 2>/dev/null; then
  echo "==> 已安装源码版 LLaMA-Factory，跳过"
else
  echo "==> 从 GitHub 安装 LLaMA-Factory@$LF_REF"
  pip install "llamafactory @ git+https://github.com/hiyouga/LLaMA-Factory.git@${LF_REF}"
fi

# ---- 5. 安装其余依赖 ----
echo "==> 安装项目依赖 (requirements.txt)"
pip install -r requirements.txt

# ---- 6. 准备目录与 .env ----
mkdir -p models data output configs .cache
[ -f .env ] || { cp .env.example .env; echo "==> 已生成 .env（可按需修改）"; }

# ---- 7. 环境体检 ----
echo "==> 运行环境检查"
python -m kit.env_check || true

echo ""
echo "============================================================"
echo "✅ 安装完成！下一步运行交互菜单："
echo "      bash run.sh"
echo "============================================================"
