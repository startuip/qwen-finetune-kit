#!/usr/bin/env bash
# 启动交互式菜单
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "❌ 未找到 .venv，请先运行: bash setup.sh"; exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate
exec python -m kit.main
