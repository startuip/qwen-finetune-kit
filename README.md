# Qwen 微调工具箱（LoRA → Ollama）

一个**交互式命令行**工具：在本机微调 Qwen（默认目标 `Qwen/Qwen3.5-4B`）等大模型，并一键导入 [Ollama](https://ollama.com) 使用。

- 🧰 引擎：[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)，LoRA / QLoRA 微调
- 🔎 模型与数据集**自由选择**：从**魔搭社区** / **HuggingFace** 搜索下载，或选**本地**文件
- 🦙 微调后**合并并导入 Ollama**（主路径免编译 llama.cpp，附带回退路径）
- 📦 **开箱即用 & 可分享**：仓库内**不含模型/数据集**，朋友拿到后 `bash setup.sh` 即可重建环境

> 适用环境：WSL / Linux + NVIDIA GPU（CUDA）+ **Python ≥ 3.11**（LLaMA-Factory 源码版要求）。当前默认忽略显存不足风险，6GB 显存默认启用 QLoRA(4bit) 兜底。

---

## 前置：Python 3.11+

LLaMA-Factory 源码版要求 Python ≥ 3.11。`setup.sh` 会自动处理：

- 优先选用已有的 `python3.12` / `python3.11`（或用 `PYTHON_BIN=/path/to/python bash setup.sh` 指定）；
- 若都没有（如 Ubuntu 22.04 只有 3.10），**自动通过 deadsnakes 安装 `python3.12`**（需要 root 或 sudo，会请求管理员权限）。

不想让脚本自动装系统包，可设 `SKIP_PYTHON_INSTALL=1 bash setup.sh`，然后手动安装：

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev
```

> 可用 `PY_INSTALL_VERSION=3.11 bash setup.sh` 改变自动安装的版本。

---

## 快速开始

```bash
# 1) 重建环境（建 venv、装依赖、检测 GPU/Ollama）
bash setup.sh

# 2) 启动交互菜单
bash run.sh
```

菜单：

```
 1) 检查环境（GPU / Ollama / 依赖 / 磁盘）
 2) 选择基础模型（魔搭 / HuggingFace / 本地）
 3) 选择数据集（魔搭 / HuggingFace / 本地）
 4) 配置并开始 LoRA 微调
 5) 交互式测试微调结果（chat）
 6) 合并并导入 Ollama
 7) 一键全流程（2→3→4→6）
 0) 退出
```

完成第 6 步后：

```bash
ollama run my-qwen-ft "你好"
```

---

## 配置

复制 `.env.example` 为 `.env` 调整默认值（`setup.sh` 会自动生成）：

| 变量 | 说明 |
|------|------|
| `DEFAULT_SOURCE` | 默认下载来源：`modelscope` / `huggingface` |
| `USE_MODELSCOPE_HUB` | 选魔搭来源时让 LLaMA-Factory 也走魔搭（`1` 开启） |
| `HF_ENDPOINT` | HuggingFace 镜像，如 `https://hf-mirror.com` |
| `HF_HOME` / `MODELSCOPE_CACHE` | 缓存目录（默认在项目内便于迁移） |

PyTorch 默认装 `cu121` wheel；如需其它 CUDA 版本：

```bash
TORCH_INDEX=https://download.pytorch.org/whl/cu124 bash setup.sh
```

---

## 数据集格式

支持本地或下载目录中的 `json / jsonl / csv / parquet`，两种格式：

- **alpaca**（指令式）：字段如 `instruction` / `input` / `output`
- **sharegpt**（多轮对话）：字段如 `conversations`（`from` / `value`）

注册时会交互式让你把语义字段映射到实际列名，写入 `data/dataset_info.json`。

---

## 目录结构

```
.
├── setup.sh / run.sh          # 引导与启动
├── requirements.txt
├── .env.example
├── kit/                       # 全部逻辑
│   ├── main.py                # 交互式主菜单
│   ├── env_check.py           # 环境检查
│   ├── sources.py             # 魔搭/HF/本地 搜索+下载
│   ├── data_prep.py           # 数据集注册
│   ├── train.py               # LoRA 训练
│   ├── chat.py                # 交互测试
│   └── export_ollama.py       # 合并 + 导入 Ollama
├── models/  data/  output/    # 运行时产物（git 忽略，不随仓库分享）
└── configs/                   # 运行时生成的 YAML 与 state.json
```

---

## 分享给朋友 / 换电脑

1. 直接拷贝/`git clone` 本仓库（不含模型与数据，体积很小）。
2. 对方 `bash setup.sh` 自动重建 `.venv` 与依赖。
3. `bash run.sh` 后在菜单里自行下载所需模型/数据集。

---

## 关于 Qwen3.5 的兼容性说明

`Qwen3.5` 是全新混合架构 + 原生多模态模型。微调（LoRA）通常没问题，但**导出到 GGUF/Ollama** 依赖 llama.cpp / Ollama 对该架构的支持：

- 工具优先走 **Ollama 直接导入合并后的 safetensors**（`ollama create --quantize`）。
- 失败时自动回退到 **llama.cpp 转 GGUF**。
- 若两者均暂不支持，工具会明确提示，并建议：改用已支持的模型（如 Qwen3 系列），或保留适配器走 `FROM <base> + ADAPTER`。

这是外部生态的支持节奏问题，非本工具缺陷。
