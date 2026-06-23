"""Qwen 微调工具箱 —— 交互式主菜单。

运行： python -m kit.main  （或 bash run.sh）
"""
from __future__ import annotations

import sys

from kit import config, ui


def _status_line() -> str:
    s = config.load_state()
    model = s.get("model_repo") or s.get("model_path") or "（未选）"
    ds = s.get("dataset_repo") or s.get("dataset_file") or "（未选）"
    run = s.get("last_run_dir") or "（无）"
    ollama = s.get("last_ollama_name") or "（无）"
    return (f"模型: {model}\n数据集: {ds}\n"
            f"最近训练: {run}\nOllama 模型: {ollama}")


def _choose_model() -> None:
    from kit import sources

    res = sources.choose("model")
    if res:
        config.update_state(
            model_source=res["source"],
            model_repo=res["repo_id"],
            model_path=res["local_path"],
        )
        ui.ok("基础模型已记录。")


def _choose_dataset() -> None:
    from kit import data_prep, sources

    res = sources.choose("dataset")
    if res:
        data_prep.register(res)


def _full_pipeline() -> None:
    from kit import export_ollama, train

    ui.title("一键全流程：选模型 → 选数据集 → 训练 → 导入 Ollama")
    _choose_model()
    if not config.load_state().get("model_path") and not config.load_state().get("model_repo"):
        return
    _choose_dataset()
    if not train.configure_and_train():
        return
    export_ollama.export_to_ollama()


MENU = [
    ("检查环境（GPU / Ollama / 依赖 / 磁盘）", "env"),
    ("选择基础模型（魔搭 / HuggingFace / 本地）", "model"),
    ("选择数据集（魔搭 / HuggingFace / 本地）", "dataset"),
    ("配置并开始 LoRA 微调", "train"),
    ("交互式测试微调结果（chat）", "chat"),
    ("合并并导入 Ollama", "export"),
    ("一键全流程（2→3→4→6）", "full"),
    ("退出", "quit"),
]


def _dispatch(action: str) -> bool:
    """返回 False 表示退出。"""
    if action == "env":
        from kit import env_check
        env_check.run()
    elif action == "model":
        _choose_model()
    elif action == "dataset":
        _choose_dataset()
    elif action == "train":
        from kit import train
        train.configure_and_train()
    elif action == "chat":
        from kit import chat
        chat.chat()
    elif action == "export":
        from kit import export_ollama
        export_ollama.export_to_ollama()
    elif action == "full":
        _full_pipeline()
    elif action == "quit":
        return False
    return True


def main() -> None:
    ui.title("Qwen 微调工具箱  (LoRA → Ollama)")
    while True:
        ui.info("\n[dim]当前状态[/dim]")
        ui.info(_status_line())
        labels = [f"{i+1}) {label}" for i, (label, _) in enumerate(MENU)]
        choice = ui.ask_select("请选择操作：", labels)
        idx = labels.index(choice)
        action = MENU[idx][1]
        try:
            if not _dispatch(action):
                ui.ok("再见！")
                break
        except KeyboardInterrupt:
            ui.warn("\n已中断当前操作，返回主菜单。")
        except Exception as e:  # 不让单步异常打断整个菜单
            ui.err(f"操作出错: {e}")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n已退出。")
        sys.exit(0)
