"""交互与输出的统一封装。

优先用 rich/questionary 提供美观体验；若尚未安装（例如首次跑 env_check），
自动回退到标准 input/print，保证任何阶段都能运行。
"""
from __future__ import annotations

from typing import Optional, Sequence

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    _console = Console()
    _HAS_RICH = True
except Exception:  # pragma: no cover - 回退路径
    _console = None
    _HAS_RICH = False

try:
    import questionary

    _HAS_Q = True
except Exception:  # pragma: no cover
    questionary = None
    _HAS_Q = False


def info(msg: str) -> None:
    if _HAS_RICH:
        _console.print(msg)
    else:
        print(msg)


def ok(msg: str) -> None:
    info(f"[green]✅ {msg}[/green]" if _HAS_RICH else f"[OK] {msg}")


def warn(msg: str) -> None:
    info(f"[yellow]⚠️  {msg}[/yellow]" if _HAS_RICH else f"[警告] {msg}")


def err(msg: str) -> None:
    info(f"[red]❌ {msg}[/red]" if _HAS_RICH else f"[错误] {msg}")


def title(text: str) -> None:
    if _HAS_RICH:
        _console.print(Panel.fit(text, style="bold cyan"))
    else:
        print("\n==== %s ====" % text)


def table(headers: Sequence[str], rows: Sequence[Sequence[str]], caption: str = "") -> None:
    if _HAS_RICH:
        t = Table(show_header=True, header_style="bold magenta", caption=caption or None)
        for h in headers:
            t.add_column(str(h))
        for r in rows:
            t.add_row(*[str(c) for c in r])
        _console.print(t)
    else:
        print("\t".join(headers))
        for r in rows:
            print("\t".join(str(c) for c in r))
        if caption:
            print(caption)


def ask_text(prompt: str, default: Optional[str] = None) -> str:
    if _HAS_Q:
        return questionary.text(prompt, default=default or "").ask() or (default or "")
    raw = input(f"{prompt}{f' [{default}]' if default else ''}: ").strip()
    return raw or (default or "")


def ask_select(prompt: str, choices: Sequence[str], default: Optional[str] = None) -> str:
    choices = list(choices)
    if _HAS_Q:
        return questionary.select(
            prompt, choices=choices, default=default if default in choices else None
        ).ask()
    # 回退：编号选择
    print(f"\n{prompt}")
    for i, c in enumerate(choices, 1):
        print(f"  {i}) {c}")
    while True:
        raw = input("请输入编号: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1]
        print("无效编号，请重试。")


def ask_confirm(prompt: str, default: bool = True) -> bool:
    if _HAS_Q:
        return bool(questionary.confirm(prompt, default=default).ask())
    suffix = "Y/n" if default else "y/N"
    raw = input(f"{prompt} [{suffix}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "是")
