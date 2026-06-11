"""保留注释的 config.yaml 就地补丁。

PyYAML 不保留注释/顺序, 故按行编辑: 给定 {点路径: 值}, 找到对应 section 下的 key 行,
只替换其值并保留行尾注释; 缺失的 key 追加到该 section 末尾; 缺失的 section 追加到文件末尾。
仅支持一层嵌套 (section.key), 满足 signal.* / execution.max_hold_seconds 的需要。
"""
from __future__ import annotations

import re
from pathlib import Path


def _format_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        s = repr(round(v, 6))
        return s
    return str(v)


def set_in_text(text: str, dotted: str, value) -> str:
    section, _, key = dotted.partition(".")
    if not key:
        return text
    val = _format_value(value)
    lines = text.split("\n")
    sec_re = re.compile(rf"^{re.escape(section)}:\s*(#.*)?$")
    start = None
    for i, ln in enumerate(lines):
        if sec_re.match(ln):
            start = i
            break
    if start is None:                       # 整个 section 缺失 -> 追加
        if lines and lines[-1].strip() != "":
            lines.append("")
        lines.append(f"{section}:")
        lines.append(f"  {key}: {val}")
        return "\n".join(lines)

    # section 范围: 到下一个顶格(非空/非注释)行为止
    end = len(lines)
    for j in range(start + 1, len(lines)):
        ln = lines[j]
        if ln.strip() == "" or ln.lstrip().startswith("#"):
            continue
        if ln[:1] not in (" ", "\t"):
            end = j
            break

    key_re = re.compile(rf"^(\s+){re.escape(key)}:[ \t]*(.*?)([ \t]*#.*)?$")
    for j in range(start + 1, end):
        m = key_re.match(lines[j])
        if m:
            indent = m.group(1)
            comment = m.group(3) or ""
            lines[j] = f"{indent}{key}: {val}{comment}"
            return "\n".join(lines)

    # key 缺失 -> 插到 section 末尾(回退过尾部空行)
    insert_at = end
    while insert_at - 1 > start and lines[insert_at - 1].strip() == "":
        insert_at -= 1
    lines.insert(insert_at, f"  {key}: {val}")
    return "\n".join(lines)


def apply_updates(path: str | Path, updates: dict) -> list[str]:
    """对文件就地应用一批 {点路径: 值}, 返回变更摘要行。"""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    summary = []
    for dotted, value in updates.items():
        text = set_in_text(text, dotted, value)
        summary.append(f"{dotted} = {_format_value(value)}")
    p.write_text(text, encoding="utf-8")
    return summary
