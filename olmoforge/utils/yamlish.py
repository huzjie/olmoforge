"""Minimal YAML-subset parser (zero dependency fallback).

Supports the common subset used by the shipped configs: key: value scalars,
inline comments, nested mappings by indentation, block lists (`- item`),
and lists of mappings (`- key: value`). Used automatically by `config.py`
when PyYAML is not installed, so the CLI keeps working in a bare Python 3.9+.
"""
import re


def _strip_comment(line):
    line = line.rstrip("\n")
    in_q = False
    q = None
    for i, ch in enumerate(line):
        if ch in ("'", '"') and not in_q:
            in_q = True
            q = ch
        elif ch == q and in_q:
            in_q = False
            q = None
        elif ch == "#" and not in_q and (i == 0 or line[i - 1].isspace()):
            return line[:i].rstrip()
    return line.rstrip()


def _scalar(v):
    v = v.strip()
    if v == "":
        return None
    if v.lower() in ("true", "yes", "on"):
        return True
    if v.lower() in ("false", "no", "off"):
        return False
    if v.lower() in ("null", "none", "~"):
        return None
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        return [_scalar(x) for x in inner.split(",")]
    return v


def _indent(line):
    return len(line) - len(line.lstrip(" "))


def parse(text):
    root = {}
    stack = [(-1, root, None, None)]  # (indent, container, parent, key)

    for raw in text.splitlines():
        if not raw.strip():
            continue
        line = _strip_comment(raw)
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped.startswith("- "):
            indent = _indent(line)
            item_raw = stripped[2:].strip()
            # pop stack until an indent < current indent
            while stack and stack[-1][0] >= indent:
                stack.pop()
            container = stack[-1][1]
            key = stack[-1][3]
            if key is not None and not isinstance(container.get(key), list):
                container[key] = []
            if ":" in item_raw and not item_raw.startswith(("{", "[")):
                k, _, v = item_raw.partition(":")
                d = {k.strip(): _scalar(v)}
                if key is not None:
                    container[key].append(d)
                else:
                    container.append(d)
                stack.append((indent, d, key, k.strip()))
            else:
                if key is not None:
                    container[key].append(_scalar(item_raw))
                else:
                    container.append(_scalar(item_raw))
            continue

        if ":" not in stripped:
            continue
        key, _, val = stripped.partition(":")
        key = key.strip()
        indent = _indent(line)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if val.strip() == "":
            parent[key] = {}
            stack.append((indent, parent[key], parent, key))
        else:
            parent[key] = _scalar(val)
    return root
