"""Verify PROJECT_REPORT.md file::function citations exist."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
report = (ROOT / "docs" / "PROJECT_REPORT.md").read_text(encoding="utf-8")
cites = re.findall(r"`([^`]+)::([^`]+)`", report)
missing = []
seen = []
for path, name in cites:
    seen.append(f"{path}::{name}")
    target = ROOT / path.replace("/", "\\") if "\\" not in path else ROOT / path
    target = ROOT / Path(*path.split("/"))
    if not target.is_file():
        missing.append(f"missing file {path}::{name}")
        continue
    if target.suffix != ".py":
        continue
    tree = ast.parse(target.read_text(encoding="utf-8"))
    names = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    assigns = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target_node in node.targets:
                if isinstance(target_node, ast.Name):
                    assigns.add(target_node.id)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assigns.add(node.target.id)
    if name not in names and name not in assigns:
        missing.append(f"missing symbol {path}::{name}")

print("citations", len(seen))
print("unique", len(set(seen)))
for row in missing:
    print("FAIL", row)
if not missing:
    print("OK")
