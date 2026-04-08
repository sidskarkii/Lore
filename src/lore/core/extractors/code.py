"""Code extraction — AST-based chunking with context preservation."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from . import ExtractedDocument

_LANG_MAP: dict[str, str] = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".tsx": "typescript", ".jsx": "javascript", ".java": "java",
    ".go": "go", ".rs": "rust", ".cpp": "cpp", ".c": "c",
    ".cs": "csharp", ".rb": "ruby", ".php": "php",
}

_SKIP_DIRS = frozenset({
    "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
    ".git", ".tox", ".mypy_cache", ".pytest_cache", "target",
})


def extract_code(path: str) -> ExtractedDocument:
    """Extract code from a source file, chunked by function/class."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    content = p.read_text(encoding="utf-8", errors="replace")
    ext = p.suffix.lower()

    if ext == ".py":
        sections = _extract_python(content, str(p))
    else:
        sections = _extract_generic(content, str(p))

    return ExtractedDocument(
        sections=sections,
        metadata={"language": _LANG_MAP.get(ext, "unknown"), "filename": p.name},
        source_type="code",
        file_path=str(p),
    )


def extract_code_repo(repo_path: str, extensions: list[str] | None = None) -> list[ExtractedDocument]:
    """Extract all code files from a directory."""
    if extensions is None:
        extensions = list(_LANG_MAP.keys())

    docs: list[ExtractedDocument] = []
    root = Path(repo_path)
    for ext in extensions:
        for fp in root.rglob(f"*{ext}"):
            if any(part in _SKIP_DIRS for part in fp.relative_to(root).parts):
                continue
            try:
                docs.append(extract_code(str(fp)))
            except Exception as e:
                print(f"  Skipping {fp}: {e}")
    return docs


def _extract_python(content: str, file_path: str) -> list[dict]:
    """Parse Python AST to extract functions and classes with context."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return _extract_generic(content, file_path)

    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            seg = ast.get_source_segment(content, node)
            if seg:
                imports.append(seg)
    import_block = "\n".join(imports)

    sections: list[dict] = []
    lines = content.split("\n")

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno - 1
            end = node.end_lineno or start + 1
            body = "\n".join(lines[start:end])
            ctx = f"# File: {file_path}\n{import_block}\n\n" if import_block else f"# File: {file_path}\n\n"
            sections.append({"title": f"function {node.name}()", "text": f"{ctx}{body}"})

        elif isinstance(node, ast.ClassDef):
            start = node.lineno - 1
            end = node.end_lineno or start + 1
            body = "\n".join(lines[start:end])
            ctx = f"# File: {file_path}\n{import_block}\n\n" if import_block else f"# File: {file_path}\n\n"
            sections.append({"title": f"class {node.name}", "text": f"{ctx}{body}"})

    if not sections:
        sections = [{"title": Path(file_path).stem, "text": f"# File: {file_path}\n\n{content}"}]

    return sections


def _extract_generic(content: str, file_path: str) -> list[dict]:
    """Fallback: split by blank-line groups."""
    blocks = re.split(r"\n\n\n+", content)
    sections: list[dict] = []
    for block in blocks:
        block = block.strip()
        if block and len(block) > 20:
            sections.append({"title": "", "text": f"// File: {file_path}\n\n{block}"})

    return sections if sections else [{"title": Path(file_path).stem, "text": content}]
