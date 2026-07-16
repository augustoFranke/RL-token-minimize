import inspect
from pathlib import Path


class ToolError(Exception):
    pass


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all files in the workspace as relative paths.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_text",
            "description": "Search all files for a literal text pattern. Returns matching lines as path:line:text.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string", "description": "Literal text to search for."}},
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file. Returns numbered lines. Optionally restrict to a line range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_text",
            "description": "Replace old_text with new_text in a file. old_text must appear exactly once.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Declare the task complete. Call this after making your edits.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


class Workspace:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _resolve(self, path: str) -> Path:
        resolved = (self.root / path).resolve()
        if not resolved.is_relative_to(self.root.resolve()):
            raise ToolError(f"path escapes workspace: {path}")
        return resolved

    def _files(self) -> list[Path]:
        return sorted(
            p for p in self.root.rglob("*")
            if p.is_file() and not any(part.startswith(".") for part in p.relative_to(self.root).parts)
        )

    def list_files(self) -> str:
        return "\n".join(str(p.relative_to(self.root)) for p in self._files())

    def search_text(self, pattern: str) -> str:
        hits = []
        for p in self._files():
            for i, line in enumerate(p.read_text().splitlines(), 1):
                if pattern in line:
                    hits.append(f"{p.relative_to(self.root)}:{i}:{line}")
        return "\n".join(hits) if hits else "no matches"

    def read_file(self, path: str, start_line: int | None = None, end_line: int | None = None) -> str:
        target = self._resolve(path)
        if not target.is_file():
            raise ToolError(f"no such file: {path}")
        lines = target.read_text().splitlines()
        start = (start_line or 1) - 1
        end = end_line if end_line is not None else len(lines)
        return "\n".join(f"{i}|{line}" for i, line in enumerate(lines[start:end], start + 1))

    def replace_text(self, path: str, old_text: str, new_text: str) -> str:
        target = self._resolve(path)
        if not target.is_file():
            raise ToolError(f"no such file: {path}")
        content = target.read_text()
        count = content.count(old_text)
        if count == 0:
            raise ToolError(f"old_text not found in {path}")
        if count > 1:
            raise ToolError(f"old_text appears {count} times in {path}; must be unique")
        target.write_text(content.replace(old_text, new_text, 1))
        return f"replaced in {path}"

    def call(self, name: str, arguments: dict) -> str:
        methods = {"list_files": self.list_files, "search_text": self.search_text,
                   "read_file": self.read_file, "replace_text": self.replace_text}
        if name == "finish":
            return "finished"
        fn = methods.get(name)
        if fn is None:
            return f"error: unknown tool: {name}"
        try:
            inspect.signature(fn).bind(**arguments)
        except TypeError as e:
            return f"error: bad arguments for {name}: {e}"
        try:
            return fn(**arguments)
        except (ToolError, OSError, UnicodeDecodeError) as e:
            return f"error: {e}"

    def snapshot(self) -> dict[str, str]:
        return {str(p.relative_to(self.root)): p.read_text() for p in self._files()}

    def changed_files(self, before: dict[str, str]) -> list[str]:
        after = self.snapshot()
        return sorted(
            set(k for k in before if before[k] != after.get(k)) | (set(after) - set(before))
        )
