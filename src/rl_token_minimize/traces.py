import difflib

MAX_CONTEXT = 10


def _hunk_replacements(buggy: str, fixed: str) -> list[tuple[str, str]] | None:
    old_lines = buggy.splitlines(keepends=True)
    new_lines = fixed.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    replacements = []
    content = buggy
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        for context in range(MAX_CONTEXT + 1):
            lo, hi = max(0, i1 - context), min(len(old_lines), i2 + context)
            old_text = "".join(old_lines[lo:hi])
            new_text = "".join(new_lines[j1:j2]) if context == 0 else (
                "".join(old_lines[lo:i1]) + "".join(new_lines[j1:j2]) + "".join(old_lines[i2:hi])
            )
            if old_text and content.count(old_text) == 1:
                replacements.append((old_text, new_text))
                content = content.replace(old_text, new_text, 1)
                break
        else:
            return None
    return replacements if content == fixed else None


def generate_trace(files: dict[str, str], reference_files: dict[str, str]) -> list[dict] | None:
    if set(files) != set(reference_files):
        return None
    changed = [p for p in sorted(files) if files[p] != reference_files[p]]
    if not changed:
        return None
    trace = []
    for path in changed:
        replacements = _hunk_replacements(files[path], reference_files[path])
        if replacements is None:
            return None
        trace.append({"name": "read_file", "arguments": {"path": path}})
        trace.extend(
            {"name": "replace_text", "arguments": {"path": path, "old_text": old, "new_text": new}}
            for old, new in replacements
        )
    trace.append({"name": "finish", "arguments": {}})
    return trace
