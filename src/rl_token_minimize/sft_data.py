import tempfile
from pathlib import Path

from .episode import SYSTEM_PROMPT
from .tasks import Task
from .tools import Workspace
from .traces import generate_trace


def build_sft_example(task: Task) -> dict | None:
    trace = generate_trace(task.files, task.reference_files)
    if trace is None:
        return None
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.prompt},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        ws = Workspace(Path(tmp))
        for name, content in task.files.items():
            target = Path(tmp) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        for call in trace:
            messages.append({
                "role": "assistant",
                "tool_calls": [{"type": "function", "function": call}],
            })
            if call["name"] == "finish":
                break
            result = ws.call(call["name"], call["arguments"])
            assert not result.startswith("error:"), f"{task.task_id}: canonical trace failed: {result}"
            messages.append({"role": "tool", "content": result})
    return {"messages": messages}
