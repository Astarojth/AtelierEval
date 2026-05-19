# question_handlers/open_ended.py
import gradio as gr
from utils.timestamp_utils import record_answer_timestamp

def display_open_ended_question(state):
    """
    Safely display current Open-Ended question with robust key fallbacks.
    """
    index = state.get("current_open_ended_index", 0)
    tasks = state.get("open_ended_tasks", [])

    # Empty or out-of-range safeguard
    if not tasks or index < 0 or index >= len(tasks):
        return (
            "No Open-Ended tasks available.",
            "### (missing task)",
            gr.update(value=""),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )

    task = tasks[index]

    # Robust description fallback
    description = (
        task.get("description")
        or task.get("task")
        or task.get("title")
        or "No description available."
    )

    # Unified task id retrieval (consistent with processing.py)
    task_id = str(task.get("task_id") or task.get("id") or f"oe_{index}")

    # Ensure answers dict exists
    if "open_ended_answers" not in state:
        state["open_ended_answers"] = {}

    answer = state["open_ended_answers"].get(task_id, "")

    progress = f"**Open-Ended Creation Question {index + 1} of {len(tasks)}**"
    question_md = f"### {description}"

    return (
        progress,
        question_md,
        gr.update(value=answer),
        gr.update(visible=index > 0),
        gr.update(visible=index < len(tasks) - 1),
        gr.update(visible=index == len(tasks) - 1)
    )

def navigate_open_ended(state, direction):
    """
    Navigate between Open-Ended questions with timestamp recording & bounds safety.
    """
    tasks = state.get("open_ended_tasks", [])
    if not tasks:
        return state

    current_index = state.get("current_open_ended_index", 0)
    # Record timestamp for the question being left
    record_answer_timestamp(state, "open_ended", current_index)

    if direction == "prev":
        current_index = max(0, current_index - 1)
    else:
        current_index = min(len(tasks) - 1, current_index + 1)

    state["current_open_ended_index"] = current_index
    return state

def store_open_ended_answer(state, prompt):
    """
    Store user's answer with unified task_id logic and timestamp.
    """
    import datetime
    index = state.get("current_open_ended_index", 0)
    tasks = state.get("open_ended_tasks", [])

    if not tasks or index < 0 or index >= len(tasks):
        return state

    task = tasks[index]
    task_id = str(task.get("task_id") or task.get("id") or f"oe_{index}")

    if "open_ended_answers" not in state:
        state["open_ended_answers"] = {}
    if "open_ended_timestamps" not in state:
        state["open_ended_timestamps"] = {}

    state["open_ended_answers"][task_id] = prompt or ""
    state["open_ended_timestamps"][task_id] = datetime.datetime.utcnow().isoformat() + "Z"

    return state