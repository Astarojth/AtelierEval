# question_handlers/constrained.py
import gradio as gr
from utils.timestamp_utils import record_answer_timestamp

def display_constrained_question(state):
    """
    根据当前状态，显示约束创作题的问题、进度和导航按钮。
    Displays the constrained creation question, progress, and navigation buttons based on the current state.
    """
    index = state.get("current_constrained_index", 0)
    tasks = state.get("constrained_tasks", [])
    
    if not tasks or index >= len(tasks):
        return "No tasks available.", "", gr.update(value=""), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    task = tasks[index]
    
    # [FIX] 统一任务ID的获取逻辑，与 processing.py 保持一致
    # [FIX] Unify the task ID retrieval logic to be consistent with processing.py
    task_id = str(task.get("task_id") or task.get("id"))

    progress_md = f"**Constrained Creation Question {index + 1} of {len(tasks)}**"
    
    description = task.get('task', 'No task description provided.')
    
    constraints_md = f"### Task: {description}\n\n**Your prompt must adhere to the following constraints:**\n"
    constraints_md += f"- **Title:** {task.get('title', 'N/A')}\n"
    constraints_md += f"- **Pairing:** {task.get('pairing', 'N/A')}\n"
    constraints_md += f"- **Constraints:** {task.get('constraint', 'N/A')}\n"
    constraints_md += f"- **Layouts:** {task.get('layout', 'N/A')}\n"
    constraints_md += f"- **Quantity:** {task.get('quantity', 'N/A')}\n"  
    constraints_md += f"- **Text:** {task.get('text', 'N/A')}\n"

    user_prompt = state.get("constrained_answers", {}).get(task_id, "")
    
    return (
        progress_md,
        constraints_md,
        gr.update(value=user_prompt),
        gr.update(visible=(index > 0)),
        gr.update(visible=(index < len(tasks) - 1)),
        gr.update(visible=(index == len(tasks) - 1))
    )

def navigate_constrained(state, direction):
    """
    导航到上一个或下一个约束性问题。
    Navigates to the previous or next constrained question.
    """
    # 在导航前记录当前题目的时间戳
    # Record timestamp before navigating
    current_index = state.get("current_constrained_index", 0)
    record_answer_timestamp(state, "constrained", current_index)

    tasks_count = len(state.get("constrained_tasks", []))

    if direction == "prev":
        state["current_constrained_index"] = max(0, current_index - 1)
    elif current_index < tasks_count - 1:
        state["current_constrained_index"] = current_index + 1

    return state

def store_constrained_answer(state, written_prompt):
    """
    存储当前约束性问题的用户答案。
    Stores the user's answer for the current constrained question.
    """
    import datetime
    index = state.get("current_constrained_index", 0)
    tasks = state.get("constrained_tasks", [])
    
    if tasks and index < len(tasks):
        task = tasks[index]
        # [FIX] 统一任务ID的获取逻辑，与 processing.py 保持一致
        # [FIX] Unify the task ID retrieval logic to be consistent with processing.py
        task_id = str(task.get("task_id") or task.get("id"))
        
        if "constrained_answers" not in state:
            state["constrained_answers"] = {}
        state["constrained_answers"][task_id] = written_prompt or ""
        
        # Store the timestamp when the answer is updated
        if "constrained_timestamps" not in state:
            state["constrained_timestamps"] = {}
        state["constrained_timestamps"][task_id] = datetime.datetime.utcnow().isoformat() + "Z"
        
    return state