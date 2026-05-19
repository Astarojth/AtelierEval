# question_handlers/imit.py
import gradio as gr
import io
from pathlib import Path
from PIL import Image
from utils.timestamp_utils import record_answer_timestamp

def display_imit_question(state):
    """
    根据 state 更新模仿题的 UI 组件。
    """
    current_index = state.get('current_imit_index', 0)
    tasks = state.get('imit_tasks', [])
    
    if not tasks or current_index >= len(tasks):
        return (
            gr.update(value="No more questions."),
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=""),
            gr.update(visible=False, value=""),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )

    task = tasks[current_index]
    total_questions = len(tasks)
    
    # --- [FIX] 从字典中重建 PIL 图像对象 ---
    image_data = task.get('origin_image_path') or task.get('origin_image')
    target_image_pil = None
    
    # 检查获取的数据是否为包含图像字节的字典
    if isinstance(image_data, dict) and 'bytes' in image_data and image_data['bytes']:
        try:
            # 从二进制数据重建 PIL.Image 对象
            target_image_pil = Image.open(io.BytesIO(image_data['bytes']))
        except Exception as e:
            print(f"Error reconstructing PIL image from bytes for task {task.get('id')}: {e}")
            target_image_pil = None
    # 备用检查：如果它已经是 PIL.Image 对象
    elif isinstance(image_data, Image.Image):
         target_image_pil = image_data
    elif isinstance(image_data, str) and image_data.strip():
        candidate = Path(image_data)
        if candidate.exists():
            try:
                with Image.open(candidate) as img:
                    target_image_pil = img.convert("RGB")
            except Exception as e:
                print(f"Error loading image from path for task {task.get('id')}: {e}")

    image_visible = target_image_pil is not None

    answers = state.get('imit_answers', {})
    current_answer = answers.get(str(task.get('id')), "")

    progress_text = f"**Imitation Question {current_index + 1} of {total_questions}**"
    task_info_text = f"### Task\nYour goal is to write a prompt that replicates the target image on the left as closely as possible."
    
    prev_btn_visible = current_index > 0
    next_btn_visible = current_index < total_questions - 1
    complete_btn_visible = current_index == total_questions - 1

    return (
        gr.update(value=progress_text),
        # 将重建后的 PIL 图像对象传递给 Gradio
        gr.update(value=target_image_pil, visible=image_visible),
        gr.update(value=task_info_text, visible=True),
        gr.update(value=current_answer),
        gr.update(visible=prev_btn_visible),
        gr.update(visible=next_btn_visible),
        gr.update(visible=complete_btn_visible)
    )


def store_imit_answer(state, prompt_input):
    """Stores the user's answer for the current imitation question."""
    import datetime
    current_index = state.get('current_imit_index', 0)
    tasks = state.get('imit_tasks', [])
    if tasks and current_index < len(tasks):
        task_id = tasks[current_index].get('id')
        if task_id is not None:
            if 'imit_answers' not in state:
                state['imit_answers'] = {}
            state['imit_answers'][str(task_id)] = prompt_input or ""
            
            # Store the timestamp when the answer is updated
            if 'imit_timestamps' not in state:
                state['imit_timestamps'] = {}
            state['imit_timestamps'][str(task_id)] = datetime.datetime.utcnow().isoformat() + "Z"
    return state

def navigate_imit(state, direction):
    """Navigates between imitation questions."""
    # 在导航前记录当前题目的时间戳
    # Record timestamp before navigating
    current_index = state.get('current_imit_index', 0)
    record_answer_timestamp(state, "imit", current_index)

    total_questions = len(state.get('imit_tasks', []))
    if direction == "next" and current_index < total_questions - 1:
        state['current_imit_index'] += 1
    elif direction == "prev" and current_index > 0:
        state['current_imit_index'] -= 1
    return state
