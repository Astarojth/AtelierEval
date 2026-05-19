# utils/hf_utils.py
import uuid
import datetime
import json
from datasets import Dataset, load_dataset
import os
import time
import urllib.parse
import requests
import base64
import random
from huggingface_hub import hf_hub_url
from PIL import Image # 新增
import io # 新增

# --- Configuration ---
HF_NAMESPACE = os.environ.get("ATELIEREVAL_HF_NAMESPACE", "your-hf-namespace")

# Repositories for the three task types
OPEN_ENDED_TASKS_REPO_ID = os.environ.get("ATELIEREVAL_HF_TASK_OE", f"{HF_NAMESPACE}/task-base-oe")
CONSTRAINED_TASKS_REPO_ID = os.environ.get("ATELIEREVAL_HF_TASK_CO", f"{HF_NAMESPACE}/task-base-co")
IMIT_TASKS_REPO_ID = os.environ.get("ATELIEREVAL_HF_TASK_IM", f"{HF_NAMESPACE}/task-base-im")

# Submission/result repositories
SUBMISSIONS_REPO_ID = os.environ.get(
    "ATELIEREVAL_HF_SUBMISSIONS",
    f"{HF_NAMESPACE}/tti-prompt-eval-submissions-prod",
)
ALL_RESULT_REPO_ID = os.environ.get("ATELIEREVAL_HF_ALL_RESULT", f"{HF_NAMESPACE}/all_result")
TIMESTAMP_REPO_ID = os.environ.get("ATELIEREVAL_HF_TIMESTAMP", f"{HF_NAMESPACE}/Timestamp")
ALL_TIMESTAMPS_REPO_ID = os.environ.get("ATELIEREVAL_HF_ALL_TIMESTAMPS", f"{HF_NAMESPACE}/all_timestamps")

# --- Replicate (yunwu) / FLUX-schnell settings ---
REPLICATE_BASE_URL = os.environ.get("REPLICATE_BASE_URL", "http://yunwu.ai/replicate")
# 优先取 REPLICATE_API_KEY；兼容你现有 OPENAI_API_KEY / API_KEY 环境变量
REPLICATE_API_KEY = (
    os.environ.get("REPLICATE_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or os.environ.get("API_KEY")
)
REPLICATE_MODEL_PATH = os.environ.get("REPLICATE_MODEL_PATH", "black-forest-labs/flux-schnell")
REPLICATE_POLL_TIMEOUT = float(os.environ.get("REPLICATE_POLL_TIMEOUT", "120"))
REPLICATE_POLL_INTERVAL = float(os.environ.get("REPLICATE_POLL_INTERVAL", "1.2"))
# 固定 seed 值，确保相同 prompt 生成相同图片
REPLICATE_SEED = int(os.environ.get("REPLICATE_SEED", "42"))

# --- Task Loading Functions ---
def load_open_ended_tasks() -> list:
    """Loads the Open-Ended creativity task set."""
    try:
        tasks_dataset = load_dataset(OPEN_ENDED_TASKS_REPO_ID, split='train', token=os.environ.get("HF_TOKEN"))
        tasks = tasks_dataset.to_list()
        for task in tasks:
            task['task_type'] = 'open_ended'
        return tasks
    except Exception as e:
        print(f"❌ Failed to load Open-Ended tasks: {e}")
        return []

def load_constrained_tasks() -> list:
    """Loads the Constrained generation task set."""
    try:
        tasks_dataset = load_dataset(CONSTRAINED_TASKS_REPO_ID, split='train', token=os.environ.get("HF_TOKEN"))
        tasks = tasks_dataset.to_list()
        for task in tasks:
            task['task_type'] = 'constrained'
        return tasks
    except Exception as e:
        print(f"❌ Failed to load Constrained tasks: {e}")
        return []

def load_imit_tasks() -> list:
    """Loads the Imitation task set."""
    try:
        tasks_dataset = load_dataset(IMIT_TASKS_REPO_ID, split='train', token=os.environ.get("HF_TOKEN"))
        tasks = tasks_dataset.to_list()
        for task in tasks:
            task['task_type'] = 'imit'
        return tasks
    except Exception as e:
        print(f"❌ Failed to load Imitation tasks: {e}")
        return []

def load_all_tasks(shuffle: bool = True) -> list:
    """
    Loads all available tasks from all repositories, combines them,
    and optionally shuffles the result. This is the recommended function to use.
    """
    print("Loading all task types...")
    open_ended_tasks = load_open_ended_tasks()
    constrained_tasks = load_constrained_tasks()
    imit_tasks = load_imit_tasks()

    all_tasks = open_ended_tasks + constrained_tasks + imit_tasks

    if shuffle:
        print("Shuffling combined task list...")
        random.shuffle(all_tasks)

    print(f"✅ Successfully loaded a total of {len(all_tasks)} tasks.")
    return all_tasks

# --- 新增: 用于下载私有图片的函数 ---
# --- NEW: Function to download private images ---
def download_private_image(repo_id: str, filename: str):
    """
    使用 HF_TOKEN 从私有的 Hugging Face 数据集仓库下载图片。
    返回一个 PIL Image 对象或一个占位符 URL。
    
    Downloads an image from a private Hugging Face dataset repo using the HF_TOKEN.
    Returns a PIL Image object or a placeholder URL.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("❌ ERROR: HF_TOKEN is not set in Space Secrets. Cannot download private image.")
        return "https://placehold.co/512x512/ff0000/FFFFFF/png?text=HF_TOKEN_MISSING"

    image_url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type='dataset')
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(image_url, headers=headers, timeout=20)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        image = Image.open(io.BytesIO(response.content))
        return image
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to download private image '{filename}': {e}")
        return f"https://placehold.co/512x512/ff0000/FFFFFF/png?text=DOWNLOAD_FAILED"

# --- Image Generation via flux-schnell (Replicate-compatible REST) ---
def generate_image_with_flux(prompt: str) -> str:
    """
    使用 yunwu 的 Replicate 兼容接口:
      1) POST /replicate/v1/models/{model}/predictions 创建任务
      2) GET  /replicate/v1/predictions/{id}           轮询结果
    成功后下载 output(临时链接，≈1h有效)，转为 PNG 的 data URI 返回。
    """
    print(f"Calling Replicate-compatible API (flux-schnell) for prompt: '{prompt[:40]}...'")

    if not REPLICATE_API_KEY:
        print("❌ ERROR: 未设置 REPLICATE_API_KEY/OPENAI_API_KEY/API_KEY。")
        return "https://placehold.co/512x512/ff0000/FFFFFF/png?text=API_KEY_MISSING"

    headers = {
        "Authorization": f"Bearer {REPLICATE_API_KEY}",
        "Content-Type": "application/json",
    }

    # 创建任务
    create_url = f"{REPLICATE_BASE_URL}/v1/models/{REPLICATE_MODEL_PATH}/predictions"
    body = {
        "input": {
            "prompt": prompt,
            "go_fast": False,  # 关闭快速模式以确保可重复性
            "megapixels": "1",
            "num_outputs": 1,
            "aspect_ratio": "1:1",
            "output_format": "png",  # 使用 PNG 格式避免 JPEG 压缩导致的差异
            "output_quality": 100,  # 最高质量，减少压缩损失
            "num_inference_steps": 4,
            "seed": REPLICATE_SEED,  # 固定 seed 值以确保可重复性
            "disable_safety_checker": True  # 禁用安全检查器，避免引入随机性
        }
    }

    try:
        resp = requests.post(create_url, headers=headers, json=body, timeout=30)
        if not resp.ok:
            raise RuntimeError(f"create failed: {resp.status_code} {resp.text}")
        data = resp.json()
        pid = data.get("id")
        if not pid:
            raise RuntimeError(f"no prediction id in response: {data}")

        # 轮询
        get_url = f"{REPLICATE_BASE_URL}/v1/predictions/{pid}"
        deadline = time.time() + REPLICATE_POLL_TIMEOUT

        while True:
            r = requests.get(get_url, headers=headers, timeout=30)
            if not r.ok:
                raise RuntimeError(f"get failed: {r.status_code} {r.text}")
            info = r.json()
            status = info.get("status")
            if status == "succeeded":
                out = info.get("output")
                urls = out if isinstance(out, list) else ([out] if out else [])
                if not urls:
                    raise RuntimeError("empty output urls")

                # 立刻下载第1张图片（链接约1小时有效）
                img_r = requests.get(urls[0], timeout=60)
                img_r.raise_for_status()

                # 为了与原项目保持 data URI（PNG）的一致性，这里把结果转成 PNG 再 base64
                try:
                    img = Image.open(io.BytesIO(img_r.content)).convert("RGB")
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    print("✅ Image generated successfully via Replicate-compatible API.")
                    return f"data:image/png;base64,{b64}"
                except Exception:
                    # 若 PIL 失败，直接用原字节（假设 JPEG）
                    b64 = base64.b64encode(img_r.content).decode("utf-8")
                    return f"data:image/jpeg;base64,{b64}"

            if status == "failed":
                raise RuntimeError(f"prediction failed: {info.get('error')}")

            if time.time() > deadline:
                raise TimeoutError("poll timeout")

            time.sleep(REPLICATE_POLL_INTERVAL)

    except Exception as e:
        print(f"❌ Replicate API call failed: {e}")
        error_text = urllib.parse.quote(str(e))
        return f"https://placehold.co/512x512/ff0000/FFFFFF/png?text=API_ERROR:{error_text}"

def safe_push_to_hub(dataset, repo_id, max_retries=3, retry_delay=5):
    """
    安全地推送数据集到 Hub，包含重试机制和错误处理
    """
    for attempt in range(max_retries):
        try:
            print(f"🔄 尝试推送到 Hub (第 {attempt + 1}/{max_retries} 次): {repo_id}")
            
            dataset.push_to_hub(
                repo_id=repo_id,
                private=True,
                token=os.environ.get("HF_TOKEN")
            )
            
            print(f"✅ 成功推送到 Hub: {repo_id}")
            return True
            
        except Exception as e:
            print(f"❌ 推送失败 (第 {attempt + 1}/{max_retries} 次): {e}")
            
            if attempt < max_retries - 1:
                print(f"⏳ 等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数退避
            else:
                print(f"💥 推送到 {repo_id} 最终失败，已达到最大重试次数")
                return False
    
    return False

# --- Submission Handling ---
def save_batch_submission(submissions: list, user_id: str = "test_user_01"):
    """
    保存批量提交，增强错误处理
    """
    if not submissions:
        print("⚠️ 没有提交数据需要保存")
        return True  # 返回 True 以继续流程
    
    batch_id = str(uuid.uuid4())
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    records_to_save = []
    
    for sub in submissions:
        record = {
            "submission_id": str(uuid.uuid4()),
            "batch_id": batch_id,
            "user_id": user_id,
            "submission_timestamp": timestamp,
            "task_id": sub.get("task_id"),
            "task_type": sub.get("task_type"),
            "response_payload": json.dumps(sub.get("payload")),
        }
        records_to_save.append(record)
    
    try:
        dataset_to_push = Dataset.from_list(records_to_save)
        
        # 使用安全推送函数
        success = safe_push_to_hub(dataset_to_push, SUBMISSIONS_REPO_ID)
        
        if success:
            print(f"✅ Batch {batch_id} with {len(records_to_save)} submissions has been saved.")
            return True
        else:
            print(f"⚠️ Batch {batch_id} 保存失败，但继续处理流程")
            # 即使保存失败，也返回 True 以继续处理流程
            return True
            
    except Exception as e:
        print(f"❌ Failed to create or save batch submission: {e}")
        # 返回 True 以继续处理流程，避免卡住
        return True

def save_complete_user_results(user_email: str, round1_state: dict, round2_state: dict):
    """
    保存用户的完整考试结果到all_result数据集，按照作答顺序记录
    Save user's complete exam results to all_result dataset in answer order

    格式：每个用户一行，第一列为user_email，后续列按照实际作答顺序记录
    每个单元格内容格式：
    轮次_题型_序号
    问题内容
    用户答案
    参考答案

    Format: One row per user, first column is user_email, subsequent columns in actual answer order
    Each cell content format:
    round_type_number
    question content
    user answer
    reference answer
    """
    try:
        print(f"🔍 开始保存用户 {user_email} 的完整结果...")

        # 构建用户的完整结果行
        result_row = {"user_email": user_email}

        # 获取两轮的答题顺序（从 answer_sequence_timestamps）
        round1_timestamps = round1_state.get('answer_sequence_timestamps', [])
        round2_timestamps = round2_state.get('answer_sequence_timestamps', [])

        # 合并所有答题记录，保持顺序
        all_answer_sequence = round1_timestamps + round2_timestamps

        print(f"📋 总共 {len(all_answer_sequence)} 道题的作答记录")

        # 为每道题构建数据
        for idx, ts_entry in enumerate(all_answer_sequence):
            label = ts_entry.get("label", "")  # 例如: "round1_open_ended_1"

            # 解析 label 获取轮次、题型、题号
            # label format examples:
            #   - round1_open_ended_1 -> round=1, type=open_ended, num=1
            #   - round1_constrained_1 -> round=1, type=constrained, num=1
            # 注意：题型名称可能包含下划线（如 open_ended）

            # 从右边分离出题号（最后一个下划线后面的部分）
            last_underscore_idx = label.rfind('_')
            if last_underscore_idx == -1:
                print(f"⚠️ 无法解析标签（缺少下划线）: {label}")
                continue

            try:
                question_num = label[last_underscore_idx + 1:]  # 例如 "1"
                question_index = int(question_num) - 1  # 转换为0-based索引
            except ValueError:
                print(f"⚠️ 无法解析题号: {label}")
                continue

            # 剩余部分是 "round1_open_ended" 或 "round1_constrained" 等
            round_and_type = label[:last_underscore_idx]

            # 从左边分离出轮次（第一个下划线前面的部分）
            first_underscore_idx = round_and_type.find('_')
            if first_underscore_idx == -1:
                print(f"⚠️ 无法解析标签（格式错误）: {label}")
                continue

            round_num = round_and_type[:first_underscore_idx]  # "round1" or "round2"
            question_type = round_and_type[first_underscore_idx + 1:]  # "open_ended", "constrained", "imit"

            # 根据轮次选择对应的state
            if round_num == "round1":
                current_state = round1_state
            else:
                current_state = round2_state

            # 获取任务列表和答案
            tasks = current_state.get(f"{question_type}_tasks", [])
            answers = current_state.get(f"{question_type}_answers", {})

            # 检查索引是否有效
            if question_index >= len(tasks):
                print(f"⚠️ 题目索引超出范围: {label}, index={question_index}, tasks_len={len(tasks)}")
                continue

            task = tasks[question_index]

            # 构建单元格内容
            cell_content = _build_answer_cell(
                label=label,
                question_type=question_type,
                task=task,
                answers=answers
            )

            # 列名格式：answer_1, answer_2, ...
            column_name = f"answer_{idx + 1}"
            result_row[column_name] = cell_content

        # 添加完成时间戳
        result_row["completion_timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"

        print(f"📊 准备保存的结果行包含 {len(result_row)} 个字段")

        # 尝试加载现有数据集并追加新行
        try:
            existing_dataset = load_dataset(ALL_RESULT_REPO_ID, split='train', token=os.environ.get("HF_TOKEN"))
            existing_data = existing_dataset.to_list()
            print(f"📊 现有 all_result 数据集包含 {len(existing_data)} 行")
            all_data = existing_data + [result_row]
            dataset_to_push = Dataset.from_list(all_data)
        except Exception as load_error:
            print(f"🔍 创建新的 all_result 数据集（可能是首次创建）: {load_error}")
            dataset_to_push = Dataset.from_list([result_row])

        # 使用安全推送函数
        success = safe_push_to_hub(dataset_to_push, ALL_RESULT_REPO_ID)

        if success:
            print(f"✅ Complete results for user {user_email} saved to all_result dataset.")
        else:
            print(f"⚠️ 用户 {user_email} 的完整结果保存失败")

        return success

    except Exception as e:
        print(f"❌ Failed to save complete user results: {e}")
        import traceback
        traceback.print_exc()
        return False


def _build_answer_cell(label: str, question_type: str, task: dict, answers: dict) -> str:
    """
    构建单个答案单元格的内容
    Build content for a single answer cell

    格式：
    轮次_题型_序号
    问题内容
    用户答案
    参考答案
    """
    # 获取 task_id
    task_id = str(task.get("task_id") or task.get("id", ""))

    # 构建内容的各个部分
    parts = [label]  # 第一行：标签

    if question_type == "open_ended":
        # Open-Ended: 任务描述、用户prompt、参考prompt
        question = task.get('description', 'No description available.')
        user_prompt = answers.get(task_id, "No prompt submitted.")
        reference_prompt = task.get('ground_truth_prompt') or task.get('prompt', 'N/A')

        parts.append(question)
        parts.append(user_prompt)
        parts.append(reference_prompt)

    elif question_type == "constrained":
        # Constrained: 任务描述、用户prompt、参考prompt
        question = task.get('task', 'No description available.')
        user_prompt = answers.get(task_id, "No prompt submitted.")
        reference_prompt = task.get('prompt', 'N/A')

        parts.append(question)
        parts.append(user_prompt)
        parts.append(reference_prompt)

    elif question_type == "imit":
        # Imitation: 只记录用户prompt和参考prompt（不包括图片）
        # 由于imit没有显式的问题描述，我们用"Imitate the target image"作为描述
        question = "Imitate the target image"
        user_prompt = answers.get(task_id, "No prompt submitted.")
        reference_prompt = task.get('prompt', 'N/A')

        parts.append(question)
        parts.append(user_prompt)
        parts.append(reference_prompt)

    # 用换行符连接所有部分
    return "\n".join(parts)

def save_user_timestamps(user_email: str, round1_timestamps: dict, round2_timestamps: dict):
    """
    保存用户的完整作答时间戳到Timestamp数据集
    """
    try:
        print(f"🕒 开始保存用户 {user_email} 的时间戳...")
        
        # 构建用户的时间戳行
        timestamp_row = {"user_email": user_email}
        
        # 处理第一轮时间戳
        print(f"🕒 第一轮时间戳数据: {round1_timestamps}")

        # 添加第一轮Open-Ended时间戳
        oe_timestamps_r1 = round1_timestamps.get('open_ended_timestamps', {})
        oe_tasks_r1 = round1_timestamps.get('open_ended_tasks', [])
        
        for i, task in enumerate(oe_tasks_r1):
            task_id = str(task.get("task_id") or task.get("id"))
            timestamp = oe_timestamps_r1.get(task_id, "Not Answered")
            timestamp_row[f"round1_open_ended_{i+1}_timestamp"] = timestamp
        
        # 添加第一轮Constrained时间戳
        con_timestamps_r1 = round1_timestamps.get('constrained_timestamps', {})
        con_tasks_r1 = round1_timestamps.get('constrained_tasks', [])
        
        for i, task in enumerate(con_tasks_r1):
            task_id = str(task.get("task_id") or task.get("id"))
            timestamp = con_timestamps_r1.get(task_id, "Not Answered")
            timestamp_row[f"round1_constrained_{i+1}_timestamp"] = timestamp
        
        # 添加第一轮Imitation时间戳
        imit_timestamps_r1 = round1_timestamps.get('imit_timestamps', {})
        imit_tasks_r1 = round1_timestamps.get('imit_tasks', [])
        
        for i, task in enumerate(imit_tasks_r1):
            task_id = str(task.get("id"))
            timestamp = imit_timestamps_r1.get(task_id, "Not Answered")
            timestamp_row[f"round1_imit_{i+1}_timestamp"] = timestamp
        
        # 处理第二轮时间戳
        print(f"🕒 第二轮时间戳数据: {round2_timestamps}")

        # 添加第二轮Open-Ended时间戳
        oe_timestamps_r2 = round2_timestamps.get('open_ended_timestamps', {})
        oe_tasks_r2 = round2_timestamps.get('open_ended_tasks', [])
        
        for i, task in enumerate(oe_tasks_r2):
            task_id = str(task.get("task_id") or task.get("id"))
            timestamp = oe_timestamps_r2.get(task_id, "Not Answered")
            timestamp_row[f"round2_open_ended_{i+1}_timestamp"] = timestamp
        
        # 添加第二轮Constrained时间戳
        con_timestamps_r2 = round2_timestamps.get('constrained_timestamps', {})
        con_tasks_r2 = round2_timestamps.get('constrained_tasks', [])
        
        for i, task in enumerate(con_tasks_r2):
            task_id = str(task.get("task_id") or task.get("id"))
            timestamp = con_timestamps_r2.get(task_id, "Not Answered")
            timestamp_row[f"round2_constrained_{i+1}_timestamp"] = timestamp
        
        # 添加第二轮Imitation时间戳
        imit_timestamps_r2 = round2_timestamps.get('imit_timestamps', {})
        imit_tasks_r2 = round2_timestamps.get('imit_tasks', [])
        
        for i, task in enumerate(imit_tasks_r2):
            task_id = str(task.get("id"))
            timestamp = imit_timestamps_r2.get(task_id, "Not Answered")
            timestamp_row[f"round2_imit_{i+1}_timestamp"] = timestamp
        
        # 添加完成时间戳
        timestamp_row["completion_timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
        
        print(f"🕒 准备保存的时间戳行包含 {len(timestamp_row)} 个字段")
        
        # 尝试加载现有时间戳数据集并追加新行
        try:
            existing_dataset = load_dataset(TIMESTAMP_REPO_ID, split='train', token=os.environ.get("HF_TOKEN"))
            existing_data = existing_dataset.to_list()
            print(f"🕒 现有时间戳数据集包含 {len(existing_data)} 行")
            
            # 合并现有数据和新数据
            all_data = existing_data + [timestamp_row]
            dataset_to_push = Dataset.from_list(all_data)
        except Exception as load_error:
            print(f"🕒 无法加载现有时间戳数据集（可能是首次创建）: {load_error}")
            # 如果数据集不存在或为空，创建新的数据集
            dataset_to_push = Dataset.from_list([timestamp_row])
        
        # 推送到HF Hub
        dataset_to_push.push_to_hub(
            repo_id=TIMESTAMP_REPO_ID,
            private=True,
            token=os.environ.get("HF_TOKEN")
        )
        
        print(f"✅ Complete timestamps for user {user_email} saved to Timestamp dataset.")
        return True
        
    except Exception as e:
        print(f"❌ Failed to save user timestamps: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_answer_sequence_timestamps(user_email: str, round1_timestamps: list, round2_timestamps: list):
    """
    保存用户按作答顺序的时间戳到 all_timestamps 数据集
    Save user's answer sequence timestamps to all_timestamps dataset

    Args:
        user_email: 用户邮箱 (user email)
        round1_timestamps: 第一轮时间戳序列 (round 1 timestamp sequence)
            格式: [{"label": "round1_open_ended_1", "timestamp": "2025-10-20 05:38:15"}, ...]
        round2_timestamps: 第二轮时间戳序列 (round 2 timestamp sequence)
            格式: [{"label": "round2_open_ended_1", "timestamp": "2025-10-20 05:38:15"}, ...]
    """
    try:
        print(f"🕒 开始保存用户 {user_email} 按作答顺序的时间戳...")

        # 构建用户的时间戳行
        timestamp_row = {"user_email": user_email}

        # 合并两轮时间戳
        all_timestamps = round1_timestamps + round2_timestamps

        # 为每个时间戳创建一列
        for i, ts_entry in enumerate(all_timestamps):
            label = ts_entry.get("label", "")
            timestamp = ts_entry.get("timestamp", "")
            # 列名格式：timestamp_1, timestamp_2, ...
            column_name = f"timestamp_{i + 1}"
            # 值格式：round1_constrained_1  2025-10-20 05:38:15
            timestamp_row[column_name] = f"{label}  {timestamp}"

        print(f"🕒 准备保存的时间戳行包含 {len(timestamp_row)} 个字段")

        # 尝试加载现有数据集并追加新行
        try:
            existing_dataset = load_dataset(ALL_TIMESTAMPS_REPO_ID, split='train', token=os.environ.get("HF_TOKEN"))
            existing_data = existing_dataset.to_list()
            print(f"🕒 现有 all_timestamps 数据集包含 {len(existing_data)} 行")

            # 合并现有数据和新数据
            all_data = existing_data + [timestamp_row]
            dataset_to_push = Dataset.from_list(all_data)
        except Exception as load_error:
            print(f"🕒 无法加载现有 all_timestamps 数据集（可能是首次创建）: {load_error}")
            # 如果数据集不存在或为空，创建新的数据集
            dataset_to_push = Dataset.from_list([timestamp_row])

        # 使用安全推送函数
        success = safe_push_to_hub(dataset_to_push, ALL_TIMESTAMPS_REPO_ID)

        if success:
            print(f"✅ Answer sequence timestamps for user {user_email} saved to all_timestamps dataset.")
        else:
            print(f"⚠️ 用户 {user_email} 的时间戳序列保存失败")

        return success

    except Exception as e:
        print(f"❌ Failed to save answer sequence timestamps: {e}")
        import traceback
        traceback.print_exc()
        return False
