# utils/timestamp_utils.py
import datetime

def record_answer_timestamp(state, question_type, question_index):
    """
    记录用户点击next或complete按钮的时间戳
    Record timestamp when user clicks next or complete button

    Args:
        state: 当前状态字典 (current state dict)
        question_type: 题型类型 (question type), e.g., 'mcq', 'open_ended', 'constrained', 'imit'
        question_index: 当前题目索引 (current question index)
    """
    # 获取当前轮次
    assessment_round = state.get('assessment_round', 1)

    # 生成时间戳标签：round1_constrained_1
    # Generate timestamp label: round1_constrained_1
    label = f"round{assessment_round}_{question_type}_{question_index + 1}"

    # 生成时间戳（格式：2025-10-20 05:38:15）
    # Generate timestamp (format: 2025-10-20 05:38:15)
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # 添加到时间戳序列
    # Add to timestamp sequence
    if "answer_sequence_timestamps" not in state:
        state["answer_sequence_timestamps"] = []

    state["answer_sequence_timestamps"].append({
        "label": label,
        "timestamp": timestamp
    })

    return state


def format_timestamps_for_upload(user_email, round1_timestamps, round2_timestamps):
    """
    格式化时间戳数据用于上传到HuggingFace
    Format timestamp data for uploading to HuggingFace

    Args:
        user_email: 用户邮箱 (user email)
        round1_timestamps: 第一轮时间戳列表 (round 1 timestamp list)
        round2_timestamps: 第二轮时间戳列表 (round 2 timestamp list)

    Returns:
        一行数据的字典，包含user_email和所有时间戳列
        A dict representing one row with user_email and all timestamp columns
    """
    row = {"user_email": user_email}

    # 合并两轮时间戳
    # Combine both rounds
    all_timestamps = round1_timestamps + round2_timestamps

    # 为每个时间戳创建一列
    # Create a column for each timestamp
    for i, ts_entry in enumerate(all_timestamps):
        label = ts_entry["label"]
        timestamp = ts_entry["timestamp"]
        # 列名格式：timestamp_1, timestamp_2, ...
        # Column name format: timestamp_1, timestamp_2, ...
        column_name = f"timestamp_{i + 1}"
        # 值格式：round1_constrained_1  2025-10-20 05:38:15
        # Value format: round1_constrained_1  2025-10-20 05:38:15
        row[column_name] = f"{label}  {timestamp}"

    return row
