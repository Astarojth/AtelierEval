# state_management.py
import random
import config
import time

def get_initial_state():
    """
    为新的多阶段测试会话初始化状态。
    Initializes the state for a new multi-stage exam session.
    """
    # 将导入语句移到函数内部，确保在调用时获取最新的任务列表
    # Move import statements inside the function to ensure the latest task lists are fetched upon call
    from data_loader import OPEN_ENDED_TASKS, CONSTRAINED_TASKS, IMIT_TASKS

    # 为每种题型从配置中获取样本大小，并与实际任务数量取较小值
    # Get sample sizes from config for each task type, taking the minimum of available tasks
    open_ended_sample_size = min(len(OPEN_ENDED_TASKS), config.OPEN_ENDED_EXAM_SIZE)
    constrained_sample_size = min(len(CONSTRAINED_TASKS), config.CONSTRAINED_EXAM_SIZE)
    imit_sample_size = min(len(IMIT_TASKS), config.IMIT_EXAM_SIZE) 
    
    # --- 随机化题型顺序 (Randomize Question Type Order) ---
    # 生成三种题型的随机顺序，每轮考试独立随机
    # Generate random order of three question types, independently randomized for each round
    question_types = ['open_ended', 'constrained', 'imit']
    randomized_order = random.sample(question_types, k=len(question_types))

    return {
        # --- 题型顺序 (Question Type Order) ---
        "question_type_order": randomized_order,  # 随机化的题型顺序
        "current_section_index": 0,  # 当前进行到第几个题型 (0-3)

        # --- 任务列表 (Task Lists) ---
        "open_ended_tasks": random.sample(OPEN_ENDED_TASKS, k=open_ended_sample_size) if OPEN_ENDED_TASKS else [],
        "constrained_tasks": random.sample(CONSTRAINED_TASKS, k=constrained_sample_size) if CONSTRAINED_TASKS else [],
        "imit_tasks": random.sample(IMIT_TASKS, k=imit_sample_size) if IMIT_TASKS else [], # 新增 (NEW)

        # --- 答案存储 (Answer Storage) ---
        "open_ended_answers": {},
        "constrained_answers": {},
        "imit_answers": {}, # 新增 (NEW)

        # --- 时间戳存储 (Timestamp Storage) ---
        "open_ended_timestamps": {},
        "constrained_timestamps": {},
        "imit_timestamps": {},

        # --- 按作答顺序存储的时间戳列表 (Timestamp Sequence in Answer Order) ---
        "answer_sequence_timestamps": [],  # 格式: [{"label": "round1_open_ended_1", "timestamp": "2025-10-20 05:38:15"}, ...]

        # --- 当前题目索引 (Current Question Indices) ---
        "current_open_ended_index": 0,
        "current_constrained_index": 0,
        "current_imit_index": 0, # 新增 (NEW)

        # --- 当前测试阶段 (Current Exam Phase) ---
        "exam_phase": "intro", # e.g., 'intro', 'open_ended', 'constrained', 'imit', 'results', 'final_complete'

        # --- 新增：轮次和用户信息 (NEW: Round and User Info) ---
        "assessment_round": 1,  # 默认为第一轮
        "user_id": None,  # 用户邮箱
        "round1_results": None,  # 第一轮结果存储
        "final_report": None,  # 最终报告

    }