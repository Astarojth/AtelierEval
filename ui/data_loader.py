# data_loader.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import config

# 为所有题型定义全局变量
# Global variables for tasks
OPEN_ENDED_TASKS = []
CONSTRAINED_TASKS = []
IMIT_TASKS = [] # 新增：为模仿题增加全局变量 (New: Add global variable for imitation tasks)


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Local dataset file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list JSON in {path}, got {type(payload).__name__}")
    out: List[Dict[str, Any]] = []
    for item in payload:
        if isinstance(item, dict):
            out.append(item)
    return out


def _normalize_task_id(task: Dict[str, Any], prefix: str, idx: int) -> str:
    raw = task.get("task_id") or task.get("id")
    text = str(raw).strip() if raw is not None else ""
    return text or f"{prefix}_{idx:03d}"


def _load_local_open_ended() -> List[Dict[str, Any]]:
    path = config.LOCAL_DATASET_ROOT / config.LOCAL_DATA_FILES["open_ended"]
    raw = _load_json_list(path)
    tasks: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw, start=1):
        task_id = _normalize_task_id(item, "oe", idx)
        mapped = dict(item)
        mapped["task_id"] = task_id
        mapped["id"] = task_id
        mapped["task_type"] = "open_ended"
        mapped["description"] = (
            str(item.get("description", "")).strip()
            or str(item.get("task", "")).strip()
            or str(item.get("title", "")).strip()
        )
        mapped["ground_truth_prompt"] = (
            str(item.get("ground_truth_prompt", "")).strip()
            or str(item.get("gt_prompt", "")).strip()
            or str(item.get("prompt", "")).strip()
        )
        tasks.append(mapped)
    return tasks


def _load_local_constrained() -> List[Dict[str, Any]]:
    path = config.LOCAL_DATASET_ROOT / config.LOCAL_DATA_FILES["constrained"]
    raw = _load_json_list(path)
    tasks: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw, start=1):
        task_id = _normalize_task_id(item, "co", idx)
        mapped = dict(item)
        mapped["task_id"] = task_id
        mapped["id"] = task_id
        mapped["task_type"] = "constrained"
        # Keep a unified reference prompt key for report rendering compatibility.
        mapped["prompt"] = str(item.get("prompt", "")).strip()
        tasks.append(mapped)
    return tasks


def _load_local_imit() -> List[Dict[str, Any]]:
    path = config.LOCAL_DATASET_ROOT / config.LOCAL_DATA_FILES["imit"]
    raw = _load_json_list(path)
    tasks: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw, start=1):
        task_id = _normalize_task_id(item, "im", idx)
        short_id = task_id.split("_", 1)[-1] if "_" in task_id else task_id

        image_rel = str(item.get("origin_image", "")).strip()
        image_path = (config.LOCAL_DATASET_ROOT / image_rel).resolve() if image_rel else None

        mapped = dict(item)
        mapped["task_id"] = task_id
        mapped["id"] = short_id
        mapped["task_type"] = "imit"
        mapped["prompt"] = (
            str(item.get("prompt", "")).strip()
            or str(item.get("gt_prompt", "")).strip()
        )
        mapped["origin_image_path"] = str(image_path) if image_path else ""
        # Local dataset has one reference image field; reuse it for report target slot.
        mapped["flux_image_path"] = str(image_path) if image_path else ""
        tasks.append(mapped)
    return tasks


def _load_tasks_from_local() -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    oe = _load_local_open_ended()
    co = _load_local_constrained()
    im = _load_local_imit()
    return oe, co, im


def load_tasks():
    """
    加载所有任务，并根据类型进行筛选。
    Load all tasks and filter by type.
    """
    # 使用 global 关键字来修改全局变量
    # Use the global keyword to modify the global variables
    global OPEN_ENDED_TASKS, CONSTRAINED_TASKS, IMIT_TASKS
    
    print("Starting application, loading tasks (source=dataset-only)...")
    print(f"Local dataset root: {config.LOCAL_DATASET_ROOT}")

    OPEN_ENDED_TASKS, CONSTRAINED_TASKS, IMIT_TASKS = _load_tasks_from_local()
    
    # 更新成功加载的日志信息
    # Update the success log message
    if OPEN_ENDED_TASKS and CONSTRAINED_TASKS and IMIT_TASKS:
        print(f"✅ Successfully loaded {len(OPEN_ENDED_TASKS)} Open-Ended, "
              f"{len(CONSTRAINED_TASKS)} Constrained, and "
              f"{len(IMIT_TASKS)} Imitation tasks.")
    else:
        # 更新警告信息，帮助调试
        # Update the warning message to help with debugging
        print(
            "⚠️ WARNING: Failed to load one or more task types. "
            "Check files under repository dataset directory."
        )
    
    # 在返回值中包含模仿题列表
    # Include the imitation tasks list in the return value
    return OPEN_ENDED_TASKS, CONSTRAINED_TASKS, IMIT_TASKS

