from __future__ import annotations

import importlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from configs.config_loader import DEFAULT_SKILL_CONFIG_FILES, load_skill_registry_from_config_dir as _load_skill_registry

@dataclass(frozen=True)
class TaskRoute:
    task_type: str
    run_prompt_subjective: bool
    run_image_subjective: bool
    run_prompt_objective: bool
    run_image_objective: bool


ROUTES: Dict[str, TaskRoute] = {
    "co": TaskRoute(
        task_type="co",
        run_prompt_subjective=True,
        run_image_subjective=True,
        run_prompt_objective=True,
        run_image_objective=True,
    ),
    "oe": TaskRoute(
        task_type="oe",
        run_prompt_subjective=True,
        run_image_subjective=True,
        run_prompt_objective=True,
        run_image_objective=True,
    ),
    "im": TaskRoute(
        task_type="im",
        run_prompt_subjective=True,
        run_image_subjective=False,
        run_prompt_objective=True,
        run_image_objective=True,
    ),
}

DEFAULT_SKILL_REGISTRY: Dict[str, Dict[str, str]] = {
    "SafetyPromptSkill": {
        "module": "safety.skills.safety_prompt_skill",
        "class": "SafetyPromptSkill",
    },
    "SafetyImageSkill": {
        "module": "safety.skills.safety_image_skill",
        "class": "SafetyImageSkill",
    },
    "S1PromptSubjectiveSkill": {
        "module": "subjective.skills.s1_prompt_subjective_skill",
        "class": "S1PromptSubjectiveSkill",
    },
    "S1ImageSubjectiveSkill": {
        "module": "subjective.skills.s1_image_subjective_skill",
        "class": "S1ImageSubjectiveSkill",
    },
    "S2PromptObjectiveSkill": {
        "module": "objective.skills.s2_prompt_objective_skill",
        "class": "S2PromptObjectiveSkill",
    },
    "S2ImageObjectiveSkill": {
        "module": "objective.skills.s2_image_objective_skill",
        "class": "S2ImageObjectiveSkill",
    },
}

def get_route(task_type: str) -> TaskRoute:
    key = task_type.strip().lower()
    if key not in ROUTES:
        raise ValueError(f"Unsupported task_type: {task_type}")
    return ROUTES[key]


def load_skill_registry_from_config_dir(
    config_dir: str | Path,
    config_files: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, str]]:
    return _load_skill_registry(config_dir=config_dir, config_files=config_files)


def load_skill_instances(skill_registry: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    registry = skill_registry or DEFAULT_SKILL_REGISTRY
    instances: Dict[str, Any] = {}
    for skill_name, cfg in registry.items():
        module_name = str(cfg.get("module", "")).strip()
        class_name = str(cfg.get("class", "")).strip()
        if not module_name or not class_name:
            raise ValueError(f"Invalid skill config for {skill_name}: missing module/class")
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        instances[skill_name] = cls()
    return instances


def _ensure_skill_instances(skill_instances: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if skill_instances is not None:
        return skill_instances
    return load_skill_instances(DEFAULT_SKILL_REGISTRY)


def _trace(
    *,
    skill_name: str,
    stage: str,
    status: str,
    duration_seconds: float,
    input_refs: Dict[str, Any],
    error: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "skill_name": skill_name,
        "stage": stage,
        "status": status,
        "duration_seconds": round(float(duration_seconds), 4),
        "input_refs": input_refs,
    }
    if error:
        payload["error"] = error
    return payload


def _run_skill(
    *,
    skill_instances: Dict[str, Any],
    skill_name: str,
    stage: str,
    input_refs: Dict[str, Any],
    kwargs: Dict[str, Any],
) -> Tuple[Any, Dict[str, Any]]:
    if skill_name not in skill_instances:
        raise KeyError(f"Skill instance not found: {skill_name}")

    start = perf_counter()
    try:
        output = skill_instances[skill_name].run(**kwargs)
        duration = perf_counter() - start
        return output, _trace(
            skill_name=skill_name,
            stage=stage,
            status="success",
            duration_seconds=duration,
            input_refs=input_refs,
        )
    except Exception as exc:  # noqa: BLE001
        duration = perf_counter() - start
        setattr(exc, "skill_trace", _trace(
            skill_name=skill_name,
            stage=stage,
            status="failed",
            duration_seconds=duration,
            input_refs=input_refs,
            error=f"{type(exc).__name__}: {exc}",
        ))
        raise


def _skip_trace(skill_name: str, stage: str, input_refs: Dict[str, Any], reason: str) -> Dict[str, Any]:
    return _trace(
        skill_name=skill_name,
        stage=stage,
        status="skipped",
        duration_seconds=0.0,
        input_refs={**input_refs, "reason": reason},
    )


def run_safety_skills(
    *,
    cfg: Any,
    sample: Any,
    safety_runtime: Any,
    run_prompt_safety_fn: Callable[..., Dict[str, Any]],
    run_image_safety_fn: Callable[..., Dict[str, Any]],
    skill_instances: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    skill_instances = _ensure_skill_instances(skill_instances)
    prompt_result, prompt_trace = _run_skill(
        skill_instances=skill_instances,
        skill_name="SafetyPromptSkill",
        stage="Safety",
        input_refs={"task_id": sample.task_id, "modality": "prompt"},
        kwargs={
            "cfg": cfg,
            "sample": sample,
            "safety_runtime": safety_runtime,
            "run_prompt_safety_fn": run_prompt_safety_fn,
        },
    )
    image_result, image_trace = _run_skill(
        skill_instances=skill_instances,
        skill_name="SafetyImageSkill",
        stage="Safety",
        input_refs={"task_id": sample.task_id, "modality": "image"},
        kwargs={
            "cfg": cfg,
            "sample": sample,
            "safety_runtime": safety_runtime,
            "run_image_safety_fn": run_image_safety_fn,
        },
    )

    traces = [prompt_trace, image_trace]

    if not cfg.run_safety:
        return {"enabled": False, "passed": True, "skip_reason": "disabled_by_flag"}, traces

    if prompt_result.get("skip_reason") or image_result.get("skip_reason"):
        reason = prompt_result.get("skip_reason") or image_result.get("skip_reason")
        return {"enabled": True, "passed": True, "skip_reason": reason}, traces

    prompt_passed = bool(prompt_result.get("passed"))
    image_passed = bool(image_result.get("passed"))
    decision_mode = str(getattr(cfg, "safety_decision_mode", "any_fail")).strip().lower()
    if decision_mode not in {"both_fail", "any_fail"}:
        decision_mode = "any_fail"

    if decision_mode == "both_fail":
        overall_passed = prompt_passed or image_passed
    else:
        overall_passed = prompt_passed and image_passed

    return {
        "enabled": True,
        "passed": overall_passed,
        "decision_mode": decision_mode,
        "prompt_safety": {
            "passed": prompt_passed,
            "reason": prompt_result.get("reason"),
            "policy_tags": prompt_result.get("policy_tags", []),
        },
        "image_safety": {
            "passed": image_passed,
            "reason": image_result.get("reason"),
        },
    }, traces


def run_system1_subjective_skills(
    *,
    route: TaskRoute,
    task_id: str,
    task_description: str,
    prompt_text: str,
    image_path: str,
    sub_mem: Any,
    sub_core: Any,
    memories: Any,
    prompt_clusterer: Any,
    image_clusterer: Any,
    top_k: int,
    skill_instances: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    skill_instances = _ensure_skill_instances(skill_instances)
    prompt_dims: Optional[Dict[str, Any]] = None
    image_dims: Optional[Dict[str, Any]] = None
    traces: List[Dict[str, Any]] = []

    if route.run_prompt_subjective and route.run_image_subjective:
        with ThreadPoolExecutor(max_workers=2) as executor:
            prompt_future = executor.submit(
                _run_skill,
                skill_instances=skill_instances,
                skill_name="S1PromptSubjectiveSkill",
                stage="System1",
                input_refs={"task_id": task_id, "task_type": route.task_type, "modality": "prompt"},
                kwargs={
                    "route": route,
                    "task_id": task_id,
                    "task_description": task_description,
                    "prompt_text": prompt_text,
                    "sub_mem": sub_mem,
                    "sub_core": sub_core,
                    "memories": memories,
                    "prompt_clusterer": prompt_clusterer,
                    "top_k": top_k,
                },
            )
            image_future = executor.submit(
                _run_skill,
                skill_instances=skill_instances,
                skill_name="S1ImageSubjectiveSkill",
                stage="System1",
                input_refs={"task_id": task_id, "task_type": route.task_type, "modality": "image"},
                kwargs={
                    "route": route,
                    "task_id": task_id,
                    "task_description": task_description,
                    "prompt_text": prompt_text,
                    "image_path": image_path,
                    "sub_mem": sub_mem,
                    "sub_core": sub_core,
                    "memories": memories,
                    "image_clusterer": image_clusterer,
                    "top_k": top_k,
                },
            )
            prompt_dims, prompt_trace = prompt_future.result()
            image_dims, image_trace = image_future.result()
            traces.extend([prompt_trace, image_trace])
    elif route.run_prompt_subjective:
        prompt_dims, prompt_trace = _run_skill(
            skill_instances=skill_instances,
            skill_name="S1PromptSubjectiveSkill",
            stage="System1",
            input_refs={"task_id": task_id, "task_type": route.task_type, "modality": "prompt"},
            kwargs={
                "route": route,
                "task_id": task_id,
                "task_description": task_description,
                "prompt_text": prompt_text,
                "sub_mem": sub_mem,
                "sub_core": sub_core,
                "memories": memories,
                "prompt_clusterer": prompt_clusterer,
                "top_k": top_k,
            },
        )
        traces.append(prompt_trace)
        traces.append(_skip_trace("S1ImageSubjectiveSkill", "System1", {"task_id": task_id, "task_type": route.task_type, "modality": "image"}, "route_disabled"))
    elif route.run_image_subjective:
        image_dims, image_trace = _run_skill(
            skill_instances=skill_instances,
            skill_name="S1ImageSubjectiveSkill",
            stage="System1",
            input_refs={"task_id": task_id, "task_type": route.task_type, "modality": "image"},
            kwargs={
                "route": route,
                "task_id": task_id,
                "task_description": task_description,
                "prompt_text": prompt_text,
                "image_path": image_path,
                "sub_mem": sub_mem,
                "sub_core": sub_core,
                "memories": memories,
                "image_clusterer": image_clusterer,
                "top_k": top_k,
            },
        )
        traces.append(image_trace)
        traces.append(_skip_trace("S1PromptSubjectiveSkill", "System1", {"task_id": task_id, "task_type": route.task_type, "modality": "prompt"}, "route_disabled"))

    return {
        "prompt_dimensions": prompt_dims,
        "image_dimensions": image_dims,
    }, traces


def run_system2_objective_skills(
    *,
    route: TaskRoute,
    run_prompt_objective_fn: Callable[[], Dict[str, Any]],
    run_image_objective_fn: Callable[[], Dict[str, Any]],
    skill_instances: Optional[Dict[str, Any]] = None,
    task_id: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    skill_instances = _ensure_skill_instances(skill_instances)
    traces: List[Dict[str, Any]] = []
    prompt_payload: Dict[str, Any] = {
        "prompt_checks": None,
        "prompt_constraint_satisfaction": None,
        "raw_result_json": None,
    }
    image_payload: Dict[str, Any] = {
        "image_checks": None,
        "image_constraint_satisfaction": None,
        "raw_result_json": None,
    }

    if route.run_prompt_objective and route.run_image_objective:
        with ThreadPoolExecutor(max_workers=2) as executor:
            prompt_future = executor.submit(
                _run_skill,
                skill_instances=skill_instances,
                skill_name="S2PromptObjectiveSkill",
                stage="System2",
                input_refs={"task_id": task_id, "task_type": route.task_type, "modality": "prompt"},
                kwargs={"run_prompt_objective_fn": run_prompt_objective_fn},
            )
            image_future = executor.submit(
                _run_skill,
                skill_instances=skill_instances,
                skill_name="S2ImageObjectiveSkill",
                stage="System2",
                input_refs={"task_id": task_id, "task_type": route.task_type, "modality": "image"},
                kwargs={"run_image_objective_fn": run_image_objective_fn},
            )
            prompt_payload, prompt_trace = prompt_future.result()
            image_payload, image_trace = image_future.result()
            traces.extend([prompt_trace, image_trace])
    elif route.run_prompt_objective:
        prompt_payload, prompt_trace = _run_skill(
            skill_instances=skill_instances,
            skill_name="S2PromptObjectiveSkill",
            stage="System2",
            input_refs={"task_id": task_id, "task_type": route.task_type, "modality": "prompt"},
            kwargs={"run_prompt_objective_fn": run_prompt_objective_fn},
        )
        traces.append(prompt_trace)
        traces.append(_skip_trace("S2ImageObjectiveSkill", "System2", {"task_id": task_id, "task_type": route.task_type, "modality": "image"}, "route_disabled"))
    elif route.run_image_objective:
        image_payload, image_trace = _run_skill(
            skill_instances=skill_instances,
            skill_name="S2ImageObjectiveSkill",
            stage="System2",
            input_refs={"task_id": task_id, "task_type": route.task_type, "modality": "image"},
            kwargs={"run_image_objective_fn": run_image_objective_fn},
        )
        traces.append(image_trace)
        traces.append(_skip_trace("S2PromptObjectiveSkill", "System2", {"task_id": task_id, "task_type": route.task_type, "modality": "prompt"}, "route_disabled"))

    raw_result_json = prompt_payload.get("raw_result_json") or image_payload.get("raw_result_json")
    return {
        "prompt_checks": prompt_payload.get("prompt_checks"),
        "image_checks": image_payload.get("image_checks"),
        "prompt_constraint_satisfaction": prompt_payload.get("prompt_constraint_satisfaction"),
        "image_constraint_satisfaction": image_payload.get("image_constraint_satisfaction"),
        "raw_result_json": raw_result_json,
    }, traces


def run_subjective_route(**kwargs: Any) -> Dict[str, Any]:
    """Backward-compatible wrapper returning only subjective payload."""
    payload, _ = run_system1_subjective_skills(**kwargs)
    return payload
