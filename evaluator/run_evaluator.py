from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import logging
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

from router import (
    get_route,
    load_skill_instances,
    load_skill_registry_from_config_dir,
    run_safety_skills,
    run_system1_subjective_skills,
    run_system2_objective_skills,
)
from configs.api_runtime import build_openai_base_url, get_model_candidates


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [Agent:AtelierJudgeAgent] | %(levelname)s | %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("atelier_evaluator")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
TASK_TYPES = ("co", "im", "oe")
HEARTBEAT_SECONDS = 20
AGENT_NAME = "AtelierJudgeAgent"


def _progress(stage: str, event: str, message: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}][Agent:{AGENT_NAME}][Stage:{stage}][Event:{event}] {message}", flush=True)


def _run_with_heartbeat(
    skill_label: str,
    fn: Any,
    *args: Any,
    heartbeat_seconds: int = HEARTBEAT_SECONDS,
    **kwargs: Any,
) -> Any:
    _progress("SkillExec", "start", f"{skill_label} started")
    start = perf_counter()
    stop_event = threading.Event()

    def _heartbeat() -> None:
        while not stop_event.wait(heartbeat_seconds):
            elapsed = perf_counter() - start
            _progress("SkillExec", "heartbeat", f"{skill_label} running ({elapsed:.1f}s)")

    beat_thread = threading.Thread(target=_heartbeat, daemon=True)
    beat_thread.start()
    try:
        result = fn(*args, **kwargs)
        elapsed = perf_counter() - start
        _progress("SkillExec", "finish", f"{skill_label} finished ({elapsed:.1f}s)")
        return result
    finally:
        stop_event.set()
        beat_thread.join(timeout=0.1)


@dataclass(frozen=True)
class SampleCase:
    task_type: str
    task_id: str
    model_llm: str
    image_model_llm: str
    model_t2i: str
    llm_json_path: Path
    prompt_text: str
    original_task: str
    image_path: Path


@dataclass(frozen=True)
class EvaluatorConfig:
    workspace_root: Path
    submit_root: Path
    evaluator_root: Path
    generated_root: Path
    subjective_dir: Path
    objective_dir: Path
    safety_dir: Path
    configs_dir: Path
    objective_config_path: Path
    llm_safety_config_path: Path
    t2i_safety_config_path: Path
    objective_output_subdir: str
    preferred_model_llm: str
    preferred_model_t2i: str
    preferred_task_ids: Dict[str, str]
    top_k: int
    run_safety: bool
    safety_decision_mode: str


@dataclass(frozen=True)
class SafetyRuntime:
    enabled: bool
    skip_reason: Optional[str]
    llm_safety: Any = None
    t2i_safety: Any = None
    llm_client: Any = None
    t2i_client: Any = None
    llm_cfg: Optional[Dict[str, Any]] = None
    t2i_cfg: Optional[Dict[str, Any]] = None
    llm_template: Optional[str] = None
    t2i_template: Optional[str] = None


class _AgentPrefixedStdout(io.TextIOBase):
    def __init__(self, stage: str, event: str, skill_label: str) -> None:
        self.stage = stage
        self.event = event
        self.skill_label = skill_label
        self._buf = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                ts = datetime.now().strftime("%H:%M:%S")
                sys.__stdout__.write(
                    f"[{ts}][Agent:{AGENT_NAME}][Stage:{self.stage}][Event:{self.event}] "
                    f"[{self.skill_label}] {line}\n"
                )
            else:
                sys.__stdout__.write("\n")
        return len(s)

    def flush(self) -> None:
        if self._buf.strip():
            ts = datetime.now().strftime("%H:%M:%S")
            sys.__stdout__.write(
                f"[{ts}][Agent:{AGENT_NAME}][Stage:{self.stage}][Event:{self.event}] "
                f"[{self.skill_label}] {self._buf}\n"
            )
        self._buf = ""
        sys.__stdout__.flush()


def _with_generated_root(cfg: EvaluatorConfig, generated_root: Path) -> EvaluatorConfig:
    return EvaluatorConfig(
        workspace_root=cfg.workspace_root,
        submit_root=cfg.submit_root,
        evaluator_root=cfg.evaluator_root,
        generated_root=generated_root,
        subjective_dir=cfg.subjective_dir,
        objective_dir=cfg.objective_dir,
        safety_dir=cfg.safety_dir,
        configs_dir=cfg.configs_dir,
        objective_config_path=cfg.objective_config_path,
        llm_safety_config_path=cfg.llm_safety_config_path,
        t2i_safety_config_path=cfg.t2i_safety_config_path,
        objective_output_subdir=cfg.objective_output_subdir,
        preferred_model_llm=cfg.preferred_model_llm,
        preferred_model_t2i=cfg.preferred_model_t2i,
        preferred_task_ids=cfg.preferred_task_ids,
        top_k=cfg.top_k,
        run_safety=cfg.run_safety,
        safety_decision_mode=cfg.safety_decision_mode,
    )


def _find_workspace_root(start: Path) -> Path:
    for cand in [start, *start.parents]:
        if (cand / "dataset").is_dir() and (cand / "generator").is_dir() and (cand / "evaluator").is_dir():
            return cand
    raise RuntimeError(f"Cannot infer workspace root from: {start}")


def build_default_config(*, run_safety: bool, top_k: int) -> EvaluatorConfig:
    current = Path(__file__).resolve()
    workspace_root = _find_workspace_root(current)
    submit_root = workspace_root
    evaluator_root = workspace_root / "evaluator"
    generated_root = workspace_root / "_placeholder_generated_results"
    return EvaluatorConfig(
        workspace_root=workspace_root,
        submit_root=submit_root,
        evaluator_root=evaluator_root,
        generated_root=generated_root,
        subjective_dir=evaluator_root / "subjective",
        objective_dir=evaluator_root / "objective",
        safety_dir=evaluator_root / "safety",
        configs_dir=evaluator_root / "configs",
        objective_config_path=evaluator_root / "configs" / "runtime" / "objective_runtime.json",
        llm_safety_config_path=evaluator_root / "configs" / "runtime" / "safety_prompt_runtime.json",
        t2i_safety_config_path=evaluator_root / "configs" / "runtime" / "safety_image_runtime.json",
        objective_output_subdir="3_check",
        preferred_model_llm="gemini_2.0_flash",
        preferred_model_t2i="sdxl_local",
        preferred_task_ids={"co": "co_01", "im": "im_01", "oe": "oe_01"},
        top_k=top_k,
        run_safety=run_safety,
        safety_decision_mode="any_fail",
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_prompt_files(generated_root: Path, task_type: str, preferred_model_llm: str) -> Iterable[Tuple[str, Path]]:
    llm_dir = generated_root / "1_llm" / f"{task_type}_output_llm"
    preferred = llm_dir / f"{task_type}_prompts_{preferred_model_llm}.json"
    if preferred.exists():
        yield preferred_model_llm, preferred
    for path in sorted(llm_dir.glob(f"{task_type}_prompts_*.json")):
        model = path.stem.replace(f"{task_type}_prompts_", "", 1)
        if model == preferred_model_llm and preferred.exists():
            continue
        yield model, path


def _normalize_model_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _resolve_prompt_file_for_model(generated_root: Path, task_type: str, model_llm: str) -> Optional[Path]:
    llm_dir = generated_root / "1_llm" / f"{task_type}_output_llm"
    expected = llm_dir / f"{task_type}_prompts_{model_llm}.json"
    if expected.exists():
        return expected
    if not llm_dir.exists():
        return None

    target = _normalize_model_name(model_llm)
    for path in sorted(llm_dir.glob(f"{task_type}_prompts_*.json")):
        suffix = path.stem.replace(f"{task_type}_prompts_", "", 1)
        if _normalize_model_name(suffix) == target:
            return path
    return None


def discover_cases_for_task_type(cfg: EvaluatorConfig, task_type: str) -> List[SampleCase]:
    tti_root = cfg.generated_root / "2_tti" / f"{task_type}_output_tti"
    if not tti_root.exists():
        logger.warning("[Agent:%s][Stage:Routing] Missing TTI directory for task_type=%s: %s", AGENT_NAME, task_type, tti_root)
        return []

    prompt_payload_cache: Dict[Path, Dict[str, Any]] = {}
    cases: List[SampleCase] = []

    for model_t2i_dir in sorted([p for p in tti_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        model_t2i = model_t2i_dir.name
        for model_llm_dir in sorted([p for p in model_t2i_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
            model_llm = model_llm_dir.name
            llm_file = _resolve_prompt_file_for_model(cfg.generated_root, task_type, model_llm)
            if llm_file is None:
                logger.warning(
                    "[Agent:%s][Stage:Routing] Prompt file not found for task_type=%s, model_llm=%s",
                    AGENT_NAME,
                    task_type,
                    model_llm,
                )
                continue

            if llm_file not in prompt_payload_cache:
                payload = _load_json(llm_file)
                if not isinstance(payload, dict):
                    logger.warning(
                        "[Agent:%s][Stage:Routing] Prompt file is not object JSON: %s",
                        AGENT_NAME,
                        llm_file,
                    )
                    prompt_payload_cache[llm_file] = {}
                else:
                    prompt_payload_cache[llm_file] = payload
            prompt_payload = prompt_payload_cache[llm_file]

            for task_dir in sorted([p for p in model_llm_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
                task_id = task_dir.name
                image_candidates = [
                    p
                    for p in sorted(task_dir.iterdir(), key=lambda p: p.name)
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                ]
                if not image_candidates:
                    continue

                item = prompt_payload.get(task_id)
                if not isinstance(item, dict):
                    logger.warning(
                        "[Agent:%s][Stage:Routing] Task %s missing in prompt source %s",
                        AGENT_NAME,
                        task_id,
                        llm_file,
                    )
                    continue
                prompt_text = str(item.get("generated_prompt", "")).strip()
                if not prompt_text:
                    logger.warning(
                        "[Agent:%s][Stage:Routing] Empty generated_prompt for task=%s in %s",
                        AGENT_NAME,
                        task_id,
                        llm_file,
                    )
                    continue

                cases.append(
                    SampleCase(
                        task_type=task_type,
                        task_id=task_id,
                        model_llm=model_llm,
                        image_model_llm=model_llm,
                        model_t2i=model_t2i,
                        llm_json_path=llm_file,
                        prompt_text=prompt_text,
                        original_task=str(item.get("original_task", "")).strip(),
                        image_path=image_candidates[0],
                    )
                )
    return cases


def _parse_image_metadata(tti_root: Path, image_path: Path) -> Optional[Tuple[str, str, str]]:
    try:
        rel = image_path.relative_to(tti_root)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 4:
        return None
    task_id = parts[-2]
    model_llm = parts[-3]
    model_t2i = "/".join(parts[:-3])
    return task_id, model_llm, model_t2i


def _find_image_for_task(
    *,
    generated_root: Path,
    task_type: str,
    task_id: str,
    model_llm: str,
    preferred_model_t2i: str,
    require_same_model_llm: bool = True,
) -> Optional[Tuple[Path, str, str]]:
    tti_root = generated_root / "2_tti" / f"{task_type}_output_tti"
    preferred_dir = tti_root / preferred_model_t2i / model_llm / task_id
    if require_same_model_llm and preferred_dir.exists():
        preferred_images = [
            p
            for p in sorted(preferred_dir.glob(f"{task_id}_*.*"))
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if preferred_images:
            return preferred_images[0], preferred_model_t2i, model_llm

    for image_path in sorted(tti_root.rglob(f"{task_id}_*.*")):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        meta = _parse_image_metadata(tti_root=tti_root, image_path=image_path)
        if meta is None:
            continue
        meta_task_id, meta_model_llm, meta_model_t2i = meta
        if meta_task_id != task_id:
            continue
        if require_same_model_llm and meta_model_llm != model_llm:
            continue
        return image_path, meta_model_t2i, meta_model_llm
    return None


def choose_sample_case(cfg: EvaluatorConfig, task_type: str) -> SampleCase:
    preferred_task_id = cfg.preferred_task_ids.get(task_type)
    for model_llm, llm_file in _iter_prompt_files(
        generated_root=cfg.generated_root,
        task_type=task_type,
        preferred_model_llm=cfg.preferred_model_llm,
    ):
        payload = _load_json(llm_file)
        if not isinstance(payload, dict):
            continue

        candidate_task_ids: List[str] = []
        if preferred_task_id:
            candidate_task_ids.append(preferred_task_id)
        candidate_task_ids.extend([k for k in sorted(payload.keys()) if k not in candidate_task_ids])

        for task_id in candidate_task_ids:
            item = payload.get(task_id)
            if not isinstance(item, dict):
                continue
            prompt_text = str(item.get("generated_prompt", "")).strip()
            if not prompt_text:
                continue
            image_meta = _find_image_for_task(
                generated_root=cfg.generated_root,
                task_type=task_type,
                task_id=task_id,
                model_llm=model_llm,
                preferred_model_t2i=cfg.preferred_model_t2i,
                require_same_model_llm=True,
            )
            if image_meta is None:
                image_meta = _find_image_for_task(
                    generated_root=cfg.generated_root,
                    task_type=task_type,
                    task_id=task_id,
                    model_llm=model_llm,
                    preferred_model_t2i=cfg.preferred_model_t2i,
                    require_same_model_llm=False,
                )
            if image_meta is None:
                continue
            image_path, model_t2i, image_model_llm = image_meta
            if image_model_llm != model_llm:
                logger.warning(
                    "[Agent:%s][Stage:Routing][Skill:SampleSelector] Fallback to cross-LLM image for %s: prompt_llm=%s, image_llm=%s, image=%s",
                    AGENT_NAME,
                    task_id,
                    model_llm,
                    image_model_llm,
                    image_path,
                )
            return SampleCase(
                task_type=task_type,
                task_id=task_id,
                model_llm=model_llm,
                image_model_llm=image_model_llm,
                model_t2i=model_t2i,
                llm_json_path=llm_file,
                prompt_text=prompt_text,
                original_task=str(item.get("original_task", "")).strip(),
                image_path=image_path,
            )
    raise RuntimeError(f"Cannot find usable sample for task_type={task_type}")


def _calc_constraint_rate(checks: Any) -> Optional[float]:
    if not isinstance(checks, dict) or not checks:
        return None
    vals = [int(v) for v in checks.values() if v in (0, 1)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _prepare_subjective_modules(cfg: EvaluatorConfig):
    if str(cfg.subjective_dir) not in sys.path:
        sys.path.insert(0, str(cfg.subjective_dir))
    if str(cfg.evaluator_root) not in sys.path:
        sys.path.insert(0, str(cfg.evaluator_root))
    import subjective.s1_subjective_memory_skill_engine as sub_mem  # type: ignore
    import subjective.subjective_eval_core as sub_core  # type: ignore
    return sub_mem, sub_core


def _prepare_objective_modules(cfg: EvaluatorConfig):
    if str(cfg.evaluator_root) not in sys.path:
        sys.path.insert(0, str(cfg.evaluator_root))
    import objective.skills.s2_prompt_objective_skill as obj_checker_mod  # type: ignore
    import objective.obj_checker_utils as obj_utils  # type: ignore
    return obj_checker_mod, obj_utils


def _prepare_safety_modules(cfg: EvaluatorConfig):
    if str(cfg.evaluator_root) not in sys.path:
        sys.path.insert(0, str(cfg.evaluator_root))
    import safety.skills.safety_prompt_skill as llm_safety  # type: ignore
    import safety.skills.safety_image_skill as t2i_safety  # type: ignore
    from openai import OpenAI  # type: ignore

    return llm_safety, t2i_safety, OpenAI


def prepare_safety_runtime(cfg: EvaluatorConfig) -> SafetyRuntime:
    if not cfg.run_safety:
        return SafetyRuntime(enabled=False, skip_reason="disabled_by_flag")

    llm_safety, t2i_safety, OpenAI = _prepare_safety_modules(cfg)
    llm_cfg = llm_safety.load_config(cfg.llm_safety_config_path)
    t2i_cfg = t2i_safety.load_config(cfg.t2i_safety_config_path)

    llm_api_cfg = llm_cfg.get("api", {})
    t2i_api_cfg = t2i_cfg.get("api", {})
    llm_runtime = llm_cfg.get("runtime", {})
    t2i_runtime = t2i_cfg.get("runtime", {})

    llm_token = llm_api_cfg.get("api_key") or os.getenv(str(llm_api_cfg.get("api_key_env", "ATELIEREVAL_API_KEY")))
    t2i_token = t2i_api_cfg.get("api_key") or os.getenv(str(t2i_api_cfg.get("api_key_env", "ATELIEREVAL_API_KEY")))
    if not llm_token or not t2i_token:
        return SafetyRuntime(enabled=True, skip_reason="missing_safety_api_key")

    llm_template = llm_safety.load_prompt_template(
        llm_safety.resolve_config_path(cfg.llm_safety_config_path, str(llm_cfg["paths"]["prompt_template_path"]))
    )
    t2i_template = t2i_safety.load_prompt_template(
        t2i_safety.resolve_config_path(cfg.t2i_safety_config_path, str(t2i_cfg["paths"]["prompt_template_path"]))
    )

    llm_client = OpenAI(
        base_url=build_openai_base_url(llm_api_cfg),
        api_key=llm_token,
        timeout=int(llm_runtime.get("timeout_seconds", 120)),
    )
    t2i_client = OpenAI(
        base_url=build_openai_base_url(t2i_api_cfg),
        api_key=t2i_token,
        timeout=int(t2i_runtime.get("timeout_seconds", 120)),
    )

    return SafetyRuntime(
        enabled=True,
        skip_reason=None,
        llm_safety=llm_safety,
        t2i_safety=t2i_safety,
        llm_client=llm_client,
        t2i_client=t2i_client,
        llm_cfg=llm_cfg,
        t2i_cfg=t2i_cfg,
        llm_template=llm_template,
        t2i_template=t2i_template,
    )


def run_prompt_safety_for_case(*, cfg: EvaluatorConfig, sample: SampleCase, safety_runtime: SafetyRuntime) -> Dict[str, Any]:
    if not cfg.run_safety:
        return {"enabled": False, "passed": True, "skip_reason": "disabled_by_flag"}
    if safety_runtime.skip_reason:
        return {"enabled": True, "passed": True, "skip_reason": safety_runtime.skip_reason}
    if not safety_runtime.llm_cfg or safety_runtime.llm_safety is None:
        return {"enabled": True, "passed": True, "skip_reason": "invalid_safety_runtime"}

    llm_cfg = safety_runtime.llm_cfg
    llm_api_cfg = llm_cfg.get("api", {})
    llm_runtime = llm_cfg.get("runtime", {})
    llm_req_cfg = llm_cfg.get("llm_request", {})

    last_error: Optional[Exception] = None
    for model_name in get_model_candidates(llm_api_cfg):
        try:
            llm_passed, llm_reason, llm_tags = safety_runtime.llm_safety.call_text_safety(
                client=safety_runtime.llm_client,
                model=model_name,
                task_id=sample.task_id,
                prompt_text=sample.prompt_text,
                prompt_template=safety_runtime.llm_template,
                temperature=float(llm_req_cfg.get("temperature", 0.0)),
                top_p=(float(llm_req_cfg["top_p"]) if llm_req_cfg.get("top_p") is not None else None),
                max_tokens=int(llm_req_cfg.get("max_tokens", 4096)),
                max_retries=int(llm_runtime.get("max_retries", 8)),
                backoff_base=float(llm_runtime.get("backoff_base", 1.5)),
                backoff_cap=float(llm_runtime.get("backoff_cap", 60.0)),
                limiter=safety_runtime.llm_safety.RateLimiter(float(llm_runtime.get("qps", 0.5))),
            )
            return {"enabled": True, "passed": bool(llm_passed), "reason": llm_reason, "policy_tags": llm_tags}
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "[Agent:%s][Stage:Safety] Prompt safety model failed for %s with model=%s: %s",
                AGENT_NAME,
                sample.task_id,
                model_name,
                exc,
            )
    return {
        "enabled": True,
        "passed": True,
        "skip_reason": f"prompt_safety_api_error:{type(last_error).__name__ if last_error else 'unknown'}",
    }


def run_image_safety_for_case(*, cfg: EvaluatorConfig, sample: SampleCase, safety_runtime: SafetyRuntime) -> Dict[str, Any]:
    if not cfg.run_safety:
        return {"enabled": False, "passed": True, "skip_reason": "disabled_by_flag"}
    if safety_runtime.skip_reason:
        return {"enabled": True, "passed": True, "skip_reason": safety_runtime.skip_reason}
    if not safety_runtime.t2i_cfg or safety_runtime.t2i_safety is None:
        return {"enabled": True, "passed": True, "skip_reason": "invalid_safety_runtime"}

    t2i_cfg = safety_runtime.t2i_cfg
    t2i_api_cfg = t2i_cfg.get("api", {})
    t2i_runtime = t2i_cfg.get("runtime", {})
    t2i_req_cfg = t2i_cfg.get("llm_request", {})

    last_error: Optional[Exception] = None
    for model_name in get_model_candidates(t2i_api_cfg):
        try:
            t2i_passed, t2i_reason = safety_runtime.t2i_safety.call_safety_image_chat(
                client=safety_runtime.t2i_client,
                model=model_name,
                image_data_url=safety_runtime.t2i_safety.to_data_url(str(sample.image_path)),
                task_id=sample.task_id,
                prompt_template=safety_runtime.t2i_template,
                temperature=float(t2i_req_cfg.get("temperature", 0.0)),
                top_p=(float(t2i_req_cfg["top_p"]) if t2i_req_cfg.get("top_p") is not None else None),
                max_tokens=int(t2i_req_cfg.get("max_tokens", 4096)),
                max_retries=int(t2i_runtime.get("max_retries", 6)),
                backoff_base=float(t2i_runtime.get("backoff_base", 1.5)),
                backoff_cap=float(t2i_runtime.get("backoff_cap", 60.0)),
                limiter=safety_runtime.t2i_safety.RateLimiter(float(t2i_runtime.get("qps", 0.5))),
            )
            return {"enabled": True, "passed": bool(t2i_passed), "reason": t2i_reason}
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "[Agent:%s][Stage:Safety] Image safety model failed for %s with model=%s: %s",
                AGENT_NAME,
                sample.task_id,
                model_name,
                exc,
            )
    return {
        "enabled": True,
        "passed": True,
        "skip_reason": f"image_safety_api_error:{type(last_error).__name__ if last_error else 'unknown'}",
    }


def run_objective_for_case(
    *,
    cfg: EvaluatorConfig,
    sample: SampleCase,
    obj_checker_mod: Any,
    obj_utils: Any,
    objective_cfg: dict,
    prompt_check_template: str,
    image_check_template: str,
) -> Dict[str, Any]:
    check_root = cfg.generated_root / cfg.objective_output_subdir
    task_dir = sample.image_path.parent
    json_path = (
        check_root
        / f"{sample.task_type}_output_check"
        / sample.model_t2i
        / sample.model_llm
        / f"{sample.task_id}.json"
    )
    job = {
        "output_prefix": sample.task_type,
        "task_id": sample.task_id,
        "task_dir": task_dir,
        "json_path": json_path,
        "model_t2i": sample.model_t2i,
        "model_llm": sample.model_llm,
    }

    checklist_cache: Dict[str, Dict[str, dict]] = {}
    prompt_source_cache: Dict[Tuple[str, str], Dict[str, dict]] = {}
    checklist_dir = obj_utils.resolve_config_path(cfg.objective_config_path, objective_cfg["paths"]["checklist_dir"])

    prefixed_stdout = _AgentPrefixedStdout(
        stage="System2",
        event="skill_log",
        skill_label=f"Task:{sample.task_id}|Skill:S2PromptObjectiveSkill+S2ImageObjectiveSkill",
    )
    with contextlib.redirect_stdout(prefixed_stdout):
        obj_checker_mod.process_single_job(
            job=job,
            generated_root=cfg.generated_root,
            checklist_dir=checklist_dir,
            check_root=check_root,
            checklist_cache=checklist_cache,
            prompt_source_cache=prompt_source_cache,
            cfg=objective_cfg,
            prompt_check_template=prompt_check_template,
            image_check_template=image_check_template,
            dry_run=False,
            max_images_per_job=1,
        )

    payload = _load_json(json_path) if json_path.exists() else {}
    prompt_checks = payload.get("prompt_checkpoint")
    relative_image = str(sample.image_path.relative_to(cfg.generated_root)).replace("\\", "/")
    image_checks = None
    for item in reversed(payload.get("results", [])):
        if item.get("image") == relative_image:
            image_checks = item.get("image_checkpoint")
            break
    if image_checks is None and payload.get("results"):
        image_checks = payload["results"][-1].get("image_checkpoint")

    return {
        "prompt_checks": prompt_checks,
        "image_checks": image_checks,
        "prompt_constraint_satisfaction": _calc_constraint_rate(prompt_checks),
        "image_constraint_satisfaction": _calc_constraint_rate(image_checks),
        "raw_result_json": str(json_path),
    }


class ObjectiveExecutionCache:
    """Ensure objective core execution happens once, while exposing prompt/image split callbacks."""

    def __init__(self, run_full_fn: Any) -> None:
        self._run_full_fn = run_full_fn
        self._lock = threading.Lock()
        self._payload: Optional[Dict[str, Any]] = None

    def _ensure(self) -> Dict[str, Any]:
        with self._lock:
            if self._payload is None:
                self._payload = self._run_full_fn()
            return self._payload

    def run_prompt_objective(self) -> Dict[str, Any]:
        payload = self._ensure()
        return {
            "prompt_checks": payload.get("prompt_checks"),
            "prompt_constraint_satisfaction": payload.get("prompt_constraint_satisfaction"),
            "raw_result_json": payload.get("raw_result_json"),
        }

    def run_image_objective(self) -> Dict[str, Any]:
        payload = self._ensure()
        return {
            "image_checks": payload.get("image_checks"),
            "image_constraint_satisfaction": payload.get("image_constraint_satisfaction"),
            "raw_result_json": payload.get("raw_result_json"),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AtelierJudge evaluator runner (Safety -> S1 -> S2).")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K exemplars for memory retrieval.")
    parser.add_argument("--disable-safety", action="store_true", help="Disable safety filter stage.")
    parser.add_argument(
        "--generated-root",
        default=None,
        help="Override generated results root (default: _placeholder_generated_results under repo root).",
    )
    parser.add_argument(
        "--eval-results-dir",
        default=None,
        help="Directory to save run summary JSON (default: evaluator/eval_results under repo root).",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only discover and print case metadata without API evaluation.",
    )
    parser.add_argument(
        "--safety-decision-mode",
        choices=["both_fail", "any_fail"],
        default="any_fail",
        help=(
            "Safety aggregation rule: 'any_fail' (strict, default) blocks when either prompt or image fails; "
            "'both_fail' (lenient) blocks only when prompt and image both fail."
        ),
    )
    return parser.parse_args()


def _check_runtime_dependencies(run_safety: bool) -> None:
    required = [
        "numpy",
        "sklearn",
        "torch",
        "sentence_transformers",
        "finch",
        "transformers",
        "torchvision",
        "requests",
        "PIL",
    ]
    if run_safety:
        required.extend(["openai", "tqdm"])
    missing: List[str] = []
    for module_name in required:
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(module_name)
    if missing:
        missing_text = ", ".join(sorted(set(missing)))
        raise RuntimeError(f"Missing runtime dependencies: {missing_text}")


def _warn_if_not_ateliereval() -> None:
    if "ateliereval" in sys.executable:
        return
    active = os.getenv("CONDA_DEFAULT_ENV", "")
    if active != "ateliereval":
        logger.warning(
            "[Agent:%s][Stage:Env][Skill:RuntimeGuard] Current env is '%s', recommended env is 'ateliereval'. "
            "Try: conda run -n ateliereval python evaluator/run_evaluator.py",
            AGENT_NAME,
            active or "<unknown>",
        )


def _write_run_summary(cfg: EvaluatorConfig, payload: Dict[str, Any], eval_results_dir_arg: Optional[str]) -> Path:
    base_dir = Path(eval_results_dir_arg).resolve() if eval_results_dir_arg else (cfg.evaluator_root / "eval_results")
    base_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = base_dir / f"evaluation_run_{ts}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path = base_dir / "latest.json"
    latest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    args = parse_args()
    cfg = build_default_config(run_safety=(not args.disable_safety), top_k=int(args.top_k))
    cfg = EvaluatorConfig(
        workspace_root=cfg.workspace_root,
        submit_root=cfg.submit_root,
        evaluator_root=cfg.evaluator_root,
        generated_root=cfg.generated_root,
        subjective_dir=cfg.subjective_dir,
        objective_dir=cfg.objective_dir,
        safety_dir=cfg.safety_dir,
        configs_dir=cfg.configs_dir,
        objective_config_path=cfg.objective_config_path,
        llm_safety_config_path=cfg.llm_safety_config_path,
        t2i_safety_config_path=cfg.t2i_safety_config_path,
        objective_output_subdir=cfg.objective_output_subdir,
        preferred_model_llm=cfg.preferred_model_llm,
        preferred_model_t2i=cfg.preferred_model_t2i,
        preferred_task_ids=cfg.preferred_task_ids,
        top_k=cfg.top_k,
        run_safety=cfg.run_safety,
        safety_decision_mode=str(args.safety_decision_mode),
    )
    if args.generated_root:
        cfg = _with_generated_root(cfg, Path(args.generated_root).resolve())
    if not cfg.generated_root.exists():
        raise FileNotFoundError(f"generated_root does not exist: {cfg.generated_root}")

    _warn_if_not_ateliereval()
    _check_runtime_dependencies(run_safety=cfg.run_safety)
    skill_registry = load_skill_registry_from_config_dir(cfg.configs_dir)
    skill_instances = load_skill_instances(skill_registry)

    _progress("Boot", "start", f"Pipeline start. generated_root={cfg.generated_root}")
    _progress("Routing", "start", "Step 1/6 Discovering all evaluable cases from generated results...")
    cases: List[SampleCase] = []
    for task_type in TASK_TYPES:
        task_cases = discover_cases_for_task_type(cfg, task_type)
        _progress("Routing", "discovered", f"[{task_type}] discovered {len(task_cases)} case(s)")
        cases.extend(task_cases)
    if not cases:
        raise RuntimeError(f"No evaluable cases found under generated_root: {cfg.generated_root}")

    if args.prepare_only:
        payload = [
            {
                "task_type": s.task_type,
                "task_id": s.task_id,
                "model_llm": s.model_llm,
                "image_model_llm": s.image_model_llm,
                "model_t2i": s.model_t2i,
                "prompt_source_file": str(s.llm_json_path),
                "image_path": str(s.image_path),
            }
            for s in cases
        ]
        prepare_payload = {"prepare_only": True, "num_cases": len(payload), "cases": payload, "samples": payload}
        print(json.dumps(prepare_payload, ensure_ascii=False, indent=2))
        out_path = _write_run_summary(cfg, prepare_payload, args.eval_results_dir)
        logger.info("Saved prepare-only summary: %s", out_path)
        _progress("Finalize", "done", f"Prepare-only done. summary={out_path}")
        return

    _progress("Prep", "start", "Step 2/6 Preparing modules and memory bank...")
    sub_mem, sub_core = _prepare_subjective_modules(cfg)
    prompt_clusterer = sub_mem.PromptClusterer()
    image_clusterer = sub_mem.ImageClusterer()
    memories = _run_with_heartbeat(
        "[Skill:MemoryBankLoader]",
        sub_mem.load_subjective_memory_bank,
        prompt_clusterer=prompt_clusterer,
        image_clusterer=image_clusterer,
    )

    obj_checker_mod, obj_utils = _prepare_objective_modules(cfg)
    objective_cfg = obj_utils.load_config(cfg.objective_config_path)
    objective_cfg["paths"]["generated_root"] = str(cfg.generated_root)
    objective_cfg["paths"]["check_output_subdir"] = cfg.objective_output_subdir
    template_path = obj_utils.resolve_config_path(cfg.objective_config_path, objective_cfg["paths"]["template_json_path"])
    prompt_check_template, image_check_template = obj_utils.load_templates_from_json(template_path)
    safety_runtime = prepare_safety_runtime(cfg)

    results: List[Dict[str, Any]] = []
    total_cases = len(cases)
    _progress("Execution", "start", f"Step 3/6 Running per-case pipeline for {total_cases} case(s): Safety -> (S1 || S2)")
    for idx, sample in enumerate(cases, start=1):
        route = get_route(sample.task_type)
        task_description = str(memories.task_description_lookup.get(sample.task_id, "")).strip() or sample.original_task or "N/A"
        _progress("Execution", "sample_start", f"Case {idx}/{total_cases} [{sample.task_type}] {sample.task_id} route={route}")

        safety_result = _run_with_heartbeat(
            f"[Task:{sample.task_id}][Skill:SafetyPromptSkill+SafetyImageSkill]",
            run_safety_skills,
            cfg=cfg,
            sample=sample,
            safety_runtime=safety_runtime,
            run_prompt_safety_fn=run_prompt_safety_for_case,
            run_image_safety_fn=run_image_safety_for_case,
            skill_instances=skill_instances,
        )
        safety_payload, safety_traces = safety_result
        if safety_payload.get("enabled") and not safety_payload.get("passed", False):
            _progress("Execution", "sample_skip", f"[Task:{sample.task_id}] safety failed, skip S1/S2")
            execution_trace = list(safety_traces)
            results.append(
                {
                    "task_id": sample.task_id,
                    "task_type": sample.task_type,
                    "task_description": task_description,
                    "model_llm": sample.model_llm,
                    "image_model_llm": sample.image_model_llm,
                    "model_t2i": sample.model_t2i,
                    "prompt_source_file": str(sample.llm_json_path),
                    "prompt": sample.prompt_text,
                    "image_path": str(sample.image_path),
                    "safety": safety_payload,
                    "subjective": "skipped_due_to_safety",
                    "objective": "skipped_due_to_safety",
                    "execution_trace": execution_trace,
                }
            )
            continue

        with ThreadPoolExecutor(max_workers=2) as executor:
            def _run_objective_full() -> Dict[str, Any]:
                return run_objective_for_case(
                    cfg=cfg,
                    sample=sample,
                    obj_checker_mod=obj_checker_mod,
                    obj_utils=obj_utils,
                    objective_cfg=objective_cfg,
                    prompt_check_template=prompt_check_template,
                    image_check_template=image_check_template,
                )
            objective_cache = ObjectiveExecutionCache(_run_objective_full)

            s1_future = executor.submit(
                _run_with_heartbeat,
                f"[Task:{sample.task_id}][Skill:S1PromptSubjectiveSkill+S1ImageSubjectiveSkill]",
                run_system1_subjective_skills,
                route=route,
                task_id=sample.task_id,
                task_description=task_description,
                prompt_text=sample.prompt_text,
                image_path=str(sample.image_path),
                sub_mem=sub_mem,
                sub_core=sub_core,
                memories=memories,
                prompt_clusterer=prompt_clusterer,
                image_clusterer=image_clusterer,
                top_k=cfg.top_k,
                skill_instances=skill_instances,
            )
            s2_future = executor.submit(
                _run_with_heartbeat,
                f"[Task:{sample.task_id}][Skill:S2PromptObjectiveSkill+S2ImageObjectiveSkill]",
                run_system2_objective_skills,
                route=route,
                run_prompt_objective_fn=objective_cache.run_prompt_objective,
                run_image_objective_fn=objective_cache.run_image_objective,
                skill_instances=skill_instances,
                task_id=sample.task_id,
            )
            subjective_payload, s1_traces = s1_future.result()
            objective_payload, s2_traces = s2_future.result()

        execution_trace = list(safety_traces) + list(s1_traces) + list(s2_traces)
        results.append(
            {
                "task_id": sample.task_id,
                "task_type": sample.task_type,
                "task_description": task_description,
                "model_llm": sample.model_llm,
                "image_model_llm": sample.image_model_llm,
                "model_t2i": sample.model_t2i,
                "prompt_source_file": str(sample.llm_json_path),
                "prompt": sample.prompt_text,
                "image_path": str(sample.image_path),
                "safety": safety_payload,
                "subjective": subjective_payload,
                "objective": objective_payload,
                "execution_trace": execution_trace,
            }
        )
        _progress("Execution", "sample_done", f"[Task:{sample.task_id}] sample complete")

    final_payload = {"pipeline": "Safety -> System1 -> System2", "results": results}
    _progress("Finalize", "summary_print", "Step 4/6 Pipeline completed, printing JSON summary")
    print(json.dumps(final_payload, ensure_ascii=False, indent=2))
    _progress("Finalize", "summary_write", "Step 5/6 Writing summary file")
    out_path = _write_run_summary(cfg, final_payload, args.eval_results_dir)
    logger.info("Saved evaluation summary: %s", out_path)
    _progress("Finalize", "done", f"Step 6/6 Done. summary={out_path}")


if __name__ == "__main__":
    main()
