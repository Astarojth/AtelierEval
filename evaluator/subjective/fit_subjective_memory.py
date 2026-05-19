from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


THIS_FILE = Path(__file__).resolve()
SUBJECTIVE_ROOT = THIS_FILE.parent
EVALUATOR_ROOT = SUBJECTIVE_ROOT.parent
REPO_ROOT = EVALUATOR_ROOT.parent

for _path in (REPO_ROOT, EVALUATOR_ROOT, SUBJECTIVE_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from subjective.s1_subjective_memory_skill_engine import (  # type: ignore
    ImageClusterer,
    PromptClusterer,
    SubjectiveMemoryBank,
    evaluate_image_dimensions_with_memory,
    evaluate_prompt_subjective_with_memory,
    load_subjective_memory_bank,
    load_task_description_lookup,
)


MODEL_LABEL_TO_KEY: Dict[str, str] = {
    "GPT-5.2": "gpt_5.2",
    "Gem-3": "gemini_3_pro_preview",
    "Cl-4.5": "claude_sonnet_4_5_20250929",
    "GPT-4.1": "gpt_4.1_2025_04_14",
    "Gem-2": "gemini_2.0_flash",
    "Qwen-L": "qwen3_235b_a22b",
    "GPT-4n": "gpt_4.1_nano_2025_04_14",
    "Qwen-S": "qwen3_8b",
}

GROUP_LABEL_TO_KEY: Dict[str, str] = {"no.": "novice", "sk.": "skilled"}
BACKEND_LABEL_TO_KEY: Dict[str, str] = {
    "nBanana": "gemini_3_pro_image_preview",
    "GI-1": "gpt_image_1_all",
    "Flux Pro": "black_forest_labs__flux_1.1_pro",
    "SDXL": "sdxl_local",
}
BACKEND_ALIAS_TO_KEY: Dict[str, str] = {
    "black-forest-labs/flux-1.1-pro": "black_forest_labs__flux_1.1_pro",
    "black-forest-labs__flux-1.1-pro": "black_forest_labs__flux_1.1_pro",
    "black_forest_labs__flux_1.1_pro": "black_forest_labs__flux_1.1_pro",
    "gemini_3_pro_image_preview": "gemini_3_pro_image_preview",
    "gpt_image_1_all": "gpt_image_1_all",
    "sdxl_local": "sdxl_local",
}

PROMPT_MEMORY_FILES: Dict[str, str] = {
    "co": "prompt_from_prompt_memory_co.json",
    "im": "prompt_from_prompt_memory_im.json",
    "oe": "prompt_from_prompt_memory_oe.json",
}
IMAGE_MEMORY_FILES: Dict[str, str] = {
    "co": "image_from_image_memory_co.json",
    "oe": "image_from_image_memory_oe.json",
}


@dataclass(frozen=True)
class PaperTargetCell:
    task_type: str
    group: str
    model_key: str
    prompt_subj: float
    image_avg_subj: float
    backend_subj: Dict[str, float]


@dataclass(frozen=True)
class SamplePlan:
    rounds: List[Dict[str, List[str]]]
    prompt_intersections: Dict[str, int]
    image_intersections: Dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit AtelierJudge subjective memory against paper-reported results.")
    parser.add_argument(
        "--expert-root",
        default=str(REPO_ROOT.parent / "generated_result" / "generated_results_expert"),
        help="Generated root for skilled/expert prompts.",
    )
    parser.add_argument(
        "--novice-root",
        default=str(REPO_ROOT.parent / "generated_result" / "generated_results_novice"),
        help="Generated root for novice prompts.",
    )
    parser.add_argument(
        "--paper-root",
        default=str(Path("/home/astar/workspace/writing/refer/_ICML2026_AtelierEval")),
        help="Paper source root used as fitting reference.",
    )
    parser.add_argument(
        "--memory-dir",
        default=str(SUBJECTIVE_ROOT / "memory"),
        help="Memory directory to evaluate.",
    )
    parser.add_argument("--judge-model", default="gpt-5.1", help="Judge model for subjective evaluation.")
    parser.add_argument(
        "--api-key-override",
        default="",
        help="Optional API key override for subjective evaluation. When set, overrides api_modellist order.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Top-K retrieved exemplars.")
    parser.add_argument("--sample-per-type", type=int, default=1, help="Number of task_ids sampled per task type per round.")
    parser.add_argument("--num-rounds", type=int, default=2, help="Number of repeated random sampling rounds.")
    parser.add_argument(
        "--eval-repeats",
        type=int,
        default=1,
        help="Number of repeated judge calls to average for each sampled cell on the same fixed sample.",
    )
    parser.add_argument("--seed", type=int, default=20260323, help="Base random seed.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries for each subjective API call.")
    parser.add_argument("--retry-delay-seconds", nargs="+", type=int, default=[3, 8, 15], help="Retry delays in seconds.")
    parser.add_argument(
        "--task-types",
        nargs="+",
        choices=["co", "im", "oe"],
        default=["co", "im", "oe"],
        help="Subset of task types to include.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=["novice", "skilled"],
        default=["novice", "skilled"],
        help="Subset of prompter groups to include.",
    )
    parser.add_argument(
        "--model-keys",
        nargs="+",
        default=None,
        help="Optional subset of internal model keys such as gpt_5.2 or gemini_3_pro_preview.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(SUBJECTIVE_ROOT / "memory_fit_runs"),
        help="Directory to save reports and suggested variants.",
    )
    parser.add_argument(
        "--mode",
        choices=["prepare", "baseline", "suggest"],
        default="prepare",
        help="prepare: inspect targets and sample plan, baseline: run subjective fit, suggest: baseline + write shifted memory variant.",
    )
    return parser.parse_args()


def _canonical_model_key(raw: str) -> str:
    text = raw.strip()
    aliases = {
        "gpt_4_1_2025_04_14": "gpt_4.1_2025_04_14",
        "gpt_4_1_nano_2025_04_14": "gpt_4.1_nano_2025_04_14",
    }
    return aliases.get(text, text)


def _canonical_backend_key(raw: str) -> str:
    text = raw.strip().replace("\\", "/")
    return BACKEND_ALIAS_TO_KEY.get(text, text.replace("/", "__").replace("-", "_"))


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def _score_from_dimensions(payload: Optional[Mapping[str, Any]]) -> Optional[float]:
    if not isinstance(payload, Mapping):
        return None
    vals: List[int] = []
    for item in payload.values():
        if not isinstance(item, Mapping):
            continue
        score = item.get("score")
        if isinstance(score, int):
            vals.append(score)
    return _safe_mean(vals)


def _spearman_like(target_map: Mapping[str, float], pred_map: Mapping[str, float]) -> Optional[float]:
    shared = [key for key in target_map.keys() if key in pred_map]
    if len(shared) < 2:
        return None

    def _ranks(values: Mapping[str, float]) -> Dict[str, float]:
        ordered = sorted(values.items(), key=lambda kv: (kv[1], kv[0]))
        ranks: Dict[str, float] = {}
        i = 0
        while i < len(ordered):
            j = i
            while j + 1 < len(ordered) and ordered[j + 1][1] == ordered[i][1]:
                j += 1
            rank_value = (i + j + 2) / 2.0
            for idx in range(i, j + 1):
                ranks[ordered[idx][0]] = rank_value
            i = j + 1
        return ranks

    target_ranks = _ranks({key: target_map[key] for key in shared})
    pred_ranks = _ranks({key: pred_map[key] for key in shared})
    target_vals = [target_ranks[key] for key in shared]
    pred_vals = [pred_ranks[key] for key in shared]
    mean_t = sum(target_vals) / len(target_vals)
    mean_p = sum(pred_vals) / len(pred_vals)
    cov = sum((t - mean_t) * (p - mean_p) for t, p in zip(target_vals, pred_vals))
    var_t = sum((t - mean_t) ** 2 for t in target_vals)
    var_p = sum((p - mean_p) ** 2 for p in pred_vals)
    if var_t <= 0 or var_p <= 0:
        return None
    return cov / math.sqrt(var_t * var_p)


def parse_paper_targets(paper_root: Path) -> Dict[str, Dict[str, Dict[str, PaperTargetCell]]]:
    tex_path = paper_root / "appendix" / "More_Results.tex"
    if not tex_path.exists():
        raise FileNotFoundError(f"Paper target source not found: {tex_path}")

    text = tex_path.read_text(encoding="utf-8")
    task_type = ""
    current_model_label = ""
    targets: Dict[str, Dict[str, Dict[str, PaperTargetCell]]] = defaultdict(lambda: defaultdict(dict))

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if "\\label{tab:co}" in line:
            task_type = "co"
            continue
        if "\\label{tab:im}" in line:
            task_type = "im"
            continue
        if "\\label{tab:oe}" in line:
            task_type = "oe"
            continue
        if not task_type:
            continue
        if "& no. &" not in line and "& sk. &" not in line:
            continue

        if line.startswith("\\multirow{2}{*}{"):
            current_model_label = line.split("{", 3)[3].split("}", 1)[0].strip()
            cells = [part.strip().rstrip("\\") for part in line.split("&")]
            group_label = cells[1]
            numeric_cells = cells[2:]
        else:
            cells = [part.strip().rstrip("\\") for part in line.split("&")]
            group_label = cells[1]
            numeric_cells = cells[2:]

        if current_model_label == "Human":
            continue
        model_key = MODEL_LABEL_TO_KEY.get(current_model_label)
        group_key = GROUP_LABEL_TO_KEY.get(group_label)
        if model_key is None or group_key is None:
            continue

        values = [float(part) for part in numeric_cells]
        backend_subj = {
            BACKEND_LABEL_TO_KEY["nBanana"]: values[4],
            BACKEND_LABEL_TO_KEY["GI-1"]: values[6],
            BACKEND_LABEL_TO_KEY["Flux Pro"]: values[8],
            BACKEND_LABEL_TO_KEY["SDXL"]: values[10],
        }
        targets[task_type][group_key][model_key] = PaperTargetCell(
            task_type=task_type,
            group=group_key,
            model_key=model_key,
            prompt_subj=values[0],
            image_avg_subj=values[2],
            backend_subj=backend_subj,
        )

    return targets


def filter_paper_targets(
    paper_targets: Mapping[str, Any],
    *,
    task_types: Sequence[str],
    groups: Sequence[str],
    model_keys: Optional[Sequence[str]],
) -> Dict[str, Dict[str, Dict[str, PaperTargetCell]]]:
    selected_task_types = set(task_types)
    selected_groups = set(groups)
    selected_models = set(model_keys or [])
    filtered: Dict[str, Dict[str, Dict[str, PaperTargetCell]]] = {}
    for task_type, task_groups in paper_targets.items():
        if task_type not in selected_task_types:
            continue
        filtered[task_type] = {}
        for group_key, model_map in task_groups.items():
            if group_key not in selected_groups:
                continue
            if model_keys is None:
                filtered_models = dict(model_map)
            else:
                filtered_models = {model_key: cell for model_key, cell in model_map.items() if model_key in selected_models}
            filtered[task_type][group_key] = filtered_models
    return filtered


def _load_prompt_map(prompt_json_path: Path) -> Dict[str, Dict[str, Any]]:
    payload = json.loads(prompt_json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict prompt payload in {prompt_json_path}")
    return payload


def discover_generated_content(
    generated_root: Path,
    *,
    group_key: str,
    task_description_lookup: Mapping[str, str],
) -> Dict[str, Any]:
    prompt_index: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    image_index: Dict[str, Dict[str, Dict[str, Dict[str, Path]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )

    llm_root = generated_root / "1_llm"
    if llm_root.exists():
        for prompt_file in sorted(llm_root.glob("*_output_llm/*_prompts_*.json")):
            task_type = prompt_file.parent.name.split("_", 1)[0]
            model_key = _canonical_model_key(prompt_file.stem.replace(f"{task_type}_prompts_", "", 1))
            prompt_map = _load_prompt_map(prompt_file)
            for task_id, payload in prompt_map.items():
                if not isinstance(payload, Mapping):
                    continue
                prompt_text = str(payload.get("generated_prompt", "")).strip()
                if not prompt_text:
                    continue
                prompt_index[task_type][model_key][task_id] = {
                    "prompt_text": prompt_text,
                    "task_description": str(task_description_lookup.get(task_id, "")).strip()
                    or str(payload.get("original_task", "")).strip(),
                }

    tti_root = generated_root / "2_tti"
    if tti_root.exists():
        for task_type_dir in sorted(tti_root.glob("*_output_tti")):
            task_type = task_type_dir.name.split("_", 1)[0]
            for image_path in sorted(task_type_dir.rglob("*")):
                if not image_path.is_file() or image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                    continue
                try:
                    rel_parts = image_path.relative_to(task_type_dir).parts
                except ValueError:
                    continue
                if len(rel_parts) < 4:
                    continue
                task_id = rel_parts[-2]
                model_key = _canonical_model_key(rel_parts[-3])
                backend_parts = rel_parts[:-3]
                backend_raw = backend_parts[-1] if backend_parts else ""
                backend_key = _canonical_backend_key(backend_raw)
                image_index[task_type][model_key][backend_key][task_id] = image_path

    return {"group": group_key, "prompt_index": prompt_index, "image_index": image_index}


def build_sample_plan(
    discovered: Mapping[str, Dict[str, Any]],
    paper_targets: Mapping[str, Any],
    *,
    sample_per_type: int,
    num_rounds: int,
    seed: int,
) -> SamplePlan:
    prompt_intersections: Dict[str, int] = {}
    image_intersections: Dict[str, int] = {}
    task_types = sorted(paper_targets.keys())

    for task_type in task_types:
        sets: List[set[str]] = []
        for group_key in sorted(paper_targets[task_type].keys()):
            group_data = discovered[group_key]["prompt_index"][task_type]
            for model_key in paper_targets[task_type][group_key].keys():
                if model_key not in group_data:
                    raise KeyError(f"Prompt content missing for group={group_key}, task_type={task_type}, model={model_key}")
                sets.append(set(group_data[model_key].keys()))
        prompt_intersections[task_type] = len(set.intersection(*sets)) if sets else 0

    for task_type in [task for task in task_types if task in {"co", "oe"}]:
        sets = []
        for group_key in sorted(paper_targets[task_type].keys()):
            group_data = discovered[group_key]["image_index"][task_type]
            for model_key, target_cell in paper_targets[task_type][group_key].items():
                if model_key not in group_data:
                    raise KeyError(f"Image content missing for group={group_key}, task_type={task_type}, model={model_key}")
                for backend_key in target_cell.backend_subj.keys():
                    if backend_key not in group_data[model_key]:
                        raise KeyError(
                            f"Backend image content missing for group={group_key}, task_type={task_type}, "
                            f"model={model_key}, backend={backend_key}"
                        )
                    sets.append(set(group_data[model_key][backend_key].keys()))
        image_intersections[task_type] = len(set.intersection(*sets)) if sets else 0

    rounds: List[Dict[str, List[str]]] = []
    for round_idx in range(num_rounds):
        rng = random.Random(seed + round_idx)
        round_plan: Dict[str, List[str]] = {}
        for task_type in task_types:
            if task_type == "im":
                source_sets: List[set[str]] = []
                for group_key in sorted(paper_targets[task_type].keys()):
                    group_data = discovered[group_key]["prompt_index"][task_type]
                    for model_key in paper_targets[task_type][group_key].keys():
                        source_sets.append(set(group_data[model_key].keys()))
            else:
                source_sets = []
                for group_key in sorted(paper_targets[task_type].keys()):
                    group_data = discovered[group_key]["image_index"][task_type]
                    for model_key, target_cell in paper_targets[task_type][group_key].items():
                        for backend_key in target_cell.backend_subj.keys():
                            source_sets.append(set(group_data[model_key][backend_key].keys()))
            shared = sorted(set.intersection(*source_sets)) if source_sets else []
            chosen = rng.sample(shared, k=min(sample_per_type, len(shared))) if shared else []
            round_plan[task_type] = chosen
        rounds.append(round_plan)

    return SamplePlan(
        rounds=rounds,
        prompt_intersections=prompt_intersections,
        image_intersections=image_intersections,
    )


def _build_memory_path_maps(memory_dir: Path) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    prompt_paths = {task_type: memory_dir / filename for task_type, filename in PROMPT_MEMORY_FILES.items()}
    image_paths = {task_type: memory_dir / filename for task_type, filename in IMAGE_MEMORY_FILES.items()}
    for path in [*prompt_paths.values(), *image_paths.values()]:
        if not path.exists():
            raise FileNotFoundError(f"Memory file not found: {path}")
    return prompt_paths, image_paths


def _mean_score(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    return _safe_mean(clean)


def _evaluate_rounds(
    sample_plan: SamplePlan,
    discovered: Mapping[str, Dict[str, Any]],
    paper_targets: Mapping[str, Any],
    *,
    memory_dir: Path,
    judge_model: str,
    top_k: int,
    eval_repeats: int,
    max_retries: int,
    retry_delays: Sequence[int],
) -> Dict[str, Any]:
    prompt_memory_paths, image_memory_paths = _build_memory_path_maps(memory_dir)
    prompt_clusterer = PromptClusterer()
    image_clusterer = ImageClusterer()
    bank: SubjectiveMemoryBank = load_subjective_memory_bank(
        prompt_memory_paths=prompt_memory_paths,
        image_memory_paths=image_memory_paths,
        prompt_clusterer=prompt_clusterer,
        image_clusterer=image_clusterer,
    )

    prompt_scores: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    image_scores: Dict[Tuple[str, str, str, str], List[float]] = defaultdict(list)
    retrieval_counts: Dict[str, Dict[str, int]] = {"prompt": defaultdict(int), "image": defaultdict(int)}
    errors: List[Dict[str, Any]] = []

    def _log(message: str) -> None:
        print(f"[fit_subjective_memory] {message}", flush=True)

    def _with_retries(label: str, fn: Any) -> Any:
        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= max_retries:
                    break
                delay = retry_delays[min(attempt - 1, len(retry_delays) - 1)] if retry_delays else 3
                _log(f"{label} failed on attempt {attempt}/{max_retries}: {type(exc).__name__}: {exc}; retry in {delay}s")
                import time
                time.sleep(delay)
        if last_error is None:
            raise RuntimeError(f"{label} failed without an exception payload")
        raise last_error

    _log(
        "Load memory bank: "
        f"memory_dir={memory_dir} judge_model={judge_model} top_k={top_k} "
        f"rounds={len(sample_plan.rounds)} eval_repeats={eval_repeats}"
    )

    active_groups = sorted({group_key for task_groups in paper_targets.values() for group_key in task_groups.keys()})
    for round_idx, round_plan in enumerate(sample_plan.rounds):
        _log(f"Round {round_idx + 1}/{len(sample_plan.rounds)} sampled task_ids={round_plan}")
        for task_type, task_ids in round_plan.items():
            for group_key in active_groups:
                if group_key not in paper_targets[task_type]:
                    continue
                prompt_group_index = discovered[group_key]["prompt_index"][task_type]
                target_group = paper_targets[task_type][group_key]
                for model_key in target_group.keys():
                    prompt_model_index = prompt_group_index.get(model_key, {})
                    for task_id in task_ids:
                        prompt_payload = prompt_model_index.get(task_id)
                        if prompt_payload is None:
                            errors.append(
                                {
                                    "round": round_idx,
                                    "kind": "missing_prompt_payload",
                                    "task_type": task_type,
                                    "group": group_key,
                                    "model": model_key,
                                    "task_id": task_id,
                                }
                            )
                            continue
                        for repeat_idx in range(eval_repeats):
                            try:
                                _log(
                                    f"Prompt eval round={round_idx + 1} repeat={repeat_idx + 1}/{eval_repeats} "
                                    f"task_type={task_type} group={group_key} model={model_key} task_id={task_id}"
                                )
                                prompt_result = _with_retries(
                                    f"prompt:{round_idx + 1}:{repeat_idx + 1}:{task_type}:{group_key}:{model_key}:{task_id}",
                                    lambda: evaluate_prompt_subjective_with_memory(
                                        task_id=task_id,
                                        task_type=task_type,
                                        task_description=prompt_payload["task_description"],
                                        prompt_text=prompt_payload["prompt_text"],
                                        model_name=judge_model,
                                        memories=bank,
                                        prompt_clusterer=prompt_clusterer,
                                        top_k=top_k,
                                    ),
                                )
                                prompt_score = _score_from_dimensions(prompt_result.get("prompt_dimensions"))
                                if prompt_score is not None:
                                    prompt_scores[(task_type, group_key, model_key)].append(prompt_score)
                                for retrieved in prompt_result.get("retrieved_prompt_exemplars", []):
                                    ex_task_id = str(retrieved.get("task_id", "")).strip()
                                    if ex_task_id:
                                        retrieval_counts["prompt"][f"{task_type}:{ex_task_id}"] += 1
                            except Exception as exc:  # noqa: BLE001
                                errors.append(
                                    {
                                        "round": round_idx,
                                        "repeat": repeat_idx,
                                        "kind": "prompt_eval_error",
                                        "task_type": task_type,
                                        "group": group_key,
                                        "model": model_key,
                                        "task_id": task_id,
                                        "error": f"{type(exc).__name__}: {exc}",
                                    }
                                )

                        if task_type not in {"co", "oe"}:
                            continue
                        image_group_index = discovered[group_key]["image_index"][task_type].get(model_key, {})
                        for backend_key in target_group[model_key].backend_subj.keys():
                            image_path = image_group_index.get(backend_key, {}).get(task_id)
                            if image_path is None:
                                errors.append(
                                    {
                                        "round": round_idx,
                                        "kind": "missing_image_payload",
                                        "task_type": task_type,
                                        "group": group_key,
                                        "model": model_key,
                                        "backend": backend_key,
                                        "task_id": task_id,
                                    }
                                )
                                continue
                            for repeat_idx in range(eval_repeats):
                                try:
                                    _log(
                                        f"Image eval round={round_idx + 1} repeat={repeat_idx + 1}/{eval_repeats} "
                                        f"task_type={task_type} group={group_key} model={model_key} "
                                        f"backend={backend_key} task_id={task_id}"
                                    )
                                    image_result = _with_retries(
                                        f"image:{round_idx + 1}:{repeat_idx + 1}:{task_type}:{group_key}:{model_key}:{backend_key}:{task_id}",
                                        lambda: evaluate_image_dimensions_with_memory(
                                            task_id=task_id,
                                            task_type=task_type,
                                            task_description=prompt_payload["task_description"],
                                            prompt_text=prompt_payload["prompt_text"],
                                            image_path=str(image_path),
                                            model_name=judge_model,
                                            memories=bank,
                                            image_clusterer=image_clusterer,
                                            top_k=top_k,
                                        ),
                                    )
                                    image_score = _score_from_dimensions(image_result.get("image_dimensions"))
                                    if image_score is not None:
                                        image_scores[(task_type, group_key, model_key, backend_key)].append(image_score)
                                    for retrieved in image_result.get("retrieved_image_exemplars", []):
                                        ex_task_id = str(retrieved.get("task_id", "")).strip()
                                        if ex_task_id:
                                            retrieval_counts["image"][f"{task_type}:{ex_task_id}"] += 1
                                except Exception as exc:  # noqa: BLE001
                                    errors.append(
                                        {
                                            "round": round_idx,
                                            "repeat": repeat_idx,
                                            "kind": "image_eval_error",
                                            "task_type": task_type,
                                            "group": group_key,
                                            "model": model_key,
                                            "backend": backend_key,
                                            "task_id": task_id,
                                            "error": f"{type(exc).__name__}: {exc}",
                                        }
                                    )

    prompt_means = {
        f"{task_type}|{group_key}|{model_key}": _mean_score(values)
        for (task_type, group_key, model_key), values in prompt_scores.items()
    }
    image_means = {
        f"{task_type}|{group_key}|{model_key}|{backend_key}": _mean_score(values)
        for (task_type, group_key, model_key, backend_key), values in image_scores.items()
    }

    return {
        "prompt_means": prompt_means,
        "image_means": image_means,
        "retrieval_counts": {
            "prompt": dict(sorted(retrieval_counts["prompt"].items(), key=lambda kv: (-kv[1], kv[0]))),
            "image": dict(sorted(retrieval_counts["image"].items(), key=lambda kv: (-kv[1], kv[0]))),
        },
        "errors": errors,
    }


def compute_fit_report(
    paper_targets: Mapping[str, Any],
    predictions: Mapping[str, Any],
) -> Dict[str, Any]:
    prompt_abs_errors: List[float] = []
    image_abs_errors: List[float] = []
    image_avg_abs_errors: List[float] = []
    prompt_rank_scores: List[float] = []
    image_rank_scores: List[float] = []
    delta_abs_errors: List[float] = []
    unit_signed_errors: Dict[str, List[float]] = defaultdict(list)
    missing_cells = 0
    total_cells = 0

    prompt_means = predictions["prompt_means"]
    image_means = predictions["image_means"]

    for task_type, task_group_targets in paper_targets.items():
        for group_key, model_targets in task_group_targets.items():
            prompt_target_rank: Dict[str, float] = {}
            prompt_pred_rank: Dict[str, float] = {}
            for model_key, target_cell in model_targets.items():
                total_cells += 1
                prompt_key = f"{task_type}|{group_key}|{model_key}"
                prompt_pred = prompt_means.get(prompt_key)
                if prompt_pred is None:
                    missing_cells += 1
                else:
                    prompt_abs_errors.append(abs(prompt_pred - target_cell.prompt_subj))
                    unit_signed_errors[f"prompt_{task_type}"].append(target_cell.prompt_subj - prompt_pred)
                    prompt_target_rank[model_key] = target_cell.prompt_subj
                    prompt_pred_rank[model_key] = prompt_pred

                if task_type in {"co", "oe"}:
                    backend_target_rank: Dict[str, Dict[str, float]] = defaultdict(dict)
                    backend_pred_rank: Dict[str, Dict[str, float]] = defaultdict(dict)
                    backend_preds: List[float] = []
                    backend_tgts: List[float] = []
                    for backend_key, backend_target in target_cell.backend_subj.items():
                        total_cells += 1
                        image_key = f"{task_type}|{group_key}|{model_key}|{backend_key}"
                        image_pred = image_means.get(image_key)
                        if image_pred is None:
                            missing_cells += 1
                            continue
                        image_abs_errors.append(abs(image_pred - backend_target))
                        unit_signed_errors[f"image_{task_type}"].append(backend_target - image_pred)
                        backend_preds.append(image_pred)
                        backend_tgts.append(backend_target)
                        backend_target_rank[backend_key][model_key] = backend_target
                        backend_pred_rank[backend_key][model_key] = image_pred

                    if backend_preds and backend_tgts:
                        image_avg_abs_errors.append(abs((sum(backend_preds) / len(backend_preds)) - target_cell.image_avg_subj))

            prompt_rank = _spearman_like(prompt_target_rank, prompt_pred_rank)
            if prompt_rank is not None:
                prompt_rank_scores.append(prompt_rank)

        if task_type in {"co", "oe"}:
            for group_key in sorted(task_group_targets.keys()):
                for backend_key in BACKEND_LABEL_TO_KEY.values():
                    target_rank: Dict[str, float] = {}
                    pred_rank: Dict[str, float] = {}
                    for model_key, target_cell in paper_targets[task_type][group_key].items():
                        image_key = f"{task_type}|{group_key}|{model_key}|{backend_key}"
                        image_pred = image_means.get(image_key)
                        if image_pred is None:
                            continue
                        target_rank[model_key] = target_cell.backend_subj[backend_key]
                        pred_rank[model_key] = image_pred
                    image_rank = _spearman_like(target_rank, pred_rank)
                    if image_rank is not None:
                        image_rank_scores.append(image_rank)

    for task_type, task_group_targets in paper_targets.items():
        if "novice" not in task_group_targets or "skilled" not in task_group_targets:
            continue
        for model_key in task_group_targets["novice"].keys():
            novice_prompt = prompt_means.get(f"{task_type}|novice|{model_key}")
            skilled_prompt = prompt_means.get(f"{task_type}|skilled|{model_key}")
            if novice_prompt is not None and skilled_prompt is not None:
                target_delta = task_group_targets["skilled"][model_key].prompt_subj - task_group_targets["novice"][model_key].prompt_subj
                pred_delta = skilled_prompt - novice_prompt
                delta_abs_errors.append(abs(pred_delta - target_delta))

            if task_type in {"co", "oe"}:
                for backend_key in BACKEND_LABEL_TO_KEY.values():
                    novice_image = image_means.get(f"{task_type}|novice|{model_key}|{backend_key}")
                    skilled_image = image_means.get(f"{task_type}|skilled|{model_key}|{backend_key}")
                    if novice_image is None or skilled_image is None:
                        continue
                    target_delta = (
                        task_group_targets["skilled"][model_key].backend_subj[backend_key]
                        - task_group_targets["novice"][model_key].backend_subj[backend_key]
                    )
                    pred_delta = skilled_image - novice_image
                    delta_abs_errors.append(abs(pred_delta - target_delta))

    prompt_mae = _safe_mean(prompt_abs_errors) or 1.0
    image_mae = _safe_mean(image_abs_errors) or 1.0
    image_avg_mae = _safe_mean(image_avg_abs_errors) or 1.0
    rank_score = _safe_mean(prompt_rank_scores + image_rank_scores) or 0.0
    delta_mae = _safe_mean(delta_abs_errors) or 1.0
    coverage = 1.0 - (missing_cells / total_cells) if total_cells else 0.0

    composite = (
        0.35 * (prompt_mae / 4.0)
        + 0.30 * (image_mae / 4.0)
        + 0.10 * (image_avg_mae / 4.0)
        + 0.15 * (1.0 - max(min(rank_score, 1.0), -1.0))
        + 0.10 * (delta_mae / 4.0)
        + 0.10 * (1.0 - coverage)
    )

    unit_shift_suggestion: Dict[str, int] = {}
    for unit_name, errors in unit_signed_errors.items():
        mean_error = _safe_mean(errors) or 0.0
        if mean_error >= 0.35:
            unit_shift_suggestion[unit_name] = 1
        elif mean_error <= -0.35:
            unit_shift_suggestion[unit_name] = -1
        else:
            unit_shift_suggestion[unit_name] = 0

    return {
        "fit_score": round(composite, 6),
        "coverage": round(coverage, 6),
        "prompt_mae": round(prompt_mae, 6),
        "image_mae": round(image_mae, 6),
        "image_avg_mae": round(image_avg_mae, 6),
        "rank_score": round(rank_score, 6),
        "delta_mae": round(delta_mae, 6),
        "missing_cells": missing_cells,
        "total_cells": total_cells,
        "unit_shift_suggestion": unit_shift_suggestion,
        "unit_mean_signed_error": {
            unit_name: round(_safe_mean(errors) or 0.0, 6)
            for unit_name, errors in sorted(unit_signed_errors.items())
        },
    }


def build_exemplar_suggestions(
    *,
    retrieval_counts: Mapping[str, Mapping[str, int]],
    fit_report: Mapping[str, Any],
    top_n_per_unit: int = 8,
) -> Dict[str, List[Dict[str, Any]]]:
    unit_shifts: Mapping[str, int] = fit_report.get("unit_shift_suggestion", {})
    suggestions: Dict[str, List[Dict[str, Any]]] = {}

    def _collect(kind: str, task_type: str, unit_name: str) -> List[Dict[str, Any]]:
        shift = int(unit_shifts.get(unit_name, 0))
        if shift == 0:
            return []
        ranked = []
        for key, count in retrieval_counts.get(kind, {}).items():
            key_task_type, task_id = key.split(":", 1)
            if key_task_type != task_type:
                continue
            ranked.append(
                {
                    "task_id": task_id,
                    "retrieval_count": int(count),
                    "suggested_score_shift": shift,
                    "scope": "all_dimensions_in_exemplar",
                    "kind": kind,
                }
            )
        ranked.sort(key=lambda item: (-item["retrieval_count"], item["task_id"]))
        return ranked[:top_n_per_unit]

    for task_type in ("co", "im", "oe"):
        prompt_unit = f"prompt_{task_type}"
        suggestions[prompt_unit] = _collect("prompt", task_type, prompt_unit)
        if task_type in {"co", "oe"}:
            image_unit = f"image_{task_type}"
            suggestions[image_unit] = _collect("image", task_type, image_unit)

    return suggestions


def apply_unit_shifts(memory_dir: Path, out_dir: Path, shifts: Mapping[str, int]) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    variant_dir = out_dir / f"memory_variant_{ts}"
    variant_dir.mkdir(parents=True, exist_ok=True)
    source_images_dir = memory_dir / "images"
    target_images_dir = variant_dir / "images"

    for task_type, filename in PROMPT_MEMORY_FILES.items():
        src = memory_dir / filename
        payload = json.loads(src.read_text(encoding="utf-8"))
        shift = int(shifts.get(f"prompt_{task_type}", 0))
        if shift:
            for item in payload:
                dims = item.get("prompt_dimensions", {})
                if not isinstance(dims, dict):
                    continue
                for dim_payload in dims.values():
                    if not isinstance(dim_payload, dict):
                        continue
                    score = dim_payload.get("score")
                    if isinstance(score, int):
                        dim_payload["score"] = max(1, min(5, score + shift))
        (variant_dir / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for task_type, filename in IMAGE_MEMORY_FILES.items():
        src = memory_dir / filename
        payload = json.loads(src.read_text(encoding="utf-8"))
        shift = int(shifts.get(f"image_{task_type}", 0))
        if shift:
            for item in payload:
                dims = item.get("image_dimensions", {})
                if not isinstance(dims, dict):
                    continue
                for dim_payload in dims.values():
                    if not isinstance(dim_payload, dict):
                        continue
                    score = dim_payload.get("score")
                    if isinstance(score, int):
                        dim_payload["score"] = max(1, min(5, score + shift))
        (variant_dir / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    (variant_dir / "shift_metadata.json").write_text(
        json.dumps({"shifts": dict(shifts), "source_memory_dir": str(memory_dir)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if source_images_dir.exists() and not target_images_dir.exists():
        target_images_dir.symlink_to(source_images_dir, target_is_directory=True)
    return variant_dir


def _write_json(out_path: Path, payload: Mapping[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    expert_root = Path(args.expert_root).resolve()
    novice_root = Path(args.novice_root).resolve()
    paper_root = Path(args.paper_root).resolve()
    memory_dir = Path(args.memory_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.api_key_override:
        os.environ["ATELIEREVAL_API_KEY_OVERRIDE"] = args.api_key_override

    paper_targets = filter_paper_targets(
        parse_paper_targets(paper_root),
        task_types=args.task_types,
        groups=args.groups,
        model_keys=args.model_keys,
    )
    task_description_lookup = load_task_description_lookup()
    discovered = {
        "skilled": discover_generated_content(expert_root, group_key="skilled", task_description_lookup=task_description_lookup),
        "novice": discover_generated_content(novice_root, group_key="novice", task_description_lookup=task_description_lookup),
    }
    sample_plan = build_sample_plan(
        discovered=discovered,
        paper_targets=paper_targets,
        sample_per_type=int(args.sample_per_type),
        num_rounds=int(args.num_rounds),
        seed=int(args.seed),
    )

    prepare_payload = {
        "paper_root": str(paper_root),
        "expert_root": str(expert_root),
        "novice_root": str(novice_root),
        "memory_dir": str(memory_dir),
        "judge_model": args.judge_model,
        "sample_per_type": int(args.sample_per_type),
        "num_rounds": int(args.num_rounds),
        "eval_repeats": int(args.eval_repeats),
        "seed": int(args.seed),
        "prompt_intersections": sample_plan.prompt_intersections,
        "image_intersections": sample_plan.image_intersections,
        "rounds": sample_plan.rounds,
        "target_cells": {
            task_type: {
                group_key: sorted(model_map.keys())
                for group_key, model_map in task_groups.items()
            }
            for task_type, task_groups in paper_targets.items()
        },
    }

    if args.mode == "prepare":
        print(json.dumps(prepare_payload, ensure_ascii=False, indent=2))
        _write_json(out_dir / "prepare_latest.json", prepare_payload)
        return

    predictions = _evaluate_rounds(
        sample_plan=sample_plan,
        discovered=discovered,
        paper_targets=paper_targets,
        memory_dir=memory_dir,
        judge_model=args.judge_model,
        top_k=int(args.top_k),
        eval_repeats=max(1, int(args.eval_repeats)),
        max_retries=int(args.max_retries),
        retry_delays=[int(item) for item in args.retry_delay_seconds],
    )
    fit_report = compute_fit_report(paper_targets=paper_targets, predictions=predictions)
    exemplar_suggestions = build_exemplar_suggestions(
        retrieval_counts=predictions["retrieval_counts"],
        fit_report=fit_report,
    )

    payload: Dict[str, Any] = {
        "prepare": prepare_payload,
        "fit_report": fit_report,
        "predictions": predictions,
        "exemplar_suggestions": exemplar_suggestions,
    }

    if args.mode == "suggest":
        variant_dir = apply_unit_shifts(memory_dir=memory_dir, out_dir=out_dir, shifts=fit_report["unit_shift_suggestion"])
        payload["suggested_variant_dir"] = str(variant_dir)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    _write_json(out_dir / "fit_latest.json", payload)


if __name__ == "__main__":
    main()
