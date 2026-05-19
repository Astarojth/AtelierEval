from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional


DEFAULT_SKILL_CONFIG_FILES: Dict[str, str] = {
    "SafetyPromptSkill": "safety_prompt_skill.json",
    "SafetyImageSkill": "safety_image_skill.json",
    "S1PromptSubjectiveSkill": "s1_prompt_subjective_skill.json",
    "S1ImageSubjectiveSkill": "s1_image_subjective_skill.json",
    "S2PromptObjectiveSkill": "s2_prompt_objective_skill.json",
    "S2ImageObjectiveSkill": "s2_image_objective_skill.json",
}


def resolve_path_from_config_dir(config_dir: str | Path, rel_or_abs: str) -> Path:
    candidate = Path(rel_or_abs)
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path(config_dir).resolve() / candidate).resolve()


def load_json_config(path: str | Path) -> dict:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def load_skill_registry_from_config_dir(
    config_dir: str | Path,
    config_files: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, str]]:
    base_dir = Path(config_dir).resolve()
    files = config_files or DEFAULT_SKILL_CONFIG_FILES
    registry: Dict[str, Dict[str, str]] = {}

    for skill_name, file_name in files.items():
        path = base_dir / file_name
        payload = load_json_config(path)
        module_name = str(payload.get("module", "")).strip()
        class_name = str(payload.get("class", "")).strip()
        if not module_name or not class_name:
            raise ValueError(f"Invalid skill config in {path}: missing module/class")
        registry[skill_name] = {"module": module_name, "class": class_name}

    return registry
