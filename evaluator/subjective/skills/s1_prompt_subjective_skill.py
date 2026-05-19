from __future__ import annotations

from typing import Any, Dict, Optional


class S1PromptSubjectiveSkill:
    name = "S1PromptSubjectiveSkill"

    def run(
        self,
        *,
        route: Any,
        task_id: str,
        task_description: str,
        prompt_text: str,
        sub_mem: Any,
        sub_core: Any,
        memories: Any,
        prompt_clusterer: Any,
        top_k: int,
    ) -> Optional[Dict[str, Any]]:
        try:
            result = sub_mem.evaluate_prompt_subjective_with_memory(
                task_id=task_id,
                task_type=route.task_type,
                task_description=task_description,
                prompt_text=prompt_text,
                model_name=sub_core.API_MODEL,
                memories=memories,
                prompt_clusterer=prompt_clusterer,
                top_k=top_k,
            )
            return result.get("prompt_dimensions")
        except Exception as exc:  # noqa: BLE001
            return {"_error": f"{type(exc).__name__}: {exc}"}
