from __future__ import annotations

from typing import Any, Callable, Dict


class S2ImageObjectiveSkill:
    name = "S2ImageObjectiveSkill"

    def run(self, *, run_image_objective_fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        return run_image_objective_fn()
