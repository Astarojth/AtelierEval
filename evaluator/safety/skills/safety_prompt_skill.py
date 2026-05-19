from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai import OpenAI

try:
    from ...configs.api_runtime import apply_api_spec_to_config
except ImportError:
    from configs.api_runtime import apply_api_spec_to_config  # type: ignore


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return apply_api_spec_to_config(cfg, config_path=config_path)


def resolve_config_path(config_path: Path, configured_path: str) -> Path:
    candidate = Path(configured_path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (config_path.parent / candidate).resolve()


def load_prompt_template(prompt_template_path: Path) -> str:
    if not prompt_template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_template_path}")
    return prompt_template_path.read_text(encoding="utf-8")


class RateLimiter:
    """Simple QPS limiter for safety requests."""

    def __init__(self, qps: float):
        self.qps = max(0.0, qps)
        self.min_interval = 0.0 if self.qps <= 0 else 1.0 / self.qps
        self._last = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        now = time.time()
        elapsed = now - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.time()


def call_text_safety(
    client: OpenAI,
    model: str,
    task_id: str,
    prompt_text: str,
    prompt_template: str,
    temperature: float,
    top_p: Optional[float],
    max_tokens: int,
    max_retries: int,
    backoff_base: float,
    backoff_cap: float,
    limiter: RateLimiter,
) -> Tuple[bool, str, List[str]]:
    user = prompt_template.format(task_id=task_id, prompt_text=prompt_text)
    last_err: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            limiter.wait()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content or ""
            parsed = json.loads(content)
            passed = bool(parsed.get("passed", False))
            reason = str(parsed.get("reason", "")).strip() or ("OK" if passed else "Blocked")
            tags = parsed.get("policy_tags", [])
            if not isinstance(tags, list):
                tags = []
            return passed, reason, [str(tag) for tag in tags]
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            sleep_s = min(backoff_cap, backoff_base * (2**attempt))
            sleep_s = sleep_s + random.uniform(0, 0.25 * sleep_s)
            time.sleep(sleep_s)
    raise RuntimeError(f"API_ERROR after retries: {last_err}")


class SafetyPromptSkill:
    name = "SafetyPromptSkill"

    def run(
        self,
        *,
        cfg: Any,
        sample: Any,
        safety_runtime: Any,
        run_prompt_safety_fn: Callable[..., Dict[str, Any]],
    ) -> Dict[str, Any]:
        return run_prompt_safety_fn(cfg=cfg, sample=sample, safety_runtime=safety_runtime)
