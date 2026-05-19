# Atelier Evaluator (Skill-Agent)

This folder contains the unified evaluator runner for AtelierJudge with a skill-agent execution style.

## Pipeline

The default pipeline follows:

1. `Safety` (prompt safety + image safety)
2. `System 1` subjective evaluation (memory-augmented)
3. `System 2` objective checklist evaluation (prompt/image binary checks)
4. Aggregation into final per-task results

Execution order is:

- Run safety first.
- If safety passes, run `System 1` and `System 2` in parallel.
- Inside each system, run prompt/image skills according to task routing (`co`, `im`, `oe`).

## Main Entry

- `run_evaluator.py`: formal runner.

## Skill Routing

- `router.py` contains:
  - task routing rules,
  - dynamic skill loading (via `configs/config_loader.py`),
  - skill-level execution trace generation,
  - safety/S1/S2 orchestration helpers.

## Skill Configs

- `configs/safety_prompt_skill.json`
- `configs/safety_image_skill.json`
- `configs/s1_prompt_subjective_skill.json`
- `configs/s1_image_subjective_skill.json`
- `configs/s2_prompt_objective_skill.json`
- `configs/s2_image_objective_skill.json`
- `configs/config_loader.py`
- `configs/runtime/safety_prompt_runtime.json`
- `configs/runtime/safety_image_runtime.json`
- `configs/runtime/subjective_runtime.json`
- `configs/runtime/objective_runtime.json`
- `configs/templates/subjective_prompt_templates.json`
- `configs/templates/objective_check_templates.json`

## Skill Files

### Safety skills
- `safety/skills/safety_prompt_skill.py`
- `safety/skills/safety_image_skill.py`

### Subjective (System 1) skills
- `subjective/skills/s1_prompt_subjective_skill.py`
- `subjective/skills/s1_image_subjective_skill.py`

### Objective (System 2) skills
- `objective/skills/s2_prompt_objective_skill.py`
- `objective/skills/s2_image_objective_skill.py`

## Core Evaluator Modules

These keep the original evaluator logic (not rewritten):

- Subjective core: `subjective/subjective_eval_core.py`
- Subjective memory: `subjective/s1_subjective_memory_skill_engine.py`
- Objective checker: `objective/skills/s2_prompt_objective_skill.py`
- Objective utils: `objective/obj_checker_utils.py`
- Safety prompt/image evaluators:
  - `safety/skills/safety_prompt_skill.py`
  - `safety/skills/safety_image_skill.py`

## Output

- Run outputs are saved under `eval_results/`.
- Latest run summary: `eval_results/latest.json`.
- Each task includes `execution_trace` entries for skill-level observability.

## Run

```bash
conda run --no-capture-output -n ateliereval \
python evaluator/run_evaluator.py \
  --generated-root <GENERATED_RESULTS_ROOT_IN_REPO>
```

Use `--prepare-only` to verify case discovery without API evaluation.
Use `--safety-decision-mode both_fail|any_fail` to control safety strictness (`any_fail` is the default, strict mode).
