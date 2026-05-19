# LLM Runner

Batch-generate prompts for:
- `co_120.json`
- `oe_120.json`
- `im_120.json`

The runner:
- reads tasks from repository `dataset/*_120.json`
- loads prompt templates from a separate JSON file
- calls one API model for all task types
- writes 3 output JSON files under `./output`

## Files

- `llm_runner.py`: main script
- `configs/experiment_api.json`: API and path config (anonymized placeholders by default)
- `prompts/prompt_templates.json`: 3 prompt templates (`co` / `oe` / `im`)

## Run

```bash
cd <RUNNER_ROOT>
python llm_runner.py
```

Default config path is built in:
- `configs/experiment_api.json`

If needed, you can still override:

```bash
python llm_runner.py --config configs/experiment_api.json
```

Optional:

```bash
python llm_runner.py --task-types co,oe,im
python llm_runner.py --max-items-per-type 2
python llm_runner.py --disable-resume
```
