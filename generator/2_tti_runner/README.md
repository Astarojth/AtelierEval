# Unified TTI Runner (Anonymous Template)

Single entrypoint for batch image generation from JSON tasks.

Supported backends:
- Local backend
- API backend

## Folder Structure

Keep these folders in this project:

- `configs/`
- `input/co_input/`
- `input/oe_input/`
- `input/im_input/`
- `output/co_output/`
- `output/oe_output/`
- `output/im_output/`

Input JSON files must be placed under `input/*_input/`.

## Config Files

- `configs/experiment_api.json`
- `configs/experiment_local.json`

These are anonymized templates. All sensitive values and non-`input/output` paths are placeholders.

## Run

```bash
cd <RUNNER_ROOT>
python tti_runner.py --config configs/experiment_api.json
python tti_runner.py --config configs/experiment_local.json
```

Optional overrides:

```bash
python tti_runner.py --config configs/experiment_api.json --task-types co,oe
python tti_runner.py --config configs/experiment_api.json --disable-resume
```

## Placeholder Policy

In config files:
- `input_dirs` and `output_dirs` use real project folders under `input/` and `output/`.
- All other path-like or sensitive fields must remain placeholders, for example:
  - `<API_BASE_URL>`
  - `<API_KEY>`
  - `<LOCAL_MODEL_PATH>`
  - `<OUTPUT_MODEL_TAG>`
  - `<DEVICE>`
  - `<TORCH_DTYPE>`
