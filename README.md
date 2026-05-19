# AtelierEval

AtelierEval is a benchmark and evaluation toolkit for studying prompting ability in text-to-image generation. It contains 360 tasks across three task families and the public release code for the human UI, baseline generation runners, and AtelierJudge evaluator.

## Task Families

- **Open-ended (OE):** creative briefs where the prompt should translate user intent into a high-quality text-to-image instruction.
- **Constrained (CO):** tasks with explicit visual, layout, quantity, and text constraints.
- **Imitation (IM):** reference-image imitation tasks where a prompt should reproduce the key content and style of a provided image.

## Repository Layout

```text
dataset/                  Benchmark task JSON files and IM reference images.
ui/                       Human evaluation / task interface.
generator/1_llm_runner/   LLM prompt generation runner.
generator/2_tti_runner/   Text-to-image generation runner.
evaluator/                AtelierJudge evaluator code, configs, skills, and memories.
results/                  Camera-ready result tables and figure source data.
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some evaluator and generation paths call external LLM or T2I APIs. Configure your own API endpoint and key through the provided config templates or environment variables. This repository does not include API keys.

## Dataset

The released task files are:

- `dataset/co_120.json`
- `dataset/im_120.json`
- `dataset/oe_120.json`
- `dataset/category_statistics.json`
- `dataset/gt_image/`

Each task JSON contains task IDs and task-specific fields used by the UI, generator, and evaluator.

## Human UI

```bash
cd ui
pip install -r requirements.txt
python app.py
```

The UI loads benchmark tasks from the local `dataset/` directory by default.

## Generation Runners

LLM prompt generation:

```bash
cd generator/1_llm_runner
python llm_runner.py --config configs/experiment_api.json --task-types co,oe,im
```

Text-to-image generation:

```bash
cd generator/2_tti_runner
python tti_runner.py --config configs/experiment_api.json --task-types co,oe,im
```

Edit the config files before running. Keep private keys outside version control.

## Evaluator

The evaluator includes:

- Safety skills in `evaluator/safety/`
- Objective checklist skills in `evaluator/objective/`
- Subjective memory-augmented skills in `evaluator/subjective/`
- Runtime and skill configs in `evaluator/configs/`

The subjective branch depends on the released memory bank under:

```text
evaluator/subjective/memory/
```

Do not remove this directory if you want to run the full AtelierJudge evaluator. Without it, only the non-memory branches can be adapted.

## Results

`results/` contains the camera-ready tables, figure data, and replacement snippets used for the final paper version.

## License

Code is released under the MIT License. Dataset, memory assets, paper assets, and result files are released under CC BY 4.0 unless a file states otherwise. See `LICENSE` for details.

## Citation

If you use AtelierEval, please cite the paper. A final venue-specific citation will be updated when available.

```bibtex
@misc{ateliereval2026,
  title = {AtelierEval: Evaluating Human and MLLM Prompting Ability for Text-to-Image Generation},
  year = {2026},
  note = {Camera-ready release}
}
```
