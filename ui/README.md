# AtelierEval UI

Gradio-based assessment UI for three task types:

- `open_ended` (mapped from `dataset/oe_120.json`)
- `constrained` (mapped from `dataset/co_120.json`)
- `imit` (mapped from `dataset/im_120.json`)

By default, UI runs in **local data mode** and reads from `../dataset`.
Task loading is fixed to repository `dataset/` only.

## Run Locally

```bash
cd UI
pip install -r requirements.txt
python app.py
```

## Dataset Source

Task files are loaded exclusively from:

- `<repo>/dataset/oe_120.json`
- `<repo>/dataset/co_120.json`
- `<repo>/dataset/im_120.json`
