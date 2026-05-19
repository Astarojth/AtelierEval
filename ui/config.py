# config.py
from pathlib import Path

import gradio as gr

# --- Constants ---
OPEN_ENDED_EXAM_SIZE = 5
CONSTRAINED_EXAM_SIZE = 5
IMIT_EXAM_SIZE = 5

UI_ROOT = Path(__file__).resolve().parent
REPO_ROOT = UI_ROOT.parent
LOCAL_DATASET_ROOT = (REPO_ROOT / "dataset").resolve()

LOCAL_DATA_FILES = {
    "open_ended": "oe_120.json",
    "constrained": "co_120.json",
    "imit": "im_120.json",
}

# Theme configuration
THEME = gr.themes.Soft()
CSS = """
.gradio-radio label { display: block !important; }
"""
TITLE = "AtelierEval Assessment"
