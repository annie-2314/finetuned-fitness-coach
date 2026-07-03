"""Builds a RunPod notebook that exports the fine-tuned adapter to GGUF for Ollama.

Run: python -m scripts.build_gguf_notebook
Output: training/export_gguf_runpod.ipynb
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "training" / "export_gguf_runpod.ipynb"


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)}
def code(t): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": t.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# Export the fine-tuned model to GGUF (for Ollama)

Deploy a GPU pod, open Jupyter, and drag in **`sft-adapter.zip`** (the model you downloaded).
Run every cell. At the end, download the `.gguf` file to your PC.
"""),
    md("## 1. Install Unsloth"),
    code("!pip install -q unsloth"),
    md("## 2. Unzip the adapter (handles the nested folder)"),
    code("""
import os, zipfile, shutil
if os.path.exists("sft-adapter"): shutil.rmtree("sft-adapter")
with zipfile.ZipFile("sft-adapter.zip") as z: z.extractall("sft-adapter")
ADAPTER_PATH = None
for root, _, fs in os.walk("sft-adapter"):
    if "adapter_config.json" in fs: ADAPTER_PATH = root; break
print("ADAPTER_PATH =", ADAPTER_PATH)
assert ADAPTER_PATH, "adapter_config.json not found in sft-adapter.zip"
"""),
    md("## 3. Load + export to GGUF (q4_k_m). Takes ~5-10 min (builds llama.cpp)."),
    code("""
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained(ADAPTER_PATH, max_seq_length=4096, load_in_4bit=True)
model.save_pretrained_gguf("fitness-coach-gguf", tok, quantization_method="q4_k_m")
print("done")
"""),
    md("## 4. Find the .gguf to download"),
    code("""
import os
for f in os.listdir("fitness-coach-gguf"):
    p = os.path.join("fitness-coach-gguf", f)
    print(f, f"({os.path.getsize(p)/1e9:.2f} GB)" if os.path.isfile(p) else "")
print("\\nRight-click the .gguf in the file browser -> Download.")
"""),
    md("""
## Next (on your PC)
1. Install Ollama: https://ollama.com/download
2. Put the `.gguf` next to `backend/Modelfile`, edit its `FROM` line to the real filename.
3. `ollama create fitness-coach -f backend/Modelfile`
4. `ollama run fitness-coach "test"` to verify.
5. Tell Claude it's running — the backend `.env` gets pointed at Ollama.
"""),
]

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}}, "cells": cells}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {OUT}")
