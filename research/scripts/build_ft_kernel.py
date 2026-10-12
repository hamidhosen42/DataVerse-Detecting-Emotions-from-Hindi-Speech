"""Build a private Kaggle notebook that fine-tunes a non-WavLM SSL backbone with the public
kunalinfinite/01-dataverse-training recipe (same folds, same hyper-parameters), then runs the public
02-dataverse-inference code on the new checkpoints.

Patches (minimal): WavLMConfig/WavLMModel -> AutoConfig/AutoModel so wav2vec2/XLS-R/HuBERT load;
NOTEBOOK_ARGS set for this run. Everything else is the public code unchanged.

Usage: python -I build_ft_kernel.py <k01.ipynb> <k02.ipynb> <out_dir> <kaggle_user> <slug> <hf_model> <backbone_lr>
"""
import json
import sys
from pathlib import Path

k01, k02, out, user, slug, model, lr = sys.argv[1:8]


def src(p):
    nb = json.load(open(p))
    return "".join("".join(c["source"]) for c in nb["cells"])


def sub(text, old, new, count=1):
    assert text.count(old) >= 1, f"pattern not found: {old[:60]}"
    return text.replace(old, new) if count == 0 else text.replace(old, new, count)


t = src(k01)
t = sub(t, "from transformers import WavLMConfig, WavLMModel, Wav2Vec2FeatureExtractor",
        "from transformers import WavLMConfig, WavLMModel, Wav2Vec2FeatureExtractor, AutoConfig, AutoModel")
t = sub(t, "config = WavLMConfig.from_pretrained(source,", "config = AutoConfig.from_pretrained(source,")
t = sub(t, 'if config.model_type != "wavlm" or getattr(config, "add_adapter", False):',
        'if config.model_type not in ("wavlm", "wav2vec2", "hubert") or getattr(config, "add_adapter", False):')
t = sub(t, "backbone = WavLMModel.from_pretrained(", "backbone = AutoModel.from_pretrained(")
t = sub(t, "self.backbone = backbone if backbone is not None else WavLMModel(config)",
        "self.backbone = backbone if backbone is not None else AutoModel.from_config(config)")
args01 = ('import glob as _glob\n'
          '_folds = sorted(_glob.glob("/kaggle/input/**/dataverse_models/train_with_folds.csv", recursive=True))\n'
          f'NOTEBOOK_ARGS = ["--models", "{model}", "--backbone-lr", "{lr}", "--output-dir", "/kaggle/working/ft_models"]'
          ' + (["--folds-csv", _folds[0]] if _folds else [])\n'
          'print("folds csv:", _folds[:1], flush=True)\n')
t = sub(t, "NOTEBOOK_ARGS = []\n", args01)

u = src(k02)
u = sub(u, "from transformers import WavLMConfig, WavLMModel, Wav2Vec2FeatureExtractor",
        "from transformers import WavLMConfig, WavLMModel, Wav2Vec2FeatureExtractor, AutoConfig, AutoModel")
u = sub(u, 'config = WavLMConfig.from_dict(specification["model_config"])',
        'config = AutoConfig.for_model(**specification["model_config"])')
u = sub(u, "self.backbone = backbone if backbone is not None else WavLMModel(config)",
        "self.backbone = backbone if backbone is not None else AutoModel.from_config(config)")
start = u.index("NOTEBOOK_ARGS = [")
end = u.index("]", start) + 1
u = u[:start] + ('NOTEBOOK_ARGS = ["--model-dir", "/kaggle/working/ft_models", "--output-dir", "/kaggle/working/ft_inference", '
                 '"--submission-path", "/kaggle/working/submission_ft.csv"]') + u[end:]

cell = lambda s: {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}
nb = {"cells": [cell(t), cell(u)], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                                "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
o = Path(out)
o.mkdir(parents=True, exist_ok=True)
(o / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1))
meta = {"id": f"{user}/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "keywords": [],
        "dataset_sources": [], "kernel_sources": ["kunalinfinite/01-dataverse-training"],
        "competition_sources": ["dataverse-detecting-emotions-from-hindi-speech"], "model_sources": [], "machine_shape": "NvidiaTeslaT4"}
(o / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
(o / "train_patched.py").write_text(t)
(o / "infer_patched.py").write_text(u)
print("built", o, "| train chars", len(t), "| infer chars", len(u))
