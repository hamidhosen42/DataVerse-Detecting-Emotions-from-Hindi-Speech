"""Fresh-path parity check for the five frozen Whisper families of notebooks/B9_hindi_ser_final.ipynb.

The cached notebook runs reuse the research features (research/work/emb/<tag>.npy). This script executes the notebook's own
configuration and data cells plus its `whisper_features` function (notebook loading code, pinned model revisions) on a few
clips for EVERY frozen model, and compares the pooled features with the research features behind all reported numbers.
Clips: the 2 shortest, the 2 longest, and the first training and test clips.

Usage: python -I b9_frozen_parity.py <repo_root> <scratch_cache_dir>
"""
import ast
import os
import shutil
import sys
import time

import nbformat
import numpy as np

P, C = sys.argv[1], sys.argv[2]
os.makedirs(C, exist_ok=True)
shutil.copy(f"{P}/research/work/dup_groups.csv", f"{C}/dup_groups.csv")
os.environ.update(B9_MODE="full", B9_CACHE=C, B9_FT_RECIPE="none")
code = [c.source for c in nbformat.read(f"{P}/notebooks/B9_hindi_ser_final.ipynb", as_version=4).cells if c.cell_type == "code"]
os.chdir(f"{P}/notebooks")                       # the notebook resolves ../data from here
g = {}
exec(compile(code[0], "notebook-cell-config", "exec"), g)
exec(compile(code[1], "notebook-cell-data", "exec"), g)
defs = [n for n in ast.parse(code[2]).body if isinstance(n, ast.FunctionDef) and n.name == "whisper_features"]
assert len(defs) == 1
exec(compile(ast.Module(body=defs, type_ignores=[]), "notebook-cell-frozen", "exec"), g)
order = np.argsort(g["DUR"])
rows = np.unique(np.concatenate([order[:2], order[-2:], g["TR"][:1], g["TE"][:1]]))
print(f"parity rows {rows.tolist()} | durations {[round(float(g['DUR'][r]), 2) for r in rows]} | device {g['DEV']}", flush=True)
worst = 0.0
for tag, (mid, recipe) in g["FROZEN"].items():
    t0 = time.time()
    F = g["whisper_features"](mid, f"{tag}_parity", rows)        # only these rows are filled; nothing is saved
    R = np.load(f"{P}/research/work/emb/{tag}.npy", mmap_mode="r")
    a, b = np.asarray(F[rows], np.float32), np.asarray(R[rows], np.float32)
    assert a.shape == b.shape, (a.shape, b.shape)
    xa, xb = (a[:, [24, 32]], b[:, [24, 32]]) if recipe == "cat24+32" else (a[:, 22:25].mean(1), b[:, 22:25].mean(1))
    rel_all = float(np.abs(a - b).max() / np.abs(b).max())
    rel_lr = float(np.abs(xa - xb).max() / np.abs(xb).max())
    cos = float(np.min([np.dot(u.ravel(), v.ravel()) / np.linalg.norm(u) / np.linalg.norm(v) for u, v in zip(xa, xb)]))
    worst = max(worst, rel_lr)
    print(f"{tag:7s} {mid} @ {g['REVISIONS'][mid][:12]}: features {a.shape[1:]} | max rel diff all layers {rel_all:.2e} | "
          f"LR-input layers {rel_lr:.2e} | min cosine {cos:.6f} | {time.time() - t0:.0f} s", flush=True)
print(f"PARITY {'OK' if worst < 1e-2 else 'MISMATCH'}: worst LR-input relative difference {worst:.2e} (fp16 storage)")
