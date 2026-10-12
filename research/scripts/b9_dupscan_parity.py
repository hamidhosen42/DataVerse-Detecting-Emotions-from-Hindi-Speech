"""Parity check for the near-duplicate scan of notebooks/B9_hindi_ser_final.ipynb.

The cached notebook runs are given research/work/dup_groups.csv, and smoke mode groups exact copies only, so neither runs the
notebook's own waveform-NCC scan (used in a fresh FULL run, e.g. on Kaggle). This script executes the notebook's configuration
and data cells in FULL mode with an EMPTY cache (so the scan runs) and checks that the partition of the 1,035 clips into
near-duplicate groups is identical to the research partition, and that the derived quantities (test twins, OOF twin rows)
match.

Usage: python -I b9_dupscan_parity.py <repo_root> <scratch_cache_dir>
"""
import os
import shutil
import sys
import time

import nbformat
import numpy as np
import pandas as pd

P, C = sys.argv[1], sys.argv[2]
shutil.rmtree(C, ignore_errors=True)
os.makedirs(C)
os.environ.update(B9_MODE="full", B9_CACHE=C, B9_FT_RECIPE="none")
code = [c.source for c in nbformat.read(f"{P}/notebooks/B9_hindi_ser_final.ipynb", as_version=4).cells if c.cell_type == "code"]
os.chdir(f"{P}/notebooks")
g = {}
t0 = time.time()
exec(compile(code[0], "notebook-cell-config", "exec"), g)
exec(compile(code[1], "notebook-cell-data", "exec"), g)
assert os.path.exists(f"{C}/dup_groups.csv"), "the scan did not run"
nb = g["GROUP"]
res = pd.read_csv(f"{P}/research/work/dup_groups.csv")
assert (res.Id.values == g["meta"].Id.values).all(), "row order differs"
rg = res.dup_group.values
pairs = pd.Series(list(zip(nb, rg)))
same = pairs.nunique() == len(np.unique(nb)) == len(np.unique(rg))
sizes_nb = pd.Series(nb).value_counts()
print(f"notebook scan: {len(np.unique(nb))} groups, {(sizes_nb > 1).sum()} with >=2 clips covering {int(sizes_nb[sizes_nb > 1].sum())} clips | "
      f"research: {len(np.unique(rg))} groups | identical partition: {same} | {time.time() - t0:.0f} s")
if not same:
    ct = pd.crosstab(nb, rg)
    print("rows in mismatching groups:", int(((ct > 0).sum(1) > 1).sum() + ((ct > 0).sum(0) > 1).sum()))
print(f"test clips with a training twin: {int(g['TEST_TWIN'].sum())} | OOF rows with twin: {g['HAS_TWIN'].mean():.3f}")
print("DUPSCAN PARITY", "OK" if same else "MISMATCH")
