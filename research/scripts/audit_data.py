"""Data audit: per-file audio metadata, exact duplicates, ID-order structure.

Usage: python -I audit_data.py <data_root (SRCASW-BhavVaani)> <out_dir>
Writes <out_dir>/meta.csv and <out_dir>/exact_duplicates.csv; prints a report.
"""
import hashlib
import sys

import numpy as np
import pandas as pd
import soundfile as sf

R, OUT = sys.argv[1], sys.argv[2]
tr = pd.read_csv(f"{R}/train.csv")
te = pd.read_csv(f"{R}/test.csv")
ss = pd.read_csv(f"{R}/sample_submission.csv")

print("== class distribution\n", pd.concat([tr.emotion.value_counts(), tr.emotion.value_counts(normalize=True).round(3)], axis=1))
print("Id overlap train/test:", len(set(tr.Id) & set(te.Id)))
print("Ids cover 1..1035 exactly:", sorted(set(tr.Id) | set(te.Id)) == list(range(1, 1036)))
print("filename == audio_{Id:06d}.wav:", (pd.concat([tr, te]).filename == pd.concat([tr, te]).Id.map(lambda i: f"audio_{i:06d}.wav")).all())
print("sample_submission order == test order:", (ss.Id.values == te.Id.values).all(), "| sample labels:", ss.emotion.unique())

rows = []
for split, df in (("train", tr), ("test", te)):
    for _, r in df.iterrows():
        p = f"{R}/{split}/{r.filename}"
        info = sf.info(p)
        y, sr = sf.read(p, dtype="float32", always_2d=True)
        mono = y.mean(1)
        raw = open(p, "rb").read()
        rows.append(dict(
            split=split, Id=r.Id, emotion=r.get("emotion", None), sr=sr, channels=info.channels,
            subtype=info.subtype, n_samples=len(mono), dur=len(mono) / sr, nbytes=len(raw),
            file_md5=hashlib.md5(raw).hexdigest(), pcm_md5=hashlib.md5(mono.tobytes()).hexdigest(),
            peak=float(np.abs(mono).max()), rms=float(np.sqrt((mono ** 2).mean())),
            clip_frac=float((np.abs(mono) > 0.999).mean()), silence_frac=float((np.abs(mono) < 1e-3).mean()),
        ))
m = pd.DataFrame(rows)
m.to_csv(f"{OUT}/meta.csv", index=False)

print("\n== audio format")
print(m.groupby("split")[["sr", "channels"]].agg(lambda s: s.value_counts().to_dict()))
print(pd.crosstab(m.split, m.subtype))
print(pd.crosstab(m.emotion, m.subtype))
print("\n== duration by split\n", m.groupby("split").dur.describe().round(2))
print("\n== per-class medians (train)\n", m[m.split == "train"].groupby("emotion")[["dur", "rms", "peak", "silence_frac"]].median().round(4))
print("\n== duration by encoding\n", m.groupby("subtype").dur.describe().round(2))

g = m.groupby("pcm_md5").agg(n=("Id", "size"), splits=("split", lambda s: ",".join(sorted(s))),
                             labels=("emotion", lambda s: ",".join(sorted(map(str, s)))),
                             ids=("Id", lambda s: " ".join(map(str, sorted(s)))))
d = g[g.n > 1].sort_values("splits")
d.to_csv(f"{OUT}/exact_duplicates.csv")
print(f"\n== exact PCM duplicate groups: {len(d)} (rows involved {d.n.sum()})")
print(d.splits.value_counts().to_string())
print(d.to_string(index=False))

trs = m[m.split == "train"].sort_values("Id")
lab = trs.emotion.values
same = (lab[1:] == lab[:-1]).mean()
rng = np.random.default_rng(0)
perm = [(p[1:] == p[:-1]).mean() for p in (rng.permutation(lab) for _ in range(2000))]
print(f"\n== adjacent-Id same-label rate {same:.3f} vs shuffled {np.mean(perm):.3f} +- {np.std(perm):.3f}")
print(pd.crosstab(pd.qcut(trs.Id, 5, labels=False), trs.emotion))
