"""emotion2vec+ (FunASR) utterance embeddings + 9-class scores, and openSMILE eGeMAPSv02 functionals.

Usage: python -I embed_e2v_egemaps.py <which: e2v|egemaps> <out_dir> <list_csv> [e2v_model_id]
Outputs: e2v.npy (N, D + 9)  /  egemaps.npy (N, 88)
"""
import sys

import numpy as np
import pandas as pd

which, out, lst = sys.argv[1:4]
paths = pd.read_csv(lst).path.tolist()
if which == "egemaps":
    import opensmile

    sm = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)
    X = np.stack([sm.process_file(p).values[0] for p in paths]).astype(np.float32)
    np.save(f"{out}/egemaps.npy", X)
    print("done egemaps", X.shape)
else:
    from funasr import AutoModel

    mid = sys.argv[4] if len(sys.argv) > 4 else "emotion2vec/emotion2vec_plus_large"
    m = AutoModel(model=mid, hub="hf", disable_update=True)
    feats = []
    for i, p in enumerate(paths):
        r = m.generate(p, granularity="utterance", extract_embedding=True)[0]
        feats.append(np.concatenate([np.asarray(r["feats"], dtype=np.float32).ravel(), np.asarray(r["scores"], dtype=np.float32)]))
        if i == 0:
            print("labels:", r["labels"], "emb dim:", np.asarray(r["feats"]).size, flush=True)
        if i % 250 == 0:
            print("e2v", i, len(paths), flush=True)
    X = np.stack(feats)
    np.save(f"{out}/e2v.npy", X)
    print("done e2v", X.shape)
