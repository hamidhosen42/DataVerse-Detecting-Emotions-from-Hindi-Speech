## Summary

Team **Hack2Publish** (hosen42, foysalemonshanto). The final model averages the probabilities of six families:
* **five frozen Whisper encoders**, each with a logistic-regression head on masked mean/std-pooled hidden states;
* **one Whisper-large-v2 fine-tuned with LoRA**.

The main lessons from this competition concern the data: near-duplicate recordings, and how they inflate naive
cross-validation. The model choices follow from them.

| Submission | Public | Private |
|---|---|---|
| **B8: LoRA Whisper-large-v2 + 5 frozen Whisper families** (final) | **0.94230** | **0.92730** |
| B6: 5 frozen Whisper families + public WavLM fine-tune ensemble | 0.93060 | 0.92691 |
| Public WavLM Base+/Large fine-tune baseline (for reference) | 0.87918 | 0.84003 |

Final standing: **13th of 54 teams** on the private leaderboard.

## 1. What the data looks like

* 826 training and 209 test clips (16 kHz, 1–11 s), four balanced classes: angry, happy, neutral, sad.
* **Near-duplicate recordings are common.** We computed the waveform normalised cross-correlation (max over lags) between
  clips of similar length:
  * 257 pairs score ≥ 0.95; no pair falls between 0.5 and 0.95, so the split is clean.
  * They form 227 groups covering 467 of the 1,035 clips.
  * **78 of the 209 test clips (37%) have a near-copy in the training set.** Labels inside training groups agree in all but
    one group.
* One exact-duplicate pair carries two different labels. We dropped it from training, leaving 824 rows.

Consequence: under ordinary stratified k-fold, about 27% of validation rows have a near-copy in the training folds, so CV is
optimistic. We therefore always report a **non-twin** score as well, and use **duplicate-group-disjoint folds** (Track B)
to choose models.

## 2. Model

**Frozen Whisper families.** We used five public checkpoints, all Apache-2.0, at pinned revisions:
* `openai/whisper-large-v2`
* `openai/whisper-large-v3`
* `ARTPARK-IISc/whisper-large-v3-vaani-hindi`
* `vasista22/whisper-hindi-large-v2`
* `openai/whisper-medium`

For each clip and each encoder layer we take the mean and standard deviation over **only the frames that cover real audio**.
Whisper pads every input to 30 s; pooling over the padding washes out the signal. The heads:
* Large models: layers 24 and 32 concatenated (dimension 5,120). Medium: the average of layers 22–24.
* `StandardScaler` + `LogisticRegression(C=0.05, class_weight="balanced")` per family, all fitted inside the folds.

| family | 5-fold OOF macro-F1 | duplicate-group-disjoint CV |
|---|---|---|
| Whisper-large-v2 | 0.936 | 0.900 |
| Whisper-large-v3 | 0.920 | 0.895 |
| Hindi Whisper-v3 (Vaani) | 0.929 | 0.892 |
| Hindi Whisper-large-v2 | 0.914 | 0.884 |
| Whisper-medium | 0.902 | 0.848 |
| **equal-weight blend of the five** | 0.943 | **0.912** |

For comparison, wav2vec2, XLS-R, WavLM, w2v-BERT and emotion2vec features stayed at or below 0.82 on the same group-disjoint
folds. Whisper's encoder was by far the best representation for this data.

**Fine-tuned family.** Whisper-large-v2 encoder with LoRA:
* r = 16, α = 32, on the attention q/v projections;
* masked mean/std pooling of layers 24 and 32, then a linear head;
* 8-s crops, 6 epochs, OneCycle schedule, label smoothing 0.1;
* fixed epochs, no early stopping on the validation fold;
* five fold models, whose test predictions are averaged.

On its own it scores OOF 0.9145. Its errors are only weakly correlated with those of the strongest frozen family
(correctness correlation ≈ 0.5), which is why it helps the blend.

**Ensemble.** Equal-weight average of the six families' probabilities; no weights were tuned on OOF. As a safety net, a
test clip that is a near-copy of training clips with one consistent label receives that label. This uses only training
labels; in our final submission it changed 0 predictions.

## 3. Validation protocol

* Track A: fixed 5 folds, reporting OOF macro-F1 overall, on non-twin rows, per class and per fold.
* Track B: `StratifiedGroupKFold` on the near-duplicate groups, with 3 seeds.
* Every candidate is compared with the current best using a **paired bootstrap over duplicate groups**.
* Keep rule: gain ≥ +0.005 with P(gain > 0) ≥ 0.9, **and** at least one test prediction must change.
* Because we tried many blends, we also ran White's Reality Check over all candidates.

## 4. What did not help (useful negative results)

* **Native 30-s fine-tuning.**
  * LoRA on layers 17–32 at Whisper's full 30-s context is a better single model than the 8-s version (+0.010 OOF, better in
    5/5 folds). A second training seed scored 0.930 on its own; averaging the two seeds gave 0.925.
  * In the ensemble, however, it changed **no** test prediction.
  * LP-FT (head initialised from the linear probe) scored *below* its own probe.
* **The ensemble saturates.** With equal weights, the five frozen families alone fix 207 of the 209 test predictions whatever
  the fine-tuned member is. No test clip had a top-2 probability margin below 0.05.
* **Other heads and classifiers:** MLP, attention pooling, learned layer weights, PCA + RBF-SVM and cosine kNN were all equal
  or worse than logistic regression. So were other layer combinations (16+24+32, 28+32) and C = 0.3.
* **Other ideas:** an ASR-transcript text branch, pseudo-labelling, external acted Hindi data, speed/noise augmentation,
  per-recording-batch normalisation and class-bias tuning gave no reliable gain.

## 5. Public vs private

* Our CV-based ordering held on the private set: B8 ≥ B6 on both public and private.
* The 0.94 → 0.93 drop from public to private sits within the expected noise. About 84 public clips means one clip moves the
  score by roughly 0.01.
* The non-twin OOF score (0.921) was a closer guide to the private result than the full OOF (0.943).

## 6. Reproducibility

* One notebook covers both training and inference. It finds Kaggle or local paths, has smoke and full modes, pins model
  revisions, caches every stage, and asserts the submission format.
* With cached artifacts it reproduces our final submission byte-for-byte.
* It was tested with transformers 4.57 and 5.19.
* It uses only the competition data and public pretrained checkpoints: no external labelled data and no hand labelling.

Thanks to the hosts for the dataset and to the authors of the public WavLM baseline notebooks.
