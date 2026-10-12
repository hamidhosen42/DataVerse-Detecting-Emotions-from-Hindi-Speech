# B9 experiment report — DataVerse: Detecting Emotions from Hindi Speech

*Status: FINAL (2026-10-12). **Decision: no B9 candidate reliably improves B8, so B8 (public LB 0.94230) stays the
recommended submission.** Nothing was submitted to Kaggle in this round.*

Evidence labels: **MEASURED** (computed in this round from files/runs in this repo), **VERIFIED-CACHED** (re-computed from cached
artifacts, no re-training), **EARLIER CLAIM** (from the research report, not re-run this round), **HYPOTHESIS**, **UNTESTED** (code
written but not executed in that configuration).

## 0. Summary

* **Baseline:** B8 reproduces exactly from cached artifacts (0 / 209 rows, identical md5). The new notebook reproduces it in
  cached FULL mode.
* **Priority experiment, two variants:** native 30-s Whisper-large-v2 LoRA with masked pooling, each run end-to-end for 5 folds
  (≈ 13 h on an Apple M4 Pro).
  * **LP-FT at 30 s** scores *below its own linear probe* (−0.0060, P(gain>0) = 0.06). Rejected.
  * **LoRA at 30 s** (B8's recipe, LoRA in layers 17–32) is a better fine-tuned member than B8's 8-s LoRA: +0.0100 OOF, better in
    5 / 5 folds, P = 0.90. In the ensemble, though, it adds only +0.0049 [0.000, +0.012] (P = 0.95; Reality Check p = 0.165).
    Its test predictions are **identical to B8's**.
* **Blends:** none of the 8 candidates passes the pre-declared keep rule. The three that would change test predictions are coin
  flips (P ≤ 0.62).
* **Deliverable:** `notebooks/B9_hindi_ser_final.ipynb` defaults to the B8 composition. `lora30top` / `lpft30` can be selected
  with one variable.

## 1. Baseline audit (B8)

| Check | Result | Label |
|---|---|---|
| B8 test predictions rebuilt from cached family probabilities (`lora_probs_wlv2.npz`, `family_probs.npz`) | **0 / 209** rows differ from the submitted CSV; md5 of `my_submissions/0.94230_…/submission.csv` == `research/submissions/sub_B8.csv` (`696f1431…`) | VERIFIED-CACHED |
| B8 OOF Macro-F1 (public-pipeline folds, 824 rows) | **0.9433**; non-twin 0.9211; twin 1.000; per fold 0.9139 / 0.9692 / 0.9470 / 0.9496 / 0.9346 | VERIFIED-CACHED |
| B8 per class (OOF) | F1 angry 0.953, happy 0.945, **neutral 0.925**, sad 0.950 | VERIFIED-CACHED |
| Class order | `["angry","happy","neutral","sad"]` everywhere (= sorted labels; every `LogisticRegression.classes_` asserted) | MEASURED |
| IDs / order | 209 test Ids in identical order in `test.csv`, `sample_submission.csv` and all probability files | MEASURED |
| Folds | 5 public-pipeline folds (166/163/164/163/168); 824 = 826 train clips − exact-duplicate pair Id 75/636 (conflicting labels) | MEASURED |
| Twin safety net | changes **0** B8 predictions | MEASURED |
| B9 notebook, FULL mode, fed cached artifacts (`FT_RECIPE=lora8`), final file run through a Jupyter kernel | re-fits the 5 frozen LR families inside the notebook (identical OOF to research, e.g. lr_wlv2 0.9362), ensemble OOF **0.9433**, `submission.csv` md5 == submitted B8 | VERIFIED-CACHED |
| B8's fine-tuned member (8-s LoRA) contribution | ensemble − frozen-only blend = **+0.0000** [−0.0073, +0.0065], P(>0) 0.54 | MEASURED |

"Cached reproduction" means probabilities from earlier runs are re-combined; the 8-s LoRA itself was **not** re-trained this round.
The training code was run from scratch in smoke mode (§8). Audit log: `research/work/b9_audit.txt`.

## 2. Validation protocol

* **Track A (comparable with the baseline):** the 5 public-pipeline folds. Reported: OOF Macro-F1 overall, **non-twin**, per
  class and per fold, plus confusion matrices. "Twin" OOF rows are those whose near-duplicate has a copy in the training folds:
  26.9 % of OOF rows vs 37.3 % of test rows.
* **Track B (novel recordings):** `StratifiedGroupKFold(5)` with near-duplicate groups as groups, 3 seeds (frozen families).
* **Paired dup-group bootstrap** (3,000 resamples of near-duplicate groups) of every candidate against B8.
* **Decision rules (pre-declared):**
  * Keep a candidate only if the mean paired gain is ≥ +0.005 **and** P(gain > 0) ≥ 0.9.
  * About 45 blend candidates were tried during the research, so a candidate must also reach **P ≥ 0.99** to count as an upgrade
    of the submitted B8.
  * It must change at least one test prediction; otherwise the choice makes no difference.
* **No leakage into the folds:**
  * Scaling, LR fitting, linear-probe initialisation and LoRA training all happen inside the training part of each fold.
  * Epochs are fixed; there is no early stopping and no checkpoint selection on validation folds.
  * Ensembles use equal weights; no weights are tuned on OOF.
* **Known optimism:** the frozen families' layer recipe (24+32) and their membership were chosen on the same 826 training clips
  (Track B). This applies equally to B8 and every candidate, so it largely cancels in paired comparisons.

## 3. Near-duplicate finding — re-verified (MEASURED)

* 257 pairs have waveform max-NCC ≥ 0.95: train–train 162, train–test 86, test–test 9.
* They form 227 groups covering 467 / 1,035 clips.
* The distribution is bimodal: **no pair falls between 0.5 and 0.95**, and 247 pairs are ≥ 0.99.
* 78 / 209 test clips (37.3 %) have a near-copy in train.
* Of 147 multi-clip train groups, 1 has inconsistent labels (Id 75 / 636 / 878).
* All 9 test–test groups and all 43 held-out OOF groups already get identical predictions from B8, so there is no consistency
  gain to be had.

## 4. What earlier rounds already established (EARLIER CLAIM, rows X00–X50 in `research/experiments.csv`)

Frozen Whisper-large encoders + LR dominate: dup-group CV 0.88–0.90, vs ≤ 0.82 for wav2vec2, WavLM, XLS-R, w2v-BERT and
emotion2vec. B5, B6 and B8 are statistically tied (≤ 0.0012 apart).

Rejected because the gain was inside noise or negative:
* Heads: MLP, attention pooling, learned layer weighting, early fusion; other classifiers; stacking; class-bias tuning.
* Extra inputs: a text (ASR transcript) branch, pseudo-labels, external acted Hindi data.
* Audio processing: train-time speed/noise augmentation, PCM_32 domain normalisation, trimming, low-pass, domain bias; the
  repeated-sentence rule.
* Other backbones: a second LoRA backbone (Hindi Whisper-v3, 8 s), Whisper-large-v1 and Qwen2-Audio encoders; 8-s LP-FT.

One diagnostic was left unexploited: **the frozen probe scores 0.908 on the 8-s-window encoder vs 0.936 with Whisper's native
30-s input**, and both fine-tuned members so far were trained at 8 s. This round tests that gap.

## 5. Ranked experiment queue (this round)

| # | Experiment | Why (evidence) | Cost (M4 Pro) | Status |
|---|---|---|---|---|
| 1 | **LP-FT at native 30 s** (Whisper-large-v2, LoRA q/v r16 all layers, head = exact linear probe, 3 ep, lr 5e-5/1e-4) | closes the 8-s gap while starting from the best known solution (probe 0.936) | 7.5 h (measured) | **done — REJECT** (B9-01) |
| 2 | **LoRA at native 30 s, top 16 layers** (= B8's recipe: random head, lr 1e-4/1e-3, 6 ep; layers 1–16 frozen & cached in fp32) | isolates the context-window hypothesis for the B8 recipe; random head moves further from the probe → more ensemble diversity | 5.6 h (measured) | **done — better member, not an upgrade** (B9-03) |
| 3 | Second backbone / neutral-class / PCM_32 targeted work | prior evidence negative (rows X20–X49); only revisit with a changed hypothesis | — | not run (no new hypothesis with positive prior evidence) |
| 4 | Ensemble changes | only if #1/#2 pass the keep rule | min | **evaluated — none passes** (B9-04) |

## 6. Executed experiments (MEASURED)

### 6.1 Fine-tuned members (Track A, 824 OOF rows)

| member | context | trainable part | OOF Macro-F1 | non-twin | per fold | paired comparison (dup-group bootstrap) |
|---|---|---|---|---|---|---|
| `lp30` = frozen linear probe of Whisper-large-v2 (identical to `lr_wlv2cat`) | 30 s | LR head | 0.9362 | 0.9117 | 0.9414 / 0.9565 / 0.9341 / 0.9137 / 0.9332 | reference for LP-FT |
| `lora8` (B8's member, earlier round) | 8 s | LoRA q/v r16 in layers 1–32, random head | 0.9145 | 0.8887 | 0.9090 / 0.9201 / 0.9082 / 0.9200 / 0.9163 | reference for 30-s LoRA |
| **B9-01 `lpft30`** | 30 s | LoRA in layers 1–32, head = exact LR probe | 0.9302 | 0.9033 | 0.9463 / 0.9499 / 0.9221 / 0.9013 / 0.9287 | vs its own probe: **−0.0060** [−0.0145, +0.0018], P(>0) 0.06 |
| **B9-03 `lora30top`** | 30 s | LoRA in layers 17–32, random head | 0.9242 | 0.8994 | 0.9194 / 0.9335 / 0.9134 / 0.9335 / 0.9231 | vs `lora8`: **+0.0100** [−0.0050, +0.0259], P 0.90; better in 5 / 5 folds |

* **The native 30-s context helps the LoRA recipe.** The gain is +0.010, every fold improves, and it matches the 0.908 → 0.936
  probe gap. The interval still includes 0, so the HYPOTHESIS is supported but not proven.
* **LP-FT drifts away from a near-optimal linear probe.** It falls below its probe in 4 / 5 folds, and its errors are nearly
  the probe's own errors (correctness correlation 0.895), so it adds no diversity.
* **No fine-tuned member beats the frozen probe (0.9362) on its own.** Their value is diversity:
  * correctness correlation with `lr_wlv2cat`: 0.501 (`lora8`), 0.520 (`lora30top`), 0.895 (`lpft30`);
  * OOF rows the member gets right and `lr_wlv2cat` gets wrong: 20, 21 and 3 respectively.

### 6.2 Ensemble candidates vs B8

All candidates are equal-weight probability means; test predictions include the twin safety net. Comparison: paired
dup-group bootstrap, 3,000 draws.

| candidate | OOF | non-twin | fold min–max | gain vs B8 [95 % CI] | P(gain>0) | test rows ≠ B8 | keep rule |
|---|---|---|---|---|---|---|---|
| B8 = `lora8` + 5 frozen | 0.9433 | 0.9211 | 0.9139–0.9692 | — | — | — | — |
| B5 = 5 frozen only | 0.9432 | 0.9214 | 0.9074–0.9692 | −0.0001 [−0.0066, +0.0072] | 0.46 | 2 | no |
| B9a = `lpft30` replaces `lora8` | 0.9445 | 0.9230 | 0.9074–0.9692 | +0.0012 [−0.0049, +0.0083] | 0.60 | 1 | no |
| B9b = B8 + `lpft30` | 0.9470 | 0.9262 | 0.9139–0.9692 | +0.0037 [+0.0000, +0.0099] | 0.86 | 0 | no |
| B9c = `lpft30` + `lp30` + 5 frozen | 0.9445 | 0.9232 | 0.9147–0.9692 | +0.0013 [−0.0054, +0.0088] | 0.62 | 3 | no |
| **B9d = `lora30top` replaces `lora8`** | **0.9482** | **0.9279** | 0.9139–0.9692 | +0.0049 [+0.0000, +0.0119] | 0.95 | **0** | no (gain < +0.005, P < 0.99) |
| B9e = B8 + `lora30top` | 0.9458 | 0.9244 | 0.9139–0.9692 | +0.0025 [+0.0000, +0.0062] | 0.93 | 0 | no |
| B9f = `lora30top` + `lpft30` + 5 frozen | 0.9457 | 0.9247 | 0.9074–0.9692 | +0.0024 [−0.0026, +0.0090] | 0.73 | 0 | no |

Per fold: B8 0.9139 / 0.9692 / 0.9470 / 0.9496 / 0.9346; B9d 0.9139 / 0.9692 / 0.9522 / 0.9496 / 0.9523.

### 6.3 Per class and confusion (OOF, 824 rows)

| class | B8 F1 | B9d F1 | B9a F1 |
|---|---|---|---|
| angry | 0.9533 | 0.9580 | 0.9557 |
| happy | 0.9452 | 0.9504 | 0.9421 |
| neutral | 0.9248 | 0.9294 | 0.9252 |
| sad | 0.9499 | 0.9549 | 0.9549 |

Confusion matrices (rows = true; columns = predicted angry / happy / neutral / sad):

| true \ pred | B8 | B9d |
|---|---|---|
| angry | 194 / 2 / 5 / 4 | 194 / 2 / 5 / 4 |
| happy | 1 / 181 / 8 / 2 | 1 / 182 / 7 / 2 |
| neutral | 2 / 8 / 203 / 3 | 2 / 7 / 204 / 3 |
| sad | 5 / 0 / 7 / 199 | 3 / 0 / 7 / 201 |

Neutral is still the weakest class in every candidate. Most confusions are happy ↔ neutral and sad → neutral.

### 6.4 What B9d actually changes (`research/work/b9_diagnostics.txt`)

* **OOF:** 4 of 824 predictions differ from B8. B9d fixes all 4 and breaks none; all 4 are non-twin rows (fold 2: 1, fold 4: 3).
* **Test:** 0 of 209 predictions differ, also *before* the twin safety net. The mean absolute probability difference is
  0.0045, and only 2 B8 test clips have a top-2 margin below 0.10.
* B9d is therefore weakly better on OOF but would score exactly the same as B8 on both the public and the private leaderboard.

### 6.5 Novel-recording view

* **Track B, frozen families** (dup-group-disjoint folds, 3 seeds, re-computed by the notebook):

  | family | Track B | Track A |
  |---|---|---|
  | lr_wlv2 | 0.9000 ± 0.0026 | 0.936 |
  | lr_wlv3 | 0.8948 ± 0.0017 | 0.920 |
  | lr_wvaani | 0.8917 ± 0.0011 | 0.929 |
  | lr_whhlv2 | 0.8839 ± 0.0016 | 0.914 |
  | lr_wmed | 0.8478 ± 0.0073 | 0.902 |
  | **B5 blend (5 frozen, equal mean)** | **0.9115 ± 0.0042** | 0.9432 |

  Track A is 0.025–0.054 higher, mostly because of twins. On novel recordings the blend beats the best single family by
  +0.0115 (`research/scripts/b9_trackb_blend.py`, `research/work/b9_trackb_blend.txt`).
* **Fine-tuned members were not re-trained on Track B folds.** That would take 5.5–7.5 h per member here (stated limitation).
* **Proxy: the non-twin slice of Track A** (602 rows in 564 groups), paired bootstrap vs B8:

  | candidate | gain [95 % CI] | P(gain>0) |
  |---|---|---|
  | B9d | +0.0068 [+0.0000, +0.0156] | 0.96 |
  | B9b | +0.0051 | 0.87 |
  | B9e | +0.0033 | 0.89 |
  | B9a | +0.0019 | 0.67 |
  | B5 | +0.0004 | 0.53 |

### 6.6 Multiple-candidate optimism

* No candidate passes the pre-declared rules (keep: gain ≥ +0.005 and P ≥ 0.9; upgrade: P ≥ 0.99).
* **White's Reality Check** over this round's 7 alternatives: bootstrap gains are centred at their means, and the statistic
  is the maximum gain.
  * The best candidate is B9d at +0.0049; the 95th percentile of the null maximum is +0.0073, so **p = 0.165 — not
    significant**.
  * Only this round's alternatives are included. About 45 blends were tried across all rounds, so the honest correction is
    stronger still.

## 7. Selection and rationale

**No candidate reliably improves B8. B8 (public 0.94230, OOF 0.9433) remains the recommended submission, and nothing new was
submitted.**

* **No candidate passes the keep rule.** The family-wise Reality Check is not significant (p = 0.165).
* **The highest-OOF candidates change nothing.** B9d, B9b, B9e and B9f make test predictions identical to B8, so submitting
  them would score exactly B8's score.
* **The candidates that do change test predictions are coin flips.** B9a changes 1 clip, B5 2 and B9c 3, each with
  P(gain>0) ≤ 0.62.
* **What B9 did establish is useful for future work, but it is not an upgrade:**
  * Native 30-s context gives a better LoRA member (+0.010, 5 / 5 folds).
  * That member is the first fine-tuned family that contributes to the blend: +0.0048 over frozen-only, P 0.98. B8's `lora8`
    contributes +0.0000.
  * LP-FT from an exact probe does not help with ~660 clips per fold.
* **Final Kaggle selection, if two are chosen:** B8 (0.94230) and B6 (0.93060). B6 differs from B8 on 3 test clips and is
  statistically tied on CV. These choices rest on CV, not on the public leaderboard.

## 8. Notebook `notebooks/B9_hindi_ser_final.ipynb`

**Structure:** 13 cells.

| cells | content |
|---|---|
| 0 | overview, modes and options |
| 1 | selection evidence |
| 2 | configuration: mode, recipes, pinned model revisions, hardware knobs, Kaggle-or-local path detection, cache restore |
| 3–4 | data, canonical folds (embedded `FOLD_MAP`), near-duplicate groups |
| 5–6 | frozen families |
| 7–8 | fine-tuned family: LoRA model with an explicit encoder forward, LP stage, training loop with a non-finite-loss guard, per-fold cache and adapters |
| 9–10 | validation report |
| 11–12 | ensemble, twin safety net, submission with assertions |

**Defaults:** `B9_MODE=full`, `B9_FT_RECIPE=lora8`, `ENSEMBLE` = fine-tuned + 5 frozen families, which is the B8 composition.

**Inputs:** only the competition data and five public Apache-2.0 Whisper checkpoints at pinned commits (licences checked through
the Hugging Face API). No private caches and no other participant's checkpoints.

Verification of the final file (notebook md5 `4924e137e20948cb6cf787b155c744da`):

| # | check | how | result | label |
|---|---|---|---|---|
| 1 | notebook format | `nbformat` 4.5 schema validation; delivered file has no outputs | valid, no warnings | MEASURED |
| 2 | frozen-feature parity, all 5 models | notebook's own config and data cells plus `whisper_features`, pinned revisions; fresh extraction on 6 clips (2 shortest, 2 longest, first train, first test) compared with the research features | **bit-identical** (relative difference 0.0) for all 5 | MEASURED (`research/work/b9_frozen_parity.log`) |
| 3 | FULL mode, default `lora8`, cached fine-tuned predictions | executed through a Jupyter kernel (`nbclient`); frozen LR families re-fitted inside the notebook | ensemble OOF 0.9433; frozen families identical to research; `submission.csv` md5 = submitted B8 (`696f1431…`) | VERIFIED-CACHED |
| 4 | FULL mode, `lora30top`, cached fine-tuned predictions | same | ensemble OOF 0.9482; `submission.csv` md5 = submitted B8 | VERIFIED-CACHED |
| 5 | near-duplicate scan parity | notebook's config and data cells in FULL mode with an **empty** cache, so its own NCC scan runs; compared with the research partition | **identical partition** (795 groups; 227 multi-clip groups covering 467 clips; 78 test twins); 56 s | MEASURED (`research/work/b9_dupscan_parity.log`) |
| 6 | SMOKE `lora8` from an empty cache | real training: 96 clips, fold 0, 1 epoch | all cells OK; manual encoder forward = transformers forward (max relative diff 0.0); 2,621,440 LoRA parameters trained, finite loss; adapter saved; writes `submission_smoke.csv` (no `submission.csv`) | MEASURED |
| 7 | SMOKE `lora30top` from an empty cache | same | all cells OK; forward equivalence 0.0; 1,310,720 LoRA parameters (layers 17–32) | MEASURED |
| 8 | SMOKE `lpft30` from an empty cache | same, including the linear-probe stage | all cells OK; forward equivalence 0.0; LP-stage probe initialises the head (valid 0.70 on 24 rows, unchanged by 1 epoch of LP-FT) | MEASURED |
| 9 | FULL `lora8` with cached fine-tuned predictions, **transformers 4.57.6 / peft 0.17.1** | same as row 3, second environment | ensemble OOF 0.9433; `submission.csv` md5 = submitted B8 | VERIFIED-CACHED |
| 10–12 | SMOKE `lora8` / `lora30top` / `lpft30` from an empty cache, **transformers 4.57.6 / peft 0.17.1** | real training, as rows 6–8 | all cells OK; forward equivalence 0.0; same LoRA parameter counts; training losses **bit-identical** to transformers 5.19 (1.9264 / 1.7898 / 0.4660) | MEASURED |

All rows refer to the final file. Rows 3–4 and 6–12 were executed through a Jupyter kernel; the executed notebooks and logs are
in `research/b9_outputs/verification/`.

As expected for 9 optimisation steps on 72 clips, the random-head smoke models collapse to one class (F1 0.07). The smoke runs test
the code, not the model.

**Version compatibility** (Kaggle's library versions are not known in advance, so the notebook was also run with
transformers 4.57.6 / peft 0.17.1; rows 9–12). That test found a real failure in an earlier draft, now fixed:
* With peft 0.17, `layers_pattern="layers"` matched no modules of a bare Whisper encoder and raised `ValueError`; peft 0.21
  accepts it. LoRA targets are now passed as exact module names (`layers.{i}.self_attn.q_proj` / `v_proj` for the trained
  layers), with an assert on the exact LoRA parameter count. Both versions give identical parameter sets (2,621,440 for all 32
  layers, 1,310,720 for layers 17–32).
* The explicit encoder forward calls each layer through a small wrapper: transformers 4.x passes `layer_head_mask` and returns
  a tuple; 5.x takes `(hidden_states, attention_mask)` and returns a tensor.

**Not executed (limitations):**
* A complete from-scratch FULL run of this notebook (5 frozen extractions + 5-fold fine-tuning). The frozen path is covered by
  check 2. The fine-tuning path is covered by the smoke runs and by the research scripts' 5-fold runs of the same model.
* The CUDA / fp16-AMP path; there is no NVIDIA GPU here.
* Kaggle runtimes.

## 9. Hardware, runtime, limitations

**Hardware and software:** Apple M4 Pro (MPS), 25.8 GB unified memory, fp32. torch 2.14.1, transformers 5.19.0, peft 0.21.2,
scikit-learn 1.9.1, Python 3.12.

| job | measured runtime |
|---|---|
| B9-01 LP-FT 30 s, 5 folds (`research/scripts/whisper_lpft.py`) | 7.5 h |
| B9-03 LoRA-top 30 s, 5 folds (`research/scripts/whisper_lora_top.py`) | 5.6 h (≈ 66 min per fold) + 16 min one-time layer-16 cache |
| B9-04 evaluation and diagnostics (`b9_eval.py`, `b9_diagnostics.py`) | < 3 min |
| notebook FULL mode with cached artifacts | 0.3 min per run |
| notebook SMOKE mode from an empty cache (`lora8` / `lora30top` / `lpft30`) | 5.5 / 8.0 / 11.7 min when run alone; 9.9 / 14.9 / 21–22 min with both environments running at once |
| notebook near-duplicate scan (FULL mode, no cache) | 56 s |
| frozen-feature parity check (5 models × 6 clips) | ≈ 1 min |

Limitations:
* **One seed per fine-tuned member.** Seed variance is unmeasured, so B9d's +0.0049 is within plausible seed noise.
* **Fine-tuned members were evaluated on Track A only**, with the non-twin slice as the novel-recording proxy.
* **The decision is robust to both.** Even if B9d were truly better, it would not change a single test prediction.
* **A fresh FULL run will not be byte-identical to B8.** It re-trains `lora8` with this notebook's seeds (42 + fold) on whatever
  GPU Kaggle provides, so expect small differences from the submitted file. Earlier notes estimated 1–2 clips; this was not
  measured.
* **The public leaderboard was not used for any decision.**

## 10. Running on Kaggle

1. **Create the notebook.** Kaggle → *Create → New Notebook → File → Import Notebook* → upload
   `notebooks/B9_hindi_ser_final.ipynb`.
2. **Attach the data.** *Add Input → Competitions → DataVerse: Detecting Emotions from Hindi Speech*. The notebook finds
   `/kaggle/input/**/SRCASW-BhavVaani/train.csv`; public notebooks use
   `/kaggle/input/competitions/dataverse-detecting-emotions-from-hindi-speech/SRCASW-BhavVaani`.
3. **Settings.** Accelerator: **GPU** (T4 or P100; one GPU is used). **Internet: On**, because the five pinned Whisper
   checkpoints are downloaded from the Hugging Face Hub (≈ 13 GB).
4. **Optional smoke test.** Add a first cell `import os; os.environ["B9_MODE"] = "smoke"` and *Run All*. Expect every
   assertion to pass and `submission_smoke.csv written (smoke mode)` at the end. **Do not submit that file.** Delete the cell
   afterwards.
5. **Full run.** *Save Version → Save & Run All (Commit)* with the defaults. Outputs:
   * `/kaggle/working/submission.csv`;
   * `/kaggle/working/b9_cache/`, holding features, per-fold predictions, LoRA adapters and `b9_probs.npz`.
6. **If a session might hit the time limit,** split the folds:
   1. Commit version 1 with `os.environ["B9_LORA_FOLDS"] = "0,1,2"`. Its `submission.csv` is partial, so do not use it.
   2. Attach version 1's output to a new version (*Add Input → Your Work*). Its `b9_cache` is restored automatically.
   3. Run that version with the default folds: folds 0–2 load from the cache and folds 3–4 train.
7. **Check the log.** Every assertion should pass. `ENSEMBLE` OOF should be ≈ 0.94; the research value is 0.9433, and a fresh
   `lora8` run will differ slightly. The run should end with `submission.csv written (full mode)`.
8. **Submitting.** Submissions in this competition were CSV uploads (*Submit Prediction*, or `kaggle competitions submit -c
   dataverse-detecting-emotions-from-hindi-speech -f submission.csv -m "..."`). **Re-submitting is not recommended:** the B8
   file is already scored, and a fresh run differs from it only by GPU / seed noise.

To use the 30-s member instead, set `os.environ["B9_FT_RECIPE"] = "lora30top"`. This is measured but not an upgrade, its
test predictions were identical to B8, and it is slower.

## 11. Artifacts

* Notebook: `notebooks/B9_hindi_ser_final.ipynb`. Generator: `research/scripts/build_b9_notebook.py`. Folds:
  `research/work/canonical_folds.json`.
* Ledger: `research/B9_ledger.csv`, rows B9-00 to B9-05.
* Scripts in `research/scripts/`:

  | script | role |
  |---|---|
  | `b9_audit.py` | baseline audit |
  | `whisper_lpft.py` | B9-01 |
  | `whisper_lora_top.py` | B9-03 |
  | `b9_eval.py` | B9-04 candidate evaluation |
  | `b9_diagnostics.py` | B9-04 diagnostics |
  | `b9_prepare_cache.py` | builds the notebook cache from research artifacts |
  | `b9_frozen_parity.py` | frozen-feature parity check |
  | `b9_dupscan_parity.py` | near-duplicate scan parity check |
  | `b9_trackb_blend.py` | Track B score of the frozen-only blend |

* Logs in `research/work/`:

  | log | covers |
  |---|---|
  | `b9_audit.txt` | baseline audit |
  | `lpft_wlv2_30s.log` | B9-01 |
  | `loratop_wlv2.log`, `loratop_wlv2_check.log` | B9-03 |
  | `b9_eval_final.txt`, `b9_eval.csv`, `b9_diagnostics.txt` | B9-04 |
  | `b9_frozen_parity.log`, `b9_dupscan_parity.log` | parity checks |
  | `b9_trackb_blend.txt` | Track B of the frozen-only blend |

* Outputs in `research/b9_outputs/`:
  * `submission.csv`: the notebook's FULL-mode output, identical to the submitted B8;
  * `b9_probs.npz`: OOF and test probabilities of every family, as written by the notebook;
  * `b9_probs_lora30top.npz`;
  * `b9_candidate_probs.npz`: the fine-tuned members;
  * `verification/`: executed notebooks and their logs.
* Candidate CSVs (not submitted): `research/submissions/sub_B9_*.csv`.

## 12. Addendum — B10: second seed of the 30-s LoRA (2026-10-12, MEASURED)

Plan declared before the run (ledger B10-01): **B10a** = mean of `lora30top` seeds 0 and 1 replaces `lora8`; **B10b** = B8 +
that mean. Keep rule: gain ≥ +0.005 with P ≥ 0.9, and at least one test prediction must change.

| item | result |
|---|---|
| seed-1 member | OOF 0.9302 (non-twin 0.9110); folds 0.9139 / 0.9566 / 0.9167 / 0.9327 / 0.9296; 5.6 h |
| seed-averaged member vs `lora8` | +0.0112 [−0.0043, +0.0264], P 0.92 |
| B10a vs B8 | OOF 0.9482, +0.0049 [+0.0000, +0.0119], P 0.95 (OOF predictions identical to B9d); **0 test rows changed** |
| B10b vs B8 | OOF 0.9458, +0.0025, P 0.93; **0 test rows changed** |
| Reality Check (9 alternatives, B9 + B10) | p = 0.165 |

**Decision: no upgrade, B8 stays.** The fine-tuned member keeps improving, but the submission cannot move this way. With
equal weights the five frozen families alone fix 207 / 209 test predictions for every fine-tuned member tried (`lora8`,
`lora30top`, seed average). No test clip has a top-2 margin below 0.05 (minimum 0.085). `sub_B10a.csv` and `sub_B10b.csv` are
byte-identical to the submitted B8.

## 13. Addendum — third-party "E11" package evaluated locally (2026-10-12, MEASURED)

The package (`DataVerse_Score_Improvement_Package.zip`) proposes:
* alternative layer recipes: 16+24+32 and 28+32 (medium: 20+24);
* LR with C = 0.3;
* PCA64 + RBF-SVM, and PCA64 + cosine kNN;
* OOF-weighted blends with B8.

Its notebook was not executed. Its recipes were re-implemented on the real cached features
(`research/scripts/e11_local_eval.py`); the re-implemented legacy LR reproduces B8's family exactly.

**Result: 52 candidates, none passes the keep rule.**
* The best is a +0.0037 gain (P 0.88) that changes **0** test rows.
* Every candidate that changes test rows has gain ≤ +0.0024 and P ≤ 0.72.
* White's Reality Check: p = 0.873.
* SVM (OOF 0.87–0.90) and kNN (0.77) are clearly weaker than LR on these features.
* Medium layers 20+24 improve the medium family itself (Track B 0.8630 vs 0.8478) but add nothing to the ensemble.

On Kaggle, running the package would have spent GPU hours re-extracting features that exist locally; here it took 34 s.
