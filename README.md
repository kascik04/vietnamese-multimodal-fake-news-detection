# VMTF-Net: Vietnamese Multimodal Transformer Fusion Network for Fake News Detection

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/Transformers-4.30+-yellow.svg?style=flat&logo=huggingface)](https://huggingface.co)
[![Kaggle GPU](https://img.shields.io/badge/Kaggle-GPU%20T4%20x2%20Ready-20BEFF.svg?style=flat&logo=kaggle)](https://kaggle.com)

---

## 1. Overview & Architecture

**VMTF-Net** (*Vietnamese Multimodal Transformer Fusion Network*) is an end-to-end deep learning system for detecting multimodal fake news on Vietnamese social media.

The model processes three modalities:
1. **Text (Text Encoder):** post text encoded by `vinai/phobert-base` (word-segmented with VnCoreNLP / pyvi) or `uitnlp/visobert` (no segmentation) → token embeddings $[B, L, 768]$ and sentence embedding $E_T \in \mathbb{R}^{768}$ (`<s>` hidden state or masked mean; the randomly-initialized HF pooler is **not** used).
2. **Visual (Visual Encoder):** image encoded by `google/vit-base-patch16-224` → patch embeddings $[B, 196, 768]$ and $E_V \in \mathbb{R}^{768}$. Posts without a usable image carry `has_image = 0` and contribute **no** visual information (see below).
3. **Social (Social Encoder):** engagement counts compressed with $\log(1 + x)$, standardized with statistics fitted **on the training split only** (stored in the checkpoint), then mapped by an MLP to $E_S \in \mathbb{R}^{128}$.
   - **Basic preset** (3 features): `like_count`, `comment_count`, `share_count`
   - **Extended preset** (8 features): adds `angry_count`, `haha_count`, `like_share_ratio`, `comment_share_ratio`, `engagement_ratio`

### Fusion & Objective
* **Stage 1 – Bidirectional Cross-Modal Attention + Contradiction**
  - $\bar{T} = \text{Pool}(\text{LN}(T + m \cdot \text{MHA}(Q=T, K=V, V=V)))$, $\bar{V} = m \cdot \text{Pool}(\text{LN}(V + \text{MHA}(Q=V, K=T, V=T)))$, with $m$ the image mask.
  - Contradiction: $C = m \cdot \text{MLP}([|W_t\bar{T} - W_v\bar{V}| \,;\, W_t\bar{T} \odot W_v\bar{V}]) \in \mathbb{R}^{384}$ (projections $W_t, W_v$ bring both encoders into a common space).
* **Stage 2 – Reliability-Aware Gated Fusion**
  - $\beta = m \cdot \text{Softmax}(\text{GateNet}([E_T + \bar{T} \,;\, E_V + \bar{V}]))_v$, $\alpha = 1 - \beta$ (so $\alpha + \beta = 1$ and $\beta = 0$ without image).
  - $F = \text{MLP}([\alpha (E_T + \bar{T}) + \beta (E_V + \bar{V}) \,;\, E_S \,;\, C]) \in \mathbb{R}^{768}$
* **Objective**: $\mathcal{L} = \mathcal{L}_{cls} + \lambda \, \mathcal{L}_{cons}$, where $\mathcal{L}_{cls}$ is CE / Focal loss (optional balanced class weights) and $\mathcal{L}_{cons}$ is computed on **learned projections** $P_t E_T$, $P_v E_V$ in a shared 256-d space: $1 - \cos$ for real news, $\max(0, \cos - \text{margin})$ for fake news, averaged only over posts that have an image.

---

## 2. Codebase Structure

```
vmtf_net/
├── config.py             # Hyperparameters, ablation flags, presets, environment auto-detection
├── dataset.py            # Pre-tokenization (+cache), word segmentation, has_image mask,
│                         # serializable social normalizer, dynamic-padding collate, seeded loaders
├── models/
│   ├── text_encoder.py   # PhoBERT / ViSoBERT wrapper (cls / mean pooling, grad checkpointing)
│   ├── visual_encoder.py # ViT wrapper
│   ├── social_encoder.py # Social MLP (BatchNorm + Dropout)
│   ├── fusion.py         # Cross-attention, contradiction, gated fusion, late concat, unimodal
│   └── vmtf_net.py       # Full model, consistency projection heads, parameter groups
├── losses.py             # Focal Loss, masked Text-Image Consistency Loss, Combined Loss
├── style_features.py     # Clickbait / sensational-style features + explanations (no torch)
├── retrieval/            # Evidence layer: claim KB search, source credibility, PostChecker
├── train.py              # AMP, grad accumulation, DataParallel, resume, history, final test
└── evaluate.py           # Metrics + bootstrap CIs, threshold tuning, interpretability, predictions CSV

scripts/
├── validate_and_preprocess_dataset.py  # Schema check, hashing, anti-leakage grouping, splits, K-fold
├── run_seeds.py                        # Multi-seed / K-fold runner + mean ± std aggregation
├── significance_test.py                # McNemar + paired bootstrap between two models
├── convert_reintel.py                  # ReINTEL (VLSP 2020) -> 55-column schema
├── resize_images.py                    # One-off image downscaling (faster data loading)
├── build_claim_kb.py                   # Index the fact-checked claim knowledge base
└── check_post.py                       # Check one post / evaluate the evidence layer

test_step1.py ... test_step5.py         # Offline tests (no model download; step 5 needs no torch)
notebooks/kaggle_quickstart.ipynb       # Ready-to-run Kaggle notebook
notebooks/colab_a100_drive.ipynb       # Colab (A100) notebook with Google Drive storage
docs/DATA_COLLECTION_GUIDE.md           # Data collection & annotation protocol (Vietnamese)
docs/EXPERIMENT_PLAN.md                 # GPU budget & experiment schedule (Vietnamese)
```

---

## 3. Dataset Preparation

### 3.1 Data sources
* **Public benchmark (strongly recommended for comparability):** ReINTEL (VLSP 2020) — Vietnamese social-network posts with text, image and `num_like/num_comment/num_share`, matching the `basic` preset. Map `post_message → text`, `num_like_post → like_count`, `num_comment_post → comment_count`, `num_share_post → share_count`, `user_name → source`, `timestamp_post → timestamp`, image file → `image_1_local_path`.
  Convert it with the dedicated script (column names, unix timestamps, counts like `1,2K`, images by id or URL are handled automatically):
  ```bash
  python scripts/convert_reintel.py --input /kaggle/input/reintel/public_train.csv \n      --image_dir /kaggle/input/reintel/images --output data/reintel_vmtf.csv
  ```
* **Self-collected data:** follow [`docs/DATA_COLLECTION_GUIDE.md`](docs/DATA_COLLECTION_GUIDE.md) (55-column schema in `data/dataset_template.csv`).

### 3.2 Validate, de-duplicate & split

```bash
# Stratified group split 70/15/15 (duplicates by claim / image hash / text never cross splits)
python scripts/validate_and_preprocess_dataset.py --input_csv data/my_collected_data.csv --image_dir data/images

# Harder, more realistic protocols
python scripts/validate_and_preprocess_dataset.py --input_csv data/raw.csv --split_by source   # unseen pages
python scripts/validate_and_preprocess_dataset.py --input_csv data/raw.csv --split_by time     # older -> newer
python scripts/validate_and_preprocess_dataset.py --input_csv data/raw.csv --kfold 5           # data/folds/fold_*
```

The script prints a leakage report and warns about empty / single-class splits.

---

## 4. Quick Start on Kaggle (GPU T4 x2 or P100) / Colab

Open [`notebooks/kaggle_quickstart.ipynb`](notebooks/kaggle_quickstart.ipynb) (Kaggle) or [`notebooks/colab_a100_drive.ipynb`](notebooks/colab_a100_drive.ipynb) (Colab + Google Drive). The GPU-hour budget and run schedule are in [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md). Or run:

```bash
!pip install -q pyvi imagehash            # torch / transformers are pre-installed on Kaggle & Colab

# Offline tests (seconds, no download)
!python test_step1.py && python test_step2.py && python test_step3.py && python test_step4.py && python test_step5.py

# Optional: shrink images once (inputs are read-only on Kaggle)
!python scripts/resize_images.py --input_dir /kaggle/input/<dataset>/images --output_dir /kaggle/working/images

!python -m vmtf_net.train \
    --train_csv data/train.csv --val_csv data/val.csv --test_csv data/test.csv \
    --image_dir /kaggle/working/images --cache_dir /kaggle/working/cache \
    --batch_size 16 --epochs 10 --lr 2e-5 --head_lr 1e-4 \
    --fusion_type hybrid_proposed --consistency_lambda 0.1 --social_preset basic \
    --no_tqdm
```

Every run writes to `checkpoints/<experiment_name>/` (or `--checkpoint_dir`). The experiment name encodes
fusion type, ablations, backbones, social preset, loss options, λ and seed
(e.g. `hybrid_proposed-no_gate_phobert-base_vit-base-patch16-224_social-basic_lambda0.1_seed42`), so different
experiments never share a folder. Training refuses to write into a folder that already holds a run unless
`--resume` or `--overwrite` is given.

| File | Content |
|---|---|
| `best_vmtf_model.pt` | Best weights (val Macro-F1), config, **training social normalizer**, tuned threshold |
| `last_checkpoint.pt` | + optimizer / scheduler / scaler / RNG state for `--resume` (≈3 GB). Exists only while a run is in progress: deleted when the run finishes (`--keep_last` to keep, `--no_save_last` to never write it) |
| `history.csv` | Per-epoch train / val metrics, LR, epoch time (for learning curves) |
| `results.json` | Test metrics with 95% bootstrap CIs, at the val-tuned threshold and at 0.5, interpretability |
| `val_predictions.csv`, `test_predictions.csv` | Per-sample `prob_fake`, `pred`, `label`, `alpha`, `beta`, `cosine`, `has_image` |
| `run_info.json` | Library versions, GPUs, parameter count, epoch time, peak VRAM, throughput |

### Kaggle / Colab tips
* **Session limits (Kaggle 12 h, Colab disconnects):** re-run with `--resume <dir>/last_checkpoint.pt`.
* **T4 x2:** both GPUs are used automatically (DataParallel; disable with `--no_multi_gpu`).
* **A100 (Colab):** `--amp_dtype auto` (default) uses **bf16 + TF32** on Ampere GPUs and fp16 on T4/P100/V100; DataLoader workers are scaled to the vCPU count. Copy data from Drive to `/content` before training.
* **Out of memory:** `--gradient_checkpointing`, or `--batch_size 8 --grad_accum_steps 2` (same effective batch).
* **Internet off:** add PhoBERT / ViT as a Kaggle Model or Dataset and pass local paths: `--text_model_name /kaggle/input/phobert-base --visual_model_name /kaggle/input/vit-base-patch16-224`.
* **Best PhoBERT segmentation:** `pip install py_vncorenlp` (Java required) and add `--vncorenlp_dir /kaggle/working/vncorenlp` (models are downloaded automatically). Without it, pyvi is used.
* **Clean logs for "Save & Run All":** `--no_tqdm`.

### Evaluate a checkpoint separately

```bash
!python -m vmtf_net.evaluate \
    --checkpoint checkpoints/<exp>/best_vmtf_model.pt \
    --test_csv data/test.csv --image_dir data/images \
    --output_json logs/evaluation_results.json --output_predictions logs/test_predictions.csv
```

The decision threshold is taken from `--threshold`, else tuned on `--val_csv`, else the value stored in the checkpoint, else 0.5. It is **never** tuned on the test set.

---

## 5. Experiments for the Report / Paper

### 5.1 Baselines and ablations

| Experiment | Command flags |
|---|---|
| **Proposed** | `--fusion_type hybrid_proposed --consistency_lambda 0.1` |
| w/o consistency loss | `--consistency_lambda 0` |
| w/o cross-attention | `--no_cross_attention` |
| w/o reliability gate (α = β = 0.5) | `--no_gate` |
| w/o contradiction feature | `--no_contradiction` |
| w/o social branch | `--no_social` |
| Late concatenation baseline | `--fusion_type late_concat --consistency_lambda 0` |
| Text only (PhoBERT) | `--fusion_type text_only` |
| Image only (ViT) | `--fusion_type image_only` |
| Social only (MLP) | `--fusion_type social_only` |
| ViSoBERT backbone | `--text_model_name uitnlp/visobert` (segmentation is disabled automatically) |
| + style / clickbait branch | `--use_style` (on top of any of the above that uses text) |
| Text + style (cheap strong baseline) | `--fusion_type text_only --use_style` |

### 5.2 Several seeds or K-fold (report mean ± std)

```bash
python scripts/run_seeds.py --seeds 13 42 2024 --out_dir checkpoints/proposed -- \
    --train_csv data/train.csv --val_csv data/val.csv --test_csv data/test.csv --image_dir data/images --no_tqdm

python scripts/run_seeds.py --folds_dir data/folds --seeds 42 --out_dir checkpoints/cv_late -- \
    --image_dir data/images --fusion_type late_concat --consistency_lambda 0 --no_tqdm
```

Re-running the same command is safe (finished runs are skipped, interrupted ones resumed), so on Kaggle you can
simply restart it after a session timeout. `summary.json` / `all_runs.csv` contain mean ± std of Accuracy, Macro-F1, Fake-F1, Precision/Recall (fake), AUC-ROC and PR-AUC.

### 5.3 Is the improvement significant?

```bash
python scripts/significance_test.py \
    --a checkpoints/<baseline>/test_predictions.csv --name_a late_concat \
    --b checkpoints/<proposed>/test_predictions.csv --name_b hybrid_proposed
```

Reports McNemar's exact test and a paired bootstrap CI / p-value for ΔMacro-F1 and ΔAUC.

### 5.4 Reporting checklist
- Dataset statistics per split (size, fake ratio, % posts with image) and the split protocol (`group` / `source` / `time`).
- Main metrics: **Macro-F1, Fake-class F1, AUC-ROC, PR-AUC** (accuracy alone is misleading on imbalanced data), with CIs or mean ± std.
- Ablation table (5.1) and significance tests against the strongest baseline.
- Interpretability from `results.json`: distribution of β per class, text-image cosine real vs. fake, metrics with vs. without image.
- Efficiency from `run_info.json`: parameters, epoch time, peak VRAM, inference throughput.
- Limitation: engagement counts are only available **after** spreading; include the `--no_social` result for an early-detection setting.

---

## 6. Style Features & Evidence Layer (Phase 1)

VMTF-Net judges how a post *looks* (style, image, engagement). Two complementary, interpretable
components are provided:

**a) Style / clickbait branch** ([`vmtf_net/style_features.py`](vmtf_net/style_features.py), `--use_style`):
14 hand-crafted signals (exclamations, repeated punctuation, CAPS ratio, alarm emojis ⚠️🔥🚨,
Vietnamese clickbait terms "sốc / khẩn cấp / thần dược...", calls to share, URLs, hashtags...),
also detected in text typed without diacritics. They are log-scaled, standardized on the training split
(stored in the checkpoint) and fed through a small MLP into the fusion head. `explain_style()` turns them
into user-facing explanations. Sensational style is a *signal*, not proof: also evaluate on real news with
sensational titles.

**b) Evidence layer** ([`vmtf_net/retrieval/`](vmtf_net/retrieval/)), independent of the neural model:

| Component | Role |
|---|---|
| `ClaimKnowledgeBase` | Fact-checked claims (debunked / verified, source link, date) + similarity search (TF-IDF char+word offline, or a sentence-embedding model). Long posts are matched window by window. |
| `SourceCredibility` | Domains in the post: `trusted` (official / mainstream, `*.gov.vn`), `lookalike` (typo-squatting `vnexpres.net`, embedding `vnexpress.net.tin-nong.xyz`, `tuoitre-news.online`), `unknown` |
| `PostChecker` | Combines KB matches, sources, style signals and the model P(fake) into a cautious hint + Vietnamese message (never a bare "FAKE" verdict) |

```bash
# 1. Fill data/kb/debunked_claims.csv (template: data/kb/debunked_claims_template.csv), then index it
python scripts/build_claim_kb.py --kb_csv data/kb/debunked_claims.csv --out_dir data/kb/index

# 2. Check one post (what an extension backend would call)
python scripts/check_post.py --kb_dir data/kb/index --post_time 2024-05-01 --model_prob 0.83 \
    --text "KHẨN CẤP!!! Chia sẻ gấp: uống nước chanh nóng diệt virus..."

# 3. Evaluate the evidence layer on a labeled split (coverage / precision per threshold, clickbait AUC)
python scripts/check_post.py --kb_dir data/kb/index --eval_csv data/test.csv --output_json logs/evidence_eval.json
```

**Anti-leakage rule:** evaluation cuts the KB at each post's `timestamp` (`--time_cutoff post`, default): a
post can only be matched with refutations published *before* it. Without the cutoff, the refutation of the
test post itself is found and results are unrealistically good (report both if you want to show the gap).

---

## 7. Social Feature Presets

| Preset | Features | Config |
|---|---|---|
| `basic` (default) | `share_count`, `like_count`, `comment_count` | `--social_preset basic` |
| `extended` | Above + `angry_count`, `haha_count`, `like_share_ratio`, `comment_share_ratio`, `engagement_ratio` | `--social_preset extended` |

Missing ratio values are computed from the base counts; existing values are kept.

---

## 8. Environment Compatibility

| Environment | Workers | AMP | Pin Memory | Multi-GPU |
|---|---|---|---|---|
| Kaggle GPU (T4 x2 / P100) | 2 | ✅ fp16 | ✅ | ✅ DataParallel |
| Google Colab | 2 | ✅ fp16 | ✅ | – |
| Local Windows | 0 | Auto | Auto | Auto |
| Local Linux | ≤ 4 | Auto | Auto | Auto |

Hardware fields (device, AMP, workers, pin memory) are re-detected when a checkpoint is loaded on another machine.

---

## 9. Future Work
* OCR of text embedded in images (screenshots are frequent in Vietnamese fake news) as a fourth modality.
* Vision-language backbones (CLIP / SigLIP) whose image and text spaces are already aligned.
* Attention-map / Grad-CAM visualizations for qualitative error analysis.
