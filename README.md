# VMTF-Net: Vietnamese Multimodal Transformer Fusion Network for Fake News Detection

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/Transformers-4.30+-yellow.svg?style=flat&logo=huggingface)](https://huggingface.co)
[![Kaggle GPU](https://img.shields.io/badge/Kaggle-GPU%20T4%20Ready-20BEFF.svg?style=flat&logo=kaggle)](https://kaggle.com)

---

## 1. Overview & Architecture

**VMTF-Net** (*Vietnamese Multimodal Transformer Fusion Network*) is an end-to-end deep learning system engineered specifically for detecting multimodal fake news on Vietnamese social media platforms.

The model simultaneously processes three modalities:
1. **Text Modality (Text Encoder):** Vietnamese post text encoded via `vinai/phobert-base` or `UIT-NLP/visobert` to extract token sequence embeddings $[B, L, 768]$ and $[CLS]$ embedding $E_T \in \mathbb{R}^{768}$.
2. **Visual Modality (Visual Encoder):** Accompanying image encoded via `google/vit-base-patch16-224` to extract patch embeddings $[B, 196, 768]$ and $[class]$ embedding $E_V \in \mathbb{R}^{768}$.
3. **Social Metrics Modality (Social Encoder):** Numerical interaction features compressed with $\log(1 + x)$ and mapped via a Deep MLP with BatchNorm to $E_S \in \mathbb{R}^{128}$.
   - **Basic preset** (3 features): `like_count`, `comment_count`, `share_count`
   - **Extended preset** (8 features): adds `angry_count`, `haha_count`, `like_share_ratio`, `comment_share_ratio`, `engagement_ratio`

### Fusion & Objective Formulation
* **Stage 1: Bidirectional Cross-Modal Attention (Contradiction Detection)**:
  - Text attends to visual patches: $\bar{T} = \text{Pool}(\text{LN}(T + \text{MHA}(Q=T, K=V, V=V)))$
  - Visual patches attend to text: $\bar{V} = \text{Pool}(\text{LN}(V + \text{MHA}(Q=V, K=T, V=T)))$
  - Contradiction vector: $C_{contra} = \text{MLP}([|\bar{T} - \bar{V}| \,;\, \bar{T} \odot \bar{V}]) \in \mathbb{R}^{384}$ to detect "Out-of-Context Fakes".
* **Stage 2: Reliability-Aware Gated Fusion**:
  - Gating network dynamically learns reliability weights $\alpha$ (text) and $\beta$ (image):
    $$[\alpha, \beta] = \text{Softmax}(\text{GateNet}([E_T \,;\, E_V])) \quad \text{subject to } \alpha + \beta = 1, \; \alpha \ge 0, \beta \ge 0$$
  - Gated representation: $M_{gated} = \alpha \cdot (E_T + \bar{T}) + \beta \cdot (E_V + \bar{V})$.
  - Multimodal fusion: $F_{fused} = \text{LayerNorm}(\text{Linear}([M_{gated} \,;\, E_S \,;\, C_{contra}])) \in \mathbb{R}^{768}$.
* **Combined Loss Objective**:
  $$\mathcal{L}_{total} = \mathcal{L}_{cls} + \lambda \times \mathcal{L}_{consistency}$$
  where $\mathcal{L}_{cls}$ is Cross-Entropy / Focal Loss and $\mathcal{L}_{consistency}$ enforces/penalizes text-image semantic alignment.

---

## 2. Codebase Structure

```
vmtf_net/
├── __init__.py           # Package exports (VMTFConfig, VMTFDataset, VMTFNet, etc.)
├── config.py             # Hyperparameters, social presets, multi-environment auto-detection
├── dataset.py            # Multimodal Dataset, PhoBERT tokenizer, ViT transforms, social normalizer
├── models/
│   ├── __init__.py
│   ├── text_encoder.py   # PhoBERT / ViSoBERT Wrapper
│   ├── visual_encoder.py # ViT Backbone Wrapper
│   ├── social_encoder.py # Social Metrics MLP (BatchNorm + Dropout)
│   ├── fusion.py         # Cross-Modal Attention & Reliability Gated Fusion
│   └── vmtf_net.py       # Full integrated architecture & differential LR grouping
├── losses.py             # Focal Loss, Text-Image Consistency Loss, Combined Loss
├── train.py              # Mixed-Precision Training, Early Stopping, Checkpointing
└── evaluate.py           # Evaluation script (Acc, Macro-F1, AUC-ROC, Interpretability)

data/
├── dataset_template.csv  # Standard 55-column schema template with example rows
└── images/               # Root directory for downloaded social media images

scripts/
└── validate_and_preprocess_dataset.py  # Schema validation, hashing, ratio computation, splitting

docs/
└── DATA_COLLECTION_GUIDE.md  # Step-by-step data collection & annotation guide
```

---

## 3. Dataset Preparation

### 3.1 Schema (55 Standard Columns)

The dataset follows a standardized 55-column schema defined in `data/dataset_template.csv`. Key column groups:

| Group | Key Columns |
|---|---|
| **Identity** | `sample_id`, `platform`, `source`, `source_url`, `timestamp` |
| **Text** | `text`, `language` |
| **Images** (up to 3) | `image_1_url`, `image_1_local_path`, `image_1_sha256`, `image_1_phash`, ... |
| **Social Metrics** | `like_count`, `comment_count`, `share_count`, `angry_count`, `haha_count` |
| **Derived Ratios** | `like_share_ratio`, `comment_share_ratio`, `engagement_ratio` (auto-computed if missing) |
| **Labels** | `label` (0=real, 1=fake), `raw_label`, `label_source`, `label_confidence` |
| **Anti-Leakage** | `claim_id`, `duplicate_group` |
| **Rebuttal Comments** | `rebuttal_comment_1..3`, timestamps, likes |
| **QA** | `data_quality`, `annotation_status`, `annotator`, `notes` |

### 3.2 Validate & Split Dataset

After collecting data into CSV format:

```bash
# Validate schema, compute hashes, auto-derive ratios, split 70/15/15
python scripts/validate_and_preprocess_dataset.py \
    --input_csv data/my_collected_data.csv \
    --image_dir data/images

# Validate only (no split)
python scripts/validate_and_preprocess_dataset.py \
    --input_csv data/my_collected_data.csv --no_split
```

For detailed collection guidelines, see [`docs/DATA_COLLECTION_GUIDE.md`](docs/DATA_COLLECTION_GUIDE.md).

---

## 4. Quick Start on Kaggle GPU (T4 x 2 or P100)

### Step 1: Install Dependencies
```bash
!pip install -r requirements.txt
```

### Step 2: Run Unit Tests
```bash
!python test_step1.py   # Config, Dataset, DataLoader
!python test_step2.py   # Model modules, Fusion, Gate constraint
!python test_step3.py   # Loss functions
```

### Step 3: Train VMTF-Net (Basic Social Features — 3 features)
```bash
!python -m vmtf_net.train \
    --train_csv data/train.csv \
    --val_csv data/val.csv \
    --image_dir data/images \
    --batch_size 16 \
    --epochs 10 \
    --lr 2e-5 \
    --head_lr 1e-4 \
    --fusion_type hybrid_proposed \
    --consistency_lambda 0.1 \
    --social_preset basic
```

### Step 4: Train with Extended Social Features (8 features)
```bash
!python -m vmtf_net.train \
    --train_csv data/train.csv \
    --val_csv data/val.csv \
    --image_dir data/images \
    --social_preset extended \
    --checkpoint_dir ./checkpoints/extended
```

### Step 5: Ablation Study (Late Concatenation Baseline)
```bash
!python -m vmtf_net.train \
    --train_csv data/train.csv \
    --val_csv data/val.csv \
    --image_dir data/images \
    --fusion_type late_concat \
    --checkpoint_dir ./checkpoints/ablation_late_concat
```

### Step 6: Evaluate & Extract Interpretability
```bash
!python -m vmtf_net.evaluate \
    --checkpoint checkpoints/best_vmtf_model.pt \
    --test_csv data/test.csv \
    --image_dir data/images \
    --output_json logs/evaluation_results.json
```

The evaluation report displays:
* Classification metrics: Accuracy, Macro F1, Binary F1, Precision, Recall, AUC-ROC.
* Interpretability: Average modality reliability weights $\alpha$ (text) vs $\beta$ (image) across Real vs Fake news.

---

## 5. Social Feature Presets

| Preset | Features | Config |
|---|---|---|
| `basic` (default) | `share_count`, `like_count`, `comment_count` | `--social_preset basic` |
| `extended` | Above + `angry_count`, `haha_count`, `like_share_ratio`, `comment_share_ratio`, `engagement_ratio` | `--social_preset extended` |

Ratios are **auto-computed** from base counts if missing in the CSV, so you only need to collect the base counts.

---

## 6. Environment Compatibility

| Environment | Workers | AMP | Pin Memory |
|---|---|---|---|
| Kaggle GPU (T4/P100) | 2 | ✅ | ✅ |
| Google Colab | 2 | ✅ | ✅ |
| Local Windows | 0 | Auto | Auto |
| Local Linux | 4 | Auto | Auto |