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
3. **Social Metrics Modality (Social Encoder):** Numerical interaction features (`like_count`, `share_count`, `comment_count`, `reaction_ratios`) compressed with $\log(1 + x)$ and mapped via a Deep MLP with BatchNorm to $E_S \in \mathbb{R}^{128}$.

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
├── config.py             # Hyperparameters, paths, multi-environment auto-detection
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
```

---

## 3. Quick Start on Kaggle GPU (T4 x 2 or P100)

### Step 1: Install Dependencies
In a Kaggle Notebook code cell:
```bash
!pip install -r requirements.txt
```

### Step 2: Run Unit Tests
Validate the environment, data loading, tensor contracts, and constraints ($\alpha + \beta = 1$):
```bash
!python test_step1.py
!python test_step2.py
!python test_step3.py
```

### Step 3: Train Proposed VMTF-Net
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
    --consistency_lambda 0.1
```

### Step 4: Run Ablation Study (Baseline Late Concatenation)
Compare against the baseline without cross-attention and gating (e.g., RoViT-Detect 2026):
```bash
!python -m vmtf_net.train \
    --train_csv data/train.csv \
    --val_csv data/val.csv \
    --image_dir data/images \
    --fusion_type late_concat \
    --checkpoint_dir ./checkpoints/ablation_late_concat
```

### Step 5: Evaluate & Extract Interpretability
```bash
!python -m vmtf_net.evaluate \
    --checkpoint checkpoints/best_vmtf_model.pt \
    --test_csv data/test.csv \
    --image_dir data/images \
    --output_json logs/evaluation_results.json
```
The evaluation report will display:
* Classification metrics: Accuracy, Macro F1, Binary F1, Precision, Recall, AUC-ROC.
* Interpretability: Average modality reliability weights $\alpha$ (text) vs $\beta$ (image) across Real vs Fake news.