# Graph-Unlearning-Inversion

PyTorch implementation of **"Unlearning Inversion Attacks for Graph Neural Networks"** (WSDM '26).  
Extended with a **Privacy-Preserving Training Framework** that adds concept-aware masking, privacy-certified embedding transformation, and adversarial inversion-aware training.

ArXiv pre-print: <https://arxiv.org/abs/2506.00808>  
Codebase adapted from [GIF-torch](https://github.com/wujcan/GIF-torch/).

---

## Project Overview

This repository contains:

1. **Baseline WSDM '26 code** — graph unlearning + inversion attack evaluation pipeline (GCN / GAT / GIN / SGC / MLP on Cora, Citeseer, PubMed).
2. **Privacy-Preserving Framework (new)** — three modular components gated behind CLI flags that reduce the success rate of unlearning inversion attacks while preserving node-classification utility.

---

## Method Description

### Architecture

```
Graph
 ↓
GNN Encoder  (existing, unchanged)
 ↓
Concept Leakage Detector  (NEW — identifies leaky embedding dimensions)
 ↓
KAN Privacy Mask Layer    (NEW — suppresses leaky dimensions via learnable mask)
 ↓
Privacy-Certified Embedding Transformation  (NEW — minimises MI with deleted nodes)
 ↓
Classifier  (existing, unchanged)
```

### Components

| Component | File | Description |
|-----------|------|-------------|
| Concept Leakage Detector | `lib_gnn_model/leakage_detector.py` | Trains an MLP classifier to predict deleted-node membership; uses gradient saliency to score each embedding dimension. |
| KAN Privacy Mask | `lib_gnn_model/privacy_mask.py` | Applies a learnable Chebyshev-polynomial-enhanced mask that suppresses the most leaky dimensions. |
| Privacy-Certified Transform | `lib_gnn_model/privacy_transform.py` | MINE-based mutual-information regulariser that penalises correlation between embeddings and deleted-node indicators. |
| Adversarial Inverter | `lib_gnn_model/adversarial_inverter.py` | Min-max adversarial training with gradient reversal: the inverter reconstructs node features while the encoder learns to resist reconstruction. |

### Training Objective

When privacy flags are enabled the total loss becomes:

```
Loss = TaskLoss + λ_adv × AttackLoss + β_mi × PrivacyLoss
```

where `TaskLoss` is the original NLL cross-entropy loss (unchanged).

---

## Environment Requirements

```
python  >= 3.6.10
pytorch == 1.9.0+cu111
torch-geometric == 2.0.3
```

### Installation

```bash
pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-geometric==2.0.3
```

---

## Datasets

| Dataset  | Nodes  | Edges   | Features | Classes |
|----------|--------|---------|----------|---------|
| Cora     | 2 708  | 10 556  | 1 433    | 7       |
| Citeseer | 3 327  | 9 104   | 3 703    | 6       |
| PubMed   | 19 717 | 88 648  | 500      | 3       |

Datasets are downloaded automatically via PyTorch Geometric on first run.

---

## Running Experiments

### Baseline (original behaviour, no privacy flags)

```bash
python main.py \
    --exp Inversion --method GIF \
    --dataset_name cora --target_model GCN \
    --attack_method trend_steal \
    --unlearn_ratio 0.05 --num_runs 5 \
    --is_gen_unlearn_request True --is_gen_unlearned_probs True
```

### Proposed (privacy framework enabled)

```bash
python main.py \
    --exp Inversion --method GIF \
    --dataset_name cora --target_model GCN \
    --attack_method trend_steal \
    --unlearn_ratio 0.05 --num_runs 5 \
    --is_gen_unlearn_request True --is_gen_unlearned_probs True \
    --concept_leakage --privacy_mask --adversarial_training
```

### Privacy Hyperparameters

| Flag | Default | Description |
|------|---------|-------------|
| `--lambda_adv` | `0.1` | Weight for adversarial reconstruction loss |
| `--beta_mi` | `0.01` | Weight for MINE mutual-information penalty |
| `--leakage_hidden_dim` | `64` | Leakage detector MLP hidden size |
| `--privacy_mask_alpha` | `0.5` | KAN privacy mask scaling factor |
| `--mine_hidden_dim` | `64` | MINE estimator hidden size |
| `--adv_hidden_dim` | `128` | Adversarial inverter hidden size |
| `--adv_inner_steps` | `3` | Adversarial inner update steps per epoch |

---

## Reproducing Comparison Tables

Use `eval_privacy.py` to run both experiments back-to-back and print a formatted table:

```bash
python eval_privacy.py \
    --dataset_name cora --target_model GCN \
    --method GIF --attack_method trend_mia \
    --is_gen_unlearn_request True --is_gen_unlearned_probs True
```

Example output:

```
Privacy Framework Comparison Table
======================================================================
Dataset    Model  Method     Attack AUC   Attack F1  Node Acc
----------  ------  ----------  ------------  ----------  ----------
cora       GCN    Baseline   0.62          0.58        0.81
cora       GCN    Proposed   0.34          0.31        0.80
======================================================================
```

---

## Output Metrics

| Metric | Description |
|--------|-------------|
| Node classification accuracy | F1-micro on held-out test nodes |
| Attack AUC | Area under ROC for the inversion attack |
| Attack F1 | F1 score of the inversion attack classifier |
| Leakage score | Mean per-dimension leakage score from the detector |
| Reconstruction MSE | Mean-squared reconstruction error from the adversarial inverter |

---

## Citation

```bibtex
@inproceedings{zhang2026unlearning,
  title   = {Unlearning Inversion Attacks for Graph Neural Networks},
  author  = {Jiahao Zhang and Yilong Wang and Zhiwei Zhang and Xiaorui Liu and Suhang Wang},
  year    = {2026},
  booktitle = {The ACM International Conference on Web Search and Data Mining (WSDM)},
}
```

