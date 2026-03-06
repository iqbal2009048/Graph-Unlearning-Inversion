# Implementation Guide

## Graph Unlearning Inversion — VS Code Setup & Experiment Walkthrough

---

## Section 1 — Project Overview

This repository provides a PyTorch implementation of **"Unlearning Inversion Attacks for Graph Neural Networks"** (WSDM '26).  
The paper demonstrates that graph unlearning algorithms — methods that allow a trained GNN to "forget" specific nodes or edges — inadvertently leak information about the deleted data through the change in model outputs.  
An adversary who observes the model before and after unlearning can therefore reconstruct sensitive information about the forgotten entities, posing a serious privacy risk.

The codebase extends the original attack framework with a **Privacy-Preserving Training Framework** composed of four modular components:

| Component | File | Purpose |
|-----------|------|---------|
| Concept Leakage Detector | `lib_gnn_model/leakage_detector.py` | Identifies embedding dimensions that leak deleted-node membership via gradient saliency |
| KAN Privacy Mask | `lib_gnn_model/privacy_mask.py` | Suppresses leaky dimensions through a learnable Chebyshev-polynomial-enhanced mask |
| Privacy-Certified Embedding Transformation | `lib_gnn_model/privacy_transform.py` | MINE-based mutual-information regulariser that penalises correlation between embeddings and deleted-node indicators |
| Adversarial Inverter | `lib_gnn_model/adversarial_inverter.py` | Min-max adversarial training with gradient reversal to harden the encoder against feature reconstruction |

When the privacy flags are enabled the combined training objective becomes:

```
Loss = TaskLoss + λ_adv × AttackLoss + β_mi × PrivacyLoss
```

where `TaskLoss` is the standard NLL cross-entropy loss (unchanged from the baseline).

---

## Section 2 — Required Software

### Core tools

| Tool | Minimum version | Notes |
|------|----------------|-------|
| Python | 3.9 | 3.9+ recommended; 3.6.10+ supported |
| Git | Any recent release | For cloning the repository |
| CUDA toolkit | 11.1 (cu111) | Optional; CPU-only mode also works |

### Visual Studio Code

Download from <https://code.visualstudio.com/>.

#### Recommended VS Code Extensions

Install each extension from the **Extensions** panel (`Ctrl+Shift+X` / `Cmd+Shift+X`):

| Extension | Publisher ID | Why it helps |
|-----------|-------------|--------------|
| Python | `ms-python.python` | IntelliSense, linting, virtual-env management |
| Pylance | `ms-python.vscode-pylance` | Fast type checking and import resolution |
| Jupyter *(optional)* | `ms-toolsai.jupyter` | Useful for interactive experimentation |

---

## Section 3 — Clone the Repository

Open a terminal (or the VS Code integrated terminal with `` Ctrl+` ``):

```bash
git clone https://github.com/iqbal2009048/Graph-Unlearning-Inversion.git
cd Graph-Unlearning-Inversion
```

---

## Section 4 — Create and Activate a Python Virtual Environment

### 4.1 Create the environment

```bash
python -m venv .venv
```

### 4.2 Activate the environment

| Platform | Command |
|----------|---------|
| macOS / Linux | `source .venv/bin/activate` |
| Windows (CMD) | `.venv\Scripts\activate.bat` |
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |

Your terminal prompt will change to show `(.venv)` once activated.

### 4.3 Select the interpreter in VS Code

1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS) to open the Command Palette.
2. Type **Python: Select Interpreter** and press `Enter`.
3. Choose the interpreter that points to `.venv/bin/python` (or `.venv\Scripts\python.exe` on Windows).

---

## Section 5 — Install Dependencies

### 5.1 Upgrade pip

```bash
pip install --upgrade pip
```

### 5.2 Install PyTorch (GPU, CUDA 11.1)

```bash
pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

> **CPU-only fallback** — if you do not have a CUDA-capable GPU, install the CPU build instead:
> ```bash
> pip install torch==1.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
> ```

### 5.3 Install PyTorch Geometric

```bash
pip install torch-geometric==2.0.3
```

### 5.4 Install remaining dependencies

```bash
pip install numpy scikit-learn tqdm
```

---

## Section 6 — Project Structure

```
Graph-Unlearning-Inversion/
├── main.py                        # Entry point for all experiments
├── eval_privacy.py                # Convenience script: runs baseline + proposed and prints table
├── parameter_parser.py            # All CLI argument definitions
├── config.py                      # Storage paths (temp_data/)
├── exp/
│   ├── exp.py                     # Base experiment class
│   ├── exp_unlearn_inv.py         # Unlearning inversion experiment
│   ├── exp_attack.py              # Attack experiment
│   ├── exp_GIF.py                 # Graph Influence Function unlearning
│   └── exp_retrain.py             # Full-retrain baseline
├── lib_gnn_model/
│   ├── node_classifier.py         # GNN node classifier wrapper
│   ├── gnn_base.py                # Base GNN training logic
│   ├── leakage_detector.py        # (NEW) Concept leakage detector
│   ├── privacy_mask.py            # (NEW) KAN privacy mask layer
│   ├── privacy_transform.py       # (NEW) Privacy-certified embedding transform
│   ├── adversarial_inverter.py    # (NEW) Adversarial inverter for min-max training
│   ├── link_stealer.py            # Link-stealing attack model
│   ├── mlp.py                     # MLP backbone
│   ├── gcn/                       # GCN architecture
│   ├── gat/                       # GAT architecture
│   ├── gin/                       # GIN architecture
│   └── sgc/                       # SGC architecture
├── lib_dataset/
│   └── data_store.py              # Dataset loading and caching
├── lib_unlearn/
│   └── gif.py                     # GIF unlearning implementation
└── lib_utils/
    ├── utils.py
    ├── distance.py
    ├── partition.py
    ├── trend_feature.py
    └── logger.py
```

Datasets (Cora, Citeseer, PubMed) are downloaded automatically via PyTorch Geometric on the first run and cached under `temp_data/`.

---

## Section 7 — Run Baseline Experiments

The baseline corresponds to the original WSDM '26 paper — no privacy flags are set.

### 7.1 Single run (Cora, GCN, GIF unlearning, trend-steal attack)

```bash
python main.py \
    --exp Inversion --method GIF \
    --dataset_name cora --target_model GCN \
    --attack_method trend_steal \
    --unlearn_ratio 0.05 --num_runs 5 \
    --is_gen_unlearn_request True --is_gen_unlearned_probs True
```

### 7.2 Other dataset / model combinations

```bash
# Citeseer + GAT
python main.py \
    --exp Inversion --method GIF \
    --dataset_name citeseer --target_model GAT \
    --attack_method trend_mia \
    --unlearn_ratio 0.05 --num_runs 5 \
    --is_gen_unlearn_request True --is_gen_unlearned_probs True

# PubMed + GIN
python main.py \
    --exp Inversion --method GIF \
    --dataset_name pubmed --target_model GIN \
    --attack_method trend_mia \
    --unlearn_ratio 0.05 --num_runs 5 \
    --is_gen_unlearn_request True --is_gen_unlearned_probs True
```

### 7.3 Key CLI flags

| Flag | Choices | Default | Description |
|------|---------|---------|-------------|
| `--exp` | `Unlearn`, `Attack`, `Inversion` | `Unlearn` | Experiment type |
| `--method` | `GIF`, `Retrain`, `IF`, `CEU`, `GA` | `GIF` | Unlearning algorithm |
| `--dataset_name` | `cora`, `citeseer`, `pubmed` | `citeseer` | Graph dataset |
| `--target_model` | `GCN`, `GAT`, `GIN`, `SGC`, `MLP` | `GCN` | GNN backbone |
| `--attack_method` | `trend_mia`, `trend_steal`, `mia_gnn`, `transfer_lp`, `steal_link`, `group_attack` | `mia_gnn` | Inversion attack |
| `--unlearn_ratio` | float | `0.05` | Fraction of nodes/edges to unlearn |
| `--num_runs` | int | `1` | Number of repeated runs |

---

## Section 8 — Run the Proposed Privacy Framework

Enable all three privacy components by adding `--concept_leakage`, `--privacy_mask`, and `--adversarial_training`:

```bash
python main.py \
    --exp Inversion --method GIF \
    --dataset_name cora --target_model GCN \
    --attack_method trend_steal \
    --unlearn_ratio 0.05 --num_runs 5 \
    --is_gen_unlearn_request True --is_gen_unlearned_probs True \
    --concept_leakage --privacy_mask --adversarial_training
```

### 8.1 Privacy hyperparameters

| Flag | Default | Description |
|------|---------|-------------|
| `--lambda_adv` | `0.1` | Weight for adversarial reconstruction loss |
| `--beta_mi` | `0.01` | Weight for MINE mutual-information penalty |
| `--leakage_hidden_dim` | `64` | Leakage detector MLP hidden size |
| `--privacy_mask_alpha` | `0.5` | KAN privacy mask scaling factor |
| `--mine_hidden_dim` | `64` | MINE estimator hidden size |
| `--adv_hidden_dim` | `128` | Adversarial inverter hidden size |
| `--adv_inner_steps` | `3` | Adversarial inner update steps per epoch |

### 8.2 Example: tuning hyperparameters

```bash
python main.py \
    --exp Inversion --method GIF \
    --dataset_name cora --target_model GCN \
    --attack_method trend_steal \
    --unlearn_ratio 0.05 --num_runs 5 \
    --is_gen_unlearn_request True --is_gen_unlearned_probs True \
    --concept_leakage --privacy_mask --adversarial_training \
    --lambda_adv 0.2 --beta_mi 0.05 \
    --adv_inner_steps 5
```

---

## Section 9 — Reproduce Experiment Tables

`eval_privacy.py` runs the baseline and proposed experiments back-to-back and prints a formatted comparison table.

```bash
python eval_privacy.py \
    --dataset_name cora --target_model GCN \
    --method GIF --attack_method trend_mia \
    --is_gen_unlearn_request True --is_gen_unlearned_probs True
```

> **Note:** Do **not** pass `--concept_leakage`, `--privacy_mask`, or `--adversarial_training` on the command line when using `eval_privacy.py` — the script manages those flags internally.

### Expected output

```
Privacy Framework Comparison Table
======================================================================
Dataset     Model   Method      Attack AUC    Attack F1   Node Acc
----------  ------  ----------  ------------  ----------  ----------
cora        GCN     Baseline    0.62          0.58        0.81
cora        GCN     Proposed    0.34          0.31        0.80
======================================================================
```

### Output metrics explained

| Metric | Description |
|--------|-------------|
| **Node classification accuracy** | F1-micro on held-out test nodes — measures utility preservation |
| **Attack AUC** | Area under ROC curve for the inversion attack — lower is better |
| **Attack F1** | F1 score of the inversion attack classifier — lower is better |
| **Leakage score** | Mean per-dimension leakage score from the concept leakage detector |
| **Reconstruction MSE** | Mean-squared reconstruction error from the adversarial inverter |

---

## Section 10 — Running Experiments from VS Code

### 10.1 Integrated terminal

Open the terminal inside VS Code with `` Ctrl+` `` and run any of the commands above directly.

### 10.2 Launch configuration

Create `.vscode/launch.json` in the project root to run experiments with a single key press (`F5`):

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Baseline — Cora GCN",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/main.py",
            "args": [
                "--exp", "Inversion",
                "--method", "GIF",
                "--dataset_name", "cora",
                "--target_model", "GCN",
                "--attack_method", "trend_steal",
                "--unlearn_ratio", "0.05",
                "--num_runs", "5",
                "--is_gen_unlearn_request", "True",
                "--is_gen_unlearned_probs", "True"
            ],
            "console": "integratedTerminal"
        },
        {
            "name": "Proposed — Cora GCN (all privacy flags)",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/main.py",
            "args": [
                "--exp", "Inversion",
                "--method", "GIF",
                "--dataset_name", "cora",
                "--target_model", "GCN",
                "--attack_method", "trend_steal",
                "--unlearn_ratio", "0.05",
                "--num_runs", "5",
                "--is_gen_unlearn_request", "True",
                "--is_gen_unlearned_probs", "True",
                "--concept_leakage",
                "--privacy_mask",
                "--adversarial_training"
            ],
            "console": "integratedTerminal"
        },
        {
            "name": "Eval — Privacy comparison table",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/eval_privacy.py",
            "args": [
                "--dataset_name", "cora",
                "--target_model", "GCN",
                "--method", "GIF",
                "--attack_method", "trend_mia",
                "--is_gen_unlearn_request", "True",
                "--is_gen_unlearned_probs", "True"
            ],
            "console": "integratedTerminal"
        }
    ]
}
```

---

## Section 11 — Datasets

| Dataset | Nodes | Edges | Features | Classes |
|---------|-------|-------|----------|---------|
| Cora | 2,708 | 10,556 | 1,433 | 7 |
| Citeseer | 3,327 | 9,104 | 3,703 | 6 |
| PubMed | 19,717 | 88,648 | 500 | 3 |

Datasets are downloaded automatically via PyTorch Geometric on the first run and cached under `temp_data/raw_data/`.  
Processed splits are cached under `temp_data/processed_data/` so subsequent runs are faster.

---

## Section 12 — Troubleshooting

### CUDA not found

If you see `CUDA error` or the model falls back to CPU unexpectedly, verify your CUDA installation:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If `False`, either install the correct CUDA toolkit version or switch to the CPU build of PyTorch (see Section 5.2).

### torch-geometric import errors

Make sure the PyTorch version and the torch-geometric version are compatible.  
The tested combination is `torch==1.9.0+cu111` + `torch-geometric==2.0.3`.

### `ModuleNotFoundError` for project modules

Ensure you are running commands from the repository root and that your virtual environment is activated:

```bash
cd Graph-Unlearning-Inversion
source .venv/bin/activate   # macOS/Linux
python main.py ...
```

### Intermediate results already exist

If you re-run an experiment, cached unlearn requests and probability files may conflict.  
Delete the `temp_data/` directory to start fresh:

```bash
rm -rf temp_data/
```

---

## Citation

```bibtex
@inproceedings{zhang2026unlearning,
  title     = {Unlearning Inversion Attacks for Graph Neural Networks},
  author    = {Jiahao Zhang and Yilong Wang and Zhiwei Zhang and Xiaorui Liu and Suhang Wang},
  year      = {2026},
  booktitle = {The ACM International Conference on Web Search and Data Mining (WSDM)},
}
```
