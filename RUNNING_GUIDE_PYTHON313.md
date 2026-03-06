# Running Guide — Python 3.13

## Graph Unlearning Inversion — Step-by-Step Setup with VS Code

---

## Section 1 — Project Overview

This repository provides a PyTorch implementation of **"Unlearning Inversion Attacks for Graph Neural Networks"** (WSDM '26).

Graph unlearning algorithms allow a trained Graph Neural Network (GNN) to "forget" specific nodes or edges after training.
The paper demonstrates that an adversary who observes the model *before* and *after* unlearning can reconstruct sensitive information about the forgotten data — a serious privacy risk known as an **unlearning inversion attack**.

The codebase evaluates these attacks across five GNN architectures (GCN, GAT, GIN, SGC, MLP) and three benchmark datasets (Cora, Citeseer, PubMed).
It also includes a **Privacy-Preserving Training Framework** with four modular components that significantly reduce the success rate of inversion attacks while preserving node-classification utility.

**This repository now fully supports Python 3.13** (and modern PyTorch / torch-geometric).
See `PYTHON313_COMPATIBILITY.md` for a summary of the compatibility changes that were made.

---

## Section 2 — System Requirements

### Core tools

| Tool | Required version | Notes |
|------|-----------------|-------|
| **Python** | **3.13** | Must be installed and on your `PATH` |
| **Git** | Any recent release | For cloning the repository |
| **Visual Studio Code** | Latest stable | Download from <https://code.visualstudio.com/> |

### Recommended VS Code extensions

Install each extension from the **Extensions** panel (`Ctrl+Shift+X` / `Cmd+Shift+X`):

| Extension | Publisher ID | Why it helps |
|-----------|-------------|--------------|
| **Python** | `ms-python.python` | IntelliSense, linting, virtual-environment management |
| **Pylance** | `ms-python.vscode-pylance` | Fast type checking and import resolution |
| Jupyter *(optional)* | `ms-toolsai.jupyter` | Useful for interactive experimentation |

---

## Section 3 — Clone the Repository

Open a terminal (or the VS Code integrated terminal with `` Ctrl+` ``):

```bash
git clone https://github.com/iqbal2009048/Graph-Unlearning-Inversion.git
cd Graph-Unlearning-Inversion
```

---

## Section 4 — Create and Activate a Python 3.13 Virtual Environment

### 4.1 Verify your Python version

```bash
python3.13 --version
# Expected output: Python 3.13.x
```

If the command is not found, make sure Python 3.13 is installed and added to your `PATH`.

### 4.2 Create the virtual environment

```bash
python3.13 -m venv .venv
```

> On some systems the executable is simply `python` or `python3` — use whichever resolves to 3.13.

### 4.3 Activate the virtual environment

| Platform | Command |
|----------|---------|
| macOS / Linux | `source .venv/bin/activate` |
| Windows (CMD) | `.venv\Scripts\activate.bat` |
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |

Your terminal prompt will show `(.venv)` once the environment is active.

### 4.4 Select the interpreter in VS Code

1. Press `Ctrl+Shift+P` (macOS: `Cmd+Shift+P`) to open the Command Palette.
2. Type **Python: Select Interpreter** and press `Enter`.
3. Choose the interpreter that points to `.venv/bin/python` (Windows: `.venv\Scripts\python.exe`).

---

## Section 5 — Install Dependencies

### 5.1 Upgrade pip

```bash
pip install --upgrade pip
```

### 5.2 Install PyTorch (latest stable)

**GPU (CUDA 12.x) — recommended if you have an NVIDIA GPU:**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CPU-only — if you do not have a CUDA-capable GPU:**

```bash
pip install torch torchvision torchaudio
```

> To check which CUDA version your GPU supports run `nvidia-smi` in a terminal.
> Replace `cu121` with the appropriate suffix (e.g. `cu118` for CUDA 11.8) if needed.
> See <https://pytorch.org/get-started/locally/> for the full matrix.

### 5.3 Install PyTorch Geometric (latest stable)

```bash
pip install torch-geometric
```

PyTorch Geometric will automatically install its required sparse/scatter dependencies.

### 5.4 Install remaining dependencies

```bash
pip install numpy scikit-learn tqdm
```

### 5.5 Verify the installation

```bash
python -c "import torch; import torch_geometric; import numpy; \
    print('PyTorch', torch.__version__); \
    print('torch-geometric', torch_geometric.__version__); \
    print('NumPy', numpy.__version__); \
    print('CUDA available:', torch.cuda.is_available())"
```

---

## Section 6 — Project Structure

```
Graph-Unlearning-Inversion/
├── main.py                        # Entry point for all experiments
├── eval_privacy.py                # Runs baseline + proposed and prints comparison table
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
│   ├── leakage_detector.py        # Concept leakage detector
│   ├── privacy_mask.py            # KAN privacy mask layer
│   ├── privacy_transform.py       # Privacy-certified embedding transform
│   ├── adversarial_inverter.py    # Adversarial inverter for min-max training
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

The baseline reproduces the original WSDM '26 paper results — no privacy flags are set.

### 7.1 Quick start (Cora, GCN, GIF unlearning, trend-steal attack)

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

## Section 8 — Run the Privacy Framework

Enable the privacy components by adding `--concept_leakage`, `--privacy_mask`, and `--adversarial_training`:

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

`eval_privacy.py` runs both the baseline and proposed experiments back-to-back and prints a formatted comparison table.

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

Open the VS Code terminal with `` Ctrl+` `` and run any command from the sections above directly.
Make sure the virtual environment is activated (your prompt should show `(.venv)`).

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

### Wrong Python version

Confirm you are using Python 3.13 inside the virtual environment:

```bash
python --version
# Expected: Python 3.13.x
```

If it shows a different version, re-create the virtual environment using `python3.13 -m venv .venv`.

### CUDA not found

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If `False`, either install the correct CUDA toolkit or use the CPU-only PyTorch build (see Section 5.2).

### torch-geometric import errors

Make sure PyTorch and torch-geometric versions are compatible.
Visit <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html> for the official compatibility matrix.

### `ModuleNotFoundError` for project modules

Ensure you are running commands from the repository root and that the virtual environment is activated:

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
