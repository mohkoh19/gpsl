# GPSL

Official PyTorch codebase for GPSL (Parallel Split Learning with Global Sampling). [ToDo: Arxiv link]

## Method

GPSL is a method for distributed deep learning. At a high level, it samples client data to form fixed-size global batches that better reflect the overall distribution. This improves performance:
- without requiring fixed or proportional local batch sizes, which introduce rounding bias and data imbalance,
- and without increasing the effective batch size with the number of clients, which harms generalization and training efficiency.

<img src="https://github.com/user-attachments/assets/316a259b-223d-431e-8499-575756d05254" alt="Alt text" width="400"/>

## Evaluation

GPSL is evaluated on the CIFAR-10 dataset under IID, mild non-IID, and severe non-IID settings with up to 128 clients.

- **Accuracy**: Under severe non-IID conditions with 128 clients, GPSL achieves **84.31%** test accuracy—matching centralized learning and outperforming baseline methods by a large margin (FLS: 58.88%, FPLS: 70.05%).
- **Batch Deviation**: GPSL maintains low and stable batch deviation across training steps, closely matching centralized sampling, while FLS and FPLS exhibit high variability and bias.
- **Training Time**: GPSL significantly reduces total training time by avoiding unnecessary client data depletion, especially in low batch size settings.
- **Robustness**: Performance is stable across a wide range of client counts (K = 16–128) and global batch sizes (B = 64–256), demonstrating strong scalability.

<div align="center">
  <img src="https://github.com/user-attachments/assets/dcc36b81-1673-48c8-9f5c-fd3eb6a8794c" alt="Test Accuracy Curve" width="52%" style="margin-right: 1%">
  <img src="https://github.com/user-attachments/assets/5f9d199e-c7b4-4f89-86ef-58a28e464504" alt="Batch Deviation" width="45%">
</div>
<p align="center">
  <b>Left:</b> Test accuracy across epochs under severe non-IID conditions.     
  <b>Right:</b> Batch deviation over time under severe non-IID conditions.
</p>


## Code Structure

```
.
├── configs/                   # Experiment configurations (Hydra-based)
│   ├── experiment/           # Setup for different experiments (IID, non-IID, etc.)
│   ├── model/                # Model hyperparameters
│   ├── data/                 # Dataset configs
│   ├── hparams_search/       # Hyperparameter search settings
│   ├── callbacks/, logger/, trainer/, etc.
│   └── train.yaml            # Default training configuration
├── data/                     # CIFAR-10 dataset files (downloaded)
├── src/                      # Core training and model code
│   ├── train.py              # Main training script
│   ├── data/                 # Data loading and augmentation logic
│   ├── models/               # GPSL and baseline model implementations
│   └── utils/                # Utility functions
├── Dockerfile                # Reproducible environment specification
├── Makefile                  # Simplified Docker and training commands
├── README.md                 # Project documentation
└── pyproject.toml            # Python package and dependency specification
```

This project builds on the [Lightning-Hydra Template](https://github.com/ashleve/lightning-hydra-template), which provides a modular, scalable, and reproducible research framework using PyTorch Lightning and Hydra. Refer to that repository for additional details on how to structure, extend, and use the components effectively.

## Launching Experiments

We provide a Docker-based setup and standardized command-line interface to reproduce all experiments from the paper.

### Setup

1. **Build the Docker container:**
   ```bash
   make docker-build
   ```

2. **Run the container (requires `.env` file):**
   ```bash
   make docker-run
   ```

   > Note: Even if you don't use environment variables, an empty `.env` file must be present.

3. **Access the running container:**
   ```bash
   make docker-exec
   ```

### Reproducing Results

Use the following commands to reproduce the tables from the paper:

#### Tables 1–3 (IID, Mild, and Severe Non-IID)
```bash
python src/train.py -m hparams_search=tables_1_2_3 experiment=gpsl/iid
python src/train.py -m hparams_search=tables_1_2_3 experiment=gpsl/non_iid_mild
python src/train.py -m hparams_search=tables_1_2_3 experiment=gpsl/non_iid_severe
```

#### Table 4 (Effect of Batch Size)
```bash
# Distributed case
python src/train.py -m hparams_search=table_4 experiment=gpsl/non_iid_severe

# Centralized baseline
python src/train.py -m hparams_search=table_4_cl experiment=gpsl/iid
```

#### Centralized Learning Baseline (Figure 3, etc.)
```bash
python src/train.py -m experiment=gpsl/cl seed=1,2,3,4,5
```

### Notes
- Some experiment-specific configurations (e.g., logging, hardware settings) may need to be adjusted before running.
- For hyperparameter search and logging, ensure your API keys are configured in the `.env` file or via your preferred method.

## License
For license details, please refer to the [LICENSE](https://github.com/mohkoh19/gpsl/LICENSE) in this repository

## Citation

If you find this repository helpful, please consider starring ⭐ the project and citing it as:
```
@article{kohankhaki2024gpsl,
  title={Parallel Split Learning with Global Sampling},
  author={Kohankhaki, Mohammad and Ayad, Ahmad and Barhoush, Mahdi and Schmeink, Anke},
  journal={arXiv preprint arXiv:2405.00000},
  year={2024}
}
```
