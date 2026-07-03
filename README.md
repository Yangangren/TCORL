# TCORL: Continual improvement of deployed autonomous vehicles from human intervention

This repository provides the processed experimental results and Python scripts used to reproduce the main quantitative figures in the manuscript:

**Continual improvement of deployed autonomous vehicles from human intervention**

## Scope of this repository

This repository is intended as a figure-reproduction and code-availability package for peer review and public release. It contains:

- processed experimental results used to generate the main quantitative plots;
- Python plotting scripts for reproducing the released quantitative figure files;
- links and descriptions for representative takeover cases, trajectory-correction results, and real-vehicle test videos.

This repository does **not** contain the raw production-vehicle logs, multi-view camera/LiDAR data, HD-map annotations, production end-to-end driving stack, full training pipeline, model weights, vehicle-log interfaces, or in-vehicle deployment code. These materials cannot be publicly released because of proprietary industrial restrictions, data-governance requirements, road-user privacy, and vehicle-safety constraints.

The algorithmic details of TCORL, including trajectory-level exploration, takeover-aware reward computation, group-relative advantage estimation, and the hybrid reinforcement-learning/imitation-learning objective, are described in the Methods section of the manuscript.

## Repository contents

The repository includes:

```text
TCORL/
├── README.md
├── requirements.txt
├── reward_data/
├── predeploy_model_on_normal_test_dataset/
├── postrain_model_on_normal_test_dataset/
├── predeploy_model_on_takeover_test_dataset/
├── postrain_model_on_takeover_test_dataset/
├── test_ADE_with_takeover_training_data/
├── test_ADE_with_mixed_training_data/
├── test_SCORE_with_takeover_training_data/
├── test_SCORE_with_mixed_training_data/
├── test_takeover_dataset.json
├── plot_takeover_reward.py
├── plot_performance_on_normal_dataset.py
├── plot_performance_on_takeover_dataset.py
└── plot_continue_learn.py
```

## Data, code, and video availability

The datasets used in the study are derived from proprietary real-world data collected from production vehicles and are not publicly available. Access to derived or anonymized data may be considered by the corresponding authors upon reasonable request and subject to review of the intended use, institutional approval, and company data-governance requirements.

Processed experimental results and Python plotting scripts used to reproduce the main quantitative plots are publicly available in this repository:

<https://github.com/Yangangren/TCORL>

Representative takeover cases are publicly available in this Google Drive folder:

<https://drive.google.com/drive/folders/1dR_USiRUyPHJ6usW0Q6AIY-n2Snn5zy6?usp=sharing>

The trajectory-correction results on takeover datasets are available in this Google Drive folder:

<https://drive.google.com/drive/folders/1AMAHWusU_N7cWxBzbbVsFwkJBEUmb3Zy?usp=sharing>

Real-vehicle test videos are publicly available in this Google Drive folder:

<https://drive.google.com/drive/folders/1Ydu5OOgs3XMfwRMQpB4K97-MU5D18OGH?usp=sharing>

These supplementary videos are not included in this repository because of file-size constraints.

## System requirements

- Operating system: Linux, macOS, or Windows
- Python: 3.10 or later
- Hardware: no non-standard hardware is required for the public plotting scripts; the scripts run on CPU
- Python dependencies: NumPy, pandas, Matplotlib, and SciPy

The scripts were tested in the following environment:

- Python 3.13.12
- numpy 2.4.3
- pandas 3.0.1
- matplotlib 3.10.8
- scipy 1.17.1

## Installation

Install the required dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Typical installation time is less than 5 minutes on a normal desktop computer with an internet connection. If the dependencies are already cached, installation is typically less than 1 minute.

## Demo

This demo provides a quick test of the repository by running one representative plotting script. It is intended to verify the installation, data paths, and figure-generation workflow. Full figure-reproduction instructions are provided in the next section.

A minimal demonstration can be run with the reward-comparison script:

```bash
python plot_takeover_reward.py
```

Expected output:

- `reward_neighbor_collision.png` and `reward_neighbor_collision.pdf`
- `reward_progress.png` and `reward_progress.pdf`
- `reward_route.png` and `reward_route.pdf`
- `reward_onroad.png` and `reward_onroad.pdf`

Expected runtime is less than 1 minute on a normal desktop computer. In our local test environment, this demo completed in approximately 1.2 seconds.

## Reproducing the figures

The main quantitative plots generated from processed numerical results can be reproduced by running the corresponding Python scripts below.

| Manuscript result | Script | Main input data | Expected output |
|---|---|---|---|
| Reward comparison before takeover | `plot_takeover_reward.py` | `reward_data/` | `reward_neighbor_collision`, `reward_progress`, `reward_route`, `reward_onroad` |
| Performance on the normal test dataset | `plot_performance_on_normal_dataset.py` | `predeploy_model_on_normal_test_dataset/`, `postrain_model_on_normal_test_dataset/` | `normal_ade`, `normal_cost` |
| Performance on the takeover test dataset | `plot_performance_on_takeover_dataset.py` | `predeploy_model_on_takeover_test_dataset/`, `postrain_model_on_takeover_test_dataset/`, `test_takeover_dataset.json` | `takeover_all_score`, `takeover_step_score` |
| Continual learning with growing takeover data | `plot_continue_learn.py` | `test_ADE_with_takeover_training_data/`, `test_ADE_with_mixed_training_data/`, `test_SCORE_with_takeover_training_data/`, `test_SCORE_with_mixed_training_data/` | `continue_learn_ade`, `continue_learn_reward` |

### Reward comparison before takeover

```bash
python plot_takeover_reward.py
```

This script reproduces the reward-distribution plots comparing pre-takeover model trajectories and human-corrected trajectories.

### Performance comparison on the normal test dataset

```bash
python plot_performance_on_normal_dataset.py
```

This script reproduces the normal-driving evaluation plots, including trajectory displacement error and collision-related metrics.

### Performance comparison on the takeover test dataset

```bash
python plot_performance_on_takeover_dataset.py
```

This script reproduces the takeover-scenario evaluation plots, including the overall driving score and the pre-takeover step-wise safety score.

### Continual learning with growing takeover data

```bash
python plot_continue_learn.py
```

This script reproduces the data-scale ablation results for post-training with increasing proportions of takeover data, with and without matched expert-like data.

Running all four scripts reproduces the released quantitative figure files. In our local test environment, the four scripts completed in approximately 5 seconds in total. On a normal desktop computer, the expected runtime is less than 2 minutes.

## Instructions for use

Each script loads the processed data files included in this repository and writes the corresponding figures to the repository root. To use the scripts on new data, replace the relevant input files with files that follow the same format and path structure, or update the input path constants at the top of each script.

Main inputs:

- `reward_data/human_traj_reward.npy` and `reward_data/model_traj_reward.npy` for `plot_takeover_reward.py`
- `predeploy_model_on_normal_test_dataset/` and `postrain_model_on_normal_test_dataset/` for `plot_performance_on_normal_dataset.py`
- `predeploy_model_on_takeover_test_dataset/`, `postrain_model_on_takeover_test_dataset/`, and `test_takeover_dataset.json` for `plot_performance_on_takeover_dataset.py`
- `test_ADE_with_takeover_training_data/`, `test_ADE_with_mixed_training_data/`, `test_SCORE_with_takeover_training_data/`, and `test_SCORE_with_mixed_training_data/` for `plot_continue_learn.py`

## Limitations of the public package

The public package supports reproduction of the released quantitative figure files from processed results. It does not support full retraining of the production driving model or redeployment on a vehicle.

Full model training and vehicle deployment require proprietary datasets, the production E2E driving stack, model weights, vehicle-log interfaces, and in-vehicle computing infrastructure. These components are not part of this repository and cannot be publicly released for the reasons stated above.

## License

The public plotting scripts and associated documentation in this repository are released under the MIT License. See `LICENSE` for details.

The license applies only to the public figure-reproduction code and documentation in this repository. It does not apply to the raw production-vehicle logs, multi-view camera/LiDAR data, HD-map annotations, production E2E driving stack, model weights, vehicle-log interfaces, or in-vehicle deployment code, which are not included in this repository.