# Continuous self-correction of end-to-end driving models from human interventions

This repository provides the data and scripts required to reproduce the main figures in the paper.

## Overview

This repository includes:

- processed experimental results used for plotting the figures;
- Python scripts for reproducing the figures with one-command execution.

The full training and deployment code is not publicly available due to proprietary restrictions.

---

## Installation

We recommend Python 3.10 or later.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Reproducing the figures

All figures can be reproduced by running the corresponding Python scripts below.

### Reward comparison before takeover
```bash
python plot_takeover_reward.py
```

### Performance comparison on normal test dataset
```bash
python plot_performance_on_normal_dataset.py
```
### Performance comparison on takeover test dataset
```bash
python plot_performance_on_takeover_dataset.py
```
### Continue learning with growing takeover data
```bash
python plot_continue_learn.py
```