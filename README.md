## ABM-RL: Learning Tax Schedules with Reinforcement Learning

This repository implements a small agent-based economy where a planner learns a revenue-maximizing marginal tax schedule across 5 income brackets using DQN. The code is organized as a package with config-driven training and evaluation.

### Structure
- `abm_rl/`: package with the Gymnasium environment and utilities
  - `env.py`: `TaxSimEnv` and `TaxSimParams`
  - `callbacks.py`: `EvaluationCallback` and `gini_coefficient`
  - `utils.py`: plotting and metrics
- `configs/`: experiment configuration files (YAML)
  - `default.yaml`: default hyperparameters and environment parameters
- `scripts/`: CLI entry points
  - `train.py`: trains a DQN agent and saves model and logs
  - `eval.py`: loads a model, evaluates metrics, and plots the tax schedule
- `Preliminary-ABM.ipynb`: original exploratory notebook (kept for reference)

### Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Train
```bash
python scripts/train.py --config configs/default.yaml
```
Artifacts are saved to the `save_dir` specified in the config (default `runs/default/`):
- `model.zip`: trained SB3 model
- `logs.csv`: evaluation logs over training
- `final_policy.txt`: comma-separated marginal tax rates

You can override output directory or timesteps:
```bash
python scripts/train.py --config configs/default.yaml --save_dir runs/exp1 --total_timesteps 20000
```

### Evaluate and plot
```bash
python scripts/eval.py --config configs/default.yaml --model runs/default/model.zip --episodes 5
```
This prints average return, GDP, and Gini, and displays a step plot of the learned marginal tax schedule.

### Configuration
Edit `configs/default.yaml` to control:
- Training: `total_timesteps`, `eval_freq`, `n_eval_episodes`, `save_dir`, `seed`
- Environment: population size, bracket thresholds, allowed marginal rates, relocation penalty, etc.

### Notes
- The action space enumerates non-decreasing 5-rate schedules across 4 thresholds.
- Redistribution is modeled via a lump-sum transfer proportional to revenue.
- Relocation reduces labor supply to zero for agents who leave.


