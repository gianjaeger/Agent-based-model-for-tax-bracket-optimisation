import argparse
import os
from pathlib import Path

import yaml
import numpy as np
import pandas as pd

from stable_baselines3 import DQN

from abm_rl.env import TaxSimEnv, TaxSimParams
from abm_rl.callbacks import EvaluationCallback


def make_env(params_dict: dict, seed: int | None = None) -> TaxSimEnv:
	params = TaxSimParams(**params_dict)
	return TaxSimEnv(params=params, seed=seed)


def main() -> None:
	parser = argparse.ArgumentParser(description="Train DQN on ABM tax environment")
	parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
	parser.add_argument("--save_dir", type=str, default=None, help="Override output directory")
	parser.add_argument("--total_timesteps", type=int, default=None, help="Override timesteps")
	args = parser.parse_args()

	with open(args.config, "r") as f:
		cfg = yaml.safe_load(f)

	seed = int(cfg.get("seed", 0))
	np.random.seed(seed)

	save_dir = Path(args.save_dir or cfg.get("save_dir", "runs/default"))
	save_dir.mkdir(parents=True, exist_ok=True)

	env_cfg = cfg.get("env", {})
	train_env = make_env(env_cfg, seed=seed)
	eval_env = make_env(env_cfg, seed=seed + 1)

	eval_freq = int(cfg.get("eval_freq", 1000))
	n_eval_episodes = int(cfg.get("n_eval_episodes", 3))
	callback = EvaluationCallback(eval_env, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes, verbose=1)

	model = DQN("MlpPolicy", train_env, verbose=1, seed=seed)
	timesteps = int(args.total_timesteps or cfg.get("total_timesteps", 10000))
	model.learn(total_timesteps=timesteps, callback=callback)

	model_path = save_dir / "model.zip"
	model.save(model_path.as_posix())

	logs_df = pd.DataFrame(callback.logs)
	logs_df.to_csv((save_dir / "logs.csv").as_posix(), index=False)

	# Save final learned tax policy (deterministic)
	obs, _ = eval_env.reset()
	action, _ = model.predict(obs, deterministic=True)
	final_policy = eval_env.decode_action(int(action))
	with open(save_dir / "final_policy.txt", "w") as f:
		f.write(",".join(str(x) for x in final_policy))

	print(f"Saved model to {model_path}")
	print(f"Saved logs to {save_dir / 'logs.csv'}")
	print(f"Saved final policy to {save_dir / 'final_policy.txt'}")


if __name__ == "__main__":
	main()


