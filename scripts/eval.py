import argparse
from pathlib import Path

import yaml
import numpy as np
import pandas as pd

from stable_baselines3 import DQN

from abm_rl.env import TaxSimEnv, TaxSimParams
from abm_rl.utils import gini_coefficient, plot_tax_schedule


def make_env(params_dict: dict, seed: int | None = None) -> TaxSimEnv:
	params = TaxSimParams(**params_dict)
	return TaxSimEnv(params=params, seed=seed)


def main() -> None:
	parser = argparse.ArgumentParser(description="Evaluate trained model on ABM env")
	parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
	parser.add_argument("--model", type=str, required=True, help="Path to saved SB3 model.zip")
	parser.add_argument("--episodes", type=int, default=5, help="Number of eval episodes")
	args = parser.parse_args()

	with open(args.config, "r") as f:
		cfg = yaml.safe_load(f)

	seed = int(cfg.get("seed", 0))
	env_cfg = cfg.get("env", {})
	env = make_env(env_cfg, seed=seed + 123)

	model = DQN.load(args.model)

	returns = []
	gdp_values = []
	gini_values = []

	for _ in range(args.episodes):
		obs, _ = env.reset()
		done = False
		episode_return = 0.0
		while not done:
			action, _ = model.predict(obs, deterministic=True)
			obs, reward, done, _, _ = env.step(action)
			episode_return += float(reward)
		active = ~env.relocated
		e = env.e[active]
		h = env.h[active]
		w = env._wage()
		income = e * w * h
		gdp_values.append(float(np.sum(income)))
		gini_values.append(gini_coefficient(income))
		returns.append(episode_return)

	print(f"Avg return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
	print(f"Avg GDP: {np.mean(gdp_values):.2f}")
	print(f"Avg Gini: {np.mean(gini_values):.4f}")

	# Plot learned tax schedule from current model policy
	obs, _ = env.reset()
	action, _ = model.predict(obs, deterministic=True)
	policy = env.decode_action(int(action))
	plot_tax_schedule(policy, env.b, title="Learned Marginal Tax Schedule")


if __name__ == "__main__":
	main()


