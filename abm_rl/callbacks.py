from __future__ import annotations

from typing import List, Dict, Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


def gini_coefficient(x: np.ndarray) -> float:
	if x.size == 0:
		return 0.0
	sorted_x = np.sort(x)
	cumx = np.cumsum(sorted_x)
	return float((x.size + 1 - 2 * np.sum(cumx) / cumx[-1]) / x.size)


class EvaluationCallback(BaseCallback):
	def __init__(self, eval_env, eval_freq: int = 500, n_eval_episodes: int = 5, verbose: int = 1):
		super().__init__(verbose)
		self.eval_env = eval_env
		self.eval_freq = eval_freq
		self.n_eval_episodes = n_eval_episodes
		self.logs: List[Dict[str, Any]] = []

	def _on_step(self) -> bool:
		if self.n_calls % self.eval_freq == 0:
			avg_reward, avg_gdp, avg_gini = 0.0, 0.0, 0.0
			for _ in range(self.n_eval_episodes):
				obs, _ = self.eval_env.reset()
				done = False
				total_reward = 0.0
				gdp_vals: List[float] = []
				while not done:
					action, _ = self.model.predict(obs, deterministic=True)
					obs, reward, done, _, _ = self.eval_env.step(action)
					total_reward += float(reward)

					active = ~self.eval_env.relocated
					e = self.eval_env.e[active]
					h = self.eval_env.h[active]
					w = self.eval_env._wage()
					income = e * w * h
					gdp_vals.append(float(np.sum(income)))

					if income.size > 0:
						avg_gini += gini_coefficient(income)

				avg_reward += total_reward
				avg_gdp += float(np.mean(gdp_vals)) if gdp_vals else 0.0

			self.logs.append(
				{
					"step": int(self.n_calls),
					"avg_reward": avg_reward / self.n_eval_episodes,
					"avg_gdp": avg_gdp / self.n_eval_episodes,
					"avg_gini": avg_gini / self.n_eval_episodes,
				}
			)
		return True


