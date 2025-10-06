from dataclasses import dataclass
from typing import Tuple, Dict, Any

import itertools
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box


@dataclass
class TaxSimParams:
	N: int = 100
	eta: float = 2.0
	psi: float = 0.8
	A: float = 1.0
	lambda_: float = 0.9
	z_foreign: float = 0.8
	# Income efficiency, relocation sensitivity, and relocation cost ranges
	epsilon_low: float = 0.8
	epsilon_high: float = 1.2
	phi_low: float = 0.2
	phi_high: float = 0.4
	kappa_low: float = 0.5
	kappa_high: float = 0.9
	# Bracket thresholds and allowed marginal rates
	brackets: Tuple[float, float, float, float] = (0.2, 0.4, 0.6, 0.8)
	tau_values: Tuple[float, float, float, float, float] = (0.05, 0.1, 0.15, 0.2, 0.25)
	max_steps: int = 50
	relocation_penalty: float = 0.05


class TaxSimEnv(Env):
	"""Gymnasium environment simulating a tax planner in a heterogeneous-agent economy.

	Observation: [mean income, income std, num relocated, total revenue]
	Action: index selecting a non-decreasing vector of 5 marginal tax rates across 4 brackets
	Reward: revenue - relocation_penalty * num_newly_relocated
	"""

	metadata = {"render_modes": []}

	def __init__(self, params: TaxSimParams | None = None, seed: int | None = None) -> None:
		super().__init__()
		self.params = params or TaxSimParams()
		self._rng = np.random.default_rng(seed)

		self.N = self.params.N
		self.e = self._rng.uniform(self.params.epsilon_low, self.params.epsilon_high, self.N)
		self.phi = self._rng.uniform(self.params.phi_low, self.params.phi_high, self.N)
		self.kappa = self._rng.uniform(self.params.kappa_low, self.params.kappa_high, self.N)

		self.z_foreign = self.params.z_foreign
		self.lambda_ = self.params.lambda_
		self.eta = self.params.eta
		self.psi = self.params.psi
		self.A = self.params.A
		self.b = list(self.params.brackets)
		self.tau_vals = list(self.params.tau_values)

		self.bracket_space = sorted(
			set(tuple(sorted(t)) for t in itertools.product(self.tau_vals, repeat=5))
		)
		self.tau_to_idx: Dict[Tuple[float, ...], int] = {tuple(x): i for i, x in enumerate(self.bracket_space)}
		self.idx_to_tau: Dict[int, Tuple[float, ...]] = {i: x for i, x in enumerate(self.bracket_space)}

		self.action_space = Discrete(len(self.bracket_space))
		self.observation_space = Box(low=0, high=np.inf, shape=(4,), dtype=np.float32)

		self.reset()

	def encode_action(self, tau_tuple: Tuple[float, float, float, float, float]) -> int:
		clean = tuple(sorted(float(x) for x in np.array(tau_tuple).flatten()))
		return self.tau_to_idx[clean]

	def decode_action(self, index: int) -> Tuple[float, float, float, float, float]:
		return self.idx_to_tau[int(index)]

	def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None):
		super().reset(seed=seed)
		if seed is not None:
			self._rng = np.random.default_rng(seed)
		self.h = np.ones(self.N)
		self.relocated = np.zeros(self.N, dtype=bool)
		self.timestep = 0
		self.total_revenue = 0.0
		self.total_relocated = 0
		return self._get_obs(), {}

	def _get_obs(self) -> np.ndarray:
		active = ~self.relocated
		z = self.e[active] * self._wage() * self.h[active]
		mu_z = float(np.mean(z)) if len(z) > 0 else 0.0
		sigma_z = float(np.std(z)) if len(z) > 0 else 0.0
		return np.array([mu_z, sigma_z, float(np.sum(self.relocated)), float(self.total_revenue)], dtype=np.float32)

	def _wage(self) -> float:
		L = float(np.sum(self.h[~self.relocated]))
		return float(self.A * self.psi * (L ** (self.psi - 1))) if L > 0 else 0.0

	def step(self, action: int):
		tau = self.decode_action(action)
		w = self._wage()
		z = self.e * w * self.h
		z[self.relocated] = 0.0

		tau_z = np.piecewise(
			z,
			[
				z < self.b[0],
				(z >= self.b[0]) & (z < self.b[1]),
				(z >= self.b[1]) & (z < self.b[2]),
				(z >= self.b[2]) & (z < self.b[3]),
				z >= self.b[3],
			],
			tau,
		)

		taxes = tau_z * z
		R_t = float(np.sum(taxes))
		x = self.lambda_ * R_t / self.N
		stay_payoff = (z - taxes) + x
		leave_payoff = self.z_foreign - self.kappa

		newly_relocated = (stay_payoff < leave_payoff) & (~self.relocated)
		self.relocated |= newly_relocated
		self.h[self.relocated] = 0.0

		self.total_revenue = R_t
		self.total_relocated = int(np.sum(self.relocated))
		self.timestep += 1
		terminated = self.timestep >= self.params.max_steps
		reward = R_t - self.params.relocation_penalty * float(np.sum(newly_relocated))

		return self._get_obs(), float(reward), bool(terminated), False, {}


