from __future__ import annotations

from typing import Tuple, Sequence

import numpy as np
import matplotlib.pyplot as plt


def gini_coefficient(x: np.ndarray) -> float:
	if x.size == 0:
		return 0.0
	sorted_x = np.sort(x)
	cumx = np.cumsum(sorted_x)
	return float((x.size + 1 - 2 * np.sum(cumx) / cumx[-1]) / x.size)


def plot_tax_schedule(
	tax_schedule: Tuple[float, float, float, float, float],
	income_brackets: Sequence[float],
	title: str = "Optimal Marginal Tax Schedule",
) -> None:
	plt.rcParams["font.family"] = "Times New Roman"
	levels_pct = [int(round(100 * t)) for t in tax_schedule]
	# Prepend 0 and append 1.0 so x matches y (N+1 points for step plot)
	brackets = [0.0] + list(income_brackets) + [1.0]
	levels_ext = levels_pct + [levels_pct[-1]]
	fig, ax = plt.subplots(figsize=(10, 5))
	ax.step(brackets, levels_ext, where="post", color="darkblue", linewidth=2, label="Learned policy")
	ax.fill_between(brackets, 0, levels_ext, step="post", color="lightblue", alpha=0.3)
	ax.set_xticks(brackets)
	ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", r"$\infty$"])
	ax.set_xlabel("Income (×10³)", fontsize=14)
	ax.set_ylabel("Marginal Tax Rate (%)", fontsize=14)
	ax.set_title(title, fontsize=16)
	ax.legend(fontsize=12)
	ax.grid(True, linestyle="--", alpha=0.5)
	plt.tight_layout()
	plt.show()


