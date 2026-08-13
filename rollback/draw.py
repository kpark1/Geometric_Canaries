from typing import Any

import numpy as np

from rollback_types import ProveRunResult


def plot_timeline(
    res: ProveRunResult,
    title: str = "CoT signal timeline",
) -> None:
    import matplotlib.pyplot as plt

    _, axes = plt.subplots(
        2, 1, figsize=(13, 5.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    x = np.arange(len(res["entropy"]))
    axes[0].plot(x, res["entropy"], lw=0.8)
    axes[0].set_ylabel("entropy")
    axes[1].plot(x, res["logprob"], lw=0.8, color="tab:orange")
    axes[1].set_ylabel("logprob")
    axes[1].set_xlabel("generated token")
    for s in res["segments"]:
        for ax in axes:
            ax.axvline(s["tok_start"], color="gray", lw=0.5, alpha=0.5)
        axes[0].text(
            s["tok_start"],
            axes[0].get_ylim()[1] * 0.95,
            f"{s['score']:+.1f}",
            fontsize=7,
            color="gray",
        )
    for e in res["events"]:
        if e["type"] == "rollback":
            for ax in axes:
                ax.axvline(e["to_tok"], color="red", lw=1.5, alpha=0.8)
    axes[0].set_title(title + f"  ({res['rollbacks']} rollbacks)")
    plt.tight_layout()
    plt.show()
