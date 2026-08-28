import numpy as np


def nom_trace(per_segment: list[float]) -> float:
    """Aggregate NoM (number-of-mismatches) per-segment 0/1 flags to a trace count."""
    return float(sum(per_segment))


def aae_trace(per_segment: list[float]) -> float:
    """Aggregate AAE per-segment angles (radians) to their trace mean."""
    return float(np.mean(per_segment)) if per_segment else 0.0
