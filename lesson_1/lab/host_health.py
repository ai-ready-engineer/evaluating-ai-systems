def deviation_score(cpu_dev: float, mem_dev: float) -> float:
    """Combined anomaly magnitude for a host (distance from the all-normal point).

    Args:
        cpu_dev: normalized CPU deviation from baseline; valid range [0.0, 1.0].
        mem_dev: normalized memory deviation from baseline; valid range [0.0, 1.0].

    Returns:
        Euclidean distance from (0, 0): 0.0 = healthy, larger = more anomalous.
        Range [0.0, sqrt(2) ~= 1.41].
    """
    if not 0.0 <= cpu_dev <= 1.0:
        raise ValueError(f"cpu_dev must be in [0, 1], got {cpu_dev}")
    if not 0.0 <= mem_dev <= 1.0:
        raise ValueError(f"mem_dev must be in [0, 1], got {mem_dev}")
    return (cpu_dev**2 + mem_dev**2) ** 0.5
