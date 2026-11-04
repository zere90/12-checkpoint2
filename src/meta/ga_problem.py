"""GA-P problem operators for cloud B2 (zones), GA with capacity penalties.

Genes are integer server indices, one per VM. Operators preserve zones but do
not repair CPU or memory overload. All stochastic choices use the supplied
numpy Generator. Evaluation calls the official checker exactly once.
"""

import numpy as np

from .interface import check


# All three weights gave five feasible tuning runs. Rho=1 had the lowest
# mean checker cost on seeds 100--104 (results/kabdulla_exp_summary.csv).
RHO = 1.0


def _validate_instance(d, variant):
    if d.get("story") != "B" or variant != 2 or d.get("variant") != 2:
        raise ValueError("ga_problem implements storyline B, variant 2")


def _assignment(individual, size=None, servers=None):
    a = np.asarray(individual)
    if a.ndim != 1 or a.dtype.kind not in "iu":
        raise ValueError("individual must be a one-dimensional integer assignment")
    if size is not None and len(a) != size:
        raise ValueError("individual must have one entry per VM")
    if np.any(a < 0) or (servers is not None and np.any(a >= servers)):
        raise ValueError("individual contains an invalid server index")
    return a


def allowed_servers(d, variant):
    """Derive each VM's legal servers from current instance data, not capacity."""
    _validate_instance(d, variant)
    zones = np.asarray(d["zone"])
    all_servers = np.arange(d["S"], dtype=np.int64)
    result = []
    for k, required in enumerate(d["vm_required_zone"]):
        allowed = all_servers if required == -1 else all_servers[zones == required]
        if len(allowed) == 0:
            raise ValueError(f"VM {k} has no server in its required zone")
        result.append(allowed.copy())
    return result


def random_individual(d, variant, rng):
    """Independently sample a zone-legal server for every VM; allow overload."""
    return np.array([rng.choice(a) for a in allowed_servers(d, variant)],
                    dtype=np.int64)


def capacity_violation(d, individual):
    """Sum relative CPU and memory overload across servers (dimensionless).

    sum_s max(cpu_load_s / cpu_cap_s - 1, 0)
        + max(mem_load_s / mem_cap_s - 1, 0).
    Normalization prevents memory units from dominating CPU units by scale.
    """
    a = _assignment(individual, d["M"], d["S"])
    cpu_cap = np.asarray(d["cpu_cap"], dtype=float)
    mem_cap = np.asarray(d["mem_cap"], dtype=float)
    if np.any(cpu_cap <= 0) or np.any(mem_cap <= 0):
        raise ValueError("server capacities must be positive")
    cpu = np.bincount(a, weights=d["cpu_req"], minlength=d["S"])
    mem = np.bincount(a, weights=d["mem_req"], minlength=d["S"])
    return float(np.maximum(cpu / cpu_cap - 1.0, 0.0).sum()
                 + np.maximum(mem / mem_cap - 1.0, 0.0).sum())


def evaluate(d, variant, individual):
    """Return (penalized selection fitness, true checker cost, feasibility).

    RHO can be set by the experiment before a solve call, since the agreed
    four-function interface has no params argument for this function. Do not
    run different RHO settings concurrently in the same Python process.
    Malformed representations raise ValueError. A structurally valid external
    assignment with a wrong zone gets infinite fitness, its true checker cost,
    and feasible=False. Our own operators never generate such assignments.
    """
    _validate_instance(d, variant)
    a = _assignment(individual, d["M"], d["S"])
    rho = float(RHO)
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("RHO must be finite and positive")
    cost, problems = check(d, variant, a)
    violation = capacity_violation(d, a)
    required = np.asarray(d["vm_required_zone"])
    wrong_zone = np.any((required != -1) & (np.asarray(d["zone"])[a] != required))
    fitness = float("inf") if wrong_zone else float(cost + rho * violation)
    return fitness, float(cost), not bool(problems)


def crossover(parent1, parent2, rng):
    """Uniform crossover; both parents must be legal for the same instance.

    Each VM inherits its server from a parent at the same position, so zone
    validity is preserved without needing instance data. Parents are unchanged.
    """
    p1 = _assignment(parent1)
    p2 = _assignment(parent2, len(p1))
    return np.where(rng.random(len(p1)) < 0.5, p1, p2).astype(np.int64)


def mutate(d, variant, individual, rng):
    """Change one or two distinct mutable VM genes, preserving only zones.

    Exclude the current server so selected genes really change. If fewer than
    two genes have an alternative server, change as many as are available.
    Return an independent array, leaving the input unchanged.
    """
    a = _assignment(individual, d["M"], d["S"]).astype(np.int64, copy=True)
    allowed = allowed_servers(d, variant)
    if any(a[k] not in choices for k, choices in enumerate(allowed)):
        raise ValueError("mutation requires a zone-valid parent")
    mutable = [k for k, choices in enumerate(allowed) if len(choices) > 1]
    if mutable:
        count = min(int(rng.integers(1, 3)), len(mutable))
        for k in rng.choice(mutable, size=count, replace=False):
            choices = allowed[k]
            a[k] = rng.choice(choices[choices != a[k]])
    return a
