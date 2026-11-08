"""Simulated annealing, configuration "SA, transfer" (role SA-solo, T12, storyline B).

File name: src/meta/sa.py is the SA-I file of the SA pair, so the solo SA lives here.
The state is `assign`, the server of every VM. The only move is a transfer: one VM goes to
another server that its zone (B2) or GPU need (B3) allows. Only feasible moves are accepted: a
transfer that overloads CPU or memory is still one evaluation, and it is rejected.

Counting: the start is one evaluation and every proposal is one evaluation. With params=None the
3000-proposal calibration walk of the rule of thumb runs inside solve and is charged to the budget;
with params={"T0": ..., "Tend": ...} there is no walk and K = budget - 1 proposals remain.
"""
import math
import random

import numpy as np

import checkpoint4_generator_PUBLIC as gen
from src.meta.interface import check

WALK_LENGTH = 3000
EPS = 1e-12


def _latency(u):
    """The same formula as the checker latency_B, so that the table holds the checker's values."""
    return gen.latency_B(min(u, 1.0))


class Problem:
    """Plain Python copies of the data, the allowed servers of each VM and a latency table."""

    def __init__(self, d, variant):
        if d.get('story') != 'B':
            raise ValueError('sa_solo.solve handles storyline B only')
        self.S, self.M, self.variant = int(d['S']), int(d['M']), variant
        self.cpu_cap = [int(x) for x in d['cpu_cap']]
        self.mem_cap = [int(x) for x in d['mem_cap']]
        self.cpu_req = [int(x) for x in d['cpu_req']]
        self.mem_req = [int(x) for x in d['mem_req']]
        self.allowed = [self._allowed(d, k) for k in range(self.M)]
        self.movable = [k for k in range(self.M) if len(self.allowed[k]) >= 2]
        total = sum(self.cpu_req)
        # the loads are integers, so latency[l][load] replaces every later floating-point evaluation
        self.latency = [[_latency(load / self.cpu_cap[l]) for load in range(total + 1)] for l in range(self.S)]

    def _allowed(self, d, k):
        servers = range(self.S)
        if self.variant == 2 and int(d['vm_required_zone'][k]) != -1:
            return [l for l in servers if int(d['zone'][l]) == int(d['vm_required_zone'][k])]
        if self.variant == 3 and int(d['needs_gpu'][k]) == 1:
            return [l for l in servers if int(d['is_gpu_server'][l]) == 1]
        return list(servers)

    def loads(self, assign):
        cpu, mem = [0] * self.S, [0] * self.S
        for k, l in enumerate(assign):
            cpu[l] += self.cpu_req[k]
            mem[l] += self.mem_req[k]
        return cpu, mem

    def cost(self, cpu):
        return sum(self.latency[l][cpu[l]] for l in range(self.S))

    def overload_of(self, l, cpu_load, mem_load):
        return max(0, cpu_load - self.cpu_cap[l]) + max(0, mem_load - self.mem_cap[l])

    def overload(self, cpu, mem):
        return sum(self.overload_of(l, cpu[l], mem[l]) for l in range(self.S))


def transfer_delta(P, cpu, k, old, new):
    """Change of the cost when VM k goes from server `old` to server `new` (only two servers change)."""
    c = P.cpu_req[k]
    return (P.latency[old][cpu[old] - c] + P.latency[new][cpu[new] + c]
            - P.latency[old][cpu[old]] - P.latency[new][cpu[new]])


def propose(P, assign, r):
    """Random transfer: a movable VM and a different allowed server, both uniform."""
    k = P.movable[int(r() * len(P.movable))]
    servers = P.allowed[k]
    new = servers[int(r() * (len(servers) - 1))]
    if new == assign[k]:
        new = servers[-1]
    return k, new


def apply_transfer(P, assign, cpu, mem, k, new):
    old = assign[k]
    cpu[old] -= P.cpu_req[k]
    mem[old] -= P.mem_req[k]
    cpu[new] += P.cpu_req[k]
    mem[new] += P.mem_req[k]
    assign[k] = new


def build_start(P, rng, tries=200):
    """Randomised greedy start; it reads only the data. The most restricted and largest VMs go first
    onto a random server that still fits. Returns the first feasible result, else the least overloaded."""
    best, best_over = None, None
    for _ in range(tries):
        order = list(range(P.M))
        rng.shuffle(order)
        order.sort(key=lambda k: (len(P.allowed[k]), -P.cpu_req[k], -P.mem_req[k]))
        assign, cpu, mem = [-1] * P.M, [0] * P.S, [0] * P.S
        for k in order:
            c, m = P.cpu_req[k], P.mem_req[k]
            fits = [l for l in P.allowed[k] if cpu[l] + c <= P.cpu_cap[l] and mem[l] + m <= P.mem_cap[l]]
            l = rng.choice(fits) if fits else min(P.allowed[k], key=lambda s: P.overload_of(s, cpu[s] + c, mem[s] + m))
            assign[k] = l
            cpu[l] += c
            mem[l] += m
        over = P.overload(cpu, mem)
        if best is None or over < best_over:
            best, best_over = assign, over
        if over == 0:
            break
    return best


def _clean_start(P, start, rng):
    """A supplied start; a VM on a server that it may not use is moved to an allowed server."""
    assign = [int(x) for x in start]
    if len(assign) != P.M:
        raise ValueError('start must have one entry per VM')
    for k in range(P.M):
        if assign[k] not in P.allowed[k]:
            assign[k] = rng.choice(P.allowed[k])
    return assign


def _random_walk(P, assign, rng, proposals):
    """Walk that accepts every feasible proposal (a walk from an overloaded state may not raise the
    overload). Returns the positive cost changes seen. `assign` is not changed."""
    assign = list(assign)
    cpu, mem = P.loads(assign)
    r, positive = rng.random, []
    if not P.movable:
        return positive
    for _ in range(proposals):
        k, new = propose(P, assign, r)
        old = assign[k]
        c, m = P.cpu_req[k], P.mem_req[k]
        before = P.overload_of(old, cpu[old], mem[old]) + P.overload_of(new, cpu[new], mem[new])
        after = P.overload_of(old, cpu[old] - c, mem[old] - m) + P.overload_of(new, cpu[new] + c, mem[new] + m)
        if after > before:
            continue
        delta = transfer_delta(P, cpu, k, old, new)
        if delta > EPS:
            positive.append(delta)
        apply_transfer(P, assign, cpu, mem, k, new)
    return positive


def temperatures_from_walk(positive):
    """Rule of thumb of Section 4: T0 = 4.5 * median, Tend = 0.22 * 5th percentile of the positive changes."""
    if not positive:
        return {'Delta': 1.0, 'Delta_min': 0.01, 'T0': 4.5, 'Tend': 0.0022, 'positive': 0}
    delta, delta_min = float(np.median(positive)), float(np.percentile(positive, 5))
    return {'Delta': delta, 'Delta_min': delta_min, 'T0': 4.5 * delta, 'Tend': 0.22 * delta_min, 'positive': len(positive)}


def rule_temperatures(d, variant, seed, proposals=WALK_LENGTH):
    """The rule-based T0 and Tend measured outside a run (used for the tuning grid)."""
    P = Problem(d, variant)
    rng = random.Random(seed)
    return temperatures_from_walk(_random_walk(P, build_start(P, rng), rng, proposals))


def solve(d, variant, seed, budget, params=None, start=None):
    """See src/meta/interface.py. Extra entry 'diag': the counters needed by the analysis."""
    P = Problem(d, variant)
    rng = random.Random(seed)
    r = rng.random
    assign = _clean_start(P, start, rng) if start is not None else build_start(P, rng)
    cpu, mem = P.loads(assign)
    cost, over = P.cost(cpu), P.overload(cpu, mem)

    budget = int(budget)
    evals = 1 if budget >= 1 else 0
    best_assign, best_cost = (list(assign), cost) if over == 0 else (None, float('inf'))
    trace = [(evals, best_cost)]

    if params is not None and 'T0' in params and 'Tend' in params:
        T0, Tend, walk = float(params['T0']), float(params['Tend']), 0
        calibration = None
    else:
        walk = min(WALK_LENGTH, max(0, budget - 1) // 10)
        calibration = temperatures_from_walk(_random_walk(P, assign, rng, walk))
        T0, Tend = calibration['T0'], calibration['Tend']
        for e in range(evals + 1, evals + walk + 1):      # the walk is charged; the best does not change
            if e % 1000 == 0:
                trace.append((e, best_cost))
        evals += walk
    if T0 <= 0 or Tend <= 0:
        raise ValueError('T0 and Tend must be positive')

    K = max(0, budget - evals)
    alpha = (Tend / T0) ** (1.0 / K) if K > 0 else 1.0
    head = K // 20                                         # the first and the last 5% of the proposals
    tail_from = K - head
    bins = 100
    bin_worse, bin_accepted, bin_T = [0] * bins, [0] * bins, [0.0] * bins
    bin_n = [0] * bins
    worse_first = accepted_first = worse_last = accepted_last = 0
    accepted = rejected_capacity = 0

    T = T0
    for i in range(K):
        evals += 1
        b = i * bins // K
        bin_T[b] += T
        bin_n[b] += 1
        if P.movable:
            k, new = propose(P, assign, r)
            old = assign[k]
            c, m = P.cpu_req[k], P.mem_req[k]
            if over > 0:                                    # only a supplied or impossible start gets here
                before = P.overload_of(old, cpu[old], mem[old]) + P.overload_of(new, cpu[new], mem[new])
                after = P.overload_of(old, cpu[old] - c, mem[old] - m) + P.overload_of(new, cpu[new] + c, mem[new] + m)
                if after <= before:
                    cost += transfer_delta(P, cpu, k, old, new)
                    apply_transfer(P, assign, cpu, mem, k, new)
                    over += after - before
                    accepted += 1
                    if over == 0 and cost < best_cost:
                        best_cost, best_assign = cost, list(assign)
            elif cpu[new] + c > P.cpu_cap[new] or mem[new] + m > P.mem_cap[new]:
                rejected_capacity += 1
            else:
                delta = transfer_delta(P, cpu, k, old, new)
                worse = delta > EPS
                take = (not worse) or r() < math.exp(-delta / T)
                if worse:
                    bin_worse[b] += 1
                    bin_accepted[b] += take
                    if i < head:
                        worse_first += 1
                        accepted_first += take
                    if i >= tail_from:
                        worse_last += 1
                        accepted_last += take
                if take:
                    apply_transfer(P, assign, cpu, mem, k, new)
                    cost += delta
                    accepted += 1
                    if cost < best_cost:
                        best_cost, best_assign = cost, list(assign)
        T *= alpha
        if evals % 1000 == 0:
            trace.append((evals, best_cost))
    if trace[-1][0] != evals:
        trace.append((evals, best_cost))

    final = best_assign if best_assign is not None else assign
    checked_cost, problems = check(d, variant, list(final))
    diag = {
        'T0': T0, 'Tend': Tend, 'alpha': alpha, 'proposals': K, 'walk': walk, 'calibration': calibration,
        'worse_first': worse_first, 'accepted_first': accepted_first,
        'worse_last': worse_last, 'accepted_last': accepted_last,
        'accepted': accepted, 'rejected_capacity': rejected_capacity,
        'overload': P.overload(*P.loads(final)),
        'bin_T': [t / n if n else float('nan') for t, n in zip(bin_T, bin_n)],
        'bin_worse': bin_worse, 'bin_accepted': bin_accepted,
        'current_assign': list(assign), 'current_cost': cost,
    }
    return {'solution': [int(x) for x in final], 'cost': float(checked_cost), 'evals': evals,
            'feasible': len(problems) == 0, 'trace': trace, 'diag': diag}
