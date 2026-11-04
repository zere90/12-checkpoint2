"""Simulated annealing, transfer + swap, for storyline B (cloud). Role SA-I, team T12.

State: assign[k] = server of VM k. Cost: sum over servers of the piecewise-linear latency of the
CPU utilisation (same formula as the checker, re-implemented here with incremental updates).

Design (see README):
  * constraint handling: feasible-only. A candidate that would overload a CPU or memory capacity is
    still one evaluation (it is evaluated and rejected). Candidates that break a zone / GPU rule are never
    generated (the target servers are chosen among the allowed ones), so they are not candidates.
  * moves: transfer (prob. 1-p_swap) and swap (prob. p_swap, default 0.3).
  * temperature: T_k = T0 * alpha^k, alpha = (Tend/T0)^(1/K), K = number of annealing proposals.
  * T0, Tend: params={"T0":..., "Tend":...}; if absent, the rule of the course (random walk of 3000
    proposals accepting everything feasible; median and 5th percentile of the positive deltas).
    The random walk is charged to the budget (every proposal = 1 evaluation); K = budget - 1 - walk.
  * evaluations: the start = 1, every proposal (random walk and annealing) = 1; the total equals budget.
"""
import math
import random

import numpy as np

TANGENT_POINTS = (0.0, 0.4, 0.7, 0.9)           # fixed for everyone (same as the generator)
_LINES = [(1.0 / (1.0 - um) ** 2, um ** 2 / (1.0 - um) ** 2) for um in TANGENT_POINTS]


def latency(u):
    if u > 1.0:
        u = 1.0
    return max(a * u - b for a, b in _LINES)


class State:
    """Assignment with incremental cost. Moves are evaluated before they are applied."""

    def __init__(self, d, variant, assign):
        self.S, self.M = int(d['S']), int(d['M'])
        self.cpu_cap = [int(x) for x in d['cpu_cap']]
        self.mem_cap = [int(x) for x in d['mem_cap']]
        self.cpu_req = [int(x) for x in d['cpu_req']]
        self.mem_req = [int(x) for x in d['mem_req']]
        self.allowed = allowed_servers(d, variant)
        self.allowed_set = [set(a) for a in self.allowed]
        self.assign = [int(a) for a in assign]
        self.cpu = [0] * self.S
        self.mem = [0] * self.S
        for k, s in enumerate(self.assign):
            self.cpu[s] += self.cpu_req[k]
            self.mem[s] += self.mem_req[k]
        self.lat = [latency(self.cpu[s] / self.cpu_cap[s]) for s in range(self.S)]
        self.cost = sum(self.lat)

    def fits(self, s, cpu, mem):
        return cpu <= self.cpu_cap[s] and mem <= self.mem_cap[s]

    def eval_transfer(self, k, t):
        """Returns (delta, feasible, new data) for moving VM k to server t."""
        s = self.assign[k]
        c, m = self.cpu_req[k], self.mem_req[k]
        cs, ct = self.cpu[s] - c, self.cpu[t] + c
        ms, mt = self.mem[s] - m, self.mem[t] + m
        ls, lt = latency(cs / self.cpu_cap[s]), latency(ct / self.cpu_cap[t])
        delta = ls + lt - self.lat[s] - self.lat[t]
        feas = self.fits(t, ct, mt)          # s only loses load, it stays feasible
        return delta, feas, (s, t, cs, ct, ms, mt, ls, lt)

    def eval_swap(self, i, j):
        """Returns (delta, feasible, new data) for exchanging the servers of VMs i and j."""
        s, t = self.assign[i], self.assign[j]
        dc = self.cpu_req[j] - self.cpu_req[i]       # change of load of s
        dm = self.mem_req[j] - self.mem_req[i]
        cs, ct = self.cpu[s] + dc, self.cpu[t] - dc
        ms, mt = self.mem[s] + dm, self.mem[t] - dm
        ls, lt = latency(cs / self.cpu_cap[s]), latency(ct / self.cpu_cap[t])
        delta = ls + lt - self.lat[s] - self.lat[t]
        feas = self.fits(s, cs, ms) and self.fits(t, ct, mt)
        return delta, feas, (s, t, cs, ct, ms, mt, ls, lt)

    def apply(self, moves, delta, data):
        """moves: list of (vm, new server)."""
        s, t, cs, ct, ms, mt, ls, lt = data
        self.cpu[s], self.cpu[t] = cs, ct
        self.mem[s], self.mem[t] = ms, mt
        self.lat[s], self.lat[t] = ls, lt
        for k, srv in moves:
            self.assign[k] = srv
        self.cost += delta


def allowed_servers(d, variant):
    S, M = int(d['S']), int(d['M'])
    allowed = []
    for k in range(M):
        ok = []
        for s in range(S):
            if variant == 2 and int(d['vm_required_zone'][k]) != -1 and int(d['zone'][s]) != int(d['vm_required_zone'][k]):
                continue
            if variant == 3 and int(d['needs_gpu'][k]) == 1 and int(d['is_gpu_server'][s]) == 0:
                continue
            ok.append(s)
        allowed.append(ok)
    return allowed


def random_start(d, variant, rng):
    """Random feasible assignment: VMs in random order, each to a random allowed server where it fits."""
    S, M = int(d['S']), int(d['M'])
    allowed = allowed_servers(d, variant)
    cpu_cap = [int(x) for x in d['cpu_cap']]
    mem_cap = [int(x) for x in d['mem_cap']]
    for _ in range(1000):
        cpu, mem = [0] * S, [0] * S
        assign = [-1] * M
        order = list(range(M))
        order.sort(key=lambda k: (len(allowed[k]), rng.random()))      # most restricted VMs first
        good = True
        for k in order:
            cands = [s for s in allowed[k]
                     if cpu[s] + int(d['cpu_req'][k]) <= cpu_cap[s] and mem[s] + int(d['mem_req'][k]) <= mem_cap[s]]
            if not cands:
                good = False
                break
            s = rng.choice(cands)
            assign[k] = s
            cpu[s] += int(d['cpu_req'][k])
            mem[s] += int(d['mem_req'][k])
        if good:
            return assign
    raise RuntimeError('no feasible random start found')


def propose(st, rng, p_swap):
    """Draw one candidate move: returns ('T', k, t) or ('S', i, j), or None if no move exists."""
    M = st.M
    if rng.random() < p_swap:
        for _ in range(50):
            i, j = rng.randrange(M), rng.randrange(M)
            si, sj = st.assign[i], st.assign[j]
            if si != sj and sj in st.allowed_set[i] and si in st.allowed_set[j]:
                return ('S', i, j)
    for _ in range(200):
        k = rng.randrange(M)
        opts = st.allowed[k]
        if len(opts) < 2:
            continue
        t = opts[rng.randrange(len(opts))]
        if t != st.assign[k]:
            return ('T', k, t)
    return None


def evaluate_move(st, mv):
    if mv[0] == 'T':
        delta, feas, data = st.eval_transfer(mv[1], mv[2])
        moves = [(mv[1], mv[2])]
    else:
        delta, feas, data = st.eval_swap(mv[1], mv[2])
        moves = [(mv[1], st.assign[mv[2]]), (mv[2], st.assign[mv[1]])]
    return delta, feas, data, moves


def solve(d, variant, seed, budget, params=None, start=None):
    if d.get('story') != 'B':
        raise ValueError('this module solves storyline B')
    params = dict(params or {})
    rng = random.Random(seed)
    p_swap = float(params.get('p_swap', 0.3))
    budget = int(budget)

    assign0 = [int(a) for a in start] if start is not None else random_start(d, variant, rng)
    st = State(d, variant, assign0)
    evals = 1                                                    # the start solution
    trace = []
    feasible_now = all(st.fits(s, st.cpu[s], st.mem[s]) for s in range(st.S))
    best_cost = st.cost if feasible_now else float('inf')
    best_assign = list(st.assign) if feasible_now else None
    next_trace = 1000

    def record():
        nonlocal next_trace
        while evals >= next_trace:
            trace.append((next_trace, best_cost))
            next_trace += 1000

    # ---------- temperatures
    if 'T0' in params and 'Tend' in params:
        T0, Tend = float(params['T0']), float(params['Tend'])
        n_walk = 0
    else:
        n_walk = int(params.get('walk', 3000))
        if budget - 1 - n_walk < 1:
            n_walk = max(0, (budget - 1) // 3)
        positives = []
        saved = list(st.assign)
        for _ in range(n_walk):
            mv = propose(st, rng, p_swap)
            evals += 1
            if mv is not None:
                delta, feas, data, moves = evaluate_move(st, mv)
                if delta > 0:
                    positives.append(delta)
                if feas:                                         # accept everything that is feasible
                    st.apply(moves, delta, data)
                    if st.cost < best_cost - 1e-12:
                        best_cost, best_assign = st.cost, list(st.assign)
            record()
        if positives:
            dbar = float(np.median(positives))
            dmin = float(np.percentile(positives, 5))
        else:
            dbar, dmin = 1.0, 0.01
        T0, Tend = 4.5 * dbar, 0.22 * dmin
        st = State(d, variant, saved)                            # SA starts from the original start
    K = max(1, budget - evals)                                   # annealing proposals
    alpha = (Tend / T0) ** (1.0 / K)

    # ---------- annealing
    T = T0
    log_idx, log_acc, log_T, log_feas = [], [], [], []           # all proposals with delta > 0
    n_prop = 0
    n_acc = 0
    while evals < budget:
        mv = propose(st, rng, p_swap)
        evals += 1
        n_prop += 1
        if mv is not None:
            delta, feas, data, moves = evaluate_move(st, mv)
            accepted = False
            if feas:
                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    accepted = True
                    st.apply(moves, delta, data)
                    n_acc += 1
                    if st.cost < best_cost - 1e-12:
                        best_cost, best_assign = st.cost, list(st.assign)
            if delta > 0:
                log_idx.append(n_prop - 1)
                log_acc.append(accepted)
                log_T.append(T)
                log_feas.append(feas)
        T *= alpha
        record()
    while next_trace <= budget:
        trace.append((next_trace, best_cost))
        next_trace += 1000

    import checkpoint4_generator_PUBLIC as gen                    # checker only for the final report
    if best_assign is None:
        best_assign = list(st.assign)
    cost, problems = gen.evaluate_B(d, best_assign)
    return {'solution': best_assign, 'cost': cost, 'evals': evals, 'feasible': len(problems) == 0,
            'trace': trace,
            'info': {'T0': T0, 'Tend': Tend, 'K': K, 'walk': n_walk, 'p_swap': p_swap,
                     'n_proposals': n_prop, 'n_accepted': n_acc,
                     'worse_idx': log_idx, 'worse_acc': log_acc, 'worse_T': log_T, 'worse_feas': log_feas}}
