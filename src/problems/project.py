"""Part C project block -- Storyline B: balancing CPU load across servers
(Team T12, cloud_variant(seed=12, variant=2)).

Checkpoint 1 assigned VMs to servers with integer, zone-restricted
variables. Here we relax it: forget individual VMs and integrality, and
split the total CPU demand D = sum(cpu_req) among the N_srv servers with
continuous shares w_j >= 0, sum_j w_j = 1. Server j has capacity
C_j = cpu_cap[j] and utilization u_j = D*w_j / C_j.

Latency model for this checkpoint: L(u) = u^2.
Objective:            J(w) = (1/N_srv) * sum_j u_j^2        (mean squared utilization)

Unconstrained reparametrization: w = softmax(z) with the last logit
pinned at 0,
    w_j = exp(z_j) / (1 + sum_{k<N_srv-1} exp(z_k))   for j = 0..N_srv-2
    w_{N_srv-1} = 1 / (1 + sum_k exp(z_k))
so z in R^{N_srv-1} = R^9 and we minimize F(z) = J(w(z)) without
constraints.
"""
import numpy as np

# ---- Data: cloud_variant(seed=12, variant=2) -------------------------
CPU_CAP = np.array([64, 63, 34, 38, 38, 37, 51, 43, 47, 39], dtype=float)
CPU_REQ = np.array([4, 5, 4, 1, 6, 3, 3, 4, 3, 4, 7, 4, 6, 8, 2, 3, 5, 2,
                     6, 6, 6, 8, 4, 8, 8, 8, 3, 1, 7, 8, 3, 6, 2, 7, 7],
                    dtype=float)

N_SRV = CPU_CAP.size            # 10
D = CPU_REQ.sum()               # 172.0
NFREE = N_SRV - 1               # 9  (dimension of z)


def softmax_pinned(z):
    """w in R^{N_SRV}, last coordinate's logit pinned to 0."""
    ez = np.exp(z)
    denom = 1.0 + ez.sum()
    w = np.empty(N_SRV)
    w[:NFREE] = ez / denom
    w[NFREE] = 1.0 / denom
    return w


def f(z):
    w = softmax_pinned(z)
    u = D * w / CPU_CAP
    return (u ** 2).sum() / N_SRV


def grad(z):
    """Analytic gradient via the chain rule:
        dJ/dw_j = (2 D^2 / N_srv) * w_j / C_j^2
        dw_j/dz_k = w_j (delta_jk - w_k)          (j=0..N_SRV-1, k=0..NFREE-1)
    =>  dF/dz_k = w_k * ( s_k - sum_j w_j s_j ),   s_j := dJ/dw_j
    """
    w = softmax_pinned(z)
    s = (2.0 * D ** 2 / N_SRV) * w / CPU_CAP ** 2
    mean_s = (w * s).sum()
    g = w[:NFREE] * (s[:NFREE] - mean_s)
    return g


def hess(z, eps=1e-5):
    """Central-difference Hessian of the verified analytic gradient,
    symmetrized."""
    n = z.size
    H = np.zeros((n, n))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = eps
        H[:, i] = (grad(z + ei) - grad(z - ei)) / (2 * eps)
    return 0.5 * (H + H.T)


START = np.zeros(NFREE)

# Closed-form minimizer over all splits (to be derived/verified, see H5
# and S2/S5): w*_j = C_j^2 / sum_k C_k^2
W_STAR = CPU_CAP ** 2 / (CPU_CAP ** 2).sum()
Z_STAR = np.log(W_STAR[:NFREE] / W_STAR[NFREE])
J_STAR = D ** 2 / (N_SRV * (CPU_CAP ** 2).sum())
