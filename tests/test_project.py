"""Tests for problems/project.py -- M5 (project lead) role.

1. Check values published in the task file at z0=0: F(z0)=0.16307,
   ||grad F(z0)||=0.0370463.
2. check_grad / check_hess against central differences.
3. The closed-form minimizer: softmax_pinned(z*) == w*, grad(z*) ~= 0.
4. H5 hand trace: the toy instance N_srv=3, C=(3,5,6), D=6 at z=(0,0),
   reproduced with the real (full-size) helper functions specialised to
   the toy data, to 1e-3.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from optim._common import check_grad, check_hess
from problems import project as P


def test_check_values_at_z0():
    assert abs(P.f(P.START) - 0.16307) < 1e-4
    assert abs(np.linalg.norm(P.grad(P.START)) - 0.0370463) < 1e-4


def test_check_grad():
    err = check_grad(P.f, P.grad, P.START, eps=1e-6)
    assert err < 1e-6


def test_check_hess():
    err = check_hess(P.grad, P.hess, P.START, eps=1e-5)
    assert err < 1e-4


def test_closed_form_minimizer():
    w = P.softmax_pinned(P.Z_STAR)
    assert np.allclose(w, P.W_STAR, atol=1e-8)
    g_star = P.grad(P.Z_STAR)
    assert np.max(np.abs(g_star)) < 1e-6
    J_star_direct = P.f(P.Z_STAR)
    assert abs(J_star_direct - P.J_STAR) < 1e-9


# ---- H5: toy instance, N_srv = 3, C = (3,5,6), D = 6 ------------------

def toy_softmax(z, C):
    n = C.size
    nfree = n - 1
    ez = np.exp(z)
    denom = 1.0 + ez.sum()
    w = np.empty(n)
    w[:nfree] = ez / denom
    w[nfree] = 1.0 / denom
    return w


def toy_f(z, C, D, N):
    w = toy_softmax(z, C)
    u = D * w / C
    return (u ** 2).sum() / N


def toy_grad(z, C, D, N):
    w = toy_softmax(z, C)
    nfree = C.size - 1
    s = (2.0 * D ** 2 / N) * w / C ** 2
    mean_s = (w * s).sum()
    return w[:nfree] * (s[:nfree] - mean_s)


def test_h5_toy_at_z0():
    C = np.array([3.0, 5.0, 6.0])
    D, N = 6.0, 3
    z0 = np.array([0.0, 0.0])
    w = toy_softmax(z0, C)
    assert np.allclose(w, np.array([1 / 3, 1 / 3, 1 / 3]), atol=1e-3)
    u = D * w / C
    assert np.allclose(u, np.array([0.6667, 0.4, 0.3333]), atol=1e-3)
    J = toy_f(z0, C, D, N)
    assert abs(J - 0.238519) < 1e-3
    g = toy_grad(z0, C, D, N)
    assert np.allclose(g, np.array([0.137284, -0.052346]), atol=1e-3)


def test_h5_toy_closed_form():
    C = np.array([3.0, 5.0, 6.0])
    D, N = 6.0, 3
    wstar = C ** 2 / (C ** 2).sum()
    assert np.allclose(wstar, np.array([0.128571, 0.357143, 0.514286]),
                        atol=1e-3)
    Jstar = D ** 2 / (N * (C ** 2).sum())
    assert abs(Jstar - 0.171429) < 1e-3
    zstar = np.log(wstar[:2] / wstar[2])
    gstar = toy_grad(zstar, C, D, N)
    assert np.max(np.abs(gstar)) < 1e-6

    wprop = C / C.sum()
    uprop = D * wprop / C
    Jprop = (uprop ** 2).sum() / N
    assert abs(Jprop - 0.183673) < 1e-3
    assert Jstar < Jprop  # equal utilization is NOT the minimizer


if __name__ == "__main__":
    test_check_values_at_z0()
    test_check_grad()
    test_check_hess()
    test_closed_form_minimizer()
    test_h5_toy_at_z0()
    test_h5_toy_closed_form()
    print("test_project.py: all tests passed")
