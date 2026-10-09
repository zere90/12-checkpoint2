"""Quadratic test problems Q1, Q2 (Team T12: c = 277, theta = 20 deg).

Q1: f(x) = x1^2 + c*x2^2            H = diag(2, 2c),  kappa = c
Q2: g(x) = f(R^T x),  R = rotation(theta_deg)   (Q1 rotated: same
    eigenvalues, same minimum value, same distance to the minimizer)
"""
import numpy as np

C = 277
THETA_DEG = 20

X0_QUAD = np.array([1.3, 0.7])


def rotation(theta_deg):
    t = np.deg2rad(theta_deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


R = rotation(THETA_DEG)


def q1_f(x):
    x1, x2 = x
    return x1 ** 2 + C * x2 ** 2


def q1_grad(x):
    x1, x2 = x
    return np.array([2.0 * x1, 2.0 * C * x2])


def q1_hess(x):
    return np.array([[2.0, 0.0], [0.0, 2.0 * C]])


def q2_f(x):
    y = R.T @ x
    return q1_f(y)


def q2_grad(x):
    # g(x) = f(R^T x)  =>  grad g(x) = R * grad f(R^T x)   (R orthogonal)
    y = R.T @ x
    return R @ q1_grad(y)


def q2_hess(x):
    # Hessian of f(R^T x) wrt x is R H(R^T x) R^T
    y = R.T @ x
    return R @ q1_hess(y) @ R.T


Q1_START = X0_QUAD.copy()
Q2_START = R @ X0_QUAD

MINIMIZER = np.array([0.0, 0.0])
F_MIN = 0.0
LAMBDA_MIN = 2.0
LAMBDA_MAX = 2.0 * C
KAPPA = C
