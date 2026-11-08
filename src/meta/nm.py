"""Nelder–Mead simplex minimizer, from scratch (SA-T)."""
import math

def nelder_mead(f, x0, step=(0.2, 0.2), max_evals=30, tol=1e-8):
    x0 = tuple(map(float, x0))
    n = len(x0)
    if n < 1 or len(step) != n or max_evals < n + 1:
        raise ValueError('invalid simplex dimensions or evaluation budget')
    simplex = [x0]
    for i in range(n):
        p = list(x0); p[i] += float(step[i]); simplex.append(tuple(p))
    history = []
    def evaluate(x):
        value = float(f(tuple(x)))
        if not math.isfinite(value): value = float('inf')
        history.append(min(value, history[-1] if history else float('inf')))
        return value
    values = [evaluate(x) for x in simplex]
    while len(history) < max_evals:
        order = sorted(range(n+1), key=lambda i: values[i]); simplex = [simplex[i] for i in order]; values = [values[i] for i in order]
        if max(max(abs(simplex[i][j]-simplex[0][j]) for j in range(n)) for i in range(1,n+1)) < tol: break
        centroid = tuple(sum(simplex[i][j] for i in range(n))/n for j in range(n))
        worst = simplex[-1]
        xr = tuple(2*centroid[j]-worst[j] for j in range(n))
        fr = evaluate(xr)
        if values[0] <= fr < values[-2]:
            simplex[-1], values[-1] = xr, fr
        elif fr < values[0]:
            if len(history) >= max_evals:
                simplex[-1], values[-1] = xr, fr; break
            xe = tuple(centroid[j]+2*(xr[j]-centroid[j]) for j in range(n))
            fe = evaluate(xe)
            simplex[-1], values[-1] = (xe,fe) if fe < fr else (xr,fr)
        else:
            if len(history) >= max_evals: break
            if fr < values[-1]:
                xc = tuple(centroid[j]+0.5*(xr[j]-centroid[j]) for j in range(n))
                threshold = fr
            else:
                xc = tuple(centroid[j]+0.5*(worst[j]-centroid[j]) for j in range(n))
                threshold = values[-1]
            fc = evaluate(xc)
            if fc <= threshold: simplex[-1], values[-1] = xc, fc
            else:
                best = simplex[0]
                for i in range(1,n+1):
                    if len(history) >= max_evals: break
                    simplex[i] = tuple(best[j]+0.5*(simplex[i][j]-best[j]) for j in range(n))
                    values[i] = evaluate(simplex[i])
    idx = min(range(n+1), key=lambda i: values[i])
    return {'x': simplex[idx], 'fun': values[idx], 'nfev': len(history), 'history': history}
