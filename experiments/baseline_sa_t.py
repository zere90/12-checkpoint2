"""Uniform independent zone-compatible assignments; infeasible samples count toward budget."""
import csv, json, os, sys, time, random
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import checkpoint4_generator_PUBLIC as gen
from src.meta.sa import allowed_servers, latency
p='results/Aitzhanov_tuning.csv'; d=gen.cp34_instance(12,'B',2)
allowed=allowed_servers(d,2); cap=np.asarray(d['cpu_cap']); memcap=np.asarray(d['mem_cap']); cpu=np.asarray(d['cpu_req']); mem=np.asarray(d['mem_req']); S=int(d['S']); M=int(d['M'])
rows=[]
for seed in range(10):
    rng=np.random.default_rng(seed); start=time.perf_counter(); best=float('inf'); best_assign=None
    # Every column is a separate candidate, evaluated once, feasible or not.
    batch=10000
    for offset in range(0,100000,batch):
        n=min(batch,100000-offset)
        assignments=np.vstack([rng.choice(allowed[k],size=n) for k in range(M)])
        cload=np.zeros((S,n),dtype=int); mload=np.zeros((S,n),dtype=int)
        for k in range(M):
            cload[assignments[k],np.arange(n)]+=cpu[k]
            mload[assignments[k],np.arange(n)]+=mem[k]
        feasible=np.all((cload<=cap[:,None]) & (mload<=memcap[:,None]),axis=0)
        ids=np.flatnonzero(feasible)
        if len(ids):
            u=cload[:,ids]/cap[:,None]
            lines=[(1/(1-x)**2,x*x/(1-x)**2) for x in (0,.4,.7,.9)]
            costs=np.maximum.reduce([a*u-b for a,b in lines]).sum(axis=0)
            j=int(np.argmin(costs))
            if float(costs[j])<best:
                best=float(costs[j]); best_assign=assignments[:,ids[j]].tolist()
    if best_assign is not None:
        check,problems=gen.evaluate_B(d,best_assign)
        assert not problems and abs(check-best)<1e-8
    rows.append(dict(team='T12',role='SA-T',configuration='random',seed=seed,budget=100000,cost=f'{best:.9f}',evals=100000,feasible=best_assign is not None,seconds=f'{time.perf_counter()-start:.4f}',params='{}'))
    print('random',seed,best,flush=True)
with open(p,'a',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['team','role','configuration','seed','budget','cost','evals','feasible','seconds','params']);w.writerows(rows)
