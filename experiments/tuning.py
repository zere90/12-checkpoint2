"""SA-T tuning experiment. Execute from project root: python experiments/tuning.py"""
import csv, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import checkpoint4_generator_PUBLIC as gen
from src.meta.sa import solve
from src.meta.nm import nelder_mead

OUT='results'; os.makedirs(OUT, exist_ok=True)
d=gen.cp34_instance(12,'B',2); VARIANT=2
TUNE_SEEDS=range(100,105); FINAL_SEEDS=range(10)
INNER_BUDGET=20000; FINAL_BUDGET=100000
FIELDS=['team','role','configuration','seed','budget','cost','evals','feasible','seconds','params']
rows=[]
def run(seed,budget,params,label):
    start=time.perf_counter(); r=solve(d,VARIANT,seed,budget,params=params)
    cost,problems=gen.evaluate_B(d,r['solution']); elapsed=time.perf_counter()-start
    assert r['evals']==budget and (not problems)==r['feasible']
    rows.append(dict(team='T12',role='SA-T',configuration=label,seed=seed,budget=budget,cost=f'{cost:.9f}',evals=r['evals'],feasible=not problems,seconds=f'{elapsed:.4f}',params=json.dumps(params or {})))
    return float(cost) if not problems else 10.0*float(cost)

# Rule-based temperatures are measured from a reproducible pilot, never final seeds.
pilot=solve(d,VARIANT,100,5000)
T0=pilot['info']['T0']; Tend=pilot['info']['Tend']
print('rule temperatures:',T0,Tend,flush=True)
rule={'T0':T0,'Tend':Tend}
for seed in TUNE_SEEDS: run(seed,INNER_BUDGET,rule,'SA rule tune')

def objective(x):
    # Optimize logarithms: both temperatures stay positive and Tend < T0.
    t0=math.exp(max(-15,min(10,x[0])))
    tend=t0*math.exp(-math.exp(max(-7,min(4,x[1]))))
    params={'T0':t0,'Tend':tend}
    vals=[run(seed,INNER_BUDGET,params,'SA NM objective') for seed in TUNE_SEEDS]
    return float(np.mean(vals))

x0=(math.log(T0),math.log(max(1e-6,math.log(T0/Tend))))
nm=nelder_mead(objective,x0,step=(0.2,0.2),max_evals=25)
xt=nm['x']; tuned={'T0':math.exp(xt[0]),'Tend':math.exp(xt[0])*math.exp(-math.exp(xt[1]))}
print('NM:',nm['fun'],tuned,flush=True)
for seed in TUNE_SEEDS: run(seed,INNER_BUDGET,tuned,'SA tuned comparison')
# Do not choose settings on the final seeds.
rule_mean=np.mean([float(r['cost']) for r in rows if r['configuration']=='SA rule tune'])
tuned_mean=np.mean([float(r['cost']) for r in rows if r['configuration']=='SA tuned comparison'])
selected='tuned' if tuned_mean < rule_mean - 1e-10 else 'rule'
params= tuned if selected=='tuned' else rule
for seed in FINAL_SEEDS: run(seed,FINAL_BUDGET,params,'SA selected final')

with open(f'{OUT}/Aitzhanov_tuning.csv','w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
with open(f'{OUT}/Aitzhanov_tuning_summary.json','w') as f:
    json.dump(dict(rule=rule,tuned=tuned,selected=selected,rule_tune_mean=rule_mean,tuned_tune_mean=tuned_mean,nm=nm,inner_budget=INNER_BUDGET),f,indent=2)
plt.figure(figsize=(6.5,3.3));plt.plot(range(1,len(nm['history'])+1),nm['history'],marker='.',markersize=3)
plt.xlabel('Nelder–Mead objective evaluations');plt.ylabel('Best tuning mean cost');plt.grid(alpha=.3);plt.tight_layout();plt.savefig(f'{OUT}/Aitzhanov_NM_trace.png',dpi=180);plt.close()
print('DONE',selected,flush=True)
