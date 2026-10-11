"""Vorgerechnete Studie: fünf Verfahren, drei Zeitgrenzen, zwei Lauflängen, zwei Budgets, je 20 feste Läufe; dazu ParEGO gegen Zufall (rund 5 bis 10 Minuten).

    python generate_study.py            # schreibt precomputed_study.json
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import cbo_constants as C
import cbo_evaluation as ev


def job(args):
    tmax, trucks, budget, method, seed = args
    r = ev.run_one(method, ev.Settings(n=budget, tmax=tmax, trucks=trucks, seed=seed), budget)
    return tmax, trucks, budget, method, {"seed": seed, "truly": r.truly_feasible, "claimed": r.claimed_feasible, "excess": r.excess, "cost": r.cost, "rec": r.rec}


def pareto_job(args):
    trucks, budget, method, seed = args
    hv, front = ev.pareto_run(method, ev.Settings(n=budget, tmax=C.DEFAULT_TMAX, trucks=trucks, seed=seed), budget)
    return trucks, budget, method, {"seed": seed, "hv": hv, "front": front}


def main(workers=14):
    t0 = time.time()
    jobs = [(tm, tr, b, m, s) for tm in C.TMAX_CHOICES for tr in C.TRUCK_CHOICES for b in C.STUDY_BUDGETS for m in C.METHODS for s in C.SWEEP_SEEDS]
    pjobs = [(tr, b, m, s) for tr in C.TRUCK_CHOICES for b in C.STUDY_BUDGETS for m in ("random", "parego") for s in C.SWEEP_SEEDS]
    runs, pareto = {}, {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for tmax, trucks, budget, method, rec in ex.map(job, jobs, chunksize=4):
            runs.setdefault(ev.study_key(tmax, trucks, budget), {}).setdefault(method, []).append(rec)
        for trucks, budget, method, rec in ex.map(pareto_job, pjobs, chunksize=2):
            pareto.setdefault(f"{trucks}_{budget}", {}).setdefault(method, []).append(rec)
    opt = {str(float(t)): ev.S.constrained_optimum(t) for t in C.TMAX_CHOICES}
    out = {"runs": runs, "pareto": pareto, "optimum": {k: list(v) for k, v in opt.items()}, "pareto_reference_hv": ev.pareto_reference(),
           "meta": {"seeds": len(C.SWEEP_SEEDS), "jobs": len(jobs) + len(pjobs), "seconds": round(time.time() - t0)}}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "precomputed_study.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print("fertig", len(jobs) + len(pjobs), "Läufe in", round(time.time() - t0), "s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 14)
