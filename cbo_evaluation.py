"""Auswertung: ein Lauf je Verfahren, Kennzahlen (wirklich zulässig? Mehrkosten), vorgerechnete Studie mit gepaarten Läufen und Vorzeichentest, ParEGO-Abschnitt."""

import json
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import cbo_algorithm as A
import cbo_constants as C
import cbo_scenario as S

_HERE = os.path.dirname(os.path.abspath(__file__))
_OFFSET = {m: 1000 * (k + 1) for k, m in enumerate(C.METHODS)}


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    tmax: float = C.DEFAULT_TMAX
    trucks: int = C.DEFAULT_TRUCKS
    seed: int = C.DEFAULT_SEED


@dataclass
class Run:
    method: str
    idx: list                 # Einstellungen in Reihenfolge der Auswertung (Gitterindex)
    t_obs: list               # beobachtete (bei Nachprüfung gemittelte) Zeit je Einstellung
    rec: int                  # empfohlene Einstellung
    claimed_feasible: bool    # hält die Empfehlung die Grenze nach dem, was das Verfahren gesehen hat?
    truly_feasible: bool      # hält sie die Grenze wirklich (Wahrheit)?
    cost: float               # Gerätekosten der Empfehlung
    excess: float             # Gerätekosten minus Kosten des wirklich zulässigen Optimums


def setup(settings):
    t_true = S.truth_time()
    cost = S.cost_grid()
    pts = S.grid_points(cost.shape)
    return cost.reshape(-1), t_true.reshape(-1), S.to_unit(pts, cost.shape), pts


def run_one(method, settings, budget=None):
    """Ein Suchlauf mit Auswertungen aus der Tabelle; Zufallsströme je Verfahren getrennt und reproduzierbar."""
    budget = budget or settings.n
    cost, t_true, cand, pts = setup(settings)
    seed = settings.seed + _OFFSET[method]
    field = S.TimeField(settings.trucks, seed)
    rng = np.random.default_rng(seed + 7)

    def ev(i):
        return field.observe(int(pts[i][0]), int(pts[i][1]))

    if method == "random":
        idx, t = A.random_search(len(cand), ev, budget, rng)
        rec, ok = A.recommend_obs(idx, t, cost, settings.tmax)
    elif method == "penalty":
        idx, t = A.penalty_search(cand, cost, ev, budget, rng, settings.tmax, C.PENALTY_PER_MIN, C.N_START)
        rec, ok = A.recommend_obs(idx, t, cost, settings.tmax)
    elif method == "pof":
        idx, t = A.pof_search(cand, cost, ev, budget, rng, settings.tmax, C.N_START)
        rec, ok = A.recommend_obs(idx, t, cost, settings.tmax)
    elif method == "pof_safe":
        idx, t = A.pof_search(cand, cost, ev, budget, rng, settings.tmax, C.N_START)
        rec, ok = A.recommend_safe(cand, idx, t, cost, settings.tmax, C.SAFE_PROB)
    else:   # pof_check
        search_budget = max(C.N_START, budget - C.CHECK_RESERVE)
        idx, t = A.pof_search(cand, cost, ev, search_budget, rng, settings.tmax, C.N_START)
        idx, t = A.check_candidates(idx, t, cost, settings.tmax, lambda i: ev(i), C.CHECK_CANDIDATES, C.CHECK_REPEATS)
        rec, ok = A.recommend_obs(idx, t, cost, settings.tmax)
    _, _, opt_cost = S.constrained_optimum(settings.tmax)
    truly = bool(t_true[rec] <= settings.tmax)
    return Run(method, [int(i) for i in idx], [float(v) for v in t], int(rec), bool(ok), truly, float(cost[rec]), float(cost[rec] - opt_cost))


@dataclass
class Analysis:
    settings: Settings
    runs: dict
    optimum: tuple            # (Platz-Index, Tempo-Index, Gerätekosten) des wirklich zulässigen Optimums
    t_true: np.ndarray        # wahre Zeit [Plätze, Tempo]
    cost: np.ndarray


def analyse(settings, method):
    runs = {"random": run_one("random", settings), "penalty": run_one("penalty", settings), method: run_one(method, settings)}
    return Analysis(settings, runs, S.constrained_optimum(settings.tmax), S.truth_time(), S.cost_grid())


# --- Vorgerechnete Studie ------------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def study():
    with open(os.path.join(_HERE, "precomputed_study.json"), encoding="utf-8") as fh:
        return json.load(fh)


def study_key(tmax, trucks, budget):
    return f"t{float(tmax)}_{int(trucks)}_{int(budget)}"


def method_summary(tmax, trucks, budget):
    """Je Verfahren: Anteil wirklich zulässiger Empfehlungen, Anteil 'zulässig behauptet, aber verletzt', Median der Mehrkosten der wirklich zulässigen."""
    raw = study()["runs"][study_key(tmax, trucks, budget)]
    out = {}
    for m, recs in raw.items():
        truly = np.array([r["truly"] for r in recs], bool)
        claimed = np.array([r["claimed"] for r in recs], bool)
        excess = np.array([r["excess"] for r in recs], float)
        out[m] = {"n": len(recs), "feasible": int(truly.sum()), "violated_but_claimed": int((claimed & ~truly).sum()),
                  "median_excess_feasible": float(np.median(excess[truly])) if truly.any() else float("nan"),
                  "median_excess_all": float(np.median(excess)), "mean_excess_feasible": float(excess[truly].mean()) if truly.any() else float("nan")}
    return out


def paired_feasibility(tmax, trucks, budget, key, base="pof"):
    """Gepaarter Vergleich der Zulässigkeit: in wie vielen Läufen ist die Empfehlung von `key` wirklich zulässig und die von `base` nicht (und umgekehrt); Vorzeichentest."""
    from scipy.stats import binomtest
    raw = study()["runs"][study_key(tmax, trucks, budget)]
    a = np.array([r["truly"] for r in raw[key]], bool)
    b = np.array([r["truly"] for r in raw[base]], bool)
    better, worse = int((a & ~b).sum()), int((~a & b).sum())
    p = float(binomtest(better, better + worse, 0.5).pvalue) if better + worse else 1.0
    return {"better": better, "worse": worse, "equal": int(len(a) - better - worse), "p": p}


# --- Zwei Ziele statt einer Grenze (ParEGO) ---------------------------------------------------------------------------------------------

PARETO_REF = (800.0, 20.0)       # Referenzpunkt (Gerätekosten EUR/h, mittlere Gesamtzeit min) des Hypervolumens


def hypervolume_2d(points, ref=PARETO_REF):
    """Hypervolumen (Minimierung, zwei Ziele) der Punktmenge gegen `ref`: Fläche der dominierten Region bis zum Referenzpunkt."""
    pts = np.array([p for p in points if p[0] < ref[0] and p[1] < ref[1]], float)
    if len(pts) == 0:
        return 0.0
    pts = pts[np.argsort(pts[:, 0])]
    hv, best_t = 0.0, ref[1]
    for c, t in pts:
        if t < best_t:
            hv += (ref[0] - c) * (best_t - t)
            best_t = t
    return float(hv)


def non_dominated(points):
    """Indizes der nicht dominierten Punkte (Minimierung beider Ziele)."""
    keep = []
    for i, p in enumerate(points):
        dominated = any((q[0] <= p[0] and q[1] <= p[1]) and (q[0] < p[0] or q[1] < p[1]) for j, q in enumerate(points) if j != i)
        if not dominated:
            keep.append(i)
    return keep


def pareto_run(method, settings, budget):
    """Zwei Ziele (Gerätekosten bekannt, Zeit verrauscht): 'random' oder 'parego' (zufällige Gewichte, Tschebyscheff-Skalarisierung, EI). Hypervolumen der nicht dominierten
    ausgewerteten Einstellungen **an ihren wahren Werten**."""
    cost, t_true, cand, pts = setup(settings)
    seed = settings.seed + (91000 if method == "random" else 92000)
    field = S.TimeField(settings.trucks, seed)
    rng = np.random.default_rng(seed + 7)
    ev = lambda i: field.observe(int(pts[i][0]), int(pts[i][1]))
    if method == "random":
        idx, t = A.random_search(len(cand), ev, budget, rng)
    else:
        idx = [int(i) for i in rng.choice(len(cand), size=C.N_START, replace=False)]
        t = [ev(i) for i in idx]
        while len(idx) < budget:
            c = cost[idx]
            Z = np.column_stack([(c - c.min()) / max(c.max() - c.min(), 1e-9), (np.array(t) - min(t)) / max(max(t) - min(t), 1e-9)])
            w = rng.uniform()
            s = np.maximum(w * Z[:, 0], (1 - w) * Z[:, 1]) + 0.05 * (w * Z[:, 0] + (1 - w) * Z[:, 1])
            gp = A.GaussianProcess().fit(cand[idx], s)
            mu, sd = gp.predict(cand)
            best = float(np.min(gp.predict(cand[idx])[0]))
            z = (best - mu) / np.maximum(sd, 1e-12)
            ei = (best - mu) * norm_cdf(z) + sd * norm_pdf(z)
            ei[idx] = -np.inf
            i = int(np.argmax(ei))
            idx.append(i)
            t.append(ev(i))
    true_pts = [(float(cost[i]), float(t_true[i])) for i in idx]
    keep = non_dominated(true_pts)
    return hypervolume_2d([true_pts[k] for k in keep]), len(keep)


def norm_cdf(z):
    from scipy.stats import norm
    return norm.cdf(z)


def norm_pdf(z):
    from scipy.stats import norm
    return norm.pdf(z)


def pareto_summary(trucks, budget):
    raw = study()["pareto"][f"{int(trucks)}_{int(budget)}"]
    out = {}
    for m, runs in raw.items():
        hv = np.array([r["hv"] for r in runs], float)
        out[m] = {"median": float(np.median(hv)), "mean": float(hv.mean()), "min": float(hv.min()), "max": float(hv.max()), "front": float(np.median([r["front"] for r in runs]))}
    return out


def pareto_reference():
    """Hypervolumen der wahren Pareto-Front aller 204 Einstellungen (Obergrenze)."""
    cost, t_true = S.cost_grid().reshape(-1), S.truth_time().reshape(-1)
    pts = list(zip(cost.tolist(), t_true.tolist()))
    return hypervolume_2d([pts[k] for k in non_dominated(pts)])


def aggregate():
    """Über alle zwölf Einstellungen (3 Grenzen x 2 Lauflängen x 2 Budgets) und je 20 Läufe: Anteil wirklich zulässiger Empfehlungen, Anteil 'zulässig behauptet, aber verletzt'
    und Mittel/Median der Mehrkosten der wirklich zulässigen."""
    out = {m: {"n": 0, "feasible": 0, "violated_but_claimed": 0, "excess": []} for m in C.METHODS}
    for runs in study()["runs"].values():
        for m, recs in runs.items():
            a = out[m]
            for r in recs:
                a["n"] += 1
                a["feasible"] += int(r["truly"])
                a["violated_but_claimed"] += int(r["claimed"] and not r["truly"])
                if r["truly"]:
                    a["excess"].append(r["excess"])
    for a in out.values():
        a["median_excess"] = float(np.median(a["excess"])) if a["excess"] else float("nan")
        a["mean_excess"] = float(np.mean(a["excess"])) if a["excess"] else float("nan")
        del a["excess"]
    return out


def paired_aggregate(key, base):
    """Gepaarter Vergleich der wirklichen Zulässigkeit über alle zwölf Einstellungen (240 Paare): besser, schlechter, p (Vorzeichentest)."""
    from scipy.stats import binomtest
    better = worse = 0
    for runs in study()["runs"].values():
        a = np.array([r["truly"] for r in runs[key]], bool)
        b = np.array([r["truly"] for r in runs[base]], bool)
        better += int((a & ~b).sum())
        worse += int((~a & b).sum())
    p = float(binomtest(better, better + worse, 0.5).pvalue) if better + worse else 1.0
    return {"better": better, "worse": worse, "p": p}


def margin_summary():
    """Wie weit liegt das wirkliche zulässige Optimum von der Grenze: T des Optimums (min) und Abstand zur Grenze je Zeitgrenze."""
    t = S.truth_time()
    out = {}
    for tmax in C.TMAX_CHOICES:
        i, j, cost = S.constrained_optimum(tmax)
        out[tmax] = {"t_opt": float(t[i, j]), "margin": float(tmax - t[i, j]), "feasible_share": float((t <= tmax).mean())}
    return out
