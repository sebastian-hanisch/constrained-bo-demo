"""Kern der Demo: Gauß-Prozess für die verrauschte Nebenbedingung, Zulässigkeitswahrscheinlichkeit, beschränkte erwartete Verbesserung, fünf Suchverfahren.

Die Zielgröße (Gerätekosten `cost`) ist bekannt; nur die Gesamtzeit T (Nebenbedingung T ≤ tmax) wird durch Läufe beobachtet und mit einem Gauß-Prozess modelliert.
Eine Einheit = eine Regel (DEMO-PLAYBOOK §1b): `matern52`, `GaussianProcess`, `probability_of_feasibility`, `constrained_improvement`, `penalized_value`,
`recommend_obs`, `recommend_safe`, die Suchen `random_search`, `penalty_search`, `pof_search`, `check_candidates`. Zufall nur über übergebene Generatoren.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import norm


def matern52(A, B, lengthscales, variance):
    """Matérn-5/2-Kern k(a, b) = σ² (1 + √5 r + 5r²/3) exp(−√5 r), r = Abstand mit je Eingabe eigenem Längenmaß."""
    r2 = np.maximum((((A[:, None, :] - B[None, :, :]) / lengthscales) ** 2).sum(-1), 0.0)
    s = np.sqrt(5.0) * np.sqrt(r2)
    return variance * (1.0 + s + s * s / 3.0) * np.exp(-s)


class GaussianProcess:
    """Gauß-Prozess-Regression, Ausgabe standardisiert; Längenmaße, Signal- und Rauschvarianz durch Maximieren der Randwahrscheinlichkeit (L-BFGS-B, feste Starts)."""

    def __init__(self, restarts=3):
        self.restarts = restarts

    def _unpack(self, theta, d):
        return np.exp(theta[:d]), np.exp(theta[d]), np.exp(theta[d + 1])

    def _neg_log_marginal(self, theta, X, z):
        ls, var, nz = self._unpack(theta, X.shape[1])
        K = matern52(X, X, ls, var) + (nz + 1e-8) * np.eye(len(X))
        try:
            c = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10
        a = cho_solve(c, z)
        return 0.5 * z @ a + np.log(np.diag(c[0])).sum() + 0.5 * len(z) * np.log(2 * np.pi)

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.X = X
        self.y_mean = float(y.mean())
        self.y_std = max(float(y.std()), 1e-9)
        z = (y - self.y_mean) / self.y_std
        d = X.shape[1]
        bounds = [(np.log(0.05), np.log(10.0))] * d + [(np.log(0.05), np.log(10.0)), (np.log(1e-6), np.log(2.0))]
        starts = np.random.default_rng(0).uniform(0.15, 1.0, (self.restarts, d))
        best = None
        for r in range(self.restarts):
            th0 = np.concatenate([np.log(starts[r]), [0.0, np.log(1e-2)]])
            res = minimize(self._neg_log_marginal, th0, args=(X, z), method="L-BFGS-B", bounds=bounds)
            if best is None or res.fun < best.fun:
                best = res
        self.lengthscales, self.variance, self.noise = self._unpack(best.x, d)
        K = matern52(X, X, self.lengthscales, self.variance) + (self.noise + 1e-8) * np.eye(len(X))
        self._chol = cho_factor(K, lower=True)
        self._alpha = cho_solve(self._chol, z)
        return self

    def predict(self, Xs):
        """Mittelwert und Standardabweichung der **Funktion** (ohne Beobachtungsrauschen)."""
        Ks = matern52(np.asarray(Xs, float), self.X, self.lengthscales, self.variance)
        mu = Ks @ self._alpha
        v = cho_solve(self._chol, Ks.T)
        var = np.maximum(self.variance - (Ks * v.T).sum(1), 1e-12)
        return mu * self.y_std + self.y_mean, np.sqrt(var) * self.y_std


def probability_of_feasibility(mu, sd, tmax):
    """P(T(x) ≤ tmax) für T(x) ~ N(mu, sd²): Φ((tmax − mu)/sd)."""
    return norm.cdf((tmax - np.asarray(mu, float)) / np.maximum(np.asarray(sd, float), 1e-12))


def constrained_improvement(cost, best_cost, pof):
    """Beschränkte erwartete Verbesserung bei **bekanntem** Ziel: (best_cost − cost)₊ · P(zulässig). Ohne zulässigen Punkt (best_cost = ∞): nur die Zulässigkeit."""
    if not np.isfinite(best_cost):
        return np.asarray(pof, float)
    return np.maximum(best_cost - np.asarray(cost, float), 0.0) * np.asarray(pof, float)


def penalized_value(cost, t_obs, tmax, penalty):
    """Strafverfahren: Kosten plus `penalty` je Minute über der Grenze."""
    return np.asarray(cost, float) + penalty * np.maximum(np.asarray(t_obs, float) - tmax, 0.0)


# --- Empfehlungsregeln ------------------------------------------------------------------------------------------------------------------

def recommend_obs(idx, t_obs, cost, tmax):
    """Billigste ausgewertete Einstellung, deren **beobachtete** Zeit die Grenze einhält; ohne solche: die mit der kleinsten beobachteten Zeit. Gibt (Kandidat, wurde_zulässig_beobachtet)."""
    ok = [k for k, t in enumerate(t_obs) if t <= tmax]
    if ok:
        k = min(ok, key=lambda q: cost[idx[q]])
        return idx[k], True
    return idx[int(np.argmin(t_obs))], False


def recommend_safe(cand, idx, t_obs, cost, tmax, prob=0.9):
    """Billigste ausgewertete Einstellung mit Modell-Zulässigkeitswahrscheinlichkeit ≥ prob (Gauß-Prozess auf allen Beobachtungen); ohne solche die mit der größten."""
    gp = GaussianProcess().fit(cand[idx], t_obs)
    mu, sd = gp.predict(cand[idx])
    pof = probability_of_feasibility(mu, sd, tmax)
    ok = [k for k in range(len(idx)) if pof[k] >= prob]
    if ok:
        k = min(ok, key=lambda q: cost[idx[q]])
        return idx[k], True
    return idx[int(np.argmax(pof))], False


# --- Suchverfahren ----------------------------------------------------------------------------------------------------------------------

def random_search(n_cand, evaluate, budget, rng):
    """Zufällig ohne Zurücklegen; gibt (Indizes, beobachtete Zeiten) zurück."""
    idx = [int(i) for i in rng.choice(n_cand, size=min(budget, n_cand), replace=False)]
    return idx, [evaluate(i) for i in idx]


def _start(n_cand, n_start, rng):
    return [int(i) for i in rng.choice(n_cand, size=n_start, replace=False)]


def penalty_search(cand, cost, evaluate, budget, rng, tmax, penalty, n_start=5):
    """BO auf dem bestraften Wert (Kosten + Strafe · Minuten über der Grenze): erwartete Verbesserung eines einzigen Gauß-Prozesses. Gibt (Indizes, Zeiten)."""
    idx = _start(len(cand), min(n_start, budget), rng)
    t = [evaluate(i) for i in idx]
    while len(idx) < budget:
        y = penalized_value(cost[idx], t, tmax, penalty)
        gp = GaussianProcess().fit(cand[idx], y)
        mu, sd = gp.predict(cand)
        best = float(np.min(gp.predict(cand[idx])[0]))
        z = (best - mu) / np.maximum(sd, 1e-12)
        ei = (best - mu) * norm.cdf(z) + sd * norm.pdf(z)
        ei[idx] = -np.inf
        i = int(np.argmax(ei))
        idx.append(i)
        t.append(evaluate(i))
    return idx, t


def choose_by_constrained_improvement(cost, best_cost, pof, idx):
    """Die Auswahlregel von `pof_search`: größte beschränkte erwartete Verbesserung unter den noch nicht ausgewerteten Einstellungen. Ist sie überall null (kein
    billigerer Kandidat hat noch eine Zulässigkeitswahrscheinlichkeit), wird der billigere Kandidat gewählt, dessen Zulässigkeit am unsichersten ist (P nahe 0,5); gibt es keinen
    billigeren, der noch nicht ausgewertet wurde, der unsicherste insgesamt."""
    taken = np.isin(np.arange(len(cost)), idx)
    a = np.where(taken, -np.inf, constrained_improvement(cost, best_cost, pof))
    if float(np.max(a)) > 1e-12:
        return int(np.argmax(a))
    uncertainty = np.minimum(pof, 1.0 - pof)
    cheaper = (~taken) & (np.asarray(cost) < best_cost)
    pool = cheaper if cheaper.any() else ~taken
    return int(np.argmax(np.where(pool, uncertainty, -np.inf)))


def pof_search(cand, cost, evaluate, budget, rng, tmax, n_start=5):
    """BO mit Zulässigkeitswahrscheinlichkeit: ein Gauß-Prozess auf der beobachteten Zeit; gewählt wird der Kandidat mit der größten beschränkten erwarteten
    Verbesserung (best_cost − cost)₊ · P(zulässig); bester Wert = billigste beobachtet zulässige Einstellung. Gibt (Indizes, Zeiten)."""
    idx = _start(len(cand), min(n_start, budget), rng)
    t = [evaluate(i) for i in idx]
    while len(idx) < budget:
        gp = GaussianProcess().fit(cand[idx], t)
        mu, sd = gp.predict(cand)
        feas = [cost[idx[k]] for k in range(len(idx)) if t[k] <= tmax]
        pof = probability_of_feasibility(mu, sd, tmax)
        best_cost = min(feas) if feas else np.inf
        i = choose_by_constrained_improvement(cost, best_cost, pof, idx)
        idx.append(i)
        t.append(evaluate(i))
    return idx, t


def check_candidates(idx, t_obs, cost, tmax, evaluate, k_candidates, repeats):
    """Nachprüfung: die `k_candidates` billigsten beobachtet zulässigen Einstellungen werden je `repeats`-mal nachgemessen; ihre Beobachtungen werden gemittelt.
    Gibt (Indizes, Zeiten) mit dem Mittel je Einstellung zurück (unveränderte Einstellungen behalten ihren Wert)."""
    ok = sorted([k for k, t in enumerate(t_obs) if t <= tmax], key=lambda q: cost[idx[q]])[:k_candidates]
    t_new = list(t_obs)
    for k in ok:
        extra = [evaluate(idx[k]) for _ in range(repeats)]
        t_new[k] = float(np.mean([t_obs[k]] + extra))
    return idx, t_new
