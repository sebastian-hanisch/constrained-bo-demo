"""Orakel-Tests constrained-bo-demo: Kern, Hypervolumen und Tabelle gegen unabhängige Rechenwege (ORACLE-PLAYBOOK).

* Gauß-Prozess gegen scikit-learn; Zulässigkeitswahrscheinlichkeit gegen numerische Integration; beschränkte Verbesserung gegen Monte-Carlo.
* Hypervolumen gegen Rasterzählung; Nicht-Dominierte gegen die Definition; zulässiges Optimum gegen Aufzählung.
* Zeit aus Gesamtkosten gegen die Kostenformel; Vorzeichentest von Hand.
"""

import numpy as np
import pytest

import cbo_algorithm as A
import cbo_constants as C
import cbo_evaluation as ev
import cbo_scenario as S


def test_gp_prediction_matches_scikit_learn_at_the_learned_hyperparameters():
    gpm = pytest.importorskip("sklearn.gaussian_process")
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern

    rng = np.random.default_rng(11)
    X = rng.random((10, 2))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2 + 0.05 * rng.standard_normal(10)
    gp = A.GaussianProcess().fit(X, y)
    z = (y - gp.y_mean) / gp.y_std
    sk = gpm.GaussianProcessRegressor(ConstantKernel(gp.variance, "fixed") * Matern(gp.lengthscales, "fixed", nu=2.5), alpha=gp.noise + 1e-8, optimizer=None).fit(X, z)
    Xs = rng.random((12, 2))
    m_ref, s_ref = sk.predict(Xs, return_std=True)
    mu, sd = gp.predict(Xs)
    assert np.allclose((mu - gp.y_mean) / gp.y_std, m_ref, atol=1e-6)
    assert np.allclose(sd / gp.y_std, s_ref, atol=1e-5)


def test_probability_of_feasibility_matches_numerical_integration():
    quad = pytest.importorskip("scipy.integrate").quad
    from scipy.stats import norm

    for mu, sd, tmax in [(10.0, 1.0, 10.5), (9.0, 0.4, 10.0), (11.0, 2.0, 10.0)]:
        ref, _ = quad(lambda t: norm.pdf(t, mu, sd), mu - 12 * sd, tmax, limit=200)
        assert A.probability_of_feasibility(np.array([mu]), np.array([sd]), tmax)[0] == pytest.approx(ref, rel=1e-6)


def test_constrained_improvement_matches_monte_carlo_for_independent_cost_and_time():
    """Bei bekanntem Ziel ist E[(best − c)₊ · 1{T ≤ tmax}] = (best − c)₊ · P(T ≤ tmax): Monte-Carlo-Gegenprobe der Formel."""
    rng = np.random.default_rng(1)
    mu, sd, tmax, c, best = 10.0, 0.8, 10.5, 3.0, 4.5
    t = mu + sd * rng.standard_normal(400_000)
    mc = (max(best - c, 0.0) * (t <= tmax)).mean()
    ref = A.constrained_improvement(np.array([c]), best, A.probability_of_feasibility(np.array([mu]), np.array([sd]), tmax))[0]
    assert ref == pytest.approx(mc, rel=0.01)


def test_hypervolume_matches_grid_counting():
    rng = np.random.default_rng(4)
    pts = [(float(a), float(b)) for a, b in rng.uniform(0, 1, (7, 2))]
    ref = (1.0, 1.0)
    xs, ys = np.meshgrid((np.arange(2000) + 0.5) / 2000, (np.arange(2000) + 0.5) / 2000)
    dominated = np.zeros_like(xs, bool)
    for a, b in pts:
        dominated |= (xs >= a) & (ys >= b)
    assert ev.hypervolume_2d(pts, ref) == pytest.approx(dominated.mean(), abs=2e-3)


def test_non_dominated_matches_the_definition():
    rng = np.random.default_rng(6)
    pts = [(float(a), float(b)) for a, b in rng.uniform(0, 1, (25, 2))]
    keep = set(ev.non_dominated(pts))
    for i, p in enumerate(pts):
        dominated = any(q[0] <= p[0] and q[1] <= p[1] and (q[0] < p[0] or q[1] < p[1]) for j, q in enumerate(pts) if j != i)
        assert (i in keep) == (not dominated)


def test_constrained_optimum_matches_enumeration():
    cost, t = S.cost_grid(), S.truth_time()
    for tmax in C.TMAX_CHOICES:
        best = min((cost[i, j], i, j) for i in range(cost.shape[0]) for j in range(cost.shape[1]) if t[i, j] <= tmax)
        i, j, c = S.constrained_optimum(tmax)
        assert (i, j) == (best[1], best[2]) and c == pytest.approx(best[0])


def test_time_from_total_inverts_the_cost_formula():
    """Gesamtkosten = Geräte + 22 EUR/h je Minute: 100 + 22 · 11 = 342 gibt T = 11."""
    assert S.time_from_total(100.0 + 22.0 * 11.0, 100.0) == pytest.approx(11.0)
    assert C.WAIT_COST_PER_MIN == pytest.approx(22.0)


def test_device_cost_by_hand():
    """Gate 4 · 20 = 80; Kran 4 · 120 · 0,49 = 235,2; Stapel 5 · 60 · 1 = 300; 5 Plätze · 4 = 20: 635,2."""
    assert S.device_cost(5, 1.0) == pytest.approx(635.2)


def test_sign_test_by_hand():
    from scipy.stats import binomtest
    assert binomtest(9, 10, 0.5).pvalue == pytest.approx(22 / 1024)
