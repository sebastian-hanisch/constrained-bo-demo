"""Einheiten des Kerns an Mini-Instanzen (Handrechnung): Zulässigkeit, beschränkte Verbesserung, Strafe, Empfehlungsregeln, Nachprüfung, Suche."""

import math

import numpy as np
import pytest

import cbo_algorithm as A


def test_matern_kernel_by_hand():
    r = 1.0 / math.sqrt(5.0)
    assert A.matern52(np.array([[0.0]]), np.array([[r]]), np.array([1.0]), 1.0)[0, 0] == pytest.approx((7 / 3) * math.exp(-1), rel=1e-12)


def test_probability_of_feasibility_by_hand():
    """Grenze 10; Mittel 10, Streuung 1: 0,5. Mittel 9: Φ(1) = 0,841345. Mittel 11: Φ(-1) = 0,158655. Ohne Streuung: Stufe."""
    one = np.array([1.0])
    assert A.probability_of_feasibility(np.array([10.0]), one, 10.0)[0] == pytest.approx(0.5)
    assert A.probability_of_feasibility(np.array([9.0]), one, 10.0)[0] == pytest.approx(0.8413447, rel=1e-6)
    assert A.probability_of_feasibility(np.array([11.0]), one, 10.0)[0] == pytest.approx(0.1586553, rel=1e-6)
    assert A.probability_of_feasibility(np.array([9.0]), np.array([0.0]), 10.0)[0] == pytest.approx(1.0)
    assert A.probability_of_feasibility(np.array([11.0]), np.array([0.0]), 10.0)[0] == pytest.approx(0.0, abs=1e-9)


def test_constrained_improvement_by_hand():
    """Beste zulässige Kosten 4: Kandidat mit Kosten 3 und Zulässigkeit 0,5 ergibt 0,5; Kandidat mit Kosten 5 (teurer) ergibt 0; ohne zulässigen Punkt zählt nur die Zulässigkeit."""
    cost = np.array([3.0, 5.0, 2.0])
    pof = np.array([0.5, 0.9, 0.1])
    assert np.allclose(A.constrained_improvement(cost, 4.0, pof), [0.5, 0.0, 0.2])
    assert np.allclose(A.constrained_improvement(cost, np.inf, pof), pof)


def test_penalized_value_by_hand():
    """Kosten 100, Zeit 12 bei Grenze 10, Strafe 50 je Minute: 100 + 50 · 2 = 200; innerhalb der Grenze keine Strafe."""
    assert A.penalized_value(100.0, 12.0, 10.0, 50.0) == pytest.approx(200.0)
    assert A.penalized_value(100.0, 9.0, 10.0, 50.0) == pytest.approx(100.0)


def test_recommend_obs_picks_the_cheapest_observed_feasible(mini):
    idx = [0, 1, 2, 3, 4]
    t_obs = [8.0, 9.0, 10.0, 10.2, 12.0]        # Index 3 wurde zu schön beobachtet (wahr 11)
    rec, ok = A.recommend_obs(idx, t_obs, mini["cost"], mini["tmax"])
    assert rec == 3 and ok                          # billigste beobachtet zulässige: Index 3 (Kosten 2), obwohl wahr unzulässig
    rec, ok = A.recommend_obs(idx, [11.0, 12.0, 13.0, 14.0, 15.0], mini["cost"], mini["tmax"])
    assert rec == 0 and not ok                      # keine zulässige Beobachtung: die mit der kleinsten Zeit


def test_recommend_safe_requires_model_confidence(mini):
    """Mit dichten, sehr eindeutigen Beobachtungen entlang der Linie: die sichere Regel wählt das billigste mit P(zulässig) ≥ 0,9, nicht das billigere Grenzwertige."""
    cand = np.linspace(0, 1, 11)[:, None]
    cost = np.linspace(10, 0, 11)
    t = np.linspace(8.0, 14.0, 11)                  # zulässig (Grenze 10): Indizes 0 bis 3 (T = 8, 8,6, 9,2, 9,8)
    idx = list(range(11))
    rec, ok = A.recommend_safe(cand, idx, list(t), cost, 10.0, prob=0.9)
    assert ok and rec in (2, 3)                     # das billigste mit hoher Zulässigkeit; Index 4 (T = 10,4) ist es nicht
    assert cost[rec] <= cost[0]


def test_check_candidates_re_evaluates_the_cheapest_feasible_and_averages(mini):
    idx = [0, 1, 2, 3, 4]
    t_obs = [8.0, 9.0, 10.0, 10.2, 12.0]
    calls = []

    def ev(i):
        calls.append(i)
        return 11.0 if i == 3 else float(mini["t"][i])

    idx2, t2 = A.check_candidates(idx, t_obs, mini["cost"], mini["tmax"], ev, k_candidates=2, repeats=2)
    assert calls == [3, 3, 2, 2]                                       # die zwei billigsten beobachtet zulässigen: Index 3 (Kosten 2), Index 2 (Kosten 3)
    assert t2[3] == pytest.approx((10.2 + 11.0 + 11.0) / 3)           # Mittel aus drei Beobachtungen
    assert t2[2] == pytest.approx(10.0)
    rec, ok = A.recommend_obs(idx2, t2, mini["cost"], mini["tmax"])
    assert rec == 2 and ok                                             # nach der Nachprüfung ist Index 3 nicht mehr zulässig


def test_searches_respect_the_budget_and_never_repeat(mini):
    cand = np.linspace(0, 1, 21)[:, None]
    cost = np.linspace(10, 0, 21)
    t_true = np.linspace(8, 14, 21)
    rng = np.random.default_rng(3)
    ev = lambda i: float(t_true[i] + 0.1 * rng.standard_normal())
    for fn in (lambda r: A.pof_search(cand, cost, ev, 9, r, 10.0, n_start=4), lambda r: A.penalty_search(cand, cost, ev, 9, r, 10.0, 100.0, n_start=4)):
        idx, t = fn(np.random.default_rng(5))
        assert len(idx) == 9 == len(set(idx)) and len(t) == 9
    idx, t = A.random_search(len(cand), ev, 9, np.random.default_rng(1))
    assert len(idx) == 9 == len(set(idx))


def test_pof_search_moves_towards_the_boundary_and_not_into_the_infeasible_region():
    """Entlang einer Linie mit rauschfreier Zeit: die Suche sammelt ihre Auswertungen nahe der Grenze T = 10 (x ≈ 0,33) und nicht im schnellen, teuren Bereich."""
    cand = np.linspace(0, 1, 41)[:, None]
    cost = np.linspace(10, 0, 41)
    t_true = np.linspace(8, 14, 41)
    idx, t = A.pof_search(cand, cost, lambda i: float(t_true[i]), 14, np.random.default_rng(2), 10.0, n_start=4)
    later = idx[4:]
    assert np.median(np.abs(cand[later, 0] - 0.3333)) < 0.2


def test_choice_uses_the_largest_improvement_and_falls_back_to_the_most_uncertain_candidate():
    """Kosten 5, 4, 3, 2, 1; bester zulässiger Wert 3 (Index 2). Zulässigkeiten (1, 1, 0,9, 0,02, 0,45): Verbesserung Index 3: 1 · 0,02, Index 4: 2 · 0,45 = 0,9, also Index 4.
    Ist niemand billiger als der beste zulässige Wert (hier 0,5), entscheidet die Unsicherheit min(P, 1 − P) unter den noch nicht ausgewerteten: Index 3 (0,4) gegen Index 4 (0,1)."""
    cost = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert A.choose_by_constrained_improvement(cost, 3.0, np.array([1.0, 1.0, 0.9, 0.02, 0.45]), [0, 1, 2]) == 4
    assert A.choose_by_constrained_improvement(cost, 0.5, np.array([1.0, 1.0, 0.9, 0.4, 0.1]), [0, 1, 2]) == 3
