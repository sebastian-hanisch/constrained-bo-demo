"""Jede Zahl aus dem README, nachgerechnet über dieselben Auswertungsfunktionen wie die App (vorgerechnete Studie: 3 Zeitgrenzen x 2 Lauflängen x 2 Budgets x 5 Verfahren x 20 Läufe)."""

import pytest

import cbo_constants as C
import cbo_evaluation as ev
import cbo_scenario as S


def test_the_constrained_optimum_sits_just_below_the_limit():
    m = ev.margin_summary()
    assert m[11.0]["margin"] == pytest.approx(0.14, abs=0.02) and m[12.0]["margin"] == pytest.approx(0.24, abs=0.02) and m[15.0]["margin"] == pytest.approx(0.06, abs=0.02)
    assert m[11.0]["feasible_share"] == pytest.approx(0.39, abs=0.02) and m[12.0]["feasible_share"] == pytest.approx(0.56, abs=0.02)
    i, j, cost = S.constrained_optimum(12.0)
    assert (C.PLACES[i], C.SPEEDS[j]) == (5, pytest.approx(0.9)) and cost == pytest.approx(578.2, abs=0.5)


def test_noise_of_the_time_observation_is_small_but_comparable_to_the_margin():
    import numpy as np
    t = S.truth_time()
    ok = t < 60
    for trucks, expected in ((3000, 0.27), (15000, 0.12)):
        sd = S.reps_time(trucks).std(-1)
        assert float(np.median(sd[ok])) == pytest.approx(expected, abs=0.04)


def test_plain_pof_rarely_recommends_something_truly_feasible():
    agg = ev.aggregate()
    assert agg["pof"]["n"] == 240
    assert agg["pof"]["feasible"] / 240 == pytest.approx(0.28, abs=0.05)
    for m in ("random", "penalty", "pof_safe", "pof_check"):
        assert agg[m]["feasible"] / 240 > 0.65, m
    assert agg["pof_check"]["feasible"] / 240 == pytest.approx(0.78, abs=0.05)


def test_plain_pof_is_provably_worse_than_random_and_nobody_is_provably_better():
    assert ev.paired_aggregate("pof", "random")["p"] < 1e-6
    assert (ev.paired_aggregate("pof", "random")["better"], ev.paired_aggregate("pof", "random")["worse"]) == (15, 124)
    for k in ("penalty", "pof_safe", "pof_check"):
        s = ev.paired_aggregate(k, "random")
        assert s["p"] > 0.2, k


def test_safety_by_distance_costs_money():
    agg = ev.aggregate()
    assert agg["pof_safe"]["median_excess"] == pytest.approx(69, abs=15)
    assert agg["random"]["median_excess"] == pytest.approx(16, abs=8)
    assert agg["pof_safe"]["median_excess"] > 2 * agg["random"]["median_excess"]


def test_one_setting_in_detail():
    s = ev.method_summary(12.0, 3000, 20)
    assert (s["pof"]["feasible"], s["random"]["feasible"], s["pof_check"]["feasible"]) == (3, 14, 16)
    assert s["pof"]["violated_but_claimed"] == 17
    s30 = ev.method_summary(12.0, 3000, 30)
    assert s30["pof"]["feasible"] == 0


def test_pareto_parego_is_no_better_than_random():
    ref = ev.study()["pareto_reference_hv"]
    assert ref == pytest.approx(2871, abs=30)
    for trucks in C.TRUCK_CHOICES:
        for b in C.STUDY_BUDGETS:
            s = ev.pareto_summary(trucks, b)
            assert s["random"]["median"] / ref > 0.88
            assert abs(s["parego"]["median"] - s["random"]["median"]) < 0.03 * ref


def test_study_geometry():
    st = ev.study()
    assert st["meta"]["seeds"] == 20
    assert len(st["runs"]) == len(C.TMAX_CHOICES) * len(C.TRUCK_CHOICES) * len(C.STUDY_BUDGETS)
    for key, runs in st["runs"].items():
        assert set(runs) == set(C.METHODS), key
        for m, recs in runs.items():
            assert len(recs) == 20, (key, m)
