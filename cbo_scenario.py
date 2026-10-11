"""Das Terminal als Aufgabe mit Zeitgrenze: bekannte Gerätekosten, verrauschte mittlere Gesamtzeit aus der Tabelle echter Simulationen."""

import json
import os

import numpy as np

import cbo_constants as C

_HERE = os.path.dirname(os.path.abspath(__file__))
_RAW = None


def _raw():
    global _RAW
    if _RAW is None:
        with open(os.path.join(_HERE, "precomputed_slice.json"), encoding="utf-8") as fh:
            _RAW = json.load(fh)
    return _RAW


def device_cost(places, speed):
    """Gerätekosten (EUR/h): Gate, Kran (Tempo 0,7), Stapel (Tempo `speed`, quadratisch) und Stellplätze. Bekannt und deterministisch."""
    return C.K_GATE * C.GATE_SERVERS + C.K_KRAN * C.KRAN_SERVERS * C.KRAN_SPEED ** 2 + C.K_STAPEL * C.STAPEL_SERVERS * speed ** 2 + C.K_PLACE * places


def cost_grid():
    """Gerätekosten je Einstellung [Plätze, Tempo]."""
    return np.array([[device_cost(k, s) for s in C.SPEEDS] for k in C.PLACES])


def time_from_total(total, dev):
    """Mittlere Gesamtzeit aus den Gesamtkosten: total = dev + WAIT_COST_PER_MIN · T (die Wartekosten sind linear in T)."""
    return (np.asarray(total, float) - dev) / C.WAIT_COST_PER_MIN


def truth_time():
    """Wahre mittlere Gesamtzeit je Einstellung [Plätze, Tempo] (Minuten): aus der Wahrheit der Tabelle (zwei Läufe à 100 000 Lkw)."""
    return time_from_total(np.array(_raw()["truth"], float), cost_grid())


def reps_time(trucks):
    """Wiederholungen der Gesamtzeit [Plätze, Tempo, r] (Minuten) für die Lauflänge `trucks` (30 echte Simulationen je Einstellung)."""
    return time_from_total(np.array(_raw()["reps"][str(trucks)], float), cost_grid()[:, :, None])


def grid_points(shape):
    rows, cols = shape
    return np.array([(i, j) for i in range(rows) for j in range(cols)], dtype=int)


def to_unit(pts, shape):
    rows, cols = shape
    pts = np.asarray(pts, float)
    return np.column_stack([pts[:, 0] / (rows - 1), pts[:, 1] / (cols - 1)])


class TimeField:
    """Verrauschte Auswertung der Gesamtzeit: je Einstellung eine noch nicht gezogene der 30 Wiederholungen (danach beginnt der Vorrat von vorn)."""

    def __init__(self, trucks, seed):
        self.reps = reps_time(trucks)
        self.rng = np.random.default_rng(seed)
        self.used = {}

    def observe(self, i, j):
        used = self.used.setdefault((i, j), [])
        free = [r for r in range(C.REPLICATES) if r not in used]
        if not free:
            used.clear()
            free = list(range(C.REPLICATES))
        r = int(self.rng.choice(free))
        used.append(r)
        return float(self.reps[i, j, r])


def constrained_optimum(tmax):
    """Billigste wirklich zulässige Einstellung (Wahrheit): (Plätze-Index, Tempo-Index, Gerätekosten)."""
    cost, t = cost_grid(), truth_time()
    c = np.where(t <= tmax, cost, np.inf)
    i, j = np.unravel_index(int(np.argmin(c)), c.shape)
    return int(i), int(j), float(c[i, j])
