"""Konstanten der Demo „Nebenbedingungen“: das Terminal als teurer Simulator mit verrauschter Zeitgrenze, Regler, Verfahren, Studie, Presets, Farben.

Zeiten in Minuten, Kosten in EUR je Stunde. Die Auswertungen stammen aus der Tabelle echter Simulationen der Terminal-Scheibe (Plätze am Stapel x Tempo des Stapels)
aus bayes-opt-demo (`precomputed_slice.json`). Die Gerätekosten sind bekannt und deterministisch; **die mittlere Gesamtzeit ist die verrauschte Nebenbedingung**.
"""

# --- Das Terminal (Annahmen der Demo, wie in bayes-opt-demo) ----------------------------------------------------------------------------
GAMMA_PER_HOUR = 33.0
GATE_SERVERS, KRAN_SERVERS, STAPEL_SERVERS = 4, 4, 5
KRAN_SPEED = 0.7
K_GATE, K_KRAN, K_STAPEL, K_PLACE = 20.0, 120.0, 60.0, 4.0
V_WAIT = 40.0
T_CAP = 300.0
WAIT_COST_PER_MIN = V_WAIT * GAMMA_PER_HOUR / 60.0     # EUR/h je Minute mittlere Gesamtzeit (22)

PLACES = tuple(range(2, 14))
SPEEDS = tuple(round(0.6 + 0.05 * i, 2) for i in range(17))
N_PLACES, N_SPEEDS = len(PLACES), len(SPEEDS)
TRUCK_CHOICES = (3000, 15000)
TRUCKS_LABEL = {3000: "3 000 Lkw (billig, verrauscht)", 15000: "15 000 Lkw (teurer, ruhiger)"}
DEFAULT_TRUCKS = 3000
REPLICATES = 30

# --- Die Nebenbedingung: mittlere Gesamtzeit höchstens TMAX Minuten ----------------------------------------------------------------------
TMAX_CHOICES = (11.0, 12.0, 15.0)
TMAX_LABEL = {11.0: "11 min (scharf)", 12.0: "12 min (mittel)", 15.0: "15 min (locker)"}
DEFAULT_TMAX = 12.0
PENALTY_PER_MIN = 100.0          # EUR/h je Minute über der Grenze (Strafverfahren)

# --- Verfahren --------------------------------------------------------------------------------------------------------------------------
METHODS = ("random", "penalty", "pof", "pof_safe", "pof_check")
METHOD_LABELS = {
    "random": "Zufall",
    "penalty": "Strafe",
    "pof": "Zulässigkeitswahrscheinlichkeit (PoF)",
    "pof_safe": "PoF, nur Empfehlungen mit P(zulässig) ≥ 90 %",
    "pof_check": "PoF mit Nachprüfung",
}
METHOD_COLORS = {"random": "#8c8c8c", "penalty": "#D68A2E", "pof": "#2F6B65", "pof_safe": "#14233B", "pof_check": "#7B5EA7"}
SAFE_PROB = 0.9                  # Mindest-Zulässigkeitswahrscheinlichkeit der „sicheren“ Empfehlung
CHECK_RESERVE = 6                # Auswertungen, die „PoF mit Nachprüfung“ für drei Kandidaten à 2 Wiederholungen zurückhält
CHECK_CANDIDATES, CHECK_REPEATS = 3, 2
N_START = 5

# --- Regler -----------------------------------------------------------------------------------------------------------------------------
N_MIN, N_MAX, DEFAULT_N = 12, 30, 20
SEED_MAX, DEFAULT_SEED = 999999, 35
SWEEP_SEEDS = tuple(range(600000, 600020))
STUDY_BUDGETS = (20, 30)

PRESETS = {
    "Standardfall (Voreinstellung)": {"n": DEFAULT_N, "tmax": DEFAULT_TMAX, "trucks": DEFAULT_TRUCKS, "method": "pof", "seed": DEFAULT_SEED},
    "Strafe": {"n": DEFAULT_N, "tmax": DEFAULT_TMAX, "trucks": DEFAULT_TRUCKS, "method": "penalty", "seed": DEFAULT_SEED},
    "Nur sichere Empfehlungen": {"n": DEFAULT_N, "tmax": DEFAULT_TMAX, "trucks": DEFAULT_TRUCKS, "method": "pof_safe", "seed": DEFAULT_SEED},
    "Mit Nachprüfung": {"n": DEFAULT_N, "tmax": DEFAULT_TMAX, "trucks": DEFAULT_TRUCKS, "method": "pof_check", "seed": DEFAULT_SEED},
    "Scharfe Grenze, ruhige Läufe": {"n": 30, "tmax": 11.0, "trucks": 15000, "method": "pof", "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Grenze 12 min, kurze verrauschte Läufe, 20 Auswertungen, Suche nach erwarteter Verbesserung mal Zulässigkeitswahrscheinlichkeit.",
    "Strafe": "Statt die Zulässigkeit zu modellieren, wird jede Minute über der Grenze mit 100 EUR/h bestraft.",
    "Nur sichere Empfehlungen": "Empfohlen wird nur, was nach dem Modell mit mindestens 90 % Wahrscheinlichkeit zulässig ist.",
    "Mit Nachprüfung": "Die letzten sechs Auswertungen prüfen die drei besten Kandidaten je zweimal nach, bevor empfohlen wird.",
    "Scharfe Grenze, ruhige Läufe": "Grenze 11 min (nur 39 % der Einstellungen zulässig), ruhige Läufe mit 15 000 Lkw, 30 Auswertungen.",
}

COLOR_FEASIBLE = "#2F6B65"
COLOR_INFEASIBLE = "#B3463A"
COLOR_NEXT = "#D68A2E"
COLOR_BASE = "#14233B"
