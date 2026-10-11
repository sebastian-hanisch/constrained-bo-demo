"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink-Einrasten."""

import cbo_constants as C
import cbo_evaluation as ev
import cbo_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_control_bounds_and_choices():
    for preset in C.PRESETS.values():
        assert C.N_MIN <= preset["n"] <= C.N_MAX and preset["tmax"] in C.TMAX_CHOICES and preset["trucks"] in C.TRUCK_CHOICES
        assert preset["method"] in C.METHODS[1:] and 0 <= preset["seed"] <= C.SEED_MAX


def test_default_preset_equals_the_default_settings():
    p = C.PRESETS["Standardfall (Voreinstellung)"]
    assert ev.Settings(n=p["n"], tmax=p["tmax"], trucks=p["trucks"], seed=p["seed"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) <= set(P.SETTING_SPECS)


def test_permalink_values_snap_to_allowed_choices():
    assert P.snap_to_choice(11.4, C.TMAX_CHOICES, 12.0) == 11.0
    assert P.snap_to_choice(100.0, C.TMAX_CHOICES, 12.0) == 15.0
    assert P.snap_to_choice("unsinn", C.METHODS, "pof") == "pof"
    assert P.snap_to_choice("pof_safe", C.METHODS, "pof") == "pof_safe"


def test_study_budgets_cover_the_slider_range():
    assert min(C.STUDY_BUDGETS) <= C.DEFAULT_N <= max(C.STUDY_BUDGETS)
    assert C.CHECK_RESERVE < C.N_MIN
