"""Gemeinsame Fixtures: Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini():
    """Fünf Einstellungen auf einer Linie: Kosten 5, 4, 3, 2, 1 (links teuer, rechts billig); wahre Zeit 8, 9, 10, 11, 12 (rechts langsam). Grenze 10,5:
    zulässig sind die ersten drei, das billigste zulässige ist Index 2 (Kosten 3)."""
    return {"cand": np.linspace(0, 1, 5)[:, None], "cost": np.array([5.0, 4.0, 3.0, 2.0, 1.0]), "t": np.array([8.0, 9.0, 10.0, 11.0, 12.0]), "tmax": 10.5}
