# Nebenbedingungen: günstig und trotzdem zulässig – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-constrained-bo-demo.streamlit.app/)**

Viertes Stück der **Bayessche-Optimierung-Linie** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“.
Es setzt [noisy-bo-demo](https://github.com/sebastian-hanisch/noisy-bo-demo) fort: dort war die Auswertung verrauscht, hier ist es die **Grenze**, an der die Empfehlung
gemessen wird. Im Unterschied zu den Fall-Demos zeigt die Demo ein einziges Verfahren an einem wachsenden Beispiel.

## Warum dieses Problem

Am Terminal soll die billigste Einstellung (Plätze am Stapel, Tempo des Stapels) gefunden werden, bei der die mittlere Gesamtzeit eines Lkw eine Grenze einhält. Die
Gerätekosten sind bekannt, die Zeit nur aus verrauschten Läufen. Die Kernfrage: **Hält die Empfehlung die Grenze wirklich ein, oder hat sie nur Glück gehabt?** Die naheliegende
Antwort (erwartete Verbesserung mal Zulässigkeitswahrscheinlichkeit) führt an den Rand des zulässigen Gebiets, und dort ist eine einzelne Beobachtung kein Beleg.

## Befunde aus der Vorarbeit

Die Phase-0-Messreihe der Linie hatte bei einer Grenze von 15 min gemessen: „EI × PoF“ findet 23 % billigere Pläne, 5 von 20 Empfehlungen verletzen die Grenze, Strafe und Zufall
bleiben sicher. Diese Demo prüft das über **drei Grenzen, zwei Lauflängen und zwei Budgets** (12 Einstellungen, 240 Läufe je Verfahren), mit der **wahren** Zeit als Maßstab. Das Bild wird
**schärfer und unbequemer**: das einfache Verfahren verletzt die Grenze in 72 % der Läufe.

## Modell

- **Aufgabe:** Gerätekosten c(x) minimieren (bekannt: 20 EUR/h je Gate-Spur, 120 je Kran, 60 je Stapelgerät, 4 je Stellplatz, Tempo quadratisch) unter T(x) ≤ T_max für die mittlere
  Gesamtzeit T. x = (Plätze am Stapel 2 bis 13, Tempo des Stapels 0,6 bis 1,4): 204 Einstellungen. Terminal wie in `bayes-opt-demo` (33 Lkw je Stunde, Platzgrenze am Stapel).
- **Beobachtung:** die Zeit je Einstellung aus der Tabelle echter Simulationen (`precomputed_slice.json`, aus `bayes-opt-demo`; Zeit = (Gesamtkosten − Gerätekosten) / 22 EUR/h je
  Minute); 30 Wiederholungen je Einstellung, Lauflänge 3 000 oder 15 000 Lkw. **Wahrheit:** Mittel zweier Läufe à 100 000 Lkw.
- **Grenzen:** 11 min (39 % der Einstellungen zulässig), 12 min (56 %), 15 min (79 %). Das wirklich zulässige Optimum liegt nur **0,14 / 0,24 / 0,06 min** unter der Grenze; ein Lauf
  streut um etwa 0,27 min (3 000 Lkw) bzw. 0,12 min (15 000 Lkw).
- **Verfahren:** Zufall; Strafe (Gauß-Prozess auf Kosten + 100 EUR/h je Minute über der Grenze); PoF (Gauß-Prozess auf der Zeit, beschränkte erwartete Verbesserung
  (best_cost − cost)₊ · P(zulässig)); PoF nur sichere Empfehlungen (P ≥ 90 % nach dem Modell); PoF mit Nachprüfung (die letzten 6 Auswertungen messen die 3 billigsten zulässigen
  Kandidaten je zweimal nach). Startdesign 5 Läufe. Empfehlung = billigste beobachtet zulässige (bzw. nach der jeweiligen Regel).
- Formeln im 📐-Expander, Kern in `cbo_algorithm.py`.

## Methodik

1. **Live:** ein Lauf je Verfahren (Zufall, Strafe und das gewählte), mit Karte der Kosten, Zeitgrenze, Modell und Zulässigkeitswahrscheinlichkeit.
2. **Studie** (vorgerechnet, `generate_study.py`): 3 Grenzen × 2 Lauflängen × 2 Budgets (20, 30) × 5 Verfahren × 20 feste Läufe, gepaart (gleiche Seeds). **Maß:** „wirklich
   zulässig“ = wahre Zeit der Empfehlung ≤ Grenze; „Mehrkosten“ = Gerätekosten der Empfehlung minus die des wirklich zulässigen Optimums.
3. **Urteil:** zweiseitiger Vorzeichentest über die Läufe mit Unterschied in der wirklichen Zulässigkeit, ohne Korrektur für mehrere Vergleiche.
4. **ParEGO:** Kosten und Zeit als zwei Ziele; Hypervolumen der nicht dominierten ausgewerteten Einstellungen an ihren wahren Werten.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. Über alle zwölf Einstellungen, je Verfahren 240 Läufe.

| Frage | Befund |
|---|---|
| Wie oft hält die Empfehlung die Grenze wirklich ein? | Zufall **73 %** (176 von 240), Strafe **74 %**, **PoF nur 28 %** (67), PoF nur sichere **74 %**, PoF mit Nachprüfung **77 %** (185). |
| Schlägt eines den Zufall? | **Nein.** Strafe 43 gegen 42 Läufe (p = 1), PoF nur sichere 47 gegen 45 (p = 0,92), PoF mit Nachprüfung 46 gegen 37 (p = 0,38). Das einfache **PoF ist nachweisbar schlechter** (15 gegen 124 Läufe, p < 10⁻²¹). |
| Wie schlecht ist das einfache PoF im Einzelfall? | Grenze 12 min, 3 000 Lkw, 20 Auswertungen: **3 von 20** Empfehlungen wirklich zulässig (Zufall 14, mit Nachprüfung 16); bei 30 Auswertungen **0 von 20**. In 17 von 20 Läufen hielt es die Empfehlung für zulässig. |
| Was kostet Sicherheit? | Mehrkosten der wirklich zulässigen Empfehlungen im Median: Zufall **16 EUR/h**, Strafe 24, PoF 8 (aber nur in 28 % der Läufe zulässig), **PoF nur sichere 69**, PoF mit Nachprüfung **16**. Abstand (sichere Empfehlung) kostet Geld, Nachprüfen kostet Läufe. |
| Warum? | Das Optimum liegt nur 0,06 bis 0,24 min unter der Grenze, ein Lauf streut um 0,12 bis 0,27 min: die billigsten **beobachtet zulässigen** Einstellungen sind überwiegend solche mit zufällig zu kleinem Messwert. |
| Zwei Ziele statt einer Grenze | ParEGO erreicht bei 20 bzw. 30 Auswertungen **91 %** bzw. **95 %** der Obergrenze (2 871) und ist vom Zufall nicht zu unterscheiden (Median 2 604 / 2 724 gegen 2 602 / 2 715). |

## Ehrliche Grenzen

- **Das Ziel ist hier bekannt.** Ist auch das Ziel verrauscht, kommt ein zweites Modell dazu; das ist nicht geprüft.
- **Eine Nebenbedingung, ein Gitter von 204 Einstellungen.** Mehrere Grenzen, mehr Entscheidungen oder stetige Räume sind nicht Teil der Demo.
- **Die Wahrheit hat einen kleinen Restfehler** (Mittel aus 200 000 Lkw); bei Abständen von 0,06 min (Grenze 15) ist „wirklich zulässig“ nur auf dieser Genauigkeit zu lesen.
- **Das einfache PoF ist hier eine bewusst einfache Fassung:** der Maßstab ist die billigste *beobachtet* zulässige Einstellung, nicht die Modell-Sicht. Die Varianten „nur sichere“ und
  „mit Nachprüfung“ sind zwei von vielen möglichen Abhilfen (Quantil-Schranken, Rauschmodell, mehr Läufe an den Kandidaten); ein Urteil über „BO mit Nebenbedingungen“ im Allgemeinen
  folgt nicht daraus.
- **Rund ein Dutzend Einzelzellen ohne Korrektur;** belastbar sind die über alle zwölf Einstellungen gemittelten Befunde, nicht jede einzelne Zelle.
- **Eigene Implementierung** (Kandidatenmenge statt stetiger Optimierung, drei feste Startpunkte der Hyperparameter-Suche).

## Verifikation

- **Kern:** Zulässigkeitswahrscheinlichkeit, beschränkte Verbesserung, Strafe, Empfehlungsregeln, Nachprüfung und Auswahl an Mini-Instanzen von Hand (`tests/test_algorithm.py`).
- **Orakel** (`tests/test_oracle_constrained_bo.py`, anderer Rechenweg):
  - Gauß-Prozess-Vorhersage gegen **scikit-learn**; Zulässigkeitswahrscheinlichkeit gegen **numerische Integration**; beschränkte Verbesserung gegen **Monte-Carlo**.
  - **Hypervolumen** gegen Rasterzählung; Nicht-Dominierte gegen die Definition; zulässiges Optimum gegen **Aufzählung**.
  - Zeit aus Gesamtkosten gegen die Kostenformel; Geräte- und Gesamtkosten von Hand; Vorzeichentest von Hand.
- **Zahlen im README:** `tests/test_claims.py`.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an Minimum und Maximum) und Footer-Test.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App (Skelett nach DEMO-PLAYBOOK §2) |
| `cbo_algorithm.py` | Kern: Gauß-Prozess, Zulässigkeit, beschränkte Verbesserung, Suchverfahren, Empfehlungsregeln |
| `cbo_scenario.py` | Gerätekosten, Zeit aus der Tabelle, verrauschte Auswertung, zulässiges Optimum |
| `cbo_evaluation.py` | Läufe, Kennzahlen, Studie, gepaarte Statistik, ParEGO, Hypervolumen |
| `cbo_constants.py` | Regler-Grenzen, feste Annahmen, Verfahren, Farben, Presets |
| `cbo_presets.py` | Permalink (Begrenzen/Einrasten), Presets, Zufalls-Seed-Button |
| `cbo_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `cbo_pdf_export.py` | PDF-Auswertung (fpdf2) |
| `generate_study.py`, `precomputed_study.json` | Studie (12 Einstellungen, 5 Verfahren, 20 Läufe; ParEGO) |
| `precomputed_slice.json` | Wahrheit und Wiederholungen der Terminal-Scheibe (aus `bayes-opt-demo`) |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit bzw. Laufzeit plus pytest und scikit-learn |
| `tests/` | Einheiten, Orakel, Behauptungen, Presets, AppTest, Footer |

## Bewusst nicht umgesetzt

- **Mehrere Nebenbedingungen und Sicherheitsabstände nach Quantilen** (Chance Constraints), **Rauschmodelle, die vom Ort abhängen**.
- **Beschränkte Verbesserung mit verrauschtem Ziel** (zwei Gauß-Prozesse mit gemeinsamer Auswahl) und **mehrere Ziele mit Hypervolumen-Akquisition** (EHVI).
- **Ranking und Selection** für die Nachprüfung der Kandidaten.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, SciPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
