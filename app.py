"""Nebenbedingungen: günstig und trotzdem zulässig - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Viertes Stück der Linie Bayessche Optimierung. Das Ziel (Gerätekosten) ist bekannt, die Grenze auf der mittleren Gesamtzeit nur durch verrauschte Simulationen. Die
billigste Einstellung liegt dann genau dort, wo die Grenze nicht mehr zu prüfen ist.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import cbo_constants as C
import cbo_evaluation as ev
from cbo_evaluation import Settings, analyse
from cbo_pdf_export import build_pdf
from cbo_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from cbo_visualization import build_excess_chart, build_feasibility_chart, build_map_chart

st.set_page_config(page_title="Nebenbedingungen – Sebastian Hanisch", layout="wide")


def de(x, nd=0):
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


def pct(x, nd=0):
    return f"{100 * x:.{nd}f} %".replace(".", ",")


@st.cache_data(show_spinner=False)
def _analysis(settings, method):
    return analyse(settings, method)


st.title("🚧 Nebenbedingungen: günstig und trotzdem zulässig")
st.markdown(
    """
Am Terminal soll die **billigste Einstellung** gefunden werden, bei der die **mittlere Gesamtzeit eines Lkw** eine Grenze nicht überschreitet. Die Kosten der Geräte und Stellplätze
sind bekannt; die Gesamtzeit kennt man nur aus **verrauschten Simulationsläufen**. Die naheliegende Antwort der Bayesschen Optimierung: ein Ersatzmodell für die Zeit und eine Suche
nach **erwarteter Verbesserung mal Zulässigkeitswahrscheinlichkeit**. Sie führt, wenn die Grenze bindet, an den Rand des zulässigen Gebiets, und dort ist eine **einzelne
verrauschte Beobachtung kein Beleg** für Zulässigkeit. Die Demo misst, wie oft die Empfehlung die Grenze **wirklich** einhält, und was Strafe, Sicherheitsabstand und Nachprüfung daran
ändern. Sie ist das vierte Stück der Linie **Bayessche Optimierung**.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("Die fünf Verfahren", expanded=True):
    st.markdown(
        """
| Verfahren | Wie es sucht | Wie es empfiehlt |
|---|---|---|
| **Zufall** | zufällige Einstellungen, ohne Modell | billigste Einstellung, die **beobachtet** zulässig war |
| **Strafe** | Ein Gauß-Prozess auf Kosten + 100 EUR/h je Minute über der Grenze; erwartete Verbesserung | wie Zufall |
| **PoF** | Gauß-Prozess auf der Zeit; erwartete Verbesserung der Kosten mal **Zulässigkeitswahrscheinlichkeit** | wie Zufall |
| **PoF, nur sichere Empfehlungen** | wie PoF | nur Einstellungen mit **Modell**-Zulässigkeitswahrscheinlichkeit ≥ 90 % |
| **PoF mit Nachprüfung** | wie PoF, aber die letzten 6 Auswertungen prüfen die 3 billigsten zulässigen Kandidaten je zweimal nach | billigste, deren **Mittel** zulässig ist |

**Maß:** „wirklich zulässig“ heißt: die **wahre** mittlere Gesamtzeit (Mittel aus 2 × 100 000 Lkw) der empfohlenen Einstellung liegt unter der Grenze. „Mehrkosten“ sind die
Gerätekosten der Empfehlung minus die des wirklich zulässigen Optimums.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Die Aufgabe**")
    tmax = st.select_slider("Zeitgrenze (mittlere Gesamtzeit)", C.TMAX_CHOICES, key="tmax_select", format_func=lambda t: C.TMAX_LABEL[t],
                            help="Je schärfer die Grenze, desto weniger Einstellungen sind zulässig und desto näher liegt das Optimum an ihr.")
    trucks = st.select_slider("Länge eines Laufs", C.TRUCK_CHOICES, key="trucks_select", format_func=lambda t: C.TRUCKS_LABEL[t],
                              help="Kurze Läufe sind billig, aber streuen stark; lange sind ruhiger. Die Tabelle enthält zu beiden je 30 echte Wiederholungen je Einstellung.")
    n = st.slider("Auswertungen (Simulationsläufe)", *bounds("n_slider"), key="n_slider", step=1, help="Das Gesamtbudget, auch bei der Nachprüfung.")
    st.markdown("**Das Verfahren**")
    method = st.radio("Verfahren (live)", C.METHODS[1:], format_func=lambda m: C.METHOD_LABELS[m], key="method_radio")
    seed = st.number_input("Zufalls-Seed des Laufs", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Lauf generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "tmax_select": float(tmax), "trucks_select": int(trucks), "method_radio": method, "seed_input": int(seed)})

settings = Settings(n=int(n), tmax=float(tmax), trucks=int(trucks), seed=int(seed))
a = _analysis(settings, method)
opt_i, opt_j, opt_cost = a.optimum
margins = ev.margin_summary()[float(tmax)]

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
st.caption(
    f"Das **wirklich zulässige Optimum**: {C.PLACES[opt_i]} Plätze, Tempo {str(C.SPEEDS[opt_j]).replace('.', ',')}, **{de(opt_cost)} EUR/h** bei einer wahren mittleren Gesamtzeit von "
    f"{de(margins['t_opt'], 2)} min. Es liegt **{de(margins['margin'], 2)} min** unter der Grenze; ein Lauf dieser Länge streut um etwa "
    f"{'0,3' if settings.trucks == 3000 else '0,1'} min."
)
shown = [m for m in C.METHODS if m in a.runs]
cols = st.columns(len(shown))
for col, m in zip(cols, shown):
    run = a.runs[m]
    verdict = "✅ wirklich zulässig" if run.truly_feasible else "❌ verletzt die Grenze"
    col.metric(C.METHOD_LABELS[m], f"{de(run.cost)} EUR/h", delta=f"{run.excess:+.0f} EUR/h gegen Optimum".replace(".", ","), delta_color="off", help=verdict)
    col.caption(f"{verdict}; das Verfahren hielt sie für {'zulässig' if run.claimed_feasible else 'unzulässig (kein zulässiger Punkt beobachtet)'}.")

st.markdown("**Was das Verfahren sieht** – Schritt für Schritt nachvollziehen")
run = a.runs[method]
c1, c2 = st.columns([2, 1])
with c1:
    step = st.slider("Nach wie vielen Auswertungen?", C.N_START, C.N_MAX, C.N_START + 5, key="map_step", help="Bis zur gewählten Zahl der Auswertungen; bei mehr als dem Budget gilt das Budget.")
step = int(min(step, len(run.idx)))
st.plotly_chart(build_map_chart(a, run, step, settings.tmax), width="stretch", key=f"map_{method}_{step}")
st.caption(f"Nach {step} Auswertungen des Verfahrens „{C.METHOD_LABELS[method]}“. Links die bekannten Kosten mit der wahren Grenze (orange Linie: links der Linie ist die Einstellung zu "
           "langsam, rechts zulässig), in der Mitte, was das Modell von der Zeit glaubt, rechts die Zulässigkeitswahrscheinlichkeit.")
st.download_button("📄 Auswertung als PDF", data=build_pdf(a, method), file_name="nebenbedingungen_auswertung.pdf", mime="application/pdf", key="pdf_button")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Hält die Empfehlung die Grenze wirklich ein?")
agg = ev.aggregate()
st.info(
    f"In diesem Lauf: {sum(r.truly_feasible for r in a.runs.values())} von {len(a.runs)} Verfahren empfehlen wirklich Zulässiges. **Über alle zwölf Einstellungen der Studie** (je 20 Läufe,"
    f" 240 je Verfahren) hält die Empfehlung die Grenze wirklich ein bei: Zufall {pct(agg['random']['feasible'] / agg['random']['n'])}, Strafe "
    f"{pct(agg['penalty']['feasible'] / agg['penalty']['n'])}, PoF {pct(agg['pof']['feasible'] / agg['pof']['n'])}, PoF nur sichere "
    f"{pct(agg['pof_safe']['feasible'] / agg['pof_safe']['n'])}, PoF mit Nachprüfung {pct(agg['pof_check']['feasible'] / agg['pof_check']['n'])}. "
    "**Das einfache PoF ist das schlechteste:** es sucht den Rand und empfiehlt, was dort zufällig zulässig aussah."
)
st.caption("Ein einzelner Lauf zeigt nur, was bei diesem Seed geschah; die Experimente mitteln über 20 feste Läufe je Einstellung.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

budget_key = min(C.STUDY_BUDGETS, key=lambda b: abs(b - settings.n))
summary = ev.method_summary(settings.tmax, settings.trucks, budget_key)

st.subheader("🔬 Experiment 1: Wie oft hält die Empfehlung die Grenze wirklich ein?")
st.caption(f"Zeitgrenze und Lauflänge wie in der Seitenleiste, Budget {budget_key} Auswertungen (das nächstliegende der Studie); 20 feste Läufe je Verfahren.")
st.plotly_chart(build_feasibility_chart(summary), width="stretch", key=f"feas_chart_{budget_key}")
rows = "\n".join(
    f"| {C.METHOD_LABELS[m]} | {summary[m]['feasible']} von 20 | {summary[m]['violated_but_claimed']} | "
    + (f"{de(summary[m]['median_excess_feasible'])}" if summary[m]["feasible"] else "–") + " |"
    for m in C.METHODS
)
st.markdown("| Verfahren | wirklich zulässig | zulässig behauptet, aber verletzt | Mehrkosten der zulässigen, Median (EUR/h) |\n|---|---|---|---|\n" + rows)
worst = min(C.METHODS, key=lambda m: summary[m]["feasible"])
st.info(f"Am seltensten wirklich zulässig: **{C.METHOD_LABELS[worst]}** ({summary[worst]['feasible']} von 20). Je mehr Läufe ein Verfahren in die Nähe der Grenze steckt, desto öfter "
        "fällt die Empfehlung auf ein Rauschereignis herein.")

st.subheader("🔬 Experiment 2: Was kostet Sicherheit?")
st.plotly_chart(build_excess_chart(summary), width="stretch", key=f"excess_chart_{budget_key}")
st.info(
    f"Über alle zwölf Einstellungen: Mehrkosten der wirklich zulässigen Empfehlungen im Median **{de(agg['random']['median_excess'])}** (Zufall), "
    f"**{de(agg['penalty']['median_excess'])}** (Strafe), **{de(agg['pof']['median_excess'])}** (PoF), **{de(agg['pof_safe']['median_excess'])}** (nur sichere), "
    f"**{de(agg['pof_check']['median_excess'])}** EUR/h (Nachprüfung). Sicherheit durch **Abstand** (nur sichere Empfehlungen) kostet Geld; Sicherheit durch **Nachprüfen** "
    "kostet Läufe."
)

st.subheader("🔬 Experiment 3: Schlägt eines der Verfahren den Zufall?")
st.caption("Gepaarter Vergleich der **wirklichen** Zulässigkeit über alle zwölf Einstellungen (240 Paare je Verfahren; Vorzeichentest über die Paare mit Unterschied).")
tab_rows = []
for k in ("penalty", "pof", "pof_safe", "pof_check"):
    s = ev.paired_aggregate(k, "random")
    tab_rows.append(f"| {C.METHOD_LABELS[k]} | {s['better']} | {s['worse']} | {de(s['p'], 3)} |")
st.markdown("| Verfahren gegen Zufall | Läufe, in denen es zulässig ist und der Zufall nicht | umgekehrt | p |\n|---|---|---|---|\n" + "\n".join(tab_rows))
st.info("**Keines der Verfahren ist bei der Zulässigkeit nachweisbar besser als zufälliges Ausprobieren**, das einfache PoF ist nachweisbar schlechter. Das Tal der Kosten ist flach: "
        "was der Zufall findet, ist selten viel teurer als das Optimum, und das **billigste** zulässige liegt direkt an der Grenze.")

st.subheader("🔬 Experiment 4: Zwei Ziele statt einer Grenze")
tr = settings.trucks
ps = ev.pareto_summary(tr, budget_key)
ref = ev.study()["pareto_reference_hv"]
st.caption(f"Kosten und Zeit als **zwei Ziele**, ParEGO (zufällige Gewichte, Tschebyscheff, erwartete Verbesserung) gegen Zufall; Hypervolumen der nicht dominierten ausgewerteten "
           f"Einstellungen an ihren wahren Werten, Referenzpunkt (800 EUR/h, 20 min); Obergrenze der 204 Einstellungen: {de(ref)}.")
st.markdown(
    "| Verfahren | Hypervolumen, Median | kleinster Lauf | größter Lauf | Punkte der Front, Median |\n|---|---|---|---|---|\n"
    + "\n".join(f"| {name} | {de(ps[k]['median'])} | {de(ps[k]['min'])} | {de(ps[k]['max'])} | {de(ps[k]['front'])} |" for k, name in (("random", "Zufall"), ("parego", "ParEGO")))
)
st.info(f"Bei {budget_key} Auswertungen liegt ParEGO im Median bei {pct(ps['parego']['median'] / ref)} der Obergrenze, der Zufall bei {pct(ps['random']['median'] / ref)}: "
        "**kein nachweisbarer Vorsprung**. Mit zwei Zielen entfällt die Frage nach der Grenze, aber nicht die nach dem Rauschen: die Front wird aus verrauschten Werten gebaut.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Ziel ist bekannt** | Hier sind die Gerätekosten eine Formel. Ist auch das Ziel verrauscht, muss die beschränkte erwartete Verbesserung beide Modelle mitführen. | Zwei Gauß-Prozesse, gemeinsame Auswahl |
| **Eine Nebenbedingung** | Mehrere Grenzen verlangen eine Zulässigkeitswahrscheinlichkeit je Grenze und ein Produkt mit der Annahme der Unabhängigkeit. | Mehrere Nebenbedingungen |
| **Das Rauschen ist gleichmäßig** | Das Modell lernt ein Rauschen für alle Einstellungen; nahe der Grenze (Last) streuen die Läufe stärker. | Rauschmodelle, die vom Ort abhängen |
| **Die Wahrheit ist bekannt** | „Wirklich zulässig“ misst gegen ein Mittel aus 200 000 Lkw; auch das hat einen kleinen Fehler. Das Optimum liegt nur 0,06 bis 0,24 min unter der Grenze. | Längere Referenzläufe |
| **Das Budget ist knapp** | Mit viel Budget je Einstellung (viele Nachmessungen) wird die Grenze prüfbar; das ist dann aber keine Suche mehr. | Ranking und Selection |
| **20 Läufe je Einstellung** | Einzelne Einstellungen der Studie haben wenige Unterschiede; belastbar ist der Befund über alle zwölf Einstellungen. | Mehr Läufe |
    """
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Aufgabe.** Gesucht ist $x^\star = \arg\min_x c(x)$ unter $T(x) \le T_{\max}$ mit bekannten Kosten $c$ und unbekannter mittlerer Gesamtzeit $T$, beobachtet als $t = T(x) + \varepsilon$.

**Ersatzmodell.** Gauß-Prozess mit Matérn-5/2-Kern auf den beobachteten Zeiten: $T(x) \mid \text{Daten} \sim \mathcal N(\mu(x), s^2(x))$.

**Zulässigkeitswahrscheinlichkeit.**

$$\mathrm{PoF}(x) = P\bigl(T(x) \le T_{\max}\bigr) = \Phi\!\left(\frac{T_{\max} - \mu(x)}{s(x)}\right)$$

**Beschränkte erwartete Verbesserung** bei bekanntem Ziel, $c^{+}$ = Kosten der billigsten beobachtet zulässigen Einstellung:

$$\mathrm{CEI}(x) = \bigl(c^{+} - c(x)\bigr)_{+}\cdot \mathrm{PoF}(x)$$

**Strafverfahren.** Der Gauß-Prozess modelliert $y = c(x) + \lambda\,(t - T_{\max})_{+}$ mit $\lambda = 100$ EUR/h je Minute; gesucht wird nach erwarteter Verbesserung von $y$.

**Sichere Empfehlung.** Nur Einstellungen mit $\mathrm{PoF}(x) \ge 0{,}9$ werden empfohlen.

**Nachprüfung.** Die drei billigsten beobachtet zulässigen Einstellungen werden je zweimal nachgemessen; empfohlen wird die billigste, deren Mittel aus drei Beobachtungen $\le T_{\max}$ ist.

**Warum der Rand täuscht.** Liegt das Optimum $T(x^\star) = T_{\max} - \delta$ nahe der Grenze, und streut eine Beobachtung mit $\sigma$, dann ist die Wahrscheinlichkeit, **beobachtet zulässig**
zu sein, selbst für Einstellungen mit $T(x) > T_{\max}$ bis zu $\Phi\bigl((T_{\max} - T(x))/\sigma\bigr)$: bei $\sigma = 0{,}27$ min und $T(x) = T_{\max} + 0{,}2$ min rund 23 %.

**Literatur.** M. A. Gelbart, J. Snoek, R. P. Adams (2014), *Bayesian Optimization with Unknown Constraints*, UAI (Zulässigkeitswahrscheinlichkeit); J. R. Gardner, M. J. Kusner,
Z. Xu, K. Q. Weinberger, J. P. Cunningham (2014), *Bayesian Optimization with Inequality Constraints*, ICML; B. Letham, B. Karrer, G. Ottoni, E. Bakshy (2019), *Constrained
Bayesian Optimization with Noisy Experiments*, Bayesian Analysis 14(2), 495–519.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
