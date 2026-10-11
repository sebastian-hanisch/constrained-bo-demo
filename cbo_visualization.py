"""Plotly-Abbildungen (alle Achsen fest, damit Touch-Scrollen auf der Seite nicht am Diagramm hängen bleibt)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import cbo_algorithm as A
import cbo_constants as C
import cbo_scenario as S


def _fixed(fig, **layout):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    return fig


def build_map_chart(a, run, step, tmax):
    """Drei Flächen über Stapeltempo (x) und Plätze (y): Gerätekosten mit der wahren Grenze, was das Modell von der Zeit glaubt (Mittelwert), und die Zulässigkeitswahrscheinlichkeit.
    Punkte: bisher ausgewertet (grün: beobachtet zulässig, rot: beobachtet unzulässig); Stern: wahres zulässiges Optimum."""
    cost, t_true = a.cost, a.t_true
    shape = cost.shape
    pts = S.grid_points(shape)
    cand = S.to_unit(pts, shape)
    idx = run.idx[:step]
    t = run.t_obs[:step]
    gp = A.GaussianProcess().fit(cand[idx], t)
    mu, sd = gp.predict(cand)
    pof = A.probability_of_feasibility(mu, sd, tmax).reshape(shape)
    xs, ys = list(C.SPEEDS), list(C.PLACES)
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Kosten, Grenze", "Zeit (Mittel des Modells)", "P(zulässig)"), horizontal_spacing=0.06)
    fig.add_trace(go.Heatmap(z=cost, x=xs, y=ys, colorscale="YlGnBu_r", showscale=False, hovertemplate="Tempo %{x}<br>Plätze %{y}<br>Kosten %{z:.0f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Contour(z=t_true, x=xs, y=ys, contours=dict(start=tmax, end=tmax, size=1, coloring="none"), line=dict(color=C.COLOR_NEXT, width=3), showscale=False,
                             hoverinfo="skip", name="wahre Grenze"), row=1, col=1)
    fig.add_trace(go.Heatmap(z=np.clip(mu.reshape(shape), 8, 20), x=xs, y=ys, colorscale="RdYlBu_r", zmin=8, zmax=20, showscale=False,
                             hovertemplate="Tempo %{x}<br>Plätze %{y}<br>Zeit %{z:.1f} min<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Heatmap(z=pof, x=xs, y=ys, colorscale="Greens", zmin=0, zmax=1, showscale=False, hovertemplate="Tempo %{x}<br>Plätze %{y}<br>P %{z:.2f}<extra></extra>"), row=1, col=3)
    feas = [t[k] <= tmax for k in range(len(idx))]
    for col in (1, 2, 3):
        for flag, color, name in ((True, C.COLOR_FEASIBLE, "beobachtet zulässig"), (False, C.COLOR_INFEASIBLE, "beobachtet unzulässig")):
            sel = [k for k in range(len(idx)) if feas[k] == flag]
            fig.add_trace(go.Scatter(x=[xs[int(pts[idx[k]][1])] for k in sel], y=[ys[int(pts[idx[k]][0])] for k in sel], mode="markers",
                                     marker=dict(size=9, color=color, line=dict(color="white", width=1)), name=name, showlegend=(col == 1)), row=1, col=col)
    oi, oj, _ = a.optimum
    fig.add_trace(go.Scatter(x=[xs[oj]], y=[ys[oi]], mode="markers", marker=dict(size=15, symbol="star", color=C.COLOR_NEXT, line=dict(color="white", width=1)),
                             name="zulässiges Optimum"), row=1, col=1)
    fig.update_xaxes(title_text="Tempo Stapel")
    fig.update_yaxes(title_text="Plätze am Stapel", col=1)
    fig.update_annotations(font_size=12)
    return _fixed(fig, height=380)


def build_feasibility_chart(summary, n_runs=20):
    """Anteil der Läufe mit wirklich zulässiger Empfehlung je Verfahren (Balken) und Anteil, in dem das Verfahren Zulässigkeit behauptet, die Wahrheit aber verletzt (Raute)."""
    keys = [m for m in C.METHODS if m in summary]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[C.METHOD_LABELS[m] for m in keys], y=[summary[m]["feasible"] / n_runs for m in keys], marker_color=[C.METHOD_COLORS[m] for m in keys],
                         name="Empfehlung wirklich zulässig", hovertemplate="%{x}<br>%{y:.0%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[C.METHOD_LABELS[m] for m in keys], y=[summary[m]["violated_but_claimed"] / n_runs for m in keys], mode="markers",
                             marker=dict(symbol="diamond", size=11, color=C.COLOR_INFEASIBLE), name="zulässig behauptet, aber verletzt"))
    fig.update_yaxes(title_text="Anteil der 20 Läufe", range=[0, 1.05], tickformat=".0%")
    return _fixed(fig, height=340)


def build_excess_chart(summary):
    """Median der Mehrkosten gegenüber dem wirklich zulässigen Optimum, nur über die Läufe mit wirklich zulässiger Empfehlung."""
    keys = [m for m in C.METHODS if m in summary and summary[m]["feasible"] > 0]
    fig = go.Figure(go.Bar(x=[C.METHOD_LABELS[m] for m in keys], y=[summary[m]["median_excess_feasible"] for m in keys], marker_color=[C.METHOD_COLORS[m] for m in keys],
                           hovertemplate="%{x}<br>%{y:.0f} EUR/h<extra></extra>"))
    fig.update_yaxes(title_text="Mehrkosten gegen zulässiges Optimum (EUR/h, Median der zulässigen)")
    return _fixed(fig, height=320)
