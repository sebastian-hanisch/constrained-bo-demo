"""PDF-Export der aktuellen Auswertung (fpdf2, Kernschriften; Umlaute sind dort erlaubt, Gedankenstrich und Eurozeichen nicht)."""

from fpdf import FPDF

import cbo_constants as C


def _num(x, nd=0):
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


def build_pdf(a, method):
    """Einseitiger Bericht: Einstellungen und Ergebnis je Verfahren (Empfehlung, wirklich zulässig?, Mehrkosten). Gibt die Bytes der PDF-Datei zurück."""
    s = a.settings
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Nebenbedingungen: günstig und zulässig", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "Sebastian Hanisch, Operations Research und Machine Learning", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Einstellungen", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for k, v in (("Zeitgrenze", f"{_num(s.tmax, 0)} min mittlere Gesamtzeit"), ("Länge eines Laufs", f"{_num(s.trucks)} Lkw"), ("Auswertungen", str(s.n)),
                 ("Gewähltes Verfahren", C.METHOD_LABELS[method].replace(chr(8805), chr(62) + chr(61))), ("Zufalls-Seed", str(s.seed))):
        pdf.cell(60, 6, k)
        pdf.cell(0, 6, v, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Ergebnis je Verfahren", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Wirklich zulässiges Optimum: {_num(a.optimum[2])} EUR/h", new_x="LMARGIN", new_y="NEXT")
    for key, run in a.runs.items():
        status = "wirklich zulässig" if run.truly_feasible else "VERLETZT die Grenze"
        pdf.cell(0, 6, f"{C.METHOD_LABELS[key].replace(chr(8805), chr(62) + chr(61))}: {_num(run.cost)} EUR/h, {status}, Mehrkosten {_num(run.excess)} EUR/h", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, "Ein einzelner Lauf zeigt nur, was bei diesem Seed geschah; die Studie der Demo mittelt über 20 feste Läufe.")
    return bytes(pdf.output())
