#!/usr/bin/env python3
"""Build docs/no-klujax-scipy-results.pdf from the stored result files.

    python docs/make_no_klujax_report.py

Needs matplotlib and reportlab (not a dependency of the simulator or of the harness).
Every timing in the tables and charts is read from bench/results/intel-xeon-platinum-8468-197g-linux/;
the few numbers typed in below (the development steps and the experiments) say where they come from.
"""
import json
import statistics
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results" / "intel-xeon-platinum-8468-197g-linux"
OUTPUT = ROOT / "docs" / "no-klujax-scipy-results.pdf"
SESSION = RESULTS / "sessions" / "2026-11-22T00-14-25Z_run.json"
EARLIER_SESSION = RESULTS / "sessions" / "2026-11-22T00-08-43Z_run.json"
VARIATIONS_BRANCH = RESULTS / "variations" / "2026-11-22T00-23-50Z.json"
VARIATIONS_MAIN = RESULTS / "variations-v10" / "2026-11-22T00-24-58Z.json"
VARIATION_WITH_ORIGINAL = RESULTS / "variations" / "2026-11-22T00-18-31Z.json"
BRANCH, MAIN, OLD = "no-klujax-scipy", "v10-host-swept-values", "v07-numpy-host"

# Ink and series colours: the reference palette of the chart guide, used unchanged and in its fixed order.
INK, SECONDARY, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
LABEL_ON = ["#ffffff", "#ffffff", "#ffffff", "#0b0b0b", "#ffffff"]
BEFORE, AFTER = "#86b6ef", "#2a78d6"

GROUPS = [
    ("Imports", ("imports",)),
    ("Build", ("initialize", "load", "build")),
    ("Trace and compile", ("forward_lower", "forward_compile", "sensitivity_lower", "sensitivity_compile", "gradient_lower", "gradient_compile")),
    ("Forward run", ("forward", "sensitivity", "host_transfer")),
    ("Gradient run", ("gradient",)),
]
PHASE_ROWS = [
    ("Imports", ("imports",)),
    ("Build the setup", ("initialize", "load", "build")),
    ("Forward: trace and compile", ("forward_lower", "forward_compile")),
    ("Forward: run", ("forward",)),
    ("Readout: trace and compile (benchmark's own function)", ("sensitivity_lower", "sensitivity_compile")),
    ("Gradient: trace and compile", ("gradient_lower", "gradient_compile")),
    ("Gradient: run", ("gradient",)),
    ("Readout run and transfer of results", ("sensitivity", "host_transfer")),
]


# ---------------------------------------------------------------------------- data
def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def session_phases(session, label):
    """Median of every phase, in ms, over the samples of one version."""
    samples = [s for s in session["samples"] if s["label"] == label and s["status"] == "ok"]
    keys = [p["key"] for p in samples[0]["phases"]]
    return {key: statistics.median(next(p["seconds"] for p in s["phases"] if p["key"] == key) for s in samples) * 1e3 for key in keys}


def session_row(session, label):
    return next(row for row in session["summary"]["rows"] if row["label"] == label)


def report_phases(paths):
    reports = [load(p) for p in paths]
    keys = [p["key"] for p in reports[0]["phases"]]
    phases = {key: statistics.median(next(p["seconds"] for p in r["phases"] if p["key"] == key) for r in reports) * 1e3 for key in keys}
    totals = [r["timing_seconds"]["end_to_end"] for r in reports]
    return phases, totals, reports


def grouped(phases, groups):
    return [sum(phases[key] for key in keys) for _, keys in groups]


session = load(SESSION)
earlier = load(EARLIER_SESSION)
phases = {label: session_phases(session, label) for label in (OLD, MAIN, BRANCH)}
rows = {label: session_row(session, label) for label in (OLD, MAIN, BRANCH)}
baseline_seconds = load(RESULTS / "baseline.json")["sample"]["end_to_end"]
aligo_main, aligo_main_totals, _ = report_phases(sorted((RESULTS / "aligo").glob("mainlatest-*.json")))
aligo_branch, aligo_branch_totals, aligo_branch_reports = report_phases(sorted((RESULTS / "aligo").glob("branch-*.json")))
aligo_original = load(RESULTS / "aligo" / "original-1.json")
cases_branch = {case["case"]: case for case in load(VARIATIONS_BRANCH)["cases"]}
cases_main = {case["case"]: case for case in load(VARIATIONS_MAIN)["cases"]}
varied_original = load(VARIATION_WITH_ORIGINAL)["cases"][0]["original_here"]["first_run_total"]
varied_runs = {"main": [], "branch": []}
for path in sorted((RESULTS / "variations").glob("*.json")) + sorted((RESULTS / "variations-v10").glob("*.json")):
    run = load(path)
    for case in run["cases"]:
        if case["case"] == "random-layout-1" and (case.get("ours") or {}).get("first_run_total"):
            if run["git_head"].startswith("f3f1454") and not run["dirty"]:
                varied_runs["branch"].append(case["ours"]["first_run_total"])
            elif run["git_head"].startswith("bf8d047"):
                varied_runs["main"].append(case["ours"]["first_run_total"])


# ---------------------------------------------------------------------------- charts
def use_font():
    for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSansMono.ttf"):
        pdfmetrics.registerFont(TTFont(name[:-4], font_manager.findfont(name[:-4].replace("DejaVuSans-Bold", "DejaVu Sans:bold").replace("DejaVuSansMono", "DejaVu Sans Mono").replace("DejaVuSans", "DejaVu Sans"))))
    pdfmetrics.registerFontFamily("DejaVuSans", normal="DejaVuSans", bold="DejaVuSans-Bold", italic="DejaVuSans", boldItalic="DejaVuSans-Bold")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": AXIS, "text.color": INK, "axes.labelcolor": SECONDARY,
                         "xtick.color": MUTED, "ytick.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def chart_phases(path):
    """Three complete first runs on one axis, each split into the same five parts."""
    labels = [("v07 (changes 1 to 12)", OLD), ("v10 (latest main)", MAIN), ("Branch", BRANCH)]
    figure, axes = plt.subplots(figsize=(6.9, 2.15))
    for position, (name, label) in enumerate(labels):
        left = 0.0
        for index, value in enumerate(grouped(phases[label], GROUPS)):
            axes.barh(position, value, left=left, height=0.46, color=SERIES[index], edgecolor=SURFACE, linewidth=1.6)
            if value >= 52:
                axes.text(left + value / 2, position, f"{value:.0f}", ha="center", va="center", color=LABEL_ON[index], fontsize=7.5)
            left += value
        axes.text(left + 10, position, f"{rows[label]['median'] * 1e3:.0f} ms", ha="left", va="center", color=INK, fontsize=8.5, fontweight="bold")
    axes.set_yticks(range(len(labels)), [name for name, _ in labels])
    axes.invert_yaxis()
    axes.set_xlim(0, 1000)
    axes.set_xticks(range(0, 1001, 200))
    axes.set_xlabel("milliseconds, median of five runs")
    axes.xaxis.grid(True, color=GRID, linewidth=0.6)
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES[index]) for index in range(len(GROUPS))]
    axes.legend(handles, [name for name, _ in GROUPS], loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=len(GROUPS), frameon=False,
                handlelength=0.9, handleheight=0.9, columnspacing=1.4, fontsize=8, labelcolor=SECONDARY)
    figure.tight_layout()
    figure.savefig(path, dpi=230)
    plt.close(figure)


SCENARIO_ORDER = ["supplied", "values-shifted", "values-random", "frequencies-51", "frequencies-201", "band-20-5000", "boundary-swap", "center-swap",
                  "detector-moved", "homodyne-readout", "frequency-modulation", "random-layout-1", "random-layout-2", "random-layout-3",
                  "uifo-20-5000", "size-2", "size-4", "size-4-full"]


def chart_scenarios(path):
    """Before and after for every scenario on the NumPy path: one line per scenario, two dots."""
    figure, axes = plt.subplots(figsize=(6.9, 4.1))
    for position, name in enumerate(SCENARIO_ORDER):
        before, after = cases_main[name]["ours"]["first_run_total"] * 1e3, cases_branch[name]["ours"]["first_run_total"] * 1e3
        axes.plot([after, before], [position, position], color=AXIS, linewidth=2, solid_capstyle="round", zorder=1)
        axes.scatter([before], [position], s=46, color=BEFORE, edgecolor=SURFACE, linewidth=1.4, zorder=3)
        axes.scatter([after], [position], s=46, color=AFTER, edgecolor=SURFACE, linewidth=1.4, zorder=3)
        axes.text(after - 14, position, f"{after:.0f}", ha="right", va="center", color=SECONDARY, fontsize=7.5)
    axes.set_yticks(range(len(SCENARIO_ORDER)), SCENARIO_ORDER)
    axes.invert_yaxis()
    axes.set_xlim(500, 1100)
    axes.set_xlabel("complete first run, milliseconds (one run each)")
    axes.xaxis.grid(True, color=GRID, linewidth=0.6)
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.tick_params(axis="y", length=0)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=color) for color in (BEFORE, AFTER)]
    axes.legend(handles, ["v10 (latest main)", "Branch"], loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, frameon=False, fontsize=8, labelcolor=SECONDARY)
    figure.tight_layout()
    figure.savefig(path, dpi=230)
    plt.close(figure)


# ---------------------------------------------------------------------------- document
use_font()
BODY = ParagraphStyle("body", fontName="DejaVuSans", fontSize=9.2, leading=13.2, textColor=colors.HexColor(INK), alignment=TA_LEFT, spaceAfter=5)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8, leading=11, textColor=colors.HexColor(SECONDARY), spaceAfter=3)
TITLE = ParagraphStyle("title", parent=BODY, fontName="DejaVuSans-Bold", fontSize=19, leading=23, spaceAfter=3)
SUBTITLE = ParagraphStyle("subtitle", parent=BODY, fontSize=9.5, leading=13, textColor=colors.HexColor(SECONDARY), spaceAfter=9)
H1 = ParagraphStyle("h1", parent=BODY, fontName="DejaVuSans-Bold", fontSize=12.5, leading=16, spaceBefore=10, spaceAfter=5)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.2, leading=10.6, spaceAfter=0)
CELL_BOLD = ParagraphStyle("cellbold", parent=CELL, fontName="DejaVuSans-Bold")
HEAD = ParagraphStyle("head", parent=CELL, fontName="DejaVuSans-Bold", textColor=colors.HexColor(SECONDARY))
BULLET = ParagraphStyle("bullet", parent=BODY, leftIndent=11, bulletIndent=1, spaceAfter=2.5)
CODE = ParagraphStyle("code", parent=BODY, fontName="DejaVuSansMono", fontSize=7.6, leading=10.4, leftIndent=6, spaceAfter=1.5)
TILE_VALUE = ParagraphStyle("tilevalue", parent=BODY, fontName="DejaVuSans-Bold", fontSize=15, leading=18, spaceAfter=0)
TILE_LABEL = ParagraphStyle("tilelabel", parent=SMALL, spaceAfter=1)


def P(text, style=BODY):
    return Paragraph(text, style)


def bullets(items):
    return [Paragraph(item, BULLET, bulletText="•") for item in items]


def table(header, body, widths, numeric=(), bold_last_column=False, repeat=1):
    data = [[P(cell, HEAD) for cell in header]]
    for line in body:
        data.append([P(str(cell), CELL_BOLD if (bold_last_column and index == len(line) - 1) else CELL) for index, cell in enumerate(line)])
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f4f3f0")),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor(GRID)),
        ("LINEABOVE", (0, 0), (-1, 0), 0.5, colors.HexColor(GRID)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    result = Table(data, colWidths=[w * mm for w in widths], repeatRows=repeat, hAlign="LEFT")
    for column in numeric:
        for line in data:
            line[column].style = ParagraphStyle("right", parent=line[column].style, alignment=2)
    result.setStyle(TableStyle(style))
    return result


def ms(seconds):
    return f"{seconds * 1e3:.0f} ms"


def signed(value, unit=""):
    """A rounded number with an explicit sign, typeset with a real minus."""
    text = f"{value:+.0f}"
    if text in ("+0", "-0"):
        return "0" + unit
    return text.replace("-", "−") + unit


def change(before, after):
    return signed((after / before - 1) * 100, " %")


def footer(canvas, document):
    canvas.saveState()
    canvas.setFont("DejaVuSans", 7.5)
    canvas.setFillColor(colors.HexColor(MUTED))
    canvas.drawString(18 * mm, 10 * mm, "Differometor without klujax: results of branch no-klujax-scipy on the judges' CPU")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {document.page}")
    canvas.restoreState()


def build():
    temporary = Path(tempfile.mkdtemp(prefix="report-"))
    chart_phases(temporary / "phases.png")
    chart_scenarios(temporary / "scenarios.png")
    width = A4[0] - 36 * mm
    old, main, branch = (rows[label]["median"] for label in (OLD, MAIN, BRANCH))
    aligo_before, aligo_after = statistics.median(aligo_main_totals), statistics.median(aligo_branch_totals)
    varied_before, varied_after = statistics.median(varied_runs["main"]), statistics.median(varied_runs["branch"])
    story = []

    story += [P("Differometor without klujax", TITLE),
              P("Results of the experiment branch <b>no-klujax-scipy</b> (commit f3f1454) on the judges' CPU. "
                "Intel Xeon Platinum 8468, one pinned logical CPU, official <font face='DejaVuSansMono'>--cpus 1</font>, 32 GiB memory cap. "
                "Measured on 22 November 2026.", SUBTITLE),
              P("After this report the branch became <font face='DejaVuSansMono'>main</font>, as version <b>v11-scipy-stacked-solver</b> "
                "(changes 16 to 18 in CHANGES.md). Wherever this report says \"main\" or \"latest main\", it means the code before that step, "
                "version v10, which is kept on the branch <font face='DejaVuSansMono'>legacy-klujax</font>.", SMALL)]

    story += [P("In short", H1),
              P(f"The branch replaces the klujax solver by SciPy's sparse LU and moves the solves out of the JAX graph. "
                f"On the supplied setup the complete first run, which is the scored number, goes from <b>{main:.2f} s</b> for the latest "
                f"<font face='DejaVuSansMono'>main</font> (v10) to <b>{branch:.2f} s</b>, in the same five-round session. "
                f"The two other setups that were timed improve by the same amount. All correctness checks pass, and in one scenario the "
                f"branch agrees better with the original than <font face='DejaVuSansMono'>main</font> does.")]
    tiles = [[P("Supplied setup (official benchmark)", TILE_LABEL), P("aLIGO (official second benchmark)", TILE_LABEL), P("A varied grid (random-layout-1)", TILE_LABEL)],
             [P(f"{main:.2f} s to {branch:.2f} s", TILE_VALUE), P(f"{aligo_before:.2f} s to {aligo_after:.2f} s", TILE_VALUE), P(f"{varied_before:.2f} s to {varied_after:.2f} s", TILE_VALUE)],
             [P(f"{change(main, branch)} against latest main; the original takes {baseline_seconds:.1f} s", TILE_LABEL),
              P(f"{change(aligo_before, aligo_after)}; the original takes {aligo_original['timing_seconds']['end_to_end']:.1f} s", TILE_LABEL),
              P(f"{change(varied_before, varied_after)}; the original takes {varied_original:.1f} s", TILE_LABEL)]]
    tile_table = Table(tiles, colWidths=[width / 3] * 3, hAlign="LEFT")
    tile_table.setStyle(TableStyle([("BOX", (0, 0), (0, -1), 0.6, colors.HexColor(GRID)), ("BOX", (1, 0), (1, -1), 0.6, colors.HexColor(GRID)),
                                    ("BOX", (2, 0), (2, -1), 0.6, colors.HexColor(GRID)), ("LEFTPADDING", (0, 0), (-1, -1), 7),
                                    ("TOPPADDING", (0, 0), (-1, 0), 6), ("BOTTOMPADDING", (0, -1), (-1, -1), 6), ("TOPPADDING", (0, 1), (-1, -1), 1),
                                    ("BOTTOMPADDING", (0, 0), (-1, -2), 1)]))
    story += [tile_table, Spacer(1, 6)]

    story += [P("Timings of three versions on three setups", H1),
              P("\"Complete first run\" is what <font face='DejaVuSansMono'>benchmark.py</font> prints and what is scored: imports, building the setup, all compilation, "
                "forward simulation, readout, gradient and transfer of the results, in a fresh process.")]
    story += [table(
        ["Setup", "How it was timed", "Original", "v07", "v10, latest main", "Branch", "Branch against v10"],
        [["Supplied grid, 101 frequencies, 193 design variables", "Harness session, 5 shuffled rounds, medians", f"{baseline_seconds:.2f} s (one stored run)", f"{old:.2f} s", f"{main:.2f} s", f"{branch:.2f} s", change(main, branch)],
         ["aLIGO, 101 frequencies, 29 design variables", f"benchmark_aligo.py, {len(aligo_main_totals)} alternating runs each, medians", f"{aligo_original['timing_seconds']['end_to_end']:.2f} s (one run)", "not timed", f"{aligo_before:.2f} s", f"{aligo_after:.2f} s", change(aligo_before, aligo_after)],
         ["random-layout-1: another 3x3 arrangement, balanced homodyne readout", f"tests/variations.py, {len(varied_runs['main'])} and {len(varied_runs['branch'])} runs, medians", f"{varied_original:.2f} s (one run)", "not timed", f"{varied_before:.2f} s", f"{varied_after:.2f} s", change(varied_before, varied_after)]],
        [40, 38, 25, 13, 23, 15, 20], numeric=(2, 3, 4, 5, 6), bold_last_column=True)]
    spread = lambda label: f"{rows[label]['min']:.2f} to {rows[label]['max']:.2f} s"
    story += [Spacer(1, 4),
              P(f"Range of the five runs in the session: v07 {spread(OLD)}, v10 {spread(MAIN)}, branch {spread(BRANCH)}. The machine was not perfectly quiet "
                f"(the harness flagged a slower control workload in 2 to 3 of 5 rounds for every version), which is why versions are compared within one session only. "
                f"An earlier session the same night, before the merge with main, gave v07 {session_row(earlier, OLD)['median']:.2f} s and the branch "
                f"{session_row(earlier, BRANCH)['median']:.2f} s.", SMALL)]

    story += [KeepTogether([P("Where the time goes on the supplied setup", H1),
                            Image(str(temporary / "phases.png"), width=width, height=width * 2.15 / 6.9)]),
              P("The same session, phase by phase (medians in milliseconds):", SMALL)]
    body = []
    for name, keys in PHASE_ROWS:
        values = [sum(phases[label][key] for key in keys) for label in (OLD, MAIN, BRANCH)]
        body.append([name] + [f"{value:.0f}" for value in values] + [signed(values[2] - values[1])])
    totals = [sum(phases[label].values()) for label in (OLD, MAIN, BRANCH)]
    body.append(["Sum of the phase medians"] + [f"{value:.0f}" for value in totals] + [signed(totals[2] - totals[1])])
    story += [table(["Phase", "v07", "v10 (latest main)", "Branch", "Branch minus v10"], body, [84, 18, 28, 18, 26], numeric=(1, 2, 3, 4), bold_last_column=True),
              Spacer(1, 4),
              P("Three things explain the difference. Compiling the forward and gradient functions takes a third of the time, because the solver is no longer part "
                "of the JAX program. The gradient run is twice as fast, because it no longer factorises anything. The forward run is slower, because it now does "
                "the one factorisation, three solves and their refinement. Importing is 22 ms cheaper without klujax.", BODY)]

    story += [P("What the branch changes", H1)]
    story += bullets([
        "<b>SciPy's sparse LU instead of klujax.</b> SciPy is already in the organizers' environment, so the main path needs no package beyond theirs. "
        "klujax is still imported by the JAX fallback, which setups with linking functions use.",
        "<b>One matrix for all frequencies.</b> The signal matrices are stacked into one block-diagonal matrix and factorised in a single call. "
        "The column order is found once per sparsity pattern, and SuperLU's supernode settings are turned down, which made the factorisation 2.25 times faster in tests.",
        "<b>One factorisation per evaluation instead of four.</b> The gradient needs none of its own. The adjoint solve of the noise is linear in a factor that is "
        "only known later, so it is done in the forward pass. The adjoint of the signal is a combination of the noise weights whenever the signal is read at the "
        "noise detector; this is checked at run time, and otherwise the matrices are factorised again.",
        "<b>One refinement step after every solve.</b> The matrices have entries from 1e-22 to 1e13. A plain SuperLU solve was up to 5e-04 off in the sensitivity "
        "in tests, depending on the column order; with one refinement step it was closer to the exact solution than klujax in every scenario tested.",
        "<b>Nothing of the solver is compiled.</b> Forward and gradient are one host callback each.",
        "<b>The dense signal and noise matrices no longer travel between JAX and the host.</b> The host path reads a few thousand of their entries, which the "
        "sparse layouts now carry. The arrays kept for the reverse pass shrink from 26.7 MB to 5.2 MB.",
        "<b>Cheaper imports of our own package.</b> It took about 33 ms to import; the import phase is now 22 ms shorter. SuperLU's compiled module is loaded "
        "from its file without importing SciPy's sparse package, the jaxlib version is read from jaxlib, and plotly is loaded when a plot is first asked for. "
        "<font face='DejaVuSansMono'>import jax</font> itself, about 280 ms, is done by benchmark.py before our code runs.",
    ])
    story += [P("The branch is merged with main at v10, so it contains changes 14 and 15 in main's version.", SMALL)]

    story += [P("Correctness", H1)]
    validation = load(RESULTS / "validation" / "no-klujax-scipy@f3f14542ea40.json")
    arrays = validation["arrays"]
    verdicts = [case["verdict"] for case in cases_branch.values()]
    main_verdicts = [case["verdict"] for case in cases_main.values()]
    story += [table(
        ["Check", "Result on the VM"],
        [["Official benchmark.py, supplied setup", f"PASS in every run; distance from the public reference {validation['official']['max_log_sensitivity_error']:.3e} in log sensitivity (limit 1e-5), the same as the original"],
         ["Official benchmark_aligo.py", f"PASS in every run; distance from the reference {aligo_branch_reports[0]['validation']['max_absolute_log_sensitivity_error']:.1e} (limit 1e-6)"],
         ["bench/validate.py: every output and all 193 gradient entries against the original's arrays from this VM",
          f"{validation['verdict']}. Signal {arrays['signal']['scale_rel']:.1e}, noise {arrays['noise']['scale_rel']:.1e}, sensitivity {arrays['sensitivity']['max_log_abs']:.1e} in log, "
          f"gradient {arrays['gradient']['scale_rel']:.1e} of its largest entry, {arrays['gradient']['entries_outside']} of {arrays['gradient']['entries']} entries outside the tolerance"],
         ["tests/test_host.py: the host path against the JAX path on 13 cases, plus jit, vmap and finite differences", "ALL CHECKS PASSED"],
         ["tests/handwritten_jacobians.py: hand-derived derivatives of the 19 formulas", "ALL RULES AGREE WITH JAX"],
         ["tests/variations.py: 19 other setups against the original, with finite differences",
          f"{verdicts.count('PASS')} PASS, {verdicts.count('DEVIATION')} DEVIATION, {verdicts.count('FAIL')} FAIL "
          f"(latest main in the same hour: {main_verdicts.count('PASS')} PASS, {main_verdicts.count('DEVIATION')} DEVIATION)"]],
        [78, 96])]
    story += [Spacer(1, 4),
              P("The one DEVIATION is the scenario band-20-5000: two small gradient entries differ from the original's by more than 0.1 %. It is the deviation already "
                "recorded for main, where a rounding-free reference showed the original to be the less accurate side. The second DEVIATION of main, a signal difference of "
                "1.6e-06 on the second public setup, does not occur on the branch, where that setup agrees with the original to 5e-11 in log sensitivity.", BODY)]

    story += [KeepTogether([P("Other setups", H1),
                            P("Every scenario of tests/variations.py, one complete first run each for the latest main and for the branch, on the VM.", BODY),
                            Image(str(temporary / "scenarios.png"), width=width, height=width * 4.1 / 6.9)])]
    body = []
    for name in SCENARIO_ORDER + ["amplitude-modulation"]:
        before, after = cases_main[name]["ours"]["first_run_total"], cases_branch[name]["ours"]["first_run_total"]
        metrics = cases_branch[name]["metrics"]
        body.append([name, cases_branch[name]["ours"]["path"], ms(before), ms(after), change(before, after),
                     f"{metrics['sensitivity']['max_log_abs']:.1e}", f"{metrics['gradient']['entries_outside']} of {metrics['gradient']['entries']}", cases_branch[name]["verdict"]])
    story += [P("The same scenarios as a table", H1),
              table(["Scenario", "Path", "v10", "Branch", "Change", "Log sensitivity against the original", "Gradient entries outside", "Verdict"], body,
                    [39, 21, 18, 18, 17, 22, 18, 21], numeric=(2, 3, 4, 5, 6)),
              Spacer(1, 4),
              P("amplitude-modulation has linking functions and takes the JAX fallback on both versions. The branch does not make that path faster "
                "(one run each; the difference is within what single runs vary by). Scenario descriptions are in tests/variations/README.md.", SMALL)]

    story += [P("How the result was reached", H1),
              P("Quick comparisons while developing: the official benchmark, three or four alternating runs per version, medians. They are not harness sessions, "
                "and the machine's speed drifted during the night, so only the two columns of one row are comparable.", BODY),
              table(["State of the branch", "v07 in the same minutes", "Branch"],
                    [["First version: SciPy solver, one factorisation, callbacks only, SuperLU loaded directly (b0b902a)", "835 ms", "725 ms"],
                     ["Dense matrices kept out of the callbacks, imports trimmed", "812 ms", "658 ms"],
                     ["Build step fixed, matrix product rewritten (05da4b9)", "834 ms", "667 ms"],
                     ["Merged with main at v10 (f3f1454); v10 itself: 879 ms", "913 ms", "720 ms"]],
                    [108, 36, 30], numeric=(1, 2), bold_last_column=True)]

    story += [P("Experiments, and what was decided", H1)]
    story += [table(
        ["Tried", "Measured", "Decision"],
        [["Switching the refinement step off", "Forward run 109 ms to 84 ms, complete first run 658 ms to 645 ms (VM, four alternating runs). Agreement with the original on the second public setup falls from 5e-11 to 2e-08", "Kept on: the margin is worth more than 2 %"],
         ["Tuning SuperLU: direct call, fixed column order, supernode settings of one", "Factorising 101 matrices: 86.7 ms to 38.5 ms (development laptop)", "Adopted"],
         ["Stacking all frequencies into one matrix", "Solve time halved, factorisation unchanged (development laptop)", "Adopted"],
         ["Scaling rows and columns before factorising", "No help without refinement, sometimes worse (up to 1.5e-04 in an adjoint solve)", "Not adopted"],
         ["Krylov methods (GMRES, BiCGSTAB), also preconditioned with the LU of the neighbouring frequency", "No convergence in 2,000 iterations (development laptop)", "Not adopted"],
         ["Banded LAPACK solver after reordering", "Three times slower than SuperLU and 9e-06 away from the klujax result (development laptop)", "Not adopted"],
         ["Loading SciPy's compiled sparse-product module directly, like SuperLU", "The module took 49 ms to load on the laptop, more than the products it would speed up", "Not adopted"],
         ["Importing jax before benchmark.py starts its timer (a start-up hook)", "Not tried", "Rejected: it would move work out of the timed region instead of removing it"]],
        [58, 82, 34])]
    story += [Spacer(1, 3), P("Numbers marked \"development laptop\" are from short runs on Windows with a newer JAX; they guided decisions and are not recorded as results.", SMALL)]

    story += [P("Limits and open points", H1)]
    story += bullets([
        "<b>The branch is now main.</b> It was an experiment on a branch when these measurements were made. It has since become main as "
        "v11-scipy-stacked-solver; the code before it is on the branch legacy-klujax.",
        "<b>A private SciPy entry point.</b> The factorisation calls SciPy's internal SuperLU function and loads its module from a file. Both exist in the pinned "
        "scipy 1.18.1; if either fails, the code falls back to the normal import and then to the public function, which is slower.",
        "<b>klujax is still a requirement</b> because of the JAX fallback. Removing it completely needs the fallback to change as well.",
        "<b>A stall that main has too.</b> On one pinned CPU, starting the gradient before the previous JAX call has finished hangs inside the host callback. "
        "benchmark.py waits after every phase, so it is not affected. v07 hangs in the same test.",
        "<b>The harness marks v10 and the branch as \"unofficial-benchmark\".</b> Their benchmark.py is the organizers' newer file (with aLIGO), "
        "which differs from the one at the baseline commit. Nobody on the team edited it.",
        "<b>Not yet done:</b> a noise-floor session on the VM in its present state, and timings of v07 on the two extra setups.",
    ])

    story += [P("Reproduce", H1)]
    for line in ("git checkout v11-scipy-stacked-solver",
                 "# the scored number, and the second official setup",
                 "python benchmark.py --cpus 1 --output benchmark-candidate.json",
                 "python benchmark_aligo.py --cpus 1 --output benchmark-aligo.json",
                 "# three versions in shuffled rounds",
                 "python bench/run.py run --rounds 5 \\",
                 "    --versions v07-numpy-host,v10-host-swept-values,v11-scipy-stacked-solver",
                 "# all outputs and gradients against the original, then the other setups",
                 "python bench/validate.py v11-scipy-stacked-solver --force",
                 "python tests/test_host.py",
                 "python tests/variations.py"):
        story.append(P(line.replace(" ", "&nbsp;"), CODE))
    story += [Spacer(1, 5), P(f"Raw data: bench/results/intel-xeon-platinum-8468-197g-linux/ (sessions/{SESSION.name}, aligo/, variations/, variations-v10/, "
                              f"validation/, no-klujax-scipy-logs/). This document is built from those files by docs/make_no_klujax_report.py.", SMALL)]

    document = SimpleDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=17 * mm,
                                 title="Differometor without klujax: results on the judges' CPU", author="Team repository pochndai/start-hack-munich-2026")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size // 1024} KiB)")
    print(f"supplied: v07 {old:.3f}, v10 {main:.3f}, branch {branch:.3f} | aLIGO: {aligo_before:.3f} to {aligo_after:.3f} (original {aligo_original['timing_seconds']['end_to_end']:.2f}) "
          f"| varied: {varied_before:.3f} to {varied_after:.3f} (original {varied_original:.2f}); runs {len(varied_runs['main'])}/{len(varied_runs['branch'])}")


if __name__ == "__main__":
    build()
