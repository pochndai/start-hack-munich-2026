#!/usr/bin/env python3
"""Statistics and reports built from stored sessions. Never runs a simulation.

    python bench/report.py                 # rebuild SUMMARY.md, history.csv, summary.png
    python bench/report.py --session FILE  # print one session's table

Everything here is a pure function of the session files, so a report can
always be regenerated and never drifts from the raw data.
"""
from __future__ import annotations

import argparse
import csv
import io
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

CANARY_SLOW = 1.05          # canary this much slower than the session's best -> "slow-machine"
MULTI_CORE = 1.15           # CPU-seconds per wall-second above this -> "multi-core"
COMPRESSED_PRESSURE = 512 * 2**20
FOOTPRINT_PRESSURE = 0.8    # peak footprint above this share of RAM -> "memory"
HISTORY_COLUMNS = 8


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def sample_flags(sample, session, best_canary):
    if sample["status"] != "ok":
        return [sample["status"]]
    flags = []
    before, after = sample.get("memory_before") or {}, sample.get("memory_after") or {}
    swapped = after.get("swapouts", 0) - before.get("swapouts", 0) > 0
    compressed = after.get("compressed_bytes_total", 0) - before.get("compressed_bytes_total", 0)
    ram, peak = session["machine"].get("ram_bytes"), sample.get("peak_footprint_bytes")
    if swapped or compressed > COMPRESSED_PRESSURE or (ram and peak and peak > FOOTPRINT_PRESSURE * ram):
        flags.append("memory")
    canary = (sample.get("canary") or {}).get("seconds")
    if canary and best_canary and canary > best_canary * CANARY_SLOW:
        flags.append("slow-machine")
    if (sample.get("cpu_per_wall") or 0) > MULTI_CORE:
        flags.append("multi-core")
    if sample.get("phases") and sample["phases"][0]["seconds"] < -1e-3:
        flags.append("phase-sum")
    return flags


def _median(values):
    return statistics.median(values) if values else None


def summarize_session(session, noise_floor=None):
    """Per-version statistics plus per-round speed-up ratios for one session."""
    samples = session["samples"]
    canaries = [s["canary"]["seconds"] for s in samples if (s.get("canary") or {}).get("seconds")]
    best_canary = min(canaries) if canaries else None
    labels = [entry["label"] for entry in session["entries"]]
    by_label = {label: {} for label in labels}
    flags = {}
    for sample in samples:
        by_label.setdefault(sample["label"], {})[sample["round"]] = sample
        flags[(sample["label"], sample["round"])] = sample_flags(sample, session, best_canary)

    def ok_samples(label):
        return [s for _, s in sorted(by_label[label].items()) if s["status"] == "ok"]

    def pressured(label):
        good = ok_samples(label)
        return bool(good) and sum("memory" in flags[(label, s["round"])] for s in good) * 2 > len(good)

    # The baseline is measured once per machine and not rerun: its stored run
    # stands in as the reference whenever the session itself did not time it.
    stored = session.get("stored_reference") or None
    stored_pressured = bool(stored) and "memory" in sample_flags(stored["sample"], session, None)

    def ratio(reference, label):
        if not reference or reference == label:
            return None
        if reference in by_label:
            values = [
                by_label[reference][index]["end_to_end"] / by_label[label][index]["end_to_end"]
                for index in sorted(by_label[label])
                if index in by_label[reference] and by_label[reference][index]["status"] == "ok" and by_label[label][index]["status"] == "ok"
            ]
            fastest = min((s["end_to_end"] for s in ok_samples(reference)), default=None)
            from_store, under_pressure = False, pressured(reference)
        elif stored and stored["label"] == reference:
            fastest = stored["sample"]["end_to_end"]
            values = [fastest / s["end_to_end"] for s in ok_samples(label)]
            from_store, under_pressure = True, stored_pressured
        else:
            return None
        if not values:
            return None
        middle = statistics.median(values)
        return {
            "reference": reference, "rounds": len(values), "median": middle, "min": min(values), "max": max(values),
            "per_round": values,
            "ratio_of_mins": fastest / min(s["end_to_end"] for s in ok_samples(label)),
            "within_noise": min(values) <= 1.0 <= max(values) or (noise_floor is not None and abs(middle - 1.0) <= noise_floor),
            "reference_memory_pressured": under_pressure, "stored_reference": from_store,
        }

    reference = session.get("reference_label")
    anchor = (session.get("registry") or {}).get("anchor")
    anchor = anchor if anchor in by_label and anchor != reference else None
    rows = []
    for entry in session["entries"]:
        label = entry["label"]
        good = ok_samples(label)
        totals = [s["end_to_end"] for s in good]
        counts = {}
        for sample in by_label[label].values():
            for flag in flags[(label, sample["round"])]:
                counts[flag] = counts.get(flag, 0) + 1
        peaks = [s.get("peak_footprint_bytes") or s.get("peak_rss_bytes") for s in good]
        peaks = [p for p in peaks if p]
        cores = [s["cpu_per_wall"] for s in good if s.get("cpu_per_wall")]
        parent = entry.get("parent") if entry.get("parent") in by_label else None
        rows.append({
            "label": label, "name": entry["name"], "id": entry["id"],
            "n_ok": len(good), "n_total": len(by_label[label]),
            "median": _median(totals), "min": min(totals) if totals else None, "max": max(totals) if totals else None,
            "spread_pct": (max(totals) - min(totals)) / statistics.median(totals) * 100 if totals else None,
            "groups": {name: _median([s["groups"][name] for s in good]) for name in common.GROUP_NAMES} if good else {},
            "peak_bytes": _median(peaks), "cpu_per_wall": _median(cores),
            "flags": counts, "official_benchmark": entry.get("official_benchmark"),
            "validation": (entry.get("validation") or {}).get("verdict"),
            "ratios": {"reference": ratio(reference, label), "anchor": ratio(anchor, label), "parent": ratio(parent, label)},
        })
    return {
        "reference_label": reference, "anchor_label": anchor, "noise_floor": noise_floor, "best_canary": best_canary, "rows": rows,
        "stored_reference": {"label": stored["label"], "end_to_end": stored["sample"]["end_to_end"],
                             "measured_utc": stored["measured_utc"], "memory_pressured": stored_pressured} if stored else None,
    }


def noise_floor_from(sessions):
    """Largest per-round deviation from 1.00x in the latest complete A/A session."""
    for session in reversed(sessions):
        if session.get("mode") != "aa" or not session.get("complete"):
            continue
        ratios = [row["ratios"]["reference"] for row in summarize_session(session)["rows"] if row["ratios"]["reference"]]
        if ratios:
            return max(abs(value - 1.0) for item in ratios for value in item["per_round"]), session["id"]
    return None, None


# --------------------------------------------------------------------------
# text rendering (ASCII only, so it prints on any console)
# --------------------------------------------------------------------------
def _seconds(value):
    return "-" if value is None else f"{value:.2f}"


def _ratio(item):
    if not item:
        return "-"
    marks = ("~" if item["within_noise"] else "") + ("*" if item["reference_memory_pressured"] else "")
    return f"{item['median']:.2f}x ({item['min']:.2f}-{item['max']:.2f}){marks}"


def _table(header, rows):
    widths = [max(len(str(row[index])) for row in [header, *rows]) for index in range(len(header))]
    line = lambda cells: "| " + " | ".join(str(cell).ljust(width) for cell, width in zip(cells, widths)) + " |"
    return "\n".join([line(header), "| " + " | ".join("-" * width for width in widths) + " |", *map(line, rows)])


def stored_reference_note(summary):
    """One line about the stored baseline run, or '' when the session did not use one."""
    stored = summary.get("stored_reference")
    if not stored:
        return ""
    return (f"Stored baseline: {stored['label']} took {stored['end_to_end']:.2f} s in its single run on {stored['measured_utc'][:10]}"
            + (", under memory pressure" if stored["memory_pressured"] else "")
            + ". It is not rerun: each factor against it is this number divided by a new sample.")


def session_table(summary):
    reference = summary["reference_label"] or "reference"
    stored = summary.get("stored_reference")
    header = ["Version", "n", "median s", "min s", "spread", f"vs {reference}" + (" (stored)" if stored and stored["label"] == reference else "")]
    if summary["anchor_label"]:
        header.append(f"vs {summary['anchor_label']}")
    header += ["vs parent", "Setup", "Compile", "Forward", "Sens.", "Grad.", "peak mem", "cpu/wall", "valid", "flags"]
    rows = []
    for row in summary["rows"]:
        flags = [f"{name} {count}/{row['n_total']}" for name, count in sorted(row["flags"].items())]
        if row["official_benchmark"] is False:
            flags.append("unofficial-benchmark")
        groups = row["groups"]
        cells = [
            row["label"], f"{row['n_ok']}/{row['n_total']}", _seconds(row["median"]), _seconds(row["min"]),
            "-" if row["spread_pct"] is None else f"{row['spread_pct']:.1f}%", _ratio(row["ratios"]["reference"]),
        ]
        if summary["anchor_label"]:
            cells.append(_ratio(row["ratios"]["anchor"]))
        cells += [
            _ratio(row["ratios"]["parent"]),
            *(_seconds(groups.get(name)) for name in ("Setup", "Compilation", "Forward simulation", "Sensitivity", "Gradient")),
            common.format_bytes(row["peak_bytes"]), "-" if row["cpu_per_wall"] is None else f"{row['cpu_per_wall']:.2f}",
            row["validation"] or "-", ", ".join(flags) or "-",
        ]
        rows.append(cells)
    return _table(header, rows)


LEGEND = """- `median s` / `min s`: complete first-run total printed by the version's own benchmark.py (the scored number).
- `vs X`: speed-up over X, median of the per-round ratios, with the lowest and highest round in brackets.
  `~` = not distinguishable from noise. `*` = X ran under memory pressure, so the factor is overstated.
  `(stored)` = X is the baseline, which ran once and is not rerun: every round is divided into that one
  stored total, so the bracket shows only this version's own scatter.
- `spread`: (max - min) / median of the totals. `cpu/wall`: CPU-seconds per wall-second, about 1.00 means one core.
- Flags (count / samples): `memory` = swapping or heavy compression during the run, `slow-machine` = the canary
  ran more than 5 % slower than the session's best, `multi-core` = more than one core was used,
  `failed` / `timeout` = no result, `unofficial-benchmark` = benchmark.py differs from the baseline's."""


# --------------------------------------------------------------------------
# files: SUMMARY.md, history.csv, summary.png
# --------------------------------------------------------------------------
def load_sessions(machine_dir):
    folder = Path(machine_dir) / "sessions"
    return [common.read_json(path) for path in sorted(folder.glob("*.json"))] if folder.is_dir() else []


def _describe_limit(session):
    limits = session.get("limits") or {}
    if limits.get("strategy") == "affinity":
        text = f"pinned to logical CPU {limits.get('bench_cpu')} with the official `--cpus 1`"
    else:
        text = "thread limits " + ", ".join(f"`{key}={value}`" for key, value in sorted((limits.get("env") or {}).items()))
    return text + ("" if session.get("calibrated") else " (NOT calibrated on this machine: run `python bench/run.py calibrate`)")


def _history_table(sessions, noise):
    recent = [s for s in sessions if s.get("complete") and s.get("mode") != "aa"][-HISTORY_COLUMNS:]
    names, cells = [], {}
    for session in recent:
        for row in summarize_session(session, noise)["rows"]:
            if row["label"] not in names:
                names.append(row["label"])
            item = row["ratios"]["reference"]
            cells[(row["label"], session["id"])] = (f"{item['median']:.2f}x" + ("*" if item["reference_memory_pressured"] else "")) if item else ("ref" if row["n_ok"] else "-")
    if not recent:
        return ""
    header = ["Version"] + [f"{s['id'][5:16]} {s['mode']}" for s in recent]
    return _table(header, [[name] + [cells.get((name, s["id"]), "") for s in recent] for name in names])


def build_summary_markdown(sessions):
    complete = [s for s in sessions if s.get("complete")]
    if not complete:
        return "# Benchmark summary\n\nNo complete session yet.\n"
    noise, noise_session = noise_floor_from(sessions)
    latest = complete[-1]
    machine = latest["machine"]
    out = io.StringIO()
    out.write(f"# Benchmark summary: {machine['machine_id']}\n\n")
    ram = f"{machine['ram_bytes'] / common.GIB:.0f} GiB RAM" if machine.get("ram_bytes") else "RAM unknown"
    out.write(f"- Machine: {machine.get('cpu_model')}, {ram}, {machine.get('platform')}\n")
    out.write(f"- CPU limit: {_describe_limit(latest)}\n")
    if noise is None:
        out.write("- Noise floor: not measured yet. Run `python bench/run.py aa`.\n")
    else:
        out.write(f"- Noise floor: +/- {noise * 100:.1f} % (A/A session `{noise_session}`). Smaller differences are not speed-ups.\n")
    out.write(f"- Sessions on record: {len(complete)}\n\n")
    full = next((s for s in reversed(complete) if s["mode"] == "full"), None)
    shown = [("Latest full session", full)] if full else []
    if latest is not full:
        shown.append(("Latest session", latest))
    for title, session in shown:
        config = session["config"]
        out.write(f"## {title}: `{session['id']}`\n\n")
        out.write(f"{config['rounds']} rounds, shuffled order per round, one fresh process per sample.\n\n")
        summary = summarize_session(session, noise)
        out.write(session_table(summary) + "\n\n")
        if stored_reference_note(summary):
            out.write(stored_reference_note(summary) + "\n\n")
    history = _history_table(sessions, noise)
    if history:
        out.write("## Speed-up over the session's reference, across sessions\n\n")
        out.write("One column per session. A version whose factor stays put from column to column is a trustworthy result.\n"
                  "Against the stored baseline every session divides into the same number, so those factors are directly comparable.\n\n")
        out.write(history + "\n\n")
    out.write(f"![First-run total per version, session {session_to_plot(complete)['id']}](summary.png)\n\n")
    out.write("## Reading the tables\n\n" + LEGEND + "\n")
    return out.getvalue()


def session_to_plot(complete):
    """Latest full session, else the latest comparison of different versions, else the latest."""
    for wanted in (lambda s: s["mode"] == "full", lambda s: s["mode"] != "aa", lambda s: True):
        chosen = next((s for s in reversed(complete) if wanted(s)), None)
        if chosen:
            return chosen


def write_history_csv(sessions, path):
    noise, _ = noise_floor_from(sessions)
    fields = ["session", "mode", "started_utc", "label", "id", "n_ok", "n_total", "median_s", "min_s", "max_s",
              "reference", "reference_stored", "speedup", "speedup_min", "speedup_max", "speedup_vs_parent", "peak_bytes", "cpu_per_wall", "validation", "flags"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for session in sessions:
            if not session.get("complete"):
                continue
            for row in summarize_session(session, noise)["rows"]:
                item, parent = row["ratios"]["reference"], row["ratios"]["parent"]
                writer.writerow({
                    "session": session["id"], "mode": session["mode"], "started_utc": session["started_utc"],
                    "label": row["label"], "id": row["id"][:12], "n_ok": row["n_ok"], "n_total": row["n_total"],
                    "median_s": row["median"], "min_s": row["min"], "max_s": row["max"],
                    "reference": item["reference"] if item else "", "reference_stored": item["stored_reference"] if item else "",
                    "speedup": item["median"] if item else "",
                    "speedup_min": item["min"] if item else "", "speedup_max": item["max"] if item else "",
                    "speedup_vs_parent": parent["median"] if parent else "", "peak_bytes": row["peak_bytes"],
                    "cpu_per_wall": row["cpu_per_wall"], "validation": row["validation"] or "",
                    "flags": ";".join(f"{name}={count}" for name, count in sorted(row["flags"].items())),
                })


# Chart colors: the first five categorical slots of the reference palette, in
# fixed order (one per phase group), on its light surface.
SURFACE, INK, SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
GROUP_COLORS = {
    "Setup": "#2a78d6", "Compilation": "#eb6834", "Forward simulation": "#1baf7a",
    "Sensitivity": "#eda100", "Gradient": "#e87ba4", "Other": "#898781",
}
PX = 1 / 96  # one CSS pixel in inches


def render_plot(session, summary, path):
    """Stacked phase bars per version with every single sample as a dot. Returns False without matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch, PathPatch, Rectangle
        from matplotlib.path import Path as MplPath
    except ImportError:
        return False

    family = "DejaVu Sans"
    for candidate in ("Segoe UI", "Helvetica Neue", "Helvetica", "Arial"):
        try:
            font_manager.findfont(candidate, fallback_to_default=False)
            family = candidate
            break
        except ValueError:
            continue

    rows = summary["rows"]
    samples = {row["label"]: [s["end_to_end"] for s in session["samples"] if s["label"] == row["label"] and s["status"] == "ok"] for row in rows}
    longest = max([value for values in samples.values() for value in values] + [1e-9])
    pitch, top, bottom, width = 0.62, 1.38, 0.95, 9.0
    height = top + pitch * len(rows) + bottom
    left, right = 2.1, 0.35
    with plt.rc_context({"font.family": family, "font.size": 10, "text.color": INK}):
        fig = plt.figure(figsize=(width, height), facecolor=SURFACE)
        ax = fig.add_axes([left / width, bottom / height, (width - left - right) / width, pitch * len(rows) / height], facecolor=SURFACE)
        limit = longest * 1.42
        ax.set_xlim(0, limit)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        per_inch_x = limit / (width - left - right)
        per_inch_y = 1 / pitch
        bar = 21 * PX * per_inch_y
        gap, round_x, round_y = 2 * PX * per_inch_x, 4 * PX * per_inch_x, 4 * PX * per_inch_y

        tips = []
        for index, row in enumerate(rows):
            center = index - 0.1
            if row["median"] is None:
                ax.text(longest * 0.01, center, "no successful sample", va="center", color=MUTED, fontsize=9)
                continue
            scale = row["median"] / (sum(row["groups"].values()) or 1.0)
            parts = [(name, row["groups"][name] * scale) for name in common.GROUP_NAMES if row["groups"].get(name, 0) > 0]
            cursor = 0.0
            for position, (name, length) in enumerate(parts):
                low, high = center - bar / 2, center + bar / 2
                if position == len(parts) - 1:  # rounded data-end, square at the baseline
                    rx, ry = min(round_x, length), min(round_y, bar / 2)
                    end = cursor + length
                    patch = PathPatch(MplPath(
                        [(cursor, low), (end - rx, low), (end, low), (end, low + ry), (end, high - ry), (end, high), (end - rx, high), (cursor, high), (cursor, low)],
                        [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO, MplPath.CLOSEPOLY],
                    ), facecolor=GROUP_COLORS[name], edgecolor="none", zorder=3)
                else:
                    patch = Rectangle((cursor, low), length, bar, facecolor=GROUP_COLORS[name], edgecolor="none", zorder=3)
                ax.add_patch(patch)
                cursor += length
                if position < len(parts) - 1:  # surface gap between touching segments
                    ax.add_patch(Rectangle((cursor - gap / 2, low), gap, bar, facecolor=SURFACE, edgecolor="none", zorder=4))
            ax.scatter(samples[row["label"]], [index + 0.24] * len(samples[row["label"]]), s=36, color=SECONDARY, edgecolors=SURFACE, linewidths=1.5, zorder=5)
            tips.append((ax.text(max(row["median"], max(samples[row["label"]])) + 10 * PX * per_inch_x, center, f"{row['median']:.2f} s", va="center", fontsize=10.5, fontweight="semibold"), row, center))

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for text, row, center in tips:
            item = row["ratios"]["reference"]
            if item:
                edge = ax.transData.inverted().transform((text.get_window_extent(renderer).x1, 0))[0]
                ax.text(edge + 8 * PX * per_inch_x, center, f"{item['median']:.2f}x vs {item['reference']}" + ("*" if item["reference_memory_pressured"] else ""), va="center", fontsize=9.5, color=SECONDARY)

        ax.set_yticks(range(len(rows)), [row["label"] for row in rows])
        ax.tick_params(axis="y", length=0, pad=10, labelsize=10.5, labelcolor=INK)
        ax.tick_params(axis="x", length=0, pad=6, labelsize=9, labelcolor=MUTED)
        ax.set_xlabel("Seconds (complete first-run total)", color=MUTED, fontsize=9, labelpad=8)
        ax.xaxis.grid(True, color=GRID, linewidth=0.75, zorder=0)
        ax.set_axisbelow(True)
        for name, spine in ax.spines.items():
            spine.set_visible(name == "left")
            spine.set_color(AXIS)
            spine.set_linewidth(0.75)

        used = [name for name in common.GROUP_NAMES if any(row["groups"].get(name, 0) > 0 for row in rows)]
        handles = [Patch(facecolor=GROUP_COLORS[name], edgecolor="none", label=name) for name in used]
        handles.append(Line2D([], [], marker="o", linestyle="none", markersize=6, markerfacecolor=SECONDARY, markeredgecolor=SURFACE, label="single sample"))
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.35 / width, 1 - 0.86 / height), ncol=len(handles), frameon=False,
                   handlelength=0.9, handleheight=0.9, handletextpad=0.5, columnspacing=1.5, fontsize=9, labelcolor=SECONDARY, borderaxespad=0, borderpad=0)
        machine = session["machine"]
        fig.text(0.35 / width, 1 - 0.36 / height, "First-run total per version", fontsize=15, fontweight="semibold", va="center")
        fig.text(0.35 / width, 1 - 0.65 / height, f"{machine.get('cpu_model')}  ·  session {session['id']}  ·  median of {session['config']['rounds']} rounds, phase split from per-phase medians",
                 fontsize=9, color=SECONDARY, va="center")
        if any((row["ratios"]["reference"] or {}).get("reference_memory_pressured") for row in rows):
            fig.text(0.35 / width, 0.22 / height, f"* {summary['reference_label']} ran under memory pressure on this machine, so factors against it are overstated.", fontsize=8.5, color=SECONDARY, va="center")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        plt.close(fig)
    return True


def write_outputs(machine_dir):
    """Regenerate every derived file for one machine from its stored sessions."""
    machine_dir = Path(machine_dir)
    sessions = load_sessions(machine_dir)
    (machine_dir / "SUMMARY.md").write_text(build_summary_markdown(sessions), encoding="utf-8")
    write_history_csv(sessions, machine_dir / "history.csv")
    complete = [s for s in sessions if s.get("complete")]
    if complete:
        chosen = session_to_plot(complete)
        noise, _ = noise_floor_from(sessions)
        if not render_plot(chosen, summarize_session(chosen, noise), machine_dir / "summary.png"):
            print("matplotlib not available: summary.png not updated")
    return machine_dir / "SUMMARY.md"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--machine", help="machine id under bench/results/ (default: this machine)")
    parser.add_argument("--session", type=Path, help="print the table of one session file and exit")
    args = parser.parse_args()
    if args.session:
        session = common.read_json(args.session)
        noise, _ = noise_floor_from(load_sessions(args.session.parent.parent))
        summary = summarize_session(session, noise)
        print("\n\n".join(filter(None, [session_table(summary), stored_reference_note(summary), "Reading the table\n" + LEGEND])))
        return
    machine_dir = common.RESULTS / (args.machine or common.machine_info()["machine_id"])
    if not (machine_dir / "sessions").is_dir():
        known = sorted(path.name for path in common.RESULTS.glob("*") if path.is_dir()) if common.RESULTS.is_dir() else []
        raise SystemExit(f"No sessions in {machine_dir}. Machines on record: {', '.join(known) or 'none'}")
    print(f"Wrote {write_outputs(machine_dir)}")


if __name__ == "__main__":
    main()
