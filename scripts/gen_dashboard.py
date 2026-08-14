#!/usr/bin/env python3
"""gen_dashboard.py: generate grafana/dashboards/rakuten-overview.json.

WHY THIS EXISTS. The dashboard is 560 lines of JSON describing 13 panels laid
out on Grafana's 24-column grid. Editing that by hand is how you get two panels
sharing a cell, a panel hanging off the right edge, or a datasource uid that no
longer matches the provisioned one, none of which any test or CI job can see,
because a dashboard is data, not code. Generating it from this script means the
layout can be ASSERTED (see validate()), and a broken edit fails here instead of
appearing as an empty panel in a browser three days later.

The generator was written in session 9 and then LOST. It lived only in a
sandbox and never made it into the repo, which is exactly the failure this file
now prevents. It has been reconstructed from the committed JSON and is verified
to reproduce it BYTE FOR BYTE: `python scripts/gen_dashboard.py --check` exits
non-zero if the file on disk and the generated output differ. Wire that into a
review habit, not into CI, until someone decides the JSON is not the source of
truth.

Usage:
  python scripts/gen_dashboard.py            # write the file
  python scripts/gen_dashboard.py --check    # verify the file matches, write nothing
  python scripts/gen_dashboard.py --stdout   # print it

AFTER REGENERATING, restart Grafana. It does not reload a bind-mounted
dashboard (wart 15's family):
  sudo docker compose restart grafana
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The datasource uid is FIXED, not generated. A generated uid would change on
# every provisioning reset and silently orphan all 13 panels; this value is
# pinned in grafana/provisioning/datasources/prometheus.yml and the two must
# agree. validate() checks every panel uses it.
DATASOURCE_UID = "rakuten-prometheus"


def datasource() -> dict:
    """A FRESH dict each call. Sharing one instance across 13 panels makes the
    object graph a web rather than a tree: editing one panel's datasource would
    silently edit every panel's. Serialises identically either way, so this is
    about the script staying honest under later edits."""
    return {"type": "prometheus", "uid": DATASOURCE_UID}

GRID_COLUMNS = 24  # Grafana's fixed grid width; nothing may extend past it.

OUTPUT = Path(__file__).resolve().parents[1] / "grafana/dashboards/rakuten-overview.json"


# --------------------------------------------------------------------------- #
# panel builders
# --------------------------------------------------------------------------- #
def _target(expr: str, ref_id: str, legend: str = "", instant: bool = False,
            fmt: str = "time_series") -> dict:
    """One Prometheus query.

    NOTE `fmt`. A Grafana table or barchart fed by an INSTANT query needs
    format="table"; without it the response is shaped as a time series and the
    panel renders "No data" even though the query is correct. Three panels shipped
    empty this way before it was caught. validate() now enforces the pairing.
    """
    return {
        "datasource": datasource(),
        "editorMode": "code",
        "expr": expr,
        "legendFormat": legend,
        "range": not instant,
        "instant": instant,
        "format": fmt,
        "refId": ref_id,
    }


def row(panel_id: int, title: str, y: int) -> dict:
    return {
        "id": panel_id,
        "type": "row",
        "title": title,
        "collapsed": False,
        "gridPos": {"h": 1, "w": GRID_COLUMNS, "x": 0, "y": y},
        "panels": [],
    }


def timeseries(panel_id: int, title: str, description: str, pos: tuple,
               expr: str, ref_id: str, legend: str, unit: str = "reqps") -> dict:
    x, y, w, h = pos
    return {
        "id": panel_id,
        "type": "timeseries",
        "title": title,
        "description": description,
        "datasource": datasource(),
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "targets": [_target(expr, ref_id, legend)],
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "custom": {"lineWidth": 2, "fillOpacity": 8, "showPoints": "never"},
            },
            "overrides": [],
        },
        "options": {
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
    }


def table(panel_id: int, title: str, description: str, pos: tuple, expr: str,
          ref_id: str, exclude: dict, legend: str = "") -> dict:
    """An instant-query table. `exclude` drops columns Prometheus always returns
    (Time, Value, __name__, instance, job) that would otherwise be noise."""
    x, y, w, h = pos
    return {
        "id": panel_id,
        "type": "table",
        "title": title,
        "description": description,
        "datasource": datasource(),
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "targets": [_target(expr, ref_id, legend, instant=True, fmt="table")],
        "transformations": [{"id": "organize", "options": {"excludeByName": exclude}}],
        "fieldConfig": {"defaults": {"custom": {"align": "auto"}}, "overrides": []},
        "options": {"showHeader": True},
    }


def stat(panel_id: int, title: str, description: str, pos: tuple, expr: str,
         ref_id: str, legend: str) -> dict:
    x, y, w, h = pos
    return {
        "id": panel_id,
        "type": "stat",
        "title": title,
        "description": description,
        "datasource": datasource(),
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "targets": [_target(expr, ref_id, legend, instant=True)],
        "fieldConfig": {
            "defaults": {
                "mappings": [],
                "thresholds": {
                    "mode": "absolute",
                    "steps": [{"color": "red", "value": None},
                              {"color": "green", "value": 1}],
                },
            },
            "overrides": [],
        },
        "options": {
            "colorMode": "background",
            "graphMode": "none",
            "textMode": "auto",
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
        },
    }


# --------------------------------------------------------------------------- #
# the dashboard
# --------------------------------------------------------------------------- #
def build() -> dict:
    panels = [
        row(1, "Service health", 0),
        stat(2, "Scrape targets up",
             "1 = Prometheus reached the service's /metrics. Says nothing about "
             "whether a model is loaded. See the serving table.",
             (0, 1, 6, 4),
             'up{job=~"image-api|text-api|gateway"}', "L", "{{job}}"),
        table(3, "Which model is each service serving?",
              "THE answer to 'which model is in production'. Read from the running "
              "process, not inferred from disk. image stays 'not-loaded' until its "
              "first real /predict, because the service resolves its model lazily.",
              (6, 1, 18, 4),
              "rakuten_model_info", "S",
              # `modality` is dropped: it duplicates `service` on every row
              # except the gateway's (fusion vs gateway), and the question this
              # panel answers is per-SERVICE. What is left is service + source,
              # which is the smallest pair that answers it.
              {"Time": True, "Value": True, "__name__": True,
               "instance": True, "job": True, "modality": True}),

        row(4, "Traffic and latency", 5),
        timeseries(5, "Request rate",
                   "Includes Prometheus's own /metrics scrapes, which is a steady "
                   "baseline of about 1/15s per service.",
                   (0, 6, 8, 8),
                   "sum by (service) (rate(rakuten_http_requests_total"
                   "[$__rate_interval]))", "S", "{{service}}"),
        timeseries(6, "p95 latency",
                   "Buckets run to 120s because the image service's first "
                   "/predict imports torch and may resolve a registry "
                   "version.",
                   (8, 6, 8, 8),
                   "histogram_quantile(0.95, sum by (le, service) "
                   "(rate(rakuten_http_request_duration_seconds_bucket"
                   "[$__rate_interval])))", "N", "{{service}} p95", unit="s"),
        timeseries(7, "5xx rate",
                   "Flat zero is the expected state. Handler exceptions are "
                   "recorded here even though the 500 itself is produced "
                   "above the middleware.",
                   (16, 6, 8, 8),
                   'sum by (service) (rate(rakuten_http_requests_total{status=~"5.."}'
                   "[$__rate_interval]))", "H", "{{service}}"),

        row(8, "Predictions", 14),
        table(9, "Predictions by product type (top 10)",
              "Cumulative since each service STARTED. These counters are "
              "in-process and reset when a container is recreated. A single "
              "class dominating is the signal worth watching: the text "
              "model's weak classes (1180, 10, 1280) are where drift would "
              "show first.",
              (0, 15, 12, 9),
              "topk(10, sum by (prdtypecode) (rakuten_predictions_total))", "G",
              {"Time": True}, legend="{{prdtypecode}}"),
        timeseries(10, "Prediction rate by service",
                   "The text service counts one per PRODUCT, so a batch of 50 "
                   "moves this by 50, not 1.",
                   (12, 15, 12, 9),
                   "sum by (service) (rate(rakuten_predictions_total"
                   "[$__rate_interval]))", "Q", "{{service}}"),

        row(11, "Fusion gateway", 24),
        table(12, "Fusion outcomes",
              "degraded=true means a modality was asked for and could not be "
              "delivered. The measured fusion score (0.7973 weighted F1) "
              "applies ONLY to fused=true rows.",
              (0, 25, 12, 7),
              "sum by (modalities, fused, degraded) (rakuten_fusion_requests_total)",
              "Q", {"Time": True}),
        timeseries(13, "Upstream failures",
                   "Non-zero here with the gateway still returning 200 means "
                   "it is silently serving one modality.",
                   (12, 25, 12, 7),
                   "sum by (upstream) (rate(rakuten_upstream_failures_total"
                   "[$__rate_interval]))", "X", "{{upstream}}"),
    ]
    return {
        "uid": "rakuten-overview",
        "title": "Rakuten services overview",
        "description": "Request rate, latency, predictions and fusion health for "
                       "the image, text and fusion services.",
        "tags": ["rakuten", "mlops"],
        "timezone": "browser",
        "schemaVersion": 39,
        "version": 1,
        "editable": False,
        "refresh": "30s",
        "time": {"from": "now-1h", "to": "now"},
        "graphTooltip": 1,
        "panels": panels,
    }


# --------------------------------------------------------------------------- #
# geometry and wiring validation
# --------------------------------------------------------------------------- #
def validate(dashboard: dict) -> list[str]:
    """Return a list of problems. Empty means the layout is sound.

    Every check here corresponds to a mistake that is INVISIBLE until the
    dashboard is opened in a browser, which is the whole reason for generating
    the JSON instead of writing it.
    """
    problems: list[str] = []
    panels = dashboard["panels"]

    seen: dict[int, str] = {}
    for panel in panels:
        pid = panel["id"]
        if pid in seen:
            problems.append(
                f"duplicate panel id {pid}: {seen[pid]!r} and {panel['title']!r}")
        seen[pid] = panel["title"]

    # Occupied grid cells, so an overlap is detected rather than reasoned about.
    occupied: dict[tuple, str] = {}
    for panel in panels:
        g = panel["gridPos"]
        if g["x"] < 0 or g["w"] < 1 or g["h"] < 1:
            problems.append(f"panel {panel['id']} has a nonsense gridPos {g}")
            continue
        if g["x"] + g["w"] > GRID_COLUMNS:
            problems.append(
                f"panel {panel['id']} {panel['title']!r} runs past the "
                f"{GRID_COLUMNS}-column grid: x={g['x']} + w={g['w']}")
        for x in range(g["x"], g["x"] + g["w"]):
            for y in range(g["y"], g["y"] + g["h"]):
                if (x, y) in occupied:
                    problems.append(
                        f"panel {panel['id']} {panel['title']!r} overlaps "
                        f"{occupied[(x, y)]!r} at cell ({x},{y})")
                    return problems  # one overlap report is enough to act on
                occupied[(x, y)] = panel["title"]

    for panel in panels:
        if panel["type"] == "row":
            if panel["gridPos"]["w"] != GRID_COLUMNS:
                problems.append(f"row {panel['id']} does not span the full width")
            continue

        if panel.get("datasource", {}).get("uid") != DATASOURCE_UID:
            problems.append(
                f"panel {panel['id']} points at a datasource other than "
                f"{DATASOURCE_UID!r}")
        if not panel.get("targets"):
            problems.append(f"panel {panel['id']} {panel['title']!r} has no query")
        for target in panel.get("targets", []):
            if target["datasource"].get("uid") != DATASOURCE_UID:
                problems.append(
                    f"panel {panel['id']} target {target['refId']} points at the "
                    f"wrong datasource")
            if target["instant"] == target["range"]:
                problems.append(
                    f"panel {panel['id']} target {target['refId']}: instant and "
                    f"range must be opposites, got both {target['instant']}")
            # The bug that shipped three empty panels in session 9.
            if panel["type"] in ("table", "barchart") and target["instant"] \
                    and target["format"] != "table":
                problems.append(
                    f"panel {panel['id']} {panel['title']!r} is a {panel['type']} "
                    f"fed by an instant query but format is {target['format']!r}, "
                    f"not 'table', so it will render 'No data'")
            if not target["expr"].strip():
                problems.append(f"panel {panel['id']} has an empty expression")

    return problems


def render(dashboard: dict) -> str:
    # Matches the committed file exactly: 2-space indent, key order as built,
    # non-ASCII left alone (the title contains an em dash), trailing newline.
    return json.dumps(dashboard, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate the Grafana dashboard JSON.")
    ap.add_argument("--check", action="store_true",
                    help="Verify the committed file matches; write nothing.")
    ap.add_argument("--stdout", action="store_true", help="Print instead of writing.")
    args = ap.parse_args()

    dashboard = build()
    problems = validate(dashboard)
    if problems:
        print("❌ Dashboard failed validation:")
        for problem in problems:
            print(f"   - {problem}")
        return 1
    print(f"✅ {len(dashboard['panels'])} panels, geometry valid.")

    rendered = render(dashboard)
    if args.stdout:
        print(rendered, end="")
        return 0
    if args.check:
        if not OUTPUT.exists():
            print(f"❌ {OUTPUT} does not exist.")
            return 1
        if OUTPUT.read_text() == rendered:
            print(f"✅ {OUTPUT.name} matches the generator byte for byte.")
            return 0
        print(f"❌ {OUTPUT.name} DIFFERS from the generator. Either it was edited by "
              f"hand (regenerate, or fold the edit into this script) or the script "
              f"changed (run it without --check).")
        return 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(rendered)
    print(f"💾 Wrote {OUTPUT}")
    print("ℹ️  Grafana does not reload a bind-mounted dashboard: "
          "sudo docker compose restart grafana")
    return 0


if __name__ == "__main__":
    sys.exit(main())
