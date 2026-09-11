#!/usr/bin/env python3
"""Run the locked 3-ROI 2x->4x panel. First ROI must archive cleanly before the others."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from lod.archive import HISTORICAL_INDEX_LOCK
from lod.lineage import write_json
from lod.panel import (
    PANEL_DIR,
    PANEL_ROIS,
    PANEL_STATUS,
    archive_gate,
    selection_payload,
    summarize_panel,
)


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _historical_unchanged(index_path: Path) -> dict:
    experiments = list((_load(index_path).get("experiments") or []) if index_path.is_file() else [])
    by_name = {item.get("name"): item for item in experiments}
    problems = []
    for name, fields in HISTORICAL_INDEX_LOCK.items():
        got = by_name.get(name)
        if got is None:
            problems.append(f"missing:{name}")
            continue
        for key, value in fields.items():
            if got.get(key) != value:
                problems.append(f"{name}.{key}={got.get(key)!r} expected {value!r}")
    return {"ok": not problems, "problems": problems}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel_dir", type=str, default=str(PANEL_DIR))
    parser.add_argument("--only", type=str, default="", help="Comma ids: building,trees,parking")
    parser.add_argument("--skip_gate", action="store_true")
    parser.add_argument("--stages", type=str, default="all")
    args = parser.parse_args()

    panel_dir = Path(args.panel_dir)
    if not panel_dir.is_absolute():
        panel_dir = (ROOT / panel_dir).resolve()
    selection_path = panel_dir / "SELECTION.json"
    if not selection_path.is_file():
        write_json(
            selection_path,
            selection_payload(locked_at=datetime.now(timezone.utc).isoformat()),
        )
    selection = _load(selection_path)
    if not selection.get("locked_before_training"):
        raise SystemExit("SELECTION.json must be locked before training.")

    by_id = {item["id"]: item for item in (selection.get("rois") or PANEL_ROIS)}
    order = list(selection.get("run_order") or [item["id"] for item in PANEL_ROIS])
    requested = [item.strip() for item in args.only.split(",") if item.strip()] or order
    for roi_id in requested:
        if roi_id not in by_id:
            raise SystemExit(f"unknown ROI {roi_id}")

    py = sys.executable
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    first_id = order[0]
    index_path = ROOT / "skyfall-gs_exp" / "lod_index.json"

    for roi_id in requested:
        if roi_id != first_id and not args.skip_gate:
            gate = archive_gate(panel_dir / first_id)
            hist = _historical_unchanged(index_path)
            if not gate["ok"] or not hist["ok"]:
                write_json(panel_dir / "GATE.json", {"first": gate, "historical": hist})
                raise SystemExit(
                    f"Refusing to start {roi_id}: first-ROI archive gate "
                    f"missing={gate.get('missing')} historical={hist.get('problems')}"
                )
        item = by_id[roi_id]
        out = panel_dir / roi_id
        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            py, "scripts/run_lod_scale_chain.py",
            "--preset", roi_id,
            "--output_dir", str(out),
            "--archive_dir", str(out),
            "--stages", args.stages,
            "--status", PANEL_STATUS,
            "--alignment", "geometry",
        ]
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)
        if roi_id == first_id:
            gate = archive_gate(out)
            hist = _historical_unchanged(index_path)
            write_json(panel_dir / "GATE.json", {"first": gate, "historical": hist})
            if not gate["ok"] or not hist["ok"]:
                raise SystemExit(
                    f"First ROI finished but archive gate failed: "
                    f"missing={gate.get('missing')} historical={hist.get('problems')}"
                )

    write_json(panel_dir / "PANEL.json", summarize_panel(panel_dir, selection=selection))
    print(json.dumps({"panel_dir": str(panel_dir), "ran": requested}, indent=2))


if __name__ == "__main__":
    main()
