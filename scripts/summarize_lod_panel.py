#!/usr/bin/env python3
"""Summarize the 3-ROI panel. Execution success is not visual quality."""

from __future__ import annotations

import argparse
from pathlib import Path

from lod.lineage import write_json
from lod.panel import PANEL_DIR, summarize_panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel_dir", type=str, default=str(PANEL_DIR))
    args = parser.parse_args()
    root = Path(args.panel_dir)
    payload = summarize_panel(root)
    out = root / "PANEL.json"
    write_json(out, payload)
    print(out)


if __name__ == "__main__":
    main()
