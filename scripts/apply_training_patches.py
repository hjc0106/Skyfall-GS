#!/usr/bin/env python3
"""Check or apply the source patches required by WideFE training.

First initialize the pinned submodules with:
  git submodule update --init --recursive
Then inspect or apply the two tracked patches:
  python scripts/apply_training_patches.py
  python scripts/apply_training_patches.py --apply

No model weights are downloaded. Already-applied patches are left untouched;
conflicting changes are never reset or overwritten. Licenses remain intact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PATCHES = (("FlowEdit", "flowedit-local.patch"), ("MoGe", "moge-sparse-depth.patch"))


def patch_state(root: Path, name: str, filename: str) -> tuple[str, str]:
    checkout, patch = root / "submodules" / name, root / "patches" / filename
    if not (checkout / ".git").exists() or not patch.is_file():
        raise FileNotFoundError(f"Initialize the pinned {name} submodule and restore {patch}")
    command = ["git", "-C", str(checkout), "apply"]
    reverse = subprocess.run(command + ["--reverse", "--check", str(patch)], capture_output=True, text=True)
    if reverse.returncode == 0:
        return "already_applied", ""
    forward = subprocess.run(command + ["--check", str(patch)], capture_output=True, text=True)
    if forward.returncode == 0:
        return "pending", ""
    return "conflict", forward.stderr.strip()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Apply pending source patches after checking all of them")
    args = parser.parse_args(argv)
    try:
        records = []
        for name, filename in PATCHES:
            state, error = patch_state(ROOT, name, filename)
            records.append({"submodule": name, "patch": filename, "status": state, "error": error})
        if any(item["status"] == "conflict" for item in records):
            print(json.dumps(records, indent=2))
            return 1
        if args.apply:
            for item in records:
                if item["status"] == "pending":
                    subprocess.run(["git", "-C", str(ROOT / "submodules" / item["submodule"]),
                                    "apply", str(ROOT / "patches" / item["patch"])], check=True)
                    item["status"] = "applied"
        print(json.dumps(records, indent=2))
        return 0
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
