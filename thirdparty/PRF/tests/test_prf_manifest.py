from __future__ import annotations

from pathlib import Path

from prf.manifest import canonical_path, describe_path, sha256_file


def test_canonical_path_does_not_follow_symlink(tmp_path: Path):
    target = tmp_path / "real" / "transforms_train.json"
    target.parent.mkdir()
    target.write_text('{"frames": []}\n')
    link_root = tmp_path / "repo_link"
    link_root.symlink_to(target.parent, target_is_directory=True)
    linked = link_root / "transforms_train.json"
    described = describe_path(linked)
    assert described["path"].endswith("repo_link/transforms_train.json")
    assert described["path_realpath"] == str(target.resolve())
    assert described["sha256"] == sha256_file(target)
    assert "/datacc05/" not in described["path"]
