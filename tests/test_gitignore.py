from pathlib import Path
import shutil
import subprocess

import pytest


def test_generated_data_is_ignored_but_dataset_source_can_be_committed(tmp_path):
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is unavailable")
    subprocess.run([git, "init", "--quiet", str(tmp_path)], check=True)
    root = Path(__file__).resolve().parents[1]
    shutil.copyfile(root / ".gitignore", tmp_path / ".gitignore")
    result = subprocess.run(
        [git, "check-ignore", "--no-index", "--stdin"], cwd=tmp_path,
        input="data/example.txt\nsrc/regionssod/data/annotations.py\n",
        text=True, capture_output=True, check=True)
    assert result.stdout.splitlines() == ["data/example.txt"]
