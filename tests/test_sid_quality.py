import json
import subprocess
import sys
from pathlib import Path


def test_quality_script_reports_collisions(tmp_path: Path) -> None:
    table = {
        "0": [0, 0],
        "1": [0, 0],
        "2": [0, 1],
        "3": [1, 1],
    }
    index = tmp_path / "index.json"
    output = tmp_path / "quality.json"
    index.write_text(json.dumps(table), encoding="utf-8")

    script = Path(__file__).parents[1] / "scripts" / "evaluate_sid_quality.py"
    result = subprocess.run(
        [sys.executable, str(script), "--index", str(index), "--codebook-sizes", "2", "2", "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(result.stdout)
    assert report["items"] == 4
    assert report["unique_sid"] == 3
    assert report["collision_item_rate"] == 0.5
    assert json.loads(output.read_text(encoding="utf-8"))["collision_groups"] == 1
