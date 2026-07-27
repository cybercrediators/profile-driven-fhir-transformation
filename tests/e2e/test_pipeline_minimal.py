import shutil
import subprocess
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MATCHBOX = "http://localhost:8080/matchboxv3/fhir/metadata"


def _matchbox_up() -> bool:
    try:
        with urllib.request.urlopen(MATCHBOX, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def test_kfdm_e2e_smoke():
    """Release smoke test: generated kfdm maps transform the REDCap test records into the
    expected 8-resource transaction Bundles (scripts/e2e_kfdm_smoke.sh, fast path).

    Requires a running matchbox (:8080) holding the kfdm_e2e resources and the generated
    projects/kfdm_e2e project — skipped otherwise so the unit suite stays self-contained.
    Full chain (FSH -> sushi -> snapshots -> maps -> NiFi): run the script with --full.
    """
    if not (REPO / "projects/kfdm_e2e/structure_maps").is_dir():
        pytest.skip("projects/kfdm_e2e not generated (run scripts/e2e_kfdm_smoke.sh --full)")
    if shutil.which("bash") is None or not _matchbox_up():
        pytest.skip("matchbox :8080 not reachable")
    res = subprocess.run(
        ["bash", str(REPO / "scripts/e2e_kfdm_smoke.sh")],
        capture_output=True, text=True, timeout=600,
    )
    assert res.returncode == 0, f"smoke failed:\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}"
    assert "SMOKE PASS (direct)" in res.stdout
