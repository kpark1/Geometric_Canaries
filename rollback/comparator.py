import json
import os
import shutil
from pathlib import Path

from lean import DEFAULT_TIMEOUT_S, LEAN_PROJECT_DIR, find_lake
from rollback_types import ComparatorStatus, ProcessResultOrTimeout, ProcessTimeout
from subprocess_utils import run_subprocess_with_timeout

COMPARATOR_HOME = Path(
    os.environ.get("COMPARATOR_HOME")
    or (Path(__file__).resolve().parent.parent / "comparator-tool")
)

PERMITTED_AXIOMS = ["propext", "Quot.sound", "Classical.choice"]


def _comparator_bin() -> Path:
    path = COMPARATOR_HOME / ".lake" / "build" / "bin" / "comparator"
    if not path.is_file():
        raise RuntimeError(
            "Could not find the comparator binary. Run scripts/setup_lean.sh to "
            "build it, or set COMPARATOR_HOME if it lives elsewhere."
        )
    return path


def _lean4export_bin() -> Path:
    path = (
        COMPARATOR_HOME
        / ".lake"
        / "packages"
        / "lean4export"
        / ".lake"
        / "build"
        / "bin"
        / "lean4export"
    )
    if not path.is_file():
        raise RuntimeError(
            "Could not find the lean4export binary. Run scripts/setup_lean.sh to "
            "build it, or set COMPARATOR_HOME if it lives elsewhere."
        )
    return path


def _fake_landrun_script() -> Path:
    path = COMPARATOR_HOME / "scripts" / "fake-landrun.sh"
    if not path.is_file():
        raise RuntimeError(
            "Could not find comparator's fake-landrun.sh. Run scripts/setup_lean.sh "
            "to clone comparator, or set COMPARATOR_HOME if it lives elsewhere."
        )
    return path


def _reset_build_dir() -> None:
    # Clears stale build output from the previous check; leaves .lake/packages
    # (Mathlib) alone since it lives in a separate tree.
    shutil.rmtree(LEAN_PROJECT_DIR / ".lake" / "build", ignore_errors=True)


def _write_files(challenge_lean: str, solution_lean: str, theorem_name: str) -> Path:
    (LEAN_PROJECT_DIR / "Challenge.lean").write_text(challenge_lean, encoding="utf-8")
    (LEAN_PROJECT_DIR / "Solution.lean").write_text(solution_lean, encoding="utf-8")
    config = {
        "challenge_module": "Challenge",
        "solution_module": "Solution",
        "theorem_names": [theorem_name],
        "permitted_axioms": PERMITTED_AXIOMS,
        # Not optional in comparator's config schema despite being a
        # backwards-compat flag: a missing key fails JSON parsing outright.
        "enable_nanoda": False,
    }
    config_path = LEAN_PROJECT_DIR / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def _run_comparator(
    lake: str, config_path: Path, timeout_s: float
) -> ProcessResultOrTimeout:
    env = {
        **os.environ,
        "COMPARATOR_LANDRUN": str(_fake_landrun_script()),
        "COMPARATOR_LEAN4EXPORT": str(_lean4export_bin()),
    }
    return run_subprocess_with_timeout(
        [lake, "env", str(_comparator_bin()), str(config_path)],
        cwd=LEAN_PROJECT_DIR,
        timeout_s=timeout_s,
        env=env,
    )


def check_solution_matches_challenge(
    challenge_lean: str,
    solution_lean: str,
    theorem_name: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> ComparatorStatus:
    lake = find_lake()
    _reset_build_dir()
    config_path = _write_files(challenge_lean, solution_lean, theorem_name)
    result = _run_comparator(lake, config_path, timeout_s)
    if isinstance(result, ProcessTimeout):
        return ComparatorStatus.TIMEOUT
    return ComparatorStatus.OK if result.exit_code == 0 else ComparatorStatus.ERROR
