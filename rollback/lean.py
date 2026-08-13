import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import Any

from rollback_types import (
    ProcessResult,
    ProcessResultOrTimeout,
    ProcessTimeout,
    ProveRunResult,
)

LEAN_PROJECT_DIR = Path(__file__).resolve().parent.parent / "lean"

# Lean renders the `hasSorry` warning as "declaration uses `<expr>`", where the
# expression is interpolated, so only this prefix is stable across sorry forms.
SORRY_MESSAGE_PREFIX = "declaration uses "

# code blocks beginning ```lean4 or ```lean
LEAN_BLOCK = re.compile(r"```lean4?[ \t]*\r?\n(.*?)```", re.DOTALL)


class LeanStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    SORRY = "sorry"
    TIMEOUT = "timeout"


def extract_lean_deepseek_prover_style(res: ProveRunResult) -> str | None:
    """Pull the final fenced Lean block out of a DeepSeek-Prover-style response.

    Returns the last closed block, since the chain-of-thought plan that precedes
    it often contains illustrative snippets.
    """
    blocks = LEAN_BLOCK.findall(res["text"])
    if not blocks:
        return None
    return blocks[-1].strip()


def _find_lake() -> str:
    """Locate `lake`, tolerating a PATH without elan's shims (e.g. batch jobs)."""
    on_path = shutil.which("lake")
    if on_path is not None:
        return on_path

    elan_home = os.environ.get("ELAN_HOME") or os.path.expanduser("~/.elan")
    fallback = Path(elan_home) / "bin" / "lake"
    if fallback.is_file():
        return str(fallback)

    raise RuntimeError(
        "Could not find `lake`. Run scripts/setup_lean.sh to install the Lean "
        "toolchain, or set ELAN_HOME if it lives outside ~/.elan."
    )


def _run_lean(lake: str, source_path: Path, timeout_s: float) -> ProcessResultOrTimeout:
    # elan resolves the toolchain by walking up from the working directory, so
    # cwd (not lake's --dir) is what selects the pinned toolchain here.
    process = subprocess.Popen(
        [lake, "env", "lean", "--json", "--error=hasSorry", str(source_path)],
        cwd=LEAN_PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        # Own process group, so a hung elaboration can be killed as a unit; the
        # default kills only `lake` and leaves `lean` orphaned on a core.
        start_new_session=True,
    )
    try:
        stdout, _ = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
        return ProcessTimeout()
    return ProcessResult(process.returncode, stdout)


def _parse_messages(stdout: str) -> list[dict[str, Any]]:
    """Read `lean --json` output, one message object per line."""
    messages = []
    for line in stdout.splitlines():
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            messages.append(parsed)
    return messages


def _classify(res: ProcessResult) -> LeanStatus:
    messages = _parse_messages(res.stdout)
    if any(
        str(message.get("data", "")).startswith(SORRY_MESSAGE_PREFIX)
        for message in messages
    ):
        return LeanStatus.SORRY
    if res.exit_code != 0 or any(
        message.get("severity") == "error" for message in messages
    ):
        return LeanStatus.ERROR
    return LeanStatus.OK


# Dominated by `import Mathlib`, which is I/O bound on the ~7 GB of .olean files
# and can take several minutes on network storage; a tighter budget would report
# healthy proofs as TIMEOUT.
DEFAULT_TIMEOUT_S = 1200.0


def lean_elaborates_no_sorry(
    lean_code: str, timeout_s: float = DEFAULT_TIMEOUT_S
) -> LeanStatus:
    lake = _find_lake()
    with tempfile.TemporaryDirectory() as workdir:
        source_path = Path(workdir) / "GeneratedProof.lean"
        source_path.write_text(lean_code, encoding="utf-8")
        result = _run_lean(lake, source_path, timeout_s)

    if isinstance(result, ProcessTimeout):
        return LeanStatus.TIMEOUT
    return _classify(result)
