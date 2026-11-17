import os
import signal
import subprocess
from pathlib import Path

from rollback_types import ProcessResult, ProcessResultOrTimeout, ProcessTimeout


def run_subprocess_with_timeout(
    argv: list[str],
    cwd: Path,
    timeout_s: float,
    env: dict[str, str] | None = None,
) -> ProcessResultOrTimeout:
    """Run a subprocess in its own process group, killing the whole group on timeout.

    The default `Popen` kill only targets the immediate child; for `lake`
    invocations that leaves the actual `lean` elaboration process orphaned on
    a hung check.
    """
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
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
