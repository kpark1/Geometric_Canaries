from pathlib import Path
from typing import Any, Protocol

from rollback_types import Challenge

FIXTURE_DIR = Path(__file__).parent / "rust_fixtures"
DSP_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 400000\n\n"


def load_fixture(filename: str) -> str:
    return (FIXTURE_DIR / filename).read_text().rstrip()


def rust_comment(name: str) -> str:
    return f"-- Aeneas-style translation of Rust `{name}`\n"


LEAN_PREAMBLE = load_fixture("preamble.lean") + "\n"

class ChallengeWithMetadata(Challenge):
    bug_type: str
    real_world_analogue: str

LEAN_MODELS: list[ChallengeWithMetadata] = [
    {
        "id": "binary_search_fixed",
        "bug_type": "progress_bug",
        "real_world_analogue": "loop-progress / midpoint bug family "
        "(cf. the JDK Arrays.binarySearch overflow, Bloch 2006)",
        "definition": (
            rust_comment("binary_search") + load_fixture("binary_search_fixed.lean")
        ),
        "spec": load_fixture("binary_search_spec.lean"),
        "buggy": False,
    },
    {
        "id": "binary_search_buggy",
        "bug_type": "progress_bug",
        "real_world_analogue": "loop-progress / midpoint bug family "
        "(cf. the JDK Arrays.binarySearch overflow, Bloch 2006)",
        "definition": (
            rust_comment("binary_search") + load_fixture("binary_search_buggy.lean")
        ),
        "spec": load_fixture("binary_search_spec.lean"),
        "buggy": True,
    },
    {
        "id": "sat_sub_fixed",
        "bug_type": "underflow_panic",
        "real_world_analogue": "integer-underflow panic class "
        "(recurring RustSec advisory category)",
        "definition": (rust_comment("sat_sub") + load_fixture("sat_sub_fixed.lean")),
        "spec": load_fixture("sat_sub_spec.lean"),
        "buggy": False,
    },
    {
        "id": "sat_sub_buggy",
        "bug_type": "underflow_panic",
        "real_world_analogue": "integer-underflow panic class "
        "(recurring RustSec advisory category)",
        "definition": (rust_comment("sat_sub") + load_fixture("sat_sub_buggy.lean")),
        "spec": load_fixture("sat_sub_spec.lean"),
        "buggy": True,
    },
    {
        "id": "clamp_fixed",
        "bug_type": "flipped_comparison",
        "real_world_analogue": "comparison-flip logic-error family",
        "definition": (rust_comment("clamp") + load_fixture("clamp_fixed.lean")),
        "spec": load_fixture("clamp_spec.lean"),
        "buggy": False,
    },
    {
        "id": "clamp_buggy",
        "bug_type": "flipped_comparison",
        "real_world_analogue": "comparison-flip logic-error family",
        "definition": (rust_comment("clamp") + load_fixture("clamp_buggy.lean")),
        "spec": load_fixture("clamp_spec.lean"),
        "buggy": True,
    },
]


class BuildPromptFunction(Protocol):
    def __call__(
        self,
        challenge: Challenge,
        use_mathlib_header: bool = True,
    ) -> str: ...
def build_prompt(
    challenge: Challenge,
    use_mathlib_header: bool = True,
) -> str:
    code_block = (
        (DSP_HEADER if use_mathlib_header else "")
        + LEAN_PREAMBLE
        + "\n"
        + challenge["definition"]
        + "\n\n"
        + challenge["spec"]
    )
    return (
        "Complete the following Lean 4 code:\n\n```lean4\n" + code_block + "\n```\n\n"
        "Before producing the Lean 4 code to formally prove the given theorem, "
        "provide a detailed proof plan outlining the main proof steps and "
        "strategies.\n"
        "The plan should highlight key ideas, intermediate lemmas, and proof "
        "structures that will guide the construction of the final formal proof. "
        "If the statement is false, state this explicitly and give a concrete "
        "counterexample instead of a proof."
    )
_: BuildPromptFunction = build_prompt