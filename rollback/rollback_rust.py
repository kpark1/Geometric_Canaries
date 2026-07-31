from pathlib import Path


FIXTURE_DIR = Path(__file__).parent / "rust_fixtures"
DSP_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 400000\n\n"


def load_fixture(filename):
    return (FIXTURE_DIR / filename).read_text().rstrip()


def rust_comment(name):
    return f"-- Aeneas-style translation of Rust `{name}`\n"


LEAN_PREAMBLE = load_fixture("preamble.lean") + "\n"

LEAN_MODELS = {
    "binary_search": {
        "bug_type": "progress_bug",
        "real_world_analogue": "loop-progress / midpoint bug family "
                               "(cf. the JDK Arrays.binarySearch overflow, Bloch 2006)",
        "fixed_defn": (
            rust_comment("binary_search")
            + load_fixture("binary_search_fixed.lean")
        ),
        "buggy_defn": (
            rust_comment("binary_search")
            + load_fixture("binary_search_buggy.lean")
        ),
        "spec": load_fixture("binary_search_spec.lean"),
    },
    "sat_sub": {
        "bug_type": "underflow_panic",
        "real_world_analogue": "integer-underflow panic class "
                               "(recurring RustSec advisory category)",
        "fixed_defn": (
            rust_comment("sat_sub")
            + load_fixture("sat_sub_fixed.lean")
        ),
        "buggy_defn": (
            rust_comment("sat_sub")
            + load_fixture("sat_sub_buggy.lean")
        ),
        "spec": load_fixture("sat_sub_spec.lean"),
    },
    "clamp": {
        "bug_type": "flipped_comparison",
        "real_world_analogue": "comparison-flip logic-error family",
        "fixed_defn": (
            rust_comment("clamp")
            + load_fixture("clamp_fixed.lean")
        ),
        "buggy_defn": (
            rust_comment("clamp")
            + load_fixture("clamp_buggy.lean")
        ),
        "spec": load_fixture("clamp_spec.lean"),
    },
}


def build_prompt(item, variant, use_mathlib_header=True):
    v = item[variant]
    code_block = ((DSP_HEADER if use_mathlib_header else "")
                  + LEAN_PREAMBLE + "\n" + v["defn"] + "\n\n" + v["spec"])
    return (
        "Complete the following Lean 4 code:\n\n```lean4\n" + code_block
        + "\n```\n\n"
        "Before producing the Lean 4 code to formally prove the given theorem, "
        "provide a detailed proof plan outlining the main proof steps and "
        "strategies.\n"
        "The plan should highlight key ideas, intermediate lemmas, and proof "
        "structures that will guide the construction of the final formal proof. "
        "If the statement is false, state this explicitly and give a concrete "
        "counterexample instead of a proof."
    )
