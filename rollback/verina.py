import re
from typing import Protocol

from datasets import load_dataset
from rollback_types import Challenge

VERINA_DATASET = "sunblaze-ucb/verina"
DSP_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 400000\n\n"

# Verina delimits solution/spec/proof regions with `-- !benchmark @start ...` and
# `-- !benchmark @end ...` marker lines. They are dataset metadata, not Lean, so
# drop them before showing the code to the model.
BENCHMARK_MARKER = re.compile(r"^[ \t]*-- !benchmark @(?:start|end).*\n?", re.MULTILINE)

# Verina ships the reference proof inline: the `proof` region holds the body of the
# main `..._spec_satisfied` theorem and the `proof_aux` region holds supporting
# lemmas. Both leak the answer, so replace the main proof body with `sorry` and drop
# the aux lemmas. `(?!_aux)` keeps `proof` from matching `proof_aux`; non-greedy
# matching pairs each start marker with its own end marker.
PROOF_REGION = re.compile(
    r"(-- !benchmark @start proof(?!_aux)[^\n]*\n).*?"
    r"(^[ \t]*-- !benchmark @end proof(?!_aux)[^\n]*\n?)",
    re.DOTALL | re.MULTILINE,
)
PROOF_AUX_REGION = re.compile(
    r"-- !benchmark @start proof_aux[^\n]*\n.*?^[ \t]*-- !benchmark @end proof_aux[^\n]*\n?",
    re.DOTALL | re.MULTILINE,
)


def _clean_lean_code(lean_code: str) -> str:
    without_proof = PROOF_REGION.sub(r"\1  sorry\n\2", lean_code)
    without_aux = PROOF_AUX_REGION.sub("", without_proof)
    return BENCHMARK_MARKER.sub("", without_aux).strip()


def load_verina_challenges(limit: int | None = None) -> list[Challenge]:
    """Load Verina examples as challenges.

    Verina embeds its reference proof inline, so ``_clean_lean_code`` strips the
    main ``..._spec_satisfied`` proof body back to ``sorry`` and drops the helper
    lemmas; each ``lean_code`` is then a full Lean file ending in a
    ``theorem ..._spec_satisfied ... := by sorry`` that the model must complete.
    Verina has no buggy/fixed distinction, so every challenge is ``buggy=False``.
    ``limit <= 0`` (or ``None``) loads all examples.
    """
    dataset = load_dataset(VERINA_DATASET, split="train")
    challenges: list[Challenge] = []
    for row in dataset:
        challenges.append(
            {
                "id": row["id"],
                "lean_code": _clean_lean_code(row["lean_code"]),
                "buggy": False,
            }
        )
        if limit is not None and limit > 0 and len(challenges) >= limit:
            break
    return challenges


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
    code_block = (DSP_HEADER if use_mathlib_header else "") + challenge["lean_code"]
    return (
        "Complete the following Lean 4 code by replacing the `sorry` with a "
        "complete proof:\n\n```lean4\n" + code_block + "\n```\n\n"
        "Before producing the Lean 4 code to formally prove the given theorem, "
        "provide a detailed proof plan outlining the main proof steps and "
        "strategies.\n"
        "The plan should highlight key ideas, intermediate lemmas, and proof "
        "structures that will guide the construction of the final formal proof."
    )


_: BuildPromptFunction = build_prompt
