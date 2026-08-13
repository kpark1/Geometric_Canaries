# Geometric_Canaries
LLM-assisted formal verification pipelines are vulnerable to a critical failure mode: adversarially perturbed theorem statements can steer models toward plausible-but-invalid proofs, with no warning until the proof checker fails. This project applies geometric chain-of-thought analysis to proof generation, using two metrics, Average Angle Error and Number of Mistakes, to detect reasoning instability mid-generation, before the checker is invoked. Both metrics are already validated as adversarial-sensitive in general reasoning 

## Rollback setup

The default rollback command loads the full DeepSeek model with 4-bit
quantization, which requires the optional `bitsandbytes` GPU dependency. Install
the project with the GPU dependency group:

```bash
uv sync --extra gpu
```

Then run the rollback program from the repository root:

```bash
uv run python rollback/rollback.py
```

For a smoke test with the smaller Qwen model, use `--dry-run`. This path does
not require `bitsandbytes`:

```bash
uv run python rollback/rollback.py --dry-run
```

## Lean setup

Checking whether a generated proof actually elaborates needs a Lean 4 toolchain
and Mathlib. Run this once per machine, from the repository root:

```bash
./scripts/setup_lean.sh
```

The script installs [elan](https://github.com/leanprover/elan) if it is missing,
then downloads the pinned toolchain and prebuilt Mathlib. It is idempotent, so
re-running it is safe.

Two things to know before you run it:

- It needs **internet access** and about **15 GB** of disk under `$HOME`
  (toolchain, vendored Mathlib source, prebuilt `.olean` files, and the
  `~/.cache/mathlib` archive store). Set `MATHLIB_CACHE_DIR` to relocate the
  last of these if space is tight.
- Installing elan **appends `~/.elan/bin` to your shell rc**, so new shells find
  `lake` and `lean`. The current shell is unaffected; open a new one, or the
  Python code will find the toolchain by absolute path regardless.

Mathlib is pinned by `lean-project/lean-toolchain` and `lean-project/lake-manifest.json`, both
committed, so everyone gets byte-identical dependencies. **Do not run
`lake update`** — it repins Mathlib and rewrites the toolchain file. The setup
script only runs it if the manifest is missing entirely.

## Development checks

Run linting, formatting validation, and type checking from the repository root:

```bash
make check
```

The checks can also be run separately:

```bash
make lint
make format-check
make type-check
```

These commands check only Python and notebook files tracked by Git.
