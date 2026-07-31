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
