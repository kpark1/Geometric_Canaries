#!/usr/bin/env bash
# One-time setup of the pinned Lean 4 + Mathlib toolchain used to check whether
# generated proofs elaborate. Safe to re-run; it is idempotent.
#
# Requires internet access and roughly 15 GB of disk under $HOME.
# See the "Lean setup" section of README.md.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Hyphenated so the directory can never be imported as a Python module and
# shadow rollback/lean.py.
LEAN_DIR="$REPO_ROOT/lean-project"
ELAN_BIN="${ELAN_HOME:-$HOME/.elan}/bin"

for tool in curl git; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "error: '$tool' is required but not installed." >&2
        exit 1
    fi
done

# Preflight rather than letting curl/git hang later on an unreachable network.
if ! curl -sSf -m 20 -o /dev/null https://github.com 2>/dev/null; then
    echo "error: cannot reach github.com. This setup step requires internet access." >&2
    echo "       (On a cluster, run it somewhere with outbound network.)" >&2
    exit 1
fi

if ! command -v elan >/dev/null 2>&1 && [ ! -x "$ELAN_BIN/elan" ]; then
    echo "==> Installing elan (Lean toolchain manager)"
    # --default-toolchain none: the toolchain is chosen by lean-project/lean-toolchain,
    # so downloading 'stable' here would waste ~2 GB. PATH modification is left
    # enabled on purpose so future shells find lake/lean.
    curl https://elan.lean-lang.org/elan-init.sh -sSf | sh -s -- -y --default-toolchain none
else
    echo "==> elan already installed"
fi

# Step above only edits shell rc files, which this process never sourced.
export PATH="$ELAN_BIN:$PATH"

# elan's shims pick a toolchain from the lean-toolchain file found by walking up
# from the *working directory*. Lake's own --dir flag is resolved too late to
# affect that, so we must actually cd here.
cd "$LEAN_DIR"

echo "==> Resolving toolchain from lean-project/lean-toolchain"
lake --version

if [ ! -f "lake-manifest.json" ]; then
    # Bootstrap only. `lake update` rewrites lean-toolchain from the dependency
    # and triggers Mathlib's post_update hook, so it must not run once the
    # manifest is committed.
    echo "==> No lake-manifest.json; resolving dependencies (first-time bootstrap)"
    lake update --keep-toolchain
else
    echo "==> Using committed lake-manifest.json"
fi

# When the system curl is older than 7.81, Mathlib's `cache` downloads a static
# curl of its own. That binary ships no CA bundle path, so on distros that keep
# certs outside its compiled-in default every HTTPS fetch dies with
# "STORE routines::unregistered scheme". Point it at the system trust store.
if [ -z "${CURL_CA_BUNDLE:-}" ]; then
    for bundle in \
        /etc/pki/tls/certs/ca-bundle.crt \
        /etc/ssl/certs/ca-certificates.crt \
        /etc/ssl/cert.pem
    do
        if [ -r "$bundle" ]; then
            export CURL_CA_BUNDLE="$bundle"
            break
        fi
    done
fi

echo "==> Downloading prebuilt Mathlib (this takes a while)"
lake exe cache get

scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT

cat >"$scratch/GoodCheck.lean" <<'EOF'
import Mathlib
import Aesop

example : 2 + 2 = 4 := by norm_num
EOF

cat >"$scratch/SorryCheck.lean" <<'EOF'
import Mathlib

theorem sorry_check : 2 + 2 = 4 := by sorry
EOF

echo "==> Verifying a Mathlib proof elaborates"
lake env lean "$scratch/GoodCheck.lean"

echo "==> Verifying the no-sorry gate rejects a sorry"
if lake env lean --error=hasSorry "$scratch/SorryCheck.lean" >/dev/null 2>&1; then
    echo "error: --error=hasSorry accepted a proof containing 'sorry'." >&2
    echo "       The no-sorry gate is not working; do not trust results." >&2
    exit 1
fi

echo
echo "Lean setup complete."
echo "Open a new shell (or add $ELAN_BIN to PATH) to use lake/lean directly."
