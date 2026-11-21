#!/usr/bin/env bash
# One-shot environment setup for the benchmark machine (macOS; also works on Linux).
# Creates .venv with Python 3.12 and the pinned packages, then installs the
# simulator in editable mode. Safe to run again.
#
#   bash bench/setup_mac.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -x .venv/bin/python ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 --seed .venv
  elif command -v python3.12 >/dev/null 2>&1; then
    # Ubuntu ships python3.12 without the venv module; add it when missing.
    if ! python3.12 -c "import ensurepip" >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
      sudo apt-get update -qq
      sudo apt-get install -y -qq python3.12-venv
    fi
    rm -rf .venv
    python3.12 -m venv .venv
  else
    echo "Python 3.12 not found. Install it first, for example:" >&2
    echo "  brew install python@3.12        (macOS with Homebrew)" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh    (uv, any system)" >&2
    exit 1
  fi
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .

echo
.venv/bin/python -c "import platform, jax, numpy; print('Python', platform.python_version(), '| jax', jax.__version__, '| numpy', numpy.__version__)"
echo "Done. Next:"
echo "  source .venv/bin/activate"
echo "  python bench/run.py selftest"
echo "  python bench/run.py doctor"
