#!/usr/bin/env bash
# Stemforge local setup for Linux/macOS:  ./scripts/setup.sh [--cpu]
set -euo pipefail
cd "$(dirname "$0")/.."
INDEX=https://download.pytorch.org/whl/cu128
if [[ "${1:-}" == "--cpu" || "$(uname)" == "Darwin" ]]; then INDEX=https://download.pytorch.org/whl/cpu; fi
command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/"; exit 1; }
uv venv --python 3.11 backend/.venv
PY=backend/.venv/bin/python
uv pip install --python $PY torch==2.11.0 torchaudio==2.11.0 --index-url "$INDEX"
uv pip install --python $PY --no-deps -r backend/requirements/lock.txt
uv pip install --python $PY --no-deps -e backend
(cd backend && .venv/bin/stemforge-models download)
(cd frontend && npm ci)
cat <<'MSG'
Done. In three terminals:
  cd backend && .venv/bin/stemforge-api
  cd backend && .venv/bin/stemforge-worker
  cd frontend && npm run dev
Then open http://localhost:3000
MSG
