#!/usr/bin/env bash
# Launch the Streamlit S-RAG demo in an ISOLATED venv.
#
# Why a venv: streamlit hard-requires anyio>=4, but the base env's
# jupyter-server / jupyterlab require anyio<4. Isolating streamlit keeps BOTH
# working. The venv REUSES the base torch / transformers / sentence-transformers
# / numpy stack via include-system-site-packages=true (no 30GB re-download);
# only streamlit + anyio>=4 are added locally. All inference is local /
# open-weight — no commercial LLM API.
#
# Usage:
#   bash run_app.sh                   # device cuda:0
#   RAG_DEVICE=cuda:1 bash run_app.sh
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv-streamlit"
PYBIN="${VENV}/bin/python"
# This box ships without python3-venv/ensurepip, so the venv has no own pip. We
# drive installs with the SYSTEM pip (made importable via PYTHONPATH) run under
# the VENV interpreter, so the packages land in the venv, not the base env.
SYS_PIP_DIR="$(python3 -c 'import pip, os; print(os.path.dirname(os.path.dirname(pip.__file__)))')"
venv_pip() { PYTHONPATH="${SYS_PIP_DIR}" "${PYBIN}" -m pip "$@"; }

# 1) Create the venv once.
if [ ! -x "${PYBIN}" ]; then
  echo "[run_app] creating ${VENV}…"
  python3 -m venv --system-site-packages "${VENV}" || true   # ensurepip may warn here; harmless
fi

# 2) FORCE system-site inheritance. Debian's venv can write this 'false' even
#    with --system-site-packages; it MUST be true BEFORE installing so pip
#    reuses the base numpy/torch instead of pulling incompatible fresh copies
#    (numpy 2.x would break torch 2.3.1) into the venv.
if [ -f "${VENV}/pyvenv.cfg" ] && ! grep -q '^include-system-site-packages = true' "${VENV}/pyvenv.cfg"; then
  echo "[run_app] forcing include-system-site-packages = true"
  sed -i 's/^include-system-site-packages = .*/include-system-site-packages = true/' "${VENV}/pyvenv.cfg"
fi

# 3) Ensure streamlit (with anyio>=4) is in the venv (idempotent). With
#    system-site on, pip reuses base for everything except streamlit + anyio>=4.
if ! "${PYBIN}" -c "import streamlit" >/dev/null 2>&1; then
  echo "[run_app] installing streamlit into ${VENV} (one-time)…"
  venv_pip install --retries 6 --timeout 60 -r requirements-streamlit.txt
fi

# 4) Launch. RAG_DEVICE pins the GPU (default cuda:0; CPU fallback is automatic).
#    --server.headless true is REQUIRED on a headless/SSH server: it skips the
#    first-run "enter your email" prompt (which would otherwise block startup)
#    and does not try to open a browser. Binds to localhost by default so an SSH
#    local-forward (ssh -L PORT:localhost:PORT) reaches it securely.
export RAG_DEVICE="${RAG_DEVICE:-cuda:0}"
# Weights are fully cached under ./.hf_cache, so force OFFLINE: model loading then
# reads straight from cache and never blocks on huggingface.co HEAD checks. On a
# flaky network those checks retry for minutes and look like a hung "loading…"
# spinner. Override with HF_HUB_OFFLINE=0 if you ever need to fetch a new model.
export HF_HOME="${HF_HOME:-$(pwd)/.hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY=1
PORT="${STREAMLIT_PORT:-8501}"
ADDR="${STREAMLIT_ADDRESS:-localhost}"
echo "[run_app] device ${RAG_DEVICE}; serving on ${ADDR}:${PORT} (headless)."
echo "[run_app] To view from your LOCAL machine, open a local terminal and run:"
echo "[run_app]     ssh -N -L ${PORT}:localhost:${PORT} <user>@<this-server>"
echo "[run_app] then browse to  http://localhost:${PORT}"
exec "${VENV}/bin/streamlit" run app.py \
  --server.headless true \
  --server.address "${ADDR}" \
  --server.port "${PORT}" \
  --browser.gatherUsageStats false \
  "$@"
