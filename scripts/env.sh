#!/usr/bin/env bash
# Source this file: source scripts/env.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROOT

if [ -z "${PY:-}" ]; then
    if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
        export PY="${CONDA_PREFIX}/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        export PY="python3"
    else
        export PY="python"
    fi
fi

# cu121 torch wheel shared libraries
_lib_paths=""
for _base in "${CONDA_PREFIX:-}" "${VIRTUAL_ENV:-}" "${HOME}/.local"; do
    [ -n "$_base" ] || continue
    for _d in "$_base"/lib/python*/site-packages/nvidia/*/lib; do
        [ -d "$_d" ] && _lib_paths="${_lib_paths}:${_d}"
    done
done
if [ -n "$_lib_paths" ]; then
    export LD_LIBRARY_PATH="${_lib_paths#:}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# This experiment runs headless OSMesa (the original server used osmesa).
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export HANOI_INC="${HANOI_INC:-${ROOT}}"
export PYTHONPATH="${ROOT}/src:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "$ROOT/logs" "$ROOT/data" "$ROOT/results"
