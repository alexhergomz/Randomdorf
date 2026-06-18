#!/usr/bin/env bash
# Convenience launcher for the RFF ocean.
# On a hybrid laptop, render on the discrete GPU (e.g. `prime-select nvidia`). Rendering on
# one GPU while presenting on another can hang.
#
# Pass extra args through, e.g.:  ./run.sh --resolution 1920x1080
cd "$(dirname "$0")" || exit 1
exec godot --path . "$@"
