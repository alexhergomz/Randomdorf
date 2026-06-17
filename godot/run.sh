#!/usr/bin/env bash
# Run the RFF ocean on the GPU that drives the display (AMD iGPU on this HP Victus),
# bypassing the NVIDIA->AMD PRIME render-offload copy that hangs in `on-demand` mode.
#
# Use this ONLY while prime-select is in `on-demand` (hybrid) mode.
# If you switch to `prime-select nvidia` (Option 1), drop the flag and just run:
#     godot --path "$(dirname "$0")"
#
# Pass extra args through, e.g.:  ./run.sh --resolution 1920x1080
cd "$(dirname "$0")" || exit 1
exec godot --path . --gpu-index 0 "$@"
