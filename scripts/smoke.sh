#!/bin/sh
set -eu
.venv/bin/emg --config configs/smoke.yaml run-all
