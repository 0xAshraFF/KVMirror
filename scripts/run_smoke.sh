#!/usr/bin/env bash
set -euo pipefail

python3 -m kvmirror.runner --policy keep_all
python3 -m kvmirror.runner --policy recent_window --window-size 256
python3 -m kvmirror.runner --policy hybrid --window-size 192 --topk 96

