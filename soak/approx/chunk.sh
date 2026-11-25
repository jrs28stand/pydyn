#!/bin/bash
cd /home/claude/work/pydyn-soak; source soak/env.sh
for p in "$@"; do timeout 590 nice -n 19 python3 soak/approx/study.py ${p%%:*} --budget ${p##*:} 2>&1 | grep -E '^\[|Traceback|Error'; done
