#!/bin/bash
# one ~8 minute chunk of the soak, all parts in rotation (resumes from soak/state)
cd /home/claude/work/pydyn-soak
source soak/env.sh
timeout 585 nice -n 19 python3 soak/run.py ${1:-all} --budget ${2:-480} 2>&1 | grep -E '^\[|Traceback|Error' | head -12
