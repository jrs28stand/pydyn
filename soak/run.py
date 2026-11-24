"""Soak driver.  python3 soak/run.py PARTS [--budget SECONDS] [--replay INDEX]
PARTS: A, B, C, D, or 'all' (budget split equally, every part resumes from soak/state/<part>.json).
Run under `nice -n 19` with the environment of soak/env.sh; one process at a time."""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402
from lib import Anom, append, install_verifiers, load_state, record_warnings, rss_mb, save_state  # noqa: E402


def get_task(part):
    if part in "AB":
        import part_a
        return part_a.task_A if part == "A" else part_a.task_B
    import part_cd
    return part_cd.task_C if part == "C" else part_cd.task_D


def run_part(part, budget, replay=None):
    st = load_state(part)
    task = get_task(part)
    install_verifiers()
    t_start = time.time()
    rss0 = rss_mb()
    n = 0
    if replay is not None:
        st = dict(next=replay, seconds=0.0, kinds={}, warnings={}, counts={}, medians={}, notes={})
        budget = 0
    while True:
        i = st["next"] if replay is None else replay
        t0 = time.time()
        anom = Anom(part, st)
        where = dict(task=i)
        try:
            with record_warnings(st, anom, dict(task=i, part=part)):
                rec = task(i, st, anom)
        except lib.Alarm:
            rec = dict(i=i, timeout=True)
            anom("timeout", task=i, what="task level")
        except Exception as exc:                                   # noqa: BLE001  (a bug of the driver, or an unexpected error)
            ex = lib.exc_info(exc)
            anom(f"driver_or_unexpected:{ex['type']}:{lib.norm_msg(ex['msg'])}", task=i, msg=ex["msg"], tb=ex["tb"])
            rec = dict(i=i, driver_error=ex["type"])
        for k, v in lib.VERIFY.items():
            lib.count(st, f"verify|{k}", v)
        lib.VERIFY.clear()
        rec["sec"] = round(time.time() - t0, 2)
        rec["rss"] = round(rss_mb(), 1)
        if anom.new:
            rec["anom"] = anom.new
        if replay is not None:
            print(json.dumps(rec, default=lib._clean)[:3000])
            print("kinds", st["kinds"])
            return
        append(f"{part}.jsonl", rec)
        st["next"] = i + 1
        st["seconds"] += time.time() - t0
        st["rss_last"] = rec["rss"]
        save_state(part, st)
        n += 1
        if time.time() - t_start > budget:
            break
    print(f"[{part}] tasks {n} (next {st['next']}), part seconds {st['seconds']:.0f}, rss {rss0:.0f}->{rss_mb():.0f} MB, "
          f"anomaly kinds {len(st['kinds'])} (total {sum(st['kinds'].values())})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("parts")
    ap.add_argument("--budget", type=float, default=480)
    ap.add_argument("--replay", type=int)
    a = ap.parse_args()
    parts = list("ABCD") if a.parts == "all" else list(a.parts)
    for p in parts:
        run_part(p, a.budget / len(parts), a.replay)
