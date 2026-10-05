#!/usr/bin/env python3
"""Aggregate per-rank stats from dispatch_serve.py.

Per model: event count, residency hit rate, swap-in / swap-out / run
percentiles (all ranks), and the end-to-end latency the HTTP caller saw
(rank 0), split into queue wait, device work, and dispatch overhead.

Usage: dispatch_report.py '<glob of dispatch_stat.rank*.json>' [out.json]
"""

import glob
import json
import statistics
import sys


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def agg(xs):
    return {
        "p50_ms": pct(xs, 50),
        "p90_ms": pct(xs, 90),
        "p99_ms": pct(xs, 99),
        "max_ms": max(xs),
        "total_s": sum(xs) / 1e3,
        "n": len(xs),
    } if xs else {}


def fmt(a):
    if not a:
        return "(none)"
    return (f"p50 {a['p50_ms']:8.2f}  p90 {a['p90_ms']:8.2f}  "
            f"p99 {a['p99_ms']:8.2f}  max {a['max_ms']:8.2f} ms  (n={a['n']})")


def main():
    pattern = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"no stat files match {pattern}")

    ranks = [json.load(open(f)) for f in files]
    r0 = next(r for r in ranks if r["rank"] == 0)
    recs = [x for r in ranks for x in r["records"]]
    e2e = r0["e2e"] or []

    report = {
        "policy": r0["policy"],
        "pattern": r0["pattern"],
        "events": r0["events"],
        "steps_per_event": r0["steps_per_event"],
        "ranks": len(ranks),
        "models": {},
    }

    print(f"policy={r0['policy']} pattern={r0['pattern']} events={r0['events']} "
          f"steps/event={r0['steps_per_event']} ranks={len(ranks)}")

    for mid in (0, 1):
        mr = [x for x in recs if x["model"] == mid]
        me = [x["e2e_ms"] for x in e2e if x["model"] == mid]
        mq = [x["queue_ms"] for x in e2e if x["model"] == mid]
        if not mr:
            continue
        hits = sum(1 for x in mr if x["hit"])
        m = {
            "events": len(mr) // len(ranks),
            "hit_rate": hits / len(mr),
            "swap_in": agg([x["swap_in_ms"] for x in mr if x["swap_in_ms"] > 0]),
            "swap_out": agg([x["swap_out_ms"] for x in mr if x["swap_out_ms"] > 0]),
            "run": agg([x["run_ms"] for x in mr]),
            "e2e": agg(me),
            "queue": agg(mq),
        }
        report["models"][f"m{mid}"] = m
        print(f"m{mid}: {m['events']} events, residency hit rate {100 * m['hit_rate']:.0f}%")
        print(f"  swap_in   {fmt(m['swap_in'])}")
        print(f"  swap_out  {fmt(m['swap_out'])}")
        print(f"  run       {fmt(m['run'])}")
        print(f"  e2e       {fmt(m['e2e'])}")
        print(f"  queue     {fmt(m['queue'])}")

    if e2e:
        run_p50 = statistics.median([x["run_ms"] for x in recs])
        e2e_p50 = pct([x["e2e_ms"] for x in e2e], 50)
        q_p50 = pct([x["queue_ms"] for x in e2e], 50)
        swap_p50 = e2e_p50 - q_p50 - run_p50
        report["overhead"] = {
            "e2e_p50_ms": e2e_p50,
            "queue_p50_ms": q_p50,
            "run_p50_ms": run_p50,
            "dispatch_p50_ms": swap_p50,
        }
        print(f"median event: e2e {e2e_p50:.2f} ms = queue {q_p50:.2f} + "
              f"run {run_p50:.2f} + dispatch(swaps+sync+http) {swap_p50:.2f} ms")

    if out_path:
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
