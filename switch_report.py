#!/usr/bin/env python3
"""Aggregate per-rank stats from tenant_switch.py into one report.

Switch time (snapshot+restore) is reported separately from op (step) time:
percentiles for each, total wall time spent in each, overhead fraction, and
drift (first vs last decile of the run) to catch degradation over many
switches.

Usage: switch_report.py '<glob of mt_stat.rank*.json>' [out.json]
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
        "mean_ms": statistics.mean(xs),
        "total_s": sum(xs) / 1e3,
        "n": len(xs),
    } if xs else {}


def drift(xs):
    """Median of last decile over median of first decile; >1 means it got slower."""
    n = max(1, len(xs) // 10)
    return statistics.median(xs[-n:]) / statistics.median(xs[:n]) if len(xs) >= 20 else None


def fmt(a):
    return (
        f"p50 {a['p50_ms']:8.2f}  p90 {a['p90_ms']:8.2f}  p99 {a['p99_ms']:8.2f}  "
        f"max {a['max_ms']:8.2f} ms  (n={a['n']})"
    )


def main():
    pattern = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"no stat files match {pattern}")

    ranks = [json.load(open(f)) for f in files]
    r0 = ranks[0]

    switch_all = [x for r in ranks for x in r["switch_ms"]]
    step_all = [x for r in ranks for n in r["tenants"] for x in r["tenants"][n]["step_ms"]]
    report = {
        "ranks": len(ranks),
        "iters": r0["iters"],
        "switches": r0["switches"],
        "switch": agg(switch_all),
        "switch_drift": drift(r0["switch_ms"]),
        "ops": agg(step_all),
        "tenants": {},
    }

    for name in r0["tenants"]:
        step = [x for r in ranks for x in r["tenants"][name]["step_ms"]]
        snap = [x for r in ranks for x in r["tenants"][name]["snapshot_ms"]]
        rest = [x for r in ranks for x in r["tenants"][name]["restore_ms"]]
        nbytes = r0["tenants"][name]["bytes"]
        report["tenants"][name] = {
            "state_mib": nbytes / 2**20,
            "step": agg(step),
            "snapshot": agg(snap),
            "restore": agg(rest),
            "step_drift": drift([x for x in r0["tenants"][name]["step_ms"]]),
            "snapshot_gbps": nbytes / 2**30 / (pct(snap, 50) / 1e3) if snap else 0,
            "restore_gbps": nbytes / 2**30 / (pct(rest, 50) / 1e3) if rest else 0,
        }

    print(f"ranks={report['ranks']} iters={report['iters']} switches={report['switches']}")
    for name, t in report["tenants"].items():
        print(f"{name:12s} state {t['state_mib']:7.0f} MiB")
        print(f"  step      {fmt(t['step'])}")
        print(f"  snapshot  {fmt(t['snapshot'])}  {t['snapshot_gbps']:.1f} GiB/s")
        print(f"  restore   {fmt(t['restore'])}  {t['restore_gbps']:.1f} GiB/s")

    sw, op = report["switch"], report["ops"]
    print(f"switch      {fmt(sw)}")
    print(
        f"time in ops {op['total_s']:.1f} s | time in switches {sw['total_s']:.1f} s | "
        f"switch overhead {100 * sw['total_s'] / (sw['total_s'] + op['total_s']):.1f}% "
        f"(at iters={report['iters']})"
    )
    if report["switch_drift"] is not None:
        print(
            f"switch drift (rank0, last decile / first decile): "
            f"{report['switch_drift']:.2f}x"
        )

    if out_path:
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
