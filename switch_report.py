#!/usr/bin/env python3
"""Aggregate per-rank stats from tenant_switch.py into one report.

Usage: switch_report.py '<glob of mt_stat.rank*.json>' [out.json]
"""

import glob
import json
import statistics
import sys


def agg(xs):
    return {
        "median_ms": statistics.median(xs),
        "mean_ms": statistics.mean(xs),
        "max_ms": max(xs),
        "n": len(xs),
    } if xs else {}


def main():
    pattern = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"no stat files match {pattern}")

    ranks = [json.load(open(f)) for f in files]
    r0 = ranks[0]
    report = {
        "ranks": len(ranks),
        "iters": r0["iters"],
        "switches": r0["switches"],
        "switch": agg([x for r in ranks for x in r["switch_ms"]]),
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
            "snapshot_gbps": nbytes / 2**30 / (statistics.median(snap) / 1e3) if snap else 0,
            "restore_gbps": nbytes / 2**30 / (statistics.median(rest) / 1e3) if rest else 0,
        }

    sw = report["switch"].get("median_ms", 0)
    print(f"ranks={report['ranks']} iters={report['iters']} switches={report['switches']}")
    for name, t in report["tenants"].items():
        print(
            f"{name:12s} state {t['state_mib']:7.0f} MiB | "
            f"step {t['step'].get('median_ms', 0):8.2f} ms | "
            f"snapshot {t['snapshot'].get('median_ms', 0):8.2f} ms "
            f"({t['snapshot_gbps']:.1f} GiB/s) | "
            f"restore {t['restore'].get('median_ms', 0):8.2f} ms "
            f"({t['restore_gbps']:.1f} GiB/s)"
        )
    print(f"context switch (snapshot+restore) median {sw:.2f} ms")
    steps = [t["step"].get("median_ms", 0) for t in report["tenants"].values()]
    if sw and all(steps):
        print(f"switch costs {sw / statistics.mean(steps):.1f}x one step")

    if out_path:
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
