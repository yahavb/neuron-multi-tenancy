#!/usr/bin/env python3
"""Side-by-side table of bench_xfer.py results: host DRAM bounce vs HBM->HBM.

Also fits time = fixed + bytes / bandwidth per mode (least squares over the
measured sizes), so the two paths can be compared at any size.

Usage: xfer_report.py xfer_stat.host.json xfer_stat.hbm.json
"""

import json
import os
import sys


def fit(rows):
    xs = [r["mib"] for r in rows]
    ys = [r["p50_ms"] for r in rows]
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if not sxx:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    fixed = my - slope * mx
    return {"fixed_ms": fixed, "gib_s": (1 / 1024) / (slope / 1e3) if slope > 0 else float("inf")}


def main():
    data = {}
    for path in sys.argv[1:]:
        if os.path.exists(path):
            rows = json.load(open(path))
            if rows:
                data[rows[0]["mode"]] = rows
        else:
            print(f"missing {path} (that mode failed; see its log)")

    host = {r["mib"]: r for r in data.get("host", [])}
    hbm = {r["mib"]: r for r in data.get("hbm", [])}
    prim = data["hbm"][0]["primitive"] if "hbm" in data else "-"

    print(f"core-to-core transfer, p50 ms (hbm primitive: {prim})")
    print(f"{'MiB':>6} | {'host total':>10} {'d2h':>8} {'handoff':>8} {'h2d':>8} {'GiB/s':>6} | "
          f"{'hbm':>9} {'GiB/s':>6} | {'host/hbm':>8} | verify")
    for mib in sorted(set(host) | set(hbm)):
        h, d = host.get(mib), hbm.get(mib)
        hs = (f"{h['p50_ms']:10.3f} {h['d2h_ms']:8.3f} {h['handoff_ms']:8.3f} {h['h2d_ms']:8.3f} "
              f"{h['gib_s']:6.2f}") if h else f"{'-':>10} {'':>8} {'':>8} {'':>8} {'':>6}"
        ds = f"{d['p50_ms']:9.3f} {d['gib_s']:6.2f}" if d else f"{'-':>9} {'':>6}"
        ratio = f"{h['p50_ms'] / d['p50_ms']:8.2f}x" if h and d else f"{'-':>9}"
        ver = "/".join(
            ("ok" if r["verified"] else "FAIL") if r else "-" for r in (h, d))
        print(f"{mib:6d} | {hs} | {ds} | {ratio} | {ver}")

    for mode, rows in data.items():
        f = fit(rows)
        if f:
            print(f"fit {mode:4s}: time ~= {f['fixed_ms']:.3f} ms + size / {f['gib_s']:.2f} GiB/s")
    if prim == "all_gather":
        print("note: hbm used all_gather (broadcast not available); it moves data both "
              "directions, so hbm times are an upper bound")


if __name__ == "__main__":
    main()
