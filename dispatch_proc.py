#!/usr/bin/env python3
"""Process-per-request dispatch: two apps take turns owning the same device.

The dispatcher itself never touches Neuron. It serves HTTP on DISPATCH_PORT:
POST /e0 runs app 0 (dit stand-in), POST /e1 runs app 1 (unrolling stand-in),
POST /shutdown stops. Requests are served one at a time, because only one app
can own the cores. For each request:

  1. launch `torchrun --nproc-per-node NPROC app_run.py --app K`
  2. the app inits, loads its graph, restores its state from host, runs
  3. on the app's MT_RESULT line, reply to the user right away
  4. the app copies its state HBM -> host and exits, releasing the cores
  5. only after the exit can the next request start

Startup runs each app once with --prewarm (compile + create state), not
counted. EVENT_SOURCE=self drives EVENTS requests through the real HTTP path
in EVENT_PATTERN order (alternate | burst | random); external waits for
callers. Writes every request's timings to MT_STAT and prints a summary.
"""

import json
import os
import queue
import random
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

NPROC = int(os.environ.get("NPROC", "8"))
EVENTS = int(os.environ.get("EVENTS", "20"))
PATTERN = os.environ.get("EVENT_PATTERN", "alternate")
BURST = int(os.environ.get("EVENT_BURST", "4"))
PORT = int(os.environ.get("DISPATCH_PORT", "8080"))
SOURCE = os.environ.get("EVENT_SOURCE", "self")
STAT = os.environ.get("MT_STAT", "/tmp/proc_stat.json")
HERE = os.path.dirname(os.path.abspath(__file__))

SHUTDOWN = -1


class Item:
    def __init__(self, app):
        self.app = app
        self.t_arrival = time.time()
        self.done = threading.Event()
        self.reply = {}


def run_app(app, prewarm=False, on_result=None):
    """Run one app process to completion. Returns the merged timing record."""
    cmd = ["torchrun", "--standalone", f"--nproc-per-node={NPROC}",
           os.path.join(HERE, "app_run.py"), "--app", str(app)]
    if prewarm:
        cmd.append("--prewarm")
    t_spawn = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, cwd=HERE)
    rec = {"app": app, "t_spawn": t_spawn}
    for line in p.stdout:
        if line.startswith("MT_RESULT "):
            rec.update(json.loads(line[len("MT_RESULT "):]))
            rec["t_reply"] = time.time()
            if on_result:
                on_result(rec)
        elif line.startswith("MT_DONE "):
            rec.update(json.loads(line[len("MT_DONE "):]))
        elif "Warning" not in line and "warn" not in line and line.strip() not in ("", "^"):
            sys.stdout.write(f"  [app{app}] {line}")
    rc = p.wait()
    rec["t_exit"] = time.time()
    rec["rc"] = rc
    if rc != 0 or "t_result" not in rec:
        raise SystemExit(f"app {app} failed (rc={rc}); see [app{app}] lines above")
    return rec


STARTUP = [  # (phase, start mark, end mark) per rank
    ("launch", "t_spawn", "t_start"),         # torchrun + Python start
    ("torch_import", "t_start", "t_torch"),
    ("neuronx_import", "t_torch", "t_neuronx"),
    ("gloo_init", "t_neuronx", "t_gloo"),     # also waits for every rank to arrive
    ("runtime_init", "t_gloo", "t_init"),     # nrt_init + first tensor on the core
    ("load", "t_init", "t_load"),             # compiled graph from cache
    ("restore", "t_load", "t_restore"),
    ("run", "t_restore", "t_run"),
]


def phases(rec):
    """Durations in ms: rank 0's, and the slowest rank's (suffix _max)."""
    out = {}
    ranks = rec.get("ranks") or [rec]
    for name, a, b in STARTUP:
        per = [((r[b] - (rec["t_spawn"] if a == "t_spawn" else r[a])) * 1e3) for r in ranks]
        r0 = next(r for r in ranks if r.get("rank", 0) == 0)
        out[f"{name}_ms"] = (r0[b] - (rec["t_spawn"] if a == "t_spawn" else r0[a])) * 1e3
        out[f"{name}_max_ms"] = max(per)
    slowest = max(ranks, key=lambda r: r["t_run"])
    out["slowest_rank"] = slowest.get("rank", 0)
    out["slowest_wait_ms"] = (rec["t_result"] - rec["t_run"]) * 1e3
    out["snapshot_ms"] = (rec["t_snapshot"] - rec["t_result"]) * 1e3
    out["exit_ms"] = (rec["t_exit"] - rec["t_snapshot"]) * 1e3
    out["device_busy_ms"] = (rec["t_exit"] - rec["t_spawn"]) * 1e3
    return out


def make_server(q):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            path = self.path.rstrip("/")
            if path == "/shutdown":
                q.put(Item(SHUTDOWN))
                self._reply(200, {"ok": True})
                return
            if path not in ("/e0", "/e1"):
                self._reply(404, {"error": f"unknown event {path}"})
                return
            item = Item(int(path[-1]))
            q.put(item)
            item.done.wait()
            self._reply(200, item.reply)

        def _reply(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ThreadingHTTPServer(("0.0.0.0", PORT), Handler)


def event_sequence():
    if PATTERN == "alternate":
        return [i % 2 for i in range(EVENTS)]
    if PATTERN == "burst":
        return [(i // BURST) % 2 for i in range(EVENTS)]
    if PATTERN == "random":
        rng = random.Random(7)
        return [rng.randint(0, 1) for _ in range(EVENTS)]
    raise SystemExit(f"unknown EVENT_PATTERN {PATTERN!r}")


def generator():
    for app in event_sequence():
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/e{app}", method="POST")
        urllib.request.urlopen(req, timeout=1800).read()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/shutdown", method="POST")
    urllib.request.urlopen(req, timeout=60).read()


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def summarize(records):
    keys = ["queue_ms", "e2e_ms"]
    for name, _a, _b in STARTUP:
        keys += [f"{name}_ms", f"{name}_max_ms"]
    keys += ["slowest_wait_ms", "snapshot_ms", "exit_ms", "device_busy_ms"]
    print(f"\n=== process-per-request dispatch: {len(records)} requests, "
          f"{NPROC} ranks/app, pattern {PATTERN} ===")
    for app in (0, 1):
        rs = [r for r in records if r["app"] == app]
        if not rs:
            continue
        print(f"app{app}: {len(rs)} requests")
        for k in keys:
            xs = [r[k] for r in rs]
            print(f"  {k:15s} p50 {pct(xs, 50):9.1f}  p90 {pct(xs, 90):9.1f}  "
                  f"max {max(xs):9.1f} ms")
    e2e = statistics.median(r["e2e_ms"] for r in records)
    busy = statistics.median(r["device_busy_ms"] for r in records)
    run = statistics.median(r["run_ms"] for r in records)
    print(f"median request: user waits {e2e:.0f} ms (of which run {run:.1f} ms); "
          f"device held {busy:.0f} ms per request")
    counts = {}
    for r in records:
        counts[r["slowest_rank"]] = counts.get(r["slowest_rank"], 0) + 1
    print("slowest rank per request (rank: times): "
          + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))


def main():
    print(f"prewarm: compiling both apps and creating their state ({NPROC} ranks each)",
          flush=True)
    for app in (0, 1):
        rec = run_app(app, prewarm=True)
        print(f"prewarm app{app}: {phases(rec)['device_busy_ms'] / 1e3:.1f} s", flush=True)

    q = queue.Queue()
    srv = make_server(q)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"serving on :{PORT} (source={SOURCE}, pattern={PATTERN}, events={EVENTS})",
          flush=True)
    if SOURCE == "self":
        threading.Thread(target=generator, daemon=True).start()

    records = []
    while True:
        item = q.get()
        if item.app == SHUTDOWN:
            break
        queue_ms = (time.time() - item.t_arrival) * 1e3

        def on_result(rec, item=item, queue_ms=queue_ms):
            e2e_ms = (rec["t_reply"] - item.t_arrival) * 1e3
            item.reply = {"app": item.app, "result": rec["result"],
                          "queue_ms": round(queue_ms, 1), "e2e_ms": round(e2e_ms, 1)}
            item.done.set()

        rec = run_app(item.app, on_result=on_result)
        row = {"app": item.app, "queue_ms": queue_ms,
               "e2e_ms": (rec["t_reply"] - item.t_arrival) * 1e3, **phases(rec)}
        records.append(row)
        print(f"request {len(records)}: app{item.app}  user waited {row['e2e_ms']:.0f} ms  "
              f"(queue {queue_ms:.0f}, launch {row['launch_ms']:.0f}, "
              f"torch {row['torch_import_ms']:.0f}, neuronx {row['neuronx_import_ms']:.0f}, "
              f"gloo {row['gloo_init_ms']:.0f}, runtime {row['runtime_init_ms']:.0f}, "
              f"load {row['load_ms']:.0f}, restore {row['restore_ms']:.1f}, "
              f"run {row['run_ms']:.1f}, slowest rank {row['slowest_rank']} "
              f"+{row['slowest_wait_ms']:.0f})  then snapshot {row['snapshot_ms']:.1f}, "
              f"exit {row['exit_ms']:.0f}", flush=True)

    srv.shutdown()
    with open(STAT, "w") as f:
        json.dump(records, f, indent=2)
    print(f"wrote {STAT}")
    if records:
        summarize(records)


if __name__ == "__main__":
    main()
