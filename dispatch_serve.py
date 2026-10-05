#!/usr/bin/env python3
"""Event-driven model dispatch on one shared Neuron device.

Preface for the real dit + unrolling pair: two models (t0_dit, t1_unroll)
time-share the device, and work is triggered by events instead of a fixed
alternation. Rank 0 runs an HTTP server; POST /e0 dispatches model m0,
POST /e1 dispatches m1. The event id is broadcast to all ranks (gloo), every
rank swaps the target model into HBM if needed, runs STEPS_PER_EVENT steps,
and per DISPATCH_POLICY writes the model state back to host memory:

  eager (default): reply to the user, then snapshot the model to host, so its
      HBM is free before the next event. The snapshot is outside the user's
      wait but the next event waits for it.
  lazy: leave the model resident; evict only when the *other* model is
      requested. Consecutive same-model events skip both copies.
  resident: both models stay in HBM for the whole run; no copies at all.
      Only possible when both fit.

Per event, per rank we record: queue wait, swap-out (eviction), swap-in
(restore), run time, and on rank 0 the end-to-end latency the HTTP caller saw.

EVENT_SOURCE=self (default) starts a generator thread on rank 0 that drives
EVENTS requests through the real HTTP path in EVENT_PATTERN order
(alternate | burst | random), then shuts the server down. EVENT_SOURCE=external
leaves the server waiting for outside callers until POST /shutdown.

Run under torchrun, one rank per logical NeuronCore (4 for 1 ND at LNC=2).
"""

import json
import os
import queue
import random
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Import order matters: tenant_switch pins NEURON_RT_VISIBLE_CORES per rank
# before torch_neuronx initializes NRT.
from tenant_switch import LOCAL_RANK, RANK, T0_DIM, T1_DIM, Tenant, median, work

import torch
import torch.distributed as dist
import torch_neuronx

EVENTS = int(os.environ.get("EVENTS", "100"))
PATTERN = os.environ.get("EVENT_PATTERN", "alternate")   # alternate|burst|random
BURST = int(os.environ.get("EVENT_BURST", "8"))
STEPS_PER_EVENT = int(os.environ.get("STEPS_PER_EVENT", "10"))
POLICY = os.environ.get("DISPATCH_POLICY", "eager")      # eager|lazy
PORT = int(os.environ.get("DISPATCH_PORT", "8080"))
SOURCE = os.environ.get("EVENT_SOURCE", "self")          # self|external
STAT = os.environ.get("MT_STAT", "/tmp/dispatch_stat")

SHUTDOWN = -1


class Item:
    def __init__(self, model_id):
        self.model_id = model_id
        self.t_arrival = time.perf_counter()
        self.done = threading.Event()
        self.reply = {}


class Dispatcher:
    """Owns the two models and the device-residency policy on this rank."""

    def __init__(self, device):
        step_fn = torch.compile(work, backend="neuron", dynamic=False)
        self.models = [
            Tenant("t0_dit", T0_DIM, device, step_fn, seed=11),
            Tenant("t1_unroll", T1_DIM, device, step_fn, seed=23),
        ]
        self.resident = None
        self.records = []
        # Warm both graphs so compile never lands inside an event.
        for m in self.models:
            m.restore()
            m.step()
            m.step()
            m.snapshot()
        self.hbm_after_warmup = hbm_bytes()
        if POLICY == "resident":
            for m in self.models:
                m.restore()
        self.hbm_ready = hbm_bytes()

    def handle(self, mid):
        """Everything the user waits for: make the model resident, run it."""
        t0 = time.perf_counter()
        m = self.models[mid]
        hit = m.dev is not None
        swap_out = swap_in = 0.0

        if POLICY != "resident" and self.resident is not None and self.resident != mid:
            swap_out = self.models[self.resident].snapshot()
            self.resident = None
        if m.dev is None:
            swap_in = m.restore()
        self.resident = mid

        t_run = time.perf_counter()
        for _ in range(STEPS_PER_EVENT):
            m.step()
        run = time.perf_counter() - t_run

        rec = {
            "model": mid,
            "hit": hit,
            "swap_out_ms": swap_out * 1e3,
            "swap_in_ms": swap_in * 1e3,
            "run_ms": run * 1e3,
            "total_ms": (time.perf_counter() - t0) * 1e3,
            "after_reply_ms": 0.0,
        }
        self.records.append(rec)
        return rec

    def after_reply(self):
        """Work done after the user has the result; the next event waits for it."""
        if POLICY == "eager" and self.resident is not None:
            t = self.models[self.resident].snapshot()
            self.resident = None
            self.records[-1]["after_reply_ms"] = t * 1e3
        self.records[-1]["hbm_bytes"] = hbm_bytes()


_HBM_WARNED = False


def _try_hbm(fns):
    global _HBM_WARNED
    errors = []
    for label, fn in fns:
        try:
            return int(fn())
        except Exception as e:
            errors.append(f"{label}: {type(e).__name__}: {e}")
    if RANK == 0 and not _HBM_WARNED:
        _HBM_WARNED = True
        names = [n for n in dir(torch_neuronx) if "mem" in n.lower()]
        print("[rank 0] HBM stats unavailable: " + " | ".join(errors)
              + f" | torch_neuronx memory names: {names}", flush=True)
    return -1


def hbm_bytes():
    """Bytes held by this core's caching allocator (tensors + cached blocks)."""
    return _try_hbm([
        ("torch_neuronx.memory_allocated", lambda: torch_neuronx.memory_allocated()),
        ("torch.neuron.memory_allocated", lambda: torch.neuron.memory_allocated()),
        ("memory_stats", lambda: torch_neuronx.memory_stats()["allocated_bytes"]["current"]),
    ])


def hbm_peak_bytes():
    return _try_hbm([
        ("torch_neuronx.max_memory_allocated", lambda: torch_neuronx.max_memory_allocated()),
        ("torch.neuron.max_memory_allocated", lambda: torch.neuron.max_memory_allocated()),
    ])


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
    rng = random.Random(7)
    if PATTERN == "alternate":
        return [i % 2 for i in range(EVENTS)]
    if PATTERN == "burst":
        return [(i // BURST) % 2 for i in range(EVENTS)]
    if PATTERN == "random":
        return [rng.randint(0, 1) for _ in range(EVENTS)]
    raise SystemExit(f"unknown EVENT_PATTERN {PATTERN!r}")


def generator():
    for mid in event_sequence():
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/e{mid}", method="POST")
        urllib.request.urlopen(req, timeout=600).read()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/shutdown", method="POST")
    urllib.request.urlopen(req, timeout=60).read()


def main():
    dist.init_process_group("gloo")
    device = torch.device("neuron")
    d = Dispatcher(device)
    hbm = f"{d.hbm_ready / 2**20:.0f} MiB" if d.hbm_ready >= 0 else "unknown"
    print(f"[rank {RANK}] warmup done, policy={POLICY}, HBM in use {hbm}", flush=True)
    dist.barrier()

    q = None
    e2e = []
    if RANK == 0:
        q = queue.Queue()
        srv = make_server(q)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print(f"[rank 0] serving on :{PORT} (source={SOURCE}, pattern={PATTERN}, "
              f"events={EVENTS}, steps/event={STEPS_PER_EVENT})", flush=True)
        if SOURCE == "self":
            threading.Thread(target=generator, daemon=True).start()

    evt = torch.zeros(1, dtype=torch.long)
    served = 0
    while True:
        item = None
        if RANK == 0:
            item = q.get()
            evt[0] = item.model_id
        dist.broadcast(evt, src=0)
        mid = int(evt.item())
        if mid == SHUTDOWN:
            break

        queue_ms = (time.perf_counter() - item.t_arrival) * 1e3 if item else 0.0
        rec = d.handle(mid)
        dist.barrier()  # event is served when every core is done

        served += 1
        if RANK == 0:
            e2e_ms = (time.perf_counter() - item.t_arrival) * 1e3
            e2e.append({"model": mid, "queue_ms": queue_ms, "e2e_ms": e2e_ms})
            item.reply = {"model": f"m{mid}", "hit": rec["hit"],
                          "queue_ms": round(queue_ms, 3),
                          "swap_out_ms": round(rec["swap_out_ms"], 3),
                          "swap_in_ms": round(rec["swap_in_ms"], 3),
                          "run_ms": round(rec["run_ms"], 3),
                          "e2e_ms": round(e2e_ms, 3)}
            item.done.set()
            if served % 50 == 0:
                print(f"[rank 0] served {served} events, recent e2e median "
                      f"{median([x['e2e_ms'] for x in e2e[-50:]]):.2f} ms", flush=True)
        d.after_reply()

    out = {
        "rank": RANK,
        "policy": POLICY,
        "pattern": PATTERN,
        "events": served,
        "steps_per_event": STEPS_PER_EVENT,
        "records": d.records,
        "e2e": e2e if RANK == 0 else None,
        "hbm_after_warmup": d.hbm_after_warmup,
        "hbm_ready": d.hbm_ready,
        "hbm_peak": hbm_peak_bytes(),
    }
    path = f"{STAT}.rank{RANK}.json"
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"[rank {RANK}] wrote {path}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
