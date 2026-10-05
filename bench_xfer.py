#!/usr/bin/env python3
"""Core-to-core tensor transfer: bounce through host DRAM vs direct HBM->HBM.

Moves one fp32 tensor from NeuronCore A (rank 0) to NeuronCore B (rank 1) on
the same device, at sizes SIZES_MIB, with the two options available today:

  host  rank 0 copies HBM -> /dev/shm-backed host buffer (D2H), the ranks
        hand off with a gloo barrier, rank 1 copies host -> HBM (H2D).
        Best case of the DRAM path: one D2H + one H2D, no extra host copy.
        Reports d2h / handoff / h2d separately.
  hbm   dist.broadcast(src=0) on the "neuron" process group: device fabric,
        never touches host memory. Falls back to all_gather (which also
        moves rank 1 -> rank 0, so it over-counts) if this torch-neuronx
        build has no broadcast; the primitive used is reported.

Each size is verified once before timing: the receiver's buffer is zeroed,
one transfer runs, and the received checksum must match the sender's.

    torchrun --standalone --nproc-per-node 2 bench_xfer.py --mode host
    torchrun --standalone --nproc-per-node 2 bench_xfer.py --mode hbm

Env: SIZES_MIB (default 1,4,16,64,256,1024), XFER_ITERS (20), XFER_STAT.
"""

import json
import os
import statistics
import sys
import time

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
RANK = int(os.environ.get("RANK", str(LOCAL_RANK)))
WORLD = int(os.environ.get("WORLD_SIZE", "1"))

_A = sys.argv[1:]
MODE = next((_A[i + 1] for i, v in enumerate(_A) if v == "--mode" and i + 1 < len(_A)), "host")
if MODE not in ("host", "hbm"):
    raise SystemExit("--mode must be host | hbm")

SIZES_MIB = [int(s) for s in os.environ.get("SIZES_MIB", "1,4,16,64,256,1024").split(",")]
ITERS = int(os.environ.get("XFER_ITERS", "20"))
STAT = os.environ.get("XFER_STAT", "/tmp/xfer_stat")
SHM = "/dev/shm/xfer_buf.npy"

# host mode pins one core per rank (same as tenant_switch.py); the neuron
# backend assigns cores itself via set_device (same as bench_assembly.py).
if MODE == "host":
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(LOCAL_RANK)
    os.environ["NEURON_RT_NUM_CORES"] = "1"
elif not os.environ.get("NEURON_RT_ROOT_COMM_ID"):
    os.environ["NEURON_RT_ROOT_COMM_ID"] = "%s:%d" % (
        os.environ.get("MASTER_ADDR", "127.0.0.1"),
        int(os.environ.get("MASTER_PORT", "29500")) + 1)

import numpy as np
import torch
import torch.distributed as dist
import torch_neuronx

DEV = torch.device("neuron")


def log(m):
    if RANK == 0:
        print(m, flush=True)


def land(t):
    t.reshape(-1)[:1].cpu()


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def checksum(t):
    return float(t.double().sum().item())


class HostPath:
    def __init__(self, host_pg):
        self.pg = host_pg
        self.primitive = "d2h+shm+h2d"

    def setup(self, n):
        if RANK == 0:
            mm = np.lib.format.open_memmap(SHM, mode="w+", dtype=np.float32, shape=(n,))
            mm.flush()
            del mm
        dist.barrier(group=self.pg)
        self.buf = torch.from_numpy(np.lib.format.open_memmap(SHM, mode="r+"))

    def teardown(self):
        self.buf = None
        dist.barrier(group=self.pg)
        if RANK == 0 and os.path.exists(SHM):
            os.remove(SHM)

    def once(self, src):
        """Returns (received device tensor or None, phase timings in ms)."""
        t0 = time.perf_counter()
        if RANK == 0:
            self.buf.copy_(src)
        t1 = time.perf_counter()
        dist.barrier(group=self.pg)
        t2 = time.perf_counter()
        out = None
        if RANK == 1:
            out = self.buf.to(DEV)
            land(out)
        t3 = time.perf_counter()
        return out, {"d2h_ms": (t1 - t0) * 1e3, "handoff_ms": (t2 - t1) * 1e3,
                     "h2d_ms": (t3 - t2) * 1e3}


class HbmPath:
    def __init__(self, host_pg):
        self.pg = host_pg
        self.stream = torch_neuronx.Stream(DEV)
        self.primitive = "broadcast"

    def setup(self, n):
        self.dst = torch.zeros(n, dtype=torch.float32).to(DEV)
        land(self.dst)

    def teardown(self):
        self.dst = None

    def once(self, src):
        t0 = time.perf_counter()
        if self.primitive == "broadcast":
            t = src if RANK == 0 else self.dst
            try:
                with torch_neuronx.stream(self.stream):
                    dist.broadcast(t, src=0)
                torch_neuronx.synchronize()
            except NotImplementedError as e:
                log(f"  broadcast unavailable ({type(e).__name__}: {e}); falling back to all_gather")
                self.primitive = "all_gather"
                return self.once(src)
            out = self.dst if RANK == 1 else None
        else:
            mine = src if RANK == 0 else self.dst
            bucket = [torch.empty_like(mine) for _ in range(WORLD)]
            with torch_neuronx.stream(self.stream):
                dist.all_gather(bucket, mine)
            torch_neuronx.synchronize()
            out = bucket[0] if RANK == 1 else None
        if out is not None:
            land(out)
        return out, {"xfer_ms": (time.perf_counter() - t0) * 1e3}


def main():
    if WORLD != 2:
        raise SystemExit("run with --nproc-per-node 2 (one sender, one receiver)")

    if MODE == "hbm":
        dist.init_process_group(backend="neuron")
        if hasattr(torch, "neuron") and hasattr(torch.neuron, "set_device"):
            torch.neuron.set_device(LOCAL_RANK)
        host_pg = dist.new_group(backend="gloo")
        path = HbmPath(host_pg)
    else:
        dist.init_process_group(backend="gloo")
        host_pg = dist.group.WORLD
        torch_neuronx._lazy_init()
        path = HostPath(host_pg)

    log(f"=== core-to-core transfer, mode {MODE}, sizes {SIZES_MIB} MiB, {ITERS} iters ===")
    results = []
    for mib in SIZES_MIB:
        n = mib * 2**20 // 4
        src = None
        if RANK == 0:
            src = (torch.arange(n, dtype=torch.float32) % 977).to(DEV)
            land(src)
        path.setup(n)

        out, _ = path.once(src)
        ok = torch.tensor([1.0])
        if RANK == 1:
            ok[0] = float(abs(checksum(out.cpu()) - checksum(torch.arange(n, dtype=torch.float32) % 977)) < 1e-3)
        dist.broadcast(ok, src=1, group=host_pg)
        verified = bool(ok.item())

        for _ in range(3):
            path.once(src)

        totals, phases = [], []
        for _ in range(ITERS):
            dist.barrier(group=host_pg)
            t0 = time.perf_counter()
            _, ph = path.once(src)
            el = torch.tensor([(time.perf_counter() - t0) * 1e3])
            dist.all_reduce(el, op=dist.ReduceOp.MAX, group=host_pg)
            totals.append(float(el.item()))
            phases.append(ph)

        # Phase times live on the rank that does them: d2h on rank 0, h2d on rank 1.
        # Handoff is taken from rank 0: rank 1 reaches the barrier first and its
        # wait there includes rank 0's whole D2H.
        ph_all = [None, None]
        dist.all_gather_object(ph_all, phases, group=host_pg)
        phase_p50 = {}
        if MODE == "host":
            phase_p50 = {
                "d2h_ms": statistics.median(p["d2h_ms"] for p in ph_all[0]),
                "handoff_ms": statistics.median(p["handoff_ms"] for p in ph_all[0]),
                "h2d_ms": statistics.median(p["h2d_ms"] for p in ph_all[1]),
            }

        p50 = pct(totals, 50)
        row = {
            "mode": MODE,
            "primitive": path.primitive,
            "mib": mib,
            "verified": verified,
            "p50_ms": p50,
            "p90_ms": pct(totals, 90),
            "min_ms": min(totals),
            "max_ms": max(totals),
            "gib_s": (mib / 1024) / (p50 / 1e3),
            **phase_p50,
        }
        results.append(row)
        extra = "  ".join(f"{k} {v:.2f}" for k, v in phase_p50.items())
        log(f"  {mib:5d} MiB  {path.primitive:12s} p50 {p50:9.3f} ms  p90 {row['p90_ms']:9.3f}  "
            f"{row['gib_s']:6.2f} GiB/s  verify {'OK' if verified else 'FAILED'}  {extra}")
        path.teardown()
        src = None

    if RANK == 0:
        with open(f"{STAT}.{MODE}.json", "w") as f:
            json.dump(results, f, indent=2)
        log(f"wrote {STAT}.{MODE}.json")
    dist.barrier(group=host_pg)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
