#!/usr/bin/env python3
"""One-container Neuron multi-tenancy PoC.

Two synthetic tenants share the same NeuronCores inside one torchrun process
group: t0 stands in for the dit workload, t1 for the unrolling workload.
Only one tenant's working set lives in HBM at a time. A context switch
snapshots the outgoing tenant's tensors to host memory and restores the
incoming tenant's tensors to HBM. Each tenant uses a different matrix size so
the two workloads compile to distinct graphs.

Per residency the active tenant runs ITERS steps (transposes + matmul that
evolve its state), then we switch. We measure step time, snapshot time,
restore time, and the full switch cost, per rank.

Run under torchrun: one rank per logical NeuronCore (4 ranks for 1 device at
LNC=2). Env knobs: T0_DIM, T1_DIM, ITERS, SWITCHES, MT_STAT.
"""

import json
import os
import time

import torch
import torch_xla.core.xla_model as xm

RANK = int(os.environ.get("RANK", "0"))
T0_DIM = int(os.environ.get("T0_DIM", "2048"))
T1_DIM = int(os.environ.get("T1_DIM", "1536"))
ITERS = int(os.environ.get("ITERS", "10"))        # steps per residency
SWITCHES = int(os.environ.get("SWITCHES", "6"))   # number of context switches
STAT = os.environ.get("MT_STAT", "/tmp/mt_stat")


def sync():
    xm.mark_step()
    xm.wait_device_ops()


class Tenant:
    """A tenant is a named working set (m1, m2) plus the op graph that evolves it."""

    def __init__(self, name, dim, device, seed):
        self.name = name
        self.dim = dim
        self.device = device
        g = torch.Generator().manual_seed(seed + RANK)
        # State is born on host; restore() moves it to HBM.
        self.host = {
            "m1": torch.randn(dim, dim, generator=g, dtype=torch.float32),
            "m2": torch.randn(dim, dim, generator=g, dtype=torch.float32),
        }
        self.dev = None

    def restore(self):
        """Host -> HBM. Returns seconds."""
        t0 = time.perf_counter()
        self.dev = {k: v.to(self.device) for k, v in self.host.items()}
        sync()
        return time.perf_counter() - t0

    def snapshot(self):
        """HBM -> host, drop device references. Returns seconds."""
        t0 = time.perf_counter()
        self.host = {k: v.cpu() for k, v in self.dev.items()}
        self.dev = None
        sync()
        return time.perf_counter() - t0

    def step(self):
        """A few transposes and a matmul; result folded back into state so the
        snapshot carries real progress."""
        m1, m2 = self.dev["m1"], self.dev["m2"]
        a = m1.t().contiguous()
        b = m2.t().contiguous()
        r = torch.matmul(a, b).t().contiguous()
        self.dev["m1"] = r / (r.norm() + 1e-6) * m1.norm()
        sync()

    def nbytes(self):
        return sum(v.numel() * v.element_size() for v in self.host.values())


def median(xs):
    return sorted(xs)[len(xs) // 2] if xs else 0.0


def main():
    device = xm.xla_device()
    tenants = [
        Tenant("t0_dit", T0_DIM, device, seed=11),
        Tenant("t1_unroll", T1_DIM, device, seed=23),
    ]

    # Warm up both graphs so compilation never pollutes the measurements.
    for t in tenants:
        t.restore()
        t.step()
        t.snapshot()

    stats = {
        t.name: {"step_ms": [], "snapshot_ms": [], "restore_ms": [], "bytes": t.nbytes()}
        for t in tenants
    }
    switch_ms = []

    active = 0
    stats[tenants[active].name]["restore_ms"].append(tenants[active].restore() * 1e3)

    for _ in range(SWITCHES):
        t = tenants[active]
        for _ in range(ITERS):
            t0 = time.perf_counter()
            t.step()
            stats[t.name]["step_ms"].append((time.perf_counter() - t0) * 1e3)

        t0 = time.perf_counter()
        snap = t.snapshot()
        active ^= 1
        rest = tenants[active].restore()
        switch_ms.append((time.perf_counter() - t0) * 1e3)
        stats[t.name]["snapshot_ms"].append(snap * 1e3)
        stats[tenants[active].name]["restore_ms"].append(rest * 1e3)

    out = {
        "rank": RANK,
        "t0_dim": T0_DIM,
        "t1_dim": T1_DIM,
        "iters": ITERS,
        "switches": SWITCHES,
        "switch_ms": switch_ms,
        "tenants": stats,
    }
    path = f"{STAT}.rank{RANK}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[rank {RANK}] wrote {path}")

    if RANK == 0:
        for name, s in stats.items():
            print(
                f"[rank 0] {name}: step {median(s['step_ms']):.2f} ms  "
                f"snapshot {median(s['snapshot_ms']):.2f} ms  "
                f"restore {median(s['restore_ms']):.2f} ms  "
                f"state {s['bytes'] / 2**20:.0f} MiB"
            )
        print(f"[rank 0] context switch (snapshot+restore): median {median(switch_ms):.2f} ms")


if __name__ == "__main__":
    main()
