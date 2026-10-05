#!/usr/bin/env python3
"""One request's worth of work for one app, run as its own torchrun job.

Launched by dispatch_proc.py once per request. Each rank owns one NeuronCore
for the life of this process; the cores are released only when the process
exits (torch-neuronx calls nrt_close() from atexit and cannot re-init).

Sequence per rank:
  start -> import torch/torch_neuronx + NRT init -> load the compiled graph
  (warm-up step on scratch tensors, so the app's state is not touched)
  -> restore state from host (/dev/shm) to HBM -> run STEPS steps
  -> barrier, rank 0 prints MT_RESULT (dispatcher replies to the user)
  -> snapshot state HBM -> host (/dev/shm) -> barrier, rank 0 prints MT_DONE
  -> exit, releasing the cores.

State lives in preallocated /dev/shm .npy files so it outlives the process and
the D2H copy lands in an existing buffer (Run 6: ~2x faster than .cpu()).
First run of an app (no state files) creates random state.

    torchrun --standalone --nproc-per-node N app_run.py --app 0|1 [--prewarm]
"""

import json
import os
import sys
import time

T_START = time.time()

# Pins NEURON_RT_VISIBLE_CORES=LOCAL_RANK before torch_neuronx initializes NRT.
from tenant_switch import LOCAL_RANK, RANK, land, work

import numpy as np
import torch
import torch.distributed as dist
import torch_neuronx

_A = sys.argv[1:]
APP = int(next((_A[i + 1] for i, v in enumerate(_A) if v == "--app" and i + 1 < len(_A)), "0"))
PREWARM = "--prewarm" in _A
DIM = int(os.environ.get("T0_DIM" if APP == 0 else "T1_DIM", "2048" if APP == 0 else "1536"))
STEPS = int(os.environ.get("STEPS_PER_EVENT", "10"))
STATE_DIR = os.environ.get("MT_STATE_DIR", "/dev/shm/mt_state")
NAMES = ("m1", "m2")


def state_path(name):
    return os.path.join(STATE_DIR, f"app{APP}.rank{RANK}.{name}.npy")


def open_state():
    """Host-side state buffers (memmaps). Creates them on the app's first run."""
    os.makedirs(STATE_DIR, exist_ok=True)
    bufs, created = {}, False
    g = torch.Generator().manual_seed(11 + 12 * APP + RANK)
    for name in NAMES:
        p = state_path(name)
        if not os.path.exists(p):
            mm = np.lib.format.open_memmap(p, mode="w+", dtype=np.float32, shape=(DIM, DIM))
            mm[:] = torch.randn(DIM, DIM, generator=g).numpy()
            mm.flush()
            del mm
            created = True
        bufs[name] = torch.from_numpy(np.lib.format.open_memmap(p, mode="r+"))
    return bufs, created


def main():
    dist.init_process_group("gloo")
    dev = torch.device("neuron")
    torch_neuronx._lazy_init()
    land(torch.zeros(1).to(dev))
    t_init = time.time()

    step_fn = torch.compile(work, backend="neuron", dynamic=False)
    scratch = torch.zeros(DIM, DIM).to(dev)
    with torch.no_grad():
        land(step_fn(scratch, scratch))
    scratch = None
    t_load = time.time()

    host, created = open_state()
    state = {k: v.to(dev) for k, v in host.items()}
    for v in state.values():
        land(v)
    t_restore = time.time()

    with torch.no_grad():
        for _ in range(STEPS):
            state["m1"] = step_fn(state["m1"], state["m2"])
            land(state["m1"])
    t_run = time.time()

    result = float(state["m1"].reshape(-1)[:1024].cpu().double().sum())
    dist.barrier()
    if RANK == 0:
        print("MT_RESULT " + json.dumps({
            "app": APP, "prewarm": PREWARM, "created_state": created, "result": result,
            "t_start": T_START, "t_init": t_init, "t_load": t_load,
            "t_restore": t_restore, "t_run": t_run, "t_result": time.time(),
        }), flush=True)

    for k, v in state.items():
        host[k].copy_(v)
    state = None
    t_snap = time.time()
    dist.barrier()
    if RANK == 0:
        print("MT_DONE " + json.dumps({"app": APP, "t_snapshot": t_snap}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
