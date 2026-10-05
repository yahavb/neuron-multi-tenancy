# Two-Container Approach (deferred)

Status: **not implemented** — the PoC uses the one-container dispatcher
(`tenant_switch.py`). This document records the two-container design so we can
return to it once the one-container PoC proves snapshot/restore cost is
acceptable. The two-container version is the honest multi-tenancy demo: the
eventual dit and unrolling workloads are separate processes owned by different
teams and will not share a container.

## Goal

One k8s Job, one pod, two containers `c_0` (dit stand-in) and `c_1` (unrolling
stand-in), both referencing the **same** device-level ResourceClaim (1 ND,
LNC=2, 4 logical cores — `k8s/s-lnc2-rct.yaml`). The containers time-share the
device: only one runs on the NeuronCores at any moment, coordinated through a
lock on a shared volume. This gives MIG-like tenancy by time-slicing instead of
space-slicing.

## Why it is harder than the one-container version

NRT grants a process exclusive ownership of its NeuronCores from `nrt_init`
until `nrt_close`/process exit. Two processes cannot hold the same core, so a
context switch between containers necessarily includes:

1. c_0 snapshots its tensors HBM → host (ms)
2. c_0 tears down the runtime — `nrt_close` or full process exit (seconds)
3. c_1 runs `nrt_init` + loads its NEFFs (seconds)
4. c_1 restores tensors host → HBM (ms)

Steps 2–3 dominate. In the one-container version both tenants live in one
runtime so those steps disappear; that asymmetry is exactly what the two
experiments are meant to quantify against each other.

## Pod spec sketch

```yaml
spec:
  template:
    spec:
      resourceClaims:
        - name: neurons
          resourceClaimTemplateName: s-lnc2
      containers:
        - name: c0            # dit stand-in
          resources: { claims: [ { name: neurons } ] }
          volumeMounts: [ { name: handoff, mountPath: /handoff } ]
        - name: c1            # unrolling stand-in
          resources: { claims: [ { name: neurons } ] }
          volumeMounts: [ { name: handoff, mountPath: /handoff } ]
      volumes:
        - name: handoff
          emptyDir: {}
```

Both containers claim the same `neurons` claim so both see the same
`/dev/neuron*`; both set `NEURON_RT_VISIBLE_CORES=0-3` and run
`torchrun --standalone --nproc-per-node 4`. Both images must carry the same
Neuron SDK version (they share the node's driver).

## Coordination: flock + turn file on the shared emptyDir

Use `flock(2)` on `/handoff/device.lock` as the mutual-exclusion primitive.
It is kernel-released when the holder dies — a crashed container cannot
deadlock the job the way a stale sentinel file would.

Bare flock does not guarantee alternation (the releasing process can
immediately re-acquire). Add a turn file for deterministic round-robin:

```bash
# each container, with ME=c0 PEER=c1 (or vice versa)
touch "/handoff/$ME.started"
for i in $(seq 1 "$SWITCHES"); do
  until [ "$(cat /handoff/turn 2>/dev/null || echo c0)" = "$ME" ]; do sleep 0.05; done
  (
    flock -x 9
    run_residency        # torchrun: restore -> ITERS steps -> snapshot -> exit
    echo "$PEER" > /handoff/turn
  ) 9>/handoff/device.lock
done
touch "/handoff/$ME.done"
```

Startup determinism: do **not** use a sleep in c_1. c_0 owns the first turn
(`turn` file absent defaults to `c0`); c_1 blocks on the turn file. If stricter
sequencing is needed, c_1 additionally waits for `/handoff/c0.started`.

Crash handling: if a container exits non-zero, write `/handoff/$ME.failed` in a
trap and have the peer treat it as terminal, otherwise the survivor spins on
the turn file until `activeDeadlineSeconds`.

## What each residency does

Same workload as the one-container PoC (`tenant_switch.py` step: transposes +
matmul, state evolved each step), but one tenant per container and the
process **fully exits** at the end of each residency, persisting its snapshot
to `/handoff/$ME.state.pt` (or `/dev/shm` for a cheaper copy). On the next
turn it restarts, reloads the snapshot, restores to HBM, continues.

## What to measure (per residency, per container)

- `t_snapshot`: HBM → host tensor copy
- `t_persist`: host → shared volume (skip if snapshotting straight to /dev/shm)
- `t_exit` + `t_init`: process teardown and `nrt_init` of the successor
  (measure as wall gap between c_0's "released" timestamp and c_1's
  "first step" timestamp, both logged to /handoff)
- `t_load`: NEFF load (keep the Neuron persistent cache on a shared volume so
  compilation happens once, ever)
- `t_restore`: host → HBM

Compare the total against the one-container switch cost from
`switch_report.py`. Expectation: copies are comparable; the two-container
switch adds seconds of runtime teardown/init/NEFF-load that the one-container
version doesn't pay.

## Open questions for the formal version

- Can `nrt_close`/re-`nrt_init` within a long-lived process replace full
  process exit? That would cut the exec/import cost and keep torchrun alive.
- Preemption: current design is cooperative (switch only at residency
  boundaries). A real scheduler needs a maximum residency and a way to ask the
  tenant to yield.
- Lock service: file lock works within one pod. Across pods on the same node
  it needs a hostPath volume or a per-node coordinator (daemonset) — that is
  the piece to formalize if the PoC numbers justify it.
