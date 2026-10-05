# neuron-multi-tenancy

PoC for time-slicing one Neuron device (ND) between two workloads — MIG-like
tenancy via context switching instead of hardware partitioning. The eventual
tenants are a dit process and an unrolling process; until the concept is
proven the tenants are synthetic (transposes + matmul over per-rank m1/m2
tensors, state evolved every step so the snapshot carries real progress).

A context switch = snapshot the outgoing tenant's HBM tensors to host memory,
restore the incoming tenant's tensors from host to HBM. The PoC measures what
that switch costs relative to a work step.

## Approach implemented: one-container dispatcher

Both tenants live in one torchrun process group (4 ranks = 1 ND at LNC=2 on
trn2). Both graphs stay loaded; only the active tenant's working set occupies
HBM. Because everything is one runtime, the switch is pure tensor movement —
no `nrt_init`/NEFF-load on the switch path.

- `tenant_switch.py` — the torchrun app: warm-up both graphs, then alternate
  residencies (`ITERS` steps, snapshot, restore peer) for `SWITCHES` switches;
  writes per-rank timing JSON.
- `switch_report.py` — aggregates rank JSONs: median step / snapshot / restore
  / switch times, copy bandwidth, switch cost as a multiple of one step.
- `k8s/neuron-mt-poc-job.yaml` — the Job: clones this repo at load time, runs
  torchrun, archives stats to the PVC. Uses the DRA claim below.
- `k8s/s-lnc2-rct.yaml` — ResourceClaimTemplate: exactly 1 neuron device,
  LNC=2 (trn2/trn3).

### Run

```bash
kubectl apply -f k8s/s-lnc2-rct.yaml
kubectl apply -f k8s/neuron-mt-poc-job.yaml
kubectl logs -f job/neuron-mt-poc
```

Knobs (env on the job): `T0_DIM` (default 2048), `T1_DIM` (1536) — distinct so
the tenants compile distinct graphs; `ITERS` steps per residency (10);
`SWITCHES` (6).

## Approach deferred: two containers, one device

The honest multi-tenancy demo — two containers sharing one ResourceClaim,
coordinated by flock + turn file on a shared emptyDir, with full runtime
teardown between residencies (NRT core ownership is exclusive per process).
Design, pod sketch, locking protocol, and measurement plan are in
[docs/two-container-approach.md](docs/two-container-approach.md).
