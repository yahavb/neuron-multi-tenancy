# neuron-multi-tenancy

PoC for letting two workloads share one Neuron device (ND) — MIG-like
tenancy by time-sharing instead of hardware partitioning. The eventual
workloads are dit and unrolling for prime-video; until the concept is proven
they are synthetic apps (transposes + matmul over per-rank m1/m2 tensors,
state evolved every step so a snapshot carries real progress).

**Start with [docs/PROJECT-LOG.md](docs/PROJECT-LOG.md)**: current decision,
every run's results, and open items.

## Where it landed

One long-running server process owns all of the device's cores and holds
both models. HTTP requests (`POST /e0`, `POST /e1`) pick the model. If both
models fit in HBM, both stay resident: about 0.6 ms of overhead per request.
If not, the idle model's state is swapped to host memory in the same process:
about 29 ms per request for the synthetic apps, growing with weight size
(about 60 ms per GB of weights on the device, per direction). Separate
processes per app cost about 18.6 s per switch, because the Neuron runtime
frees cores only at process exit.

## Files

- `dispatch_serve.py` + `dispatch_report.py` — the server (Part 2 / 2b):
  policies `resident` (keep both in HBM), `eager` (reply, then swap out),
  `lazy` (swap only when the other model is requested).
- `tenant_switch.py` + `switch_report.py` — Part 1: fixed alternation and
  the `Tenant` class (restore / step / snapshot) everything else reuses.
- `app_run.py` + `dispatch_proc.py` — Part 3: a separate process per
  request (run, reply, snapshot, exit to release the cores).
- `bench_xfer.py` + `xfer_report.py` — core-to-core transfer: through host
  memory vs direct HBM→HBM `broadcast`.
- `k8s/neuron-mt-poc-job.yaml` — **the only job spec**. `EXPERIMENT` picks
  `switch`, `dispatch`, `xfer` or `proc`. Clones this repo at start, archives
  results to the PVC.
- `k8s/s-lnc1-rct.yaml`, `k8s/s-lnc2-rct.yaml` — ResourceClaimTemplates,
  1 device each, LNC=1 or LNC=2.
- `docs/two-container-approach.md` — deferred two-container design.

## Run

```bash
kubectl delete job neuron-mt-poc --ignore-not-found
kubectl apply -f k8s/neuron-mt-poc-job.yaml
kubectl wait --for=condition=Ready pod -l job-name=neuron-mt-poc --timeout=600s
kubectl logs job/neuron-mt-poc -f | tee /tmp/neuron-mt-run
```

Defaults: `EXPERIMENT=dispatch`, `DISPATCH_POLICIES="resident eager"`,
LNC=1 with 8 ranks (`NPROC=8`), 200 alternating requests. To change LNC,
change the claim template, the two LNC env vars and `NPROC` together.
