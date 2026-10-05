# Project Log — Neuron Device Multi-Tenancy PoC

Handoff document. Everything needed to continue this work is here or in the
referenced files. Last updated: 2026-10-05.

## Goal and context

Help prime-video run two workloads — **dit** (p_0) and **unrolling** (p_1) —
on the **same Neuron device** (nd_0), MIG-like sharing. Slack discussion with
Liran Alon established:

- NRT does **not** support two processes on the same NeuronCore. Core
  ownership is exclusive per process from `nrt_init` to `nrt_close`/exit.
- NRT **does** support assigning different cores to different processes via
  `NEURON_RT_VISIBLE_CORES` / `NEURON_RT_NUM_CORES`.
- HBM↔host copies use the same NRT paths as input/output tensor movement.

So multi-tenancy = **time-slicing** the device: snapshot tenant A's HBM state
to host memory, restore tenant B's state from host, run B, switch back.
Dispatch latency is acceptable; this is the "context switch".

Two architectures were considered:

1. **Two containers, one device** (c_0, c_1 in one pod sharing a
   ResourceClaim, flock+turn-file coordination, full NRT teardown per switch).
   Honest demo but switch cost = seconds (nrt_init + NEFF reload dominate).
   **Deferred** — fully designed in `docs/two-container-approach.md`.
2. **One container, one torchrun, both models in one runtime** — only tensor
   data moves on a switch; both compiled graphs (NEFFs) stay resident in HBM.
   Switch cost = milliseconds. **This is what's implemented.**

Synthetic tenants for now (transposes + matmul over per-rank m1/m2 square
matrices, state evolved every step so snapshots carry real progress). Real
dit/unrolling models come after the concept is proven. **It is now proven.**

## Repo and infra

- Repo: **github.com/yahavb/neuron-multi-tenancy** (private, branch `main`).
  Local clone: `/Users/yahavb/multi-tenant-neuron`. HEAD: `9d79a25`.
- Cluster: EKS with DRA Neuron driver. Node: trn2.48xlarge
  (`node-type: trn2` selector). 1 device claimed via
  `k8s/s-lnc2-rct.yaml` (ResourceClaimTemplate `s-lnc2`, LNC=2 → **4 logical
  cores**, each = 2 physical NeuronCore-v3).
- Image: `421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b@sha256:404df22...`
  — ships the **eager torch-neuronx stack** (`/opt/torch-neuronx/.venv`,
  Python 3.12, NeuronX cc 2.0.404830, NRT 2.x.79381, driver 2.30.2).
  **No torch_xla.** Idioms (copied from the working
  `~/neuron-gridsample/frame_torchrun.py`):
  - `torch.device("neuron")`, `torch_neuronx._lazy_init()`
  - compile with `torch.compile(backend="neuron", dynamic=False)`
  - force execution / sync for timing by landing one element:
    `t.reshape(-1)[:1].cpu()`
  - **pin cores per rank BEFORE importing torch_neuronx**:
    `NEURON_RT_VISIBLE_CORES=$LOCAL_RANK`, `NEURON_RT_NUM_CORES=1`
    (without this every rank requests all 4 logical cores →
    `nrt_allocate_neuron_cores ... Requested:4 Available:0, cores busy`)
- Job pattern (from `~/neuron-gridsample/univr-2x4-tiles-4k-job.yaml`): the
  k8s Job clones this repo at start (`GITHUB_TOKEN` from secret
  `github-token`), runs torchrun from the clone, archives
  `/tmp/mt_out` tarball + report JSON to PVC
  `621547421844-ap-southeast-4-pvc` under `/var/mdl/mt/`.
- `NEURON_CC_FLAGS="--target trn2 --lnc 2"`, `NEURON_LOGICAL_NC_CONFIG=2`.

## Files

| file | what |
|---|---|
| `tenant_switch.py` | Part 1: alternating residency loop. Tenant class (host/dev state dicts, `restore()`, `snapshot()`, `step()`), `work()` = 2 transposes + matmul + norm-rescale, per-rank core pinning at module top (import side effect — `dispatch_serve.py` relies on it), per-rank stats JSON. Knobs: `T0_DIM`(2048), `T1_DIM`(1536), `ITERS`(10 steps/residency), `SWITCHES`, `MT_STAT`. |
| `switch_report.py` | Aggregates rank JSONs: p50/p90/p99/max per phase, time-in-ops vs time-in-switches, overhead %, drift (last/first decile). |
| `dispatch_serve.py` | Part 2: event-driven dispatch. Rank 0 runs HTTP server (:8080): `POST /e0`→m0, `POST /e1`→m1, `POST /shutdown`. Event id broadcast to all 4 ranks via gloo (`init_process_group("gloo")`); each rank swaps target model in if absent, runs `STEPS_PER_EVENT` steps, then per `DISPATCH_POLICY`: `eager`=snapshot to host after every event (device empty between events); `lazy`=stay resident, evict only when the other model is requested. Barrier before HTTP response. `EVENT_SOURCE=self` → generator thread drives `EVENTS` requests through real HTTP (`EVENT_PATTERN`: alternate/burst/random, `EVENT_BURST`); `external` → waits for outside callers. |
| `dispatch_report.py` | Per model: hit rate, swap_in/swap_out/run/e2e/queue percentiles; median e2e decomposition = queue + run + dispatch(swaps+sync+http). |
| `k8s/neuron-mt-poc-job.yaml` | Job `neuron-mt-poc` → tenant_switch (user renamed this file; keep the name). SWITCHES=1000 currently. |
| `k8s/neuron-mt-dispatch-job.yaml` | Job `neuron-mt-dispatch` → dispatch_serve. EVENTS=200, eager/alternate currently. |
| `k8s/s-lnc2-rct.yaml` | ResourceClaimTemplate (1 device, LNC=2, trn2/trn3). |
| `docs/two-container-approach.md` | Full deferred design: shared ResourceClaim across two containers, flock + turn-file protocol on emptyDir (crash-safe, deterministic alternation, no startup sleeps), what to measure, open questions (nrt_close vs process exit, preemption, cross-pod locking via hostPath/daemonset). |

## Run history (all on 1 ND, LNC=2, 4 ranks, fp32)

### Run 1 — FAILED: `ModuleNotFoundError: No module named 'torch_xla'`
First version used torch_xla idioms. Image is eager torch-neuronx. Ported
(commit `166365a`).

### Run 2 — FAILED: `nrt_allocate_neuron_cores Requested:4 Available:0`
Every rank claimed all cores. Fixed with per-rank pinning before import
(commit `80fff7e`).

### Run 3 — SUCCESS: 6 switches, ITERS=10
First working numbers: switch (snapshot+restore) median 8.12 ms = 12.7× one
step. Copies ~5–7 GiB/s.

### Run 4 — SUCCESS: 1000 switches endurance (commit `cc7b585`)
4000 switch samples pooled over 4 ranks:
```
t0_dit  (2048², 32 MiB/rank): step p50 0.83  snapshot p50 5.36 (5.8 GiB/s)  restore p50 3.94 (7.9 GiB/s)
t1_unroll(1536², 18 MiB/rank): step p50 0.46  snapshot p50 2.29 (7.7 GiB/s)  restore p50 2.20 (8.0 GiB/s)
switch  p50 7.05  p90 10.36  p99 12.76  max 15.02 ms
time in ops 25.8 s | switches 28.9 s | overhead 52.9% at ITERS=10
drift (last/first decile) 0.85x  → NO degradation over 1000 HBM alloc/free cycles
```
Step sanity: 2048³ matmul ≈ 17.2 GFLOP in 0.83 ms ≈ 21 TFLOPS/logical core —
real PE work, not overhead. Copy BW ~6–8 GiB/s = pageable-host-memory bound,
not DMA limit; pinned buffers are the known next optimization if needed.
Overhead model: `switch/(switch + ITERS*step)` → ITERS=100 ≈ 11%, 1000 ≈ 1%.

### Run 5 — SUCCESS: event dispatch, eager policy, alternate pattern (commit `9d79a25`)
Job `neuron-mt-dispatch`, log at `/tmp/neuron-mt-dispatch`. Full flow and
results are in the "Event-driven dispatch" section below. Summary: 200 HTTP
events, 0% residency hits (expected for eager+alternate, the worst case), median
request time 26.73 ms, of which 19.97 ms is dispatch overhead (~75%). Latency
stayed flat for all 200 events.

## Event-driven dispatch (`dispatch_serve.py`) — flow and results

### Why

Part 1 (`tenant_switch.py`) switched tenants on a fixed schedule. The real
dit/unrolling service will run a model when a request arrives. Part 2 replaces
the fixed loop with a request-driven dispatcher: a request names the model it
wants, the dispatcher puts that model's state in HBM, runs it, and (by policy)
moves the state back to host memory.

### Process layout

One container runs one `torchrun --nproc-per-node 4`, so there are 4 Python
processes, each pinned to one logical core (`NEURON_RT_VISIBLE_CORES=$LOCAL_RANK`).
Each rank holds its own copy of both models (m0 = `t0_dit` 2048², m1 =
`t1_unroll` 1536²) as a `Tenant` object, and both compiled graphs stay loaded
on its core for the whole run. Ranks talk over a CPU-side gloo process group.
Only rank 0 runs the HTTP server.

### HTTP API (rank 0, port 8080, `DISPATCH_PORT`)

| request | effect |
|---|---|
| `POST /e0` | run model m0 (dit stand-in); response returns when all 4 cores finish |
| `POST /e1` | run model m1 (unrolling stand-in) |
| `POST /shutdown` | all ranks exit the loop, write stats, exit cleanly |
| anything else | 404 `{"error": "unknown event ..."}` |

The request blocks until the event is served on all 4 cores. The response
carries that event's rank-0 timings, for example:

```json
{"model": "m0", "queue_ms": 0.12, "hit": 0, "swap_out_ms": 7.43,
 "swap_in_ms": 9.70, "run_ms": 8.59, "total_ms": 26.1, "e2e_ms": 27.18}
```

(Quirk: `hit` is returned as 1/0 rather than true/false because the reply
rounds every numeric field. Harmless.)

### What happens for one event, step by step

1. **Request arrives.** An HTTP handler thread on rank 0 receives `POST /e0`,
   creates an item stamped with the arrival time, puts it on a queue, and waits.
2. **Dequeue.** Rank 0's main loop takes the item off the queue. The time spent
   waiting there is `queue_ms`. Events are served one at a time, in arrival order.
3. **Broadcast.** Rank 0 broadcasts the model id to ranks 1–3 over gloo. Ranks
   1–3 sit blocked in this broadcast between events, so all 4 cores start the
   same event together.
4. **Swap out** (every rank). If the other model is still in HBM, snapshot it to
   host memory (HBM→host copy, device tensors freed). Under eager policy this
   never happens here, because the previous event already evicted its model.
5. **Swap in** (every rank). If the requested model isn't in HBM, restore it
   (host→HBM copy). Timed as `swap_in_ms`. Under eager policy this happens every
   event.
6. **Run** (every rank). `STEPS_PER_EVENT` (10) steps of transposes + matmul,
   each forced to complete on the device. Timed as `run_ms`.
7. **Write back** (eager policy only). Snapshot the model's state to host memory
   right away, so HBM is empty between events. Added to `swap_out_ms`. Under lazy
   policy the model stays in HBM until a request for the other model evicts it
   in step 4.
8. **Barrier.** All 4 ranks wait for each other. The event counts as served
   only when every core is done.
9. **Reply.** Rank 0 measures `e2e_ms` from arrival to now, fills the reply,
   and wakes the HTTP handler, which returns the JSON response.

```
caller        rank0 HTTP thread     rank0 main loop          ranks 1-3
  | POST /e0 ->|                      |                          |
  |            |-- queue.put(item) -->|                          |
  |            |   (waits)            |-- broadcast(model=0) --->|
  |            |                      |  swap in m0 / run 10 /   |  same, on own core
  |            |                      |  snapshot m0 to host     |
  |            |                      |<======= barrier ========>|
  |            |<-- reply + done -----|                          |
  |<- 200 JSON-|                      |                          |
```

### Where events come from

- `EVENT_SOURCE=self` (default, used in Run 5): a generator thread on rank 0
  sends `EVENTS` real HTTP requests to `127.0.0.1:8080`, one at a time (each
  waits for its response), then sends `/shutdown`. The order comes from
  `EVENT_PATTERN`: `alternate` = e0,e1,e0,e1…; `burst` = 8×e0, 8×e1, …
  (`EVENT_BURST`); `random` = seeded random choice.
- `EVENT_SOURCE=external`: the server waits for outside callers. To drive it
  by hand:
  ```bash
  POD=$(kubectl get pod -l job-name=neuron-mt-dispatch -o name)
  kubectl port-forward "$POD" 8080:8080 &
  curl -s -X POST localhost:8080/e0
  curl -s -X POST localhost:8080/e1
  curl -s -X POST localhost:8080/shutdown   # required, or the job runs until activeDeadlineSeconds
  ```

### Run 5 settings

`DISPATCH_POLICY=eager`, `EVENT_PATTERN=alternate`, `EVENTS=200`,
`STEPS_PER_EVENT=10`, `EVENT_SOURCE=self`, `T0_DIM=2048`, `T1_DIM=1536`,
4 ranks, fp32. Commit `9d79a25`.

### Run 5 results (verbatim from `/tmp/neuron-mt-dispatch`)

```
[rank 0] served 50 events, recent e2e median 26.93 ms
[rank 0] served 100 events, recent e2e median 26.73 ms
[rank 0] served 150 events, recent e2e median 26.92 ms
[rank 0] served 200 events, recent e2e median 26.87 ms
policy=eager pattern=alternate events=200 steps/event=10 ranks=4
m0: 100 events, residency hit rate 0%
  swap_in   p50     9.70  p90     9.75  p99     9.77  max    11.96 ms  (n=400)
  swap_out  p50     7.43  p90     8.64  p99    10.52  max    12.67 ms  (n=400)
  run       p50     8.59  p90     8.71  p99     8.96  max    10.85 ms  (n=400)
  e2e       p50    27.18  p90    28.06  p99    30.30  max    30.30 ms  (n=100)
  queue     p50     0.12  p90     0.16  p99     0.33  max     0.33 ms  (n=100)
m1: 100 events, residency hit rate 0%
  swap_in   p50     5.52  p90     5.56  p99     5.58  max     5.61 ms  (n=400)
  swap_out  p50     3.50  p90     4.80  p99     5.15  max     6.42 ms  (n=400)
  run       p50     4.74  p90     4.86  p99     5.00  max     5.03 ms  (n=400)
  e2e       p50    15.37  p90    15.65  p99    16.09  max    16.09 ms  (n=100)
  queue     p50     0.14  p90     0.17  p99     0.20  max     0.20 ms  (n=100)
median event: e2e 26.73 ms = queue 0.13 + run 6.63 + dispatch(swaps+sync+http) 19.97 ms
```

`n=400` = 100 events × 4 ranks (per-rank timings pooled). `e2e` and `queue`
are rank 0 only (`n=100`), since only rank 0 sees the HTTP request.

### What Run 5 shows

- **The flow works.** 200 requests through the real HTTP path, each fanned out
  to 4 cores and answered after all of them finished. Nothing hung or crashed,
  and shutdown was clean.
- **Stable.** The request-time median stayed at 26.7–26.9 ms across all four
  50-event checkpoints. p99 sits close to p50 (m0: 30.3 vs 27.2 ms).
- **Queueing is negligible** (~0.1 ms), as expected with one request in flight
  at a time.
- **Compute matches Part 1.** m0 run 8.59 ms / 10 steps = 0.86 ms per step
  (Part 1: 0.83 ms). m1: 0.47 ms per step (Part 1: 0.46 ms).
- **Copies dominate.** Per m0 request: 9.70 swap-in + 8.59 run + 7.43 write-back
  ≈ 25.7 of the 27.2 ms e2e. The remaining ~1.5 ms is broadcast, barrier and
  HTTP. For m1: 5.52 + 4.74 + 3.50 ≈ 13.8 of 15.4 ms.
- **The pooled breakdown line is approximate.** It takes medians across both
  models (run 6.63 ms sits between m0's 8.59 and m1's 4.74), so use the
  per-model lines for exact numbers.
- **Open question: swap-in is about 2.5× slower than in Part 1.** m0 restore:
  9.70 ms here vs 3.94 ms in Run 4; m1: 5.52 vs 2.20 ms. Compute is unchanged,
  and the spread is very tight (m0 p50 9.70, p90 9.75), which points to a fixed
  per-restore cost rather than noise. Not yet explained. Things to check: extra
  host threads (HTTP server, generator, gloo) competing with the copy at
  `OMP_NUM_THREADS=1`; eager eviction freeing and re-allocating device memory
  every event; or the restore including time waiting on the device. Worth
  resolving before quoting dispatch numbers for dit/unrolling.

## Immediate next step (queued, not yet run)

Lazy-policy / burst-pattern comparison — expected: ~87% hit rate (7 of 8
events in a burst skip both copies), e2e collapsing toward pure run time:

```bash
kubectl delete job neuron-mt-dispatch
sed -e 's/value: "eager"/value: "lazy"/' -e 's/value: "alternate"/value: "burst"/' \
  k8s/neuron-mt-dispatch-job.yaml | kubectl apply -f -
kubectl logs job/neuron-mt-dispatch -f
# then compare dispatch_report output vs Run 5
```

The eager-vs-lazy delta is the headline for prime-video: event latency is
dominated by state-movement policy, not device work.

## Roadmap after that

1. **Dim sweep** (`T0_DIM`/`T1_DIM` → 4096, 8192 ≈ 512 MiB/rank): fit
   `switch_cost = a + bytes/BW`; check BW holds at size. Extrapolation at
   ~6 GiB/s: 10 GiB tenant state ≈ 1.7 s/direction → argues for snapshotting
   only the live working set of the real models.
2. **Capacity proof**: make combined tenant state exceed one core's HBM —
   a config impossible without switching. The demo-able proof.
3. **Pinned host memory / dev/shm staging** if copy BW matters.
4. **Replace synthetic tenants with real dit + unrolling** behind the same
   `Tenant` interface (restore/step/snapshot) and the same event dispatcher.
5. **Two-container version** (docs/two-container-approach.md) to quote the
   cross-process switch cost honestly.

## Operational notes

- User drives all `kubectl` runs manually and pastes logs to
  `/tmp/neuron-mt-poc` / `/tmp/neuron-mt-dispatch` via `tee`. Jobs are
  immutable — delete before re-apply.
- Don't rename `k8s/neuron-mt-poc-job.yaml` (user chose that name).
- `gh` CLI authenticated as `yahavb`; push to `main` directly.
- Smoke-test pattern used before every push: `python3 -m py_compile` + run
  report scripts against synthetic rank JSONs.
- gridsample reference code for idioms: `~/neuron-gridsample/`
  (`frame_torchrun.py` = torchrun/core-pinning/compile/land idioms;
  `gridsample_nki.py` = NKI kernel path, only needed if plain
  torch.compile ops ever misbehave).
