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
| `k8s/neuron-mt-poc-job.yaml` | **The only job spec** (job `neuron-mt-poc`). `EXPERIMENT` picks what runs: `switch` (tenant_switch + switch_report), `dispatch` (dispatch_serve + dispatch_report), `xfer` (bench_xfer both modes + xfer_report), `proc` (dispatch_proc). Defaults: `proc`, LNC=1 (`s-lnc1`), `NPROC=8`. To change LNC, change the claim template, the two LNC env vars and `NPROC` together. |
| `bench_xfer.py` | Core-to-core transfer benchmark: host DRAM bounce vs HBM→HBM `broadcast` on the `neuron` backend, 2 ranks, size sweep, checksum-verified. |
| `xfer_report.py` | Side-by-side table for the two transfer modes plus a fixed-cost + bandwidth fit. |
| `app_run.py` | Part 3: one request for one app as its own torchrun job — init, load graph, restore state from `/dev/shm`, run, print MT_RESULT, snapshot to `/dev/shm`, exit (releases the cores). |
| `dispatch_proc.py` | Part 3 dispatcher (no Neuron): HTTP `/e0` `/e1` `/shutdown`, launches `app_run.py` per request, replies on MT_RESULT before the snapshot, starts the next request only after the app exits. |
| `k8s/s-lnc1-rct.yaml` | ResourceClaimTemplate (1 device, LNC=1). |
| `k8s/s-lnc2-rct.yaml` | ResourceClaimTemplate (1 device, LNC=2, trn2/trn3). |
| `docs/two-container-approach.md` | Full deferred design: shared ResourceClaim across two containers, flock + turn-file protocol on emptyDir (crash-safe, deterministic alternation, no startup sleeps), what to measure, open questions (nrt_close vs process exit, preemption, cross-pod locking via hostPath/daemonset). |

## Run history (1 ND, fp32; LNC=2 with 4 ranks unless noted)

Runs 1–7 used separate job specs (`neuron-mt-dispatch`, `neuron-mt-xfer`, …).
Those were merged into `k8s/neuron-mt-poc-job.yaml`; select the experiment
with `EXPERIMENT`.

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
  # in k8s/neuron-mt-poc-job.yaml set EXPERIMENT=dispatch, EVENT_SOURCE=external
  POD=$(kubectl get pod -l job-name=neuron-mt-poc -o name)
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
kubectl delete job neuron-mt-poc --ignore-not-found
sed -e 's/value: "proc"/value: "dispatch"/' -e 's/value: "eager"/value: "lazy"/' \
    -e 's/value: "alternate"/value: "burst"/' \
  k8s/neuron-mt-poc-job.yaml | kubectl apply -f -
kubectl logs job/neuron-mt-poc -f
# then compare dispatch_report output vs Run 5
```

The eager-vs-lazy delta is the headline for prime-video: event latency is
dominated by state-movement policy, not device work.

## Core-to-core transfer benchmark (Run 6 — results at the end of this section)

Prompted by feedback: when a tensor moves from one NeuronCore to another,
don't bounce it through host DRAM. Note this does **not** apply to the current
tenancy loop. There, each tenant's state leaves a core and comes back to the
*same* core, and it goes to host only to free HBM. It does apply wherever data
crosses cores: dit output feeding unrolling on a different core, or moving a
tenant to another core.

`bench_xfer.py` moves one fp32 tensor from rank 0's core to rank 1's core (2
of the 4 logical cores on the claimed device), at 1–1024 MiB, using the two
options available today:

- **host**: rank 0 copies HBM → `/dev/shm`-backed host buffer, gloo barrier
  hand-off, rank 1 copies host → HBM. This is the best case of the DRAM path
  (no extra host copy), and it reports d2h / handoff / h2d separately.
- **hbm**: `dist.broadcast(src=0)` on the `neuron` process group, which runs
  over the device fabric and never touches host memory. Setup copied from
  `~/neuron-gridsample/bench_assembly.py`, which already ran `all_gather` on
  this image (23.2 ms frame assembly, FINDINGS.md): no core pinning,
  `NEURON_RT_ROOT_COMM_ID`, `torch.neuron.set_device(LOCAL_RANK)`, the
  collective inside a `torch_neuronx.Stream`, then `torch_neuronx.synchronize()`.

Why broadcast: TorchNeuronEager's `ProcessGroupNeuron`
(`torch_neuronx/distributed/backend.py`) has no point-to-point `send`/`recv`.
On a 2-rank group, `broadcast` is exactly one A→B copy. Mainline has it; if
this image's build doesn't (it raises `NotImplementedError`), the script falls
back to `all_gather` and says so. That fallback moves data both ways, so its
times are an upper bound.

Every size is checksum-verified before timing. Each timed iteration starts
from a host barrier, and the reported time is the slower of the two ranks.
`xfer_report.py` prints both modes side by side and fits
`time ≈ fixed + size / bandwidth` for each.

```bash
# in k8s/neuron-mt-poc-job.yaml set EXPERIMENT=xfer
kubectl apply -f k8s/neuron-mt-poc-job.yaml
kubectl logs job/neuron-mt-poc -f | tee /tmp/neuron-mt-xfer
```

### Run 6 results (commit `89c7b60`, log `/tmp/neuron-mt-xfer`)

`broadcast` exists in this image's build (no fallback was used). All 12
transfers verified. p50 ms, 20 iterations each:

| MiB | via host: total | d2h | h2d | GiB/s | direct HBM→HBM | GiB/s | host ÷ direct |
|---|---|---|---|---|---|---|---|
| 1 | 0.306 | 0.109 | 0.153 | 3.19 | 0.340 | 2.87 | 0.90× |
| 4 | 0.769 | 0.332 | 0.388 | 5.08 | 0.402 | 9.71 | 1.91× |
| 16 | 2.564 | 1.216 | 1.301 | 6.09 | 0.819 | 19.09 | 3.13× |
| 64 | 9.727 | 4.746 | 4.922 | 6.43 | 2.313 | 27.02 | 4.21× |
| 256 | 38.154 | 18.820 | 19.273 | 6.55 | 8.169 | 30.61 | 4.67× |
| 1024 | 152.087 | 75.100 | 76.880 | 6.58 | 31.461 | 31.79 | 4.83× |

Fits: via host ≈ 0.19 ms + size / 6.58 GiB/s; direct ≈ 0.33 ms + size / 32.1 GiB/s.

The run's printed "handoff" column was wrong (≈ d2h) because it was read from
rank 1, whose barrier wait includes rank 0's whole D2H. The real hand-off is
total − d2h − h2d ≈ 0.04–0.1 ms. Fixed after the run: handoff is now read from
rank 0. Totals were never affected.

What it shows:

- **Direct is ~4.8× faster for large tensors** (32 vs 6.6 GiB/s). The two
  paths cross at about 1 MiB; below that the collective's ~0.33 ms fixed cost
  makes the host path slightly faster.
- **The host path's cost is two serialized copies.** Each direction alone runs
  at ~13.3 GiB/s at 1 GiB, and doing D2H then H2D halves that.
- **For dit→unrolling across cores:** at 32 GiB/s a 1 GiB activation hand-off
  costs ~31 ms direct vs ~152 ms via host.
- **Side finding for the tenancy switch:** D2H into a *preallocated* host buffer
  reached ~13 GiB/s at large sizes. Part 1's snapshot uses `.cpu()` into fresh
  memory and measured 5.8–7.7 GiB/s. Size doesn't explain the gap: m0's
  snapshot is two 16 MiB tensors in 5.36 ms, while one 16 MiB D2H here took
  1.216 ms (two would be ~2.4 ms). So snapshotting into preallocated host
  buffers looks like ~2× on the switch cost. Worth testing in `tenant_switch.py`.
- **Idea, not tested:** if HBM on a neighbouring core has room, parking an idle
  tenant's state there via the direct path (32 GiB/s, one copy) could beat
  parking it in host DRAM (~13 GiB/s each way).

### Run 7 — same benchmark at LNC=1 (commit `42c4b7f`, log `/tmp/neuron-mt-xfer-lnc1`)

`k8s/neuron-mt-xfer-job.yaml` now uses the `s-lnc1` claim with LNC settings of
1, so rank 0 and rank 1 are physical cores 0 and 1 (at LNC=2 they were cores
0–1 and 2–3). All 12 transfers verified. p50 ms:

| MiB | via host LNC=1 | via host LNC=2 | direct LNC=1 | direct LNC=2 |
|---|---|---|---|---|
| 1 | 0.304 | 0.306 | 0.291 | 0.340 |
| 16 | 2.573 | 2.564 | 0.846 | 0.819 |
| 256 | 38.172 | 38.154 | 9.553 | 8.169 |
| 1024 | 152.074 | 152.087 | 37.326 | 31.461 |

Fits at LNC=1: via host ≈ 0.19 ms + size / 6.58 GiB/s; direct ≈ 0.26 ms +
size / 26.97 GiB/s.

- **Via host is identical at both LNC settings** (6.58 GiB/s). It's bound by
  the PCIe and host-memory copies, not by the core.
- **Direct is ~16% slower at LNC=1** (26.8 vs 31.8 GiB/s at 1 GiB), but has a
  lower fixed cost (0.26 vs 0.33 ms), so it now beats the host path even at
  1 MiB. A likely reason is that an LNC=2 core has the DMA engines of two
  physical cores. That's an assumption; it hasn't been checked.
- **Direct is still ~4× faster than via host** at 1 GiB under LNC=1 (4.07×,
  vs 4.83× at LNC=2).
- The handoff fix is confirmed: the hand-off now reads 0.04–0.09 ms.

## Part 3 — process per request (Run 8 results at the end of this section)

The flow the user specified: a request is dispatched to p_0, p_0 runs and the
result goes back to the user, then p_0 copies its HBM state to host and
releases the cores. Same for p_1. This is the real two-process design, unlike
Parts 1–2 where both apps live in one long-running process.

Releasing the cores means the process exits. TorchNeuronEager only calls
`nrt_close()` from an atexit handler (`csrc/core/NeuronBindings.cpp`), and
`_lazy_init()` returns early once initialized, so a live process can neither
give its cores back nor re-acquire them. So every request starts a fresh app
process.

Per request (`dispatch_proc.py` + `app_run.py`):

1. Dispatcher launches `torchrun --nproc-per-node 8 app_run.py --app K`
   (LNC=1, so 8 ranks own the device's 8 cores).
2. Each rank: imports and runtime init → loads the compiled graph (a warm-up
   step on scratch tensors, so the app's state isn't touched; the graph comes
   from the compile cache after prewarm) → restores its state from
   `/dev/shm/mt_state` into HBM → runs 10 steps.
3. Barrier, then rank 0 prints `MT_RESULT`; the dispatcher replies to the user
   right away.
4. Each rank copies its state HBM → its preallocated `/dev/shm` buffer
   (the faster D2H path from Run 6), then the process exits and the cores are
   released.
5. Only then does the dispatcher start the next request. A request that arrives
   during step 4 waits, and that wait shows up as `queue_ms`.

Startup runs each app once with `--prewarm` (compile, create state), not
counted. Timed per request: launch (torchrun + Python start), init, load,
restore, run, snapshot, exit, the user's wait (`e2e`), and how long the device
was held. Phase times are rank 0's.

```bash
# EXPERIMENT=proc is the default in k8s/neuron-mt-poc-job.yaml
kubectl delete job neuron-mt-poc --ignore-not-found
kubectl apply -f k8s/neuron-mt-poc-job.yaml
kubectl logs job/neuron-mt-poc -f | tee /tmp/neuron-mt-proc
```

### Run 8 — Part 3 on hardware (commit `413ec9d`, log `/tmp/neuron-mt-proc`)

LNC=1, 8 ranks per app, 20 alternating requests, 10 steps per request. All
20 requests served. Prewarm (first compile): app0 29.2 s, app1 21.6 s.

Per request, p50 (rank 0's phases):

| phase | app0 | app1 |
|---|---|---|
| user waited (e2e) | 25.2 s (p90 31.2) | 18.3 s (p90 27.3) |
| queue (waiting for the previous app to exit) | 2.15 s | 2.11 s |
| launch (torchrun + Python start) | 2.79 s | 2.80 s |
| init (imports + runtime init + gloo) | 11.2 s | 11.7 s |
| load compiled graph (from cache) | 1.17 s | 1.18 s |
| restore state host → HBM | 3.1 ms | 2.1 ms |
| run 10 steps | 136 ms | 127 ms |
| snapshot HBM → host (after the reply) | 18.6 ms | 10.6 ms |
| exit (releases the cores) | 2.09 s | 2.14 s |

Overall median: the user waits **18.7 s** for **0.13 s** of work, and the
device is held 18.6 s per request. Part 2 (both apps in one process) served
the same kind of request in 15–27 ms, so process-per-request is ~1000× slower.

- **The state copies are not the problem.** Restore 2–5 ms, snapshot 11–19 ms.
- **Process lifecycle is the whole cost**: ~2.8 s launch + ~11.5 s init +
  ~1.2 s graph load before any work, then ~2.1 s exit that the next request
  waits behind.
- **Unexplained gap in some requests: up to ~10 s.** In requests 1, 4, 5, 11,
  13 and 19 the user's wait is 8–10 s more than queue + rank-0 phases. The
  reply goes out after a barrier across all 8 ranks, so this is most likely
  some rank initializing much more slowly than rank 0 (whose phases are the
  only ones logged). In the other requests the gap is ~0.1 s. That's also why
  app0's p50 (25.2 s) is higher than app1's: more of its requests hit the gap.
- **Next measurement needed:** split init into import torch, import
  torch_neuronx, runtime init and gloo init, and log every rank, not just
  rank 0. That shows where the 11 s goes and which rank causes the ~10 s gap.

## Part 2b — one server, keep both models in HBM (Run 9 results at the end of this section)

Decision after Run 8: one long-running server process owns all 8 cores and
holds both models, so the ~13 s process start-up is paid once. Weights are
most of the memory, and at LNC=1 each core has about 12 GB of HBM. If both
models fit, keep both resident and copy nothing. If not, swap the idle
model's weights to host memory inside the same process (milliseconds).

Changes to `dispatch_serve.py`:
- New policy `resident`: both models are loaded into HBM before serving and
  never copied out.
- `eager` now replies to the user first, then copies the state to host
  (`after_reply_ms`). The next request still waits for that copy.
- HBM use per core is logged (torch_neuronx caching-allocator stats) after
  warm-up, when serving starts, and at peak.

The job (`EXPERIMENT=dispatch`, the default now) runs `DISPATCH_POLICIES`
back to back, `resident` then `eager`, at LNC=1 with 8 ranks and 200
alternating requests each, and prints one report per policy.

```bash
kubectl delete job neuron-mt-poc --ignore-not-found
kubectl apply -f k8s/neuron-mt-poc-job.yaml
kubectl logs job/neuron-mt-poc -f | tee /tmp/neuron-mt-dispatch-lnc1
```

### Run 9 results (commit `8ea14b4`, log `/tmp/neuron-mt-dispatch-lnc1`)

LNC=1, 8 ranks, 200 alternating requests per policy, 10 steps per request.

Keep both resident (no copies), p50:
- m0 (2048², dit stand-in): user waited 16.9 ms, run 16.3 ms.
- m1 (1536², unrolling stand-in): user waited 8.6 ms, run 8.1 ms.
- Overhead on top of the run is about 0.6 ms (broadcast, barrier, HTTP).
  Stable over all 200 requests (rolling median 16.65–16.85 ms).

Swap after reply, p50:
- m0: user waited 46.1 ms = queue 10.2 + swap-in 14.8 + run 16.0 + ~5.
  Copy-back after the reply took 15.0 ms.
- m1: user waited 37.7 ms = queue 18.6 + swap-in 8.3 + run 8.0 + ~3.
  Copy-back 8.8 ms.
- The queue is each request waiting for the previous model's copy-back. So
  replying first moves that copy out of the request that caused it, but with
  alternating traffic the next request pays it instead.

Findings:
- Keeping both models resident makes switching essentially free: the user
  waits for the run plus about half a millisecond.
- Swapping roughly doubles to triples the wait for these small states, and
  cost grows with weight size (see next point).
- Copies from all 8 cores share the device's host link. m0 moves
  8 × 32 MiB = 256 MiB per direction in ~15 ms, about 17 GiB/s for the whole
  device. For real models, budget roughly 60 ms per direction per GB of
  weights on the device. Multi-GB weights would cost seconds per switch,
  which is why keeping both resident matters.
- Per-step time at LNC=1 is about 1.6 ms for m0 versus 0.86 ms at LNC=2,
  as expected with half the compute per core.
- HBM stats came back empty (`-0 MiB`): `torch_neuronx.memory_stats` failed
  in this build. The server now tries `memory_allocated` variants and prints
  the error plus the memory-related names `torch_neuronx` does have, so the
  next run shows what's available.

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
