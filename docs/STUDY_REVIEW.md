# Inspection of the original study

## Study objective and interpretation

The objective is a small network that first learns a plant's input/output
relationship, serves many instances efficiently on one RTX 3060, and continues
learning from new measurements on a second RTX 3060. Resource scheduling and
adaptation are both essential parts of the study. Running many untrained random
models alone does not establish the result.

The supplied code shows two related directions: many independent networks and
one adaptive MIMO network. The minimal continuation supports independent model
IDs while allowing clients to share an ID if they model the same plant. This
preserves the many-network interpretation rather than silently replacing it
with 200 clients of one globally shared model.

The code learns a forward map `y = f(x)` with measured input/output pairs. That is
online **system identification**. It can be a component of adaptive control, but
the supplied study does not contain a controller, plant dynamics with a time
state, actuator constraints, a tracking objective, or a stability argument. The
synthetic MIMO function is a memoryless nonlinear regression target. A real
dynamic plant may require lagged inputs/outputs, normalized units, and a revised
input interface. No physical closed-loop success is claimed.

## Script inventory and findings

Paths below are relative to `study-scripts/`. The copies are unchanged, including
historical comments and shortcomings.

| Original source | What it contributes | Missing or problematic behavior |
| --- | --- | --- |
| `Accurate Adaptive control.txt` | Original nonlinear synthetic plant, reproducible coefficient seed, 10→128→128→10 model, offline Adam fitting | Evaluates on training data, not a separate holdout; no streaming or GPU selection |
| `Accurate Server.txt` and `Accurate Client.txt` | Initial supervised fitting followed by labelled prediction requests | CPU model; fitting runs inside HTTP requests; no trainer/inference separation |
| `Adaptive Changing Server.py` | CUDA placement, initial fitting, batch adaptation | Uses one default GPU for both roles; 1,000-epoch HTTP request; training mutates the serving model; debug reloader; no checkpoints or bounds |
| `Accurate_gpu.py` | Gradual coefficient drift and per-request MAE | Hard-coded LAN address; no model IDs, timeouts, or HTTP failure handling; single sequential stream |
| `Multiple inferences.py` | Pool-based inference timing and multiple model IDs | Creates random untrained networks; 50 tasks with 10 workers are not 50 simultaneous resident clients; no training handoff |
| `Parallel training.py` | Explicit MPS startup/thread-percentage comments, concurrent training attempts, CPU/GPU timing experiments | Several complete experiments concatenated in one file; top-level process creation and later main blocks; default multiprocessing context rather than explicit CUDA-safe spawn; every training worker targets `cuda:0`; unbounded CUDA-error retries; random input/output pairs have no learnable plant relationship |
| `parallel train and inferences.py` | Manager-based trained-model handoff | Completes all training before starting inference; both phases use `cuda:0`; no live updates, checkpoint protocol, or version tracking |
| `MPGU adaptive model servers/accurate_adaptive.py` | Another original CPU adaptive server | Same synchronous mutation and lifetime issues as the text server |
| `MPGU adaptive model servers/adaptive_changing_gpu.py` | Another original default-GPU adaptive server | Substantially duplicates `Adaptive Changing Server.py`; does not allocate distinct training/inference GPUs |
| `MPGU adaptive model servers/adaptive_flask.py` | Update stages of 100→50→10 samples | Smaller update batches are a scheduling policy, not a demonstrated accuracy improvement; no separate training process or initial fitting |
| `MPGU adaptive model servers/fft_cuda.cu` | cuFFT/cuRAND workload and repeat-count argument | Not a neural network; repeated FFT with synchronization, not a multi-GPU adaptive pipeline |
| `MPGU adaptive model servers/fft_mps.cu` | Explicit CUDA stream for random generation and FFT | A single stream with per-iteration synchronization; not evidence of overlapping independent streams; error helper functions are declared but not defined in this translation unit |
| `MPGU adaptive model servers/matmul_flask.py`, `matmul_flask_verbose.py`, `matmul_batch_flask.py` | GPU request and batching experiments | Matrix multiplication rather than learned models; timings and request batching do not demonstrate model adaptation or two-GPU weight transfer |
| `MPGU adaptive model servers/mpiTest*.py` | Two MPI ranks explicitly mapped to different GPUs | Sparse linear-solver experiments, not neural training/inference; randomly generated matrices are not constructed symmetric although MINRES requires symmetry |

The original `compute_session*.py` scripts were also inspected. They are outside
the minimal neural-network experiment and are omitted from the publishable source
selection. They execute request-supplied Python operations with `exec`; compressed
serialization does not change that behavior. `compute_session_exec.py` also leaves
session shutdown as an ellipsis and discards asynchronous computed results.
The original compiled `fft_cuda` binary is omitted; source is provided instead.

### Scheduling and timing caveats

1. The original MPS commands initialize a service and optionally constrain
   client execution-resource availability. They do not allocate two distinct
   neural-network roles, guarantee occupancy, or establish an optimal client
   count. MPS has finite context limits. Host 100–200 logical networks in a few
   processes before considering one CUDA process per network.
2. A pool size of 10 processing 50 tasks does not demonstrate 50 concurrent
   instances. Distinguish total task count, resident model count, CUDA context
   count, concurrent HTTP clients, and actual execution overlap.
3. PyTorch CUDA work is asynchronous. Kernel-only benchmarks need warm-up and
   CUDA events or explicit synchronization. End-to-end wall-clock measurements
   need a complete result transfer before stopping the clock. The original
   training timers do not consistently make this distinction.
4. `mpiTest_memsave.py` performs 20 timed solves but divides by 100 and labels the
   result as 100 solves. Its per-solve value is understated by a factor of five.
   The other timing scripts should not inherit that result.
5. The handwritten conclusion of a tenfold GPU gain in `Parallel training.py`
   is not supported merely by its adjacent CPU/GPU timings. Compare equal total
   workloads, process counts, data sizes, training budgets and synchronization.
6. An MLP this small may spend more time on launch, process scheduling, model
   publication, HTTP, and JSON overhead than on arithmetic. More processes or a
   lower MPS percentage can reduce throughput. Thread ceilings are benchmark
   variables, not settings with a universally optimal value.

## Minimum additions and deliberate scope

The original network class is retained verbatim in `adaptive_mgpu/model.py`.
Input/output dimensionality, ReLU activations, MSE objective, and Adam are
retained. All original external files remain unchanged. Existing study scripts
are evidence and reference material; they are not imported by the runnable path.

The minimum missing connections are implemented as follows:

| Required connection | Minimal implementation |
| --- | --- |
| Distinct GPU roles | Explicit `cuda:0` inference and `cuda:1` training, checked at startup |
| CUDA-safe process creation | A single spawned trainer; no CUDA tensors sent through a manager |
| Many independent small networks | Lazy per-ID models in one bank per role, capacity configured by `--instances` |
| Initial learning | Queued `/train` with a bounded optimizer-step budget |
| Continuing learning | `/observe` accepts paired delayed measurements independently of inference |
| Reuse of older examples | Bounded per-ID recent replay plus the latest labelled batch |
| Serving a consistent version | Trainer writes a temporary checkpoint and atomically renames it; inference loads a complete candidate under its request lock |
| Restart | Persist model weights, Adam state, and version per ID |
| Overload and failure visibility | Bounded queue, 503 on rejected labelled jobs, trainer-process liveness, pure inference retained after trainer failure |
| Efficient-resource comparison | Preserve the MPS/stream experiments and add an optional UUID-based MPS launcher |
| Evidence of adaptation | Fixed changed plant, holdout inputs not sent to training, comparison against frozen pre-adaptation model |

The original per-update exponential learning-rate decay is intentionally removed
from the live path: repeatedly multiplying by 0.99 approaches zero and can make
later adaptation ineffective. Decay by itself does not preserve past knowledge.
The replacement uses a configurable fixed rate and bounded replay. Replay here
is a recent rolling window, not a lifelong retention guarantee or a solution to
all catastrophic forgetting. Recent and replay batches are mixed equally;
this fraction and update budget may need tuning for actual drift rates.

The minimal version publishes every completed training job with finite weights.
It does not establish that every update improves accuracy. There is no automatic
validation-gated promotion or rollback. The API does not include persistent job
IDs, durable queues, authentication, or multi-host transport. The Flask harness
is intended for local research. These limitations are explicit rather than
hidden behind a claim of production or control-system readiness.

## Capacity reasoning, without a hardware claim

The original MIMO network contains 19,210 trainable parameters. Its float32
weights occupy 76,840 bytes; 200 weight sets contain about 14.66 MiB of raw
parameters. That makes a many-network bank plausible, but weight size alone
does not establish RTX 3060 capacity or throughput.

Training also retains gradients, two Adam moment buffers, activations and replay
batches, plus runtime and allocator memory. PyTorch/CUDA contexts and Python
objects add overhead. Checkpoints include optimizer state, so loading and saving
them costs more than a weights-only copy. Replay is capped at 2,048 observations
per initialized model by default; host storage uses Python objects and therefore
uses more memory than raw float32-array estimates suggest.

Start with a small bank, then measure 100 and 200 independently initialized IDs.
Both training service rate and inference deadline matter. If labelled jobs
arrive faster than the single trainer can process them, the bounded queue will
fill. Batch labels and adjust step budgets before adding more processes. If
sequential per-ID kernel launches become the bottleneck, batched parameter banks
or `torch.func`/`vmap` are later optimization candidates, not prerequisites for
the minimal prototype. Cross-ID vectorization is not implemented here.

## Concrete success criteria for the real machine

Before a hardware run, choose prediction frequency, maximum p95 latency,
acceptable MSE/MAE, drift size/rate, labelled sample rate, and maximum time from
measurement arrival to served version. These depend on the intended application
and cannot be inferred from a GPU name alone.

For each of 100 and 200 initialized model IDs:

- Record both GPUs' identity and VRAM capacity; driver and PyTorch/CUDA versions;
  full command and thread settings; number of model IDs and HTTP clients.
- Verify through `nvidia-smi` that the two roles use the intended devices.
- Measure initial held-out error, post-drift frozen-model error, and adapted
  error on the same unseen evaluation inputs. Track per-instance error and a
  worst-case summary, not just an average that hides poorly adapted instances.
- Sustain the required prediction and labelled-data rates together. Record
  HTTP p50/p95, rejection rate, publication delay, peak VRAM, host RAM, and
  training throughput. Include checkpoint reloads in service latency.
- Compare the same workloads with default thread availability, selected MPS
  ceilings, and an appropriate batched CPU baseline. Preserve the original
  scripts separately from corrected benchmark variants.
- Exercise trainer failure and restart on the real host; ensure the last served
  weights remain usable and clients retain/retry unaccepted labels.

Acceptance requires the chosen error/latency limits and sustainable training
throughput together. The implementation, source archive, and experiment tools
prepare that measurement; they do not substitute for it.

## Primary technical references

- [PyTorch multiprocessing best practices](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html): CUDA subprocess start methods and accelerator lifetime considerations.
- [PyTorch saving/loading](https://docs.pytorch.org/tutorials/beginner/saving_loading_models): `state_dict`, evaluation mode, and device mapping.
- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html): asynchronous execution and CUDA timing.
- [NVIDIA MPS overview](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf): client context limits and active-thread percentage semantics. Limits vary by CUDA generation; check the installed stack rather than extrapolating a universal 200-client limit.
- [SciPy MINRES](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.minres.html): symmetry requirement of the underlying algorithm used by the sparse-solver experiments.
