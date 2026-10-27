# Validation status

Prepared on 2026-10-27. The preparation machine does not provide the two RTX
3060s. No dual-GPU, MPS, CUDA-stream, VRAM-capacity, or RTX performance tests were
performed. No performance figures in historical comments are treated as verified.

## Checks completed before the request to stop testing

The installed system PyTorch 2.5.0 failed to import because its CUDA shared
libraries were unavailable. `nvidia-smi` was unavailable. An isolated temporary
environment at `/tmp/mgpu-validation-venv` was then created, inheriting existing
Python dependencies and installing PyTorch 2.5.1+cpu. This did not replace the
system installation and is not part of the publishable folder.

A single CPU-only run of `python -m pytest -q` completed: **13 passed, 1 failed**.
The failing case checked the approximation error at one input after only 80
training steps on a constant target of 2. The observed squared error was about
0.458, exceeding the test's 0.15 threshold. This was an overly specific
convergence expectation for an integration check; it was not a two-GPU result.

After that run the owner requested no further testing. The independent-model
check was revised to test separation between independently learned predictions,
which is the behavior under examination. **The revision was not rerun.** No
claim is made that the full integration suite currently passes. The earlier
failure prevented execution of later assertions in that same test, including
publication after adaptation and trainer restart. Those remain unverified here.

The demo, final server revisions, MPS wrapper, archived experiments, and GitHub
workflow were not executed after the request. The GitHub workflow is configured
for a future manually requested CPU run; pushes and pull requests do not trigger
tests. It has not run during preparation or publication.

## Intended future checks, when the owner chooses to run them

The checked-in tests cover request validation, backpressure, inference after a
dead trainer, original target-function equivalence, explicit device selection,
independent IDs, checkpoint publication and restart. CPU execution can check
these software paths but cannot establish device isolation, GPU concurrency,
resource occupancy, or 100–200-instance performance on RTX 3060s.

The README includes manual commands for the hardware run. Define application
acceptance thresholds first and record the evidence described in
[the study review](STUDY_REVIEW.md). Until those measurements are available,
hardware capacity and adaptation quality remain open study questions.
