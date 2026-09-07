# Owned replies: performance evidence

The handler returns an owned `Reply`; the driver chooses how to execute it.
The ECS driver uses the existing executor and a private bounded receiver, with
64 queue checks per effect and 4,096 across a host tick. This recovers burst
throughput and moves expensive native stream polls away from collection while
keeping the public API small.

The [reproducibility archive](reproducibility-bundle.tar.gz) contains all 88
workloads, offline provider fixtures, source reconstruction inputs, build/run
scripts, complete raw results, summaries and the experiment report. Extract it
and follow its README. Final measurements comprise 1,320 timing samples, 264
allocation samples, 135 focused repeats and 45 separate pacing diagnostics.
Exploratory results are preserved separately and are not mixed into that matrix.

Medians of five initial runs on an M2 Max, Rust 1.95.0, repository release profile:

| Workload / metric | #2443 | Original #2473 | Follow-up |
|---|---:|---:|---:|
| Ready ECS stream, completion ms | 0.251 | 6.969 | 0.361 |
| 64 ready streams, 60 Hz, completion ms | 24.345 | 2,135.594 | 37.842 |
| 1,024 ready streams, maximum tick ms | 97.064 | 13.622 | 9.817 |
| Heavy verdict, maximum tick ms | 0.382 | 67.503 | 0.328 |
| OpenAI 4 KiB, 60 Hz, completion ms | 203.843 | 5,270.274 | 203.607 |
| Sparse 4,096 streams, 1 kHz, CPU ms | 69.155 | 146.898 | 73.451 |

This is an overall design improvement with explicit costs. The 240 Hz paced
latency target failed: across 20 runs, p95 arrival was 11.506 ms versus original
#2473's 4.295 ms, exceeding the experimental one-tick allowance. Paced 60 Hz CPU
is 17.279 ms versus original #2473's 8.065 ms. The 1,024-ready workload uses
316.218 ms CPU versus 224.470 ms, while completing faster and shortening host
ticks. The writer helper retains a throughput cost despite fewer allocations.
These tradeoffs favor a single small executor policy over additional scheduling
modes. They do not establish universal speedup or latency guarantees.

The archive preserves the failed target, spread, instrumentation limits and
nonexclusive desktop conditions. Its source hashes were verified by reconstructing
each measured variant. The measured runtime matches this follow-up; later
test-only fixture/lint corrections and documentation/package exclusions do not
change that runtime. Verification status in the archived report describes the
measurement handoff; consult the PR's checks for the final committed head.

These evidence files are excluded from the published facade crate.
