# Agent/ECS regression comparisons

Original/native correspondences live in executable tests and their shared
helpers. Use a compiled nextest listing to locate a provider's test modules;
a listed or ignored registration does not establish execution or parity.
Models, prompts, budgets, tool definitions and expected values belong in the
tests, not a second handwritten inventory.

## Each runtime owns its record

A native cell's effect log is the world's own: its records carry a run
`scope`, and its header carries the run's required row and policy hash under
that scope (`rig_cassette::ecs::identity::stamp_run`). An agent producer's
log is the agent's own, with its builder spec, hook list and required row.
The two are not compared, normalized into each other, or pinned against a
second fixture. A native cell asserts what its program contracts: the run's
ending, the committed history against the last request and the answer, the
tool invocations and their results, the usage the wire reported, and the
family-specific facts named below. The agent producer asserts the same
contract on its side and writes the golden that replay consumes.

The world interpreters under `crates/rig-cassette/tests/corpus/` and
`tests/world_replay.rs` are replay, not comparison: they answer every effect
from a golden by id and check the bus reproduces the trace it was given.

## Execution and comparison boundaries

Native cases use the actual ECS schedule and provider/tool adapters, with the
same HTTP recordings as their agent counterparts. Shared neutral tools,
configuration and assertion helpers are allowed; invoking the other interpreter
or replaying its expected effect answers does not establish independent native
execution. Preserve strict request matching, interaction consumption, propagated
errors and teardown requirements as well as explicit assertions. Cassette replay
does not establish an operating-system network barrier or live provider behavior.

The helpers in [shared test drivers](../../test-support/rig-test-support/src) define the important boundaries:

- `ecs_agent.rs` waits for actual settlement and a run result. Its success path
  rejects all public stream errors, including errors after a terminal or with
  event recording disabled. Collector selection of a last final does not prove
  terminal uniqueness. Expected failures must inspect actual failure categories,
  budgets, reasons and run-owned history, rather than manufacture an agent error.
- Native `Settled` precedes possible memory-append acknowledgement. Comparisons
  to an agent response wait for the real append outcome, accepting either
  `Appended` or a recorded append error where the original returns its answer.
  Clear-at-start/settled policies await their own acknowledgements. Success is
  not a durability or exactly-once external-write guarantee.
- Application gates and observers in these helpers are generally scoped to one
  active run per test world. They are not a general concurrent HookStack or a
  fresh-world persistence framework. Host notes and nested calls must complete
  before teardown; gated controls establish those waits independently of quick
  cassette responses.
- `goldens.rs` compares an agent producer's log to its committed golden as
  data, header included. No helper compares a native log to that golden.
- Per-run settings remain separate from agent defaults, including undeclared
  default budgets. Host bus policy can be undeclared in a comparison header
  without changing actual serving policy. Declared policy names do not hash
  application implementation or establish its correctness.

Keep HTTP recordings and original goldens unchanged during ordinary regression
runs. Never weaken a native cell's assertions to make it pass.

## Family-specific obligations

Locate each original/native source pair in the provider's test modules. The table summarizes
the distinctions worth preserving when editing those tests; executable assertions
and fixtures remain authoritative for case-specific values and configuration.

| Families | Distinct observations and limits |
| --- | --- |
| Smoke, additional provider completions | Preserve successful nonempty/marker predicates and provider-specific configuration. A final answer alone proves neither tool invocation counts nor exact answer equality. Stream smoke also checks actual provider final/usage where asserted. |
| Agent goldens, request shape | Preserve full header and payload equality under the comparator rules above, including undeclared settings and per-run overrides. Required-tool budget failures remain failures. |
| Output modes, shaping | Use native mode resolution, validation and retries. Preserve tool/native degradation, output names/schema, real-tool collisions, non-sticky patches, ordered patch merging, route membership and required-row versus signature distinctions. |
| Extractor smoke and usage | Require actual submit output, deserialization and observed usage. Preserve each shared field validator's exact acceptance rules: `Usage::has_values()` does not imply every counter is positive. Each extraction is an independent run; these blocking zero-retry cases do not establish streaming or retry accumulation. |
| Tool sessions, Chat tool lifecycle | Observe actual invocation logs and ordered run history. Preserve arguments, tool/result correlation, per-turn identity and final markers; wire assertions differ from runtime invocation assertions. Chat required-tool cells intentionally end at MaxTurns after executing tools. Native JSON extraction retains its tested first-value parsing/error rules; the image case retains actual image bytes and media type. Direct model cases remain shared-provider coverage. |
| Gemini tools | Preserve real counters, exactly correlated calls/results, integer/JSON/string result semantics and protocol hygiene. Generated call IDs need not be globally unique; correlate within their actual turn. |
| Gemini streamed access and diagnostics | Observe and resolve native invalid calls; repair continues the same stream, skip preserves abandoned history/usage, and failures retain actual diagnostics. Drain issued completions where the original requires it. Synthetic gated tests, not cassette timing, establish intervention before EOF and persistence of repairs. |
| Turn termination, Anthropic reasoning/stop matrices | Reconcile provider finish reasons with actual tool content. Preserve tiny/roomy cap transitions, rejected-attempt observation before retry, recorded stop-sequence/empty-output checks and provider-specific usage arithmetic. Blocking and streaming assertions may differ. |
| Cache growth | Inspect successful actual completion usage in turn order through the shared validator, retaining prefix and provider breakpoint assertions. Recorded cache ratios do not establish live TTL, economics or broader caching guarantees. |
| Hooks, layers | Gate patches/denials affect actual dispatch; Judge replacements affect consumption while the recorder retains the inner answer. Await startup effects before model advancement. Preserve middleware order, observed memory load and original versus replaced values. |
| Endings, outcomes | Distinguish run cancellation, issued-effect despawn and provider failure. Preserve actual settled/failure observations, partial stream prefixes and error item positions. Exact cancellation prefixes rely on test-owned backpressure; they do not establish unrestricted transport timing equivalence. |
| Serving, concurrency | Preserve dispatch versus semantic result publication order. Gated tools demonstrate overlap; ordinary serial/concurrent golden equality does not. ECS command intake limits and agent command queues are different mechanisms. Capacities do not bound all buffering, CPU or payload memory. |
| Host custom effects, causal calls | Wait for actual acknowledgements and child completion, retain full causal relationships and original tool payload predicates. An absent native handler refuses an unissued intent; that is not an identical bind API. Completion nesting does not cover every effect family or arbitrary concurrent application gating. |
| Memory | Reuse real backends across repeated runs; explicit history bypasses load/append without losing required-row identity. Preserve operation/message order, append-error behavior and every recorded scope. These tests do not imply arbitrary backend durability. |
| Lifecycle | Observe real startup, model and settlement boundaries and wire middleware phases. Entry serialization is not fresh-world restoration; incidental elapsed time and test names do not add timing guarantees. |
| Gemini stress: context, tools, streaming, main | Keep ordered patch/Gate/Judge systems, real invocation tallies, distinct response/turn boundaries and call-result correlation within each turn. Committed tool results come from materialized history, not issuance alone. Preserve each original collector's error/final obligations; single-active-run controls do not establish general interruption or concurrent application scoping. |

The [ECS runtime contract](../../crates/rig-ecs/CONTRACT.md) documents supported
library semantics. Its replay, cancellation and snapshot guarantees are separate
from the narrower observations of any one provider family above.
