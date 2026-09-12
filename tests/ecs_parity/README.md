# Agent/ECS regression comparisons

[scenarios.json](scenarios.json) indexes original test registrations and their
native ECS counterparts. It records classifications and configurations, plus a
deduplicated inventory of referenced sources, helpers and fixtures. `ecs: null`
means no native mapping is recorded. Shared-provider correspondences do not count
as agent migrations. An ignored registration or a source outside the compiled
configuration is not an executed test.

`cargo xtask check-ecs-scenarios` checks files and mapping structure. Supplying a
fresh, unfiltered root nextest JSON listing also checks mapped registrations; see
[the test guide](../README.md#agentecs-regression-scenarios). This is an index,
not a result archive or a proof of exhaustive behavioral equivalence. Models,
prompts, budgets, tool definitions and expected values belong in the tests and
their shared helpers, not a second handwritten inventory.

## Execution and comparison boundaries

Native cases use the actual ECS schedule and provider/tool adapters, with the
same HTTP recordings as their agent counterparts. Shared neutral tools,
configuration and assertion helpers are allowed; invoking the other interpreter
or replaying its expected effect answers does not establish independent native
execution. Preserve strict request matching, interaction consumption, propagated
errors and teardown requirements as well as explicit assertions. Cassette replay
does not establish an operating-system network barrier or live provider behavior.

The helpers in [tests/common](../common) define the important boundaries:

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
- `ecs_goldens.rs` compares complete logs. It maps strictly increasing nominal
  effect IDs bijectively by position, including parent and error references;
  parents must exist and precede children. Native records must have corresponding
  scoped program identities. Cross-runtime comparison omits those native-only
  scopes/programs; separate native goldens retain them. Scheduling-dependent
  delivery batches are excluded from stable golden equality. Requests, responses,
  usage, errors and positions, tool publications, descriptors, builder identity
  and causal relationships remain compared. These goldens do not certify
  delivery-sensitive policy equivalence.
- Per-run settings remain separate from agent defaults, including undeclared
  default budgets. Host bus policy can be undeclared in a comparison header
  without changing actual serving policy. Declared policy names do not hash
  application implementation or establish its correctness.

Keep HTTP recordings unchanged during ordinary regression runs. Derived native
goldens are produced by real native execution against replayed HTTP, after the
cross-runtime comparison succeeds. Do not handwrite them or normalize away a
new semantic difference.

## Family-specific obligations

Locate each original/native source pair through the catalog. The table summarizes
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
