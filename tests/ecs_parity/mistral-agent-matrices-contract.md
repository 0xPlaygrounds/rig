# Mistral tool-lifecycle and tool-truncation agent matrices

The 24 `Surface::Agent` cells of `tool_lifecycle_matrix` (12) and
`tool_truncation_matrix` (12) execute independently through native ECS
systems and the real Mistral adapter, on the unchanged original cassettes.
Original source edits widen sibling visibility only (`pub(super)` on the
cell enums, constants, tool types, `Observation` and `assert_cell`). The
catalog identifies each original/native pair and its fixture. Owner: the
`mistral-deepseek` lane.

## Independent execution and configuration

Each native cell uses `EcsAgent` with ordinary `BusPlugin`/`AgentPlugin`
configuration, the real `CompletionAdapter` for `mistral-small-latest` or
`ministral-3b-latest` through the per-poll Tokio bridge, and real
`ToolAdapter`s over the unchanged original tool types (`Ping`,
`RecordPayload`, `Alpha`, `Beta`; `FileReport`) sharing the original
invocation log or counter. No legacy builder, runner, policy or effect
replayer participates.

Settings mirror the original builder: `Preamble` is the matrix `PREAMBLE`,
`DefaultMaxTurns(Some(1))` mirrors `default_max_turns(1)`, `MaxTokens` is 128
(lifecycle) or the cell's cap 4/16/48 (truncation), `AdditionalParams` is the
original `tool_choice: any` object with the lifecycle shape's
`parallel_tool_calls` flag. Blocking cells pass no run override; streaming
cells pass the run override `Some(1)` that the original `stream_chat(..)
.max_turns(1)` supplies. Tools are granted in the original order. History is
empty in both runtimes.

## Endings and the tolerated-error obligation

Every agent cassette holds exactly one interaction. A one-turn budget with a
real tool call cannot settle: `advance` refuses the second model turn before
assembling a request and the run fails with `Failure::MaxTurns { limit: 1 }`.
The native cells accept only `Ok` or that variant for lifecycle cells and
truncation-complete cells; any `Provider`, `Cancelled`, `Tool`, `Unsupported`,
`UnknownToolCall`, `OutputToolCollision` or `Memory` failure fails the test. This is the original
"no `ProviderResponseError` in the errors" obligation matched on the failure
variant, and it is stricter: the original tolerated any other error string.
Truncated (low/mid) cells carry no complete call and must settle
successfully, as the DeepSeek truncation counterparts already require; native
success strengthens the original tolerated-error check.

The rendered `MaxTurns` failure and every `Streamed.errors` item become the
`errors` of an original `Observation`; the invocation log or counter becomes
its `invocations`. The unchanged original `assert_cell` then runs on that
observation: the fixture half (recorded request model, cap, `tool_choice`,
tool count, parallel flag, transport; recorded finish reason, wire call
names, ids and arguments, or the truncation premises) and the agent half
(exact-once invocation order, or the count of 1 / 0, and the error filter).
Cassette exhaustion is enforced by the same literal wrapper call and its
`?`. Native-only checks add: exactly one completion effect in the world,
materialised `ToolCallSlot` names in recorded order, one tool result per
call (lifecycle, truncation-complete), and no materialised call at all
(truncation low/mid).

## Parallel shape

The parallel cells' original assertion reads the order `alpha` then `beta`
from the tools' own log. Natively both tool effects are spawned as separate
pool tasks in one dispatch pass, so the log's push order is a race the
runtime does not promise (it held across five reruns, which proves nothing).
Concurrency therefore asserts a partial order and an exact result set: the
materialised `ToolCallSlot` order equals the recorded call order, and the
log is exactly the multiset of expected names (each tool invoked once). The
log is then presented to the unchanged original `assert_cell` in slot order.
This normalisation reorders only; it never adds, drops or deduplicates an
entry, and it applies to every shape (a no-op for single-call shapes).

## Limits

These 24 recorded cells establish provider request/response fidelity, real
tool dispatch and the budget ending for one-turn forced-tool workflows. They
do not establish effect-golden equality, hook or policy delivery, retry
behaviour, or multi-turn scheduling. The 24 `Surface::Model` cells remain
shared-provider coverage and are not counted here.
