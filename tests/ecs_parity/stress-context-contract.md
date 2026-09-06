# Gemini request-patch and context stress parity

Scope: all4 hook_stress_patch and all8 hook_stress_context cases at pinned
805fb18e6135c9050ee7ca4295d96ddf08cb223f, native host/default root features.
Original source/support/tool bodies remain unchanged; batch freezes those files
and all12 original provider cassettes. The other19 stress cases have separate
tools, streaming and main contracts and execution/review records.
These twelve have no original effect golden; every original response/probe/counter
assertion plus strict fixture matching/exhaustion is retained.

Native execution owns App, real Gemini2.5Flash adapter and original neutral
CountingAdd/CountingSubtract/CountingMultiply implementations/counters. It keeps
original preambles/prompts, name stress-agent or absent, explicit temperature0
where originally set (otherwise unset), undeclared effective default1 and exact
run override2/4/5/6. Every cell is blocking with initially empty history.

Four patch cases preserve preamble codeword, first-turn-only Required choice,
per-turn prior-fact history, and combined preamble/context. The two composition
patch cases preserve both appended context documents and active-tool intersection
{add,subtract} ∩ {add,multiply}={add}, with actual excluded tool counts0.
Each application patch system queries a Fresh turn after Select/before Assemble,
reads real Cursor for first-only behavior, and uses native RequestPatch.merge.
Explicit PatchSlot set ordering ensures the next system sees the prior deferred
patch. No original ApplyPatch/FirstTurnPatch implementation executes. Fixture
matching protects non-sticky later-turn configuration and actual outbound values.

Context observers follow actual effect->Turn->Run relationships. CompletionCall
comes from a new model intent; CompletionResponse observes a published successful
completion outcome after bus Collect/before bus Judge. A separate system observes
actual completed turn Outputs after agent Fold/before agent Judge, marking the
turn once for ModelTurnFinished. Actual issued tool slots
produce ToolCall, published tool outcomes produce ToolResult, both using the same
ToolCallSlot ID. Turn stamps read native Cursor; run identity is the native Entity;
stream mode is the actual run component. Configured optional display name is
explicit application metadata separate from the native owner. These preserve
original stability/absence/nonempty/correlation assertions without claiming the
same legacy UUID/block-ID representation or built-in HookContext API.

Agent-default taps live on the agent, request taps/readers on the actual run.
Both lists are visited per observed event, preserving the original separate
probe counts and builder/request append semantics. Each tap increments run-owned
Tally on actual tool issuance, as each original EventTap increments shared
Scratchpad. A separate reader samples that component at model-turn completion.
Original monotonicity, growth, final tally=real tool counters, and observed
ToolCall count=real executions remain exact assertions. No tally is reconstructed
from expected answers or exported transcript. Multiple tap increments retain the
original shared-scratchpad semantics, although the tally assertion cell has one tap.

The stress-main batch extended this helper with a stream-mode parameter and
read-only breadcrumbs, and separated these response/turn boundaries. Its
stress-main-context-regression report revalidates all twelve original/native
pairs against current source; the original stress-context artifacts remain bound
to the historical implementation. Streaming main coverage is reported separately.

This helper is explicit native application machinery, not an agent runtime or
legacy hook facade. Scope is blocking valid calls in one-active-run test Apps.
It does not implement streaming delta hooks, invalid-call/retry event timing,
failed settlement, arbitrary simultaneous request-scoped patch systems, or
fresh-world application-state restoration. Patch systems act on this App's
single run; no general scoping guarantee is inferred.

Negative control stress-context-tally-mutation.json replaces the reader's actual
component read with0. The real original cross-turn-growth assertion must reject
that fault; exact source restored before final paired replay. Existing stream
consumer/comparator controls remain separate evidence, not new stress claims.
An initial local generator dependency mismatch prevented registration and produced
an empty exploratory selection; that was not counted as execution. The repaired
Rust generator and exact-ID paired batch select all12 cases explicitly.

No production ECS change, paid provider call or cassette recapture. Full
inventory/features/network/WASM/interruption/capabilities/performance/aggregate,
final full review/publication and CI remain required. The other19 stress cases
are accounted for separately, not counted in this twelve-case scope.
