# Serving family contract

Ten Anthropic corpus_serving producers at baseline
805fb18e6135c9050ee7ca4295d96ddf08cb223f run through native provider adapters,
native agent/bus systems and original tools. Original constants and complete
assertion tails remain. No original AgentBuilder, runner or hook executes natively.

## Serving and concurrency

The five two_tools cells share the original streaming_tools cassette, preamble,
AlphaSignal/BetaSignal registration order, unset temperature, Sonnet4.6/golden,
max turns8, and event retention choice. Run ToolPolicy sets concurrency1 or2;
actual native Policy sets original serial_per_handler and command/stream capacities.
The original five-second timeout wraps the native full-success consumer and fails
on timeout. That consumer waits for channel closure, rejects stream errors and
requires a real RunResult, preserving final_output's per-item expect, EOF drainage
and required final response. Original nonempty output, four families, exact bus
header and tool dispatch name order assertions remain.

Native command_capacity is a per-pass dispatch intake bound, leaving other effects
pending without blocking the world; legacy command_capacity bounds its command
queue. Stream capacity bounds the actual native handler channel. The capacity-one
cell preserves both configured values and recorded behavior, not identical
backpressure, queue occupancy or memory bounds. The original cells assert dispatch
order and completion; they do not independently establish two-tool overlap or
reverse completion. Prior concurrency-family gates provide separate overlap
coverage. No performance or arbitrary scheduling claim follows from these goldens.

## Memory and model routes

serial_memory_tools uses the original InMemory backend via MemoryAdapter, registered
before model/tools, golden-conversation and native Remembers. It sets serial policy,
original temperature0/preamble/Adder, max3 and waits for actual append termination.
The answer42, five memory/model/tool effect families and serial header assertions
remain. No expected memory contents drive the run.

Routing registers real Haiku fast before Adder and retains Route metadata even
when never selected. The selected application policy observes fresh turns after
Advance/before Select and chooses fast after the first model turn. In these
sequential successful turns Cursor.turn>1 maps to the original previous_model
presence predicate. This does not assert equivalence for hypothetical selection
before a first completed model or unrelated retry/restore behavior. The default
model handles the tool call, fast handles the answer. The unselected cell has no
policy and still advertises fast in the required row. Original key-order, required
row, answer42 and full log assertions remain. Declared RouteAfterFirstTurn and
native PolicyVersion identify the interoperability composition; execution source
is separately bound by provenance.

## Host-owned bus

The App owns its bus and handler registry independently of the agent entity. Model
registers first, then Adder. Original owner, preamble/temp0, max3 and unary/stream
choices remain. declare_bus_policy=false explicitly maps the original over_bus
ownership/bus_config None and undeclared header.bus; actual native Policy remains
at defaults. This is host ownership metadata, not a changed dispatch policy.

The consumer requires all retained PendingEffects to have terminal EffectOutcome
before dropping the host App. Native tasks remain owned by effects; there is no
detached legacy driver to join. Original answer42, exact three families, absent
bus header, stamped run_spec and required.len2 checks remain. Both host cases
preserve their original fixture and full EOF/failure obligations.

## Evidence and limits

The batch selects ten exact baseline/native tests, freezes original corpus,
goldens and support source ASTs, and checks sixteen distinct original fixtures
(six HTTP cassettes shared by ten consumers, ten effect goldens). Original wrapper
matching, interaction exhaustion, golden helper and tool method assertions require
independent review in addition to hashes. Full original golden comparison uses
existing nominal identity rules; there is no new normalization. Native scopes,
program identities and raw delivery artifacts remain.

Scope is default root features/native host/replay. Full feature inventory,
backpressure/concurrency capability matrix, network barrier, performance and full
programme proof remain separate unfinished requirements.

## Reviewed runtime correction

Independent review identified that the shared success consumer's recorder-header
check missed errors after Final when event recording was disabled. Native
Streamed now retains all error items and mixed-item positions regardless of
recording; the consumer checks that public state. Original event retention and
first-fold outcomes remain unchanged. Live collection and policy replay populate
the field, completed scenes/reflection retain it, and unfinished-scene guards
consider it progress. A retention-disabled late-error control runs in all three
provider targets. Native bus tests cover absent/disabled recording, errors before
and after Final, positions, unchanged first outcome, serde and policy replay.
The shared consumer checks its whole test App, as the earlier recorder check did;
reusing that helper after a failed streamed run is not established by this family.
