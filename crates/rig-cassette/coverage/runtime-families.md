# Runtime families

The cassette families that test Rig's runtime (the rig-agent loop, the
rig-ecs world, tool lifecycle, turn endings, resume, memory, the effect bus)
rather than a provider's wire. For each: what it asserts, the providers it
repeats over (fixtures per provider), what is provider-specific and what is
runtime, and where it now runs once. The once-per-scenario copies live in the
`runtime` target (`crates/rig-cassette/tests/runtime/`), over the reply bank
(`crates/rig-cassette/fixtures/bank/`, `cargo xtask cassette bank`). The
per-provider copies are kept; the prune step chooses which of them go.

A reply is either bank-matched (`bank::script`: the bank's reply of each
shape and called tools the scenario recorded) or pinned (`bank::recorded`:
the scenario's own replies, listed with a reason in `fixtures/bank/pinned.txt`)
when an assertion reads what the reply says.

## ECS contract matrix

- Where: `corpus_matrix*` producers and `ecs_matrix*` world cells over
  `tests/common/ecs_matrix/{cells,agent,world}.rs`. deepseek 85, doubleword
  85, gemini 76, openai chat 85 and responses 72, venice 80 fixtures; about
  90 producer and 90 world tests per wire. Anthropic's producers are its own
  `corpus_<family>` modules (91 fixtures), whose goldens the top-level
  `corpus_*` targets replay.
- Asserts: the ending and a hook's reason, the record's families, the
  header's hooks, the replaced answer, the world's graph, despawn and cut
  and resume; the producer's log equals the golden `<wire>_<cell>`, the
  world's its world golden.
- Provider-specific: the wire's models and thinking parameters, its ignore
  reasons, and the decode frozen in each golden. Everything asserted is
  runtime.
- Runtime target: `cells`, 122 rows, once per cell (a reasoning cell once
  per wire, because each wire carries reasoning in its own shape), rotating
  the six wires, and six cells again on the wire whose replies take a branch
  the others do not (a committed output call missing a field, an output tool
  on Venice); producer and world compared record by record. 114 are
  bank-matched, 8 pinned (an answer marker, non-empty reasoning, the
  arguments a reprompt reads).

## Focused families

- Checkpoint (`checkpoint_matrix`, 4 fixtures on anthropic, deepseek, gemini,
  openai chat and responses): exact step tokens, the completion count, cuts
  1, 2, 3 and final resumed in a fresh world. Runtime; content-dependent.
  `families::checkpoint_*`, 15 rows on four wires, pinned.
- Long loop (`long_loop_matrix`, 6 or 7 fixtures on five wires): every call
  against the repository it edits, history growth, usage, cuts. Runtime,
  with usage read per wire. `families::long_*`, 11 rows, pinned.
- Long tasks (`long_task_matrix`, deepseek 4, gemini 4, openai 8): the state
  a task leaves and a restore. Runtime; its request check
  (`long_tasks::assert_requests`) reads encoder fields and is
  provider-specific. `families::task_*`, 5 rows, pinned.
- Image (`image_matrix`, anthropic 8, gemini 6, openai 8 and 8): the image in
  every request and in history, the answer names the colour. `cells::image_*`,
  8 rows, pinned.
- Extra (`ecs_matrix_extra*`, 5 or 6 per wire over other cells' fixtures):
  batch-held approval, despawn waiting on a stream, a rejection's error
  facts. `extra`, 5 rows, bank-matched.
- Stream delivery (anthropic only, 2 tests): already once; unchanged.

## Faults

- Where: `corpus_faults*` (4 fixtures per wire) and `ecs_faults*` (20 or 21
  tests per wire) over `common/ecs_matrix/faults.rs`, on deepseek,
  doubleword, gemini, openai chat and responses, venice.
- Asserts: the report's kind, status, code and retryability, the kept stream
  prefix, the history without the cut turn, tools that never ran.
- Provider-specific: the recorded status and code, each wire's stream shape
  (where a stream's terminal sits, its error frame), refusal semantics.
- Runtime target: `faults`, 9 recorded rows once (2 setup rows on Gemini, 7
  others rotating wires; one pinned) and the 11 scripted rows once per
  stream shape (Chat, Responses, Gemini), the frames read from the bank
  (`faults::Frames::Bank`). The refusal rows stay with their wires.

## Turn termination

- Where: `turn_termination_matrix` and `ecs_termination`, 8 cells each on
  anthropic, deepseek, doubleword, gemini, openai, venice (6) and llamacpp (5
  fixtures).
- Asserts: the hook sees `Length`, `Stop` or `ToolCalls` and the cap that
  attempt ran under; the escalation retries once with each attempt's cap;
  the recorded wire reason and request cap.
- Provider-specific: how the wire spells the reason (decode), and the cap
  field the request carries (encode, pinned by the request snapshots).
- Runtime target: `termination`, 16 cells, on rig-agent and in a world, each
  over every bank reply of every provider that decodes to the cell's ending.

## Tool sessions

- Where: `agent_tool_sessions` (deepseek 10, groq 10, mistral 4, openrouter
  7, xai 8 fixtures) and `ecs_tool_sessions` (deepseek, openrouter, xai).
- Asserts: exact tool arguments, sequential call and result pairing in
  history, answer tokens; the direct rows read wire metadata (usage, ids,
  `tool_choice`, response formats).
- Runtime target: `sessions`, Groq's 10 rows over their pinned replies.
  The other wires' shared rows repeat the same runtime; their direct rows and
  the world copies are provider-specific and stay.

## Lifecycle

- Where: `lifecycle_matrix` and `ecs_lifecycle`, 5 fixtures each on
  anthropic, gemini, openai.
- Asserts: HTTP middleware phases and one exchange, run-start rewrites, the
  settle outcome, the durable counter.
- Runtime target: `lifecycle`, Anthropic's 10 rows over their pinned replies
  through the same middleware stack.

## Families that stay per provider

- `session_matrix` (5 on each of six wires): history through serde, a
  checkpoint and memory, asserted on the recorded continuation requests and
  each dialect's reasoning state. Provider-specific.
- `tool_lifecycle_matrix` (mistral 24, openrouter 24, openai chat 24): half
  its rows decode streamed call fragments on the model surface.
- `ecs_extractor*`, `ecs_ordering`, `ecs_parity`, `ecs_prompt_caching`,
  `ecs_completion`, `ecs_agent_smoke` and Anthropic's and Gemini's other
  `ecs_*` modules: world twins of wire-specific cells (prompt-cache prefixes,
  typed extraction, raw provider data).
- The long loop's scripted rows (`long_loop::Scripted`) rewrite the row-1
  recording's arguments per wire; not moved.

## Provider-independent targets

`verify` (the `corpus_*` targets, `durable_execution`, `golden_refusal`,
`golden_replay`, `interpreters_agree`, `log_header`, `record_replay`),
`world_replay` and `world_replay_world` replay effect goldens or run
`MockCompletionModel`; none reads a cassette.

## Decode

`decode` decodes every bank reply of 17 providers (all but Bedrock's 11,
which answer through the AWS SDK and have no bank transport) through its
provider's decoder: a rejection is an error, and a reply decodes to the
tools and the ending the bank read off its bytes.

## Coverage

Measured with `cargo xtask coverage --per-test` (10119 tests) over the
runtime files: `crates/rig-agent/src`, `crates/rig-ecs/src`,
`crates/rig-cassette/src/{agent,ecs,effect_log}`, and rig-core's
`operation` and `streaming`. The per-provider copies are the 1720 tests of
the families above that the runtime target replaces (Anthropic's
`corpus_<family>` producers and their world twins excluded: they write the
goldens the `verify` target replays and stay). The runtime target is 261
tests.

- The copies cover 10308 runtime lines and 1026 branches; the runtime target
  10354 and 1037.
- The runtime target alone misses 4 lines and 2 branches the copies reach:
  `rig-ecs/src/bus/record.rs` 359-362 (a delta stop that lands after the
  stream finished: the cassette socket delivered a short stream in one read,
  while the bank transport yields between frames, so its stop always lands
  first), branch `rig-ecs/src/bus/dispatch.rs` 108 (a serial handler found
  busy, which only some schedules reach), and branch
  `rig-ecs/src/systems/mod.rs` 2224 (a provider retry under a witness, from
  the long loop's scripted rows, which are not moved). All three are runtime
  timing or an unmoved row, not a provider's decode.
- The whole suite without the copies misses none of the lines or branches
  the copies cover.
