# Runtime families

The cassette families that test Rig's runtime (the rig-agent loop, tool
lifecycle, turn endings, resume, memory, the effect bus)
rather than a provider's wire. For each: what it asserts, the providers it
repeats over (fixtures per provider), what is provider-specific and what is
runtime, and where it now runs once. The once-per-scenario copies live in the
`runtime` target (`crates/rig-cassette/tests/runtime/`), over the reply bank
(`crates/rig-cassette/fixtures/bank/`, `cargo xtask cassette bank`). The
per-provider copies the runtime target makes redundant were deleted by the
cassette prune (`cargo xtask cassette prune`, listed in `pruned.tsv`); the
copies that remain hold a request fact, a reply shape the reply bank does
not, or a region nothing else does. The fixture counts below are the
repetition each family had before the prune.

A reply is either bank-matched (`bank::script`: the bank's reply of each
shape and called tools the scenario recorded) or pinned (`bank::recorded`:
the scenario's own replies, listed with a reason in `fixtures/bank/pinned.txt`)
when an assertion reads what the reply says.

## Corpus matrix

- Where: the `corpus_matrix*` producers over
  `tests/common/corpus_matrix/{cells,agent}.rs`. deepseek 85, doubleword 85,
  gemini 76, openai chat 85 and responses 72, venice 80 fixtures; about 90
  producer tests per wire. Anthropic's producers are its own
  `corpus_<family>` modules (91 fixtures), whose goldens the top-level
  `corpus_*` targets replay.
- Asserts: the ending and a hook's reason, the record's families, the
  header's hooks, the replaced answer; the producer's log equals the golden
  `<wire>_<cell>`.
- Provider-specific: the wire's models and thinking parameters, its ignore
  reasons, and the decode frozen in each golden. Everything asserted is
  runtime.
- Runtime target: `cells`, 122 rows, once per cell (a reasoning cell once
  per wire, because each wire carries reasoning in its own shape), rotating
  the six wires, and six cells again on the wire whose replies take a branch
  the others do not (a committed output call missing a field, an output tool
  on Venice); each runs the producer with every assertion it makes. 114 are
  bank-matched, 8 pinned (an answer marker, non-empty reasoning, the
  arguments a reprompt reads).

## Focused families

- Checkpoint (`checkpoint_matrix`, 4 fixtures on anthropic, deepseek, gemini,
  openai chat and responses): exact step tokens, the completion count, the
  tool arguments and outputs. Runtime; content-dependent.
  `families::checkpoint_*`, 4 rows on four wires, pinned.
- Long loop (`long_loop_matrix`, 6 or 7 fixtures on five wires): every call
  against the repository it edits, history growth, usage. Runtime, with
  usage read per wire. `families::long_*`, 7 rows, pinned.
- Image (`image_matrix`, anthropic 8, gemini 6, openai 8 and 8): the image in
  every request and in history, the answer names the colour. `cells::image_*`,
  8 rows, pinned.
- Stream delivery (anthropic only, 2 tests): already once; unchanged.

## Faults

- Where: `corpus_faults*` (4 fixtures per wire) over
  `common/corpus_matrix/faults.rs`, on deepseek, doubleword, gemini, openai
  chat and responses, venice.
- Asserts: the report's kind, status, code and retryability, the kept stream
  prefix, the history without the cut turn, tools that never ran.
- Provider-specific: the recorded status and code, each wire's stream shape
  (where a stream's terminal sits, its error frame), refusal semantics.
- Runtime target: `faults`, 6 recorded rows once (2 setup rows on Gemini, 4
  others rotating wires) and the 9 scripted rows once per stream shape
  (Chat, Responses, Gemini), on rig-agent's runner, the frames read from
  the bank (`faults::Scripted`). The refusal rows stay with their wires.

## Turn termination

- Where: `turn_termination_matrix`, on venice and llamacpp.
- Asserts: the hook sees `Length`, `Stop` or `ToolCalls` and the cap that
  attempt ran under; the escalation retries once with each attempt's cap;
  the recorded wire reason and request cap.
- Provider-specific: how the wire spells the reason (decode), and the cap
  field the request carries (encode, pinned by the request snapshots).
- Runtime target: `termination`, 8 cells on rig-agent, each over every bank
  reply of every provider that decodes to the cell's ending.

## Tool sessions

- Where: `agent_tool_sessions` (deepseek 10, groq 10, mistral 4, openrouter
  7, xai 8 fixtures).
- Asserts: exact tool arguments, sequential call and result pairing in
  history, answer tokens; the direct rows read wire metadata (usage, ids,
  `tool_choice`, response formats).
- Runtime target: `sessions`, Groq's 10 rows over their pinned replies.
  The other wires' shared rows repeat the same runtime; their direct rows
  are provider-specific and stay.

## Lifecycle

- Where: `lifecycle_matrix`, 5 fixtures each on anthropic, gemini, openai.
- Asserts: HTTP middleware phases and one exchange, run-start rewrites, the
  settle outcome, the durable counter.
- Runtime target: `lifecycle`, Anthropic's rows over their pinned replies
  through the same middleware stack.

## Families that stay per provider

- `tool_lifecycle_matrix` (two cells each on mistral, openrouter and openai
  chat): streamed call fragments decoded on the model surface.
- The long loop's scripted rows (`long_loop::Scripted`) rewrite the row-1
  recording's arguments per wire; not moved.

## Provider-independent targets

`verify` (the `corpus_*` targets, `durable_execution`, `golden_refusal`,
`golden_replay`, `interpreters_agree`, `log_header`, `record_replay`)
replays effect goldens or runs `MockCompletionModel`; it reads no cassette.

## Decode

`decode` decodes every bank reply of 17 providers (all but Bedrock's 11,
which answer through the AWS SDK and have no bank transport) through its
provider's decoder: a rejection is an error, and a reply decodes to the
tools and the ending the bank read off its bytes.

## Coverage

`cargo xtask coverage --per-test` measures each test over the runtime files:
`crates/rig-agent/src`, `crates/rig-cassette/src/{agent,effect_log}`, and
rig-core's `operation` and `streaming`. The per-provider copies are the tests
of the families above that the runtime target replaces (Anthropic's
`corpus_<family>` producers excluded: they write the goldens the `verify`
target replays and stay). The cassette prune keeps a copy only when it holds
a region the runtime target does not reach.
