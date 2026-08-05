# PR #2258 — per-finding regression coverage map

Every finding from the last three review/fix passes on
`refactor/streaming-canonical-grammar`, mapped to the test that would fail if
the fix were reverted. Every citation below was read and executed; the exact
commands and their results are in [Verification](#verification) at the end.

## The three coverage tiers

Coverage is ranked by how much of the real system a failing test would have
exercised:

1. **Live cassette** (strongest) — a recording of real provider traffic
   replayed through the *complete* pipeline: HTTP transport → SSE/NDJSON
   decode → wire classification → adapter interpretation → `PartsAccumulator`
   → aggregated `StreamingCompletionResponse::choice` + `StreamFinal`. The
   frames are the provider's own bytes, so the test also pins *that the shape
   exists on the wire*. Lives in `tests/providers/<provider>/cassette/`, backed
   by `tests/cassettes/<provider>/`.
2. **Corpus scenario** (middle) — synthetic frames, but driven through the same
   complete pipeline via `WireDriver`/`ProviderWireFixture`. Everything below
   the transport is real. Lives in
   `crates/rig-core/src/test_utils/streaming_conformance.rs` (scenario bodies),
   invoked from `tests/core/streaming_conformance.rs` and expanded per wire
   family by `streaming_conformance_suite!` in
   `tests/core/streaming_conformance_suites.rs`.
3. **Unit test** (weakest) — exercises one adapter or one accumulator directly,
   usually against hand-written frames or a `MockStreamingClient`. Lives beside
   the code in `crates/rig-core/src/...`.

**Why many findings can only reach tier 2 or tier 3.** A live cassette requires
that a real, reachable provider *actually emits the shape*. Several classes of
finding are structurally unrecordable:

- **Malformed frames** (A4). A duplicate JSON key on the discriminator is not
  something any conforming server emits; it is precisely the frame a *broken or
  hostile* upstream would send. There is nothing to record.
- **Frames that do not exist yet** (C1). A "novel nested `content_block_delta`
  type Anthropic ships tomorrow" is by definition absent from today's wire; the
  forward-compat contract can only be probed synthetically.
- **In-band server faults** (A2). An `{"type":"error"}` mid-stream, or a
  `MALFORMED_FUNCTION_CALL` terminal followed by more frames, is a provider
  incident. It cannot be requested, and recording one would be a flake.
- **Degenerate field values** (A5, C3, C5, C6). Empty name fragments, empty
  authoritative end names, empty decoration ids, negative block indices — these
  are defensive branches against wire misbehaviour, not observed traffic.
- **Scale** (A10). The 50 000-event non-yielding run that used to overflow the
  stack is a synthetic stress shape; no turn produces it.
- **Log-output and meta-process findings** (A1, A6, C2, C7-log). What is being
  asserted is a `tracing` field, a `.gitattributes` rule, or the repo's own
  source scanner — none of which a cassette can observe.
- **Shapes the accessible models refuse to produce** (A7-strong, A8/A9, B3).
  Documented individually below with the recording evidence.

---

## Coverage table

| Finding | Fix commit | Live cassette | Corpus scenario | Unit test | Note |
|---|---|---|---|---|---|
| **A1** unknown-frame payload logging redacted to `event_type` + byte size | `5a13b98e` (batch shipped alongside `fb06cc31`) | — | `unknown_event_is_skipped` (5 suites: openai_responses, gemini_rest, gemini_interactions, anthropic, cohere) | — | Log-field assertion; nothing observes `tracing` output in this repo (grep: `payload_bytes` appears only at `adapter.rs:190` and `adapter.rs:332`, no test references it). The redaction is instead structurally enforced — `unknown_payload_bytes` returns `u64`, so the payload *cannot* be formatted into the event. The corpus scenario pins the surviving escape hatch: the frame is skipped semantically, surfaces as exactly one `Unknown` passthrough item, and does not perturb the aggregated choice (control-run byte-comparison). |
| **A2** `is_finished` after in-band errors | `fb06cc31` | — | — | anthropic `provider_error_event_stops_the_stream_before_a_later_terminal`; gemini REST `tool_protocol_failure_ends_the_stream_without_draining_later_frames`; gemini Interactions `provider_error_event_ends_the_stream_without_draining_later_frames` | An in-band provider fault followed by well-formed frames is a server incident, not a requestable shape. All three tests assert the *strong* form: the error is the last item the consumer sees — no later text, no `Unknown` passthrough, no terminal record (`stream.response.is_none()`), which is what "wasted reads" means observably. The transport-level sibling *is* covered at tier 2 by `transport_error_after_tool_call_yields_err_then_end` (all 7 suites), but that is a different failure mode (socket death, not an in-band error frame). |
| **A3** gemini thought signatures preserved (Interactions + gRPC) | `fb06cc31` | gemini `streaming_grammar/interactions_thinking_stream` → `interactions_thinking_stream_keeps_reasoning_and_text_discrete` | — | Interactions: `thought_signature_completes_the_accumulated_reasoning_block`, `signature_only_thought_still_carries_the_signature`. gRPC: `signed_thought_part_restates_accumulated_text_with_signature`, `signature_on_empty_trailer_part_still_carries_the_signature`, `signature_without_any_thought_text_still_surfaces` | The Interactions half reaches tier 1: the recorded cassette genuinely contains `{"delta":{"signature":"signature_REDACTED_1","type":"thought_signature"},"event_type":"step.delta"}`, and the test asserts a non-empty signature survives onto the aggregated reasoning *and* that the signed restatement does not duplicate the summary text (`matches(...).count() == 1`). The gRPC half has no live tier: `crates/rig-gemini-grpc` has no cassette harness at all (its only integration test is `tests/streaming_conformance.rs`), so gRPC signature handling is unit-only. |
| **A4** duplicate discriminator keys classify `Corrupt` (both classifiers) | `5a13b98e` (extended in `02f25094`) | — | — | `tagged_duplicate_discriminator_is_corrupt`, `chat_duplicate_object_discriminator_is_corrupt`, `chat_duplicate_choices_key_is_corrupt`, plus the negative control `tagged_duplicate_non_discriminator_key_still_classifies` | Malformed frames no server emits. The whole point is that `serde_json::Value` keeps the *last* duplicate key, so a spoofed `{"type":"text.delta","type":"future.event",...}` would demote a defective known frame to a skippable `Unknown`. Both classifiers (`classify_tagged_frame`, `classify_chat_completions_frame`) are covered, and the fourth test pins that duplication of *non*-discriminator keys stays tolerated (historical marker-key semantics). |
| **A5** empty tool-name FRAGMENT no longer erases an established name | `5a13b98e` | — | — | `an_empty_name_fragment_does_not_erase_an_established_name` | An empty `name` fragment mid-assembly is a wire defect; recorded traffic always sends the name once, non-empty. The test drives `tool_name_delta("call_1","get_weather")` → `tool_name_delta("call_1","")` and asserts the call still finalizes as `get_weather` rather than dropping as nameless. |
| **A6** cassette EOF whitespace | `82e0522f` (reverting `9f1d0f9e`) | — | — | — | Meta/process finding, resolved in `.gitattributes` (`tests/cassettes/** whitespace=-blank-at-eof`), not in code. The trailing blank line is *recorded SSE content* — SSE bodies end with a blank line and YAML block scalars represent it at EOF. There is nothing to unit-test; the regression check is that cassette replay still passes, which every tier-1 run in [Verification](#verification) performs (28 cassette tests across 5 providers). The original symptom of stripping it was 9 replay failures. |
| **A7** Responses same-item text split across interleaved reasoning | `c674af54` | openai `streaming_grammar/reasoning_then_text` → `reasoning_and_answer_text_aggregate_as_discrete_parts` **(new)** | `two_message_items_aggregate_as_two_text_parts_on_the_responses_wire` | `same_item_text_resumes_as_one_part_across_interleaved_reasoning` — **twice**: SSE (`responses_api/streaming.rs:2741`) and WebSocket (`responses_api/websocket.rs:1206`) | Full three-tier coverage, with one caveat on the live tier. The new cassette pins the achievable half: reasoning interleaved with the answer text, aggregating to *exactly one* text part carrying the whole streamed answer, with every reasoning part preceding it in wire order. The corpus scenario pins the complementary direction (two *distinct* `item_id`s must stay two parts — the identity that makes reactivation safe). The unit tests pin the mechanism directly: two `TextStart{msg_1}` emissions, one `"hello world"` text part. See [Non-inducible: A7's strong form](#a7s-strong-form-visible-text-before-a-function-call). |
| **A8** empty `fc_*` ids minted | `c674af54` | — | `id_less_parallel_tool_calls_assemble_distinct_on_the_chat_wire` | `parallel_id_less_function_calls_assemble_distinctly` | The corpus scenario asserts the minted identities literally (`tool-0` / `tool-1`) with uncorrupted per-slot arguments on the chat-compat wire; the unit test does the same on the Responses wire in the `output-{index}` namespace. See [Non-inducible: id-less tool-call events](#id-less-tool-call-events). |
| **A9** id-less args deltas surfaced instead of dropped | `c674af54` | — | `incomplete_mid_tool_call_ends_with_length_and_no_fabricated_call` (truncation half only) | `id_less_args_deltas_surface_and_truncation_fabricates_no_call` | The unit test covers both halves in one stream: the id-less `function_call_arguments.delta` surfaces as `ToolCallDelta{id:"output-0", ...}` (previously dropped), *and* the truncation policy still withholds the call when no `output_item.done` restatement arrives. The corpus scenario covers the truncation half on the real driver but with wire ids present. |
| **A10** `poll_next` iterates instead of recursing | `5a13b98e` | — | — | `a_long_run_of_non_yielding_events_does_not_grow_the_stack` | A 50 000-event run of non-yielding frames is a synthetic stress shape; no provider turn produces it. Pre-fix each such frame was one recursive `poll_next` stack frame and the run overflowed in debug builds. The test also asserts last-id-wins (`msg_49999`) so it cannot pass by short-circuiting. |
| **A11** Keep-mode `tool_input_end` contract note | `5a13b98e` | — | — | (nearest behavioural pin: `keep_mode_leaves_the_call_open_for_later_fragments`, `parts.rs:1095`) | **Doc-only.** The change is four lines of rustdoc on `PartsAccumulator::tool_input_end` (`parts.rs:388-391`) warning out-of-tree adapters that a `Keep`-mode end for an id that never opened still finalizes a call with `{}` arguments. No code changed, so there is nothing to regress; the documented behaviour itself is pinned by the existing Keep-mode test. |
| **B1** mixed id/id-less events splitting slot assembly keys | `02f25094` | — | — | `mixed_id_and_id_less_events_share_one_slot_key` | Requires one output slot whose `added` event carries `fc_real` but whose args delta arrives id-less — a wire inconsistency no server produces (and the reason the bug survived review as "comment-only"). The test asserts the *key set* collapses to `["fc_real"]` across name delta, args delta, and input end, then that the call finalizes with `tool_a` / `{"x":1}`. Slot-scoped `ToolCallBridge` makes key-splitting unrepresentable, so this is the whole surface. |
| **B2** single-pass frame classification | `02f25094` | (indirect: all 28 tier-1 cassette tests) | (indirect: all 145 conformance-suite tests, incl. `malformed_frame_*`, `unknown_event_is_skipped`, `defective_known_event_surfaces_err` × 7 wire families) | the whole `providers::internal::wire::tests` module — **20 tests, unchanged, all passing** | Behaviour-preserving perf refactor (three tokenization passes → one fused scan; `Known` frames now run the irreducible two passes). Equivalence is proven by *unchanged* tests, not new ones: no wire test was edited by `02f25094` and both A4 duplicate-key probes still pass. **Correction to the commit message:** it claims "all 35 wire tests"; the module actually contains 20 `#[test]`/`#[tokio::test]` functions, all of which pass. |
| **B3** gemini REST signature-only thought part emitted | `cee4d6e3` | — (see note) | `gemini_rest_signed_full_does_not_erase_prior_thought` (signed-restatement path, not the signature-only path) | `a_signature_with_no_thought_text_still_emits_a_signed_block` | **Not reachable from any recorded REST traffic.** The fixed branch requires a part with `thought: Some(true)` *and* `thoughtSignature`. A scan of all 124 gemini cassettes in this repo found no such part: the 90 recordings that contain `thoughtSignature` attach it to a non-thought part — either `{"text":"","thoughtSignature":"…"}` (the shape in `streaming_grammar/thinking_stream.yaml`) or a `functionCall` part (`streaming_grammar/thinking_then_tool_call.yaml`) — while `"thought":true` parts carry thought text and never a signature. `thinking_stream`'s own module doc records the same recording experience ("across repeated attempts … the wire only ever attaches `thoughtSignature` to a trailing **empty** text part or to a functionCall part"). That cassette therefore does **not** pin B3; it pins the adjacent non-erasure/non-duplication contract. Coverage for B3 proper is unit-only. |
| **C1** anthropic novel nested `content_block_delta` types are `Known` no-ops; known tags with defective payloads stay `Corrupt` | `66800ae2` | — | `defective_known_event_surfaces_err` (anthropic suite — the `Corrupt` half on the real driver) | `novel_nested_delta_type_is_a_known_noop`, `known_nested_delta_tag_with_defective_payload_is_corrupt` | A delta type that does not exist yet cannot be recorded — that is the definition of forward compatibility. The two unit tests pin both sides of the discrimination the hand-written `Deserialize` exists to make: `{"delta":{"type":"banana_delta","x":1}}` → `Known` + empty interpret output; `{"delta":{"type":"text_delta","text":42}}` → `Corrupt`. |
| **C2** policy-wall scanner bypass routes | `66800ae2` (scan extension in `cee4d6e3`) | — | — | `serde_wall_scopes_by_machinery_content`, `foreign_adapter_files_are_not_exempt`, `shipped_portion_ignores_cfg_test_mentions_in_comments`, plus the live scans `provider_streaming_modules_never_raw_parse_the_wire`, `every_triage_site_runs_on_the_single_policy_driver`, `serde_policy_scanner_catches_raw_parses` (all in `crates/rig-core/tests/driver_adoption.rs`) | Meta-test about the repo's own source scanner; there is no provider traffic involved. Each of the three closed routes has its own self-test: content-based scoping (a compat/sse helper opts in the moment it names the machinery, and stays out when it doesn't), full-path-suffix policy homes (a foreign `crates/rig-bedrock/src/streaming/adapter.rs` is scanned by both guards), and line-anchored `#[cfg(test)]` truncation (a doc-comment mentioning the attribute no longer exempts shipped code). The three self-tests run alongside the three real scans in the same binary. |
| **C3** authoritative empty END name no longer erases an established name | `66800ae2` | — | — | `an_empty_authoritative_end_name_does_not_erase_an_established_name` | The `Some("")` authoritative-name shape is a wire defect. The test is the end-event twin of A5: it sets `done.name = Some(String::new())` and asserts the call finalizes as `get_weather` rather than dropping as nameless. |
| **C4** `text_additional_params` closes the minted-reasoning boundary | `66800ae2` | — | (adjacent: `interleaved_constant_id_reasoning_preserves_order` × gemini_rest / interactions / ollama — the same boundary via text *deltas*) | `text_metadata_closes_a_minted_id_reasoning_item` | The only producer of `RawStreamingChoice::TextAdditionalParams` is anthropic's citations path (`anthropic/streaming.rs:625`), and the finding is specifically *standalone* metadata with no surrounding text deltas — a shape no recorded turn contains. The test asserts `reasoning_delta("reasoning-0","A")` → `text_additional_params(...)` → `reasoning_delta("reasoning-0","B")` yields two reasoning parts in arrival order, with the metadata landing on the text part between them. |
| **C5** `ToolCallBridge::decorate` empty-id matching + first-wins fields | `66800ae2` | — | — | `an_empty_id_decoration_never_matches_an_id_less_slot`, `decoration_fields_are_first_wins_per_field` | `decorate` has exactly one production caller (`openai_chat_completions_compatible.rs:438`), and both bugs need degenerate input: an empty `tool_id` (which pre-fix decorated an arbitrary id-less slot in `HashMap` iteration order) and a later params-only decoration clobbering an earlier signature. Neither is recordable. The first-wins test drives the real gemini-style signature-then-params sequence and then a third decoration that must not overwrite either field. |
| **C6** negative-index mints (`tool-n1`) agree with the provenance gate | `66800ae2` | (indirect: `openai/streaming_grammar/three_turn_tool_session` pins the gate on *real* `rs_*` ids across three turns) | — | `every_for_index_rendering_is_provenance_gated` | Negative indices originate from Bedrock's `i32` Converse content-block index; a negative index on the wire would itself be a server fault. The test is property-style over the signed domain (`i64::MIN`, `-1_000_000`, `-7`, `-1`, `0`, `1`, `7`, `1_000_000`, `i64::MAX`): every rendering passes `is_boundary_minted_id`, no minted id contains a `-` past the namespace separator, and the sign-free rendering is asserted literally (`for_index(-1i32) == "tool-n1"`). The live cassette covers the gate's other direction — real wire ids replayed across turns must *not* be treated as minted. |
| **C7** unary unparseable-args tool-call drop logs at `warn` | `66800ae2` | openai `streaming_grammar/incomplete_mid_tool_call` → `incomplete_mid_tool_call_normalizes_to_length` (**the streaming sibling of the same drop policy**, not the log level) | `incomplete_mid_tool_call_ends_with_length_and_no_fabricated_call` | — | The change itself is `tracing::debug!` → `tracing::warn!` in `responses_api/mod.rs:2207` on the *unary* `From<Output>` path; log level is not asserted anywhere (same limitation as A1). The *drop policy* it annotates is covered at tiers 1 and 2. The cassette is genuine truncated traffic — the recorded frame carries `"arguments":"{\"x\":48151"` cut mid-JSON with item and response status `incomplete` — and the tests assert a `Length` terminal with no fabricated call and no corrupted arguments. |

---

## New live cassettes recorded for this map

Both were recorded fresh, exist on disk, and pass in replay (see
[Verification](#verification)).

### `llamafile/streaming_grammar` — the compat-family identity pin

Files: `tests/providers/llamafile/cassette/streaming_grammar.rs` →
`tests/cassettes/llamafile/streaming_grammar/{single_tool_call,parallel_tool_calls}.yaml`
(registered via `tests/providers/llamafile/mod.rs`). Tests:
`single_tool_call_keeps_the_wire_id`, `parallel_tool_calls_stay_distinct`.

**Verified wire fact (recorded against Ollama's OpenAI-compat `/v1` endpoint
with `qwen3:4b`):** this wire streams each tool call as a single complete
`tool_calls` delta carrying **both** a `call_*` id **and** an `index`. The
recorded frames are literally:

```
data: {"choices":[{"delta":{"content":"","role":"assistant","tool_calls":[
  {"function":{"arguments":"{}","name":"lookup_harbor_label"},
   "id":"call_REDACTED_1","index":0,"type":"function"}]},
  "finish_reason":null,"index":0}], ... }
```

So this recording pins the **wire-id-preserving branch** of the identity family
— `ToolCallBridge::open(index, Some("call_…"), …)` keeps the wire id, parallel
calls stay distinct, arguments assemble uncorrupted — and **not** the minting
branch. The tests assert the ids start with `call_` (the prefix survives cassette
scrubbing) and are derived from the recorded turn, never minted literally.

### `openai/streaming_grammar/reasoning_then_text` — the text-boundary pin

File: appended to `tests/providers/openai/cassette/streaming_grammar.rs` →
`tests/cassettes/openai/streaming_grammar/reasoning_then_text.yaml`. Test:
`reasoning_and_answer_text_aggregate_as_discrete_parts`. This is the live tier
for A7; see the non-inducibility note immediately below.

---

## Non-inducibility notes

### A7's strong form: visible text before a function call

The strongest shape for the text-reactivation contract would be *visible
message text preceding a function call in the same turn* — text opens, a
non-text item closes it, and the text must resume as the same part.

**Not inducible on gpt-5.6.** Verified over two recording attempts under a
strict two-part output instruction: the model answers a tool-bearing turn with
the call alone and emits no visible message text before it. The reasoning
interleave recorded in `reasoning_then_text.yaml` is the closest real traffic
gets to the contract, and it is what the new test asserts:

- exactly **one** text part in the aggregated choice, not split around the
  interleaved reasoning;
- that single part carries the *whole* streamed answer (`text_parts[0] ==
  run.text`);
- every reasoning part precedes the text in wire order.

The stronger shape stays pinned synthetically by the two
`same_item_text_resumes_as_one_part_across_interleaved_reasoning` unit tests
(SSE and WebSocket), which assert the mechanism directly — two `TextStart{msg_1}`
emissions collapsing to one `"hello world"` part.

### Id-less tool-call events

**No accessible live wire emits id-less tool-call deltas.** Verified against the
recordings in this repo:

- **openai chat-compat** (`streaming_grammar_chat/parallel_tool_calls`): real
  `call_*` ids on every delta.
- **ollama /v1 compat via llamafile** (new recording, above): `call_*` ids *and*
  `index`.
- **ollama native NDJSON** (`ollama/streaming_grammar/parallel_tool_calls`): the
  recorded frames do carry `"id":"call_REDACTED_n"`, but rig's ollama typed model
  (`ollama::ToolCall`, `providers/ollama.rs:1057-1061`) has **no `id` field at
  all** — it is not deserialized. The adapter keys the call on the tool *name*
  (`ollama.rs:909-917`), so this cassette pins "parallel calls keep distinct
  identities and uncorrupted arguments on a wire whose model carries no id", not
  the `tool-{index}` mint. (The module doc on
  `tests/providers/ollama/cassette/streaming_grammar.rs` describes it as pinning
  the `tool-{index}` mint; that description does not match the current adapter.)

The `tool-{index}` minting branch is therefore reached only through
`ToolCallBridge::new()`'s default `SyntheticIds::tool()` namespace, used by
`openai_chat_completions_compatible.rs` and `rig-bedrock`. Its coverage is tier 2
(`id_less_parallel_tool_calls_assemble_distinct_on_the_chat_wire`, which asserts
`tool-0` / `tool-1` literally) and tier 3.

### B3's shape on the gemini REST wire

See the B3 row. Summary: across all 124 gemini cassettes (90 contain
`thoughtSignature`, 10 contain `"thought":true`), the two never co-occur on one
part. The signature always lands on a non-thought part. B3's fixed branch is
unit-only.

---

## How to re-record

All cassette suites take `RIG_PROVIDER_TEST_MODE=record` and must run
single-threaded (`-- --test-threads=1`) — the cassette layer is process-global.
IDs in recordings are scrub placeholders (the `call_` / `chatcmpl-` / `rs_` /
`toolu_` / `msg_` prefixes are preserved); **derive expected ids from the
recorded turn, never mint literals**.

| Suite | Command | Credentials / prerequisites |
|---|---|---|
| llamafile grammar | `RIG_PROVIDER_TEST_MODE=record cargo test --test llamafile streaming_grammar -- --test-threads=1` | local Ollama daemon, `qwen3:4b` pulled; no key |
| ollama grammar | `RIG_PROVIDER_TEST_MODE=record cargo test --test ollama streaming_grammar -- --test-threads=1` | local Ollama daemon, `qwen3:4b` pulled; no key |
| openai Responses grammar | `RIG_PROVIDER_TEST_MODE=record OPENAI_API_KEY=... cargo test --test openai streaming_grammar -- --test-threads=1` | `OPENAI_API_KEY` |
| openai chat grammar | `RIG_PROVIDER_TEST_MODE=record OPENAI_API_KEY=... cargo test --test openai streaming_grammar_chat -- --test-threads=1` | `OPENAI_API_KEY` |
| gemini grammar (REST + Interactions) | `RIG_PROVIDER_TEST_MODE=record GEMINI_API_KEY=... cargo test --test gemini streaming_grammar -- --test-threads=1` | `GEMINI_API_KEY` |
| anthropic grammar | `RIG_PROVIDER_TEST_MODE=record ANTHROPIC_API_KEY=... cargo test --test anthropic streaming_grammar -- --test-threads=1` | `ANTHROPIC_API_KEY` |

Recordings are byte-faithful: do not strip the trailing blank line at EOF (see
A6 — it is SSE content, and `.gitattributes` exempts `tests/cassettes/**` from
the blank-at-eof whitespace check).

---

## Verification

Every command below was run in this worktree
(`refactor/streaming-canonical-grammar` @ `66800ae2` + the uncommitted new
cassette suites). No test was cited that was not read *and* executed.

### Tier 1 — live cassettes

| Command | Result |
|---|---|
| `cargo test --test llamafile streaming_grammar -- --test-threads=1` | **ok. 2 passed; 0 failed** (`single_tool_call_keeps_the_wire_id`, `parallel_tool_calls_stay_distinct`) |
| `cargo test --test openai streaming_grammar -- --test-threads=1` | **ok. 14 passed; 0 failed** (incl. `reasoning_and_answer_text_aggregate_as_discrete_parts`, `incomplete_mid_tool_call_normalizes_to_length`, `reasoning_summary_stream_aggregates_each_part_once`, `encrypted_reasoning_keeps_summary_parts_and_encrypted_payload`, `parallel_tool_calls_both_survive_aggregation`, `tool_call_then_followup_text_across_turns`, `three_turn_tool_session_replays_rs_ids_across_turns`, and the 4 `streaming_grammar_chat` tests) |
| `cargo test --test gemini streaming_grammar -- --test-threads=1` | **ok. 8 passed; 0 failed** (incl. `interactions_thinking_stream_keeps_reasoning_and_text_discrete`, `thinking_stream_aggregates_all_reasoning_text`, `thinking_and_tool_call_interleave_as_discrete_parts`, `max_tokens_truncation_normalizes_to_length`) |
| `cargo test --test anthropic streaming_grammar -- --test-threads=1` | **ok. 2 passed; 0 failed** (`thinking_multi_block_turn_keeps_discrete_parts`, `parallel_tool_use_stays_distinct`) |
| `cargo test --test ollama streaming_grammar -- --test-threads=1` | **ok. 2 passed; 0 failed** (`thinking_and_tool_call_in_one_stream`, `parallel_id_less_tool_calls_stay_distinct`) |

### Tier 2 — corpus

| Command | Result |
|---|---|
| `cargo test -p rig --test core` | **ok. 169 passed; 0 failed** |
| `cargo test -p rig --test core streaming_conformance` | **ok. 145 passed; 0 failed** (7 wire-family suites × 9 canonical scenarios + anti-tamper, plus `tests/core/streaming_conformance.rs`'s wire-specific scenarios and the registry test) |

### Tier 3 — unit

| Command | Result |
|---|---|
| `cargo test -p rig-core --features test-utils,websocket --lib` | **ok. 1185 passed; 0 failed; 3 ignored** |
| `cargo test -p rig-core --features test-utils,websocket --lib providers::internal::wire` | **ok. 20 passed** (B2's equivalence set) |
| `cargo test -p rig-core --features test-utils --test driver_adoption` | **ok. 6 passed** (C2: 3 bypass self-tests + 3 live scans) |
| `cargo test -p rig-gemini-grpc --lib streaming` | **ok. 4 passed** (A3 gRPC: all three signature tests) |

Per-finding filtered runs (`cargo test -p rig-core --features test-utils,websocket --lib <filter>`), each confirming the named test exists and passes:

| Filter | Test(s) resolved | Result |
|---|---|---|
| `duplicate_discriminator` | `providers::internal::wire::tests::tagged_duplicate_discriminator_is_corrupt` | ok. 1 passed |
| `duplicate_object_discriminator` | `…::chat_duplicate_object_discriminator_is_corrupt` | ok. 1 passed |
| `duplicate_choices_key` | `…::chat_duplicate_choices_key_is_corrupt` | ok. 1 passed |
| `duplicate_non_discriminator` | `…::tagged_duplicate_non_discriminator_key_still_classifies` | ok. 1 passed |
| `an_empty_name_fragment_does_not_erase` | `streaming::parts::tests::an_empty_name_fragment_does_not_erase_an_established_name` | ok. 1 passed |
| `an_empty_authoritative_end_name` | `streaming::parts::tests::an_empty_authoritative_end_name_does_not_erase_an_established_name` | ok. 1 passed |
| `text_metadata_closes_a_minted` | `streaming::parts::tests::text_metadata_closes_a_minted_id_reasoning_item` | ok. 1 passed |
| `a_long_run_of_non_yielding` | `streaming::tests::a_long_run_of_non_yielding_events_does_not_grow_the_stack` | ok. 1 passed |
| `same_item_text_resumes` | `…responses_api::streaming::tests::…` **and** `…responses_api::websocket::tests::…` | ok. 2 passed |
| `mixed_id_and_id_less` | `…responses_api::streaming::tests::mixed_id_and_id_less_events_share_one_slot_key` | ok. 1 passed |
| `parallel_id_less_function_calls` | `…responses_api::streaming::tests::parallel_id_less_function_calls_assemble_distinctly` | ok. 1 passed |
| `id_less_args_deltas_surface` | `…responses_api::streaming::tests::id_less_args_deltas_surface_and_truncation_fabricates_no_call` | ok. 1 passed |
| `provider_error_event` | `gemini::interactions_api::…::provider_error_event_ends_the_stream_without_draining_later_frames` **and** `anthropic::…::terminal_emission::provider_error_event_stops_the_stream_before_a_later_terminal` | ok. 2 passed |
| `thought_signature_completes` | `gemini::interactions_api::…::thought_signature_completes_the_accumulated_reasoning_block` | ok. 1 passed |
| `signature_only_thought` | `gemini::interactions_api::…::signature_only_thought_still_carries_the_signature` | ok. 1 passed |
| `tool_protocol_failure_ends` | `gemini::streaming::tests::tool_protocol_failure_ends_the_stream_without_draining_later_frames` | ok. 1 passed |
| `a_signature_with_no_thought_text` | `gemini::streaming::tests::terminal_emission::a_signature_with_no_thought_text_still_emits_a_signed_block` | ok. 1 passed |
| `novel_nested_delta` | `anthropic::streaming::tests::novel_nested_delta_type_is_a_known_noop` | ok. 1 passed |
| `known_nested_delta_tag` | `anthropic::streaming::tests::known_nested_delta_tag_with_defective_payload_is_corrupt` | ok. 1 passed |
| `every_for_index_rendering` | `providers::internal::adapter::tests::every_for_index_rendering_is_provenance_gated` | ok. 1 passed |
| `an_empty_id_decoration` | `providers::internal::tool_call_bridge::tests::an_empty_id_decoration_never_matches_an_id_less_slot` | ok. 1 passed |
| `decoration_fields_are_first_wins` | `providers::internal::tool_call_bridge::tests::decoration_fields_are_first_wins_per_field` | ok. 1 passed |

### Non-test verification (grep / cassette scans)

- **A1**: `grep -rn payload_bytes --include='*.rs'` → only `adapter.rs:190`, `adapter.rs:201`, `adapter.rs:332`. No test asserts the log field.
- **A6**: `.gitattributes` contains `tests/cassettes/** whitespace=-blank-at-eof` (added by `82e0522f`).
- **A11**: the doc note lives at `crates/rig-core/src/streaming/parts.rs:388-391`; `git log -S "Out-of-tree adapters beware"` attributes it to `5a13b98e`, whose only other `parts.rs` change is the A5 fix.
- **B3**: scanned all 124 `tests/cassettes/gemini/**/*.yaml` (90 contain `thoughtSignature`, 10 contain `"thought":true`); no part object carries both. `streaming_grammar/thinking_stream.yaml`'s signature part is `{"text":"","thoughtSignature":"signature_REDACTED_1"}` with no `thought` key; `streaming_grammar/thinking_then_tool_call.yaml`'s is on a `functionCall` part.
- **Id-less tool calls**: inspected the raw recorded frames in `tests/cassettes/{llamafile,ollama}/streaming_grammar/parallel_tool_calls.yaml`; both carry `id` values on every tool call. `crates/rig-core/src/providers/ollama.rs:1057-1061` shows `ollama::ToolCall` has no `id` field, and `ollama.rs:909-917` shows the adapter keying on the function name.
- **C4**: the only production emitter of `RawStreamingChoice::TextAdditionalParams` is `crates/rig-core/src/providers/anthropic/streaming.rs:625`.
- **C5**: the only production caller of `ToolCallBridge::decorate` is `crates/rig-core/src/providers/internal/openai_chat_completions_compatible.rs:438`.
- **B2**: `git show 02f25094` touches no test in `wire.rs`; the module's 20 tests are unchanged and pass.
