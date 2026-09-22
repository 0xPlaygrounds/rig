# CONTRACT — ECS agent behavior and replay evidence

This reference describes request assembly, steering, persistence and replay in
`rig-ecs`. Tables link behavior to maintained corpus fixtures and regression
tests; they do not establish equivalence for arbitrary application systems.
The provider comparison harness and its narrower observation boundaries are
explained in [the test guide](../../tests/ecs_parity/README.md).

Concrete log interpretation lives in `rig-cassette` with its `ecs` feature:
`rig_cassette::ecs::{Replay, ReplayDelivery, ReplayFailure, EffectLogResource}`
and `rig_cassette::ecs::identity`. Add its `ReplayPlugin` after the runtime,
then call `Replay::register(&mut World, &EffectLog)` before replaying effects.
The runtime owns scheduling, observation and reflected world checkpoints;
it has no normal dependency on cassette or the classic agent runtime.

## 1. The request: a walk over the graph

`CompletionRequest` fields are serialized in the effect request (`record_telemetry_content` never appears in a recorded request). Each is derived by `policy::fold_request` from one source in the graph, in one order. Source entities and components are `rig_ecs::agent`'s.

| field | source (the walk) | order | pinned by |
|---|---|---|---|
| `model` | none: always `null`; the model is the handler key the effect is dispatched to (`UsesModel` on the run, else the agent, → the handler entity's `Bound.key`) | — | every golden `/records/0/kind/request/model` |
| `chat_history[0]` | the effective preamble as `system`, when there is one: the agent's `Preamble` (the run's override first) joined with the output mode's augmentation by `"\n\n"` (§3); no preamble and no augmentation is no system message | first | `anthropic_completion_smoke` `/…/chat_history/0`; `anthropic_request_shape_without_preamble` (no system message); `anthropic_request_shape_append_preamble` (the preamble's own `"\n"` join is the program's, stored already joined) |
| `chat_history[1..]` | every `Utterance` `ChildOf` the run, in sibling (`Children`) order, reconstructed from its typed content children in sibling (`Children`) order: the prior history the run was spawned with, then the prompt, then — turn by turn — the assistant utterance `Materialise` spawned and the reprompt utterance it added | sibling (`Children`) order | `anthropic_request_shape_prior_history` `/…/chat_history` (system, user, assistant(id null), user); `mock_output_tool_text_reprompt` `/records/1/…/chat_history` (…, assistant text, user reprompt); `mock_output_tool_missing_field_reprompt` `/records/1/…/chat_history` (…, assistant call, user tool result) |
| `documents` | the turn's `Attachment` links, in sibling (`Children`) order, each to a document entity (`DocumentId`, `DocumentText`, `DocumentProps`) — the agent's `Context` links are attached to every turn by `Advance` | link sibling (`Children`) order | `anthropic_request_shape_static_context` `/…/documents` (`static_doc_0`, `static_doc_1`; no `additional_props` key when empty) |
| `tools` | the turn's `Advert` links, in sibling (`Children`) order, each to a tool handler entity whose `Bound.descriptor.family` is `Tool { name, description, parameters }` — the agent's `Grant` links, advertised by `Advance`; then the output tool (§3) when the resolved mode is `Tool` | grant sibling (`Children`) order, output tool last | `anthropic_request_shape_tool_choice_none` `/…/tools/0` (`add`, its description and parameters verbatim from the descriptor); `anthropic_output_tool_unary` `/…/tools/0` (`final_result`) |
| `temperature` | `Temperature` (run, else agent) | — | `anthropic_completion_smoke` (`null`), `anthropic_output_tool_unary` (`0.0`) |
| `max_tokens` | `MaxTokens` | — | `anthropic_request_shape_max_tokens` (`32`) |
| `tool_choice` | `ToolChoiceSpec`, unchanged (`"none"`, `"required"`, `{"specific":{"function_names":[…]}}`) | — | `anthropic_request_shape_tool_choice_none`; `anthropic_output_tool_choice_required`; `anthropic_output_tool_choice_specific_output` |
| `additional_params` | `AdditionalParams`, verbatim | — | `anthropic_request_shape_thinking_unary` (`{"thinking":{"type":"enabled","budget_tokens":1024}}`) |
| `output_schema` | `Output.schema` when the resolved mode is `Native`; `null` otherwise | — | `anthropic_request_shape_output_schema_unary` (the schema verbatim, unsorted); `anthropic_output_prompted_unary` (`null`); `anthropic_output_tool_under_none_degrades` (the schema: `Tool` degraded to `Native`) |
| `stream` (the effect's, beside the request) | the run's `StreamRequested` | — | every `*_streamed` golden |

The fold is the only constructor of a `CompletionRequest` in the crate (`policy::fold_request`; `tests/core/dependency_graph.rs` guards what rig-ecs may depend on).

A run is spawned through `systems::RunCommands` — `spawn_run(agent, history,
prompt, streamed, max_turns)` on `World` (at once) or on `Commands` (the
entity reserved, the work queued: the run exists when the commands apply,
and `Advance` sees it on the first schedule pass after that). Either form
spawns the run — `Run` `#[require]`s `Cursor`, `OutputRetries`,
`InvalidRetries`, `ProviderRetried`, `OutputToolName`, `Usage` and
`StreamRequested`, and its `on_add` hook stamps `RunSeq`, `Scope` and
`Name` — with `RunOf`, the `Prompt` component and an optional `MaxTurns`, the
history utterances `ChildOf` the run in sibling (`Children`) order, then
`Ready`, then opens the run. A host assembling a run by hand spawns
`(Run, RunOf(agent), Prompt, ..., Ready)`, its utterances, and writes
`Ready` last. `open_runs` (first in
`RigSet::Advance`) opens every `Ready` run that has no phase and no
ending: the `Prompt` is spawned as the run's last utterance and taken off
the run, then the first phase — `Assembling`, or `LoadingMemory` with the
conversation's `Load` effect for an agent that `Remembers` given no history
(§11). `Advance` takes only `Ready` runs in `Assembling`: before `Ready`
nothing reads the run, so assembly never sees a half-populated prompt or
history. Pinned by `tests/run_commands.rs`.

A user utterance's content is the caller's, verbatim: `spawn_run` takes a
`Prompt` (a user message's parts), and those parts — text and image, in the order given, each
image's bytes or URL, media type and options unchanged — are what every
request of the run carries, what memory appends and loads (§11), and what a
checkpoint saves and restores (§13); the assistant's canonical part order
(`rig_core::message::ordered_assistant_content`) does not apply to user
content. Pinned by the `*_image_*` goldens
(`inline_mixed_order`: text, image, text, the same image, text; `inline_tool_unary`:
the image in the second request and after a checkpoint load; `inline_followup`: the
image loaded from memory, once, before a text-only prompt).

Content entities carry one reflected `agent::content::parts::ContentPart` enum:
`Text`, `Image`, `Audio`, `Video`, `Document`, `ToolCall`, `Reasoning`, `ToolResult`,
or `Json`. Query the component and match its variant; media metadata structs
are fields, not independent components. A `ToolResult` owns ordered `Text`,
`Image`, or `Json` child entities. `Role` and the assistant's `MessageId` belong
to the utterance. Sibling order is `Children` order. Conflicting payload types
are unrepresentable; missing required components, children on leaf parts,
nested tool results, and invalid role/content combinations are rejected.
`read_message` and the `ContentGraph` system parameter reconstruct transport DTOs
at conversion boundaries. `write_message` replaces persistent utterance content;
it does not implement a request-only patch. Individual component edits affect
that part only; reordering/reparenting changes subsequent graph reads.

`BinaryAssets` shares raw and base64 image/audio/video/document payloads by
SHA-256 of decoded bytes. Per-use `PartSource` keeps URLs, file IDs, literal
strings, explicit unknown sources, media/detail metadata, and the binary source
representation. Base64 padding and noncanonical trailing symbols are preserved
without retaining the entire spelling again. Repeated spellings use a bounded,
transient hash index to avoid another decode while indexed. Default limits are
64 MiB per payload, 256 MiB retained decoded bytes, and 65,536 distinct assets.
Invalid binary input fails the run as `Failure::Content`, before model dispatch.

Collection is explicit: `collect_binary_assets` scans every content entity in
the world, including other runs and conversation owners, and also retains the
host's supplied pins. A missing root refuses collection without changing the
store. Despawning one owner does not itself drop a shared payload. A host should
collect after releasing owners/pins; otherwise the retained store remains subject
to its allocation limit. `run_content_binary` and `run_content_parts` cover source spelling,
all existing content variants, nested JSON/image results, metadata and shared
asset lifetime.

History is rendered from the entity graph every turn: `gather_turn` walks each
utterance's part subtree anew, so a graph edit is in the next request with no
cache to invalidate.

## 2. The verbatim strings

| string | value | pinned by |
|---|---|---|
| the output tool's name | `final_result`; on a collision with a granted tool's name, `final_result_1`, `final_result_2`, … | `anthropic_output_tool_unary` `/…/tools/0/name`; numbering: `rig_agent::run::prepare` docs (no golden collides) |
| the output tool's description | `Call this tool exactly once with your final answer when you are done. Its arguments are the structured result and must satisfy the output schema.` | `anthropic_output_tool_unary` `/…/tools/0/description` |
| the output tool's parameters | the program's schema, unchanged | `anthropic_output_tool_unary` `/…/tools/0/parameters` |
| the tool-mode augmentation | ``When you have gathered enough information to answer, call the `{name}` tool exactly once with your final answer. Its arguments are the structured result and must satisfy the required schema. Do not return the final answer as plain text.`` | `anthropic_output_tool_unary` `/…/chat_history/0/content` |
| the prompted augmentation | `Respond with ONLY a single JSON object that conforms to this JSON Schema. Do not include any prose, explanation, or markdown code fences.` + `"\n"` + `to_canonical_string(schema)` (keys sorted, no whitespace) | `anthropic_output_prompted_unary` `/…/chat_history/0/content` |
| the augmentation separator | `"\n\n"` between the preamble and an augmentation | the two rows above |
| the text-answer reprompt | ``Provide your final answer by calling the `{name}` tool with the structured result as its arguments, not as plain text.`` — a plain user message | `mock_output_tool_text_reprompt` `/records/1/…/chat_history/3` |
| the missing-field reprompt | ``The `{name}` arguments were missing required field(s): {a, b}. Call `{name}` again with every required field.`` — a tool result on the call (`call`, `provider`, `name`, one text part) | `mock_output_tool_missing_field_reprompt` `/records/1/…/chat_history/3` |

`OutputToolConfig` overrides the default name and description above and can
disable tool-mode preamble augmentation. Settings resolve run first, then agent;
an explicit default component on the run resets the agent's customization.
Without a configured name, automatic collision-safe naming still applies.
With a schema, a reserved name commits `Tool` mode regardless of the requested
mode or tool choice. A granted tool using a reserved or already minted name
causes `Failed(OutputToolCollision)` before provider dispatch. Later configuration
changes cannot rename an already minted output tool. Without a schema,
configuration alone does not advertise a tool. Checkpoints and replay
identity include all three configuration fields. These contracts are exercised
by `tests/run_output_tool_config.rs`; the root extractor-usage cassette counterparts
exercise the original `submit` name, extraction description, and unaugmented
preamble through real provider adapters.

## 3. Output modes

`resolve_output(mode, has_schema, granted_tools, callable, provider_composes_native)`, never `Auto`:

| program | resolved | request | pinned by |
|---|---|---|---|
| no schema | `Native` (plain text) | no augmentation, no output tool, `output_schema: null` | `anthropic_completion_smoke` |
| `Native` + schema | `Native` | `output_schema` set | (by construction; no golden asks `Native` explicitly) |
| `Auto` + schema, provider composes native output with tools (`Bound.descriptor.family.capabilities.composes_native_output_with_tools`) | `Native` | `output_schema` set, no augmentation | `anthropic_request_shape_output_schema_unary` |
| `Auto` + schema + a granted tool + a permitting choice, provider does not compose | `Tool` | as `Tool` | `rig_agent::run::prepare` docs (no golden in this set) |
| `Tool` + schema, choice permits | `Tool` | the output tool advertised last, the augmentation, `output_schema: null` | `anthropic_output_tool_unary`, `gemini_breadth_output_tool_unary`, `openai_output_tool_unary` |
| `Tool` + schema, `tool_choice: none` | `Native` (degrades; the constraint is still enforced) | `output_schema` set, no augmentation, `tools: []` | `anthropic_output_tool_under_none_degrades` |
| `Prompted` + schema | `Prompted` | the augmentation with the canonical schema, `tools: []`, `output_schema: null` | `anthropic_output_prompted_unary`, `mock_oracle_prompted_unvalidated` |

`callable`: `None`/`Auto`/`Required` permit; `None` forbids; `Specific` permits iff it names the output tool (`anthropic_output_tool_choice_specific_output`). The mode is pinned on the turn (`systems::Folded`) once folded.

## 4. Reading the answer (`Materialise`)

The set is a chain, one system per transition, in run (`RunSeq`) order:
`land_memory` (§11), `resolve_invalid_defaults`, `land_batch` (§8.1),
`record_usage` (the completion's usage into the run's `Usage`, once),
`judge_invalid_calls` (§8.2), `read_turn` (`Materialised`; a provider
retry or failure, §5; an invalid call as an entity; else the turn's content
and its granted tools on the turn as `systems::TurnRead` for the rest of
the pass), `materialise_assistant` (the assistant utterance, a `Retry`,
§9.4, or the empty answer), `materialise_batch` (§8.1),
`materialise_reprompt` (the two reprompts while the budget lasts),
`materialise_answer` (the run settles on what is left). A content error
writing the graph — an image the store refuses, a part subtree that does
not render — ends that run `Failed(Content)` and no other: the chain goes
on to the next run in the same pass
(`run_content_parts::a_content_failure_ends_its_run_and_the_next_run_is_read_in_the_same_pass`).

| the turn | what happens | pinned by |
|---|---|---|
| a provider error | the run fails `Provider(report)` at the record | `anthropic_outcome_model_error` (`provider_response`, 401) |
| the effect despawned mid-stream | the record is `Cancelled`; the run is `Failed(Cancelled)` | `anthropic_cancelled_stream`, `anthropic_outcome_cancel_after_tool_call_delta` |
| an empty turn (no parts, or one unannotated empty text) | not history; the run settles with `""` | `anthropic_request_shape_tool_choice_none` (`choice: []`, the run settles) |
| text, mode `Native`/`Prompted` | the assistant utterance is history; the run settles with the text (prompted output is never validated) | `anthropic_completion_smoke`; `mock_oracle_prompted_unvalidated` (`"not an object"`) |
| the output tool called with every required field | the record retains the call; committed history preserves non-tool parts (including reasoning) and replaces tool calls with the arguments as JSON text; the run settles with those arguments serialised | `anthropic_output_tool_unary`; the six-wire `output_tool_thinking` cells; `output_tool_history_preserves_reasoning_and_commits_arguments_as_text` |
| the output tool called without a required field, reprompts left (`OutputRetries < 1`) and turns left | history gains the assistant call and a user tool result (the missing-field reprompt); another turn | `mock_output_tool_missing_field_reprompt` |
| text where the output tool was due, reprompts left | history gains the assistant text and the user reprompt; another turn — unless the text already parses as JSON with every required field of the schema: then it is the answer, as the model gave it through the wrong channel | `mock_output_tool_text_reprompt`; `openai_chat_shaping_tool_choice_none_on_committed_output` and `openai_responses_shaping_tool_choice_none_on_committed_output` (the JSON text under `tool_choice: none` settles the run) |
| the output tool called with a missing field, or text where it was due, with **no reprompt left** | the run settles with what it has (the partial arguments, or the text) — unpinned: no golden exhausts the budget; rig-agent's docs accept a text that already satisfies the schema and are silent on the rest | (none) |
| a granted tool named like the minted output tool | `Failed(OutputToolCollision)` | `rig_agent::run::prepare` docs |
| a call to a tool neither granted nor the output tool | an `InvalidCall` entity `ChildOf` the turn; `resolve_invalid_defaults` writes the run's `InvalidCalls.unhandled` unless a system wrote a `Resolution` first; `Fail` → `Failed(UnknownToolCall)`; `Ignore` → the call is dropped, what is left is the answer (an empty rest settles as `""`) | `mock_outcome_invalid_call_unhandled`, `mock_invalid_mixed_fail` (`Fail`); `mock_invalid_ignore_unary`, `mock_delta_ignore` (`Ignore`) |
| a call to a granted tool | dispatch and resolve the tool batch (§8) | `anthropic_tool_call_turn` |
| `Retry`, `Repair`, `Skip` written as a resolution | apply the invalid-call policy (§8.2) | `mock_invalid_tool_call_recovery`, `mock_invalid_repair_to_add`, `mock_invalid_skip_under_auto` |
| reasoning reported by a provider | reasoning stays out of the result and its complete payload survives history and scene persistence; usage preserves the wire counter, an absent counter (Venice's reasoning count) is `None`, and Responses' encrypted-only capped block with no textual delta | `venice_reasoning_text_unary`, `venice_reasoning_tool_unary`, `openai_responses_reasoning_capped_streamed` |
| a turn that delivered no answer (no text, no call, no image; reasoning is not an answer) and stopped with a truncating reason (`Length`, `ContentFilter`: `FinishReason::truncated_output`) | the run is `Failed(Provider(report))`, `kind: response`, the message `FinishReason::no_answer_message` — rig-agent's rule (rig#2322), one wording; nothing is committed on either runtime, a reasoning-only turn included | every `ecs_faults` `filtered_empty` cell; rig-agent's `a_truncated_reasoning_only_turn_commits_nothing` |
| a provider's refusal of the prompt (the REST Gemini wire's `promptFeedback.blockReason` `SAFETY` / `BLOCKLIST` / `PROHIBITED_CONTENT`; the gRPC wire does not read `prompt_feedback` yet) | the run is `Failed(Provider(report))` with `kind: provider_response`, `refusal: true`, `retryable: false`, `code` the block reason, no status; a refusal the model *says* (OpenAI's `refusal` deltas) is an answer, read as usual | `gemini::cassette::ecs_faults::refusal`, `ecs_stream_faults::blocked_prompt_is_a_provider_refusal_not_a_truncation` |

## 5. Budgets and endings

| rule | pinned by |
|---|---|
| `MaxTurns` counts model calls; `Advance` fails a run whose `Cursor.turn` reached it (`Failed(MaxTurns { limit })`); the default is 1 | `rig_agent::run::spec` docs (`effective_max_turns`) |
| the output-tool reprompt budget is 1 | `rig_agent::run::spec` docs (`DEFAULT_OUTPUT_RETRIES`) |
| every record of a `MaxTurns` run is a success; the run is not | corpus `Ending::MaxTurns` docs |
| a completion whose outcome is a provider `ErrorReport` with `retryable` set, while `ProviderRetried < ProviderRetries` (the run's, else the agent's, else 3), is re-issued: the lost turn is read (`Materialised`) and leaves no utterance, `ProviderRetried` counts up, the run is `Assembling` under `ProviderRetrying`, and `Advance` spawns the next turn without checking or spending `MaxTurns`; the witness gets `rig-ecs/agent/provider_retry` (attempt, budget, reason) | `tests/run_provider_retry.rs` |
| a provider's reply becomes an `ErrorReport` through the one funnel (`rig_core::provider_response`): `kind: provider_response`, `http_status` the reply's, `retryable` by the status table (`rig_core::error::retryable_status`), `code` the transport's own when it gave one apart from the body, else the string the body names under `error.code`, `error.status` or `error.type` (`ProviderResponseError::machine_code`), else `None`; the body itself stays on `provider_response.body` | every `ecs_faults` setup cell; `provider_response::tests` |
| a non-retryable report, a cancellation, or a spent budget ends the run `Failed(Provider)` with that report, as before; no tool is ever re-run by a retry, and the log holds every attempt as its own effect | `tests/run_provider_retry.rs` |
| a backoff is a hold on the re-issued effect: with `agent::Backoff { base, max }` on the run (else the agent), `systems::backoff::hold_retries` (after `RigSet::Patch`, before `Release`) holds the retry turn's completion as `rig-ecs/backoff` the pass it is folded and a `bevy_time` delayed command releases it after `base × 2^(attempt-1)` (≤ `max`, `systems::backoff::RetryAttempt(n)` on the retry turn) on the world's clock — a paused `Time<Virtual>` holds it until the host advances the clock, so a replay sees no wall time; without `Backoff` the retry is re-issued on the next pass, and a host `Gate` system may hold it itself | `tests/run_provider_retry.rs` (`a_backoff_on_the_agent_delays_the_retry_on_the_worlds_clock`, `a_host_hold_is_where_a_backoff_goes...`) |
| `RunCommands::cancel_run(run, reason)` writes `Cancelled(reason)` (§9.1) — at once on `World`, when the commands apply on `Commands`; a cancel queued behind a `spawn_run` ends the run before its first pass, and no request is ever made; an ended run keeps its ending; a non-run is left alone; a run cancelled before it opened (`Ready` by hand, no pass yet) fails with its unread `Prompt` still on it, and a checkpoint saves that prompt with the failed run | `tests/run_commands.rs` (`a_cancel_queued_behind_the_spawn_ends_the_run_before_it_starts`) |
| `RunCommands::despawn_run(run)` takes an ended run and its whole graph out of the world; refused (`RunBusy::{NotARun, Unsettled, InFlight}`, nothing despawned) while the run has not ended or an effect of it is in flight or unanswered. The `World` form returns the refusal; the `Commands` form triggers `RunDespawnRefused { entity, reason }` on the run when the command applies, `NotARun` on an id an earlier command already despawned. A run despawned before or while `spawn_run` populates it (a host `Add<Run>` observer refusing it) is simply gone: the steps after the despawn do nothing, and no utterance is spawned under it | `tests/run_lifetime.rs`; `tests/run_commands.rs` (`a_queued_despawn_is_refused_by_event_while_the_run_lives`, `a_reserved_run_despawned_before_it_was_populated_is_gone`, `a_second_queued_despawn_of_a_gone_run_is_refused_as_not_a_run`) |

## 6. The header

A rig-agent golden's header carries the builder configuration as its `run_spec` hash (`rig_cassette::effect_log::stable_hash`, keys canonicalised). A world writes no builder header: its log names each run's program under its scope (§10), and the two headers are never compared. The world corpus at `crates/rig-cassette/fixtures/effects/world/` pins the world's own header and scoped program identities. Effective run compatibility uses `spec_json(world, run)`, `stamp_run` and scoped program identity.

```json
{"preamble": <Preamble>, "static_context": [{"id","text",…props}], "additional_params": <AdditionalParams>,
 "max_tokens": <MaxTokens>, "temperature": <Temperature>, "tool_choice": <ToolChoiceSpec>,
 "max_turns": <DefaultMaxTurns or 1>, "max_invalid_tool_call_retries": 0, "output_schema": <Output.schema>,
 "output_mode": "Auto"|"Native"|"Tool"|"Prompted", "output_tool_name": null, "output_tool_description": null,
 "augment_output_preamble": true, "unhandled_invalid_tool_call": "fail"}
```

The builder `/header/run_spec` excludes per-run overrides: `mock_delta_fail` and `mock_delta_ignore` share that builder hash. This does not imply interchangeable runs: scoped identity includes supported effective settings, including turn budget, stream mode and invalid-call policy; application-specific semantics require a nonempty `PolicyVersion` (§10).

`required`: the model's key as `completion`, every granted tool's key by its family (`anthropic_request_shape_tool_choice_none` `/header/required`; a world's row is under `programs[scope].required`). `hooks`: the agent program’s declared hooks and layers (§10). `signature`: written by the recorder from what was dispatched.

## 7. Keys

`<owner>/model:<label>` (`golden/model:default`), `<owner>/tool:<name>#<n>` (`golden/tool:add#0`), `<owner>/memory`. `rig_cassette::ecs::identity::{model_key, tool_key}`; `HandlerKey::parts()` reads them.

## 8. Tools and batches

A call to a granted tool is an effect entity `ChildOf` the turn; the turn's tool children are the batch; the results are one user utterance. Every row cites the golden and JSON pointer that pin it; the batch semantics are `rig_agent::agent::engine::drive_tool_calls`'s docs and `rig_agent::run::AgentRun::tool_results`'s, read as specification.

### 8.1 The batch

| the turn | what happens | pinned by |
|---|---|---|
| one call to a granted tool | `Materialise` spawns one `PendingEffect { key: the advert's handler key, kind: ToolCall { name, args } }` `ChildOf` the turn, `args` the call's arguments rendered by `serde_json::Value::to_string`; the effect carries `bus::ToolInputs(context)` — the run's `ToolContextSpec` (else the agent's, else empty) as `for_dispatch()`; the run is `ResolvingTools` | `anthropic_tool_call_turn` `/records/1` (key `golden/tool:add#0`, `args` `{"x":17,"y":25}`) |
| several calls in one turn | one child per call, in call order; the record order is the call order under every serving policy and concurrency | `anthropic_concurrent_tools_serial` `/records/1..2`; `anthropic_serving_{serial_concurrency_one,concurrent_concurrency_one,concurrent_concurrency_two,capacity_one}` (four policies, one trace) |
| `ToolPolicy { concurrency }` (the run's, else the agent's, else 1) | `release_batch`, before the bus's `Gate`, holds every tool child beyond the concurrency as owner `rig-ecs/batch` with `BatchHeld` for slot accounting, then releases its ownership in call order as earlier calls land; 1 is serial. Policies acquire and release distinct stable owner names through `bus::acquire_hold` / `bus::release_hold`. `Held` remains while any owner holds it; repeated acquisition by one owner is idempotent. A policy-held call within the concurrency keeps its slot. Bare `Held` barriers remain independently unknown. The batch's marker and owner entry go with the hold: a host that approves a batch-held call by removing `Held` or by releasing `rig-ecs/batch` dispatches it and the batch counts it as active, so a scene saved afterwards loads (`approving_a_batch_held_call_by_any_route_keeps_the_batch_and_the_scene_consistent`). | `a_gate_hold_on_a_call_the_batch_also_holds_survives_the_batch_release`; `named_owners_release_independently_without_changing_dispatch`; `rig_agent::AgentRunner::tool_concurrency` docs |
| every child has an outcome | one user utterance with one `ToolResult` part per call, in call order: `tool_result_output(call.id, call.provider, name, result.output())` — the output's content items verbatim; then the run is `Assembling` | `anthropic_tool_call_turn` `/records/2/kind/request/chat_history/3` (`content: [{"type":"json","value":42}]`); `anthropic_concurrent_tools_serial` `/records/3/…/chat_history/3` (two parts) |
| a tool answered `status: error` — a `Tool::Error` of any type and a `ToolExecutionError` alike, since `ErasedTool::execute` renders every tool's `Err` into the result before the adapter answers; an error the tool's `map_error` marks a refusal is `status: refused` instead | the part is the error's model output; the run goes on | `anthropic_outcome_tool_error` `/records/2/…/chat_history/3` (`the adder is broken`); every `ecs_faults` `tool_error` cell |
| a `Denied` outcome (a layer's `deny`, a `Gate` system's `EffectOutcome(Err(Denied))`) | a skipped result: one text part, the report's message; no record for a `Gate` denial | `mock_leftovers_denied_tool` `/records/1/…/chat_history/3` (`denied by the host`); `mock_layers_suspend_deny` |
| a `Cancelled` outcome (a child despawned, a `Judge` rewrite to `Err(Cancelled)`) | the run is `Failed(Cancelled)` once the batch has landed; nothing is committed to history | `anthropic_endings_tool_outcome_cancelled` (`[Completion, Tool]`, the tool's real result in the record) |
| `BusClosed`, `HandlerUnavailable` or `Divergence` | the run is `Failed(Tool(report))`: a replay that went on after an answer the record never gave would be a passed test with another trace | `rig_agent::agent::engine::dispatch_tool_call` docs |
| any other `Err` report | a failed result: `ToolExecutionError::other(message).with_model_feedback(message)`, so the part is the message | `mock_layers_wrong_family_patch` `/records/1/…/chat_history/3` (the layer's `Internal` message) |
| an `Ok` of another family | a failed result: `the tool handler answered with a {family} outcome` | `rig_agent::agent::engine::dispatch_tool_call` docs |
| the batch lands at the turn budget | the batch still runs; `Advance` fails the run `MaxTurns` when it wants the next turn | `anthropic_outcome_max_turns_exhausted` (`[Completion, Tool]`, then `MaxTurns { limit: 1 }`) |
| the output tool's call in a turn beside a granted tool's call | unpinned: no golden; the batch runs and the output tool's call is read when it lands | (none) |
| a replaced result | history holds the replacement, the record the handler's answer (a `Judge` rewrite of the child's `EffectOutcome`) | `anthropic_hooks_replace_tool_result` `/records/1/outcome` (`42`) vs `/records/2/…/chat_history/3` (`99`) |
| a patched call | the record holds the patched arguments, history the model's | `anthropic_hooks_patch_tool_args` `/records/1/kind/args` (`{"x":40,"y":2}`) vs `/records/2/…/chat_history/2` (`{17, 25}`) |
| the status, as data | beside each `ContentPart::ToolResult` the batch lands, a `ToolResultStatus`: `Ok`, `Error` (a `status: error` result; any other `Err` report above), `Refused`, `Skipped` (a `skipped` result; every synthetic result — §8.2 feedback, the invalid-peer notice, an output-tool reprompt), `Denied`, `WrongFamily`. Graph data only: `read_message` and the request are the same whatever the status; a result written from a DTO (`write_message`, a memory load, prior history) has none; a scene saves it (`tool_result_status`) and refuses one off a tool-result part | `tool_result_status.rs`; the goldens above, unchanged |
| the size policy | `ToolResultLimit { max_bytes, marker }` on the run (else the agent; absent is verbatim): `gather_turn` cuts each *text* item of each tool-result part longer than `max_bytes` to its head and tail — together at most `max_bytes`, each on a UTF-8 character boundary (the head floors, the tail start ceils; an odd budget gives the tail the extra byte) — around `marker` with `{omitted}` the omitted byte count (`TOOL_RESULT_LIMIT_MARKER` by default). JSON and image items, assistant and plain user text are never cut. Request shaping (like `RequestPatch`, §9.3): applied to the folded DTOs after the turn's `RequestPartEdit`s (an edited text is what is measured and cut; a removed part is gone) and before the fold; history, the graph, memory and a scene keep the full text (§1, §13); a `RequestPatch.history` replacement is the host's own and not cut; a change between turns affects later turns only; not replay identity (§10) | `tool_result_limit.rs`; `policy::tests::the_tool_result_cut_keeps_at_most_the_limit_on_character_boundaries` |

### 8.2 Invalid calls beside the batch

For an active streamed completion, `discover_streamed_invalid_calls` publishes
an unknown delivered `ToolName` before EOF. A system in `RigSet::Judge` can
write its `Resolution`. The call retains the actual delivered assistant prefix
and event position. Effective failure, including exhausted retry, takes effect
immediately. Repair, Ignore, Skip and retry with budget wait for the producer's
real completion; its usage enters the run total once. Repair follows core block
assembly to the final provider identity. Skip/retry retain the early prefix,
not arguments delivered after the decision. Ignore suppresses later names on
that open call but does not suppress a later reuse of the same block identifier.
The controlled tests in `run_stream_boundary` verify these boundaries, including
completed tool blocks without prior name deltas and names buffered with EOF.
Discovery stops at the first retained stream error: later name events cannot
replace that earlier provider failure or trigger invalid-call policy. A name
delivered before the error remains actionable; a deferred repair does not wait
for the discarded event tail after that error.
They are synthetic scheduling evidence, separate from provider-cassette parity.

`ToolAccess` separates executable name-to-handler bindings and allowed names from
the actual advertisements. Configure it on the run or agent; `None` bindings use
advertised handlers, `None` permissions use executable names, and an explicit
empty set denies ordinary tools without changing the provider request. Explicit
bindings may reference unadvertised handlers. Assembly saves the effective
snapshot on each turn; changing the run policy affects later turns. That
snapshot exposes executable and allowed diagnostic sets after a failure and
survives a checkpoint. Reserved output names cannot collide with execution bindings;
automatically selected output names avoid both bindings and advertisements.
Replay identity includes explicit run/agent access and handler descriptors;
required rows also retain dependencies from saved turns after a policy change.

`Resolution` gains its payloads here: `Retry { feedback }`, `Repair { to }`, `Skip { reason }`. Written by a user system before `Materialise` (§9), or by `resolve_invalid_defaults` from `InvalidCalls.unhandled` (`Fail` / `Ignore`).

| resolution | what happens | pinned by |
|---|---|---|
| `Retry { feedback }`, retries left (`InvalidRetries < InvalidCalls.retries`) | history gains the assistant utterance (the turn as the model gave it) and a user utterance of tool results: for the invalid call the feedback, for every other call of the turn `Tool not executed because another tool call in the same assistant turn was invalid.`; nothing is dispatched; another turn | `mock_invalid_tool_call_recovery` `/records/1/…/chat_history/2..3`; `mock_hooks_retry_twice` (two retries) |
| `Retry`, no retries left | `Failed(UnknownToolCall)` | `rig_agent::run::AgentRun::resolve_invalid_tool_call` docs ("while budget remains") |
| `Repair { to }` | the call's name becomes `to`; `to` must be a granted tool (else the call stays invalid and is judged again); the assistant utterance carries the repaired name; the call is dispatched as a granted call with the model's arguments | `mock_invalid_repair_to_add` `/records/1` (key `add`, `args` `{"x":2,"y":3}`), `/records/2/…/chat_history/2` (`"name":"add"`) |
| `Skip { reason }` | the invalid call's result is the reason (one text part); no call of the turn is dispatched, each other call's result is the invalid-peer text above; refused under `tool_choice: none` (`Failed(UnknownToolCall)`) | `mock_invalid_skip_under_auto` `/records/1/…/chat_history/3` (`no such tool; skipped`); `mock_invalid_skip_under_none` (`UnknownToolCall`, one record) |
| `Ignore` beside a valid call | the invalid call is dropped from the turn; the valid call is dispatched; the assistant utterance carries the valid call only | `mock_invalid_mixed_ignore` `/records/1` (`add` dispatched), `/records/2/…/chat_history/2` |
| `Fail` beside a valid call | `Failed(UnknownToolCall)` at the completion record; nothing dispatched | `mock_invalid_mixed_fail` (one record) |
| a retry under `tool_choice: required` | the retried turn calls the tool; the budget ends the run `MaxTurns` | `mock_invalid_retry_under_required` (`[Completion, Completion, Tool]`, `MaxTurns { limit: 2 }`) |

### 8.3 Nested dispatch

A tool served by a system (a world-served handler) answers by submitting `WorldOutcome::new(outcome)` on its effect entity; `Collect` publishes the `EffectOutcome` in submission order; what it dispatches on the way is a `PendingEffect` it spawns `ChildOf` the effect it serves, so the child's record names the tool's record as `parent` and inherits the run's `Scope`.

| cell | the world | pinned by |
|---|---|---|
| a completion nested from the tool | the tool's system spawns `Completion { request, stream: false }` on the model's key `ChildOf` the tool effect and answers when the child lands | `anthropic_causal_completion_serial` `/records/2/parent` = 2 |
| a note, a relay (depth two) | likewise, the relay's system spawning the note `ChildOf` the relay | `mock_causal_depth_two` `/records/2..3/parent` (2, 3) |
| the same key under serial serving | the child is refused before dispatch (`Request`, no record); the tool answers `refused:Request` | `mock_causal_same_key_serial_refused` (three records), `mock_causal_same_key_from_thread_refused` (the threaded case exercises `rig_agent::bus`; in the world the refusal is the same query) |
| the same key under concurrent serving | served: `T, T←parent` | `mock_causal_same_key_concurrent_served` `/records/2/parent` = 2 |
| the parent cancelled with the child in flight or queued | the run despawned once the never-answering handler was reached: the tool effect and its child are `Cancelled` (the queued second child never began, no record) | `mock_causal_parent_cancelled_child_in_flight`, `mock_causal_parent_cancelled_child_queued` (`[Completion, Tool✗, Custom✗←2]`) |
| the detached resolver | a system answers later; the same records | `mock_causal_detached_resolver` |

### 8.4 Tool context beside the effect

`EffectKind::ToolCall { name, args }` and `Outcome::ToolResult { result }` carry no context. The inbound values reach the adapter through the dispatch scope (`Dispatch::scope::<ToolContext>()`), attached by the driver — the world from the effect entity's `bus::ToolInputs`, `rig_agent::bus` from the dispatch's own context — and the values a tool published come back beside the reply (`rig_core::tool::PublishedContext`), which the world reads into `bus::ToolOutputs` when the outcome lands. Published context is recorded separately from the tool outcome and restored by the consumer’s replay projection. Applications must separately persist their own state and any approval decisions needed for continuation.

## 9. Steering: every hook is a system

No hook trait: a user system writes a component at a set boundary and a library system reads it later. The cases in `crates/rig-cassette/tests/corpus/world_hooks.rs` exercise the boundaries below. The moments, in schedule order: `On<Add, Run>` (run start) · a system after `RigSet::Advance` and before `RigSet::Select` (model selection) · before `RigSet::Assemble` (the completion call: `RequestPatch`, a hook's own dispatch) · `RigSet::Patch` (the folded effect) · the bus's `Gate` (a dispatch: deny, patch, hold) · the bus's `Judge` (an outcome: replace) · after `RigSet::Fold` (deltas) · `RigSet::Judge` (the model turn: retry, replace, stop) · before `RigSet::Materialise` (an invalid call) · `On<Add, Settled>` / `On<Add, Failed>` (run settled).

### 9.1 Stopping: `Cancelled(reason)` on the run

| the stop | the write | what the library does | pinned by |
|---|---|---|---|
| any hook's `stop(reason)` | `agent::Cancelled(reason)` inserted on the run, at any moment before it ended (after `Settled` or `Failed` it is a no-op: a run has one ending) | the observer `run_cancelled` fails the run `Failed(Cancelled(report))` with `report.message == reason` and `kind == Cancelled`, removes its phase marker (`Assembling`, `AwaitingModel`, `ResolvingTools`, `LoadingMemory`), marks its current turn read, and despawns every effect of the run that was never issued (never dispatched: no record); an effect in flight is left to its handler, so the record is the handler's — a stream that ended stays a completion, one still streaming ends as the replayer or provider ends it | every `Ending::Cancelled` cell; the reason asserted |
| `on_run_start` → stop | an `On<Add, Run>` observer inserts `Cancelled` | no record | `mock_endings_stop_at_start` (`[]`, `stopped at run start`) |
| `on_model_select` → stop | a system between `Advance` and `Select` | no record: the turn is fresh, the effect not yet folded | `mock_endings_stop_at_model_select` |
| `on_completion_call` → stop | a system before `Assemble` (on `Added<Fresh>`), or in `Patch` | no record: the completion effect is despawned before `Dispatch` | `mock_endings_stop_at_completion_call` |
| `on_dispatch` → `Deny(Cancelled)` on a tool | a system in the bus's `Gate` on the tool child | the tool child is never issued: `[Completion]` | `anthropic_endings_tool_dispatch_cancelled{,_streamed}`, `gemini_breadth_tool_dispatch_cancelled`, `openai_breadth_tool_dispatch_cancelled` |
| `on_outcome` → stop on a tool result | `On<Add, EffectOutcome>` on the tool child, or a system in the bus's `Judge` | the tool's record holds its real answer; nothing is committed | `anthropic_endings_tool_outcome_cancelled{,_streamed}` (`[Completion, Tool]`) |
| `on_outcome` → stop on an answer | a system in `RigSet::Judge` | `[Completion]`, the real answer in the record | `anthropic_endings_answer_outcome_cancelled` |
| `on_model_turn_finished` → stop | a system in `RigSet::Judge` (before `Materialise`) | the turn is not history | `anthropic_endings_turn_finished_stop{,_streamed}`, `anthropic_endings_answer_turn_stop` (`[C, T, C]`), `anthropic_oracle_stop_after_turn_two` (the stateful hook: `Cursor.turn == 2`) |
| a delta stop | a system after `RigSet::Fold` on `Changed<Outputs>` (text), `Changed<Streamed>` (a tool-call delta: `Delta::ToolName` / `Delta::ToolArguments` among the new events) | the record is the handler's timing: the replayer ends the stream as it was recorded (`Cancelled` where the producer dropped it, a whole completion where the mock had already finished) | `anthropic_endings_text_delta_stop`, `anthropic_endings_tool_call_delta_stop`, `gemini_breadth_text_delta_stop`, `openai_breadth_text_delta_stop` (`Cancelled`, events kept); `mock_delta_stop_on_name`, `mock_delta_stop_on_arguments` (whole) |

### 9.2 Selecting: `UsesModel` on the run

| the hook | the write | pinned by |
|---|---|---|
| `on_model_select` → `select(label)` | a system after `Advance`, before `Select`, inserting `UsesModel(the route's handler entity)` on the run; `Select` copies the agent's only when the run has none, so a route persists until replaced — a system that routes one turn re-inserts the default on the next; the route is declared on the agent by a `Route` link (`agent::Route(entity)`, `ChildOf` the agent) so the required row names it, or bound after the agent exists and not in the row (`late_route`) | `anthropic_serving_model_route` (`fast` after the first turn: `Cursor.turn > 1`), `anthropic_shaping_route_on_first_turn` (`fast` on turn 1 only), `anthropic_shaping_late_route` (`late` on every turn; `/header/required` without it) |

### 9.3 The completion call: `RequestPatch` on the turn

`content::parts::RequestPartEdit` adds entity targeting: spawn a link
`(RequestPartEdit, EditTarget(part), ChildOf(fresh_turn))` before Assemble.
`Text` replaces only a `ContentPart::Text`'s text, preserving annotations; `Remove` omits the
part and its nested result items from this request. Stored history and siblings
are unchanged. Edits run in sibling (`Children`) order; the last edit to a
target wins. A removed parent takes precedence over edits to its children.
Targets must belong to this run's history. Missing targets, wrong types and
combination with replacement RequestPatch.history fail before
model dispatch. Successful folds consume the edit links; a checkpoint
remaps them when saved before folding. Gate/Judge systems can query these same
part entities and edit their typed components persistently; those writes affect
subsequent folds, while an already captured PendingEffect remains a request
snapshot. Effect denial uses the existing Gate outcome path and does not mutate
content implicitly. `run_content_edits` covers targeting, ownership and remapping.


`agent::RequestPatch` (the corpus's `rig_agent::agent::RequestPatch` as data: `preamble`, `temperature`, `max_tokens`, `tool_choice`, `active_tools`, `additional_params`, `extra_context`, `history`) inserted on the fresh turn before `Assemble` (a system on `Added<Fresh>`, reading `Cursor.turn` for the turn number); `gather_turn` folds it in as `prepare_request` did. Several hooks patching one turn merge in registration order (`RequestPatch::merge`: `extra_context` appends, object `additional_params` shallow-merge with later keys winning, `active_tools` intersect, scalars and `history` last-writer-wins); a user system that finds a patch on the turn merges over it.

| field | what the fold does | pinned by |
|---|---|---|
| `preamble` | replaces the preamble the system message is built from (the augmentation still applies) | `anthropic_hooks_preamble_override` `/records/0/…/chat_history/0`; `anthropic_shaping_preamble_second_turn` `/records/2/…/chat_history/0` (turn 2 only; turn 1 the agent's) |
| `history` | replaces the utterances (the system message stays) | `anthropic_shaping_history_first_turn` `/records/0/…/chat_history/1..3` |
| `extra_context` | appended after the turn's attachments | `anthropic_shaping_extra_context{,_streamed}` `/records/0/…/documents/0` (`shaping-context`) |
| `max_tokens`, `temperature`, `additional_params` | replace the setting for the turn (an object `additional_params` merges over the agent's) | `anthropic_shaping_max_tokens_second_turn` (`5` on turn 2), `anthropic_shaping_thinking_second_turn` (`1.0`, thinking on turn 2) |
| `tool_choice` | replaces the choice for the turn; a committed output tool (`OutputToolName` minted) stays advertised whatever the choice | `anthropic_shaping_tool_choice_required_first` (`required` on turn 1), `anthropic_shaping_tool_choice_none_on_committed_output` `/records/2/…/tools` (`add`, `final_result` under `none`; then a reprompt) |
| `active_tools` | narrows the adverts to the names listed | `anthropic_shaping_active_tools_none_second_turn` `/records/2/…/tools` (`[]`) |
| three hooks on one turn | the merge above | `anthropic_shaping_merged_three` (the pirate preamble, the document, `required` on turn 1) |

### 9.4 The model turn: `Retry` and a replaced answer

| the hook | the write | what `materialise_assistant` does | pinned by |
|---|---|---|---|
| `on_model_turn_finished` → `retry_with_feedback(text)` | `agent::Retry { feedback: Some(text) }` on the turn, in `RigSet::Judge` | the turn (unless empty) and a user utterance of the feedback become history; another turn; nothing is committed as an answer; text turns only (a tool-bearing turn is refused, `Failed(Unsupported)`); an empty turn with a retry asks again instead of settling on the empty answer | `anthropic_hooks_demand_done` `/records/1/…/chat_history/2..3`; `steer_hooks::a_retry_written_on_an_empty_turn_asks_again` |
| `repeat` | `Retry { feedback: None }` | nothing becomes history; another turn | `rig_agent::run::AgentRun::retry_model_turn` docs |
| `on_outcome` → replace a completion | a system in `RigSet::Judge` rewriting the turn's `Outputs.content` (or the bus's `Judge` rewriting the `EffectOutcome`) | what is read is the replacement; the record holds the model's | `anthropic_hooks_replace_answer` (the run's output `REPLACED`, the record's text) |

### 9.5 A hook's own dispatch

A `PendingEffect` the system spawns `ChildOf` the run at the hook's moment (its record names no parent — the run is not an effect — and carries the run's `Scope`):

| moment | the write | pinned by |
|---|---|---|
| run start | `On<Add, Run>` observer: before the first completion is folded, so its `Seq` is lower | `anthropic_host_custom_at_start{,_streamed}` (`[Custom, Completion]`), `anthropic_host_custom_twice_{serial,concurrent}` (`[X, X, C]`), `anthropic_hooks_lookup_before_run` (`[Tool, C, Tool, C]`: a tool call `add(1, 2)` on the tool's key), `openai_host_embed_prompt{,_streamed}` / `gemini_breadth_embed_prompt` (`Embed { inputs: Texts([prompt]) }`), `mock_oracle_rerank` (`Rerank { query: prompt, documents }`), `mock_leftovers_five_thousand_events` (two hundred notes) |
| before a completion | a system on `Added<Fresh>` before `Assemble` (the note's `Seq` precedes the completion's) | `anthropic_host_custom_at_completion_call` (`[X, C]`) |
| after a tool answered | `On<Add, EffectOutcome>` on a tool child | `anthropic_host_custom_at_outcome{,_streamed}`, `{gemini,openai}_breadth_custom_at_outcome` (`[C, T, X, C]`), `anthropic_oracle_concurrent_notes` (`[C, T, T, X, X, C]`) |
| settled | `On<Add, Settled>` observer; the effect it dispatches is taken on the app's next update | `anthropic_host_custom_at_settled` (`[C, X]`), `anthropic_host_custom_start_and_settled` |
| a key nothing serves | the system finds no `Bound` for the key and dispatches nothing | `anthropic_host_custom_unserved` (`[C]`) |
| an effect with no wire form | `PendingEffect::custom` refuses it; nothing is spawned | `mock_leftovers_unserializable_from_hook` (`[C]`) |

### 9.6 Layers

A layer is the handler's: the world registers the layered `ErasedHandler` (`handler.layered(intercept)`) exactly as the corpus runner's `Replay::open` does, under the key the header names. A decision the layer makes before the handler is served leaves no record, and a verdict after leaves the handler's answer in the record: the world's `Dispatch` installs a recording observer (`bus::WorldObserver`, its slots on the entity as `bus::Observed`) for every task-served handler — the one way a layer's `discard` and `patch` reach any recorder, and the innermost handler's outcome the observer is told is what `settle` records (`bus_world::a_layers_decisions_reach_the_record_through_the_sinks_observer`). The suspending layer (`ApprovalLayer`) is answered by a thread the interpreter spawns as the program says.

| cell | records | pinned by |
|---|---|---|
| a denying layer on the tool | `[C, C]`, the reason as the skipped result | `anthropic_layers_deny_tool`, `anthropic_layers_host_deny_over_host_bus`, `mock_layers_suspend_deny`, `mock_leftovers_denied_tool{,_streamed}` |
| a patching layer | the record holds the layer's arguments | `anthropic_layers_patch_tool_args`, `anthropic_layers_patch_beneath_hook_patch` (the hook's patch in `Gate`, the layer's beneath it: `{30, 12}`) |
| a replacing layer | the record holds the handler's answer, history the replacement | `anthropic_layers_replace_tool_result`, `anthropic_layers_two_layers` |
| a wrong-family patch | `Internal`, no record, a failed result the model sees | `mock_layers_wrong_family_patch` |
| a cancelling layer on the model | `[C]` with events; `Failed(Cancelled)` at the record | `mock_layers_replace_streamed_cancelled` |
| a denying layer on the model, on memory | `[]`; `Failed(Provider(Denied))` / `Failed(Memory)` | `mock_leftovers_denied_completion`, `mock_leftovers_denied_memory_load` (§11) |
| a denied note from a hook | the system's note is answered `Denied`; the run goes on | `mock_leftovers_denied_custom_from_hook{,_streamed}` |
| the suspended tool cancelled | `[C, T✗]`: the run despawned while the layer waits | `mock_layers_suspend_cancelled` |

## 10. Identity as data

The world corpus at `crates/rig-cassette/fixtures/effects/world/` pins native headers, including every run's `programs[scope]`, independently of the agent corpus.

| what | where | pinned by |
|---|---|---|
| `LogHeader::hooks` | the rig-agent program's declaration — the corpus's `hook_name` list, then `layer_names`; a world has no hook stack to name and leaves it empty | every golden's `/header/hooks`, asserted by the agent producer |
| `LogHeader::programs: BTreeMap<String, ProgramIdentity { required: EffectRow, policy: u64 }>` | written per run scope by `stamp_run` (`policy` = `stable_hash(spec_json(world, run))`), using supported effective run-over-agent settings; rig-agent's builder-only goldens carry none | `run_identity.rs`, `run_replay_policy.rs` |
| `rig_cassette::ecs::identity::check_replayable(world, run, &log)` | selects the run's exact `Scope`; rejects policy/row differences and missing handlers. Missing scoped identity or nonempty `PolicyVersion` declaration is unverified, with no builder-header fallback. Custom systems, ordering and otherwise-unhashed settings are the application's version declaration, not automatically fingerprinted code | `run_identity.rs`, `run_replay_policy.rs` |
| `required` with a route | the agent's `Route` links' keys as `completion` | `anthropic_serving_model_route{,_unselected}` `/header/required` |

`ToolResultLimit` (§8.1) is request shaping, like `RequestPatch`: it is
not an input of `spec_json`, so a limit set, changed or removed between
runs does not change the policy hash, and a recorded run replays under any
limit — the request the log holds is the cut one, as it was sent.

Replayers retain recorded model identity/capabilities and include every
`programs[*].required` row. They clear executable layer metadata until the
application reapplies its middleware: a program recorded under a layer is
replayed by the replayer wrapped in that same layer (`.layered(..)`), and only
then does its spec hash match the record's — a bare replayer under a layered
program, or another layer, is refused
(`a_layered_program_replays_under_the_same_layer_and_refuses_another`). The positive compatibility check runs
in the fresh replay world, followed by execution; model, capability, effective
setting and application-version changes are negative cases
(`run_replay_metadata.rs`). No policy-hash inputs were dropped to accommodate
synthetic replayer labels.

### 10.1 Delivery observation

ECS records outcome visibility and stream item batches separately from
exchange dispatch order in `LogHeader::deliveries`. `Replay::policy_visible`
enforces these batches between schedule passes and requires kept stream
bytes. Default replay supports final-answer/exchange consumption and returns
recorded cancellation errors; policy replay instead requires the same policy
to cancel the effect without inventing an outcome for it. Missing metadata
cannot establish policy fidelity; inconsistent metadata or an unreproduced
cancellation is a diagnostic. A pass is one `RigSchedule` run (one
`app.update()`); cancellation is checked in `RigEnd`, after the pass and its
deferred policy commands, and only when the pass was idle: a pass in which
nothing raised `bus::Wake` (no task landed, no delivery arrived, no system
signalled). A policy that needs another pass before removing the effect
raises `wake.signal()` and is given that pass first; replay does not insert a
cancellation outcome between those passes. A policy that goes idle without
cancelling is refused. Batch identity is not elapsed time.

A future effect missing at Collect is diagnosed the same way: after an idle
pass. Later continuation systems can advance through multiple passes, raising
`Wake`, before minting that effect. Queued intake and held requests can await
another update or host input. A refusal installs `ReplayFailure`; subsequent
recorded effects also fail instead of falling through to ordinary collection
and exposing unpaced successful answers. Cassette's `ReplayPlugin` installs
idle replay diagnosis in the runtime's `RigEnd` (after `RigSchedule` in
`MainScheduleOrder`) while a `ReplayDelivery` plan is installed; a host
driving `RigSchedule` by hand runs `RigEnd` after it. The two host systems in
`tests/bus_delivery.rs` show a policy signalling `Wake` while it deliberates.

Policy observers may use `On<Add, EffectOutcome>` or systems ordered after
all of `BusSet::Collect`, preserving their relevant live/replay ordering.
Intermediate collector state and handler inboxes are not policy replay
surfaces. World handlers submit `WorldOutcome` (or typed `Answer<E>`), which
Collect publishes in submission order; submissions after Collect land in the
next pass. A direct in-flight `EffectOutcome` insertion records a delivery
limitation and makes policy replay refuse that recording. Gate denials and
Judge replacements remain supported. World answer inboxes are transient;
collect them before a checkpoint save to preserve the submitted result.

`run_delivery.rs` covers reversed concurrent arrivals and coincident answers
under different archetypes; `bus_delivery.rs` covers opposite first-visible
winners, single/multiple event batches, cancellations and resumed subsets.
Agent materialisation and batch landing use `RunSeq` for coincident turns.
Program-created IDs still depend on the reproduced causal dispatches; saved
IDs alone do not prove program replay. Arbitrary world state and application
system ordering remain outside automatic verification.

## 11. Memory is the graph

The conversation graph *is* memory; a memory handler is where it persists. `agent::Remembers(entity)` on the agent names the memory handler entity; `agent::Conversation(id)` on the agent the conversation (the run carries a copy of it once spawned). Every op is an effect entity `ChildOf` the run, recorded like any other.

| moment | what happens | pinned by |
|---|---|---|
| a run spawned with no history, on an agent that remembers | before its first turn the run dispatches `Memory { Load { conversation } }` (the run is `LoadingMemory`); when it lands, the loaded messages become utterances *before* the prompt (each marked `Remembered`), and the run is `Assembling` | `anthropic_memory_conversation` `/records/0` (`load`, `golden-conversation`), `/records/1/…/chat_history` (the loaded history, then the prompt) |
| a run spawned with history | no load, no append: the history is the run's; memory stays in the required row | `anthropic_memory_history_bypass` (`[Completion]`, `golden/memory` in `/header/required`) |
| the run settles | `Memory { Append { conversation, messages } }` is dispatched `ChildOf` the run, `messages` every utterance of the run that is not `Remembered`, in order (the prompt, the turns' assistant utterances and tool results, the answer) | `anthropic_memory_conversation` `/records/2` (user, assistant); `anthropic_serving_serial_memory_tools` `/records/4` (user, assistant call, user result, assistant) |
| the load fails | the run is `Failed(Memory(report))` at the record; no completion | `mock_memory_failing_load` (`[Load(err)]`, `MemoryError`); `mock_leftovers_denied_memory_load` (`[]`: a layer's denial, no record) |
| the append fails | the record holds the store's error; the answer stands | `anthropic_memory_failing_append{,_streamed}` |
| a hook clears | `Memory { Clear { conversation } }` spawned by a user system `ChildOf` the run: after the load landed (`On<Add, EffectOutcome>` on the load) — the run has already read the store; after the append was spawned (a system after `RigSet::Settle` on `Added<PendingEffect>` of an append) | `anthropic_memory_clear_at_start` (`[Load, Clear, C, Append]`), `anthropic_memory_clear_at_settled` (`[Load, C, Append, Clear]`) |
| two runs on one agent | the second run's load returns what the first appended; each run appends its own | `anthropic_memory_two_runs{,_streamed}`, `openai_breadth_memory_two_runs`, `anthropic_memory_clear_at_{start,settled}_two_runs` |
| a layer replaces the load | the run's history is the replacement; the record the store's answer | `anthropic_layers_memory_load_replaced` (`/records/1/…/chat_history` has four messages, `/records/0/outcome` none) |
| memory over a host's bus, under serial serving with two tools | the same ops in the same order | `anthropic_memory_host_bus`, `anthropic_memory_serial_two_tools` |

The required row names `<owner>/memory` as `memory` from `Remembers`. `Memory { Load }` and `Append` carry `conversation` verbatim (`ConversationId`).

## 12. Retrieval attaches; routes bind

| what | the walk | pinned by |
|---|---|---|
| `agent::Retrieves(entity)` link entities `ChildOf` the agent with `Retrieval { samples, what: Documents \| Tools }` | `Advance` marks the fresh turn `Retrieving`; `Assemble`'s first pass over it spawns, `ChildOf` the turn and before the fold, one `Retrieve` effect per link in link order — `TopN` for documents, `TopNIds` for tools — with `VectorSearchRequest::builder().query(q).samples(n).build()` (`threshold`, `additional_params`, `filter` null); the turn stays `Fresh` and `Retrieving` until they land (`attach_retrieved`, after `Advance` and before `Select`, reads the results) | `gemini_retrieval_context_and_tools` `/records/0..1` (context then tools, before every completion), `/records/4..5` (again on turn 2) |
| the query | the last utterance with text, from the end (`Message::rag_text`): the prompt on turn 1, still the prompt after a tool turn | every retrieval cell's `/records/*/kind/query/req/query` |
| documents | each result `(score, id, value)` becomes a document entity (`DocumentId(id)`, `DocumentText(serde_json::to_string_pretty(value))` — a string value keeps its quotes) attached to the turn after its static attachments, in result order; an existing document entity with that id is reused | `gemini_retrieval_dynamic_context_over_sampled` `/records/1/…/documents` (three, in score order); `gemini_retrieval_dynamic_context_empty_index` (none) |
| tools | the retrieved ids name tools among the agent's `Grant` links marked `Retrievable` (never advertised otherwise); the turn advertises the retrieved tools first, in result order, then the static grants | `gemini_retrieval_retrieved_tools_with_static` `/records/1/…/tools` (`subtract`, `add`) |
| the required row | `<owner>/retrieve:context#0`, `<owner>/retrieve:tools#0` as `retrieve`; a `Retrievable` grant's key as `tool_call` | `gemini_retrieval_context_and_tools` `/header/required`; `hooks: ["DynamicContext"]` from the program's declaration (§10) |
| a route bound after the agent exists (`late_route`) | `UsesModel` inserted on the run by a system (§9.2); not in the row | `anthropic_shaping_late_route` |
| a route never selected | in the row, never dispatched | `anthropic_serving_model_route_unselected` |

### 12.1 Host assembly, runtime execution

The host constructs an HTTP or SDK `CompletionModel`, wraps it in a
`CompletionAdapter`, and installs the erased handler under a dispatch key.
ECS stores the live handler and its `Bound` descriptor, not provider launch
configuration. It performs no provider discovery, credential resolution,
transport selection or SDK preparation.

For declarative HTTP launching, the host may use rig-core's bounded
`providers::registry::{ProviderRef, ProviderConfig}` constructors. They are
optional host utilities, not ECS components or a universal SDK recipe. The
host retains launch settings separately and reapplies explicit endpoint,
authentication, proxy/redirect/TLS and runtime policy on live reconstruction.
An explicitly selected default-looking endpoint remains explicit.

A dispatch key, a model label and a connection are different facts.
`HandlerDescriptor` advertises execution metadata; equality does not attest
endpoint, credential, tenant or transport policy. Catalog membership likewise
does not prove launch readiness. SDK credential refresh and its runtime are
host/client-lifetime resources, not per-effect tasks.

`ProviderBinding`, `Materializer`, credential-reference reflection and
provider-specific ECS diagnostics are removed. Old checkpoints containing
those unregistered component paths are refused, not silently stripped. The
host must explicitly migrate launch data out of such checkpoints before
resuming them. Execution diagnostics remain on their existing runtime paths.

## 13. Resume is a checkpoint load; two runs

`rig_ecs::checkpoint::Checkpoint` is the world as reflected data:
`entities` (every entity with a registered reflected component — the graph,
the effects, the handlers' `Bound`s, a host's own components once the host
registers the type with `#[reflect(Component)]` in the world's
`AppTypeRegistry` — parents before children, siblings in `Children` order,
each component under its Bevy reflected `TypePath`, an `Entity` in a component
as the index of that entity in the checkpoint), `counters` (`next_run`, `next_id`) and
`binaries` (the binary store, every retained payload once per content hash).
`save_world(&mut World)` takes it;
`load_world(&Checkpoint, &mut World, RestoreMode, handlers)` validates its
complete implementation set, spawns it and returns `Loaded` (the entities by checkpoint index;
`Loaded::with::<C>` filters them). `Checkpoint::{to_json, from_json}` are the
wire form. `checkpoint::register_types` registers every component and wrapper
of the crate; `RigPlugin` calls it. The envelope's `format` is 2; format 1's
separate content payload components are refused without compatibility adapters.

In-flight state (`Serving`, `Streaming`, `Handler`, the cache views) is not
reflected and never saved; relationship targets (`Children`, `Serves`, …) are
rebuilt by their sources' hooks at load, in checkpoint order. An unregistered
host component is simply not saved
(`run_scene_extensions.rs::unregistered_state_is_outside_the_checkpoint_contract`);
a saved type path the loading world has not registered is refused
(`missing_registration_and_invalid_payload_are_refused_before_spawning`).
Resources, system-local and external state are host-owned.

Each supplied `(HandlerKey, ErasedHandler)` pair authorizes the dispatch key
where that implementation is installed. `load_world` normalizes the supplied
descriptor key to that key; `Strict` compares every remaining field against
the original saved contract, including family, model label, capabilities and
layer order. Saved `Bound.key` and descriptor key must agree in both restore
modes. `Replace` permits changed metadata within the same effect family.
Equal descriptors still install the supplied implementation (including rotated
credentials); omitted handlers retain matching installed implementations in
`Strict`. Destination operations already running retain their captured handlers.
Current component lookup uses reflected type identity. Removed provider component
paths remain frozen historical wire identifiers for named migration refusals.

Loading validates the whole checkpoint in a scratch world first — the binary
store merged with the destination's under the destination's limits, every
utterance readable, content parents and part components well formed, request
edits linking a turn to a part, `Ready` and `Prompt` on runs only, batch holds
with their barrier, owner and slot, and an original saved `Bound` contract for
every unfinished effect — so a refusal leaves the destination untouched; `run_content_scene` exercises this with populated destination
worlds and shared payloads (hash mismatches, missing handles, invalid parents).
Part sources reference payloads by SHA-256 identity; the store retains the
asset-count and decoded-byte limits from section 1.

Saving captures data, not proof of resumability: an unfinished streamed effect
without a continuation cursor, or an unfinished effect whose handler was never
registered or was removed before saving, can still be saved for inspection but
cannot resume. A destination binding cannot invent the absent saved contract.
Register required handlers before saving resumable work, or settle unsupported
effects before saving. An offline migration must explicitly author any missing
contract and validate the resulting graph; `Replace` is not a validation bypass.
This deliberately moves missing-contract failure ahead of resumed dispatch,
rather than silently converting an incomplete checkpoint into new execution.
Open world handlers retain their ability to inspect arbitrary input kinds;
their advertised family is not an input validation rule.


A checkpoint preserves consumed effect ids even when their entities have been
removed: `counters.next_id` is taken as the maximum with the destination's
`IdCounter`, and every `Reserved` or `Issued` loaded bumps the counter past
it. `Seq` is re-stamped in saved order after everything already in the
world. `u64::MAX` as the id counter means exhausted: no fresh id is minted
or recorded, and dispatch returns a request error; a reserved id of
`u64::MAX` is refused at dispatch. Existing reserved ids below that sentinel
still resume. This prevents id reuse; it does not make an unanswered
external write safe to repeat (`bus_scene.rs`).

`Streamed.errors` retains every error report and its zero-based position among
all stream items, even after the first terminal outcome and without a recorder
or kept event bytes. This is live consumer evidence; `Streamed.outcome` and
`EffectOutcome` still carry the first folded outcome. Consumers that require
error-free drainage through EOF must inspect errors as well as that outcome.
Policy-visible replay reconstructs the same error observations when the log kept
the error items. A folded-only log does not acquire omitted error evidence.

Completed stream effects retain `Streamed` events, errors, text and terminal outcome
through a checkpoint and load without serving again (`bus_scene.rs`). Load is
fallible: an unfinished stream with observed progress (events or errors, no
`EffectOutcome`) is refused in the scratch world, before the destination is
touched. No restart cursor is recorded. An effect taken but unanswered
(`InFlight`, `Issued`, no outcome) loads as `Reserved(id)` and `Dispatch`
re-issues it under its saved id; this is a cut restriction, not arbitrary
mid-flight forking.

Insertion observers/change detection can fire on load; install application
observers afterward or guard restoration.

`Checkpoint::requirements()` exposes the original saved descriptors;
`Checkpoint::validate(&World)` validates saved data before the host performs
side-effectful construction. Loading accepts `(HandlerKey, ErasedHandler)`
pairs already constructed by the host and an explicit `RestoreMode`:

| mode/input | contract |
|---|---|
| `Strict` | every supplied or retained implementation matches the original saved descriptor before aliasing |
| `Replace` | the host explicitly accepts descriptor changes within the same effect family |
| supplied key | installs that implementation even when its descriptor matches, so credential rotation is real |
| omitted key | retains the destination's already-serving implementation; an unserved or absent key refuses the whole load |
| duplicate, extraneous or mismatched key | refuses before state or handler installation |

Every saved handler requirement must be satisfied before any resumed state
is installed. A failed batch leaves destination state and handlers unchanged.
Aliasing reuses destination entities, but cannot erase the original descriptor
before strict comparison. Existing operations retain their captured handlers;
replacement applies to new dispatches. The host chooses whether to cancel or
drain existing work before replacement.

Replay selects and installs recorded handlers before this boundary, without
credential lookup, secret collection, SDK initialization or live transport
construction. It can load strictly over matching preinstalled replayers with
an empty supplied set. Logical effects, HTTP recordings, execution checkpoints
and host launch settings remain separate products. Pinned by `run_binding.rs`
and the cassette world's resume/replay matrix.

The binding's saved shape changed with the provider vocabulary, and there is
no legacy reader: a checkpoint written before the cutover does not load.
Before, a binding saved as
`{"key":"k","kind":"anthropic","model":"m","label":"l","base_url":"http://h","credential":"c","extra_params":{"anthropic_betas":["b"]}}`;
now it saves as `{"key":"k","provider":"anthropic/anthropic:m","label":"l","credential":"c"}`
for a registered selection, or
`{"key":"k","provider":{"config":{"anthropic":{"api_key":"[redacted]","base_url":"http://h","version":"2023-06-01","betas":["b"],"dialect":"anthropic"}},"model":"m"},"label":"l","credential":"c"}`
when it names a host or an option of its own. This repository ships no
binding fixture, scene or golden, and the cassette harness rebuilds its
binding from the live wire on every run, so nothing in-tree carries the old
shape; a host holding saved worlds rewrites those two fields.

`Ready` and an unread `Prompt` are checkpoint data: a run
saved before it opened loads unopened — `Ready`, `Prompt`, no phase — and
opens on the loaded world's first pass; a run saved mid-run loads with its
`Ready` and its phase and goes on. A run without `Ready` never advances,
loaded or not. Pinned by `tests/run_commands.rs`
(`a_ready_run_saved_before_it_opened_starts_after_the_load`) and by every
resumed world cell.


The live-handler regressions in `tests/memory_resume.rs` cover memory finalization before
scheduling, while queued, after an external write but before its outcome,
and after completion. `MemoryAppendScheduled` and the child effect persist
separately from Bevy change-detection ticks: loading `Settled` is not a new
append transition. An unanswered write is retried under its saved effect id;
external deduplication or reconciliation is the host's responsibility.

| what | how | pinned by |
|---|---|---|
| a run saved after its first tool turn's results (the head), resumed in a fresh world over the log's tail | `checkpoint::save_world` the pass after `land_batch` put the run back in `Assembling` (one schedule pass per `update`); a fresh world binds the tail's replayers (positional per key, as the corpus's resumed engine does), `load_world`s the checkpoint, installs its hooks after the load (no run-start observer fires), ticks to the ending: the head's records are the golden's to the cut, the tail's from it, the answer the golden's | every `corpus_resume.rs` row (18) as `world_resumed`; every `corpus_checkpoint.rs` program (14) as `world_payload`, `world_hash`, `world_full_log_refused` |
| the log's `Checkpoint::state` | the world's `Checkpoint` itself: `rig_cassette::effect_log::Checkpoint<S>` is generic over the driver's state and a world cuts a `Checkpoint<rig_ecs::checkpoint::Checkpoint>` (the classic engine stores its serialized state in `Checkpoint<serde_json::Value>`); it is a cut of the log beside the world, `EffectLog::from_checkpoint(&checkpoint, tail)` the continuation; a full log in the tail's place is refused by its first id | `corpus_checkpoint.rs`, `corpus/world_resume.rs` |
| a resumed run loads nothing and appends | the loaded utterances are `Remembered` in the checkpoint; the append is the resumed run's (the world keeps its state, the agent driver restores its serialized state) | `anthropic_serving_serial_memory_tools` resumed: `[Load, C, Tool, C, Append]` with the append in the tail |
| durable execution | the same property as `durable_execution.rs`: nothing of the first world survives but two JSON strings — the head's log and the checkpoint (the world as its `state`) — and the second world resumes to the same answer and the same tail | every world resume cell round-trips both |
| an effect in flight at the cut | saved as intent, re-issued in the second world under its saved id, answered by the tail's replayer there: a note an outcome hook dispatched just before the cut is the tail's first record | `anthropic_host_custom_at_outcome{,_streamed}` resumed |
| a system that answers an open key (the nesting program's `lookup`) | runs after `Dispatch` and before `Collect`, so `settle` records its answer in the same pass: a checkpoint saved when the run wants its next turn has no open record | `mock_causal_depth_two` checkpointed |
| two runs on one agent | two `spawn_run`s in sequence, the second after the first ended and the world went quiet; the conversation shared through memory (§11) | `anthropic_memory_two_runs`, `openai_breadth_memory_two_runs` |

## Tool-turn checkpoint boundary

`agent::checkpoint::ToolTurnCommit` is durable state on an existing tool-bearing
turn. Its `turn` is the model-turn cursor, not a count of successful batches.
`TurnAssistant` links that turn to its committed assistant utterance;
`TurnResults` links it to the one user utterance containing all batch results in
call order. The result link and commit are published only after the complete
accepted batch lands. An individual outcome or `Materialised` marker is not a
commit. A failed/cancelled batch has no new commit; previously committed turns
remain history. Invalid-call recovery without dispatched tools does not create
a tool-batch commit. A mixed real-tool/output-tool batch commits its results
before its existing settlement; an output-tool-only answer has no batch commit.

`hold_after_tool_turn(world, run, owner, minimum_turn)` arms a named owner's hold
on the run. Model turns start at one. It permits the awaited batch to execute,
then blocks `Advance` once any committed turn reaches that threshold, including
in the pass it lands. Each owner must release its own hold with
`release_tool_turn_hold`; re-arming cannot move an existing hold later. Unknown
owner releases are idempotent. Owner names are a host coordination convention,
not a security boundary. Arming cannot retract a turn already advanced or an
already dispatched request. Unrelated runs continue. Budget checks precede the
hold, so an exhausted run fails normally; cancellation and terminal cleanup do
not require a release. Hosts may release residual holds on terminal runs.

`ToolTurnCommitted { run, turn }` is a live event explicitly triggered after
committed graph writes and phase changes are visible. Independent observers
receive the same event. If an earlier terminal component observer deletes the
owning graph, the subsequent live notification is suppressed; that observer can
inspect durable commit state before deleting it. `RigSet::Checkpoint`, after `Materialise` and before
`Settle`, is the public schedule slot for inspection and checkpoint handling.
The next pass cannot advance a held run. A checkpoint load restores durable commit
state and relationships and retains every owner hold, but does not trigger this
live event. Generic `On<Add, ToolTurnCommit>`/change detection can still fire on
load; applications must use the live event or explicitly account for restoration.

A committed boundary is run-local. The checkpoint's whole-world validation
still rejects unsafe unrelated effects/streams. The checkpoint remaps utterance links
and validates their roles and run ownership before destination mutation. Saving
and restoring host tools, workspace state and external side effects remains the
host's responsibility; no exactly-once external-write guarantee is added. A
restored run stays held until explicit release. Holds do not alter provider
requests, effect identities, usage, retry budgets or the existing run policy hash.

## Live stream delivery

`bus::StreamItemsDelivered { effect, id, start, items }` is a synchronous Bevy
event for newly collected items of a streaming effect. `On<StreamItemsDelivered>`
observers independently receive the same batch, including interleaved
`StreamEvent` and `ErrorReport` values. `start` counts all prior events and
errors for that effect; `id` is its issued effect identity, so retries remain
distinct attempts. Provider block/call identity is carried unchanged in the
items. Run/turn correlation uses existing relationships outside the bus.

The collector updates accumulated `Streamed` before delivery and publishes its
`EffectOutcome` afterward. Live notifications execute during `BusSet::Collect`,
before agent folding, tool-turn commit, or run settlement. A terminal stream
record is an item, not stream closure: the existing first-outcome rule and
accepted post-terminal metadata/errors are preserved. EOF and cancellation are
not fabricated stream items. A unary request folded from a stream does not
expose streamed consumer delivery. Recording and recorder event retention do
not determine live visibility.

Each notification owns only its new batch; there is no second retained history
or subscriber backlog. Observers run on the world thread and must return
promptly. Long-running work must be handed to host-owned bounded processing.
No observer order is promised. If one removes the graph, the payload remains
readable to the others; entity queries must tolerate absence. An observer that
removes a sibling effect cannot suppress the sibling's already accepted batch:
live collection and policy-visible replay both capture every batch accepted in
a pass, as an owned payload, before any observer runs. Collected items are
delivered before normal outcome/terminal cleanup in the same pass. Notifications do not alter collection limits or execute
tools from incomplete arguments.

Late or re-enabled consumers receive future notifications only. Hydrate from
durable `Streamed`/committed history while the host is not advancing the world,
attach or enable observers, then resume updates. A checkpoint load restores durable
state and emits no live delivery notification. Existing refusal of unfinished
streams with observed progress remains unchanged; a held tool-turn checkpoint
resumes only after host prerequisites are restored and its owner releases it.
Notification observers, UI state, and queued host work are not checkpoint data.

Ordinary live/cassette collection preserves item order, not identical network
timing or batch grouping. Policy-visible replay emits the recorded delivery
batches. Both paths update the same durable state and use the same public
notification. `tests/bus_stream_delivery.rs` exercises independent consumers,
errors, late hydration, bounded bursts, terminal deletion, and checkpoint/replay
behavior; provider matrix evidence compares actual items with the independent
producer logs.

## Returned replies and collection cadence

An initial `Serve` task prepares the reply and folds a unary stream to its first
answer on the executor. A streaming reply gets one long-lived worker and a
bounded private delivery queue. Native source polls, parsing, writer work and
streamed verdicts run on pool threads. Browser `!Send` streams run on the local
executor selected by the target-specific `bevy_platform/web` feature; synchronous
source work still occupies the browser thread.

Live Collect performs at most 64 queue checks per effect per pass, with
`STREAM_WORK_PER_TICK` (4,096) shared across the pass. Pending ends that
effect's turn. The next pass resumes after the last served dispatch sequence,
wrapping once, so a hot early effect cannot permanently exclude later ones.
Every pass — one `RigSchedule` run — gets a fresh allowance. Deltas raise
`Wake` (another update is worth running) but are not completion. These are
streaming delivery limits, not total CPU-time or payload-size guarantees.
Hosts must keep updating to expose ready items and settle replies.

The task sits in the non-send `Executions` table under the effect's entity,
beside the `Serving` marker or `Streaming { events, fold, .. }` component, and
is removed when the effect leaves `InFlight`.
Task cancellation reaches pending setup, unary folding and full delivery queues.
An active native poll cannot be interrupted synchronously; observation closes
atomically with cancellation so a late result cannot mutate the closed record.
Registry replacement does not replace in-flight work. Streamed effects retain
their serial slot through EOF, including post-final frames. Both reply forms
converge through Collect, durable publication, `EffectOutcome` and `settle`.

The observer records the innermost request and original answer. Denial discards
the exchange. An observed original answer survives cancellation during an outer
verdict or before collection; it does not claim successful consumer delivery.
Cancelling an external deferred waiter closes its resolver and releases driver
accounting even while the external host retains that resolver.

Install the replay plan before dispatching its effects. For an implicitly
cancelled kept prefix, replay limits source polling as well as visible delivery:
it parks before any synthesized cancellation error and retains the source until
policy cancellation. Re-recording therefore preserves the original error items.

Each live `DeliveryKind::Stream { items }` records the number of items exposed
for that effect in one Collect pass. Replay preserves both single-item
and multi-item boundaries, including error positions. `bus_delivery` compares
live policy observations with replay of the actual recorded cadence.
`ServingPolicy::stream_capacity` supplies shared queue slots in ECS, clamped to
at least one, with one additional sender-reserved slot. Awaiting each send
prevents another source poll while that sender is parked. This transport bound
is independent of the collection allowance and any source-internal buffers.

Cancellation after an original handler answer has been observed records a
`DeliveryKind::Cancelled` boundary instead of an outcome delivery. This preserves
that original answer (including one awaiting a layer verdict) without claiming
that the consumer received it. Policy replay must reproduce the cancellation;
exchange replay returns cancellation and leaves undelivered terminal items hidden.
Ordinary cancellation without an observed answer retains its existing encoding.
