# CONTRACT — the completion-only run, as the goldens pin it

The bytes rig-ecs's agent modules are held to, extracted from the corpus (`crates/rig-verify/fixtures/*.effects.json`) and from `rig_agent::run`'s docs read as a specification. Every row cites the golden(s) that pin it by fixture name and JSON pointer; every `policy::text` constant has a test in `src/policy/tests.rs` that compares it to its golden. Nothing here is read off a function body of the frozen crate.

The set this stage covers: every golden whose records are all `completion` and whose header `hooks` is empty — 43 at 207 goldens (the superseded prompt counted 32 at an older head; the set is derived, never listed) (`crates/rig-verify/tests/corpus/world.rs::unsupported` derives the set; `anthropic_memory_history_bypass` is among the 43 but waits for stage 5's memory, since its required row names `golden/memory`).

## 1. The request: a walk over the graph

`CompletionRequest` has nine fields on the wire (`record_telemetry_content` never appears in a recorded request). Each is derived by `policy::fold_request` from one source in the graph, in one order. Source entities and components are `rig_ecs::agent`'s.

| field | source (the walk) | order | pinned by |
|---|---|---|---|
| `model` | none: always `null`; the model is the handler key the effect is dispatched to (`UsesModel` on the run, else the agent, → the handler entity's `Bound.key`) | — | every golden `/records/0/kind/request/model` |
| `chat_history[0]` | the effective preamble as `system`, when there is one: the agent's `Preamble` (the run's override first) joined with the output mode's augmentation by `"\n\n"` (§3); no preamble and no augmentation is no system message | first | `anthropic_completion_smoke` `/…/chat_history/0`; `anthropic_request_shape_without_preamble` (no system message); `anthropic_request_shape_append_preamble` (the preamble's own `"\n"` join is the program's, stored already joined) |
| `chat_history[1..]` | every `Utterance` `ChildOf` the run, in `Order`: the prior history the run was spawned with, then the prompt, then — turn by turn — the assistant utterance `Materialise` spawned and the reprompt utterance it added | `Order` ascending | `anthropic_request_shape_prior_history` `/…/chat_history` (system, user, assistant(id null), user); `mock_output_tool_text_reprompt` `/records/1/…/chat_history` (…, assistant text, user reprompt); `mock_output_tool_missing_field_reprompt` `/records/1/…/chat_history` (…, assistant call, user tool result) |
| `documents` | the turn's `Attachment` links, in `Order`, each to a document entity (`DocumentId`, `DocumentText`, `DocumentProps`) — the agent's `Context` links are attached to every turn by `Advance` | link `Order` | `anthropic_request_shape_static_context` `/…/documents` (`static_doc_0`, `static_doc_1`; no `additional_props` key when empty) |
| `tools` | the turn's `Advert` links, in `Order`, each to a tool handler entity whose `Bound.descriptor.family` is `Tool { name, description, parameters }` — the agent's `Grant` links, advertised by `Advance`; then the output tool (§3) when the resolved mode is `Tool` | grant `Order`, output tool last | `anthropic_request_shape_tool_choice_none` `/…/tools/0` (`add`, its description and parameters verbatim from the descriptor); `anthropic_output_tool_unary` `/…/tools/0` (`final_result`) |
| `temperature` | `Temperature` (run, else agent) | — | `anthropic_completion_smoke` (`null`), `anthropic_output_tool_unary` (`0.0`) |
| `max_tokens` | `MaxTokens` | — | `anthropic_request_shape_max_tokens` (`32`) |
| `tool_choice` | `ToolChoiceSpec`, unchanged (`"none"`, `"required"`, `{"specific":{"function_names":[…]}}`) | — | `anthropic_request_shape_tool_choice_none`; `anthropic_output_tool_choice_required`; `anthropic_output_tool_choice_specific_output` |
| `additional_params` | `AdditionalParams`, verbatim | — | `anthropic_request_shape_thinking_unary` (`{"thinking":{"type":"enabled","budget_tokens":1024}}`) |
| `output_schema` | `Output.schema` when the resolved mode is `Native`; `null` otherwise | — | `anthropic_request_shape_output_schema_unary` (the schema verbatim, unsorted); `anthropic_output_prompted_unary` (`null`); `anthropic_output_tool_under_none_degrades` (the schema: `Tool` degraded to `Native`) |
| `stream` (the effect's, beside the request) | the run's `Streamed` | — | every `*_streamed` golden |

The fold is the only constructor of a `CompletionRequest` in the crate (`tests/core/rig_ecs_bus_module.rs::the_agent_modules_hold_the_discipline`).

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

| the turn | what happens | pinned by |
|---|---|---|
| a provider error | the run fails `Provider(report)` at the record | `anthropic_outcome_model_error` (`provider_response`, 401) |
| the effect despawned mid-stream | the record is `Cancelled`; the run is `Failed(Cancelled)` | `anthropic_cancelled_stream`, `anthropic_outcome_cancel_after_tool_call_delta` |
| an empty turn (no parts, or one unannotated empty text) | not history; the run settles with `""` | `anthropic_request_shape_tool_choice_none` (`choice: []`, the run settles) |
| text, mode `Native`/`Prompted` | the assistant utterance is history; the run settles with the text (prompted output is never validated) | `anthropic_completion_smoke`; `mock_oracle_prompted_unvalidated` (`"not an object"`) |
| the output tool called with every required field | the assistant utterance is history; the run settles with the call's arguments serialised | `anthropic_output_tool_unary`; `golden_answer` in the corpus |
| the output tool called without a required field, reprompts left (`OutputRetries < 1`) and turns left | history gains the assistant call and a user tool result (the missing-field reprompt); another turn | `mock_output_tool_missing_field_reprompt` |
| text where the output tool was due, reprompts left | history gains the assistant text and the user reprompt; another turn | `mock_output_tool_text_reprompt` |
| the output tool called with a missing field, or text where it was due, with **no reprompt left** | the run settles with what it has (the partial arguments, or the text) — unpinned: no golden exhausts the budget; rig-agent's docs accept a text that already satisfies the schema and are silent on the rest | (none) |
| a granted tool named like the minted output tool | `Failed(OutputToolCollision)` | `rig_agent::run::prepare` docs |
| a call to a tool neither granted nor the output tool | an `InvalidCall` entity `ChildOf` the turn; `resolve_invalid_defaults` writes the run's `InvalidCalls.unhandled` unless a system wrote a `Resolution` first; `Fail` → `Failed(UnknownToolCall)`; `Ignore` → the call is dropped, what is left is the answer (an empty rest settles as `""`) | `mock_outcome_invalid_call_unhandled`, `mock_invalid_mixed_fail` (`Fail`); `mock_invalid_ignore_unary`, `mock_delta_ignore` (`Ignore`) |
| a call to a granted tool | `Failed(Unsupported)`: tool dispatch is stage 3's | (no golden in this set) |
| `Retry`, `Repair`, `Skip` written as a resolution | `Failed(Unsupported)`, named | (stage 4) |

## 5. Budgets and endings

| rule | pinned by |
|---|---|
| `MaxTurns` counts model calls; `Advance` fails a run whose `Cursor.turn` reached it (`Failed(MaxTurns { limit })`); the default is 1 | `rig_agent::run::spec` docs (`effective_max_turns`) |
| the output-tool reprompt budget is 1 | `rig_agent::run::spec` docs (`DEFAULT_OUTPUT_RETRIES`) |
| every record of a `MaxTurns` run is a success; the run is not | corpus `Ending::MaxTurns` docs |

## 6. The header

`replay::spec_json(agent)` is the JSON the header's `run_spec` hashes (`rig_effect_log::stable_hash`, keys canonicalised): the agent's identity, not a run's.

```json
{"preamble": <Preamble>, "static_context": [{"id","text",…props}], "additional_params": <AdditionalParams>,
 "max_tokens": <MaxTokens>, "temperature": <Temperature>, "tool_choice": <ToolChoiceSpec>,
 "max_turns": <DefaultMaxTurns or 1>, "max_invalid_tool_call_retries": 0, "output_schema": <Output.schema>,
 "output_mode": "Auto"|"Native"|"Tool"|"Prompted", "output_tool_name": null, "output_tool_description": null,
 "augment_output_preamble": true, "unhandled_invalid_tool_call": "fail"}
```

Pinned by every golden's `/header/run_spec` (`anthropic_completion_smoke` = `171082663332529849`); the world interpreter asserts equality for all 42 it replays. A run's `max_turns`, its history, its stream flag and its invalid-call policy are **not** identity: `mock_delta_fail` and `mock_delta_ignore` share one hash.

`required`: the model's key as `completion`, every granted tool's key by its family (`anthropic_request_shape_tool_choice_none` `/header/required`). `hooks`: `[]`. `signature`: written by the recorder from what was dispatched.

## 7. Keys

`<owner>/model:<label>` (`golden/model:default`), `<owner>/tool:<name>#<n>` (`golden/tool:add#0`), `<owner>/memory`. `replay::{model_key, tool_key}`; `HandlerKey::parts()` reads them.

## 8. Tools and batches

Stage 3 (`rig-ecs-pr3-whole-corpus-prompt.md` ruling 3). A call to a granted tool is an effect entity `ChildOf` the turn; the turn's tool children are the batch; the results are one user utterance. Every row cites the golden and JSON pointer that pin it; the batch semantics are `rig_agent::agent::engine::drive_tool_calls`'s docs and `rig_agent::run::AgentRun::tool_results`'s, read as specification.

### 8.1 The batch

| the turn | what happens | pinned by |
|---|---|---|
| one call to a granted tool | `Materialise` spawns one `PendingEffect { key: the advert's handler key, kind: ToolCall { name, args } }` `ChildOf` the turn, `args` the call's arguments rendered by `serde_json::Value::to_string`; the effect carries `bus::ToolInputs(context)` — the run's `ToolContextSpec` (else the agent's, else empty) as `for_dispatch()`; the run is `ResolvingTools` | `anthropic_tool_call_turn` `/records/1` (key `golden/tool:add#0`, `args` `{"x":17,"y":25}`) |
| several calls in one turn | one child per call, in call order; the record order is the call order under every serving policy and concurrency | `anthropic_concurrent_tools_serial` `/records/1..2`; `anthropic_serving_{serial_concurrency_one,concurrent_concurrency_one,concurrent_concurrency_two,capacity_one}` (four policies, one trace) |
| `ToolPolicy { concurrency }` (the run's, else the agent's, else 1) | `release_batch`, before the bus's `Gate`, holds (`Held`) every tool child beyond the concurrency and releases them in call order as earlier ones land; 1 is serial | `rig_agent::AgentRunner::tool_concurrency` docs; `anthropic_serving_concurrent_concurrency_two` (the trace does not change) |
| every child has an outcome | one user utterance with one `ToolResult` part per call, in call order: `tool_result_output(call.id, call.provider, name, result.output())` — the output's content items verbatim; then the run is `Assembling` | `anthropic_tool_call_turn` `/records/2/kind/request/chat_history/3` (`content: [{"type":"json","value":42}]`); `anthropic_concurrent_tools_serial` `/records/3/…/chat_history/3` (two parts) |
| a tool answered `status: error` | the part is the error's model output; the run goes on | `anthropic_outcome_tool_error` `/records/2/…/chat_history/3` (`the adder is broken`) |
| a `Denied` outcome (a layer's `deny`, a `Gate` system's `EffectOutcome(Err(Denied))`) | a skipped result: one text part, the report's message; no record for a `Gate` denial | `mock_leftovers_denied_tool` `/records/1/…/chat_history/3` (`denied by the host`); `mock_layers_suspend_deny` |
| a `Cancelled` outcome (a child despawned, a `Judge` rewrite to `Err(Cancelled)`) | the run is `Failed(Cancelled)` once the batch has landed; nothing is committed to history | `anthropic_endings_tool_outcome_cancelled` (`[Completion, Tool]`, the tool's real result in the record) |
| `BusClosed`, `HandlerUnavailable` or `Divergence` | the run is `Failed(Tool(report))`: a replay that went on after an answer the record never gave would be a passed test with another trace | `rig_agent::agent::engine::dispatch_tool_call` docs |
| any other `Err` report | a failed result: `ToolExecutionError::other(message).with_model_feedback(message)`, so the part is the message | `mock_layers_wrong_family_patch` `/records/1/…/chat_history/3` (the layer's `Internal` message) |
| an `Ok` of another family | a failed result: `the tool handler answered with a {family} outcome` | `rig_agent::agent::engine::dispatch_tool_call` docs |
| the batch lands at the turn budget | the batch still runs; `Advance` fails the run `MaxTurns` when it wants the next turn | `anthropic_outcome_max_turns_exhausted` (`[Completion, Tool]`, then `MaxTurns { limit: 1 }`) |
| the output tool's call in a turn beside a granted tool's call | unpinned: no golden; the batch runs and the output tool's call is read when it lands | (none) |
| a replaced result | history holds the replacement, the record the handler's answer (a `Judge` rewrite of the child's `EffectOutcome`) | `anthropic_hooks_replace_tool_result` `/records/1/outcome` (`42`) vs `/records/2/…/chat_history/3` (`99`) |
| a patched call | the record holds the patched arguments, history the model's | `anthropic_hooks_patch_tool_args` `/records/1/kind/args` (`{"x":40,"y":2}`) vs `/records/2/…/chat_history/2` (`{17, 25}`) |

### 8.2 Invalid calls beside the batch

`Resolution` gains its payloads here: `Retry { feedback }`, `Repair { to }`, `Skip { reason }`. Written by a user system before `Materialise` (stage 4's systems), or by `resolve_invalid_defaults` from `InvalidCalls.unhandled` (`Fail` / `Ignore`).

| resolution | what happens | pinned by |
|---|---|---|
| `Retry { feedback }`, retries left (`InvalidRetries < InvalidCalls.retries`) | history gains the assistant utterance (the turn as the model gave it) and a user utterance of tool results: for the invalid call the feedback, for every other call of the turn `Tool not executed because another tool call in the same assistant turn was invalid.`; nothing is dispatched; another turn | `mock_invalid_tool_call_recovery` `/records/1/…/chat_history/2..3`; `mock_hooks_retry_twice` (two retries) |
| `Retry`, no retries left | `Failed(UnknownToolCall)` | `rig_agent::run::AgentRun::resolve_invalid_tool_call` docs ("while budget remains") |
| `Repair { to }` | the call's name becomes `to`; `to` must be a granted tool (else the call stays invalid and is judged again); the assistant utterance carries the repaired name; the call is dispatched as a granted call with the model's arguments | `mock_invalid_repair_to_add` `/records/1` (key `add`, `args` `{"x":2,"y":3}`), `/records/2/…/chat_history/2` (`"name":"add"`) |
| `Skip { reason }` | the invalid call's result is the reason (one text part); no call of the turn is dispatched, each other call's result is the invalid-peer text above; refused under `tool_choice: none` (`Failed(UnknownToolCall)`) | `mock_invalid_skip_under_auto` `/records/1/…/chat_history/3` (`no such tool; skipped`); `mock_invalid_skip_under_none` (`UnknownToolCall`, one record) |
| `Ignore` beside a valid call | the invalid call is dropped from the turn; the valid call is dispatched; the assistant utterance carries the valid call only | `mock_invalid_mixed_ignore` `/records/1` (`add` dispatched), `/records/2/…/chat_history/2` |
| `Fail` beside a valid call | `Failed(UnknownToolCall)` at the completion record; nothing dispatched | `mock_invalid_mixed_fail` (one record) |
| a retry under `tool_choice: required` | the retried turn calls the tool; the budget ends the run `MaxTurns` | `mock_invalid_retry_under_required` (`[Completion, Completion, Tool]`, `MaxTurns { limit: 2 }`) |

### 8.3 Nested dispatch (matrix Q)

A tool served by a system (a world-served handler, §8.4) answers by inserting the `EffectOutcome` on its effect entity; what it dispatches on the way is a `PendingEffect` it spawns `ChildOf` the effect it serves, so the child's record names the tool's record as `parent` and inherits the run's `Scope`.

| cell | the world | pinned by |
|---|---|---|
| a completion nested from the tool | the tool's system spawns `Completion { request, stream: false }` on the model's key `ChildOf` the tool effect and answers when the child lands | `anthropic_causal_completion_serial` `/records/2/parent` = 2 |
| a note, a relay (depth two) | likewise, the relay's system spawning the note `ChildOf` the relay | `mock_causal_depth_two` `/records/2..3/parent` (2, 3) |
| the same key under serial serving | the child is refused before dispatch (`Request`, no record); the tool answers `refused:Request` | `mock_causal_same_key_serial_refused` (three records), `mock_causal_same_key_from_thread_refused` (the thread is rig-bus's axis; in the world the refusal is the same query) |
| the same key under concurrent serving | served: `T, T←parent` | `mock_causal_same_key_concurrent_served` `/records/2/parent` = 2 |
| the parent cancelled with the child in flight or queued | the run despawned once the never-answering handler was reached: the tool effect and its child are `Cancelled` (the queued second child never began, no record) | `mock_causal_parent_cancelled_child_in_flight`, `mock_causal_parent_cancelled_child_queued` (`[Completion, Tool✗, Custom✗←2]`) |
| the detached resolver | a system answers later; the same records | `mock_causal_detached_resolver` |

### 8.4 The tool context off the wire (format 5)

`EffectKind::ToolCall { name, args }` and `Outcome::ToolResult { result }` carry no context. The inbound values reach the adapter through the sink's scope (`OutcomeSink::scope::<ToolContext>()`), attached by the driver — the world from the effect entity's `bus::ToolInputs`, rig-bus from the dispatch's own context — and the values a tool published come back beside the sink (`rig_core::tool::PublishedContext`), which the world reads into `bus::ToolOutputs` when the outcome lands. Every golden's context was empty (`{}` on 0 of 207 records' inbound or outbound maps); the re-stamp moves only `/records/*/kind/context` and `/records/*/outcome/Ok/context` and `/header/format` 4 → 5 (the PR pastes the count).
