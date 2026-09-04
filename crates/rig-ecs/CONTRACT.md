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

## 9. Steering: every hook is a system

Stage 4 (ruling 4). No hook trait: a user system writes a component at a set boundary and a library system reads it later. The spellings are `how-the-ecs-dissolves-rig-agent.md` §3's table; each row names the cell that pins it (`crates/rig-verify/tests/corpus/world_hooks.rs` writes them for the corpus). The moments, in schedule order: `On<Add, Run>` (run start) · a system after `RigSet::Advance` and before `RigSet::Select` (model selection) · before `RigSet::Assemble` (the completion call: `RequestPatch`, a hook's own dispatch) · `RigSet::Patch` (the folded effect) · the bus's `Gate` (a dispatch: deny, patch, hold) · the bus's `Judge` (an outcome: replace) · after `RigSet::Fold` (deltas) · `RigSet::Judge` (the model turn: retry, replace, stop) · before `RigSet::Materialise` (an invalid call) · `On<Add, Settled>` / `On<Add, Failed>` (run settled).

### 9.1 Stopping: `Cancelled(reason)` on the run

| the stop | the write | what the library does | pinned by |
|---|---|---|---|
| any hook's `stop(reason)` | `agent::Cancelled(reason)` inserted on the run, at any moment | the observer `run_cancelled` fails the run `Failed(Cancelled(report))` with `report.message == reason` and `kind == Cancelled`, removes its phase marker, marks its current turn read, and despawns every effect of the run that was never issued (never dispatched: no record); an effect in flight is left to its handler, so the record is the handler's — a stream that ended stays a completion, one still streaming ends as the replayer or provider ends it | every `Ending::Cancelled` cell; the reason asserted |
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

`agent::RequestPatch` (the corpus's `rig_agent::agent::RequestPatch` as data: `preamble`, `temperature`, `max_tokens`, `tool_choice`, `active_tools`, `additional_params`, `extra_context`, `history`) inserted on the fresh turn before `Assemble` (a system on `Added<Fresh>`, reading `Cursor.turn` for the turn number); `assemble` folds it in as `prepare_request` did. Several hooks patching one turn merge in registration order (`RequestPatch::merge`: `extra_context` appends, object `additional_params` shallow-merge with later keys winning, `active_tools` intersect, scalars and `history` last-writer-wins); a user system that finds a patch on the turn merges over it.

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

| the hook | the write | what `materialise` does | pinned by |
|---|---|---|---|
| `on_model_turn_finished` → `retry_with_feedback(text)` | `agent::Retry { feedback: Some(text) }` on the turn, in `RigSet::Judge` | the turn (unless empty) and a user utterance of the feedback become history; another turn; nothing is committed as an answer; text turns only (a tool-bearing turn is refused, `Failed(Unsupported)`) | `anthropic_hooks_demand_done` `/records/1/…/chat_history/2..3` |
| `repeat` | `Retry { feedback: None }` | nothing becomes history; another turn | `rig_agent::run::AgentRun::retry_model_turn` docs |
| `on_outcome` → replace a completion | a system in `RigSet::Judge` rewriting the turn's `Outputs.content` (or the bus's `Judge` rewriting the `EffectOutcome`) | what is read is the replacement; the record holds the model's | `anthropic_hooks_replace_answer` (the run's output `REPLACED`, the record's text) |

### 9.5 A hook's own dispatch

A `PendingEffect` the system spawns `ChildOf` the run at the hook's moment (its record names no parent — the run is not an effect — and carries the run's `Scope`):

| moment | the write | pinned by |
|---|---|---|
| run start | `On<Add, Run>` observer: before the first completion is folded, so its `Seq` is lower | `anthropic_host_custom_at_start{,_streamed}` (`[Custom, Completion]`), `anthropic_host_custom_twice_{serial,concurrent}` (`[X, X, C]`), `anthropic_hooks_lookup_before_run` (`[Tool, C, Tool, C]`: a tool call `add(1, 2)` on the tool's key), `openai_host_embed_prompt{,_streamed}` / `gemini_breadth_embed_prompt` (`Embed { inputs: Texts([prompt]) }`), `mock_oracle_rerank` (`Rerank { query: prompt, documents }`), `mock_leftovers_five_thousand_events` (two hundred notes) |
| before a completion | a system on `Added<Fresh>` before `Assemble` (the note's `Seq` precedes the completion's) | `anthropic_host_custom_at_completion_call` (`[X, C]`) |
| after a tool answered | `On<Add, EffectOutcome>` on a tool child | `anthropic_host_custom_at_outcome{,_streamed}`, `{gemini,openai}_breadth_custom_at_outcome` (`[C, T, X, C]`), `anthropic_oracle_concurrent_notes` (`[C, T, T, X, X, C]`) |
| settled | `On<Add, Settled>` observer; the world ticks to quiescence after the run ends | `anthropic_host_custom_at_settled` (`[C, X]`), `anthropic_host_custom_start_and_settled` |
| a key nothing serves | the system finds no `Bound` for the key and dispatches nothing | `anthropic_host_custom_unserved` (`[C]`) |
| an effect with no wire form | `PendingEffect::custom` refuses it; nothing is spawned | `mock_leftovers_unserializable_from_hook` (`[C]`) |

### 9.6 Layers

A layer is the handler's: the world registers the layered `ErasedHandler` (`handler.layered(intercept)`) exactly as `Replay::open` does, under the key the header names. A decision the layer makes before the handler is served leaves no record, and a verdict after leaves the handler's answer in the record: the world's `Dispatch` installs a sink observer (`bus::WorldObserver`, its slots on the entity as `bus::Observed`) for a handler whose descriptor names layers — the one way a layer's `discard` and `patch` reach any recorder, and the innermost handler's outcome the observer is told is what `settle` records (`bus_world::a_layers_decisions_reach_the_record_through_the_sinks_observer`). This is the substrate gap the PR reports beyond ruling 1's two: without the observer the world recorded a layer's denial and verdict. The suspending layer (`ApprovalLayer`) is answered by a thread the interpreter spawns as the program says.

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

| what | where | pinned by |
|---|---|---|
| `LogHeader::hooks` | the program's declaration — the corpus's `hook_name` list, then `layer_names` — passed to `replay::stamp_header(world, agent, recorder, bus, hooks)`; the world has no hook stack to name | every golden's `/header/hooks`, asserted by the interpreter |
| `LogHeader::programs: BTreeMap<Arc<str>, ProgramIdentity { required: EffectRow, policy: u64 }>` | written per run scope by `stamp_header` (`policy` = `stable_hash(spec_json)`), `#[serde(default, skip_serializing_if = "BTreeMap::is_empty")]`: rig-agent's goldens carry none, no re-stamp; a world's own log carries its scopes | `rig-effect-log`'s round-trip test |
| `replay::check_replayable(world, agent, &log)` | refuses a foreign golden: the scope's `required` not served by the log's handlers, or its `policy` ≠ the agent's `spec_hash`; falls back to `run_spec`/`required` for a golden without `programs` | `run_identity.rs` |
| `required` with a route | the agent's `Route` links' keys as `completion` | `anthropic_serving_model_route{,_unselected}` `/header/required` |

## 11. Memory is the graph

Stage 5 (ruling 5). The conversation graph *is* memory; a memory handler is where it persists. `agent::Remembers(entity)` on the agent names the memory handler entity; `agent::Conversation(id)` the conversation (the run's, else the agent's). Every op is an effect entity `ChildOf` the run, recorded like any other.

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
| `agent::Retrieves(entity)` link entities `ChildOf` the agent with `Retrieval { samples, what: Documents \| Tools }` | `Advance` spawns, `ChildOf` the fresh turn and before the fold, one `Retrieve` effect per link in link order — `TopN` for documents, `TopNIds` for tools — with `VectorSearchRequest::builder().query(q).samples(n).build()` (`threshold`, `additional_params`, `filter` null); the turn stays `Fresh` (`Retrieving`) until they land | `gemini_retrieval_context_and_tools` `/records/0..1` (context then tools, before every completion), `/records/4..5` (again on turn 2) |
| the query | the last utterance with text, from the end (`Message::rag_text`): the prompt on turn 1, still the prompt after a tool turn | every retrieval cell's `/records/*/kind/query/req/query` |
| documents | each result `(score, id, value)` becomes a document entity (`DocumentId(id)`, `DocumentText(serde_json::to_string_pretty(value))` — a string value keeps its quotes) attached to the turn after its static attachments, in result order; an existing document entity with that id is reused | `gemini_retrieval_dynamic_context_over_sampled` `/records/1/…/documents` (three, in score order); `gemini_retrieval_dynamic_context_empty_index` (none) |
| tools | the retrieved ids name tools among the agent's `Grant` links marked `Retrievable` (never advertised otherwise); the turn advertises the retrieved tools first, in result order, then the static grants | `gemini_retrieval_retrieved_tools_with_static` `/records/1/…/tools` (`subtract`, `add`) |
| the required row | `<owner>/retrieve:context#0`, `<owner>/retrieve:tools#0` as `retrieve`; a `Retrievable` grant's key as `tool_call` | `gemini_retrieval_context_and_tools` `/header/required`; `hooks: ["DynamicContext"]` from the program's declaration (§10) |
| a route bound after the agent exists (`late_route`) | `UsesModel` inserted on the run by a system (§9.2); not in the row | `anthropic_shaping_late_route` |
| a route never selected | in the row, never dispatched | `anthropic_serving_model_route_unselected` |

## 13. Resume is a scene load; two runs

| what | how | pinned by |
|---|---|---|
| a run saved after its first tool turn's results (the head), resumed in a fresh world over the log's tail | `agent::scene::save_world` after `land_batch` put the run back in `Assembling`; a fresh world binds the tail's replayers (positional per key, as the corpus's resumed engine does) and the model handler, `load_world`s the pair, ticks to the ending: the tail's records are the golden's from the cut, the answer the golden's | every `corpus_resume.rs` row (18) and `corpus_checkpoint.rs` program (14), as `world_resume` cells |
| `Checkpoint::state` | the `WorldScene` as JSON: a checkpoint is a cut of the log beside the scene, `EffectLog::from_checkpoint(&checkpoint, tail)` the continuation; a full log in the tail's place is refused by its first id | `corpus_checkpoint.rs` |
| a resumed run loads nothing and appends | the loaded utterances are `Remembered` in the scene; the append is the resumed run's (the world keeps its state, the frozen engine's driver had to) | `anthropic_serving_serial_memory_tools` resumed: `[Load, C, Tool, C, Append]` with the append in the tail |
| durable execution | the same property as `durable_execution.rs`: a run interrupted after tool call *n*, its scene and log so far, resumes in a fresh process image to the same answer and the same tail | `world_durable` |
| two runs on one agent | two `spawn_run`s in sequence, the second after the first ended and the world went quiet; the conversation shared through memory (§11) | `anthropic_memory_two_runs`, `openai_breadth_memory_two_runs` |
