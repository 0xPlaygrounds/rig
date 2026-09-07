# OpenAI Chat tool-truncation matrix

Scope: 24 original registrations in
`tests/providers/openai/cassette/chat_tool_truncation_matrix.rs`, one fixture
each under `tests/cassettes/openai/chat_tool_truncation_matrix/`. Twelve agent
cells have native counterparts in `ecs_chat_tool_truncation.rs`; twelve model
cells remain shared provider coverage and count as zero migrated agent cases.
No original effect golden exists for this family.

## Classification rules

- Rule T-agent (`Surface::Agent`, 12 cells): `run_cell` selects `run_agent`,
  which builds `client.completions_api().agent(model)` with the preamble, the
  counting `FileReport` tool, `tool_choice: required`, the cell's `max_tokens`
  and `default_max_turns(1)`, then drives `agent.prompt(PROMPT)` (blocking) or
  `agent.stream_chat(PROMPT, []).max_turns(1).stream()` drained by
  `collect_stream_observation` (streaming). Classification `agent`.
- Rule T-model (`Surface::Model`, 12 cells): `run_cell` selects `run_model`,
  which calls `completions_api().completion_model(model).raw_completion(req)`
  then `normalize("openai")`, or `.stream(req)` drained to EOF collecting
  `BlockEnd` tool calls and the `Final` finish reason. No agent builder or
  runner is constructed. Classification `shared_provider`.

Every cell then runs the unchanged `execute` → `assert_cell` after the
cassette closure: request model, `max_tokens` cap, absence of
`max_completion_tokens`, required tool choice, one tool schema, stream flag;
exactly one wire call; `tool_calls` finish with exact complete arguments for
the 48-token control, or `length` finish with non-empty unparseable arguments
for the 16/32-token cells; then the surface-specific observation checks.

## Configuration

Both transports, gpt-4o-mini and gpt-4.1-mini, budgets 16 (low), 32 (mid) and
48 (complete). Chat Completions route, original preamble and incident prompt,
unset temperature, `tool_choice: required` through `additional_params`, one
`file_report` tool. Agent `default_max_turns` 1 is explicit on the agent;
streaming also sets run `max_turns` 1 with empty history; blocking has no run
override. Bus/task-pool defaults are unchanged.

## Native producer

The native producer owns App/plugins and uses the real Chat Completions
adapter and `ToolAdapter`. It registers the original `FileReport` tool type
itself (sibling visibility only; the original file changes no behavior), so
the advertised schema and the invocation counter are the original's. The
shared `ecs_observation` observers and the `ecs_termination` probe are
installed as read-only systems; no original `run_agent`, `AgentBuilder` or
agent runner executes.

The original shared `execute`/`assert_cell` runs unchanged on the native
`Observation`: the wire checks above and, for agent cells, exact invocation
counts (0 for partial, 1 for complete) and absence of `ProviderResponseError`
in the collected errors.

## Endings

Complete cells: the required call is dispatched and invoked once, then the
one-turn budget ends the run. The original tolerates that ending and only
counts the invocation; native `wait_for_outcome` must return the actual
`Failure::MaxTurns { limit: 1 }`, the termination probe must report
`FinishReason::ToolCalls`, and exactly one native `ToolCallSlot` named
`file_report` with `summary` equal to the prompt must exist. The actual native
diagnostic populates `Observation.errors`; no legacy error text is invented.

Partial cells: the provider truncates the only call and decoding drops it, so
the turn carries no call and the run settles with a successful outcome. The
original only requires zero invocations and no `ProviderResponseError`; native
settlement is the stronger check and additionally requires the probe to report
`FinishReason::Length`, zero `ToolCallSlot`s, an empty observed tool-call list
and no `Streamed.errors`. The native run result text is not asserted; the
original does not observe it either.

## Limits

Helper anchors retain whole shared functions; `assert_cell`'s common branch
plus the selected surface branch is the applicable obligation. Only the
selected transport branch of `recorded_finish_and_arguments` is in each call
chain. The original agent finish reason and argument checks inspect the wire
fixture; the runtime assertion is the invocation count and provider-error
guard, and the native slot/probe checks are labeled separately. Caps below 16
are not cassette cells; synthetic empty/no-call `length` shapes live in the
shared unit suite and are outside this family.
