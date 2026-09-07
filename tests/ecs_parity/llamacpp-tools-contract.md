# llama.cpp agent tools family

Agent cells that register real tools and let rig-agent orchestrate one or more
tool round trips. Native counterparts belong in
`tests/providers/llamacpp/cassette/ecs_tools.rs`.

## Rules

- **agent-chat-tools** — `client.agent(..).tool(..).default_max_turns(n).build()
  .chat(prompt, &mut history).await.expect(..)`; the answer is checked with
  `assert_mentions_expected_number`.
- **agent-prompt-tools** — `client.agent(..).tool(..).build().prompt(..)
  .max_turns(n).await.expect(..)`; real tools (`Adder`/`Subtract`, `Population`,
  `Vault`, `CacheProbeLookupTool`) execute and their counters or results are
  asserted; `prompt_caching::agent_loop_does_not_move_its_own_prefix` also
  asserts `response.completion_calls().len() >= 2` and, after closure,
  `assert_prefix_stable` over the recorded requests.
- **agent-stream-tools** — `.stream_prompt(..).max_turns(n).stream().await`
  drained by `collect_stream_final_response` or `collect_stream_observation`
  and checked with `assert_mentions_expected_number`,
  `assert_two_tool_roundtrip_contract` or `assert_tool_call_precedes_later_text`.
- **agent-typed-tools** — `.tool(WeatherTool).build().prompt_typed::<WeatherResponse>(..)
  .max_turns(4)` (the verbatim variant adds `.add_hook(StepLogger)`), with
  `anyhow::ensure!(call_count >= 1)` and `assert_weather_tool_roundtrip_response`.

## Exceptions

The three `streaming_tools::raw_*` cells and nine `tool_matrix` cells call
`model.stream` / `model.completion` directly and are provider-only
(`llamacpp-provider-contract.md`); `tool_matrix::the_smoke_tier_round_trip_is_covered_elsewhere`
is a fixture guard. `tool_matrix` agent cells and no other cell here run on the
competent tier.

## Phase 2 obligations

Tool execution counts come from the real tool types through native
`ToolAdapter`s, never from the final text alone; `completion_calls().len()`
maps to counting this run's successful Completion effect outcomes; `chat`
history obligations map to actual ordered run-child utterances; per-run
`max_turns` overrides map to `prompt_with_max_turns`; the `?`-propagated
`with_llamacpp_cassette_result` bodies keep their `ensure!` obligations. The
answer-text assertions stay exactly the shared helpers with their accepted
spellings.
## Fixture kind

Every fixture in this family is a **local-model recording**: it was recorded
against a local `llama-server` b10499 (commit 6d05498) built from source, with
generation pinned (`--seed 42 --temp 0`), and is replayed through the
provider's own wrapper (`with_llamacpp_cassette` and the per-configuration
wrappers in `tests/providers/llamacpp/cassette_support.rs`: strict ordered
`ProviderCassette`, `finish_after_test` exhaustion, credential-free
`llamacpp::Client::from_url_with`). No hosted provider and no credential are
involved, and replay needs no local server. The default smoke tier is
`unsloth/Qwen3-1.7B-GGUF` Q4_K_M (`--jinja -c 4096`); cells that name the
competent tier (`Qwen3-8B` Q4_K_M, `-c 8192`) or another server say so in
their rule.

## What a classification here claims

A row classified `agent` means the original test constructs a rig-agent
builder (`client.agent(..)`, `AgentBuilder::new(..)`, `client.extractor(..)`,
`into_agent_builder()`) and drives its orchestration (`prompt`, `prompt_typed`,
`chat`, `stream_prompt`, `stream_chat`, `runner(..).run()`, `extract`). A row
classified `shared_provider` means the reviewed body calls a provider
capability directly (`CompletionModel::{completion,raw_completion,stream}`,
`completion_request(..).send()/.stream()`, embeddings, rerank, model listing,
`verify`, transcription, image generation) or a shared driver that does so, and
never constructs an agent. A row classified `infrastructure` executes no rig
provider client and no agent at all. Classification is from source reading and
the base listing `superset-lanes/ecs-tests-a7eb63a91.json`; it is not an
execution result. Ignored rows are discovered, never executed; nothing in this
contract is a parity or superset verdict.

## Cells

| Cell | Class | Rule | Fixture(s) | Ignored |
| --- | --- | --- | --- | --- |
| `cassette::prompt_caching::agent_loop_does_not_move_its_own_prefix` | agent (tools) | agent-prompt-tools | `llamacpp/prompt_caching/agent_loop.yaml` | no |
| `cassette::streaming_tools::example_streaming_with_tools` | agent (tools) | agent-stream-tools | `llamacpp/streaming_tools/example_streaming_with_tools.yaml` | no |
| `cassette::streaming_tools::streaming_tools_emit_tool_call_before_later_text` | agent (tools) | agent-stream-tools | `llamacpp/streaming_tools/streaming_tools_emit_tool_call_before_later_text.yaml` | no |
| `cassette::streaming_tools::streaming_tools_smoke` | agent (tools) | agent-stream-tools | `llamacpp/streaming_tools/streaming_tools_smoke.yaml` | no |
| `cassette::streaming_tools::streaming_tools_surface_two_distinct_tool_calls_before_final_answer` | agent (tools) | agent-stream-tools | `llamacpp/streaming_tools/streaming_tools_surface_two_distinct_tool_calls_before_final_answer.yaml` | no |
| `cassette::tool_matrix::a_one_argument_tool_round_trips_its_value` | agent (tools) | agent-prompt-tools | `llamacpp/tool_matrix/one_argument_tool.yaml` | no |
| `cassette::tool_matrix::a_tool_that_errors_reports_the_error_back_to_the_model` | agent (tools) | agent-prompt-tools | `llamacpp/tool_matrix/tool_that_errors.yaml` | no |
| `cassette::tools::tools_roundtrip` | agent (tools) | agent-chat-tools | `llamacpp/tools/tools_roundtrip.yaml` | no |
| `cassette::tools::tools_smoke` | agent (tools) | agent-prompt-tools | `llamacpp/tools/tools_smoke.yaml` | no |
| `cassette::typed_prompt_tools::prompt_typed_with_tool_call_roundtrip` | agent (tools) | agent-typed-tools | `llamacpp/typed_prompt_tools/prompt_typed_with_tool_call_roundtrip.yaml` | no |
| `cassette::typed_prompt_tools::prompt_typed_with_tool_call_verbatim_roundtrip` | agent (tools) | agent-typed-tools | `llamacpp/typed_prompt_tools/prompt_typed_with_tool_call_verbatim_roundtrip.yaml` | no |
