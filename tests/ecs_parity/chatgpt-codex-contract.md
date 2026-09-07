# ChatGPT Codex agent family

The ten recorded agent cells against the Codex Responses backend: instruction
merging, sequential and parallel tool sessions, nested tool arguments, a
reasoning-enabled tool session, usage accumulation and the non-interactive
OAuth streaming smoke. Native counterparts belong in
`tests/providers/chatgpt/cassette/ecs_codex.rs`.

## Rules

- **agent-chat** — `client.agent(GPT_5_4).preamble(..).build().chat(prompt, &mut history)
  .await.expect(..)`; the caller-owned history may carry a mid-conversation
  `Message::system`; output markers asserted.
- **agent-chat-tools** — `client.agent(GPT_5_4).preamble(..).tool(..)
  .default_max_turns(n).build().chat(.., &mut history).await.expect(..)`;
  besides the answer, the history's tool calls and results are paired by
  `call_id` and their order asserted (`history_tool_calls` /
  `history_tool_results`), or the recorded tool-call arguments checked.
- **agent-stream-tools** — `client.agent(GPT_5_4).preamble(..).tool(..)
  [.max_tokens(6000).additional_params({"reasoning": {"effort": "low"}})].build()
  .stream_chat(.., Vec::new()).max_turns(n).stream().await` or
  `.stream_prompt(..).max_turns(n).stream().await`, collected by
  `collect_stream_observation`, `reasoning::collect_stream_stats` or a manual
  drain for `StreamUserItem` / `FinalResponse` usage; exact tool-call order,
  result counts, reasoning blocks and final text asserted as the cell states.
- **agent-stream** — the non-interactive OAuth cell: `client.authorize()` then
  `client.agent(GPT_5_4).stream_prompt(BASIC_PROMPT).stream()` drained by
  `collect_stream_final_response`.

## Exceptions

`codex_behaviors::{strict_tools_opt_in_roundtrip,store_false_and_prompt_cache_fields_roundtrip}`,
`codex_sessions::long_history_replay_nonstreaming`, four `codex_tool_args`
cells, all `codex_tool_choice`, `http_errors` and `cassette::streaming_tools`
call the model directly (`chatgpt-provider-contract.md`).

## Phase 2 obligations

History correlation (call id to result, order), exact tool-call sequences,
aggregated usage across turns, reasoning block presence and the default
instructions / preamble merge are level-1/2 obligations on actual run-child
utterances and effect outcomes; the OAuth client construction is configuration
to preserve.

## Fixture kind

Cassette cells are **hosted ChatGPT Codex backend recordings**
(`https://chatgpt.com/backend-api/codex`, model `GPT_5_4`) replayed through
`with_chatgpt_cassette` / `with_chatgpt_cassette_default_instructions` (strict
ordered `ProviderCassette`, placeholder `CHATGPT_ACCESS_TOKEN` /
`CHATGPT_ACCOUNT_ID`, `finish_after_test` exhaustion). The non-interactive
OAuth cell seeds a temp `auth.json` from the cassette keys with
`allow_device_flow(false)`. Every `tests/providers/chatgpt/*.rs` cell outside
`cassette/` is `#[ignore]` live-only: ChatGPT needs an OAuth device flow or a
cached `auth.json`, neither available here; they stay `supplemental_live`
(OAuth), not counted as ported. The `unrecorded` raw matrices have complete
bodies but no fixture.

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
| `cassette::codex_behaviors::default_instructions_merge_with_explicit_preamble` | agent (codex) | agent-chat | `chatgpt/codex_behaviors/default_instructions_merge_with_explicit_preamble.yaml` | no |
| `cassette::codex_behaviors::explicit_preamble_and_mid_conversation_system_messages_are_instructions` | agent (codex) | agent-chat | `chatgpt/codex_behaviors/explicit_preamble_and_mid_conversation_system_messages_are_instructions.yaml` | no |
| `cassette::codex_sessions::parallel_tool_calls_single_turn_nonstreaming` | agent (codex) | agent-chat-tools | `chatgpt/codex_sessions/parallel_tool_calls_single_turn_nonstreaming.yaml` | no |
| `cassette::codex_sessions::parallel_tool_calls_single_turn_streaming` | agent (codex) | agent-stream-tools | `chatgpt/codex_sessions/parallel_tool_calls_single_turn_streaming.yaml` | no |
| `cassette::codex_sessions::reasoning_session_two_tool_calls_streaming` | agent (codex) | agent-stream-tools | `chatgpt/codex_sessions/reasoning_session_two_tool_calls_streaming.yaml` | no |
| `cassette::codex_sessions::sequential_tool_calls_nonstreaming` | agent (codex) | agent-chat-tools | `chatgpt/codex_sessions/sequential_tool_calls_nonstreaming.yaml` | no |
| `cassette::codex_sessions::sequential_tool_calls_streaming` | agent (codex) | agent-stream-tools | `chatgpt/codex_sessions/sequential_tool_calls_streaming.yaml` | no |
| `cassette::codex_sessions::usage_accumulates_across_streaming_multi_turn` | agent (codex) | agent-stream-tools | `chatgpt/codex_sessions/usage_accumulates_across_streaming_multi_turn.yaml` | no |
| `cassette::codex_tool_args::nested_arguments_roundtrip_nonstreaming` | agent (codex) | agent-chat-tools | `chatgpt/codex_tool_args/nested_arguments_roundtrip_nonstreaming.yaml` | no |
| `cassette::noninteractive_oauth::cached_oauth_allows_noninteractive_streaming_completion` | agent (codex) | agent-stream | `chatgpt/noninteractive_oauth/cached_oauth_allows_noninteractive_streaming_completion.yaml` | no |
