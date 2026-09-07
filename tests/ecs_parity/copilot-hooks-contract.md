# Copilot agent hooks family

## Rules

- **agent-dispatch-hook** — `client.agent(LIVE_LIGHT_MODEL).tool(ReadFileHead)
  .tool(ReadFileTail).build().prompt(..).max_turns(5).add_hook(PermissionHook)`;
  `on_dispatch` skips the first tool call with a substituted message,
  `on_outcome` captures the tool result; `ensure!(last == "hello world")` and
  `call_count >= 2`. The streaming twin (`permission_control_streaming_example`)
  drives the same hook through `live_client()...stream_prompt(..).max_turns(5)
  .add_hook(PermissionHook).stream()` via `stream_to_stdout` and is live-only
  (`#[ignore]`, `supplemental_live`).
- **agent-completion-hook** — `client.agent(LIVE_MODEL).build().prompt(..)
  .add_hook(SessionIdHook)`; `ensure!` on one prompt call, one response call and
  the captured prompt/response.

## Phase 2 obligations

Level-3 obligations: the denied dispatch and its substituted result must be
policy-visible; one completion-call observation and one outcome observation.
The permission cell writes `test.txt` in the working directory (unlike the
llama.cpp copy); the native port keeps the tool descriptions so recorded bodies
match.

## Fixture kind

Cassette cells are **hosted GitHub Copilot recordings** replayed through
`with_copilot_cassette` / `with_copilot_cassette_result` (strict ordered
`ProviderCassette`, placeholder `GITHUB_COPILOT_API_KEY`, `finish_after_test[_result]`
exhaustion). Chat models (`GPT_4O`, `GPT_4O_MINI`) route through Chat
Completions, codex models (`live_responses_model()`, default `GPT_5_3_CODEX`)
through the Responses route. `#[ignore]` cells are live-only: Copilot needs an
API key, a GitHub access token or an OAuth device-flow cache, none configured
here; they stay `supplemental_live` (OAuth), not counted as ported. The
`unrecorded` raw matrices have complete bodies but no fixture on disk.

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
| `permission_control::permission_control_prompt_example` | agent (hooks) | agent-dispatch-hook | `copilot/permission_control/permission_control_prompt_example.yaml` | no |
| `permission_control::permission_control_streaming_example` | agent (hooks) | agent-dispatch-hook | none | yes |
| `request_hook::request_hook_records_prompt_and_response` | agent (hooks) | agent-completion-hook | `copilot/request_hook/request_hook_records_prompt_and_response.yaml` | no |
