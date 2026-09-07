# llama.cpp agent hooks family

Agent cells whose obligations are policy-visible hook observations. Native
counterparts belong in `tests/providers/llamacpp/cassette/ecs_hooks.rs`; the
anthropic-owned `hooks-contract.md` documents the native hook mapping and is
referenced, not edited.

## Rules

- **agent-runner-hook** — `client.agent(..).temperature(0).max_tokens(cap)
  .add_hook(TurnTerminationProbe)[.tool(Adder)][.add_hook(EscalateCapOnTruncation)]
  .build().runner(prompt)[.max_turns(n)].run().await.expect(..)`; the probe's
  `ModelTurnFinished` observations (`finish_reason`, `max_tokens` per attempt)
  are asserted after closure together with the recorded caps and wire reasons.
- **agent-stream-hook** — `client.agent(..).max_tokens(24).build().stream_prompt(..)
  .add_hook(TurnTerminationProbe).stream().await` drained by
  `collect_stream_final_response`; the probe must observe `Length` and the cap.
- **agent-dispatch-hook** — `client.agent(..).tool(ReadFileHead).tool(ReadFileTail)
  .build().prompt(..).max_turns(5).add_hook(PermissionHook)` (or the
  `stream_prompt` twin drained by `collect_stream_observation`); the hook's
  `on_dispatch` skips the first tool call with a substituted message and
  `on_outcome` captures tool results; `ensure!` on the captured result,
  `call_count >= 1`, and for streaming no errors, a final response,
  `tool_results >= 1` and the skipped `read_file_head` call in the stream.
- **agent-completion-hook** — `client.agent(..).build().prompt(..).add_hook(SessionIdHook)`
  whose `on_completion_call` / `on_outcome` record the prompt and response;
  `ensure!` on `prompt_calls == 1`, `response_calls == 1` and the captured text.

## Phase 2 obligations

These are level-3 (policy-visible) obligations: per-attempt finish reason and
budget, denied/skipped dispatch with the substituted tool result reaching the
model, and one observation per completion call. `EscalateCapOnTruncation`'s
`RequestPatch` retry maps to a native per-attempt budget override; the recorded
request caps (`24` then `512`) are the wire evidence. Scratch files are
per-cell temp paths (the tool descriptions still say `test.txt`), which the
port keeps so recorded bodies match.
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
| `cassette::permission_control::permission_control_prompt_example` | agent (hooks) | agent-dispatch-hook | `llamacpp/permission_control/permission_control_prompt_example.yaml` | no |
| `cassette::permission_control::permission_control_streaming_example` | agent (hooks) | agent-dispatch-hook | `llamacpp/permission_control/permission_control_streaming_example.yaml` | no |
| `cassette::request_hook::request_hook_records_prompt_and_response` | agent (hooks) | agent-completion-hook | `llamacpp/request_hook/request_hook_records_prompt_and_response.yaml` | no |
| `cassette::turn_termination_matrix::blocking_completed_turn_reports_stop` | agent (hooks) | agent-runner-hook | `llamacpp/turn_termination_matrix/blocking_completed_turn.yaml` | no |
| `cassette::turn_termination_matrix::blocking_escalating_retry_reports_each_attempts_own_cap` | agent (hooks) | agent-runner-hook | `llamacpp/turn_termination_matrix/escalating_retry.yaml` | no |
| `cassette::turn_termination_matrix::blocking_tool_turn_reports_tool_calls` | agent (hooks) | agent-runner-hook | `llamacpp/turn_termination_matrix/blocking_tool_turn.yaml` | no |
| `cassette::turn_termination_matrix::blocking_truncated_turn_reports_length_and_cap` | agent (hooks) | agent-runner-hook | `llamacpp/turn_termination_matrix/blocking_truncated_turn.yaml` | no |
| `cassette::turn_termination_matrix::streaming_truncated_turn_reports_length_and_cap` | agent (hooks) | agent-stream-hook | `llamacpp/turn_termination_matrix/streaming_truncated_turn.yaml` | no |
