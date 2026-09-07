# Additional provider completion parity

This family covers 16 original blocking agent scenarios across 11 providers:
Bedrock (4), Cohere (1), Copilot (1), DeepSeek (1), Doubleword (1), llama.cpp (1),
Ollama (2), OpenRouter (1), Perplexity (2), Venice (1), and xAI (1). Exact original
and native executable IDs and fixture paths are in the `scenarios/` catalog directory. Unrelated
provider and ignored live tests are outside this family.

## Independent execution and settings

Each native producer uses `EcsAgent` with ordinary `BusPlugin` and `AgentPlugin`
configuration. The real provider's `CompletionAdapter` is polled through the
existing `RuntimeHandler`, which enters Tokio on every poll while ECS owns the
future. Bedrock tools use real `ToolAdapter`s through the same bridge. No legacy
agent builder, runner, policy or effect replayer drives a native case.

Original model expressions, preambles, prompts, context, options and assertion
arguments are retained. The Rust AST conversion replaced the agent builder and
prompt await boundary; it preserved the original response assertion tail.
Both runtimes use empty initial history, blocking execution and no per-run turn
limit override. Unconfigured builders have an effective limit of one turn:
original `AgentConfig::new` stores `max_turns = 1`; native `DefaultMaxTurns(None)`
marks the unset override and `MaxTurns(1)` supplies the effective limit. Bedrock's
tool case explicitly sets the default limit to two in both runtimes.

| Cases | Distinct configuration preserved |
| --- | --- |
| Cohere completion | `COMMAND_A_03_2025`, basic inputs, temperature 0.2 |
| Copilot completion | `LIVE_MODEL` (`GPT_4O`), basic inputs, existing API-key cassette client |
| DeepSeek completion | `DEEPSEEK_V4_FLASH`, basic inputs |
| Doubleword, OpenRouter, Venice completion | Each original parent's `DEFAULT_MODEL` and original adapter, basic inputs |
| llama.cpp completion | `Qwen3-1.7B-Q4_K_M`, basic inputs, original credential-free boxed Reqwest client |
| Ollama completion and limit | `qwen3:4b`, `think: false`, basic inputs; only limit case sets `max_tokens = 24` |
| Perplexity completion | `SONAR`, basic inputs, temperature 0.2 |
| Perplexity options | `SONAR`, exact original custom prompt/preamble, `return_related_questions: true`, `search_context_size: low`, temperature unset |
| Bedrock completion and caching | `AMAZON_NOVA_LITE`, basic inputs; caching case preserves the actual model's `with_prompt_caching()` |
| Bedrock context | Exact original preamble/prompt and all three `CONTEXT_DOCS`, document IDs `static_doc_0` through `static_doc_2`, empty additional properties, original order |
| Bedrock tools | Original tool prompt/preamble, `max_tokens = 1024`, Adder then Subtract, default turn limit 2 |
| xAI completion | `GROK_3_MINI`, basic inputs |

## Original obligations and observations

All 16 original `prompt(...).await.expect(...)` boundaries require success. The
native helper requires actual settlement and `RunResult`, fails on native
`Failed`/`Failure`, and treats its timeout as failure. Its result is observed
from the native world, never synthesized from the expected response.

Fourteen cases call the unchanged `assert_nonempty_response`, which trims the
answer and rejects whitespace-only output. Bedrock context calls unchanged
`assert_contains_any_case_insensitive`, including its nonempty prerequisite,
and requires `ancient tool` OR `farm`. Bedrock tools calls unchanged
`assert_mentions_expected_number(-3)`, including its nonempty prerequisite,
and requires substring `-3`, `minus 3` OR `negative 3`. The last helper is not a
numeric parser. The original tool case has no tool-count assertion; the port
retains real tools and the original provider exchange without inventing one.
No case asserts exact equality of final answer strings between runtimes.

Every native case retains its provider-specific cassette wrapper and literal
fixture path. The wrappers construct clients pointing at the same loopback
replay server, catch closure panics, and call `finish_after_test` on success.
`finish -> ReplayServer::assert_consumed -> assert_replay_finished` rejects
unused interactions and recorded misses before shutting down the replay
server. If the body panics, that panic is already failure; teardown is not
claimed to run on that path. The original ordered matcher checks method,
route, query, applicable headers and request bodies under each provider's
unchanged policy. Both runtimes consume the same immutable recordings.

Bedrock's direct recorder is a recording-mode distinction. Replay uses the
same strict server and original AWS SDK configuration with explicit dummy
credentials, fixed region and loopback endpoint. Copilot uses its cassette
API-key constructor, not the ignored live all-model/OAuth workflow. Ollama and
llama.cpp do not require a local model server during replay. These facts do not
establish operating-system network isolation.

The claim is scoped provider request/response and actual final-result fidelity
for these recorded blocking workflows, with root default features plus
`bedrock` on the native host. There are no original effect goldens, stream
collectors, hooks or raw-response/usage assertions in this family. It does not
prove stream terminal counts, policy delivery grouping, general tool scheduling,
all feature combinations, ignored live behavior, or all provider agent scenarios.
Other agent families remain outside this scope.
