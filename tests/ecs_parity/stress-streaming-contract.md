# Gemini streaming stress parity

Scope: all six hook_stress_streaming executable tests on the native host.
The blocking/streaming comparison test contains two workflows under one test ID;
all seven original cassettes are frozen and executed. Original producers and
support remain unchanged. No original effect golden exists for this family.

Each native App uses the actual Gemini2.5Flash adapter and unchanged neutral
CountingAdd/CountingSubtract implementations. Prompts, preambles, owner name,
tool order, empty initial history, undeclared effective default1 and original
per-run limits2/4/5/6 are preserved. Temperature is0 except the active-tools case,
whose builder leaves it unset and whose per-turn patch sets0. The comparison
test constructs separate blocking and streaming Apps as the original does.

The independent run-owned EventTap stores only observations used by these cases.
Stream mode comes from the actual run component. TextDelta observations read
published bus Streamed.events, selecting actual BlockDelta::Text items and
maintaining a per-effect read cursor. CompletionResponse observes each successful
new EffectOutcome after Collect and before bus Judge. A separate system observes
the native turn's Outputs.done after agent Fold and before agent Judge, marking
the turn once for ModelTurnFinished. These are two actual schedule boundaries,
not duplicate reads of a final answer. ToolCall observes actual new tool intents.
The original valid-call cases retain >=1 text/response/finished observations,
>=1 tool call, >=2 finished turns and exact response/finished count equality.
The text test's name mentions stream finish but has no separate StreamFinish
event assertion. No invented assertion or invalid-call/retry timing claim is made.

Result redaction reuses the reviewed native tools.rs outcome policy; the final
response must contain STREAM-REDACTED-Q3 and exclude10. Active-tool narrowing
installs the original native RequestPatch on each Fresh turn before Assemble:
add remains executable and subtract's actual counter stays0. Skipping supplies
the actual structured ToolResult::skipped with the original reason in native Gate
before issuance; it does not run the handler. The original subtract counter0 and
nonempty final response remain. All next requests, including rewritten/skipped
tool results, are checked against original strict provider cassettes.

The original collect_stream_final_response drains through EOF, propagates every
item error and requires a final response. Native bus collection drains before
publishing its outcome; the shared success consumer requires actual settlement
and RunResult and rejects every public Streamed.errors item, including errors
after Final. This is the semantic collector mapping, not a legacy stream facade.
The original collector selects the last final without asserting uniqueness.
The comparison test reuses assert_mentions_expected_number12, including its
nonempty check and case-insensitive substring matching; it does not assert exact
equality of the two answer strings or parse an integer.

Scope is one active run per App with these valid calls. No original AgentHook or
agent orchestration executes. Broader concurrency, interruption and capability guarantees remain outside this
single-active-run scope.
