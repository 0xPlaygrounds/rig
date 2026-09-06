# Complex tool-session parity

This family covers the 27 scenarios in the original DeepSeek, xAI and OpenRouter
agent_tool_sessions files:14 invoke agent orchestration and13 call provider
models directly. The latter execute unchanged in both revisions and are shared
provider coverage, not migrated agent behavior. The xAI image case is agent
behavior; its filename neighbors are not grounds for excluding it.

## Independent execution

The twelve sequential/parallel agent cases use ordinary native plugins, real
provider adapters and the exact original Tool implementations. Their schemas,
descriptions, serialization, invocation log, and validators are shared unchanged;
only sibling visibility widens. No legacy runner or policy drives native cases.
`run_session` reads actual settled output and run-child Utterance/Order/Parts,
sorted by runtime Order. It does not manufacture history from expected tools.
These scenarios begin with empty history, matching the originals.

Streaming uses the existing native schedule observers. Names come from actual
ToolCallSlots, results from actual EffectOutcome, text from bus stream deltas,
and final output from RunResult/Settled. The bus requires channel closure before
publishing completion outcome. The native wait rejects all retained public
stream errors, retaining the original collector's EOF/error obligations. Fields
not observed by this helper are not consumed by these scenarios' assertions.
This maps the recorded event constraints, not universal scheduling equivalence.

## Settings and original obligations

| Surface | Settings and observations |
| --- | --- |
| Sequential blocking, three providers | Exact four original tools in order, original complex prompt/preamble, parallel_tool_calls false. Builder default_max_turns10; no run override. Exact invocation order/count and the original argument predicates. Actual caller history has expected calls, one result per call, and call-before-result-before-next-call ordering. Final text contains all four markers. |
| Sequential streaming, three providers | Same tools/inputs/options; unset builder default (effective1), run override10. Exact four streamed tool names in order, exactly4results, required final response text and markers, exact invocation validator. OpenRouter additionally requires got_final_response. |
| Parallel blocking, three providers | AlphaSignal then BetaSignal, original two-tool prompt/preamble. Builder default5; no run override. Exactlytwo history calls containing both names on one assistant message, exactlytwo results, final markers. DeepSeek/xAI explicitly enable parallel_tool_calls; OpenRouter leaves it unset. |
| Parallel streaming, three providers | Unset builder default (effective1), run override5; same provider option distinction. Same shared assert_two_tool_roundtrip_contract requires no errors, final response, final text equal to final-turn deltas, sufficient tool results, calls/results before first text, expected initial unique tools, and final markers. |

DeepSeek additionally retains thinking disabled, merged through the unchanged
non_thinking_params/json_utils_merge helpers. Every provider retains its exact
SESSION_MODEL. The complex validator requires an empty ping object, the original
project/flags/note, two manifest steps, exact labels/separator, and escaped echo
text. It does not independently assert every step name or weight.
Original lock, serialization, optional-event, indexing and `?`
requirements remain part of the contract beyond direct assertion macros.

OpenRouter's nested typed scenario retains its original model, preamble and
provider order/require_parameters JSON. Schema and explicit Native mode belong
to the run, with no added budget override. The independently implemented pure
JSON decoder preserves the original direct-parse then first-JSON-value fallback
for fences/prose; invalid first JSON remains an error. Decoder controls include
plain/fenced/prose/array/error inputs. All original canary/low and required
compile/replay checks remain on the actual decoded result.

xAI's image case retains VISION_MODEL, original preamble, original JPEG bytes,
media type and base64 helper. Before scheduling, the fresh native user utterance
is assigned the actual text-image-text parts in the original order. Native
settlement must succeed and its output must pass the original trim-nonempty
check. The original JPEG remains a consumed fixture in Git.

## Shared provider partition

DeepSeek6: raw tool-argument stream, long history, required/specific/none tool
choice, reasoning stream/usage, chat/reasoner aliases, JSON-object format.
xAI5: raw tool-argument stream, long history, tool choice, reasoning metadata,
JSON-schema response format. OpenRouter2: raw tool-argument stream and long
history. These construct requests directly; definitions do not execute tools.
Raw parsing/normalization, metadata helpers, stream drainage and their original
assertions execute unchanged in both checkouts. No ECS mapping is invented.

The same result wrappers require complete strict ordered cassette consumption.
These scenarios do not establish exhaustive capability, concurrency, feature or
interruption coverage.
