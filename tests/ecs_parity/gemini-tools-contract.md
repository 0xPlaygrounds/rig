# Gemini tool execution conformance

All six original `agent_tools_e2e` cases have native `ecs_tools_e2e` counterparts.
The adjacent streamed-run cases have separate access and diagnostics contracts.

## Independent execution and configuration

All six use GEMINI_2_5_FLASH and temperature 0. Chained cases keep the original
FORCE_TOOLS_PREAMBLE, prompt, add/subtract registration order, unset default
turn override (effective one), and per-run max_turns 5. They reuse the original
CountingAdd/CountingSubtract and real counters from tools_support.

Conformance scenarios retain original helper-specific preambles/prompts, tool
definitions and registration order. Both parallel cases set default and run
turn limits to 3; zero-argument sets both to 2; serialization sets both to 3.
Default tool concurrency is sequential in both engines. The explicit-one case
sets ToolPolicy on the actual run, preserving per-run override semantics.

Native execution installs real provider/tool adapters via EcsAgent and drives
native systems. NativeResponse is observed data, not a legacy PromptResponse:
output from actual settlement, usage from the run Usage component, requests
from successful Completion effect outcomes, and history from actual ordered
run-child Utterances. A one-run assertion bounds outcome counting. These
successful cases have no rejected attempts; failure and multi-run counting
would require broader observations.

The conformance module copies neutral tool definitions and assertion helpers
from model_conformance.rs while replacing execution with native ECS.
It imports only neutral ScenarioError/ScenarioReport and the pure protocol
validator from rig-agent; it never calls the legacy conformance runner. The
local contract error constructor preserves the original error data. Duration
in reports is incidental elapsed time, not a performance comparison.

## Preserved assertion closure

| Cells | Original obligations retained |
| --- | --- |
| Chained unary | Successful prompt; final output mentions 37; add/subtract execute once each; at least two requests; positive aggregate total usage; messages present and contain a user tool-result message |
| Chained streamed | Drain through EOF; no errors; add and subtract call names present; at least two results; each tool executes once; terminal response exists and mentions 37 |
| Both parallel | History present and correlated; an assistant turn has exactly two calls named add/subtract; immediately following result message has exactly two semantic integer values 7 and 8; each tool executes once |
| Zero argument | Correlated history; one ping execution; verbatim marker in result values and final response |
| Serialization | Correlated history; verbatim two-line motto; configuration JSON equals the original object, accepting the same semantic JSON string alternative; one call to each tool |

Correlation counts matches by assistant-turn number, native call handle and
optional provider call ID. Each call must have exactly one result; result and
call totals must match. The original predicate does not independently require
all call handles to be globally unique. No ID normalization is applied here.

All four helper-driven cases retain report_from_response's protocol-hygiene
validator over the actual output and history, including the original forbidden
marker list. Error-return predicates and Option/error propagation are treated
as obligations, not only assert macros. Tool result values keep original text
and JSON handling; no stripping of quotes or normalization of user arrays.

The chained stream uses existing native schedule observers for actual calls,
results and final publication. EcsAgent additionally rejects all retained bus
stream errors before successful settlement, including errors the narrower
StreamObservation does not itself store. Fields unused by the original
assertions remain unobserved; no stronger event-uniqueness claim is made.

Both runtimes use unchanged strict ordered cassette wrappers and complete
consumption/teardown. Replay uses the committed fixtures. This family does not establish OS-level
network isolation, universal scheduling behavior or a complete superset verdict.
