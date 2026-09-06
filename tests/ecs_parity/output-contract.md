# Output-mode family contract

Scope: all ten Anthropic corpus_output producer cells using the original scenarios. Each native case executes real provider
adapters with ordinary ECS plugins against the unchanged original HTTP cassette.
Original SUM_EVENT_PROMPT, request_at, tool_names and assert_event are shared by
sibling visibility only. The exact original event schema is converted to JSON
for native Output; no output validation or mode resolution is implemented in the
parity harness.

Every case preserves the original owner, model, preamble, schema and mode. Native
OutputKind is Tool or Prompted as originally requested, including the None-choice
case: the native runtime must itself degrade Tool to Native. Temperature is zero
except tool_thinking, where it remains unset and the original thinking JSON with
budget1024 is inserted. Agent default_max_turns is None, effective one. The two
real-tool cases override the run to three turns; Required overrides to two; all
other cases keep the default. Streamed cases retain all stream events.

Every original assertion remains: JSON parsing and title/summary string checks,
full effect-family sequence, actual tool name lists, preamble augmentation,
native-schema presence/absence, event retention and the Tool-with-real-tool sum.
The synthetic final_result is settled by native output systems without tool
handler dispatch. Adder is a real ToolAdapter execution in the two real-tool
cases. Native success requires settlement and rejects all failures and stream
errors through the completed provider stream.

The full original golden comparator and normalization rules are unchanged from
request-shape-contract.md. Actual native stamp_header/stamp_run produce identity;
no golden header or outcome supplies execution. Every original header, request,
response, event, usage and tool-output field compares; only the documented nominal
IDs/parents and native-only scope/program/delivery representation differ. Native
stable goldens retain scoped identity; delivery grouping is not compared. Stable
native goldens are generated from provider replay, never handwritten.

Shared comparator negative tests reject changed requests, reordered effects and
missing builder identity. This family does not establish exhaustive parity.
