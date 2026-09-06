# Concurrent streamed tool family contract

Scope: the two Anthropic streaming_tools cells named
serial_serving_reproduces_the_recorded_request_order and
streaming_tool_concurrency_surfaces_results_in_call_order_after_batch_settles.
Both use the original shared provider cassette, Sonnet 4.6, unchanged two-tool
prompt/preamble and gated OutOfOrderAlphaSignal/OutOfOrderBetaSignal implementations.
The gates require overlapping execution without sleeps; they do not pin local
completion order. Run budget eight, streaming,
tool concurrency two, default temperature and a five-second failing deadline are
preserved. The serial case enables per-handler serial serving; the other uses
ordinary concurrent serving. Distinct tool keys must remain concurrently runnable.

Native EcsAgent drives ordinary plugins and real CompletionAdapter/ToolAdapter
handlers. It does not replay effect answers or invoke the legacy runner. Original
helper/tool visibility is widened only to the sibling module. All tool fields and
implementation methods remain unchanged .

The original collector drains through EOF, records errors and requires a final
response via assertions. Native success waits for settlement, rejects stream
errors, and inspects its separate native observation's error/final fields. Actual
completion tool calls supply call order and the shared original event helper
checks that both calls precede the first result. Final response text retains both
original signal assertions in the concurrent case.

Legacy streamed tool-result items publish atomically in call order after the
whole batch settles. The native analogue is the supported world publication
boundary: an observer after Materialise sees Added<Parts> on Utterance entities,
asserts every tool has an EffectOutcome, and records result names resolved through
actual ToolCallSlot IDs. The concurrent case requires exactly one published batch
in original call order. Independently, the final world history is traversed in
Order, requiring both flattened history and its final result message to preserve
call order. The serial case retains its original flattened-history assertion.
There is no claim that raw bus completion order equals semantic publication order.

Original cassette grouping and ordering helpers are reused without strengthening
or weakening their meaning. They inspect committed provider request shapes;
native history/publication assertions and replay request matching provide the
runtime evidence. No new effect golden or builder-identity equivalence is claimed
for these two original non-golden producer cases.

These two cases do not establish all concurrency or interruption guarantees.
