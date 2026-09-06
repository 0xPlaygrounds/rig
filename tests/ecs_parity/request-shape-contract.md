# Request-shape family contract

Scope: the 13 Anthropic corpus_request_shape cassette producer scenarios. The separate schema-literal unit
case stays in its original module and does not receive an agent mapping.

## Execution and configuration

Each native case owns a fresh App using ordinary native BusPlugin/AgentPlugin,
CompletionAdapter and ToolAdapter. Existing provider wrappers perform request
matching and exhaustion against the original fixtures. Original constants and
reasoning/text helpers are imported, with visibility-only changes to the original
module. No original runner, hook, replayer, or golden response executes natively.

The model is CLAUDE_SONNET_4_6. Original owner/model/tool keys are declared as
`golden`, `golden/model:default`, `golden/tool:add#0` for log interoperability.
DefaultMaxTurns remains None, with effective default 1; tool cases override the
run budget to 3, forced Required/Specific cases to 2. Temperature is zero except
both thinking cases, where it remains unset. The exact original thinking JSON,
preambles, appended newline/paragraph, no-preamble None, cap32, schema, two context
documents with static_doc IDs, and prior two-message history are preserved.

Native expected-failure observation returns the actual Failed(Failure::MaxTurns
{limit:2}) after native tool completion. It does not convert errors to successful
answers. The shared success helper still rejects all failures and recorded stream
errors, including errors after a final event. Its synthetic budget control checks
all four completed effect records remain after failure and the run identity is
recorded. Ordinary native stamp_header computes the builder fingerprint from the
actual graph; stamp_run computes the effective scoped identity. Neither hash is
copied from the original golden.

## Assertions and golden equivalence

Every explicit original assertion is retained, with `response.output` observed as
the native final String and MaxTurnsError mapped to native Failure::MaxTurns with
the same limit. The native tests still inspect actual recorded requests, full
family sequence, nonempty/fixed text, empty tool-choice-none answer, reasoning
blocks, event presence, structured output, and prior-history message count.

The original complete effect golden is compared as serde data, including builder
fingerprint, handler descriptors/signature/required rows, bus policy, hook list,
requests, outcomes/raw responses, usage, provider IDs, events and tool outputs.
Only these representation differences are normalized:

- Effect IDs are nominal across independent runtimes. Both logs must already be
  strictly increasing in dispatch order and have equal length. A bijection by
  position maps native IDs and parents to the original IDs. Parents must exist
  and precede children; the resulting causal relationships compare exactly.
- Legacy producer records have no Scope, and their headers have no scoped
  programs. Every native record must retain a scope and the native program map
  must be nonempty. These native-only fields are omitted only from cross-runtime
  comparison and retained in the separate native stable golden.
- Legacy producer headers lack delivery/poll-batch traces (asserted). Native
  traces depend on transport polling and are omitted from stable golden
  comparison. These tests do not claim timing or delivery-group equivalence.

No request, response, event, original header field or causal relationship is
removed to make the comparison pass. Shared negative controls demonstrate that
normalization rejects changed request caps, reordered effects and a missing
builder fingerprint. Each `ecs_parity/anthropic_request_shape_*.effects.json` is generated
by the real native provider run through unchanged cassettes, never handwritten.
It additionally checks stable native scope/program metadata and native IDs.

## Recording

Record mode follows the original golden helper: reject simultaneous golden
regeneration, then allow the wrapper to scrub/save the cassette before comparing
placeholder-bearing goldens on replay. Live raw logs are not written by this
helper. Ordinary regression runs use replay with golden regeneration disabled.
