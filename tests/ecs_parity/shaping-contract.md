# Per-turn shaping family contract

Scope: all 12 Anthropic corpus_shaping producer cells at baseline
805fb18e6135c9050ee7ca4295d96ddf08cb223f, using the unchanged original HTTP cassettes
and original full effect goldens. The native module shares original prompt
constants, request extraction, schemas, and assertion helpers. Every runtime
operation uses native ECS plugins, real provider adapters and real Adder execution.

## Configuration and policy timing

All cases use Sonnet 4.6, owner `golden`, temperature zero, and the original
preamble and tool configuration. Agent DefaultMaxTurns is None (effective one);
unary runs override to three, while streamed extra context keeps the default.
The committed-output cell uses the same event schema and Tool output mode, and
must succeed through native output validation and reprompting on turn three.

Concrete request policy systems act on Added<Fresh> after Select and before
Assemble. They read the owning run's one-based Cursor and insert a native
RequestPatch on that turn only. The macro shares traversal, not policy decisions
or a legacy hook interpreter. Each patch preserves the original values and turn
predicate. The three merged systems run in registration order with deferred
commands applied between systems; native RequestPatch::merge combines their data.

Route selection runs after Advance and before Select. The first-turn route uses
the registered Haiku 4.5 handler only on turn one, explicitly restoring the agent's
default model on later turns. Its Route relationship is present when required-row
identity is stamped. The late-route case registers a real handler without adding
a builder Route relationship; it selects that handler on every fresh turn. Thus
it appears in signature/descriptors but not the builder's required row.

Native stamp_header computes the builder identity from the actual graph. The
legacy hook names are explicit semantic declarations accompanying the installed
native systems; they are not evidence that a builder hash fingerprints system
code. Native PolicyVersion declares `ecs-shaping/v1:<ordered-system-names>` and
stamp_run records that declaration in scoped identity. Source-index provenance
binds execution to the concrete system implementation.

## Assertions and comparison

All original assertion tails are preserved, including request resets, complete
family sequences, schema-run success, document contents, emitted handler keys,
late-route required/signature/descriptor distinctions, stream event retention,
preamble prefixes, active-tool lists, and inserted history. The success helper
requires actual native settlement and rejects failures and stream errors.

The full original golden comparison uses the previously reviewed
`tests/common/ecs_goldens.rs` rules described in request-shape-contract.md: only
nominal effect IDs/parents and native-only scope/program/delivery representations
are normalized. No original header, request, response, event or tool-output field
is dropped. Complete raw native logs retain poll delivery traces in immutable
content-addressed artifacts. Stable native goldens retain native scoped identity
and normalize only nondeterministic delivery grouping.

Original policies in tests/common/goldens.rs remain unchanged; the original
producer changes only shared constant/helper visibility. The batch runner checks
source AST fidelity and all 24 original fixture hashes before executing exact
original/native test IDs in separate build directories with regeneration disabled.
This is scoped provider-integrated evidence, not an all-feature or whole-programme
completion claim. Execution and independent review results are recorded with the
batch artifacts and manifest.
