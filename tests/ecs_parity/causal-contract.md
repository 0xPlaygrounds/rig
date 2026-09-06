# Native causal completion parity

Scope: all three corpus_causal cells, pinned baseline
805fb18e6135c9050ee7ca4295d96ddf08cb223f, default root features/native host.
`batches/causal.json` retains six original fixtures and original sources;
`evidence/causal-report.json` pairs exact original/native execution.

The original agent's Lookup dispatches a nested completion through its scoped
sink dispatcher. Native execution registers the same neutral descriptor as an
open world handler and reuses `crates/rig-verify/tests/corpus/world_nesting.rs`'s
existing graph-producing systems. Original Lookup Serve/nest, legacy ToolServer,
AgentBuilder, runner and bus driver do not execute. No record supplies an answer:
all three model requests reach the actual Anthropic adapter and strict cassette.
The shared systems' model/handler source is included in candidate source identity.

The world nests a PendingEffect ChildOf the actual issued tool, submits its real
child completion through the model handler, and publishes a WorldOutcome built
from the child's trimmed text pieces joined with spaces. Child completion success
is required. This preserves the original Lookup's hidden model-handle/completion
expectations and dispatcher.parent()==sink.id() observation through the actual
recorded causal parent relationship, not a claim that native ECS exposes the
same sink-dispatch API. Full parent_positions is [None,None,Some(1),None].

All cases retain Sonnet4.6, temperature0, original tool description/schema,
owner golden, research preamble/prompt, default1 undeclared, run max3, model then
lookup registration order, host-owned default capacity policy, undeclared bus
metadata None, no hook declarations. completion_serial sets actual native
serial_per_handler=true; completion_concurrent and completion_streamed false.
The nested model request is unary, temperature0 and NESTED_PREAMBLE in all cases.
PolicyVersion explicitly names the reused native application systems.

Original helper obligations remain: successful answer contains Paris; all four
record families completion/tool/completion/completion; exact parent positions;
host effects finish before App teardown (maps joined original driver); full
original golden. The streamed case additionally retains events on outer records
0/3 and no events on nested record2, keeps stream events in recorder, waits for
actual EOF/fold and rejects all item errors through the shared native consumer.
Original producer changes are only sibling visibility of two neutral constants.

Existing full comparator rules apply unchanged: nominal IDs/parents,
native-only programs/scopes comparison separation and separately retained raw
native delivery scheduling. Original requests, outputs, raw response metadata,
headers, events, outcomes and causal positions remain compared. Existing negative
controls protect the shared comparator and stream consumer; this family does not
claim a new dedicated causal mutation test.

Only Completion nesting is verified here. Same-key/thread/detached/relay/never,
interruption and cancellation variants remain separate obligations. The reused
baseline installer leaks one model-key string per installation; these tests use
three bounded Apps and make no resource-lifetime or performance superiority claim.
Native intake defaults are used, not the effect-replay world's capacity1000 or
recorded leaf replayers. No production runtime changes, paid calls or cassette
recapture were needed. Full inventory/feature/WASM/network/performance/aggregate,
capability comparisons, final review/publication/CI remain unfinished.
