# Native handler-layer parity

Scope: all seven `corpus_layers` registrations at pinned baseline
805fb18e6135c9050ee7ca4295d96ddf08cb223f, default root features, native host.
`batches/layers.json` freezes the original sources and twelve distinct original
provider/effect fixtures. `evidence/layers-report.json` pairs exact executed IDs.

The native producer uses App, BusPlugin, AgentPlugin, native handler registration,
Grant and real provider/ToolAdapter/MemoryAdapter handlers. Shared rig-core
Intercept layers are host middleware, not the legacy agent runner or hooks.
RuntimeHandler wraps each leaf before ErasedHandler layering; this preserves the
recorder's inner-exchange position. The same original Adder implementation and
schema execute. Model is Sonnet 4.6, temperature zero, owner golden, ordinary
recorder (events off), default max turns undeclared/effective one; tool runs
explicitly override to three. Registration order and handler descriptors remain
part of the exact original golden comparison. Layer declarations derive from
actual handler descriptor order. No original fixture is regenerated.

| Original case | Native behavior and retained obligations |
| --- | --- |
| deny_tool | DenyAddLayer denies actual tool dispatch; two completion records, exact hook header and full golden. Shared tool consumer requires successful nonempty answer. |
| patch_tool_args | PatchAddArgsLayer changes the actual inner request to 40+2; exact tool arguments and hook header. |
| replace_tool_result | Real tool answers 42, recorded as 42; outer layer replaces delivery with REPLACED_RESULT. The following real provider request and golden assert the replacement reached the agent. |
| two_layers | Actual outer PatchAddArgsLayer and inner ReplaceAddResultLayer: argument 40+2, recorded result 42, ordered header, replaced result in subsequent provider request. |
| host_deny_over_host_bus | Host-owned App with actual default Policy and undeclared agent-owned bus metadata; success/nonempty, all effects terminal before App teardown, bus None, two completion records and DenyAddLayer header. Native App teardown maps the legacy joined host driver, not the same async driver API. |
| patch_beneath_hook_patch | Independent native Gate system patches actual unissued ToolCallSlot to 40+2; real host PatchAgainLayer changes the inner request to 30+12. Actual recorded args PATCHED_AGAIN_ARGS, recorded answer 42 and ordered PatchAddArgs/PatchAgainLayer header. Original AgentHook is not invoked. |
| memory_load_replaced | Real empty InMemory store, layered Load replacement, memory registered before model, original conversation and default one-turn prompt. Native system observes ordered actual Remembered utterances after Select and before Assemble and compares them to replaced_history; a required count of one prevents vacuous success. This maps HistoryIsReplaced.on_run_start's hidden assertion. Answer contains Ada; families Load/completion/Append, inner recorded Load stays empty, exact headers/full golden. Actual append acknowledgement is awaited by shared consumer. |

Shared neutral original constants and tool-record extraction helpers have only
sibling visibility changes. Original producer bodies, all assertion expressions,
layer methods and golden helpers are source-frozen. The manifest records direct
assertions plus helper and method obligations, because free-function discovery
alone does not inventory assertions inside AgentHook implementations.

The existing full-log comparator permits only its documented nominal ID mapping,
removal of native-only scope/program fields from the comparison copy (retained in
native goldens), and native delivery schedule separation (raw logs retain it).
No request, inner result, outer-result-bearing follow-up request, handler layer,
builder fingerprint, terminal item, error, or legacy header field is omitted.
Its existing negative controls remain applicable. Native policy versions declare
the two application systems; they do not assert automatic code hashing.

The history observer is scoped to this single-turn cassette, not a general
multi-turn on_run_start API. This family does not establish arbitrary concurrent
host scheduling, WASM execution, full feature/inventory coverage, network isolation,
performance, or the whole functional-superset programme. Those remain separate
required work. No paid provider call or cassette recapture was needed.
