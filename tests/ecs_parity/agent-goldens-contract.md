# Anthropic completion and conversation-memory goldens

The two cells in `batches/agent-goldens.json` independently execute the original
`cassette::agent` producers and native `cassette::ecs_agent_goldens` producers.
Both intentionally reuse `agent/completion_smoke.yaml`. The original completion
and memory effect logs remain immutable oracles, with separate native goldens
under `crates/rig-verify/fixtures/ecs_parity/`.

## Configuration and execution

Both use CLAUDE_SONNET_4_6, BASIC_PREAMBLE, BASIC_PROMPT, owner `golden`, an unset
default turn override (effective one turn), no per-run override, and unary
recording without stream events. Temperature and other model options remain
unset. The memory case uses a fresh InMemoryConversationMemory and the original
`golden-conversation` key. Its memory handler is registered before the model,
matching the original recorder header's handler order.

Native EcsAgent installs Recording and drives ordinary BusPlugin/AgentPlugin
systems. RuntimeHandler keeps provider and memory futures within native effect
ownership. Memory is attached with Remembers and Conversation. No legacy agent
runner, policy, builder, or recorded answer drives execution. Success requires
actual settlement and the actual append acknowledgement/error boundary; exact
golden equality additionally requires the successful original append outcome.

## Original obligations and native assertions

| Original obligation | Native evidence |
| --- | --- |
| prompt await succeeds; output is trim-nonempty | Native success wait and identical assert_nonempty_response helper |
| take_effect_log returns a recording | Recording installed before handler registration; actual recorder output must match the nonempty original log |
| Complete golden data equality including header | ecs_goldens compares the entire native log with the original under the explicit normalization below |
| Memory families are exactly load/completion/append | Identical EffectFamily vector assertion; complete golden also checks operations, conversation, messages and outcomes |
| Golden read/deserialization/serialization succeeds | Same golden path/read expectations, full JSON comparison, then separate native golden serialization comparison |
| Cassette request matching, consumption and teardown | Unchanged with_anthropic_cassette wrapper after each test closure |

The original goldens helper has explicit record/regeneration modes. Those modes
are disabled in paired verification. Initial native-golden generation runs only
these two new producers against replayed HTTP; ecs_goldens first compares with
the original oracle, then writes only the separate native golden. It never
regenerates the original effect log in this path.

## Normalization and scope

The existing ecs_goldens helper bijectively maps increasing nominal dispatch
IDs, including parent/error references. It checks native program/scope presence,
then omits native-only program tables/scopes from the comparison with legacy
logs that lack them. Full native goldens retain those identities. Native poll
delivery batches are retained verbatim in content-addressed runtime evidence
but omitted from stable golden equality; originals have no delivery metadata.
Handlers and order, bus policy, builder fingerprint, request payloads, outcomes,
events, errors and parent relationships remain compared. No response fields,
history entries or duplicate effects are discarded.

These two unary sequential scenarios establish provider and effect-log fidelity.
They do not establish delivery-sensitive policy equivalence, concurrency,
arbitrary memory-backend failure behavior, or OS-level network isolation.
Existing comparison-helper mutation controls are separate supplemental evidence;
a successful native-golden generation alone is not a paired parity verdict.
