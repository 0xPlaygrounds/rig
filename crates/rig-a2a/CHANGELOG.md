# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial release of rig-a2a
- `A2AClient` for consuming remote A2A agents over HTTP (JSON-RPC + REST),
  with well-known `AgentCard` discovery, a 1 MiB card-size cap, and
  same-origin validation of the selected interface
- Spec-ordered interface selection and propagation of the selected interface's
  optional tenant on direct and tool requests
- `A2AClient::tool` projecting the whole remote agent as one Rig `DynamicTool`
  taking a single `prompt`, with the card's skills rendered into the tool
  description. A2A carries no skill selector, so declared skills are
  documentation rather than separate endpoints
- `A2AAgentBuilderExt::a2a_tool` for binding a remote agent onto a Rig agent at
  build time, and `A2AClientBuilder::tool_name` for disambiguating remotes whose
  card names collide
- `A2AModel`, a `CompletionModel` implementation backing a Rig `Agent` with a
  remote A2A agent — `A2AClient::model`, `A2AClient::agent`, and their
  `_for_conversation` variants — supporting both completion and streaming.
  Request fields A2A cannot express (`tools`, `output_schema`, a demanding
  `tool_choice`) fail loudly instead of being silently dropped
- `ConversationId` and `conversation_context` for host-side conversation
  threading: server-issued `contextId` and `taskId` values are recorded per
  conversation and re-attached automatically, so identifiers never appear in a
  tool's schema or output. Task ids are retained only for tasks paused in
  `input-required`; conversations are bounded and evicted least-recently-used
- `A2AThreadInfo`, published on every tool call through
  `ToolContext::insert_result`, exposing the remote identifiers to hooks without
  showing them to the model
- Typed tool errors for remote task states the caller cannot act on:
  `failed` → `Provider`, `rejected` → refusal, `canceled` → `Cancelled`,
  `auth-required` → `PermissionDenied`, each carrying the remote's status text
- `A2AClient::message(..)` with `.context(..)` / `.task(..)` for direct requests
  with explicit, caller-supplied threading
- An 8 KiB cap on the tool description rendered from a remote-controlled card
- Five-minute default HTTP timeout for agent-card discovery and protocol
  requests, configurable through `A2AClientBuilder::timeout`
- Redirect-free HTTP clients for card discovery and protocol requests, with
  safe customization through `A2AClientBuilder::http_client_builder`
