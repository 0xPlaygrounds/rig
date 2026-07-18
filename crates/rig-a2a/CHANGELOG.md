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
- `A2ATool` wrapping a remote A2A skill as a Rig tool, with optional
  `contextId` / `taskId` arguments for echoing server-generated ids to
  continue conversations and resume paused tasks, including identifier
  markers for direct `Message` responses
- `A2AAgentBuilderExt::a2a_tools` for binding every remote skill onto a Rig
  agent at build time, and `A2AClient::dynamic_tools` for registering the
  same tools on a shared `ToolServerHandle`
- `A2AClient::message(..)` with `.context(..)` / `.task(..)` for direct,
  optionally threaded requests
- Five-minute default HTTP timeout for agent-card discovery and protocol
  requests, configurable through `A2AClientBuilder::timeout`
- Redirect-free HTTP clients for card discovery and protocol requests, with
  safe customization through `A2AClientBuilder::http_client_builder`
