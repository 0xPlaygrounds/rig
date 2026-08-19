# rig-custom-agent

A custom observable agent companion crate for Rig.

This crate provides a `CustomAgent` and `CustomAgentBuilder` that demonstrate how to build an orchestrator with heavy telemetry and explicit tracing for maximum observability. It enables developers to integrate Custom Agent runtimes (such as Claude Code, Codex, or ACP) on top of Rig's canonical abstractions.

## Usage

```toml
[dependencies]
rig-custom-agent = "0.42.0"
```

## Example

See `examples/custom_agent_telemetry` for a fully functioning example of initializing and tracing the custom agent loop.
