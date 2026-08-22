# rig-run

The sans-IO agent-run protocol for [Rig](https://github.com/0xPlaygrounds/rig).

`AgentRun` is a steppable, serializable state machine: it owns every *decision*
the agent loop makes — turn budget, tool-call validation and recovery, history
threading, structured-output policy, usage accounting, final response — and
performs no IO. A *driver* calls `next_step()` and acts on the returned
`AgentRunStep` (`CallModel`, `CallTools`, `Done`), feeding results back with
`model_response` / `tool_results`. `rig-agent` is the futures driver; an ECS
plugin can be another. The crate depends on `rig-core` only: no async runtime,
no hooks, no tool registry.
