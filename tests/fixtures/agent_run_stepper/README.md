# agent_run_stepper

A downstream crate that steps rig-agent's sans-IO `AgentRun` by hand with
rig-agent's default features off, to prove that the run layer needs no
futures runtime: it depends on rig-agent, and the root guard
(`tests/core/agent_run_stepper.rs`) permits that.

This is **not** the shape of `rig-bevy`. A host stepping `AgentRun` was the
retired direction (2026-09-01): the Bevy runtime is a second interpreter
over rig-core + rig-bus that never depends on rig-agent, and the fixture
for that shape is `tests/fixtures/bevy_bus_host`. This one proves one
property of rig-agent's run layer — sans-IO — and nothing about hosts.
