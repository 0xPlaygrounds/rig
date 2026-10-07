# agent_run_stepper

A downstream crate that steps rig-agent's sans-IO `AgentRun` by hand with
rig-agent's default features off, to prove that the run layer needs no
futures runtime: it depends on rig-agent, and the root guard
(`tests/core/agent_run_stepper.rs`) permits that.

It proves one property of rig-agent's run layer, that it is sans-IO, and
nothing about hosts.
