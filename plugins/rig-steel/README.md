# rig-steel

Code mode for rig-ecs agents and the rig agent (`SteelPlugin`): the
`run_steel` tool, whose [Steel](https://github.com/mattwparas/steel) (Scheme)
program spawns agents, sends them requests, waits for their replies and calls
the model's own tools, in Steel's sandboxed engine. The [root
README](../../README.md#the-rig-coding-agent) shows how to enable it and an
example program.

A plugin crate of the [rig coding agent](../../crates/rig-harness), built only
on the public API of the crates it depends on. It is optional and not in the
default `plugins.toml`. It is not published yet: the agent builds it from a
rig checkout (`RIG_SOURCE`).
