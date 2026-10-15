# rig-steel

Code mode (`SteelPlugin`), an optional plugin of the
[rig coding agent](../../crates/rig-harness), not in the default
`plugins.toml`: the `run_steel` tool, whose
[Steel](https://github.com/mattwparas/steel) (Scheme) program spawns agents,
sends them requests, waits for their replies and calls the model's own tools,
in Steel's sandboxed engine. It builds on rig-ecs alone, so any rig-ecs app can
add it. The [root README](../../README.md#the-rig-coding-agent) shows how to
enable it and an example program.
