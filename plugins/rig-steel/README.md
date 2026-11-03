# rig-steel

Code mode (`SteelPlugin`), an optional plugin of the
[rig coding agent](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-harness), not in the default
`plugins.toml`: the `run_steel` tool, whose
[Steel](https://github.com/mattwparas/steel) (Scheme) program spawns agents,
sends them requests, waits for their replies and calls the model's own tools,
in Steel's sandboxed engine. It builds on rig-ecs, and on rig-harness for
its settings: `[steel]` in `RIG_HOME/settings.toml` sets `max_seconds` and
`max_calls`, unset by default, and `max_output_bytes`, 64 KiB by default. The [agent guide](https://github.com/0xPlaygrounds/rig/blob/main/docs/agent.md) shows how to
enable it and an example program.
