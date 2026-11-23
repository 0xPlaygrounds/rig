# rig-steel

Code mode (`SteelPlugin`), an optional plugin of the
[rig coding agent](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-harness), off by default
(`rig plugin enable rig_steel::SteelPlugin` turns it on): the `run_steel` tool, whose
[Steel](https://github.com/mattwparas/steel) (Scheme) program calls the model's
own tools, in Steel's sandboxed engine. With Subagents on
(`rig plugin enable rig_basics::subagents::SubagentsPlugin`), the program also
spawns agents, sends them requests and waits for their replies. It builds on
rig-ecs, and on rig-harness for
its settings: `[steel]` in `RIG_HOME/settings.toml` sets `max_seconds` (60 by
default), `max_calls` (1000) and `max_output_bytes` (64 KiB). The
[agent guide](https://github.com/0xPlaygrounds/rig/blob/main/docs/agent.md)
shows how to enable it and an example program.
