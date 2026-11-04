# rig-cli

The `rig` command: the launcher of the rig coding agent, a terminal coding
agent built on [rig-harness](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-harness).

```bash
cargo install rig-cli
rig
```

From a rig checkout, `cargo install --path crates/rig-cli` installs it, and
the agent is then built from that checkout.

It generates a small Cargo project for the agent under `RIG_HOME` (default
`~/.rig`) with the plugins `RIG_HOME/settings.toml` turns on, builds it, and
runs it. `settings.toml` is the one file you edit: its `[plugins]` table holds
only your changes to rig's built-in list (`enable`, `disable`, and
`[[plugins.extra]]` crates). Turning a plugin on or off rebuilds the agent at
the next start: only the generated crate compiles again, and a plugin's own
crate the first time it is on. A project's
`.rig/settings.toml` never changes which plugins run.

`rig help` lists the commands, `rig plugin --help` the plugin commands, and
`rig help-agents` how a script drives rig headless. The
[agent guide](https://github.com/0xPlaygrounds/rig/blob/main/docs/agent.md)
describes using it.
