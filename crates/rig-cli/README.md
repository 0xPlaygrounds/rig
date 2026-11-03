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
`~/.rig`) with every one of rig's own plugins and the extra crates of
`RIG_HOME/settings.toml`, builds it, and runs it with the plugins the home's
and the project's `settings.toml` turn on. `settings.toml` is the one file you
edit: its `[plugins]` table holds only your changes to rig's built-in list
(`enable`, `disable`, and the home's `[[plugins.extra]]` crates). Turning a
plugin on or off never rebuilds; an extra crate does.

`rig help` lists the commands, `rig plugin --help` the plugin commands, and
`rig help-agents` how a script drives rig headless. The
[agent guide](https://github.com/0xPlaygrounds/rig/blob/main/docs/agent.md)
describes using it.
