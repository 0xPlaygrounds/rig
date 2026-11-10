# rig-cli

The `rig` command: the launcher of the rig coding agent, a terminal coding
agent built on [rig-harness](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-harness).

`rig-cli` is not on crates.io yet. Until the next release, `cargo install rig`
installs the last release, 0.46.0. To install `main`, build it from a checkout,
and the agent is then built from that checkout (`--force` replaces a `rig`
installed by another crate):

```bash
git clone https://github.com/0xPlaygrounds/rig && cd rig
cargo install --locked --force --path crates/rig-cli
rig
```

To update, run `rig update` (`/update` in the agent): it reinstalls rig from
where it came from, crates.io or the checkout as it is now (`git pull` first
is yours), printing the cargo command first, and builds the agent at the new
version. `/update` then resumes the session on it.

It generates a small Cargo project for the agent under `RIG_HOME` (default
`~/.rig`) with the plugins `RIG_HOME/settings.toml` turns on, builds it, and
runs it. `settings.toml` is the one file you edit: its `[plugins]` table holds
only your changes to rig's built-in list (`enable`, `disable`, and
`[[plugins.extra]]` crates). Turning a plugin on or off rebuilds the agent at
the next start: only the generated crate compiles again, and a plugin's own
crate the first time it is on.

`rig help` lists the commands, `rig plugin --help` the plugin commands, and
`rig help-agents` how a script drives rig headless. The
[agent guide](https://github.com/0xPlaygrounds/rig/blob/main/docs/agent.md)
describes using it, from the
[quick start](https://github.com/0xPlaygrounds/rig/blob/main/docs/agent.md#install-and-run) on.
