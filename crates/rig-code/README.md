# rig-code

A small coding agent built on [Rig](https://github.com/0xPlaygrounds/rig) and
[Bevy](https://bevy.org). Agents are entities: the conversation, model,
effort, system prompt, tool access and status are components. Model calls and
tool calls run on Bevy's task pools, go through one dispatch path and are
recorded with rig-core's effect types into the session's `effects.jsonl`.

Tools, slash commands and the terminal view are Bevy plugins. A plugin
registers its own tools and commands with `AgentAppExt::add_tool` and
`AgentAppExt::add_command`, the calls the built-in ones use.

The `rig` binary of the `rig` crate is the usual way to run it. It
generates a small Cargo project from `plugins.toml` in its config directory,
builds it, runs the agent, and restarts it when `/reload` has built a new
binary. A new binary that crashes during startup is rolled back to the last
one that worked. `RIG_HOME` moves every directory under one root, and
`RIG_CODE_SOURCE` builds rig-code from a local checkout:

```sh
cargo install rig
RIG_HOME=/path/to/rig-home RIG_CODE_SOURCE=/path/to/rig rig -j 12
```

A plugin is a Bevy `Plugin` that implements `Default`, listed in
`plugins.toml` with its crate and source (`path`, `git` or `version`):

```toml
[[plugin]]
type = "hello_plugin::HelloPlugin"
crate = "hello-plugin"
path = "/path/to/hello-plugin"
bevy_features = []
```

Without the launcher the agent needs a data directory, from `RIG_DATA_DIR`
or `$RIG_HOME/data`, and `/reload` is not available:

```sh
RIG_HOME=/path/to/rig-home cargo run -p rig-code
```

Models come from rig-core's model catalog. `/model` lists the models whose
provider's API key variable is set, and `/effort` the reasoning settings the
chosen model takes.
