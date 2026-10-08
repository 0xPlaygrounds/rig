# rig-code

A minimal coding agent built on [Rig](https://github.com/0xPlaygrounds/rig) and Bevy.

Agents are Bevy entities, their model and tool calls are entities tied to them, and every
call goes through one dispatch path that records it in a Rig effect log. Tools and slash
commands are registered by Bevy plugins; the built-in `read`, `edit`, `write`, `shell` and
`search` tools and the `/model`, `/effort`, `/help` and `/quit` commands use the same API a
third-party plugin does.

Run it through the `rig` launcher, which generates and builds an agent project from
`plugins.toml`, then restarts it on each `/reload`:

```sh
cargo install rig
OPENAI_API_KEY=... rig
```

A plugin is a crate with a type implementing Bevy's `Plugin + Default`, listed in
`<config>/plugins.toml` (`$RIG_HOME/config` when `RIG_HOME` is set):

```toml
[[plugin]]
crate = "rig-hello"
path = "/home/me/rig-hello"          # or git = "..." (branch, tag, rev) or version = "..."
plugin = "rig_hello::HelloPlugin"
bevy_features = []                   # optional
```

`/reload` (only while no turn runs) rebuilds the agent with `rig build`, shows the compile
progress, saves the session and restarts on the new build; on a failed build the errors show
and the current build keeps running. Set `RIG_SOURCE=<rig repository>` to build rig-code from
a local checkout and `RIG_JOBS` to limit cargo's jobs. Without the launcher,
`cargo run -p rig-code` runs the built-in plugins only.

Models come from Rig's model catalog; `/model` lists those whose provider has a key in the
environment. Session logs, effect logs and the saved session (`state.json`) are written under `$RIG_HOME/data/sessions/`
(or the platform data directory).
