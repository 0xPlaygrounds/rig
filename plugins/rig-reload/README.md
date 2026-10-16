# rig-reload

`/reload` and the `reload` tool (`ReloadPlugin`), one of the
[rig coding agent](../../crates/rig-harness)'s default plugins: the agent
rebuilds itself through the `rig` launcher with the plugins in `plugins.toml`
and restarts on the new build in the same session, once no turn runs. A failed
build keeps the running one, and its first errors go to the model as a note.
When the launcher started the agent, it also adds the system prompt's
`self_extension` section on how the agent writes and loads plugins: fixed
text that names `$RIG_HOME/plugins.toml`, the `$RIG_LAUNCHER plugin`
commands and the plugin guide, so it never changes. With the default `tui`
feature it shows the rebuild's progress in the status line. Without it, the
agent runs the build it was started with until it exits.

`/plugins` (`PluginsPlugin`, also a default plugin) lists every entry of
`plugins.toml`, enabled and disabled (commented out), with its crate, the
comment above it and how it differs from the running build. In the terminal
view it is a picker: tick and untick plugins and press Enter, and it runs
`rig plugin enable` and `disable` through the launcher and reloads at once.
Without a terminal it prints the list.
