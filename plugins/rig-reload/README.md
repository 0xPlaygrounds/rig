# rig-reload

`/reload` and the `reload` tool (`ReloadPlugin`): the agent rebuilds itself
through the `rig` launcher with the plugins in `plugins.toml` and restarts on
the new build in the same session, once no turn runs. A failed build keeps the
running one, and its first errors go to the model as a note. It also adds the
system prompt's section on what the agent is (its plugins, commands and
tools) and how it writes and loads plugins. With the `tui` feature (on by
default), the terminal view's status line shows how a rebuild goes.

A plugin crate of the [rig coding agent](../../crates/rig-harness), built only
on the public API of the crates it depends on. Its plugin is in the default
`plugins.toml`, and can be removed from it: the agent then runs the build it
was started with until it exits. It is not published yet: the agent builds it
from a rig checkout (`RIG_SOURCE`).
