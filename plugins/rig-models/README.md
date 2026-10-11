# rig-models

Choosing a model: `-m`, `/model`, `/effort`, and with the default `tui`
feature their pickers and both in the terminal view's status line
(`ModelsPlugin`),
and the model and reasoning setting a new session starts with, the last ones
chosen (`DefaultsPlugin`).

A plugin crate of the [rig coding agent](../../crates/rig-harness), built only
on the public API of the crates it depends on. Its plugins are in the default
`plugins.toml`, and each can be removed from it. It is not published yet: the
agent builds it from a rig checkout (`RIG_SOURCE`).
