# rig-telemetry

What the agent records about itself: what model calls cost, in `/usage` and
the terminal view's status line with the default `tui` feature
(`UsagePlugin`), every model and tool call in the session's `effects.jsonl`
(`EffectLogPlugin`), and the process's warnings and errors, Bevy's included,
as the `Diagnostics` resource (`DiagnosticsPlugin`).

A plugin crate of the [rig coding agent](../../crates/rig-harness), built only
on the public API of the crates it depends on. Its plugins are in the default
`plugins.toml`, and each can be removed from it. It is not published yet: the
agent builds it from a rig checkout (`RIG_SOURCE`).
