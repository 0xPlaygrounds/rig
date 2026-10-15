# rig-telemetry

What the [rig coding agent](../../crates/rig-harness) records about itself, in
three default plugins:

- `UsagePlugin`: what model calls used and cost, per agent, turn and session,
  split per model and purpose (`AgentUsage`, `SessionUsage`), in `/usage`
  (plugins add `UsageRow`s), in each session's `/resume` line and, with the
  default `tui` feature, in the status line;
- `EffectLogPlugin`: every settled effect in the session's `effects.jsonl`;
- `DiagnosticsPlugin`: the process's warnings and errors, Bevy's included, as
  the `Diagnostics` resource.
