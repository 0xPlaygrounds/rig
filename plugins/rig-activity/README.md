# rig-activity

What each agent is doing, kept for views (`ActivityPlugin`), one of the
[rig coding agent](../../crates/rig-harness)'s default plugins:

- every agent's `Activity`: its status, from a question waiting for the user
  to a plugin's work after the turn, and its open tool calls;
- the `MessageFeed` of recent deliveries;
- with the default `tui` feature, each agent's status in the terminal view's
  status line.
