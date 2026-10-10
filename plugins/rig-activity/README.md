# rig-activity

What each agent is doing, kept for views: every agent's `Activity` (its
status and open tool calls) and the `MessageFeed` of recent deliveries. `ActivityPlugin` keeps them; the terminal view adds it
unless it is there, and a window or panel plugin reads them.

A plugin crate of the [rig coding agent](../../crates/rig-harness), built only
on the public API of the crates it depends on. Its plugin is in the default
`plugins.toml`, and can be removed from it. It is not published yet: the agent
builds it from a rig checkout (`RIG_SOURCE`).
