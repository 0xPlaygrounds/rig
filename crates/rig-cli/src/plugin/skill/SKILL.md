---
name: writing-rig-plugins
description: Write, change or debug a plugin for the rig agent, a Bevy plugin crate built into it. Use when asked to add a slash command, tool, panel or other behaviour to rig, or when working on a crate in RIG_HOME/plugins.
---

# Writing rig plugins

Rig writes this file at each build of the agent, for rig __VERSION__, and
overwrites changes to it.

1. Read the plugin guide first: `__GUIDE__`. It is the guide of the rig
   crates this agent is built from, with the rules every plugin follows and an
   example of each extension point.
2. `rig plugin new <name>` makes a plugin crate in `RIG_HOME/plugins/<name>`,
   with a working, commented slash command, and adds it to
   `[[plugins.extra]]` in `RIG_HOME/settings.toml`.
3. Copy from working plugins in `__CORPUS__`: one plugin per module,
   `src/<plugin>.rs`, with its tests in `src/<plugin>/tests.rs`.
4. `rig plugin check --build` builds the agent with the plugin, names
   systems that use the same data in no order, and fails on settings a
   start would refuse. `/reload` in the agent, or
   the next start, runs the new build. `rig plugin list` shows every plugin.
