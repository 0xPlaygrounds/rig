---
name: writing-rig-plugins
description: Write, change or debug a plugin for the rig agent, a Bevy plugin crate built into it. Use when asked to add a slash command, tool, panel or other behaviour to rig, or when working on a crate in RIG_HOME/plugins.
---

# Writing rig plugins

Rig wrote this file when it set up this home, and never writes it again:
change it or delete it as you like.

1. Read the plugin guide first: `PLUGINS.md` of the `rig-harness` crate the
   agent is built from, which `RIG_HOME/project/Cargo.toml` names. For
   `rig-harness = { path = "<dir>" }` it is `<dir>/PLUGINS.md`; for a version
   from crates.io it is
   `https://docs.rs/rig-harness/<version>/rig_harness/plugin_guide/`. It has
   the rules every plugin follows and an example of each extension point.
2. `rig plugin new <name>` makes a plugin crate in `RIG_HOME/plugins/<name>`,
   with a working, commented slash command, and adds it to
   `[[plugins.extra]]` in `RIG_HOME/settings.toml`.
3. Copy from working plugins in `corpus/` of that rig checkout (next to its
   `crates/`), or `https://github.com/0xPlaygrounds/rig/tree/v<version>/corpus`
   for a crates.io version: one plugin per module, `src/<plugin>.rs`, with its
   tests in `src/<plugin>/tests.rs`.
4. `rig plugin check --build` builds the agent with the plugin, names
   systems that use the same data in no order, and fails on settings a
   start would refuse. `/reload` in the agent, or
   the next start, runs the new build. `rig plugin list` shows every plugin.
