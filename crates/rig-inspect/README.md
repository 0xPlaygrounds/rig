# rig-inspect

An optional plugin of the [`rig-harness`](../rig-harness) coding agent: the
`inspect` tool, with which the agent reads its own running Bevy world through
Bevy Remote, answered in process (no transport, no port) and read-only. The
agent asks it which plugins are loaded and what each added, its agents, the
state saved with the session, the warnings logged (the `Diagnostics`
resource), schedules and every reflected type, before reading rig's source.

It is not in the default `plugins.toml`, because Bevy Remote makes the agent's
build heavier and, on Linux, needs the ALSA development headers
(`libasound2-dev` on Debian and Ubuntu, `alsa-lib-devel` on Fedora,
`alsa-lib` on Arch). Enable it by uncommenting its entry in
`$RIG_HOME/plugins.toml`, or with:

```bash
rig plugin add rig_inspect::InspectPlugin --crate rig-inspect --version 0.44.0
```

where the version is your `rig`'s. Only read-only methods are sent; a plugin
that adds its own Bevy Remote method marks it with `ReadOnlyMethod` (see the
crate docs). An answer over 16 KiB is cut and kept whole in the session's
spill directory for the agent to read.
