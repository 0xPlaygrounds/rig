# rig-inspect

An optional plugin of the [`rig-harness`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-harness) coding agent: the
`inspect` tool, with which the agent reads its own running Bevy world through
Bevy Remote, answered in process (no transport, no port) and read-only. The
agent asks it which plugins are loaded and what each added, its agents, the
state saved with the session, the warnings logged (the `Diagnostics`
resource), schedules and every reflected type, before reading rig's source.

It is off by default. Name it in `[plugins] enable` of
`$RIG_HOME/settings.toml`, or run the command below; the next start or
`/reload` builds the agent with it:

```bash
rig plugin enable rig_inspect::InspectPlugin
```

Only read-only methods are sent; a plugin
that adds its own Bevy Remote method marks it with `ReadOnlyMethod` (see the
crate docs). An answer over 16 KiB is cut and kept whole in the session's
spill directory for the agent to read.
