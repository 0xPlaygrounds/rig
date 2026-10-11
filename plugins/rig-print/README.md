# rig-print

`--print` (`PrintPlugin`): one prompt to the primary agent, its answer on
stdout, and an exit once no agent works. It is the front of every run that is
not interactive: one with `-p`, or with stdin piped in, which answers what is
piped in.

A plugin crate of the [rig coding agent](../../crates/rig-harness), built only
on the public API of the crates it depends on. Its plugin is in the default
`plugins.toml`, and can be removed from it. It is not published yet: the agent
builds it from a rig checkout (`RIG_SOURCE`).
