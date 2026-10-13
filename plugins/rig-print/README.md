# rig-print

`--print` (`PrintPlugin`): one prompt to the primary agent, its answer on
stdout, and an exit once no agent works, unless a plugin keeps the run going
(rig-harness's `KeepRunning`) and sends more work, each answer printed as it
comes. It is the front of every run that is
not interactive: one with `-p`, or with stdin piped in, which answers what is
piped in. Nobody answers its questions unless `--answer=ID[,ID…]` gives the
choices to answer with; other plugins print after the answer with
`PrintSection`s.

A plugin crate of the [rig coding agent](../../crates/rig-harness), built only
on the public API of the crates it depends on. Its plugin is in the default
`plugins.toml`, and can be removed from it. It is not published yet: the agent
builds it from a rig checkout (`RIG_SOURCE`).
