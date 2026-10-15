# rig-print

`--print` (`PrintPlugin`), one of the [rig coding agent](../../crates/rig-harness)'s
default plugins: one prompt to the main agent, its answer on stdout, and an
exit once no agent works, unless a plugin keeps the run going. It is the front
of every run that is not interactive. See the crate docs.
