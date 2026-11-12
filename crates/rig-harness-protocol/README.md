# rig-harness-protocol

The protocol between the `rig` launcher and the rig-harness agent it builds and
runs: the `RIG_HOME` layout, session ids, the environment the launcher passes
to the agent, and the exit code with which the agent asks for a restart. Its
`launch` module builds every process the launcher starts, so each gets only the
variables and secrets meant for it. Its one
dependency is `toml_edit`, for the settings both read.
