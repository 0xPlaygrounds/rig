# rig-harness-protocol

The protocol between the `rig` launcher and the rig-harness agent it builds and
runs: the `RIG_HOME` layout, session ids, the environment the launcher passes
to the agent, and the exit code with which the agent asks for a restart. It
uses only std, so it adds nothing to the launcher's dependencies.
