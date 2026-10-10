# rig-login-chatgpt

`/login` and `/logout` for the ChatGPT plan's models (`ChatgptLoginPlugin`):
sign-in in the browser or with a device code, and the credential every model
call of the plan reads, refreshed when it has expired.

A plugin crate of the [rig coding agent](../../crates/rig-harness), built only
on the public API of the crates it depends on. Its plugin is in the default
`plugins.toml`, and can be removed from it. It is not published yet: the agent
builds it from a rig checkout (`RIG_SOURCE`).
