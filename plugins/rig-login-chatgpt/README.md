# rig-login-chatgpt

`/login` and `/logout` for the ChatGPT plan's models (`ChatgptLoginPlugin`),
one of the [rig coding agent](../../crates/rig-harness)'s default plugins:
sign-in in the browser or with a device code, and the credential every model
call of the plan reads, refreshed when it has expired.
