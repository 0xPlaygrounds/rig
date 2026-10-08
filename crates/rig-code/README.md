# rig-code

A minimal coding agent built on [Rig](https://github.com/0xPlaygrounds/rig) and Bevy.

Agents are Bevy entities, their model and tool calls are entities tied to them, and every
call goes through one dispatch path that records it in a Rig effect log. Tools and slash
commands are registered by Bevy plugins; the built-in `read`, `edit`, `write`, `shell` and
`search` tools and the `/model`, `/effort`, `/help` and `/quit` commands use the same API a
third-party plugin does.

```sh
OPENAI_API_KEY=... cargo run -p rig-code
```

Models come from Rig's model catalog; `/model` lists those whose provider has a key in the
environment. Session logs and effect logs are written under `$RIG_HOME/data/sessions/`
(or the platform data directory).
