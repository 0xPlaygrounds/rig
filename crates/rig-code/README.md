# rig-code

A small coding agent built on [Rig](https://github.com/0xPlaygrounds/rig) and
[Bevy](https://bevy.org). Agents are entities: the conversation, model,
effort, system prompt, tool access and status are components. Model calls and
tool calls run on Bevy's task pools, go through one dispatch path and are
recorded with rig-core's effect types into the session's `effects.jsonl`.

Tools, slash commands and the terminal view are Bevy plugins. A plugin
registers its own tools and commands with `AgentAppExt::add_tool` and
`AgentAppExt::add_command`, the calls the built-in ones use.

The agent needs a data directory, from `RIG_DATA_DIR` or `$RIG_HOME/data`:

```sh
RIG_HOME=/path/to/rig-home cargo run -p rig-code
```

Models come from rig-core's model catalog. `/model` lists the models whose
provider's API key variable is set, and `/effort` the reasoning settings the
chosen model takes.
