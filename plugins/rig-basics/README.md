# rig-basics

The [rig coding agent](../../crates/rig-harness)'s default plugins, one
module per group. `plugins.toml` lists each plugin by its own type, so each
stays individually removable:

- at the root: `BasicCommandsPlugin` (`/help`, `/retry` and `/quit`) and
  `DefaultPromptPlugin`, rig's own fixed system prompt for every agent
  without one (`DEFAULT_SYSTEM_PROMPT`);
- `models`: `/model`, `/effort` and the model picker (`ModelsPlugin`), and
  the model a new session starts with (`DefaultsPlugin`);
- `login_chatgpt`: `/login` and `/logout` for the ChatGPT plan's models;
- `sessions`: `/new`, `/resume`, `/name` (`SessionsPlugin`), and `/fork`,
  `/agents` and rewind (`AgentsPlugin`);
- `compaction`: summarizing a conversation near the model's context
  window, and `/compact`;
- `telemetry`: what model calls cost and `/usage` (`UsagePlugin`), the
  session's effect log (`EffectLogPlugin`) and the process's warnings and
  errors as data (`DiagnosticsPlugin`);
- `activity`: what each agent is doing, kept for views;
- `coding_tools`: the `read`, `edit`, `write` and `shell` tools, and the
  files the user names as `@path` (`AttachPlugin`);
- `reload`: `/reload` and the `reload` tool (`ReloadPlugin`), and
  `/plugins` (`PluginsPlugin`);
- `print`: `--print`, the front of a run with `-p` or with stdin piped in;
- `subagents`: subagents, with the `task`, `message` and `wait` tools.

With the default `tui` feature, each also draws its part in the terminal
view; an app without it leaves that out with `default-features = false`.
