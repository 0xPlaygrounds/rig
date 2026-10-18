# rig-basics

The default plugins of the [rig coding agent](../../crates/rig-harness), and
the services only some plugins add. One crate, one module per group, all on
rig-harness's public API. Each plugin is its own entry in `plugins.toml`,
so each can be turned off alone.

- the root: `BasicCommandsPlugin` (`/help`, `/retry`, `/quit`);
- `prompt`: rig's fixed system prompt for an agent without one
  (`DefaultPromptPlugin`, `DEFAULT_SYSTEM_PROMPT`);
- `models`: `/model`, `/effort` and the model picker (`ModelsPlugin`), and
  the model a new session starts on (`defaults::DefaultsPlugin`);
- `login_chatgpt`: `/login` and `/logout` for the ChatGPT plan's models
  (`ChatgptLoginPlugin`);
- `session_picker`: the session and conversation lists of `/resume`,
  `/fork` and `/agents` (`SessionPickerPlugin`);
- `sessions`: `/new`, `/resume` and `/name` (`SessionsPlugin`), and `/fork`,
  `/agents` and rewind (`agents::AgentsPlugin`);
- `compaction`: summarizing a conversation near the model's window, and
  `/compact` (`CompactionPlugin`);
- `telemetry`: what model calls cost and `/usage` (`usage::UsagePlugin`),
  the effect log (`effect_log::EffectLogPlugin`), and warnings and errors
  as data (`diagnostics::DiagnosticsPlugin`);
- `activity`: what each agent is doing, for views (`ActivityPlugin`);
- `coding_tools`: the `read`, `edit`, `write` and `shell` tools (`ReadTool`,
  `EditTool`, `WriteTool`, `ShellTool`) and `@path` attachments
  (`AttachPlugin`);
- `approval`: one question for every effect a plugin wants approved
  (`ApprovalPlugin`);
- `preview`: what a tool call would change on disk, before it runs
  (`PreviewPlugin`);
- `checkpoint`: what each tool call changed on disk (`CheckpointPlugin`);
- `reload`: `/reload` and the `reload` tool (`ReloadPlugin`), and
  `/plugins` (`plugins::PluginsPlugin`);
- `print`: `--print`, the front of a run with `-p` or piped stdin
  (`PrintPlugin`);
- `subagents`: the `task`, `message` and `wait` tools (`SubagentsPlugin`);
- `headless`, no plugin: the driver of a run nobody sits at
  (`AppHeadlessExt`).

An entry names the crate and the plugin's type by the module that defines
it, the one path the type has, which `/plugins` and inspect show too:

```toml
[[plugin]]
crate = "rig-basics"
plugin = "rig_basics::compaction::CompactionPlugin"
```

`rig plugin disable rig_basics::compaction::CompactionPlugin` comments an
entry out, and `rig plugin enable <type>` puts it back; `/plugins` does the
same from inside the agent. `ApprovalPlugin`, `PreviewPlugin`,
`CheckpointPlugin`, `SubagentsPlugin`, `EffectLogPlugin` and
`DiagnosticsPlugin` are listed disabled by default. A plugin that needs
another adds it when it is missing, as `SessionsPlugin` adds
`SessionPickerPlugin` and `SubagentsPlugin` adds `ApprovalPlugin`.

With the default `tui` feature, each plugin also draws its part in the
terminal view; `default-features = false` leaves that out.
