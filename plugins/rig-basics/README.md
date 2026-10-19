# rig-basics

The default plugins of the [rig coding agent](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-harness), and
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
  `EditTool`, `WriteTool`, `ShellTool`), `@path` attachments
  (`AttachPlugin`), and `HOST:PATH` paths on other machines
  (`RemoteHostsPlugin`, below);
- `approval`: one question for every effect a plugin wants approved
  (`ApprovalPlugin`);
- `preview`: what a tool call would change on disk, before it runs
  (`PreviewPlugin`);
- `checkpoint`: what each tool call changed on disk (`CheckpointPlugin`);
- `reload`: `/reload` and the `reload` tool (`ReloadPlugin`), and
  `/plugins` (`plugins::PluginsPlugin`);
- `print`: `--print` and `--json`, the fronts of a run with `-p`, piped
  stdin or JSON lines (`PrintPlugin`), and the loop of any run nobody sits
  at (`AppHeadlessExt`);
- `helpers`: the `helper` tool, which runs `rig -p --json` children
  (`HelpersPlugin`);
- `subagents`: the `task`, `message` and `wait` tools (`SubagentsPlugin`).

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
`CheckpointPlugin`, `HelpersPlugin`, `SubagentsPlugin`, `EffectLogPlugin` and
`DiagnosticsPlugin` are listed disabled by default. A plugin that needs
another adds it when it is missing, as `SessionsPlugin` adds
`SessionPickerPlugin` and `SubagentsPlugin` adds `ApprovalPlugin`.

With the default `tui` feature, each plugin also draws its part in the
terminal view; `default-features = false` leaves that out.

`coding_tools::remote::RemoteHostsPlugin` makes a path written `HOST:PATH` another
machine's for every tool: `read bench:~/app/src/lib.rs`, `edit` and `write`
there, and `shell` with `cwd: "bench:~/app"`, which runs the command there.
Other paths stay on this machine. Each host is a transport, a command that
runs its last argument (a shell line) on the host, in
`RIG_HOME/settings.toml`:

```toml
[remote.hosts]
# ssh's options go in the command.
box = ["ssh", "-o", "BatchMode=yes", "dev@box"]
# koh exec runs the line given to `--sh` with the remote shell.
bench = ["koh", "exec", "bench", "--sh"]
```

With hosts configured, the system prompt names them and the path form, and
the tools' `path` and `cwd` arguments say they take it. Exit status 255 from
a transport means the transport failed: a file operation fails saying so,
and a `shell` call ends with that status. Stopping a `shell` call stops the
local transport; whether the command on the host stops with it is the
transport's to say. A project's `.rig/settings.toml` cannot add hosts.
