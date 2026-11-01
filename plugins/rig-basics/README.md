# rig-basics

The default plugins of the [rig coding agent](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-harness), and
the services only some plugins add. One crate, one module per group, all on
rig-harness's public API. Each plugin is its own entry in `plugins.toml`,
so each can be turned off alone.

- the root: `BasicCommandsPlugin` (`/help`, `/quit`);
- `prompt`: rig's fixed system prompt for an agent without one
  (`DefaultPromptPlugin`, `DEFAULT_SYSTEM_PROMPT`);
- `agents_md`: the project's and the user's `AGENTS.md` files in the system
  prompt (`AgentsMdPlugin`);
- `skills`: the skills listed in the system prompt, and `/skill:<name>`
  (`SkillsPlugin`);
- `models`: `/model`, `/effort` and the model picker (`ModelsPlugin`), and
  the model a new session starts on (`defaults::DefaultsPlugin`);
- `login`: `/login` and `/logout` for the ChatGPT plan's models and the
  providers' API keys (`LoginPlugin`);
- `session_picker`: the session and conversation lists of `/resume`,
  `/fork` and `/agents` (`SessionPickerPlugin`, which the plugins of those
  commands add, so it is not listed on its own);
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
- `reload`: `/reload` (`ReloadPlugin`), and
  `/plugins` (`plugins::PluginsPlugin`);
- `print`: `--print` and `--json`, the fronts of a run with `-p`, piped
  stdin or JSON lines (`PrintPlugin`), and the loop of any run nobody sits
  at (`AppHeadlessExt`);
- `helpers`: the `helper` tool, which runs `rig -p --json` children
  (`HelpersPlugin`);
- `subagents`: the `task`, `message` and `wait` tools (`SubagentsPlugin`).

In TUI, print and JSON runs, `--tools=read,write` limits that run to those
tools, and `--tools=` leaves it none. Each agent keeps its own access within
them, at its first request and at execution. The limit is never saved, so a
later run without `--tools` has every tool the session kept.
The harness owns this check (`rig_harness::run`), the same in every front:
once every plugin finished and no `rig_harness::run::Starting` entity is
left, a run refuses a name no registered tool has, with exit code 2, before
any model or tool call: a print or JSON run with its lines, a TUI run as its
view closes. A plugin whose tools come in the background, such as
rig-mcp-client's MCP servers, holds the start until they are in, so their
tools can be named; until then the agent runs no turn, request or prompt.
This limit also applies to subagents, including their normally supplied
`message` and `wait` tools: include those names when children need them. A
subagent never has a tool its parent lacks: it gets `message` and `wait`
only when its parent keeps them, and asking for another tool is refused. A
tool its parent keeps but the run leaves out is given, and cannot be called
in that run.
The kernel keeps every agent within the run's tools (`rig_ecs::agent::RunTools`)
before anything is admitted, as it keeps each within its parents.

Without an explicit output limit, a turn asks for 64,000 tokens plus any
thinking budget and 1,024, within the model's output limit, or the server's
default for a model whose limits are unknown. The room left in the window caps it.

`/usage` reports provider-supplied cost when available, otherwise API list-price
estimates. These are not subscription charges or verified invoices. Missing
usage details or prices leave a partial estimate; previously saved estimates
are not retroactively recalculated.

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
and a `shell` call ends with that status. A transport must forward its stdin
and close it there when it ends: a `shell` command on the host stops when
that stdin closes, so a call stopped at its timeout, interrupted, or cut off
by a transport killed on this machine leaves nothing running there. `ssh`
does this; `koh exec` does on a stop or when its stdin closes. A project's
`.rig/settings.toml` cannot add hosts.
