# The rig coding agent

`rig` is a terminal coding agent. It is a Bevy app built on the
[`rig-harness`](../crates/rig-harness) crate, and every part of it but the
harness is a plugin, listed in `plugins.toml` and built from a plugin crate.
The `rig` command is its launcher: it generates a small Cargo project for the
agent with those plugins, builds it, and runs it.

The agent's coding tools come from the [`rig-tools`](../crates/rig-tools)
crate, usable by any rig agent, and its kernel (agents, turns and tool calls
as Bevy entities, one recorded effect path, session logs) is the
[`rig-ecs`](../crates/rig-ecs) crate, for building your own harness.

- [Install and run](#install-and-run)
- [Models and sign-in](#models-and-sign-in)
- [The terminal view](#the-terminal-view)
- [Sessions](#sessions)
- [Context, compaction and retries](#context-compaction-and-retries)
- [Tools](#tools)
- [Subagents](#subagents)
- [Without the terminal view](#without-the-terminal-view)
- [Reload and builds](#reload-and-builds)
- [Plugins](#plugins)
- [Files](#files)
- [Running from a checkout](#running-from-a-checkout)

## Install and run

```bash
cargo install rig
export OPENAI_API_KEY=...   # or any other provider key in the model catalog
rig                         # builds the agent (later starts rebuild what changed), then opens it
```

The agent builds from rig's crates on crates.io, at the launcher's version,
or from a rig checkout ([Running from a checkout](#running-from-a-checkout)).
It needs Rust 1.97.1 or newer and runs on Linux and macOS; Windows is not
supported yet.

## Models and sign-in

`/model` picks a model (providers that need no key, such as a local Ollama,
are listed last), and `/effort` its reasoning setting. `-m vendor/model` picks
one on the command line; otherwise a session starts with the last model and
reasoning chosen, else the first model with a key.

`/login` without arguments lets you pick the ChatGPT plan or a provider that
takes an API key. `/login <provider>`, such as `/login anthropic`, asks for
the provider's key, hidden as you type or paste it. The key is saved in
`RIG_HOME/auth/keys/<provider>.json`, readable by you alone, and is used
before the provider's environment variable (`ANTHROPIC_API_KEY`). With a saved
key the provider's other variables, such as a base URL, are not read. If the
agent has no model yet, the model picker opens on the provider's models.
`/logout` lets you pick what to remove of what `/login` stored, or takes its
name (`/logout anthropic`); environment variables are never touched.

A ChatGPT subscription can pay for the model calls instead of an API key.
`/login chatgpt` opens your browser on the ChatGPT sign-in page and shows its
URL in case the browser does not open; the page returns to a one-shot
listener on `127.0.0.1` port 1455 (1457 when 1455 is taken). Without a
graphical session (no `DISPLAY` or `WAYLAND_DISPLAY` on Linux and the BSDs),
over SSH, or when both ports are taken, it shows a code to enter at
`https://auth.openai.com/codex/device` instead, as `/login chatgpt --device`
does. Print mode prints the URL or code, and `rig -p "/login chatgpt"` signs in
without the terminal view. Esc or a second `/login` cancels a sign-in that waits.

Once signed in, `/model` lists the plan's models as `chatgpt/...`, marked
"(ChatGPT plan)", and a successful sign-in switches the agent to the plan's
latest frontier model. The credential is kept in `RIG_HOME/auth/chatgpt.json`,
readable by you alone, and refreshed before a request when it has expired, so
a long session keeps working (rig processes running at once take turns
refreshing it). `/logout chatgpt` asks OpenAI to revoke it, then deletes it.
It is rig's own sign-in, separate from the Codex CLI's. A
`CHATGPT_ACCESS_TOKEN` in the environment (with `CHATGPT_ACCOUNT_ID`) is used
instead when set.

## The terminal view

The input takes several lines. Enter sends; Shift+Enter, Ctrl+J or `\` before
Enter starts a new line; Up and Down move between lines and through the
prompts sent before in this project, each once, the shown conversation's
first (kept in `RIG_HOME/projects/<name>-<key>/tui/history.jsonl`). Up on
the first line of your own text goes to its start, and only Up again from
there shows an earlier prompt; Down past the newest brings your text back.
While an earlier prompt is shown unchanged, Up and Down keep walking the
prompts, slash commands included, and no completion list opens until you
type or press Tab (while a turn runs, Tab queues it). Text you drop without sending (Ctrl+C, sending an earlier
prompt in its place) goes into the history, secrets redacted, so Up brings
it back. Ctrl+P and Ctrl+N move the same way, even with a completion list
open, but Ctrl+P goes to the history from anywhere on the first line. The
usual emacs keys edit (Ctrl+A/E/K/U/W, Alt+B/F/D); Ctrl+C clears the input,
and Ctrl+D on an empty input quits. A
leading `/` completes command names and `@` completes the project's paths
(skipping what `.gitignore` leaves out); Tab or Enter takes the selected one.
A command that is unknown or refused comes back in the input with the reason,
and Enter then sends it to the model as it is. `/help` lists the commands.

While nothing is typed, the input's top row shows the keys and its bottom row
the status line: the model, the reasoning setting, what the agent is doing,
the session's tokens (uncached input, output, cache reads and writes), its
cost (the provider's figure, else the catalog's list price; `+` when some
calls had no price) and the context in use against the model's window.
`/usage` breaks them down.

Typing while the agent works is fine. Enter steers the running turn (the
message goes to the model with its next call, after the tool results it waits
for) and Tab queues a follow-up that is sent when the turn would end. Both
wait under the transcript. Esc stops the turn and puts them back in the input
(a plugin may keep them for the next turn instead); what the reply had
streamed stays in the transcript, saved as stopped, and is not sent to the
model again. Messages still waiting when the agent exits come back when the
session is resumed.

`@path` to a PNG, JPEG, GIF or WebP file attaches the image when the model
reads images. Ctrl+V pastes the clipboard's image (through `wl-paste`, `xclip`
or `pngpaste`) as such a path, and a dropped image file becomes one too.

Answers are drawn as markdown, edits as diffs, and each built-in tool's call
in its own way, always whole: every call's arguments and result, every diff
and every note, nothing cut or folded. A call's row is tinted by where it
stands (waiting, running, done, failed). Code is highlighted in the terminal's
own palette, so it follows the terminal's theme. PageUp, PageDown and
Shift+Up/Down scroll the transcript. With `NO_COLOR` set, nothing is drawn in
color. A plugin draws its tools' calls with ordinary systems in
`rig_tui::TranscriptRender`, selecting inputs through
`rig_tui::RenderBatch::tools_mut` and writing their output lines.

## Sessions

Each `rig` starts a new session. `/quit` ends it, `/new` starts a new one and
`/name` names it. A session is kept once it has a user or assistant message; one
that only changed settings is removed when it ends, and `/resume` never
lists it. `/resume` lists the earlier sessions (name or first message, what
it spent, its main agent's context, the end of its directory and its age; →
shows the whole directory) and resumes the one picked, in its own
directory; `rig --resume <id or name>` (or `/resume <id or name>`) resumes a
given one, such as after a crash.

The session is written as it happens. Each agent, subagents included, has a
log, `sessions/<id>/logs/<agent-id>.jsonl`, of its messages, settings, usage
and compactions, appended to as it runs and rewritten only through
rig-memory (a plugin's redaction, say, keeps each record's place and time).
Images are stored once in `files/blobs/`, and `files/meta.json` caches what
`/resume` lists. An older session format is resumed and written on in its
own; one from before `rig-session/2` is refused, unchanged.

Quitting mid-turn (Ctrl+D, `/quit`, closing the terminal, or a signal to a
headless run) stops the turn as Esc does: the reply so far is kept, each
unfinished tool call says it was stopped, and the model is told the request
was stopped. Resuming shows it so and runs nothing until you send a message;
a message you had queued comes back in the input. A plugin may carry a turn
across the quit instead, such as one waiting on a batch API.

A tool call that may change something starts only once the reply that asked
for it, and a mark that it started, are on disk. After a crash, resuming the
session reads the logs back and stops the turn the crash cut short in the
same way: a tool call that had finished keeps its result, one that had
started is answered as cut short by the restart, and nothing runs again. The
text the model was streaming is lost. A plugin may carry such a turn on
instead, such as by asking you first.

## Context, compaction and retries

The system prompt is the same on every request of a session, so the
provider's prompt cache keeps the whole conversation behind it: no date or
tool rules in it, and each tool says in its own description when to pick it.
Models whose catalog entry lists a prompt cache get it, with the agent's id
as the cache key where the provider takes one.

Two default plugins add to it when the agent starts. `AgentsMdPlugin` adds
the `AGENTS.md` files of the working directory and of each directory above
it up to the repository root (the nearest holding `.git`), nearest last,
after your own `RIG_HOME/AGENTS.md`. `CLAUDE.md` is not read; a repository
that has only one can link `AGENTS.md` to it. `SkillsPlugin` lists your
skills: each directory with a `SKILL.md` (the Agent Skills format, a `name`
and a `description` in its frontmatter) under `.rig/skills` or
`.agents/skills` of those same directories, `RIG_HOME/skills` or
`~/.agents/skills`. The model reads a skill's file when a task matches its
description, and `/skill:<name> [text]` sends one with your message. A skill
with `disable-model-invocation: true` is left out of the list and only sent
by hand. Both are read once, so an edit takes effect on `/reload` or `/new`.

When the conversation comes within 16k tokens of the model's window, older
tool outputs are cleared and, if that is not enough, the older messages are
summarized by the model into a structured checkpoint (goal, progress,
decisions, next steps, files read and changed) that requests send in their
place; the newest 20k tokens stay as they are. When the conversation outgrows
the window anyway, older tool outputs are cleared and the call is sent again.
`/compact` compacts now, and `/compact <focus>` says what the summary should
keep. The summarized messages stay in the transcript, under a line that shows
the summary; a resumed session starts from the messages the summary kept.

A model call that fails on a rate limit, an overloaded provider or a dropped
connection is retried up to four times, waiting as long as the provider's
`Retry-After` asks (or 2, 4, 8, 16 seconds). If it still fails, your message
goes back into the input: press Enter to send it again, or switch models
first.

## Tools

The default tools are `read`, `edit`, `write` and `shell`, each its own
plugin. A `shell` call runs until its command ends, unless the model sets
`timeout_secs` or you interrupt it; either stops the command, on a remote
host too. The whole output
of a `shell` call that was cut is kept in `RIG_HOME/spill/`, and the model
reads it by the path the cut output names.

With `rig_basics::checkpoint::CheckpointPlugin` enabled (it is listed
disabled by default), every `shell` call, and any other call that may write
anywhere, reports the files it changed, from snapshots of the project before
and after it kept in the session's scratch directory, never in the
repository's `.git`. `[checkpoints] enabled = false` in
`RIG_HOME/settings.toml` turns that off.

With `[remote.hosts]` in `RIG_HOME/settings.toml`, such as
`bench = ["koh", "exec", "bench", "--sh"]` or `box = ["ssh", "dev@box"]`, a
path written `HOST:PATH` (`bench:~/app/src/lib.rs`) is that host's for every
tool: `read`, `edit`, `write` and `search` act on the file there, and `shell`
with `cwd: "box:~/app"` runs the command there, through the host's command,
while other paths stay on this machine
([rig-basics](../plugins/rig-basics/README.md)).

## Subagents

With `rig_basics::subagents::SubagentsPlugin` enabled (it is listed disabled
by default; `/plugins` or `rig plugin enable
rig_basics::subagents::SubagentsPlugin` enables it), the model can hand work
to subagents with the `task` tool. Each is a new agent in the same process,
with its own conversation and, if the call asks, another model, reasoning
setting or a subset of the tools. `task` returns the subagent's id at once,
and `message` sends one of the model's own subagents a follow-up, which it
reads with its conversation kept.

Each `task` or `message` call is a request, and exactly one report for it
(done, failed or stopped) arrives later as a message to the agent that asked,
which starts a turn of an idle agent or follows the running one (a stopped
subagent's report is a note that starts no turn); the reports of one reply's
`task` calls reach it together. Nothing waits for them, so you can keep
talking to the main agent, or steer it, meanwhile. An agent that needs a
message before it goes on calls `wait`, which keeps its turn open until that
agent sends it one, and a subagent whose turn ends while a request it made is
open is waiting, not done: it reports the answer it gives once that request
is answered.

Esc stops the shown agent's turn and the turns of every subagent below it,
even when the shown agent is idle: stopping follows who started whom.
`/agents` (`rig_basics::sessions::agents::AgentsPlugin`) lists every agent
with its model and state, and shows the one picked: its transcript, and what
you type then goes to it. A subagent can start subagents of its own, one
level deep. Subagents started with `peers` can also send each other requests
with `message` and `wait` for each other; a request or `wait` that would make
agents wait on each other is refused. In the effect log, a subagent's model
calls name the call that gave it its work as their parent.

## Without the terminal view

`rig -p "fix the failing test"` answers one prompt and exits: the answer goes
to stdout, failures to stderr, and the exit code is 0 when the turn ended with
an answer, 2 for a refused slash command. Text piped in follows the prompt
(`git diff | rig -p "review this"`), and piped in without `-p` it is the
prompt: `rig` runs the terminal view only on an interactive terminal.

`-r <id or name>` resumes a given session instead of starting a new one; it
goes wherever an option goes, but not among a command's words or after `--`,
and `rig -r <id or name> <command>` runs a plugin's command in that session.
`rig -r <id or name> -p` with no prompt goes on only with a turn a plugin
carries on (such as a batch that waits) and prints its answer; a turn a quit,
a signal or a crash cut short is stopped, not finished, so with nothing carried
on it exits 2. A plugin may keep such a run going, sending the agent work over
time (`KeepRunning`); each answer is then printed as it comes.

To drive rig from a script or another agent, over pipes and nothing else,
`rig -p --json "task"` writes the run as JSON lines, ending with an `exit`
line that carries the final answer, and `rig --json` stays up, one request
per line on stdin and one event per line on stdout, until stdin closes.
`rig help-agents` is the whole protocol, with a Python driver. The optional
`rig_basics::helpers::HelpersPlugin` gives the model the `helper` tool, which
hands tasks to such one-shot rig children (one, up to eight side by side, or a
chain) and returns their final answers.

## Reload and builds

`/reload` rebuilds the agent and restarts it on the same session. Typed while
a turn runs, it waits until no turn runs (`/reload cancel` drops it), and a
turn started during the build delays the restart until it ends. It shows
cargo's latest line, keeps the current build running if the new one does not
compile (its error notice names the build log), and rolls back to it if the
new one crashes during startup.

A `-p` run in which a plugin asks for a reload waits for it and carries on on
the new build.

Every build, the one before each start and `/reload`'s, writes its whole
output (the launcher's and cargo's) to `RIG_HOME/build.log`, and a failure
message names its reason and first compiler error. The agent sees build
failures too: the reason, the first errors and the path of `build.log` go into
its conversation as a message from the `build` plugin, which starts no turn
when the agent is idle and is saved with the session. Ask it why the build
failed and it can read the log and fix the cause; `/reload` then tries again.

`rig build` regenerates and builds the agent without starting it.
`CARGO_BUILD_JOBS` sets cargo's `-j` for these builds, as for any cargo build.

## Plugins

Plugins are Bevy plugins, listed in `$RIG_HOME/plugins.toml`. Everything but
the harness (the session and the launcher protocol) is an entry in that list:
the project context, models, sign-in, sessions, compaction, usage, activity,
each tool, `@path` attachments, `/reload`, `/plugins`, the commands,
`--print` and the terminal view, each from a plugin crate in
[`plugins/`](../plugins), most from [`rig-basics`](../plugins/rig-basics), so
any of them can be removed or replaced
([`crates/rig-harness`](../crates/rig-harness/README.md) lists them):

```toml
[[plugin]]
crate = "rig-basics"              # one of rig's own plugin crates: no source
plugin = "rig_basics::coding_tools::ReadTool"

[[plugin]]
crate = "rig-tui"
plugin = "rig_tui::TuiPlugin"

[[plugin]]
crate = "hello"                   # the package name
path = "plugins/hello"            # relative to plugins.toml; or git = "..." (branch, rev), or version = "..."
plugin = "hello::HelloPlugin"     # implements Plugin + Default
```

`rig plugin new hello` makes that crate in `$RIG_HOME/plugins/hello`, outside
any workspace, and adds its entry; `rig plugin add <type> --path <dir>` (or
`--git`, `--version`) adds an entry for an existing crate; `rig plugin add
<type>` with no source adds one for a plugin rig ships, at the launcher's
version; and `rig plugin remove <type>` takes one out (its crate stays), each
checked before `plugins.toml` is written. A commented-out entry is a disabled
plugin: the template lists every plugin rig ships, the ones not on by default
commented out. `rig plugin list` shows each entry, `on` or `off`, with its
crate and the comment above it, and `rig plugin enable <type>` and `disable
<type>` uncomment and comment out an entry, keeping every other line. In the
agent, `/plugins` is a picker of the same list: tick and untick plugins, press
Enter, and the agent rebuilds and restarts at once (without a terminal it
prints the list).

`rig plugin check` checks the list. With `--build` it also builds the agent
with its plugins in `$RIG_HOME/target` without staging it, and names the
plugins whose systems in one agent-loop stage use the same data with no order
between them; a debug build of the agent warns of those at start.

A plugin crate depends on `rig-harness`, and on another plugin crate whose
types it uses (such as `rig-tui` for a terminal panel), and registers tools,
slash commands, transcript render systems, terminal panels or a window the way
the built-in ones do, never by editing rig's crates.
[`crates/rig-harness/PLUGINS.md`](../crates/rig-harness/PLUGINS.md) is the
plugin guide, with a copy-ready example of each kind, its names from
`rig_harness::prelude` and the plugin crates, and the `src/lib.rs` that `rig
plugin new` writes is a working, commented slash command.

The default [`rig-traffic`](../plugins/rig-traffic) plugin optionally records
provider HTTP exchanges in the session's `traffic.jsonl`. Its README covers
`--traffic-log`, persistent settings, transport coverage, and joining traffic
with the separately enabled effect log.

The optional [`rig-inspect`](../plugins/rig-inspect) plugin adds the `inspect`
tool, with which the agent reads its own Bevy world, read-only.

The optional [`rig-mcp-client`](../plugins/rig-mcp-client) plugin offers the
tools of MCP servers, stdio or streamable HTTP, named in `$RIG_HOME/mcp.json`
and, once you approve each, in the project's `.rig/mcp.json`: `rig plugin
enable rig_mcp_client::McpPlugin`. Its README covers the file format, approval,
`/mcp` and headless `--tools`.

The optional [`rig-claude-cloak`](../plugins/rig-claude-cloak) plugin adds
`claude-cloak/*` models signed in with a Claude subscription instead of an API
key, on Linux and macOS: `rig plugin add rig_claude_cloak::ClaudeCloakPlugin`.

Code mode is the optional [`rig-steel`](../plugins/rig-steel) plugin: a
`run_steel` tool whose [Steel](https://github.com/mattwparas/steel) (Scheme)
program spawns agents, messages them and calls tools, in a sandbox. Enable it
with:

```toml
[[plugin]]
crate = "rig-steel"
plugin = "rig_steel::SteelPlugin"
```

A fan-out that asks three agents at once and keeps the shortest answer:

```scheme
(define agents (map (lambda (i) (spawn-agent (string-append "solver-" (number->string i)))) (range 0 3)))
(define requests (map (lambda (agent) (send agent "How does src/lib.rs load plugins? Three sentences.")) agents))
(define answers (map reply requests))
(foldl (lambda (answer best) (if (< (string-length answer) (string-length best)) answer best))
       (car answers) (cdr answers))
```

## Files

Every file lives under `RIG_HOME` (default `~/.rig`), and several `rig`
processes can share one:

| Path | What it holds |
| --- | --- |
| `plugins.toml` | The plugin list. |
| `settings.toml` | Settings, such as `[shell]`, `[checkpoints]`, `[remote.hosts]` and `[traffic]`. |
| `auth/` | `/login`'s credentials: `chatgpt.json` and the API keys in `keys/`. |
| `project/`, `target/`, `bin/` | The generated agent project, cargo's build of it (a few gigabytes), and the builds. |
| `build.log` | The last build's whole output. |
| `mcp.json` | The MCP servers of the optional `rig-mcp-client` plugin, in every project. |
| `defaults.json` | The last chosen model and reasoning, which a new session starts with. |
| `data/` | Plugins' data across sessions and projects. |
| `projects/` | Plugins' data per project, such as the prompt history `projects/<name>-<key>/tui/history.jsonl`. |
| `sessions/<id>/` | A session: the agent logs in `logs/`, the name `/name` gave it in `name`, and `files/` (`meta.json`, the images in `blobs/`, the effect log `effects.jsonl` with `EffectLogPlugin`, opt-in `traffic.jsonl` with `TrafficPlugin`, plugins' files). |
| `spill/` | The whole output of a `shell` call that was cut. |
| `agent.log` | The agent's log. |

Set `RIG_HOME` to put everything elsewhere, for example under a cache
directory.

## Running from a checkout

To run the agent from a rig checkout instead of crates.io, install the
launcher from it or point `RIG_SOURCE` at it:

```bash
cd rig && cargo install --path . --root /some/dir
RIG_HOME=/some/dir/home RIG_SOURCE=$PWD /some/dir/bin/rig
```
