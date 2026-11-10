# The rig coding agent

`rig` is a terminal coding agent. It is a Bevy app built on the
[`rig-harness`](../crates/rig-harness) crate, and every part of it but the
harness is a plugin, built from a plugin crate and turned on or off in
`settings.toml`. The `rig` command (the `rig-cli` crate) is its launcher: it
generates a small Cargo project for the agent with the plugins you turned on,
builds it, and runs it.

The agent's coding tools come from the [`rig-tools`](../crates/rig-tools)
crate, usable by any rig agent, and its kernel (agents, turns and tool calls
as Bevy entities, one recorded effect path, session logs) is the
[`rig-ecs`](../crates/rig-ecs) crate, for building your own harness.

- [Install and run](#install-and-run)
- [Update](#update)
- [What is per home and what is per project](#what-is-per-home-and-what-is-per-project)
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

`rig-cli` is not on crates.io yet. Until the next release, `cargo install rig`
installs the last release, 0.46.0, whose `rig` crate carries the launcher.
This guide describes `main`; [0.46.0's guide](https://github.com/0xPlaygrounds/rig/blob/v0.46.0/docs/agent.md)
is on its tag. To install `main`, build the launcher from a checkout
(`--force` replaces a `rig` installed by another crate):

```bash
git clone https://github.com/0xPlaygrounds/rig && cd rig
cargo install --locked --force --path crates/rig-cli
```

Then:

1. `rig` builds the agent, which takes several minutes the first time, and
   opens it. Later starts rebuild only what changed.
2. `/login` signs in with a ChatGPT plan or saves a provider's API key
   ([Models and sign-in](#models-and-sign-in)). Or export a key, such as
   `OPENAI_API_KEY`, before starting `rig`.
3. `/model` picks a model.
4. Type a prompt and press Enter. `/help` lists the commands.
5. `rig -p "fix the failing test"` answers one prompt without the terminal
   view ([Without the terminal view](#without-the-terminal-view)).

The agent builds from rig's crates on crates.io, at the launcher's version,
or from the rig checkout the launcher was installed from
([Running from a checkout](#running-from-a-checkout)). It needs Rust 1.97.1 or
newer and runs on Linux and macOS. Windows is not supported.

## Update

rig has no update command. Install the new launcher, and the next start
rebuilds the agent at its version:

- From crates.io: 0.46.0 is the latest release. Once `rig-cli` is released,
  `cargo install --force rig-cli` installs it; `--force` replaces the `rig`
  that the `rig` crate installed.
- From a checkout: `git pull`, then `cargo install --locked --force --path
  crates/rig-cli` again.

A running agent keeps its build until you restart it. The rebuild takes
several minutes, and `RIG_HOME/target` keeps the old version's build too.
Sessions resume while their format is the same. A session in another format
is refused, unchanged, with `this session was made by another version of rig;
start a new session`.

Coming from 0.46.0: it kept the plugin list in `RIG_HOME/plugins.toml` and the
saved model in `RIG_HOME/defaults.json`. `main` reads neither. Turn your
plugins off again with `rig plugin disable <type>`, add the plugin crates it
listed as `[[plugins.extra]]` tables of `settings.toml` ([Plugins](#plugins)),
and pick the model again with `/model`.

## What is per home and what is per project

`RIG_HOME` (default `~/.rig`) is the home. The project is the directory found
as [Context, compaction and retries](#context-compaction-and-retries)
describes.

| What | Home | Project |
| --- | --- | --- |
| Plugins (`[plugins]`) | `RIG_HOME/settings.toml` | no |
| Model and reasoning (`[model]`) and other settings | `RIG_HOME/settings.toml` | no: a project has no settings |
| Instructions | `RIG_HOME/AGENTS.md` | `AGENTS.md` from the working directory up to the project root |
| Skills | `RIG_HOME/skills`, `~/.agents/skills` | `.rig/skills` and `.agents/skills` from the working directory up to the project root |
| MCP servers, with the optional `rig-mcp-client` plugin | `RIG_HOME/mcp.json` | `.rig/mcp.json` |
| Sign-in, sessions, builds | `auth/`, `sessions/`, `target/` | no |

Plugins are compiled into one agent per home, so a repository you clone
cannot name crates for rig to build. To use other plugins for a project, give
it its own home. A second home has its own sign-in, sessions and build (a few
gigabytes). With [direnv](https://direnv.net), an `.envrc` in the project sets
it whenever you enter the directory:

```bash
# .envrc
export RIG_HOME="$HOME/.rig-homes/myproject"
```

Keep the home outside the repository, so the build is not in your working
tree. Run `direnv allow` once, then `rig plugin enable <type>` there changes
only that home. An exported provider key works in every home; a `/login` is
per home.

## Models and sign-in

`/model` picks a model, and `/effort` its reasoning setting. It lists the
models you can use now: those of a provider whose key is exported or saved
with `/login`, those of a sign-in, and those of a local server that needs no
key (Ollama, llama.cpp) once `OLLAMA_API_BASE_URL` or `LLAMACPP_API_BASE_URL`
names it or its key variable is set, listed last; the picker opens on the current model, which it always
lists. With none to list it says how to get one. `/model vendor/model` names
any catalog model, such as a local one at its default address; one that does
not connect, such as one with no key, is refused and the current model stays.
A new session starts with the model and reasoning you last picked with
`/model` and `/effort`, saved once that model has answered (an `/effort` is
saved with a model you picked with `/model` or saved before). They are saved
as `[model]` in `RIG_HOME/settings.toml`, `default = "vendor/model"` and
`effort = "high"` (`default`, `off`, a level, or `"1024 tokens"`), which you
may also set by hand; rig rewrites only those two keys and keeps your
comments. A project's `[model]` is not read. `-m
vendor/model` picks one on the command line for that run only, with the saved
reasoning setting when that model takes it, and none when it does not
connect (a headless run then exits 2); an editor's
choice over ACP and an automatic pick are never saved either. With nothing
saved or named, the terminal view asks you to pick one, and a headless run
or an editor's new session takes the first model that can be reached.

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
"(ChatGPT plan)": those ChatGPT says your account may use, asked once per run
when first needed (and once more if it refuses a model it listed). Until it
answers they are marked as being checked; a timeout or an
unavailable backend is retried as a model call is. If it still cannot be
asked, the plan's models are refused with the reason; the next request asks
again (after any wait ChatGPT asked for), and so does `/model` 10 s later.
Only a failed sign-in asks you to sign in again. A
successful sign-in switches the agent to the plan's latest frontier model for
that session. The credential is kept in `RIG_HOME/auth/chatgpt.json`,
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
type or press Tab (while a turn runs, Tab queues it). Text you drop
without sending (Ctrl+C, sending an earlier prompt in its place) goes into
the history, so Up brings it back. While redaction is on (below), every
prompt Up brings back, sent or not, shows its secrets and placeholders as
`[REDACTED]`, and sending it again sends `[REDACTED]`. Ctrl+P and Ctrl+N
move the same way, even with a completion list open, but Ctrl+P goes to the
history from anywhere on the first line. The usual emacs keys edit
(Ctrl+A/E/K/U/W, Alt+B/F/D); Ctrl+C clears the input, and Ctrl+D on an
empty input quits. A leading `/` completes command names and `@` completes
the paths of the shown agent's working directory (skipping what
`.gitignore` leaves out); Tab or Enter takes the selected one. A path a
space or trailing punctuation would cut is inserted quoted,
`@"my notes.md"`, which is also how to type one by hand. A command that is
unknown or refused comes back in the input with the reason, and Enter then
sends it to the model as it is.
`/help` lists the commands.

While an earlier prompt is shown, the input's top row says "An earlier
prompt · Down back to your draft" (or "to an empty input"). While nothing is typed, the top row shows
the keys and the bottom row the status line: the model, the reasoning
setting, what the agent is doing, the session's tokens (uncached input,
output, cache reads and writes), its cost (the provider's figure, else the
catalog's list price; `+` when some calls had no price) and the context in
use against the model's window. `/usage` breaks them down. With the optional
`rig_basics::budget::BudgetPlugin` (`rig plugin enable
rig_basics::budget::BudgetPlugin`), `/budget 2.00` or `/budget 500000 tokens`
bounds what the session spends, helpers included: at the limit the running
turns stop and new model calls are refused, naming `/budget`; a model call
in flight then is cancelled, and what its provider bills for it is not
counted, so the bill can pass the limit. `RIG_BUDGET_USD` and
`RIG_BUDGET_TOKENS`, which a helper gets from its parent, hold a run the same
way, and their refusal names them. `[budget] usd`
and `tokens` in `settings.toml` set it for new sessions. Dollars count only
priced calls: a subscription model's, such as a `chatgpt/*` one, has no
price, so bound those by tokens.

Typing while the agent works is fine. Enter steers the running turn (the
message goes to the model with its next call, after the tool results it waits
for) and Tab queues a follow-up that is sent when the turn would end. Both
wait under the transcript. Esc stops the turn and puts them back in the input
(a plugin may keep them for the next turn instead); what the reply had
streamed stays in the transcript, saved as stopped. A tool call cut while its
arguments streamed stays as a row that says it never ran; the model later
reads the reply's text and a note that the call never ran, never the call.
Messages still waiting when the agent exits come back when the session is
resumed.

`@path` attaches the file: its text, or, for a PNG, JPEG, GIF or WebP file,
the image when the model reads images. It resolves as the tools resolve a
path, so `@bench:~/notes.md` is the host `bench`'s file. A reference that
looks like a path (it has a `/` or an extension) with no file there, or one
that cannot be read, is warned about with the reason. A message waiting for
files on another machine holds back those sent after it, and Esc meanwhile
puts them all back in the input. Ctrl+V pastes the clipboard's image
(through `wl-paste`, `xclip` or `pngpaste`) as such a path, and a dropped
image file becomes one too.

Answers are drawn as markdown, edits as diffs, and each built-in tool's call
in its own way. By default every call is drawn whole: its arguments and
result, every diff and every note, nothing cut or folded. The optional
[`rig_fold::FoldPlugin`](../plugins/rig-fold) (off by default, `rig plugin
enable rig_fold::FoldPlugin`) instead folds each finished call to its header,
its first and last lines, and a line counting those left out. Ctrl+O (`[fold]
key`) shows the call selected with Alt+Up whole, or every call when none is
selected, and folds them again. A failed, stopped or running call stays whole,
and the model, the session log, `rig -p` and exports still get every line.
A diff is drawn from what the session keeps of it, so with redaction on (below) a secret in it shows as a
placeholder and an edit whose diff holds a private key block shows its reply
instead, as one whose diff passes 256 KiB always does. A call's row is
tinted by where it stands (waiting, running, done, failed). Code is
highlighted in the terminal's own palette, so it follows the terminal's
theme.
PageUp, PageDown and Shift+Up/Down scroll the transcript. With `NO_COLOR` set,
nothing is drawn in color. A plugin draws its tools' calls with ordinary systems in
`rig_tui::TranscriptRender`, selecting inputs through
`rig_tui::RenderBatch::tools_mut` and writing their output lines.

## Sessions

Each `rig` starts a new session. `/quit` ends it, `/new` starts a new one and
`/name` names it. A session is listed while it has a user or assistant
message. One that logged nothing is removed when it ends. One that only
changed settings, or whose only message failed and came back to the input (a
model that refuses it, say), stays on disk when it ends, but offers no `rig
--resume`, and `/resume` never lists it. `/resume` lists the earlier
sessions (name or first message, what it spent, its main agent's context, the
end of its directory and its age; → shows the whole directory) and resumes
the one picked, in its own directory; `rig --resume <id or name>` (or
`/resume <id or name>`) resumes a given one, such as after a crash.

Rig redacts nothing unless you turn redaction on with `rig plugin enable
rig_basics::redact::RedactPlugin` (then `/reload`, or the next start). With it,
API keys, tokens, private keys and the values of secret-named variables
(`GITHUB_TOKEN=…`) become placeholders, `‹SECRET_1›`, one per distinct
secret, where they come in: in the model's reply as it streams, in each
tool's output and result, and in your messages and the files they name. The
model, the session's logs, the terminal view, `--json`, ACP, subagents and a
resumed session see only the placeholders, so an image or audio link that
holds a key, such as a presigned URL, reaches the model with its
placeholder and no longer fetches. A tool call that names one runs
with the real value, so the model can still put your key in a file or a
command without seeing it. A shell command naming one whose value the shell
would read as code, such as a quote, a space or `$`, is refused. The values
are kept in memory for the run only: after a restart, `/reload` or
`/resume` a call naming an old placeholder is refused, and you give the
value again. A file that already holds text such as `‹SECRET_1›` is not
told apart: a call that writes it gets this run's first secret, or is
refused. What a session saved while redaction was off stays as it was, and
one saved while it was on keeps its placeholders after you turn it off,
where a tool call writes them as they are; resuming it then notes that no
plugin saves `rig_basics::redact::Issued`, the count of placeholders handed
out, which is kept for when you turn redaction on again.

The session is written as it happens. Each agent, subagents included, has a
log, `sessions/<id>/logs/<agent-id>.jsonl`, of its messages (each tool result
with what its call published for its row, such as an edit's diff, a read's
lines and a command's exit status, so a resumed session draws its calls as
they were drawn), settings, usage and compactions, appended to as it runs and
rewritten only through rig-memory (a plugin's redaction, say, keeps each
record's place and time). Images are stored once in `files/blobs/`, and
`files/meta.json` caches what `/resume` lists. A session in another format,
written by another version of rig, is refused, unchanged
([Update](#update)).

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

The project is the working directory or the nearest directory above it
that holds `.git` or a `.rig` directory, below your home directory: rig
started anywhere inside one reads the same `.rig/mcp.json`, `AGENTS.md`
files, skills and per-project data. A project has no settings: rig never
reads a `.rig/settings.toml`. Your home directory is not a project: started
there, rig reads only `RIG_HOME`'s MCP servers. Commands still start in the
working directory.

Two default plugins add to it when the agent starts. `AgentsMdPlugin` adds
the `AGENTS.md` files of the working directory and of each directory above
it up to the project's root, nearest last, after your own
`RIG_HOME/AGENTS.md`. `CLAUDE.md` is not read; a repository
that has only one can link `AGENTS.md` to it. `SkillsPlugin` lists your
skills: each directory with a `SKILL.md` (the Agent Skills format, a `name`
and a `description` in its frontmatter) under `.rig/skills` or
`.agents/skills` of those same directories, `RIG_HOME/skills` or
`~/.agents/skills`; each directory is searched once, and a link that leads
back to one being searched is named once. The model reads a skill's file
when a task matches its description, and `/skill:<name> [text]` sends one
with your message. A skill with `disable-model-invocation: true` is left
out of the list and only sent by hand. Both are read once, so an edit takes
effect on `/reload` or `/new`.

When the conversation comes within 16k tokens of the model's window, older
tool outputs are cleared and, if that is not enough, the older messages are
summarized by the model into a structured checkpoint (goal, progress,
decisions, next steps, files read and changed) that requests send in their
place; the newest 20k tokens stay as they are, at most a quarter of the
model's input limit (`[compaction] keep_recent` sets a number kept as set).
A turn's answer and its thinking may take the model's whole output limit,
within the room the window leaves; `[compaction] answer_tokens` sets a
smaller one. Old outputs are cleared only
after the model wrote notes on them, or when every one is a `read` whose
file reads back unchanged; notes that would not fit give way to a summary,
and when notes fail the outputs stay, a notice says why, and the older
messages are summarized next. When the conversation outgrows the window
anyway, the outputs are cleared if every one reads back unchanged, and the
older messages summarized otherwise, and the call is sent again.
`/compact` compacts now, and `/compact <focus>` says what the summary should
keep. The summarized messages stay in the transcript, under a line that shows
the summary; a resumed session starts from the messages the summary kept.

A model call that fails on a rate limit, an overloaded provider or a dropped
connection is retried up to four times. It waits as long as the provider's
`Retry-After` asks, up to 60 seconds; a longer requested wait ends retries.
Without a retry delay, it waits 2, 4, 8, then 16 seconds, and 30 seconds for
any retry after that, each shortened at random by up to a quarter. `[retry]`
in `RIG_HOME/settings.toml` changes both, with no upper bound: `max_retries`
and `max_wait_secs`. A model that calls a
tool it does not have three times in a turn fails the turn;
`[tools] max_unavailable_calls` changes that. When the turn fails, your
message goes back into the input: press Enter to send it again, or switch
models first.

## Tools

The default tools are `read`, `edit`, `write` and `shell`, each its own
plugin. The `rig-tools` library also provides `search`, `list` and
`delete`; the stock agent leaves those operations to `shell`.
A `shell` call runs until its command ends, unless the model sets
`timeout_secs` or you interrupt it; either stops the command, on a remote
host too. A job the command leaves running on this machine in the
background with its output elsewhere (`server > log 2>&1 &`) keeps running
between calls and through an interrupt of a later one, and ends when rig
exits; `setsid server > log 2>&1 &` keeps it running after rig (on macOS,
which has no `setsid` command, `perl -MPOSIX -e 'POSIX::setsid(); exec
@ARGV' server > log 2>&1 &`). On a remote
host such a job keeps running after the call. A command finds no editor
and no pager: `git commit` without `-m` fails at once, saying to pass `-m`
or `--no-edit`. The whole output
of a `shell` call that was cut is kept in `RIG_HOME/sessions/<id>/spill/`,
and the model reads it by the path the cut output names.

With `[remote.hosts]` in `RIG_HOME/settings.toml`, such as
`bench = ["koh", "exec", "bench", "--sh"]` or `box = ["ssh", "dev@box"]`, a
path written `HOST:PATH` (`bench:~/app/src/lib.rs`) is that host's for every
default coding tool: `read`, `edit` and `write` act on the file there, and `shell`
with `cwd: "box:~/app"` runs the command there, through the host's command,
while other paths stay on this machine
([rig-basics](../plugins/rig-basics/README.md)).

## Subagents

With `rig_basics::subagents::SubagentsPlugin` enabled (it is off by
default; `/plugins` or `rig plugin enable
rig_basics::subagents::SubagentsPlugin` enables it), the model can hand work
to subagents with the `task` tool. Each is a new agent in the same process,
with its own conversation and, if the call asks, another model, reasoning
setting or a subset of the tools; a call that names no model or tools gets
`[subagents] model` and `tools` from `settings.toml` when they are set, else
the parent's, and `task`'s description says which. `task` returns the
subagent's id at once, and `message` sends one of the model's own subagents a
follow-up, which it reads with its conversation kept.

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
with its model, state and what it and the agents below it spent, and shows
the one picked: its transcript, and what
you type then goes to it. A subagent can start subagents of its own, as
deep as they go. Subagents started with `peers` can also send each other
requests with `message` and `wait` for each other; a request or `wait` that
would make agents wait on each other is refused. In the effect log, a
subagent's model calls name the call that gave it its work as their parent.

## Without the terminal view

`rig -p "fix the failing test"` answers one prompt and exits: the answer goes
to stdout, failures to stderr, and the exit code is 0 when the turn ended with
an answer, 1 when it failed or a plugin such as the budget stopped its work
(answer or not), or when the agent did not start because
`RIG_HOME/settings.toml` cannot be read, is not TOML or has a `[plugins]`
table that does not read (stderr says why), 2 for a refused slash command, a
plugin's settings table that does not read, or piped input that is over
64 MiB, fails to read or is not UTF-8. Piped input is read until stdin closes; give `</dev/null` for none. A
command that would open a list to pick from, such as `/model`, `/effort`,
`/resume`, `/login` or `/fork at`, is refused there with the commands to
type instead (`/model chatgpt/gpt-6.1-sol`), as is text after a command
that takes none (`/new hello`). Text piped in follows the prompt (`git diff
| rig -p "review this"`), and piped in without `-p` it is the prompt: `rig`
runs the terminal view only on an interactive terminal.

On a terminal, `rig -p` draws the answer's markdown, word-wrapped to the
terminal's width (`COLUMNS`, else the terminal's size) with code lines kept
whole, and shows on stderr a dim line for each tool call, after the narration
before it. `NO_COLOR` leaves out its colors. Piped or redirected, stdout
gets the answer's raw markdown, unchanged.

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
hands tasks to such one-shot rig children and returns their final answers.
What a child spends, its own helpers included, is the caller's as it reports
it: `/usage` shows it live as purpose `helper`, beside what each agent and
the agents below it spent.
A call's tasks run four at a time while the rest wait their turn, or in
order as a chain, each for at most 600 s unless the call gives its own
`timeout`. `[helpers] at_once` and `timeout_secs` change both, with no upper
bound.

## Reload and builds

`/reload` rebuilds the agent when the plugins `settings.toml` turns on or a
plugin crate changed, and restarts it on the same session. The agent holds
only the plugins on, so turning one on or off rebuilds it: only the generated
crate compiles again, and a plugin's own crate the first time it is on. Typed while
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

Plugins are Bevy plugins. Everything but the harness (the session and the
launcher protocol) is one: the project context, models, sign-in, sessions,
compaction, usage, activity, each tool, `@path` attachments, `/reload`,
`/plugins`, the commands, `--print` and the terminal view, each from a plugin
crate in [`plugins/`](../plugins), most from
[`rig-basics`](../plugins/rig-basics), so any of them can be turned off or
replaced ([`crates/rig-harness`](../crates/rig-harness/README.md) lists them).
The agent is built with exactly the plugins on, which it adds as one Bevy
`PluginGroup`.

`$RIG_HOME/settings.toml` is the one file you edit. Its `[plugins]` table holds
only your changes to rig's built-in list, so a plugin rig adds to its defaults
later reaches you without a change:

```toml
[plugins]
enable  = ["rig_basics::budget::BudgetPlugin"]
disable = ["rig_basics::skills::SkillsPlugin"]

[[plugins.extra]]                 # a crate that is not rig's
crate = "hello"                   # the package name
path = "plugins/hello"            # relative to RIG_HOME; or git = "..." (branch, rev), or version = "..."
plugin = "hello::HelloPlugin"     # implements Plugin + Default
```

A mistake in this file (a name rig does not know, an `extra` path that holds
no crate, a file that is not TOML) or a `RIG_SOURCE` that is not a rig
checkout is refused, naming it, and `rig` starts no agent until it is fixed:
a build from before was built from other settings. `rig plugin disable <name>`
takes a name rig does not know out of the file. A build that fails to compile
instead starts the previous build, whose conversation is told why, except
under `rig -p`, which refuses with the compiler's errors on stderr.

Turning a plugin on or off, or changing an `extra` crate, rebuilds the agent
at the next start or `/reload`: only the generated crate compiles again, and
a plugin's own crate the first time it is on. Only your `settings.toml` says
which plugins run. A helper, `/reload`
and anything else a session starts run the same home's build.

`rig plugin list` shows every plugin, `on` or `off`, with its crate and
what it adds. `rig plugin enable <type>` and `disable <type>` change
`[plugins]`, keeping every comment. `rig plugin new hello` makes that crate in
`$RIG_HOME/plugins/hello`, outside any workspace, and adds its
`[[plugins.extra]]`, checked before `settings.toml` is written. Another
crate's plugin is a `[[plugins.extra]]` table you write. In the agent, `/plugins` is a picker of the same list: tick and
untick plugins, press Enter, and the agent rebuilds and restarts with them at
once; a failed build keeps the running one (without a terminal it prints the
list).

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

Each build of the agent writes the `writing-rig-plugins` skill to
`$RIG_HOME/skills/writing-rig-plugins/SKILL.md`, so the agent finds it
(`/skill:writing-rig-plugins`). It names `rig plugin new`, the plugin guide and
the plugin examples of the rig crates that build uses, and is rewritten at each
build.

The default [`rig-traffic`](../plugins/rig-traffic) plugin optionally records
provider HTTP exchanges in the session's `traffic.jsonl`. Its README covers
`--traffic-log`, persistent settings, transport coverage, and joining traffic
with the separately enabled effect log.

The optional [`rig-inspect`](../plugins/rig-inspect) plugin adds the `inspect`
tool, with which the agent reads its own Bevy world, read-only.

The optional [`rig-mcp-client`](../plugins/rig-mcp-client) plugin offers the
tools of MCP servers, stdio or streamable HTTP, named in `$RIG_HOME/mcp.json`
and in the project's `.rig/mcp.json`, which start with no question: `rig
plugin enable rig_mcp_client::McpPlugin`. Its README covers the file format,
`/mcp`, `--tools`, which can name an MCP server's tools in every front, and
`approval::McpApprovalPlugin`, which asks before a project's server starts.

The optional [`rig-acp`](../plugins/rig-acp) plugin adds `rig acp`, which
serves the Agent Client Protocol on stdin and stdout, so an editor such as Zed
uses rig's models, tools, sessions and slash commands: `rig plugin enable
rig_acp::AcpPlugin`. Its README has the editor's settings.

The optional [`rig-claude-cloak`](../plugins/rig-claude-cloak) plugin adds
`claude-cloak/*` models signed in with a Claude subscription instead of an API
key, on Linux and macOS: `rig plugin enable rig_claude_cloak::ClaudeCloakPlugin`.

Code mode is the optional [`rig-steel`](../plugins/rig-steel) plugin: a
`run_steel` tool whose [Steel](https://github.com/mattwparas/steel) (Scheme)
program spawns agents, messages them and calls tools, in a sandbox. Enable it
with `rig plugin enable rig_steel::SteelPlugin`, or:

```toml
[plugins]
enable = ["rig_steel::SteelPlugin"]
```

A fan-out that asks three agents at once and keeps the shortest answer:

```scheme
(define agents (map (lambda (i) (spawn-agent (string-append "solver-" (number->string i)))) (range 0 3)))
(define requests (map (lambda (agent) (send agent "How does src/lib.rs load plugins? Three sentences.")) agents))
(define answers (map reply requests))
(foldl (lambda (answer best) (if (< (string-length answer) (string-length best)) answer best))
       (car answers) (cdr answers))
```

A program runs for at most 60 seconds of execution (waits on agents and
tools not counted) and 1000 calls to agents and tools, and keeps 64 KiB of
what it prints and returns; Esc stops it sooner. `[steel]` in
`RIG_HOME/settings.toml` changes them, with no upper bound: `max_seconds`,
`max_calls` and `max_output_bytes`.

## Files

Every file lives under `RIG_HOME` (default `~/.rig`), and several `rig`
processes can share one:

| Path | What it holds |
| --- | --- |
| `names.lock` | Held while a rig checks and writes a session's name, so two never take one. |
| `settings.toml` | The one file you edit: which plugins run (`[plugins]`), the model a new session starts with (`[model]`), and settings, such as `[remote.hosts]` and `[traffic]`. A file that cannot be read or is not TOML, or a table that does not read, such as one with a misspelled key, refuses the start and names the key. A table no enabled plugin reads gets a notice at startup, an error when it looks like a misspelling of one (`[trafic]`). |
| `auth/` | `/login`'s credentials: `chatgpt.json` and the API keys in `keys/`. |
| `project/`, `target/`, `bin/` | The generated agent project, cargo's build of it (a few gigabytes), and the builds. |
| `build.log` | The last build's whole output. |
| `mcp.json` | The MCP servers of the optional `rig-mcp-client` plugin, in every project. |
| `data/` | Plugins' data across sessions and projects. |
| `projects/` | Plugins' data per project, such as the prompt history `projects/<name>-<key>/tui/history.jsonl`. |
| `sessions/<id>/` | A session: the agent logs in `logs/`, the name `/name` gave it in `name`, and `files/` (`meta.json`, the images in `blobs/`, the effect log `effects.jsonl` with `EffectLogPlugin`, opt-in `traffic.jsonl` with `TrafficPlugin`, plugins' files). |
| `sessions/<id>/spill/` | The whole output of a `shell` call that was cut. |
| `sessions/<id>/agent.log` | The session's agent log. |

Set `RIG_HOME` to put everything elsewhere, for example under a cache
directory.

## Running from a checkout

To run the agent from a rig checkout instead of crates.io, install the
launcher from it or point `RIG_SOURCE` at it:

```bash
cd rig && cargo install --path crates/rig-cli --root /some/dir
RIG_HOME=/some/dir/home RIG_SOURCE=$PWD /some/dir/bin/rig
```
