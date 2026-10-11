Writing a plugin for the rig agent: the rules every plugin follows, then
one short, copy-ready example per extension point. Every name they use
comes from `use rig_harness::prelude::*;`, except a plugin crate's own
types, such as the terminal view's panels from `rig_tui`.

- [Making one](#making-one) and [finding names](#finding-names)
- [How a plugin steps in: effects and stages](#how-a-plugin-steps-in-effects-and-stages): the rules, and [ordering against other plugins](#ordering-against-other-plugins)
- [A tool](#a-tool) and [gating and reviewing its calls](#gating-and-reviewing-tool-calls), with [why a held call holds back the later ones](#a-held-call-holds-back-the-later-calls-of-its-reply)
- [Asking the user: questions and pickers](#asking-the-user-questions-and-pickers)
- [A slash command](#a-slash-command) and [what the user types](#what-the-user-types)
- [Model requests](#model-requests), [what a model streams](#what-a-model-streams) and [how a turn ends](#how-a-turn-ends)
- [Effects of a plugin's own](#effects-of-a-plugins-own) and [subagents](#subagents), with [extending `task`](#extending-task)
- [The effect log](#the-effect-log-reading-redacting-replaying): reading, redacting, replaying
- [Settings and command-line options](#settings-and-command-line-options)
- [Saved state](#saved-state-counting-every-tool-call), [time](#time), [the process: project, files and child processes](#the-process-project-files-and-child-processes), [sessions and exiting](#sessions-and-exiting)
- [Adding to another plugin's lists](#adding-to-another-plugins-lists): `/model`, `/resume`, `/context`, `/agents`, `/usage`, `--print`; spending and budgets
- [Turning off or replacing what another plugin added](#turning-off-or-replacing-what-another-plugin-added)
- [Testing a plugin](#testing-a-plugin)
- [A front](#a-front), [the status line](#the-status-line), [a terminal panel](#what-agents-do-and-say-in-a-terminal-panel), [keys and the input](#keys-and-the-input-in-the-terminal-view), [a window](#a-window)

A plugin is a Bevy plugin: a type implementing `Plugin + Default` in a
crate that depends on `rig-harness`. The agent is a Bevy app, and its
tools, commands, subagents, compaction, sessions and terminal view are
plugins of the same kind, crates in the rig repository's `plugins/`
folder, listed in `RIG_HOME/plugins.toml` beside any other and built only
on the same public API. A plugin never edits rig-harness, rig-ecs or
rig-tools. One that needs another adds it unless it is there, with Bevy's
`app.is_plugin_added::<P>()`.

# Making one

```sh
rig plugin new hello    # RIG_HOME/plugins/hello, listed in plugins.toml
rig plugin add viz::VizPlugin --path ~/viz   # an entry for an existing crate
rig plugin remove viz::VizPlugin             # take an entry out; its crate stays
rig plugin check --build  # check plugins.toml and build the agent, as /reload would
```

`rig plugin new` writes a crate whose `src/lib.rs` is a working, commented
slash command, and adds this entry:

```toml
[[plugin]]
crate = "hello"           # the package name
path = "plugins/hello"    # or git = "…" (branch, rev), or version = "…"
plugin = "hello::HelloPlugin"
```

Each plugin crate is its own workspace in `RIG_HOME/plugins/<name>`. It
depends on `rig-harness` by version, on another rig plugin crate it uses
the same way (such as `rig-tui` for a terminal panel), and on `bevy_ecs`
and `bevy_reflect` (or `bevy`) at the agent's exact version, because
Bevy's derives expand to the Bevy crate the plugin's own `Cargo.toml`
names. The agent is built from a rig checkout (`RIG_SOURCE`): a
`[patch.crates-io]` table points the rig crates at it, so every plugin uses
the agent's own crates.

`/reload` in the agent, or the agent's `reload` tool (both from the
`rig-reload` plugin), rebuilds and restarts in the same session once no
turn runs. A build that fails leaves the running one, and one that crashes
at startup is rolled back.

# Finding names

The prelude holds Bevy's app, ECS, reflection and time preludes, the agent
runtime's components, stages and registries (`rig_harness::rig_ecs`), and
the harness's services. rig-core is `rig_harness::rig_core`, and each
default plugin's types are in its crate: the terminal view's panels, tool
renderers, status line, key bindings, input and pickers in `rig_tui`
(ratatui is `rig_tui::ratatui`, crossterm `rig_tui::crossterm`), what agents do in `rig_activity`, `/reload`'s state in
`rig_reload`, and so on for each crate `plugins.toml` names.

What every plugin shares lives in rig-harness, never in another plugin:
how the agent was started (`Invoked`, with the plugins' options), the
user's settings (`Settings`, `add_setting`), the fronts (`add_front`), the
questions any front answers (`Ask`, `Answerer`), why the app exits
(`ExitReason`), the project and `RIG_HOME` (`ProjectRoot`, `RigHome`),
how child processes start (`Processes`), file watches (`FileWatch`),
wall-clock wakes (`WakeAt`) and rig's tokio runtime (`TokioRuntime`).

Every one of these types is reflected, so the running agent can list
them with the `inspect` tool of the optional `rig-inspect` plugin (enable it
by uncommenting its entry in `plugins.toml`): ask it before reading rig's
source. `world.list_components` and `world.list_resources` name what exists,
`registry.schema` gives a type's fields (with `type_limit: {with:
["Saved"]}`, the state saved with the session), `world.query` with
`ProvidedBy` shows what each plugin added, and the `Diagnostics` resource
holds the warnings logged. A plugin's own Bevy Remote method is sent by
`inspect` once its system's entity has `rig_inspect::ReadOnlyMethod`.

# How a plugin steps in: effects and stages

Every step of an agent's work is an entity, an **effect**, whose
components are its request and, once answered, its `Outcome`: a model call
(`ModelRequest`), a tool call (`ToolRequest`), a message to an agent
(`Incoming`), a question for the user (`Ask`), or a plugin's own
(`EffectRequest`). Each works `ForAgent` an agent (or for the app, without
one) and is recorded once it settles (`Recorded`, the session's effect
log). Each frame, the `AgentLoop` system in `Update` runs the `AgentPass`
schedule, one pass through the stages in this order, again and again until
nothing moves, so an effect goes as far as it can in the frame:

| Stage | Who | What happens there |
|---|---|---|
| `Arrive` | kernel | finished work becomes an `Outcome`; a failed model call gets a `RetryVerdict` |
| `Review` | plugins | outcomes not settled yet (`Unreviewed`): read, edit in place, hold, or take a failure over; a turn's proposed end |
| `Commit` | kernel | settles what is answered and unheld (`Settled`), appends results, carries turns on or ends them |
| `Admit` | kernel | `Admitted` on every new effect; builds turn requests |
| `Prepare` | plugins | what was admitted this pass (`Added<Admitted>`): edit the request, once |
| `Gate` | plugins | as it opens, the kernel gives each tool call its `Footprint` and marks `Waiting` a call behind a conflicting one; then what is about to start (`Undecided`): hold it or decide it (`Decision`) |
| `Start` | kernel | applies the decisions as `Outcome`s, then starts what is admitted, unheld and unanswered |

**A hook is an ordinary system in a stage** that queries stage data:
`app.add_systems(AgentPass, my_gate.in_set(Stage::Gate))`. There are no
hook events. Order between plugins is Bevy's `.before` and `.after`,
against the public sets other plugins put their stage systems in
([ordering](#ordering-against-other-plugins)). Every effect is admitted before it starts, wherever and whenever
it was spawned, so every Prepare and Gate system sees it, in their order.
Prepare sees an effect once; Gate sees it each pass it is ready to start,
so once, and again after each hold on it is released; Review sees an
outcome each pass until it settles.

**An observer is for reactions that need no order**: counting, a status
item, a span, the plugin's own entities (`On<Add<Started>>`,
`On<Add<Settled>>`, `On<Add<ToolRequest>>`). Observers of one event run in
no fixed order, so an observer never edits stage data another plugin
edits; it may decide an effect, since the first decision wins.

**The conflict rule.** A hold is any entity `HoldOf(effect)`: it keeps the
effect from starting, or its outcome from settling, until the hold is
despawned or loses its `HoldOf`. Holds add up: releasing yours never
releases another plugin's, and "allow" means doing nothing. The first
decision wins: a gate decides through the effect's `Decision`
(`decision.decide(Outcome::refused(..))`), so a later gate can never
overturn a refusal; the kernel applies it as the `Outcome` at Start. A
reviewer edits the `Outcome` in place (`Query<&mut Outcome, Unreviewed>`),
ordered against other reviewers, once (a marker), since Review sees it
each pass; the record keeps both the first and the reviewed outcome.
Edits nobody ordered must commute. Two
unordered gates in one pass may hold and answer the same call: that is
harmless, the hold goes when the call settles.

**Deciding once.** A gate that only reads (a path rule) decides afresh
each time it sees a call. One whose decision asks someone or has side
effects marks the call it decided on (a `SparseSet` marker such as
`Approved`), skips marked calls, and leaves held calls alone, so a gate
ordered after it decides only once the first released the call. A
decision is applied when the pass reaches Start, so until then the call
is still `Undecided` for the gates after the one that decided: a gate
that holds or asks checks `decision.is_decided()` first.

**Failing closed.** A stage system that fails or panics closes its stage
for that pass, whatever the app's error handler: a failed Prepare system
answers what that pass admitted with an error, a failed Gate system starts
nothing, a failed Review system settles nothing. A notice names the
system. So a broken guard never lets a call through.

**Taking over and trying again.** A reviewer that takes a failure over
spawns a successor that `Supersedes(failed)`, with the request it needs
and the same `CallOf`: the failed effect is recorded as superseded and
never reaches the conversation. Work that takes time holds the outcome
first, then supersedes it.

**The loop's edge.** Work spawned, answered or released after `AgentLoop`
ran (in `PostUpdate` or `Last`) waits for the next frame. A system that
sends messages, answers effects or releases holds every frame runs in
`Update` `.before(AgentLoop)`.

```rust,no_run
use rig_harness::prelude::*;
use rig_harness::rig_core::effect::Outcome as Answered;
use rig_harness::rig_core::tool::ToolOutput;

#[derive(Default)]
pub struct WriteGuardPlugin;

impl Plugin for WriteGuardPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            AgentPass,
            (
                src_only.in_set(Stage::Gate),
                shorten_errors.in_set(Stage::Review),
            ),
        );
    }
}

/// Refuses a `write` outside `src/` when it is about to run.
fn src_only(mut calls: Query<(&ToolRequest, &mut Decision), (Undecided, ToolCalls)>) {
    for (request, mut decision) in &mut calls {
        let path = request.call.function.arguments["path"].as_str().unwrap_or_default();
        if request.call.function.name.as_str() == "write" && !path.starts_with("src/") {
            decision.decide(Outcome::refused("write only under src/"));
        }
    }
}

/// Keeps the first line of a failed `shell` call's output, before it is
/// committed.
fn shorten_errors(
    mut calls: Query<
        (Entity, &ToolRequest, &mut Outcome),
        (Unreviewed, ToolCalls, Without<Shortened>),
    >,
    mut commands: Commands,
) {
    for (call, request, mut outcome) in &mut calls {
        let Ok(Answered::ToolResult { result }) = &outcome.0 else {
            continue;
        };
        let text = result.output().as_text().unwrap_or_default();
        if request.call.function.name.as_str() == "shell" && !text.contains("[exit code 0]") {
            let first = text.lines().next().unwrap_or_default().to_owned();
            *outcome = Outcome::from(ToolResult::success(ToolOutput::text(first)));
            // Edit once: Review sees the outcome each pass until it settles.
            commands.entity(call).insert(Shortened);
        }
    }
}

/// On a call whose outcome was shortened: once is enough.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Shortened;
```

## Ordering against other plugins

Debug builds warn about two unordered stage systems that write the same
component through their queries, such as two `Query<&mut ModelRequest>`
in Prepare, two gates that decide (`&mut Decision`) or two reviewers that
edit (`&mut Outcome`). Writes through `Commands`, such as holds, are not
seen, and need no order. Queries of different kinds of effect never meet
when they say so with a kind filter (`ToolCalls`, `ModelCalls`,
`IncomingMessages`, `PluginEffects`), and two plugin effects when one
names its marker (`With<Ask>`) and the other excludes it. Inside one
system, two queries that write the same component must rule each other
out this way, or Bevy refuses the system when it is initialized (error
B0001). So:

- **A stage system that edits or answers data another plugin may also
  edit goes in a public `SystemSet` of its plugin**, configured in its
  stage: `app.configure_sets(AgentPass, MySystems.in_set(Stage::Prepare))`.
- **Another plugin orders against the set**, never against a private
  system: `.before(MySystems)` or `.after(MySystems)`.

The default plugins' stage systems that edit shared data are in these
sets:

| Set | Crate | Stage | What it does | Order against it |
|---|---|---|---|---|
| `AttachSystems` | `rig_coding_tools` | Prepare | reads the files a user's message names (`@path`) into its `attachments` | `.after` it to see or change the attached text |
| `ProjectContextSystems` | `rig_basics` | Prepare | puts the default system prompt at the head of a request whose agent has none of its own | `.after` it to see the whole prompt |
| `CompactionSystems::Prepare` | `rig_compaction` | Prepare | measures each turn's request against the window of the model it goes to, after the two sets above; clears old tool outputs or holds the call for a summary | `.before` it to route a call or change its request, so what it measures is what is sent |
| `CompactionSystems::Recover` | `rig_compaction` | Review | takes over a turn's call the model refused as too long | `.before` or `.after` it to take a failed call over first or last |
| `CommandSystems` | `rig_harness` | Gate | takes over a user's line that starts with `/` and runs it | `.before` it to take a line over first |
| `AskSystems` | `rig_harness` | Gate | answers the oldest open question that takes text with the user's next line, after `CommandSystems` | `.after` it to answer questions in Gate, so a line the user typed wins |
| `PrintSystems` | `rig_print` | Gate | answers a `--print` run's questions from `--answer`, after `AskSystems` | `.before` it to answer a question first |
| `SubagentSystems::Gate` | `rig_subagents` | Gate | refuses a `SpawnSubagent` deeper than the policy allows; stops the restored turn of an agent that owed reports | `.after` it to decide or hold a request it let through |

```rust,no_run
use rig_basics::ProjectContextSystems;
use rig_coding_tools::AttachSystems;
use rig_compaction::CompactionSystems;
use rig_harness::prelude::*;

/// What never goes to a model.
const TOKEN: &str = "sk-live-0000";

/// The redaction plugin's stage systems, for others to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RedactSystems;

#[derive(Default)]
pub struct RedactPlugin;

impl Plugin for RedactPlugin {
    fn build(&self, app: &mut App) {
        app.configure_sets(AgentPass, RedactSystems.in_set(Stage::Prepare))
            .add_systems(
                AgentPass,
                (
                    // After the files a message names were attached.
                    redact_messages.after(AttachSystems),
                    // After the default prompt went in, and before
                    // compaction measures the request.
                    redact_prompts
                        .after(ProjectContextSystems)
                        .before(CompactionSystems::Prepare),
                )
                    .in_set(RedactSystems),
            );
    }
}

/// Takes the token out of each message as it comes in, attached files
/// included.
fn redact_messages(mut messages: Query<&mut Incoming, Added<Admitted>>) {
    for mut message in &mut messages {
        message.text = message.text.replace(TOKEN, "[token]");
        for attachment in &mut message.attachments {
            if let UserContent::Text(text) = &mut attachment.content {
                text.text = text.text.replace(TOKEN, "[token]");
            }
        }
    }
}

/// Takes the token out of each request's system prompt.
fn redact_prompts(mut calls: Query<&mut ModelRequest, Added<Admitted>>) {
    for mut call in &mut calls {
        if let Some(prompt) = call.system_prompt_mut() {
            *prompt = prompt.replace(TOKEN, "[token]");
        }
    }
}
```

# A tool

Any `PortableTool` (or `rig_core::tool::Tool`). Its rules go into the
system prompt of every agent it is offered to. What each call touches
lets calls on different things run side by side: the tool says it per
call (`Tool::footprint`, as rig-tools' `read` says it reads the path it
names), else `footprint` says it for every call, else the tool's
annotations do (`Reads(Anything)` for one that says it only reads,
`Writes(Anything)` otherwise). Blocking work goes in `blocking`, on a thread
of its own. `args_schema` derives the parameters from the arguments type,
whose field doc comments describe them.

```rust,no_run
use rig_harness::prelude::*;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Default)]
pub struct WordCountPlugin;

impl Plugin for WordCountPlugin {
    fn build(&self, app: &mut App) {
        app.add_tool_with(
            WordCount,
            ToolOptions {
                rules: &["Use `word_count` to count a file's words."],
                footprint: Some(Touches::Reads(Scope::Anything)),
            },
        );
    }
}

struct WordCount;

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Args {
    /// The file.
    path: String,
}

impl PortableTool for WordCount {
    const NAME: &'static str = "word_count";
    type Args = Args;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Count the words of a file.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<Args>()
    }

    async fn call(&self, args: Args) -> Result<String, ToolExecutionError> {
        blocking(move || {
            let text =
                std::fs::read_to_string(&args.path).map_err(ToolExecutionError::from_error)?;
            Ok(format!("{} words", text.split_whitespace().count()))
        })
        .await
    }
}
```

A tool the plugin answers itself, later, is an open tool:
`app.add_open_tool(name, description, options, observer)` parses each
call's arguments into the type the observer takes and triggers
`ToolCalled<Args>` with the call's entity (`call`), the calling `agent`
and the `args`. The call ends when the plugin decides it, through its
`Decision` (`Query<&mut Decision, ToolCalls>`), such as with
`ToolResult::success("Done.".into())`: the first decision counts. A tool
found while the agent runs, such as one an MCP server lists, is
registered from a system with
`commands.queue(move |world: &mut World| { world.spawn_tool(definition, handler, options); })`:
its `ToolDefinition`, and a rig-core `ErasedHandler` that answers its
calls; despawning the entity it returns removes it. How the
terminal view (`rig-tui`) draws a tool's calls is a renderer the tool's
plugin adds; the first one of a tool stays. A crate that should also
build without the terminal view puts it behind a default cargo feature,
as `rig-coding-tools` does with `tui = ["dep:rig-tui"]`.

```rust,no_run
use rig_harness::prelude::*;
use rig_tui::{AppToolRenderersExt, RESULT_LINES};

fn build(app: &mut App) {
    app.add_tool_renderer("word_count", |call| {
        let mut lines = vec![call.header("Count", call.argument("path").unwrap_or_default())];
        lines.extend(call.result_lines(RESULT_LINES));
        lines
    });
}
```

# Gating and reviewing tool calls

A tool call is an effect with a `ToolRequest` (the model's `ToolCall`)
and the `CallHandler` that will run it, on the call's entity. A Prepare
system edits them as the call is admitted: the arguments, or the
handler, wrapped with
`ErasedHandler::layered` so the call runs through it (to give it a secret
the model never sees, say). Replacing a call's `CallHandler` runs that one call another way, such
as one agent's `shell` calls in their own directory or sandbox (below),
while `Processes` stays the way every other call starts. When Gate
opens, the kernel gives each call the `Footprint` of the call as Prepare
left it. A Gate system sees each call when it is about to run: a call
that conflicts with an earlier one of the same reply
(`Touches::conflicts_with`) is `Waiting` until that one settles, so a gate
judges it with the earlier calls' effects done. It decides it with
`Outcome::refused(why)`, which the model reads as the tool's refusal, or
with any `Outcome` (a cached result), or holds it. A Review
system reads or edits each result in place before it is committed, as
[above](#how-a-plugin-steps-in-effects-and-stages). The reply's results go
back in the order the model asked for them, refused ones included. What a
tool published beside its text, such as rig-tools' `FilesChanged` or
`ExitStatus`, is the call's `Published` component. A call of a tool the
agent's `ToolAccess` leaves out is refused at Start (a `Denied` refusal
in its record), and the model reads it as a tool it does not have.

```rust,no_run
use std::path::PathBuf;

use rig_harness::prelude::*;
use rig_harness::rig_core::serve::ErasedHandler;
use rig_harness::rig_core::serve::adapters::ToolAdapter;
use rig_harness::rig_ecs::effects::Handler;
use rig_tools::Shell;

/// On an agent whose `shell` calls run in its own directory, sandboxed.
#[derive(Component)]
pub struct Sandboxed(pub PathBuf);

/// A sandboxed agent's `shell` calls get a handler of their own.
fn sandbox(
    mut calls: Query<(&ToolRequest, &ForAgent, &mut CallHandler), Added<Admitted>>,
    agents: Query<&Sandboxed>,
    processes: Res<Processes>,
) {
    for (request, &ForAgent(agent), mut handler) in &mut calls {
        let Ok(Sandboxed(dir)) = agents.get(agent) else {
            continue;
        };
        if request.call.function.name.as_str() == "shell" {
            let launcher = processes.launcher().in_dir(dir).wrapped(["firejail", "--quiet"]);
            let shell = Shell { launcher, ..Shell::default() };
            handler.0 = Handler(ErasedHandler::new(ToolAdapter::new(shell)));
        }
    }
}
```

## A held call holds back the later calls of its reply

A call's `Footprint` says what it may read and change (`Touches`):
`Reads` or `Writes` some paths (a directory covers what is under it) or
`Anything`, or `Independent`, which changes nothing the reply's other
calls see, such as starting a subagent. A call waits for an earlier call
of its reply when one of them writes what the other touches; an
`Independent` call counts as reading anything. A held call is not
settled, so it holds back what conflicts with it like a running one.
Say the model's reply asks for five calls, and a permission gate holds
the `edit`:

| # | Call | Footprint | Runs |
|---|---|---|---|
| 1 | `edit src/a.rs` | `Writes([src/a.rs])` | once the user said yes |
| 2 | `read src/b.rs` | `Reads([src/b.rs])` | at once |
| 3 | `search docs/` | `Reads([docs])` | at once |
| 4 | `read src/a.rs` | `Reads([src/a.rs])` | once 1 settled: it ran, or was refused |
| 5 | `task` | `Independent` | once 1 settled |

A `search` without a path reads the working directory, and `shell` may
change anything (`Writes(Anything)`), so every call after a held `shell`
call waits, reads too. That is on purpose: the model wrote its calls in
order, and a later call may depend on what an earlier one changes
(`mkdir x` then `ls x`, `git stash` then `git checkout`), and only the
tool knows what its calls touch. So a gate that cannot get an answer
refuses at once rather than holding: a refused call settles, and the
next ones run.

# Asking the user: questions and pickers

A question for the user is an `Ask` entity: a title, a body, the choices
offered and whether a line of text answers it too. It is an effect like
any, recorded with its answer, and any front answers it by deciding it
(`Decision`, with `Answer::Choice(id).outcome()` or `Answer::Text(..)`);
the first decision counts and closes it in every other front. A plugin that waits on
the answer spawns the `Ask` as a hold on what it asks about, so the call
waits for it, and an interrupt, which despawns its turn's holds, takes the
question away everywhere. A front that shows questions spawns an
`Answerer` (for one agent, or `Answerer::default()` for all); a plugin
asks only when `Answerers::can_answer` for the agent, and applies its own
policy otherwise, such as refusing. While a question that takes text is
open, the user's next line to its agent answers it instead of reaching the
model. Tests answer with `rig_harness_test_support::AnswerAsks` or
`answer_open` ([testing](#testing-a-plugin)).

A choice from a list is a `Picker` entity of the plugin's
(`rig_harness::pick`, in the prelude): a title, a body, items that each
may run a command line, and whether typed text answers it too
(`free_text`). Inserting a new `Picker` replaces what it shows, and
despawning it closes it. Any front shows pickers its own way and answers
one, or an open `Ask`, by triggering `Pick { entity, chosen }`: on a
picker, the harness runs the chosen item's command, if it has one, for
the picker's agent or else the agent the user looks at (`Shown`), triggers
`Picked` on it (`Chosen::Item(index)`, `Chosen::Text` for typed text,
`None` when closed) and despawns it; on an `Ask`, it decides the answer.
So a plugin that offers a picker works in every front, and a front that
answers with `Pick` works with every plugin.

```rust,no_run
use rig_harness::prelude::*;

#[derive(Default)]
pub struct ApproveCommitsPlugin;

impl Plugin for ApproveCommitsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(AgentPass, (answered, ask).chain().in_set(Stage::Gate));
    }
}

/// On a call the user allowed.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Approved;

/// Holds each `git commit` about to run with a question, or refuses it
/// when nobody can answer.
fn ask(
    mut calls: Query<
        (Entity, &ToolRequest, &ForAgent, &mut Decision),
        (Undecided, ToolCalls, Without<Approved>),
    >,
    answerers: Answerers,
    mut commands: Commands,
) {
    for (call, request, &ForAgent(agent), mut decision) in &mut calls {
        let command = request.call.function.arguments["command"].as_str().unwrap_or_default();
        // A gate before this one decided it already: nothing to ask.
        if !command.contains("git commit") || decision.is_decided() {
            continue;
        }
        if !answerers.can_answer(Some(agent)) {
            decision.decide(Outcome::refused("nobody can approve a commit"));
            continue;
        }
        let ask = Ask::yes_no("Commit?").body(command.to_owned());
        commands.spawn((ask, ForAgent(agent), HoldOf(call)));
    }
}

/// Lets the call run, or refuses it, once the user answered; the question
/// lets go of it either way.
fn answered(
    asks: Query<(Entity, &Outcome, &HoldOf), (With<Ask>, Added<Outcome>)>,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut commands: Commands,
) {
    for (ask, outcome, &HoldOf(call)) in &asks {
        if Answer::of(outcome).is_some_and(|answer| answer.is("yes")) {
            commands.entity(call).insert(Approved);
        } else if let Ok(mut decision) = calls.get_mut(call) {
            decision.decide(Outcome::refused("the user said no"));
        }
        commands.entity(ask).remove::<HoldOf>();
    }
}
```

# A slash command

A command is a one-shot system, `app.add_command(name, help, system)`,
taking `In<CommandArgs>`: the `agent` it was typed for, the `args` after
its name, and the `mode` its line was sent with (`Queue` when the user
queued it; `Steer` from a key or a picker), for a command that sends text
on. `RunCommand::new(agent, line)` runs one. An error notice about the
agent while the command runs, from it or from what it triggers, refuses
it: the line goes back in the input with the error, and Enter then sends
it to the model as it is (a `literal` `Deliver`).

Commands are the harness's, not the kernel's: a `Gate` system in the
public set `CommandSystems` takes over a user's line that starts with `/`
(`incoming.command()` is its line, unless it is `literal`) and runs it,
in the pass it is sent. A plugin that takes some `/` lines over itself
orders its `Gate` system `.before(CommandSystems)`.

```rust,no_run
use rig_harness::prelude::*;

#[derive(Default)]
pub struct RemindPlugin;

impl Plugin for RemindPlugin {
    fn build(&self, app: &mut App) {
        app.add_command("remind", "Remind the agent of something after its turn", remind);
    }
}

/// `/remind <text>`.
fn remind(In(args): In<CommandArgs>, mut commands: Commands, mut notices: MessageWriter<Notice>) {
    if args.args.is_empty() {
        notices.write(Notice::error(args.agent, "/remind needs a text"));
        return;
    }
    // A message in the agent's conversation: `Steer` goes with the running
    // turn's next model call, `Queue` once that turn would end, together
    // with everything else queued; an idle agent starts a turn. A `Note`
    // needs no answer and starts no turn: it goes with the next call
    // built after it is accepted (at Start, after Admit built that pass's).
    let reminder = Deliver::new(args.agent, format!("Reminder: {}", args.args), DeliveryMode::Queue);
    commands.trigger(reminder.with_origin(Origin::plugin("remind")));
}
```

# What the user types

What the user types, in any front, is a message to its agent from
`Origin::user()`, and so is what an agent or a plugin sends. `Deliver`
spawns it as an effect, an `Incoming` that is `ForAgent` the agent, and it
moves through the stages like a tool call. A `Prepare` system may change
it before it is sent (`Added<Admitted>`): rewrite its `text`, add
`attachments` (the `rig-coding-tools` plugin reads `@path` files this
way). A `Gate` system may take it over by deciding it, such as with
`Outcome::custom(..)`, and then it never reaches the agent; the harness
takes a user's `/` line over this way and runs it as a command (see
above). At `Start` the kernel accepts what is left: it goes to the agent,
which starts a turn with it or keeps it for its running turn as its
`mode` says. The messages an agent accepted
and has not sent yet are its inbox,
`Query<(&Incoming, &ForAgent), (With<Started>, Without<Outcome>)>`, and
`On<Add<Started>>` sees each message as it is accepted. A message's record
in the effect log is the custom kind `message`, and
`MessageRecord::read(&record.kind)` reads its text, origin and mode back.

**With the user's next message.** What a plugin wants the agent to read
the next time the user writes, such as a background check's failure, is
a Prepare edit of that message: the plugin keeps it in its own state and
adds it to the next `Incoming` from `OriginKind::User` as an
`Attachment { label, content }` (`Added<Admitted>`). It then starts no
turn of its own, needs no hold, and is taken back by clearing that state.
To have the agent act on it at once, `Deliver` it instead.

A plugin that needs a mark on its own message spawns it:
`(Incoming::new(text, mode), ForAgent(agent), MyMark)` is all `Deliver`
does (the kernel records it under the agent's work and wakes the loop).
Once the kernel delivers a message into a turn, the turn it started or
the running turn it steered or carried on, the message gets
`DeliveredIn(turn)`, and a turn's first model call is recorded under the
message that started it. So a plugin that tags its message finds the
turn, and from it the turn's calls (`CallOf(turn)`), such as to route
them. A user's message may carry a title, `Origin::user().titled(..)`: it
still goes to the model as the user's words, and the conversation keeps
the origin (`Conversation::origin`), so a view can show where it came
from.

```rust,no_run
use rig_harness::prelude::*;

#[derive(Default)]
pub struct ShortcutsPlugin;

impl Plugin for ShortcutsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            AgentPass,
            (
                (expand, review_on_reviewer).in_set(Stage::Prepare),
                // Before the harness's systems that decide messages too.
                keep_asides.in_set(Stage::Gate).before(CommandSystems),
            ),
        )
        .add_command("review", "Asks for a review, on the reviewing model", review);
    }
}

/// On a message `/review` sent.
#[derive(Component)]
struct Review;

/// `/review` asks the model for a review, titled so the transcript says so.
fn review(In(args): In<CommandArgs>, mut commands: Commands) {
    let text = "Review the change you just made.";
    let origin = Origin::user().titled("review");
    let message = Incoming::new(text, args.mode).with_origin(origin);
    commands.spawn((message, ForAgent(args.agent), Review));
}

/// The model that reviews.
const REVIEWER: &str = "deepseek/deepseek-flash";

/// The calls of the turn a review went with go to the reviewing model.
fn review_on_reviewer(
    mut calls: Query<(&mut ModelRequest, &CallOf), (Added<Admitted>, With<TurnCall>)>,
    reviews: Query<&DeliveredIn, With<Review>>,
    models: Res<Models>,
) {
    for (mut call, &CallOf(turn)) in &mut calls {
        if reviews.iter().any(|&DeliveredIn(into)| into == turn) {
            call.connection = models.connect(REVIEWER).ok();
        }
    }
}

/// `lgtm` typed alone asks the model to commit.
fn expand(mut typed: Query<&mut Incoming, Added<Admitted>>) {
    for mut typed in &mut typed {
        if typed.origin.kind == OriginKind::User && typed.text.trim() == "lgtm" {
            typed.text = "Looks good to me: commit it with a short message.".to_owned();
        }
    }
}

/// A line that starts with `#` is the user's own aside: it never reaches
/// the model.
fn keep_asides(
    mut typed: Query<(&Incoming, &ForAgent, &mut Decision), (Undecided, IncomingMessages)>,
    mut notices: MessageWriter<Notice>,
) {
    for (typed, &ForAgent(agent), mut decision) in &mut typed {
        if typed.origin.kind == OriginKind::User && typed.text.starts_with('#') {
            decision.decide(Outcome::custom("aside"));
            notices.write(Notice::info(agent, "Noted; not sent."));
        }
    }
}
```

# Model requests

A model call is an effect entity with a `ModelRequest`: the rig-core
`CompletionRequest` it sends (system prompt as its first message, the
messages, the tools offered, the options), the `connection` it goes to,
else its agent's model (`Models::connect` makes one); its `Adapters` are a
component beside it.
A turn's own call is a `TurnCall`, whose request the kernel builds from
its agent when it is admitted (an `On<Add<Admitted>>` observer runs
before that; a Prepare system sees it built); a call built from an agent outside its
turns, such as a preview, is `BuildFrom(agent)`. A plugin changes a
request with a system in the `Prepare` stage, which sees each call once
(`Added<Admitted>`), ordered against
[the default plugins' sets](#ordering-against-other-plugins): a router
or a request edit runs `.before(CompactionSystems::Prepare)`, so
compaction measures what is sent. Start then fits the request to the
model it goes to (an option that model cannot take is left out, plugin
edits stay) and makes the tool rules in the prompt follow the tools it
carries. Changing the system prompt or the tools misses the provider's
prompt cache.

- **Routing** a call to another model is a Prepare edit of its
  `connection`.
- **Adapting** the model, whichever it is, such as wrapping its handler so
  it reads tools as text or so its stream is rewritten, is an `Adapter`
  pushed on the call's `Adapters` (`Query<&mut Adapters, Added<Admitted>>`,
  so it never conflicts with a request edit). Start applies the adapters
  in the order pushed, those placed `outermost()` last, to
  the connection the call finally goes to, after every Prepare edit, then
  fits the request to the adapted spec: a router and an adapter never undo
  each other, and neither needs to come after the other. An adapter wraps
  `connection.handler` with rig-core's `ErasedHandler::layered` (see each
  request and answer) or `map_stream` (rewrite each streamed item and the
  response the reply ends with, recorded as rewritten), and says what the
  wrapped model takes on `connection.spec`.
- **What a call was sent as.** From `Start` on, a model call has
  `SentTo(connection)`: the model it went to with every adapter applied,
  whose spec says what that model finally takes, and its
  `ModelRequest.request` is the request as sent. A plugin that must know
  what the other adapters made of a call reads these in `Review` (or on
  `Add<Settled>`) instead of guessing from the order adapters were pushed
  in. A `map_stream` closure is given the call's `PublishedContext`: a
  fact it inserts about the reply, such as that it read tool calls from
  text, is on the call as `Published` once it is answered, as a tool's
  facts are, and recorded with it.
- **Tools.** The kernel offers an agent the tools its `ToolAccess` allows
  and leaves taking them away to the model the call goes to: a call routed
  from a model that calls no tools to one that does is sent its tools. A
  tool left out of a request is not offered but still runs if called: a
  `Gate` system refuses a call (`Outcome::refused`), and `ToolAccess` on
  the agent is the hard limit.
- **Below every adapter: the HTTP client.** A provider plugin may send a
  vendor's requests through its own client:
  `connector.provide(vendor, ConnectOptions::new().http(client))`, with
  any rig-core `http_client::HttpClientExt` ([Adding to another plugin's
  lists](#adding-to-another-plugins-lists) says how a provider plugin
  provides). Every call to that vendor's models goes through it, a turn's
  and a plugin's own alike, under every adapter: the place to rotate API
  keys, or to send a request that got a 429 again on another key
  (`http_client::Error::non_success_status` and `non_success_headers`
  read the status and `Retry-After`), which no effect then sees fail.
- **What the model is sent in place of old messages.** A compaction
  splices its summary into the conversation in place of the messages it
  summarized (`Commit::splice`), as one user item whose origin is
  `rig_compaction::summary_origin()`; requests send the conversation as
  it is. A `Committed::Splice` whose changed messages hold such an item
  is a compaction.
- `rig_ecs::prompt::system_prompt` and `tool_rules` put a prompt together
  exactly as the kernel does, such as for a view of what an agent is told.
  `rig_memory::HeuristicTokenCounter` estimates tokens as rig-compaction
  does.

A failed call carries the kernel's `RetryVerdict` (`Retry`, `GaveUp`,
`Overflow`, `Final`) in `Review`. A plugin that takes the failure over
spawns a new call that `Supersedes` the failed one (the old one is
recorded as superseded and never reaches the conversation), or holds it
(`HoldOf`) while it works, as compaction (the `rig-compaction` crate)
does to summarize first; otherwise the kernel retries after a `Backoff`
or proposes the turn's end ([How a turn ends](#how-a-turn-ends)). A
call refused before it was sent fails the turn and keeps the user's
message.

A part of the system prompt that does not change from request to request
is a `PromptSection` entity, `PromptSection::new(order, tag, text)`, in
every agent's prompt, or with a `SectionOf(agent)` in that agent's
alone, despawned with it.

```rust,no_run
use std::sync::Arc;

use rig_compaction::CompactionSystems;
use rig_harness::prelude::*;
use rig_harness::rig_core::streaming::Relayed;
use rig_harness::rig_core::tool::{ContextValue, PublishedContext};
use rig_harness::rig_ecs::effects::Handler;
use serde::{Deserialize, Serialize};

/// The model a turn falls back to when its own gave up.
const FALLBACK: &str = "deepseek/deepseek-flash";

/// On an agent that only plans.
#[derive(Component)]
pub struct PlanMode;

#[derive(Default)]
pub struct ModelRequestsPlugin;

impl Plugin for ModelRequestsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            AgentPass,
            (
                (add_the_time, plan_only, tools_as_text)
                    .in_set(Stage::Prepare)
                    .before(CompactionSystems::Prepare),
                (fall_back, read_from_text).in_set(Stage::Review),
            ),
        );
    }
}

/// What the text-tools rewrite tells a call about its reply: how many tool
/// calls it read from the text.
#[derive(Serialize, Deserialize)]
struct ReadAsText(usize);

impl ContextValue for ReadAsText {
    const KEY: &'static str = "read-as-text";
}

/// Reads the tool calls a model writes as text into calls, item by item
/// (the plugin's own parser; here it passes each item on), and tells the
/// call how many it read once the reply is done.
fn read_calls(published: Arc<PublishedContext>) -> impl FnMut(Relayed) -> Vec<Relayed> {
    move |item| {
        if let Relayed::Done(_) = &item {
            published.insert(ReadAsText(0)).ok();
        }
        vec![item]
    }
}

/// Says when a reply's tool calls were read from its text, naming the
/// model the call finally went to.
fn read_from_text(
    calls: Query<(&SentTo, &Published, &ForAgent), (Unreviewed, With<TurnCall>)>,
    mut notices: MessageWriter<Notice>,
) {
    for (sent, published, &ForAgent(agent)) in &calls {
        if let Ok(Some(ReadAsText(calls))) = published.0.get::<ReadAsText>() {
            let model = &sent.0.spec.display_name;
            notices.write(Notice::info(agent, format!("{model} wrote {calls} tool calls as text.")));
        }
    }
}

/// Every turn's call goes to its model with the tools as text, when that
/// model calls none, whichever model a router picks.
fn tools_as_text(mut calls: Query<&mut Adapters, (Added<Admitted>, With<TurnCall>)>) {
    for mut adapters in &mut calls {
        adapters.0.push(Adapter::new("text-tools", |mut connection: Connection| {
            if !connection.spec.tools {
                let mut spec = (*connection.spec).clone();
                spec.tools = true;
                connection.spec = std::sync::Arc::new(spec);
                connection.handler = Handler(connection.handler.0.map_stream("text-tools", read_calls));
            }
            connection
        }));
    }
}

/// The turns' calls, as Prepare sees each once.
type NewTurnCalls<'w, 's> =
    Query<'w, 's, (&'static mut ModelRequest, &'static ForAgent), (Added<Admitted>, With<TurnCall>)>;

/// Every turn's request tells the model how long the agent has run, at
/// its end, where the prompt cache is not disturbed.
fn add_the_time(mut calls: NewTurnCalls, time: Res<Time<Real>>) {
    let note = format!("(The agent has run for {} s.)", time.elapsed().as_secs());
    for (mut call, _) in &mut calls {
        if let Some(message::Message::User { content }) = call.request.chat_history.last_mut() {
            content.push(UserContent::text(note.clone()));
        }
    }
}

/// An agent in plan mode is offered only the tools that read, and told so.
fn plan_only(mut calls: NewTurnCalls, planning: Query<(), With<PlanMode>>) {
    for (mut call, &ForAgent(agent)) in &mut calls {
        if planning.contains(agent) {
            let reads = |name: &str| matches!(name, "read" | "search");
            call.request.tools.retain(|tool| reads(tool.name.as_str()));
            if let Some(prompt) = call.system_prompt_mut() {
                prompt.push_str("\n\nPlan only: change nothing yet.");
            }
        }
    }
}

/// A turn's call whose retries gave up is sent again at once, on the
/// fallback model, unless it went there already.
fn fall_back(
    failed: Query<
        (Entity, &RetryVerdict, &ModelRequest, &Effect, &CallOf, &ForAgent),
        (Unreviewed, With<TurnCall>, Without<SupersededBy>),
    >,
    models: Res<Models>,
    mut commands: Commands,
) {
    for (call, verdict, request, effect, of, agent) in &failed {
        let routed = request.connection.as_ref().map(|to| to.spec.reference());
        if !matches!(verdict.0, Verdict::GaveUp(_)) || routed.as_deref() == Some(FALLBACK) {
            continue;
        }
        let Ok(fallback) = models.connect(FALLBACK) else {
            continue;
        };
        commands.spawn((
            request.clone().on(fallback),
            TurnCall,
            Supersedes(call),
            CallOf(of.0),
            ForAgent(agent.0),
            Effect::child_of(effect.parent),
        ));
    }
}
```

# What a model streams

Every model call streams, a turn's and a plugin's own alike. Each fragment
of text, reasoning or a tool call's argument JSON is a
`StreamDelta { effect, agent, kind, part, text }` message (`part` tells
parallel tool calls apart), written in `Arrive` in the order it came, before the call's
outcome is applied; Bevy keeps a message at least two frames, so a reader in
`PostUpdate` gets the last fragments of a reply that settled in the same
frame. They are what the call's adapters left: a `map_stream` rewrite
comes before them. A turn's call also adds its text and reasoning up in its `Partial`,
which is gone once the reply is in the conversation. From its stream's
first item of any kind, text, a tool call or the end, the call has a
`Streamed { first, last, items }` beside its `Started { at }`: `last` is
its latest item, the reply's end once it ended.

```rust,no_run
use rig_harness::prelude::*;

#[derive(Default)]
pub struct StreamTapPlugin;

impl Plugin for StreamTapPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PostUpdate, print_text).add_observer(time_to_first);
    }
}

/// Prints every model call's answer text as it comes.
fn print_text(mut deltas: MessageReader<StreamDelta>) {
    for delta in deltas.read() {
        if delta.kind == DeltaKind::Text {
            print!("{}", delta.text);
        }
    }
}

/// Says how long each model call took to start answering.
fn time_to_first(
    streamed: On<Add<Streamed>>,
    calls: Query<(&Streamed, &Started, Option<&ForAgent>)>,
    mut notices: MessageWriter<Notice>,
) {
    if let Ok((streamed, started, agent)) = calls.get(streamed.entity) {
        let waited = streamed.first.duration_since(started.at).unwrap_or_default();
        let text = format!("First token after {} ms.", waited.as_millis());
        notices.write(Notice::info(agent.map(|agent| agent.0), text));
    }
}
```

# How a turn ends

A turn is an entity `TurnOf` its agent, with a `Turn`: its `index` among
the agent's turns over the whole session (saved with the agent) and its
`first_message`, the `MessageId` of the message it begins at
(`conversation.start_of(turn.first_message)` is its index now, as splices
move messages). The kernel makes it `ForAgent` its
agent when it is spawned, so it goes with the agent. It is spawned before
its first request is built, so an `On<Add<TurnOf>>` observer's edit, such
as a prompt section's new text, is in that request; once the turn ends it
loses `TurnOf` and keeps `ForAgent`. A turn's
end is a proposal: when the model answers without calling tools, or the
turn fails, the kernel proposes a `TurnOutcome` on the turn in `Commit`
(`Answered(message)`, `Failed(reason)`), and a `Review` system sees it in
the next pass (`Added<TurnOutcome>`). It may edit it in place
(`Query<&mut TurnOutcome, With<TurnOf>>`), hold
it (spawn an entity `HoldOf(turn)`, despawn it when done), or carry the
turn on: a `Queue` or `Steer` message the agent accepts takes an
`Answered` proposal away, one on its way keeps the turn from ending
meanwhile, and a new `(TurnCall, CallOf(turn), ForAgent(agent))` takes
any away once it is admitted; a proposal taken away takes its holds with
it (they are despawned), and the next proposal is reviewed afresh. A
plugin may insert `TurnOutcome::Failed(reason)` or `Stopped(Some(reason))`
at any time, with a plain `insert`, such as a budget: the turn's calls that have not started
never start, the others are answered as stopped, and the turn ends. Once
nothing holds the outcome and every call of the turn settled, the turn
ends: the agent is idle, and the turn entity is `Settled` until the next
frame. An `On<Add<Settled>>` observer may retire the agent
(`Commit::retire`), such as one that did its one job; the turn goes with
it. `Interrupt` stops a turn at once; it cannot be held.

**Ending a turn with an answer already known**, such as the JSON a
`submit` tool was called with: a Gate system decides the turn's next
model call itself, with a completion of that text. Nothing is sent, the
turn ends `Answered` with it, and `--print` prints it as the answer.

```rust,no_run
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::{CompletionResponse, Usage};
use rig_harness::rig_core::effect;
use rig_harness::rig_core::message::{AssistantContent, Origin as Made, Text};

/// On a turn whose answer is known: the text it ends with.
#[derive(Component)]
pub struct EndWith(pub String);

/// Answers the turn's next model call with the known text.
fn end_with(
    mut calls: Query<(&CallOf, &mut Decision), (Undecided, ModelCalls, With<TurnCall>)>,
    turns: Query<&EndWith>,
) {
    for (&CallOf(turn), mut decision) in &mut calls {
        if let Ok(EndWith(text)) = turns.get(turn) {
            let choice = vec![AssistantContent::Text(Text::new(text.clone()))];
            let made = Made::new("plugin", "submit", "submit");
            let reply = CompletionResponse::new(choice, Usage::default(), made, serde_json::Value::Null);
            decision.decide(Outcome(Ok(effect::Outcome::Completion(reply))));
        }
    }
}
```

```rust,no_run
use rig_harness::prelude::*;

#[derive(Default)]
pub struct TurnEndsPlugin;

impl Plugin for TurnEndsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(AgentPass, check_the_tests.in_set(Stage::Review))
            .add_observer(announce);
    }
}

/// A turn that would end with an answer is asked once more, as its next
/// step, whether the tests pass.
fn check_the_tests(
    proposed: Query<(&TurnOf, &TurnOutcome, &Turn), Added<TurnOutcome>>,
    mut asked: Local<Vec<(Entity, u64)>>,
    mut commands: Commands,
) {
    for (&TurnOf(agent), outcome, turn) in &proposed {
        let first = !asked.contains(&(agent, turn.index));
        if matches!(outcome, TurnOutcome::Answered(_)) && first {
            asked.push((agent, turn.index));
            let ask = Deliver::new(agent, "Do the tests pass?", DeliveryMode::Queue);
            commands.trigger(ask.with_origin(Origin::plugin("check-the-tests")));
        }
    }
}

/// Says how each turn ended.
fn announce(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent)>,
    mut notices: MessageWriter<Notice>,
) {
    if let Ok((outcome, &ForAgent(agent))) = turns.get(settled.entity) {
        let how = match outcome {
            TurnOutcome::Answered(_) => "answered".to_owned(),
            TurnOutcome::Failed(why) => format!("failed: {why}"),
            TurnOutcome::Stopped(why) => format!("stopped {}", why.as_deref().unwrap_or("by you")),
        };
        notices.write(Notice::info(agent, format!("The turn {how}.")));
    }
}
```

# Effects of a plugin's own

A plugin's own work gets the same stages, recording and visibility as the
kernel's when it is an effect: an `EffectRequest` with its rig-core
`EffectKind` (`EffectKind::Custom` for a kind of its own) and a
`handler`, which the kernel runs at Start, or none, an open effect the
plugin (or anyone) decides (`Decision`) whenever it is done. Other
plugins' Gate systems can refuse it, Review systems edit its outcome,
and a replayer may answer it from the effect log; `CallOf(turn)` keeps a turn
open until it settles. Work off the main thread is a `Running` task on the
effect's entity, `Running::answering(pool, wake, future)`, whose output
becomes its outcome.

**An owner follows its effect's successor.** Another plugin may take the
effect over ([Taking over and trying
again](#how-a-plugin-steps-in-effects-and-stages)), such as a watchdog
that sends a stalled model call again: the effect then settles with no
outcome applied, recorded as superseded, and its `SupersededBy` names
the successor that carries the work on (`successor()`, which may itself
be superseded later). A plugin waiting for its own effect's outcome, on
`Add<Settled>`, checks for `SupersededBy` and moves its markers and any
hold it keeps with the effect to the successor. Commit runs before Admit,
so they are there when the successor is prepared. The kernel follows its
own turn and tool calls this way.

```rust,no_run
use rig_harness::prelude::*;
use rig_harness::rig_core::effect::EffectKind;

/// `/deploy`: a deploy effect any plugin can gate, answered by the plugin
/// itself once its pipeline reports back (here at once).
fn deploy(In(args): In<CommandArgs>, mut commands: Commands) {
    let kind = EffectKind::Custom {
        kind: "deploy".into(),
        payload: serde_json::json!({ "target": args.args }),
    };
    let request = EffectRequest { kind, handler: None };
    commands.spawn((request, ForAgent(args.agent)));
}

/// Answers each deploy that started: nothing refused it.
fn report(mut deploys: Query<(&EffectRequest, &mut Decision), (Added<Started>, PluginEffects)>) {
    for (request, mut decision) in &mut deploys {
        if request.kind.name() == "deploy" {
            decision.decide(Outcome::custom("deployed"));
        }
    }
}

fn build(app: &mut App) {
    app.add_command("deploy", "Deploy a target", deploy)
        .add_systems(AgentPass, report.in_set(Stage::Gate));
}
```

# Subagents

The `rig-subagents` plugin spawns an agent for a `SpawnSubagent` effect
(`rig_subagents`) `ForAgent` its parent: the model's `task` tool makes
one, and a plugin spawns one itself for agents of its own, such as the
sides of a debate. Like any effect, Prepare systems edit it
(`Query<&mut SpawnSubagent, Added<Admitted>>`: its model, tools, system
prompt) and Gate systems refuse or hold it. When it starts, the plugin
spawns the agent with every setting the request leaves out taken from the
parent, puts it on the request as `SpawnedAgent(agent)`, and answers with
the new agent's id, `{"agent": "<AgentId>"}`, which the record keeps.

- `SpawnSubagent::task(title, brief)` works on a task and reports to its
  parent, as `task` starts it.
- `SpawnSubagent::helper(title)` works for the plugin, which talks to it:
  it has a `Helper(title)`, reports to no one, and `message` cannot reach
  it; `.brief(text)` sends it a first message. With `.once()` it does one
  job: it retires when its first turn ends, so it is despawned and a
  restart does not bring it back. Its answer is read in an
  `On<Add<Settled>>` observer of that turn.

## Extending `task`

A plugin adds an argument to `task` by adding a property to its
`ToolDef`'s `parameters`, and a line to its `ToolRules` on when to use it
([editing another plugin's tool](#turning-off-or-replacing-what-another-plugin-added)).
`task` puts each argument it does not take itself into the request's
`SpawnSubagent.extra`, by name, recorded with the rest, so the plugin
reads it there in Prepare.

```rust,no_run
use rig_harness::prelude::*;
use rig_subagents::{Helper, SpawnSubagent, TASK};

#[derive(Default)]
pub struct ReviewersPlugin;

impl Plugin for ReviewersPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(AgentPass, as_reviewer.in_set(Stage::Prepare))
            .add_command("critic", "Ask a one-job critic about the last change", critic)
            .add_observer(show_critique);
    }

    /// In `finish`, so `task` exists whatever the plugins' order.
    fn finish(&self, app: &mut App) {
        let world = app.world_mut();
        let mut tools = world.query::<(&mut ToolDef, &mut ToolRules)>();
        for (mut def, mut rules) in tools.iter_mut(world) {
            if def.0.name.as_str() != TASK {
                continue;
            }
            if let Some(properties) = def.0.parameters.get_mut("properties")
                && let Some(properties) = properties.as_object_mut()
            {
                let profile = serde_json::json!({
                    "type": "string",
                    "enum": ["reviewer"],
                    "description": "`reviewer`: reads and reports, changes nothing.",
                });
                properties.insert("profile".to_owned(), profile);
            }
            rules.0.push("Give `task` the `reviewer` profile for a review.".to_owned());
        }
    }
}

/// A task the model gave the reviewer profile may only read.
fn as_reviewer(mut requests: Query<&mut SpawnSubagent, Added<Admitted>>) {
    for mut request in &mut requests {
        if request.extra.get("profile").and_then(|profile| profile.as_str()) == Some("reviewer") {
            request.tools = Some(vec!["read".to_owned(), "search".to_owned()]);
        }
    }
}

/// `/critic`: a helper that criticizes the last change, then retires.
fn critic(In(args): In<CommandArgs>, mut commands: Commands) {
    let critic = SpawnSubagent::helper("critic")
        .system_prompt("You criticize code changes in three lines.")
        .brief("Criticize the last change in this project.")
        .once();
    commands.spawn((critic, ForAgent(args.agent)));
}

/// Shows the critic's answer as its one turn ends.
fn show_critique(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent)>,
    helpers: Query<&Helper>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((TurnOutcome::Answered(message), &ForAgent(agent))) = turns.get(settled.entity) else {
        return;
    };
    if helpers.get(agent).is_ok_and(|helper| helper.0 == "critic")
        && let Some(answer) = final_answer(message)
    {
        notices.write(Notice::info(None, format!("Critic: {answer}")));
    }
}
```

# The effect log: reading, redacting, replaying

Every effect that settles is a `Recorded(EffectRecord)` message, in settle
order: its request (`kind`), its answer (`outcome`, and a reviewer's
replacement as `reviewed`), its times, the agent it worked for (`scope`,
the agent's `AgentId`), the agent's `turn` it was made in (`Turn.index`),
and the effect it was made under (`parent`). The `rig-telemetry` plugin
writes them to the session's effect log in `Last`, in `WriteJournal`, and
reads them back through its `EffectLog` resource, there when the session
has a directory: `read()` gives every record, and `read_matching(&filter)`
those an `EffectFilter` keeps, by agent, kind name (`completion`,
`tool_call`, `message`, `subagent` or a plugin's own) and turn. It holds
what was written by the end of the last frame. A message's record reads
back with `MessageRecord::read(&record.kind)`.

**Redacting.** A record holds what its effect really carried: tool output
and the diffs of changed files, the request as the model got it. A plugin
that keeps a secret out of the log changes the records in place with
Bevy's `MessageMutator<Recorded>`, in `Last`
`.after(StopTurns).before(WriteJournal)` (both in `rig_harness::rig_ecs`),
so it sees the records a stop on exit makes too, and the log gets only
what it left. A reader earlier in the frame sees the record unchanged.

**Replaying.** A replayer answers effects from a recorded session's log
before they start (a Gate system that decides), taking each agent's
recorded answers by kind, in order. Agent ids are new in each session, so
it pairs each agent with the recorded one it stands for: the first agent
with the first recorded agent, and a subagent through the `subagent` effect that
spawned it. That effect's recorded answer names the recorded agent, and
its answer now names the new one, so the n-th spawn of a paired parent
pairs with that parent's n-th recorded spawn, whatever order the
subagents start in. A replayer runs `subagent` effects again rather than
answering them from the record: only running one spawns the agent.

```rust,no_run
use std::collections::{HashMap, VecDeque};

use rig_harness::prelude::*;
use rig_harness::rig_core::effect::{EffectRecord, Outcome as Answered};
use rig_harness::rig_ecs::{StopTurns, WriteJournal};
use rig_subagents::{SUBAGENT, SpawnSubagent};
use rig_telemetry::{EffectFilter, EffectLog};

/// What never reaches the effect log.
const TOKEN: &str = "sk-live-0000";

fn build(app: &mut App) {
    app.add_command("effects", "What the agent did in its turn N: /effects N", effects)
        .add_systems(Last, redact.after(StopTurns).before(WriteJournal))
        .init_resource::<Pairs>()
        .add_systems(PostStartup, pair_first)
        .add_observer(pair_spawned);
}

/// `/effects N`: each effect of the agent's turn `N`.
fn effects(
    In(args): In<CommandArgs>,
    log: Option<Res<EffectLog>>,
    ids: Query<&AgentId>,
    mut notices: MessageWriter<Notice>,
) {
    let (Some(log), Ok(id), Ok(turn)) = (log, ids.get(args.agent), args.args.parse()) else {
        notices.write(Notice::error(args.agent, "/effects needs a turn number and a session log"));
        return;
    };
    let filter = EffectFilter::default().agent(id.0.clone()).turn(turn);
    for record in log.read_matching(&filter).unwrap_or_default() {
        let line = format!("{} {} {:?}", record.id, record.kind.name(), record.disposition);
        notices.write(Notice::info(args.agent, line));
    }
}

/// Takes the token out of every record before it is written.
fn redact(mut records: MessageMutator<Recorded>) {
    for Recorded(record) in records.read() {
        let Ok(json) = serde_json::to_string(&*record) else {
            continue;
        };
        if json.contains(TOKEN)
            && let Ok(clean) = serde_json::from_str(&json.replace(TOKEN, "[token]"))
        {
            *record = clean;
        }
    }
}

/// The recorded session's effect log, which the replayer read.
#[derive(Resource)]
struct Recording(Vec<EffectRecord>);

/// The recorded agent each agent of the replay stands for, by id, and the
/// agents each recorded agent spawned, in order.
#[derive(Resource, Default)]
struct Pairs {
    paired: HashMap<String, String>,
    spawned: HashMap<String, VecDeque<String>>,
}

/// The agent a `subagent` effect's answer names.
fn spawned_agent(answer: Option<&Answered>) -> Option<String> {
    let Some(Answered::Custom { payload }) = answer else {
        return None;
    };
    payload["agent"].as_str().map(str::to_owned)
}

/// Pairs the first agent with the recorded first, and notes what each
/// recorded agent spawned.
fn pair_first(
    recording: Res<Recording>,
    agents: Query<&AgentId, Without<SpawnedBy>>,
    mut pairs: ResMut<Pairs>,
) {
    let first = recording.0.iter().find_map(|record| record.scope.as_deref());
    if let (Some(first), Some(agent)) = (first, agents.iter().next()) {
        pairs.paired.insert(agent.0.clone(), first.to_owned());
    }
    for record in recording.0.iter().filter(|record| record.kind.name() == SUBAGENT) {
        if let (Some(parent), Some(agent)) = (&record.scope, spawned_agent(record.outcome.as_ref().ok())) {
            pairs.spawned.entry(parent.to_string()).or_default().push_back(agent);
        }
    }
}

/// Pairs each subagent the replay spawns with the one its parent's
/// recorded spawn at the same place made.
fn pair_spawned(
    settled: On<Add<Settled>>,
    spawns: Query<(&Outcome, &ForAgent), With<SpawnSubagent>>,
    ids: Query<&AgentId>,
    mut pairs: ResMut<Pairs>,
) {
    let Ok((outcome, &ForAgent(parent))) = spawns.get(settled.entity) else {
        return;
    };
    let (Some(agent), Ok(parent)) = (spawned_agent(outcome.0.as_ref().ok()), ids.get(parent)) else {
        return;
    };
    let Some(recorded_parent) = pairs.paired.get(&parent.0).cloned() else {
        return;
    };
    let recorded = pairs.spawned.get_mut(&recorded_parent).and_then(VecDeque::pop_front);
    if let Some(recorded) = recorded {
        pairs.paired.insert(agent, recorded);
    }
}
```

# Settings and command-line options

A plugin's settings are its table of the user's `RIG_HOME/settings.toml`:
`app.add_setting::<T>("section")` in `build` inserts the resource `T`
read from `[section]`, or `T::default()` when the user set nothing; a
table that does not read as `T` keeps the default and is shown at startup.
A test inserts `Settings::from_toml(..)` (or `Settings::default()`, which
the test support's apps do) before adding the plugin, so it never reads
the user's file.

A plugin's command-line options are declared with `app.add_flag`:
`rig --name` or `rig --name=VALUE` reaches the agent, which refuses an
option no plugin of its build declared, and `rig help` lists them. The
plugin reads them from `Invoked` (`flag(name)`, `flag_value(name)`).

```rust,no_run
use rig_harness::prelude::*;
use serde::Deserialize;

/// The `[budget]` table: `max_calls = 50`.
#[derive(Resource, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Budget {
    pub max_calls: usize,
}

impl Default for Budget {
    fn default() -> Self {
        Self { max_calls: 100 }
    }
}

#[derive(Default)]
pub struct BudgetPlugin;

impl Plugin for BudgetPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<Budget>("budget")
            .add_flag(Flag::new("budget", "Stop a turn after N tool calls").value("N"));
        let limit = app.world().get_resource::<Invoked>().and_then(|invoked| {
            invoked.flag_value("budget").and_then(|n| n.parse().ok())
        });
        if let Some(max_calls) = limit {
            app.insert_resource(Budget { max_calls });
        }
    }
}
```

# Saved state: counting every tool call

An agent component that derives `Reflect` and says
`#[reflect(Component, Saved)]` is kept with the session: each change is
logged, and a restart, `/reload` or `/resume` brings the newest value
back, inserted on the restored agent before `SessionRestored` exists, so
an `On<Insert<T>>` observer can re-arm what it says. A resource that says
`#[reflect(Resource, Saved)]` is kept the same way, for what belongs to
the whole session; the plugin inserts it in `build`, and the saved value
replaces it. A tool call is an entity with a `ToolRequest` and the
agent it works for (`ForAgent`), so `On<Add<ToolRequest>>` sees every
tool call as the model asks for it; a call a restart runs again is
marked `Restored`, like all the work a restart carries on, and is gated
again (holds and approvals are not saved). The calls
made before the plugin was added are in the conversation, which a
restored agent is spawned with: the observer that sees the plugin's
component added counts them, and a saved count, if the session has one,
replaces that right after. An assistant message is the tuple variant
`Message::Assistant(AssistantMessage)`.

```rust,no_run
use std::collections::HashMap;

use rig_harness::prelude::*;
use rig_harness::rig_core::message::Message;

/// How often each agent called each tool, kept across `/compact`,
/// `/reload` and `/resume`.
#[derive(Component, Reflect, Default)]
#[reflect(Component, Saved)]
struct ToolCounts(HashMap<String, u32>);

#[derive(Default)]
pub struct ToolCountsPlugin;

impl Plugin for ToolCountsPlugin {
    fn build(&self, app: &mut App) {
        app.register_required_components::<Agent, ToolCounts>()
            .add_observer(count)
            .add_observer(count_the_past);
    }
}

/// Counts the calls already in an agent's conversation when it gets its
/// count, as a restored agent does on the first restart with the plugin.
fn count_the_past(added: On<Add<ToolCounts>>, mut agents: Query<(&Conversation, &mut ToolCounts)>) {
    let Ok((conversation, mut counts)) = agents.get_mut(added.entity) else {
        return;
    };
    for message in conversation.messages() {
        if let Message::Assistant(assistant) = message {
            for call in assistant.tool_calls() {
                *counts.0.entry(call.function.name.as_str().to_owned()).or_default() += 1;
            }
        }
    }
}

/// Counts each call as the model asks for it, once: not again when a
/// restart runs it again.
fn count(
    call: On<Add<ToolRequest>>,
    calls: Query<(&ToolRequest, &ForAgent), Without<Restored>>,
    mut counts: Query<&mut ToolCounts>,
) {
    if let Ok((request, ForAgent(agent))) = calls.get(call.entity)
        && let Ok(mut counts) = counts.get_mut(*agent)
    {
        let name = request.call.function.name.as_str().to_owned();
        *counts.0.entry(name).or_default() += 1;
    }
}
```

A generic type is saved only once registered (`app.register_type::<T>()`).

A conversation changes only through `Commit`, so the log records every
change: `commit.splice(agent, &mut conversation, range, entries)`
replaces messages after the fact, such as `len - 1..` with nothing to
rewind, or `conversation.entry(i)` passed back changed to edit one in
place, and `commit.retire(agent)` despawns an agent for good, so a
restart brings back neither it nor the agents it spawned. Every message
has a `MessageId` (`conversation.id(i)`, `index_of(id)`) that a splice
keeps, so remember messages by id, not by index.
`rig_ecs::journal::read_session(store)` reads a session's logs, another
session's too, without restoring or writing anything: each agent's
conversation as it is and its saved components. `fork_session(from, to,
before)` copies a session as it was when a message was made, `before`
being its agent's id and the message's id: every log up to the records
made before that message's first record, in the session's order of
records, not by time.

# Time

Time is Bevy's `Time`. `.run_if(on_real_timer(Duration))` runs a system
once per interval, and a one-off wait is a delayed command,
`commands.delayed().duration(Duration)`. The loop sleeps when nothing
happens and wakes on its own for a delayed command; for an interval, an
entity with a `KeepAwake(Duration)` keeps it running at that pace while it
lives. None needs a thread. Bevy's clock stops while the machine sleeps; a
time of day is a `WakeAt(SystemTime)` on an entity, which gets `Due` in
`First` once that time passed (the loop wakes for it), and is armed again
by inserting a new `WakeAt`. A delayed command cannot be taken back, so
a wait that may be put off or called off, such as a debounce, is a
`WakeAt::after(wait)` on an entity of the plugin's: inserting a new one
puts it off, and removing it or despawning the entity calls it off.

```rust,no_run
use rig_harness::prelude::*;

/// How often a running turn says it is still running.
const EVERY: Duration = Duration::from_secs(60);

#[derive(Default)]
pub struct StillWorkingPlugin;

impl Plugin for StillWorkingPlugin {
    fn build(&self, app: &mut App) {
        app.add_observer(keep_awake)
            .add_observer(nightly)
            .add_systems(Update, still_working.run_if(on_real_timer(EVERY)))
            .add_systems(Startup, |mut commands: Commands| {
                commands.spawn(WakeAt::after(Duration::from_secs(24 * 3600)));
            });
    }
}

/// A turn keeps the loop awake while it runs; the `KeepAwake` goes with
/// the turn's entity.
fn keep_awake(turn: On<Add<TurnOf>>, mut commands: Commands) {
    commands.entity(turn.entity).insert(KeepAwake(EVERY));
}

fn still_working(turns: Query<&TurnOf>, mut notices: MessageWriter<Notice>) {
    for TurnOf(agent) in &turns {
        notices.write(Notice::info(*agent, "Still working."));
    }
}

/// Once a day, even across a suspend; then the next day.
fn nightly(due: On<Add<Due>>, mut commands: Commands, mut notices: MessageWriter<Notice>) {
    notices.write(Notice::info(None, "A day has passed."));
    commands
        .entity(due.entity)
        .insert(WakeAt::after(Duration::from_secs(24 * 3600)));
}
```

# The process: project, files and child processes

`ProjectRoot` is the directory the session works in. `RigHome` is the
`RIG_HOME` layout (`home.0.session(..)`, `home.0.auth(..)`). `Processes`
is how every child process starts, a rig-tools `Launcher` (a wrapper
command, a directory, environment): the shell tool reads it for each
command, so a sandbox plugin that sets it (`processes.set(..)`) wraps
every process from then on, and a plugin that starts processes builds
them with `processes.launcher().command(program)`. An entity with a
`FileWatch` gets `FileChanges` messages: first one listing the files
under its path when it was armed (read them then, so none made in between
is missed), then one about the files changed under it. `TokioRuntime` is rig's own tokio runtime, for a library that needs
tokio. A plugin that should also run in an app without the harness's
plugins inits the ones it reads (`app.init_resource::<ProjectRoot>()`).
A long-lived source, such as a socket, a server's notifications or a
thread's channel, is a `Running::stream(pool, &wake, items)` on an
entity: each item is inserted on it as a `Done<T>`, in order, and wakes
the loop, and the `Running` goes once the stream ended. Several items may
come in one frame, so the reader observes `Insert<Done<T>>`.

```rust,no_run
use rig_harness::prelude::*;

/// The templates in `.rig/templates`, watched.
#[derive(Component)]
struct Templates;

fn watch(mut commands: Commands, root: Res<ProjectRoot>) {
    commands.spawn((Templates, FileWatch::tree(root.0.join(".rig/templates"))));
}

fn reload(
    mut changes: MessageReader<FileChanges>,
    templates: Query<(), With<Templates>>,
    mut notices: MessageWriter<Notice>,
) {
    for change in changes.read().filter(|change| templates.contains(change.watch)) {
        notices.write(Notice::info(None, format!("{} templates changed", change.paths.len())));
    }
}

/// Runs every child process under `nice`.
fn be_nice(processes: Res<Processes>) {
    processes.set(processes.launcher().wrapped(["nice", "-n", "10"]));
}

/// The lines a chat bridge receives, here a fixed list.
#[derive(Component)]
struct Received;

fn listen(mut commands: Commands, wake: Res<Wake>) {
    let pool = bevy_tasks::IoTaskPool::get_or_init(bevy_tasks::TaskPool::default);
    let lines = ["hello", "are you there?"].map(str::to_owned);
    let lines = bevy_tasks::futures_lite::stream::iter(lines);
    commands.spawn((Received, Running::stream(pool, &wake, lines)));
}

/// Shows each line as it comes.
fn show(
    line: On<Insert<Done<String>>>,
    lines: Query<&Done<String>, With<Received>>,
    mut notices: MessageWriter<Notice>,
) {
    if let Ok(Done(text)) = lines.get(line.entity) {
        notices.write(Notice::info(None, format!("Received: {text}")));
    }
}
```

# Sessions and exiting

A plugin that ends the run triggers `Exit(reason)`: `ExitReason::Quit`,
`Reload` (the launcher restarts the session on its staged build),
`Switch(Some(id))` or `Switch(None)` (the launcher runs another session,
or a new one). The harness writes the exit code and tells the launcher
what to run; any exit's reason is the `ExitReason` resource from then on,
for code on the way out in Bevy's `OnAppExitSystems` (a background job
kept across a reload, stopped on a quit). `ForkSession { before }` copies
this session into a new one (`SessionForked` names it; the copy names
its parent and where it was cut, with the text the user wrote in the
message it stopped before, `ForkedFrom::of(&dir)`, so a view of a
session's family reads no other session's log), and
`DeleteSession(id)` removes another (`SessionDeleted`), for plugins that
keep files of their own per session.

```rust,no_run
use rig_harness::prelude::*;

/// `/fork`: continue in a copy of this session.
fn fork(_: In<CommandArgs>, mut commands: Commands) {
    commands.trigger(ForkSession::default());
}

/// Switches to each fork once it is written.
fn switch(mut forked: MessageReader<SessionForked>, mut commands: Commands) {
    for fork in forked.read() {
        commands.trigger(Exit(ExitReason::Switch(Some(fork.to.clone()))));
    }
}

fn build(app: &mut App) {
    app.add_command("fork", "Continue in a copy of this session", fork)
        .add_systems(Update, switch);
}
```

# Adding to another plugin's lists

A plugin that owns a list builds it from a query over entities other
plugins spawn, the way the system prompt is built from `PromptSection`s:
spawn one with your text and keep its text current (`set_if_neq` when it
is a field you change in place). An empty text, or Bevy's `Disabled`,
leaves it out, and despawning it takes it back. Never replace another
plugin's command to add a line to it.

| List | Spawn | Crate |
|---|---|---|
| `/model` and its picker: a choice that is no catalog model, run by name | `ModelItem::new(name, label, command)`; an agent's `ModelLabel` says what its model is in the status line | `rig_models` |
| `/resume`: a part of this session's line, kept with it | `ResumeLabel::new(tag, text)` | `rig_sessions` |
| `/context`: what you put in the context besides prompt sections | `ContextEntry::new(tag, text)` | `rig_basics` |
| `/agents`: a column of one agent's line | `(AgentColumn::new(tag, text), ColumnOf(agent))`, despawned with the agent | `rig_basics` |
| `/usage`: a line after the usage | `UsageRow::new(tag, text)` | `rig_telemetry` |
| `--print`: text printed after the answer | `PrintSection::new(tag, text)` | `rig_print` |

Models a provider plugin connects are catalog models: it declares its
vendor and adds its models to the catalog in `Models` (rig-core's
`ProviderId::declare`, `Connector::provide` and `catalog_mut`), so
`--model`, `/model`, its picker, the remembered default and subagents
take them like any other. A plugin may add them at any time, such as once
it found a local server: the frame after `Models` changed, every agent
whose chosen model was not usable, such as a restored choice, connects to
it, with no need to choose it again.

Spending is data too. rig-telemetry adds each settled model call's
reported usage to its agent's `Spending`, to the session's
`SessionSpending` (both saved with the session) and to its turn's
`TurnSpending`; a `Spending` keeps the total and its split per model and
per purpose (`turn`, `compaction`, a plugin's `Effect::with_purpose`).
A budget reads it; a plugin that pays for work that is no model request
`record`s it in both.

A `--print` run has nobody to answer questions: it is no `Answerer`, so a
plugin applies its policy for a run nobody answers, and a question asked
anyway is refused with why on stderr. `--answer=ID[,ID…]` answers each one
with the first of those choices it offers.

```rust,no_run
use rig_harness::prelude::*;
use rig_print::PrintSection;
use rig_telemetry::{SessionSpending, UsageRow};

/// What the session may spend, in USD.
const CAP: f64 = 5.0;

/// Refuses the turns' model calls once the session spent `CAP`, and says
/// what is left in `/usage` and after a `--print` answer.
#[derive(Default)]
pub struct BudgetPlugin;

impl Plugin for BudgetPlugin {
    fn build(&self, app: &mut App) {
        app.world_mut()
            .spawn((UsageRow::new("budget", ""), PrintSection::new("budget", "")));
        app.add_systems(AgentPass, refuse_over_budget.in_set(Stage::Gate))
            .add_systems(
                Update,
                show_budget.run_if(resource_exists_and_changed::<SessionSpending>),
            );
    }
}

fn refuse_over_budget(
    mut calls: Query<&mut Decision, (Undecided, ModelCalls, With<TurnCall>)>,
    spent: Option<Res<SessionSpending>>,
) {
    if spent.is_none_or(|spent| spent.0.total.cost < CAP) {
        return;
    }
    for mut decision in &mut calls {
        let why = format!("The session spent its ${CAP:.2} budget.");
        decision.decide(Outcome::refused(why));
    }
}

fn show_budget(
    spent: Res<SessionSpending>,
    mut shown: Query<(&mut UsageRow, &mut PrintSection)>,
) {
    let left = (CAP - spent.0.total.cost).max(0.0);
    let text = format!("Budget: ${left:.2} of ${CAP:.2} left.");
    for (mut row, mut section) in &mut shown {
        row.text.clone_from(&text);
        section.text.clone_from(&text);
    }
}
```

A choice in `/model` that is no catalog model, and what an agent on it
shows:

```rust,no_run
use rig_basics::{AgentColumn, ColumnOf};
use rig_harness::prelude::*;
use rig_models::{ModelItem, ModelLabel};

/// `/model router` (or its picker item) runs `/route on`.
#[derive(Default)]
pub struct RouterPlugin;

impl Plugin for RouterPlugin {
    fn build(&self, app: &mut App) {
        let item = ModelItem::new("router", "router  the cheapest model that fits", "route on");
        app.world_mut().spawn(item);
        app.add_command("route", "Pick the model per request: /route on", route);
    }
}

fn route(In(args): In<CommandArgs>, mut commands: Commands) {
    if args.args == "on" {
        let router = ModelLabel("router".to_owned());
        commands.entity(args.agent).insert(router);
        commands.spawn((AgentColumn::new("route", "routed"), ColumnOf(args.agent)));
    }
}
```

# Turning off or replacing what another plugin added

Tools, slash commands, tool renderers and prompt sections are entities.
Bevy's `Disabled` on one turns it off: no query finds it, so agents are
not offered the tool, the command is unknown, the section is left out and
the tool's calls are drawn plainly, and its name is free for a
replacement (a second tool, command or renderer of a taken name is
refused). Do it in `Plugin::finish`, which runs once every plugin's
`build` did, so what they added exists whatever their order. What a
plugin spawns in `build`, `finish` and `cleanup` is marked `ProvidedBy`
it; what it spawns later it marks itself, with its entity from
`Res<PluginEntity<MyPlugin>>`. Changing how another plugin's tool runs is
rarely a replacement: a Prepare or Gate system on its calls does it
without taking the tool over, a Review system changes its results, and a
default tool may have a switch of its own, such as
`rig_coding_tools::ShellOutput`, whose `set(None)` lets `shell` give the
model all of a command's output, for a plugin that cuts it itself.

```rust,no_run
use rig_harness::prelude::*;

#[derive(Default)]
pub struct NoShellPlugin;

impl Plugin for NoShellPlugin {
    fn build(&self, _app: &mut App) {}

    fn finish(&self, app: &mut App) {
        let world = app.world_mut();
        let mut tools = world.query::<(Entity, &ToolDef)>();
        let shell = tools.iter(world).find(|(_, def)| def.0.name.as_str() == "shell");
        if let Some((shell, _)) = shell {
            world.entity_mut(shell).insert(Disabled);
        }
        // `shell` is free again: `app.add_tool(MyShell)` would replace it.
        // A command is found by its `Name`, such as "/help", with
        // `SlashCommand`; a renderer by its `Name`, such as
        // "renderer:shell"; a section by its `PromptSection`'s `tag`.
    }
}
```

What a tool tells the model is data on its entity too, and another plugin
may change it: its `ToolDef` (the rig-core `ToolDefinition`: name,
description, the JSON schema of its `parameters`) and its `ToolRules`
(lines of the system prompt of every agent it is offered to). Change them
in `finish`, or at any time: each request is built from them when it is
admitted, so the next one has the change. A property added to another
tool's `parameters` is one the model may send, since the kernel refuses
only arguments the schema does not declare; the tool still parses its
arguments its own way, so the plugin that added the property reads it,
and takes it out if the tool would refuse it, from the call's
`ToolRequest` in Prepare (`task` hands such arguments on instead:
[extending `task`](#extending-task)). Edit lines of `ToolRules` you added, found by
their text, and leave the others' alone: two plugins editing the same
schema or rules must order their systems, as for any shared data.

```rust,no_run
use rig_harness::prelude::*;

/// On a call whose model asked for it to be quick.
#[derive(Component)]
struct Quick;

/// Lets the model ask any `shell` call to be quick, read in Prepare.
fn add_quick(world: &mut World) {
    let mut tools = world.query::<(&mut ToolDef, &mut ToolRules)>();
    for (mut def, mut rules) in tools.iter_mut(world) {
        if def.0.name.as_str() != "shell" {
            continue;
        }
        if let Some(properties) = def.0.parameters.get_mut("properties")
            && let Some(properties) = properties.as_object_mut()
        {
            let quick = serde_json::json!({ "type": "boolean", "description": "Stop after 10 s." });
            properties.insert("quick".to_owned(), quick);
        }
        rules.0.push("Ask `shell` to be quick when a command may hang.".to_owned());
    }
}

/// Takes `quick` out of each `shell` call, which would refuse it, and
/// marks the call.
fn take_quick(
    mut calls: Query<(Entity, &mut ToolRequest), Added<Admitted>>,
    mut commands: Commands,
) {
    for (call, mut request) in &mut calls {
        if request.call.function.name.as_str() != "shell" {
            continue;
        }
        let quick = request.call.function.arguments.remove("quick");
        if quick.and_then(|quick| quick.as_bool()) == Some(true) {
            commands.entity(call).insert(Quick);
        }
    }
}

fn build(app: &mut App) {
    // In `finish`, so `shell` exists whatever the plugins' order.
    add_quick(app.world_mut());
    app.add_systems(AgentPass, take_quick.in_set(Stage::Prepare));
}
```

# Testing a plugin

A plugin's test builds a small app: the agent core (`AgentPlugin`,
`JournalPlugin`), the plugin, and what it needs, run frame by frame with
scripted models. The rig repository's `rig-harness-test-support` crate
(`test-support/rig-harness-test-support` in the rig checkout, a path
dev-dependency) has the parts every plugin test needs:

- **Apps.** `app_in(&home, &id)` is an app on a session directory of
  `scratch_home(name)`, as the harness keeps one, so `SessionPaths`,
  forks and the effect log (`rig_telemetry::EffectLogPlugin`, read back
  with `EffectLog`) work; its failed assertions still print. `app_on(&store)`
  keeps the session in memory instead. Both have no user settings and
  return the loop's wakes.
- **Models.** `scripted_models([("vendor/name", &model)])` is a catalog
  whose `vendor/name` is answered by rig-core's `MockCompletionModel`;
  `reply(text)` and `calls(id, tool, args)` script its replies, and
  `connect(&model)` gives one as the catalog's DeepSeek model.
- **Running.** `run_until(&mut app, &wakes, done)` runs frames until
  `done(world)` or a deadline. `advance(&mut app, by)` runs one frame `by`
  later on Bevy's clock, for timers and delayed commands, and
  `pass_wall_clock(world, by)` makes the `WakeAt`s that would have passed
  due. `first_agent(world)` finds the agent; `start_call` spawns a tool
  call as a model would.
- **What happened.** `collect::<M>(&mut app)` keeps every message `M`
  (`Notice`, `Recorded`, `StreamDelta`) in `Collected<M>`, read in `Last`;
  the observers `log_ended` and `log_tool_outputs` keep how each turn
  ended (`Ended`) and each tool call's output by call id (`ToolOutputs`).
- **Questions.** The `AnswerAsks` front answers every question from an
  `Answers` script; `answer_open(world, title, &answer)` answers one
  later, as the user would.

```rust,no_run
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::SessionId;
use rig_harness::prelude::*;
use rig_harness_test_support::{
    Ended, ToolOutputs, app_in, calls, first_agent, log_ended, log_tool_outputs, reply,
    run_until, scratch_home, scripted_models,
};

/// The plugin under test: refuses a `write` outside `src/`.
#[derive(Default)]
struct SrcOnlyPlugin;

impl Plugin for SrcOnlyPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(AgentPass, src_only.in_set(Stage::Gate));
    }
}

fn src_only(mut calls: Query<(&ToolRequest, &mut Decision), (Undecided, ToolCalls)>) {
    for (request, mut decision) in &mut calls {
        let path = request.call.function.arguments["path"].as_str().unwrap_or_default();
        if request.call.function.name.as_str() == "write" && !path.starts_with("src/") {
            decision.decide(Outcome::refused("write only under src/"));
        }
    }
}

/// A `write` that only says it wrote.
fn wrote(called: On<ToolCalled<serde_json::Value>>, mut calls: Query<&mut Decision, ToolCalls>) {
    if let Ok(mut call) = calls.get_mut(called.call) {
        call.decide(ToolResult::success("written".into()));
    }
}

#[test]
fn a_write_outside_src_is_refused() {
    let model = MockCompletionModel::from_stream_turns([
        calls("c1", "write", serde_json::json!({ "path": "README.md" })),
        reply("done"),
    ]);
    let home = scratch_home("src-only");
    let (mut app, wakes) = app_in(&home, &SessionId::generate());
    app.insert_resource(scripted_models([("test/model", &model)]))
        .add_plugins((AgentPlugin, JournalPlugin, SrcOnlyPlugin))
        .add_open_tool("write", "Write a file.", ToolOptions::default(), wrote)
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let model = "test/model".to_owned();
    app.world_mut().trigger(SetModel { entity: agent, model });
    app.world_mut().trigger(Deliver::new(agent, "Fix the readme.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| !world.resource::<Ended>().0.is_empty());
    let outputs = &app.world().resource::<ToolOutputs>().0;
    let output = outputs.get("c1").map(String::as_str).unwrap_or_default();
    assert!(output.contains("only under src/"), "{output}");
}
```

# A front

A front shows the agents to someone and takes what they type: the
terminal view (`rig-tui`), `--print` (`rig-print`), a window, a web
server, a chat bot. It offers itself in `build` with `app.add_front(..)`,
saying which runs it takes (`on_terminal()`, `headless()`, or
`on_flag(name, help)` for a run given its option), and one front takes
each run: the one whose option was given, else the first offered for the
kind of run. It asks `app.runs_front(name)` in `finish` and sets itself
up only then. It sends what the user types as `Deliver` from
`Origin::user()`, shows `Notice`s and each agent's `Conversation` (or
each change as `Committed`) and its reply as it streams
([`StreamDelta`](#what-a-model-streams)), and sets `Shown` to the agent
the user looks at. If it can show questions, it spawns an `Answerer` and
answers [`Ask`s and `Picker`s](#asking-the-user-questions-and-pickers)
with `Pick`. A front nobody sits at drives its run with
`app.run_headless(prompt)` and writes out `HeadlessEnded`; one whose runs
leave nothing to resume is `sessionless()`. Several fronts may run side
by side (a web server beside the terminal view); only the one that takes
the run owns stdin and stdout.

When the user stops a turn before it answered, `Committed::Halt { agent,
reason }` says so with `Some(Halt::Stopped)` (`rig_ecs::agent::Halt`;
`Kept` when the message was left for the next turn, such as after its
turn failed, and `None` when it was asked for again): a front shows a stopped request from that, never by reading
text. For the model, the kernel puts the `STOPPED` text between the
stopped request and the next message that joins it; a view that draws a
whole conversation, such as a restored one, recognizes that item by the
constant `rig_ecs::agent::STOPPED`, not by its words.

```rust,no_run
use rig_harness::prelude::*;
use rig_harness::rig_ecs::agent::Halt;

#[derive(Default)]
pub struct JsonFrontPlugin;

impl Plugin for JsonFrontPlugin {
    fn build(&self, app: &mut App) {
        app.add_front(Front::new("json").on_flag("json", "Print every message as a JSON line"));
    }

    fn finish(&self, app: &mut App) {
        if app.runs_front("json") {
            app.world_mut().spawn(Answerer::default());
            app.add_systems(Update, (print_messages, answer_yes));
        }
    }
}

fn print_messages(mut committed: MessageReader<Committed>) {
    for change in committed.read() {
        match change {
            Committed::Message { message, .. } => {
                println!("{}", serde_json::to_string(message).unwrap_or_default());
            }
            Committed::Halt {
                reason: Some(Halt::Stopped),
                ..
            } => println!(r#"{{"stopped":true}}"#),
            _ => {}
        }
    }
}

/// Answers every question and picker with its first choice, as `--yes`
/// would.
fn answer_yes(
    asks: Query<Entity, (With<Ask>, Without<Outcome>)>,
    pickers: Query<Entity, With<Picker>>,
    mut commands: Commands,
) {
    for entity in asks.iter().chain(&pickers) {
        commands.trigger(Pick { entity, chosen: Some(Chosen::Item(0)) });
    }
}
```

# The status line

The terminal view's row under the transcript shows the `StatusItems` of
the agent shown and of its running turn, which every agent and turn has,
and the app's `AppStatus`, all `rig_tui`'s. A plugin shows an item there in
a system in `StatusSystems` that runs when what it shows changed:
`items.show(item)` puts the item at its place (its side and order) in
place of the one there, and an empty item clears the place. A plugin that
should also build without the terminal view puts this behind a default
`tui` cargo feature, as the default plugins do, and one that writes
`AppStatus` registers it with `app.init_resource::<AppStatus>()`, since the
view may not be added. The
default plugins' places, and how long each item stays when the line is too
narrow (`keep`: the lowest goes first, `u8::MAX` never):

| side | order | item | keep | plugin |
|---|---|---|---|---|
| left | 10 | the session's name (`AppStatus`) | 2 | rig-sessions |
| left | 20 | `⤷` a spawned agent's name | 14 | rig-basics |
| left | 30 | the model | always | rig-models |
| left | 40 | the reasoning setting | 12 | rig-models |
| left | 50 | the status: idle, thinking, … | always | rig-activity |
| left | 60 | the agent's subagents at work | 15 | rig-basics |
| left | 70 | the other agents at work | 13 | rig-basics |
| left | 80 | what the turn spent (on the turn) | 11 | rig-telemetry |
| left | 90 | the rebuild of `/reload` (`AppStatus`) | 16 | rig-reload |
| right | 10 | tokens in and out | 4 | rig-telemetry |
| right | 20 | cached input | 1 | rig-telemetry |
| right | 30 | cost | 3 | rig-telemetry |
| right | 40 | the context in use | 5 | rig-telemetry |

```rust,no_run
use rig_harness::prelude::*;
use rig_tui::{Side, StatusItem, StatusItems, StatusSystems, Tone};

/// After the status; gone before the reasoning setting.
const MESSAGES: StatusItem = StatusItem::at(Side::Left, 55, 10);

#[derive(Default)]
pub struct MessagesItemPlugin;

impl Plugin for MessagesItemPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PostUpdate, show.in_set(StatusSystems));
    }
}

fn show(mut agents: Query<(&Conversation, &mut StatusItems), Changed<Conversation>>) {
    for (conversation, mut items) in &mut agents {
        let text = format!("{} messages", conversation.messages().len());
        items.show(MESSAGES.says(text, Tone::Dim));
    }
}
```

# What agents do and say, in a terminal panel

Each agent is an entity with an `Agent`, a `Name`, an `AgentId` and a
`Conversation`; a subagent is `SpawnedBy` its parent, which lists it in
`Spawned`. The activity plugin's `rig_activity::Activity` says what an
agent does now (its status and running tools), and its
`MessageFeed` resource holds the latest deliveries. `MessageReader<Committed>` sees every change
of every conversation as the session log records it, and `On<Add<Settled>>`
on a turn with a `TurnOutcome` each turn's end ([How a turn ends](#how-a-turn-ends)).

A `TuiPanel` entity takes a side of the transcript (`Top`, `Bottom`,
`Left`, `Right`, with a ratatui `Constraint`) or a box over the screen
(`Over`). The plugin's own system draws it into its `PanelCanvas` in
`TuiSystems::Draw`, which runs only for a frame that is drawn. Frames are
drawn when agents or panels change; a plugin whose own state changed
writes a `RequestRedraw`. `TuiScreen` is the terminal's size and `Focused`
marks the agent shown. These are `rig_tui`'s, so the crate depends on
`rig-tui` and `rig-activity` beside `rig-harness`. They are there in any
run, `--print` too, which draws no frame.

```rust,no_run
use std::collections::HashMap;

use rig_activity::Activity;
use rig_harness::prelude::*;
use rig_tui::ratatui::layout::Constraint;
use rig_tui::ratatui::widgets::{Block, Paragraph};
use rig_tui::{PanelCanvas, Placement, TuiPanel, TuiSystems};

#[derive(Default)]
pub struct AgentsPanelPlugin;

#[derive(Component)]
struct AgentsPanel;

/// The words of each agent's answers since the agent started.
#[derive(Resource, Default)]
struct Written(HashMap<Entity, usize>);

impl Plugin for AgentsPanelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Written>()
            .add_systems(Startup, spawn)
            .add_systems(Update, tally)
            .add_systems(PostUpdate, draw.in_set(TuiSystems::Draw));
    }
}

fn spawn(mut commands: Commands) {
    commands.spawn((AgentsPanel, TuiPanel::new(Placement::Right(Constraint::Length(32)))));
}

fn tally(mut committed: MessageReader<Committed>, mut written: ResMut<Written>) {
    for change in committed.read() {
        if let Committed::Message { agent, message, .. } = change
            && let Some(answer) = final_answer(message)
        {
            *written.0.entry(*agent).or_default() += answer.split_whitespace().count();
        }
    }
}

fn draw(
    mut panels: Query<&mut PanelCanvas, With<AgentsPanel>>,
    agents: Query<(Entity, &Name, &Activity, Has<SpawnedBy>)>,
    written: Res<Written>,
) {
    let lines: Vec<String> = agents
        .iter()
        .map(|(agent, name, activity, sub)| {
            let indent = if sub { "  " } else { "" };
            let words = written.0.get(&agent).copied().unwrap_or_default();
            format!("{indent}{name} {}, {words} words", activity.status)
        })
        .collect();
    for mut canvas in &mut panels {
        canvas.render(Paragraph::new(lines.join("\n")).block(Block::bordered().title("agents")));
    }
}
```

# Keys and the input in the terminal view

The terminal view (`rig_tui`) takes part in the input as data a plugin
reads and changes, as it does for panels:

- **Keys.** Every key the view acts on is a `KeyBinding` entity, its own
  editing keys included, in the input's `KeyScope`. A plugin binds a key
  or a sequence (`ctrl+x ctrl+e`) with `app.add_key_binding(binding,
  observer)`: the observer hears `KeyPressed` on the binding (and, for one
  `with_releases()`, the key let go), and `run_command("compact")` is an
  observer that runs a slash command; it returns the binding's entity. As
  with commands, the first binding of a key stays: a binding that another
  of an overlapping scope takes, starts, or is started by, is refused with
  a warning, and `Disabled` on a binding frees its keys, in
  `Plugin::finish`. A global binding shadows the input's binding of the
  same keys instead. `Keys` parse from text (`"alt+m".parse()`), for keys
  from the user's settings.
- **A panel's keys.** `KeyFocus(Some(panel))` gives a `TuiPanel` the keys:
  its bindings in `KeyScope::Panel(panel)` and the global ones apply, not
  the input's, and Esc it does not bind, or the panel going, gives the
  keys back. While it has them, a key it does not bind edits nothing, so
  a panel with a text field of its own, such as a palette's filter, reads
  the typed keys in order from `TerminalEvent` (below).
- **Completion.** An `ArgCompletion::new("/model").candidate(text,
  detail)` entity offers candidates for a command's argument, which the
  input completes as it does commands and `@` paths.
- **The input.** `TuiInput` is the draft: `text()`, `cursor()`, `set`,
  `insert`, `take`, `set_cursor`; `is_changed()` says it was edited. Enter
  sends it to the agent shown (`Focused`).
- **The terminal.** `TerminalEvent` messages are what the terminal
  reported, keys and pastes; a plugin may read them, and write some.
  `TerminalFocus` says whether the terminal has the user's focus. A
  `TerminalWrite(bytes)` message goes to the terminal between frames,
  such as a notification's escape code, and `CopyToClipboard(text)` puts
  text on the clipboard (OSC 52). A `RunInTerminal(command)` entity runs
  a program with the terminal (the view steps aside and draws again
  after), and gets its `TerminalExit`.
- **Pickers and questions.** The view shows each
  [`Picker` and open `Ask`](#asking-the-user-questions-and-pickers) over
  the transcript, one at a time in the order they came, and answers it
  with `Pick`.

These are there in any run, and inert without the terminal view's run.
This plugin writes the draft in `$EDITOR` on Ctrl+E, which shadows the
view's end of the line, and offers a few prompts on Ctrl+T:

```rust,no_run
use std::path::PathBuf;
use std::process::Command;

use rig_harness::prelude::*;
use rig_tui::{
    AppKeyBindingsExt, KeyBinding, KeyPress, KeyPressed, RunInTerminal, TerminalExit, TuiInput,
};

#[derive(Default)]
pub struct DraftKeysPlugin;

/// A program writing the draft in this file.
#[derive(Component)]
struct Draft(PathBuf);

const PROMPTS: [&str; 2] = ["Explain this code: ", "Write a test for "];

impl Plugin for DraftKeysPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, take_draft);
        app.add_key_binding(KeyBinding::new(KeyPress::ctrl('e'), "Write the draft in $EDITOR"), edit);
        app.add_key_binding(KeyBinding::new(KeyPress::ctrl('t'), "Start from a prompt"), prompts);
    }
}

fn edit(_: On<KeyPressed>, input: Res<TuiInput>, mut commands: Commands) {
    let path = std::env::temp_dir().join("rig-draft.md");
    if std::fs::write(&path, input.text()).is_err() {
        return;
    }
    let mut editor = Command::new(std::env::var("EDITOR").unwrap_or_else(|_| "vi".to_owned()));
    editor.arg(&path);
    commands.spawn((RunInTerminal(editor), Draft(path)));
}

/// The draft as the editor left it, unless the editor failed.
fn take_draft(
    done: Query<(Entity, &TerminalExit, &Draft), Added<TerminalExit>>,
    mut input: ResMut<TuiInput>,
    mut commands: Commands,
) {
    for (run, exit, Draft(path)) in &done {
        if let (true, Ok(text)) = (exit.success(), std::fs::read_to_string(path)) {
            input.set(text.trim_end());
        }
        commands.entity(run).despawn();
    }
}

fn prompts(_: On<KeyPressed>, mut commands: Commands) {
    let picker = PROMPTS
        .iter()
        .fold(Picker::new("Prompt"), |picker, prompt| picker.item(*prompt, ""));
    commands.spawn(picker).observe(
        |picked: On<Picked>, mut input: ResMut<TuiInput>| {
            if let Some(Chosen::Item(index)) = picked.chosen
                && let Some(prompt) = PROMPTS.get(index)
            {
                input.set_cursor(0);
                input.insert(prompt);
            }
        },
    );
}
```

# A window

`rig_harness::windowed(DefaultPlugins)` is Bevy's `DefaultPlugins`
without the log, task pools, clock, signal handler and loop the agent
already has; its winit plugin then runs the app's loop. In `finish`, where
winit's event loop exists, the plugin points the agent's `Wake` at it, so
agent activity and terminal input wake the window instead of a poll. The
crate depends on `bevy` at exactly the agent's version, with the features
it draws with, such as
`bevy = { version = "=0.20.0", default-features = false, features = ["ui"] }`
for Bevy UI. A window beside the terminal view is no front; a window that
is the front offers itself with `on_flag` (`--gui`). (This example is not
built with the agent's docs, which have no window.)

```rust,ignore
use std::time::Duration;

use bevy::prelude::*;
use bevy::window::ExitCondition;
use bevy::winit::{EventLoopProxyWrapper, UpdateMode, WinitSettings, WinitUserEvent};
use rig_activity::Activity;
use rig_harness::prelude::{Invoked, Wake};

#[derive(Default)]
pub struct DashboardPlugin;

impl Plugin for DashboardPlugin {
    fn build(&self, app: &mut App) {
        // A window beside the terminal view only.
        if !app.world().get_resource::<Invoked>().is_some_and(Invoked::interactive) {
            return;
        }
        // Closing the window leaves the agent running.
        let window = WindowPlugin {
            exit_condition: ExitCondition::DontExit,
            ..default()
        };
        // Frames when woken, and at least every second as without a window.
        let mode = UpdateMode::reactive_low_power(Duration::from_secs(1));
        app.add_plugins(rig_harness::windowed(DefaultPlugins.set(window)))
            .insert_resource(WinitSettings { focused_mode: mode, unfocused_mode: mode })
            .add_systems(Update, show.run_if(any_changed_activity));
    }

    fn finish(&self, app: &mut App) {
        if let Some(proxy) = app.world().get_resource::<EventLoopProxyWrapper>() {
            let proxy = (**proxy).clone();
            app.insert_resource(Wake::new(move || {
                proxy.send_event(WinitUserEvent::WakeUp).ok();
            }));
        }
    }
}

fn any_changed_activity(agents: Query<(), Changed<Activity>>) -> bool {
    !agents.is_empty()
}

fn show(agents: Query<&Activity>) {
    // Update the window's UI entities from the agents' activity here.
    let _busy = agents.iter().filter(|activity| activity.is_busy()).count();
}
```
