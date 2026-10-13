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
- [The effect log](#the-effect-log-reading-redacting-replaying): reading, redacting, replaying; [settings](#settings)
- [Saved state](#saved-state-counting-every-tool-call), [changing history](#changing-history), [time](#time), [the workspace, files and child processes](#the-workspace-files-and-child-processes), [sessions: their store, secrets and exiting](#sessions-their-store-secrets-and-exiting)
- [Adding to another plugin's lists](#adding-to-another-plugins-lists): `/model`, `/resume`, `/context`, `/agents`, `/usage`, `--print`; spending and budgets
- [Turning off or replacing what another plugin added](#turning-off-or-replacing-what-another-plugin-added), and [testing a plugin](#testing-a-plugin)
- [Fronts, options and commands](#fronts-options-and-commands), with [commands of your own](#commands-of-your-own); [the transcript](#the-transcript-notes-looks-and-scrolling), [the status line](#the-status-line), [a terminal panel](#what-agents-do-and-say-in-a-terminal-panel), [keys and the input](#keys-and-the-input-in-the-terminal-view), [a window](#a-window)

A plugin is a Bevy plugin: a type implementing `Plugin + Default` in a
crate that depends on `rig-harness`. The agent is a Bevy app, and its
tools, commands, subagents, compaction, sessions and terminal view are
plugins of the same kind, listed in `RIG_HOME/plugins.toml` beside any
other and built only on the same public API. One that needs another adds
it unless `app.is_plugin_added::<P>()`.

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

Each plugin crate is its own workspace in `RIG_HOME/plugins/<name>`,
depending on `rig-harness` and the rig plugin crates it uses (`rig-tui`
for a terminal panel) by version, and on `bevy_ecs` and `bevy_reflect` (or
`bevy`) at the agent's exact version. The agent builds from a rig checkout
(`RIG_SOURCE`), which a `[patch.crates-io]` table points at.

`/reload` in the agent, or its `reload` tool, rebuilds and restarts in the
same session once no turn runs; a failed build leaves the running one, and
one that crashes at startup is rolled back.

# Finding names

The prelude holds Bevy's app, ECS, reflection and time preludes, the agent
runtime's components, stages and registries (`rig_harness::rig_ecs`), and
the harness's services. rig-core is `rig_harness::rig_core`, rig-memory
`rig_harness::rig_memory` and rig-tools `rig_harness::rig_tools`; each
default plugin's types are in its crate, such as the terminal view's in
`rig_tui` (with `rig_tui::ratatui` and `rig_tui::crossterm`). What every
plugin shares lives in rig-harness, never in another plugin; what hosts
without Bevy need too, in the libraries: rig-core's `redact`, schema check
and `conversation`, rig-memory's saving, rig-tools' tools and workspace.

Every one of these types is reflected, so the running agent lists them
with the `inspect` tool of the optional `rig-inspect` plugin: ask it
before reading rig's source. `registry.schema` gives a type's fields
(`type_limit: {with: ["Saved"]}`: what is saved), `world.query` with
`ProvidedBy` what each plugin added.

# How a plugin steps in: effects and stages

Every step of an agent's work is an entity, an **effect**, whose
components are its request and, once answered, its `Outcome`: a model call
(`ModelRequest`), a tool call (`ToolRequest`), a message to an agent
(`Incoming`), a question for the user (`Ask`), or a plugin's own
(`EffectRequest`). Each works `ForAgent` an agent (or for the app, without
one) and is recorded once it settles (`EffectRecorded`, the session's
effect log). Each frame, the `AgentLoop` system in `Update` runs the `AgentPass`
schedule, one pass through the stages in this order, again and again until
nothing moves, so an effect goes as far as it can in the frame:

| Stage | Who | What happens there |
|---|---|---|
| `Arrive` | kernel | finished work becomes an `Outcome`; a failed model call gets a `RetryVerdict` |
| `Review` | plugins | outcomes not settled yet (`Unreviewed`): read, edit in place, hold, or take a failure over; a turn's proposed end (phases: `ReviewSystems`) |
| `Commit` | kernel | settles what is answered and unheld (`Settled`), appends results, carries turns on or ends them |
| `Admit` | kernel | `Admitted` on every new effect; builds turn requests |
| `Prepare` | plugins | what was admitted this pass (`Added<Admitted>`): edit the request, once (phases: `PrepareSystems`) |
| `Gate` | plugins | as it opens, the kernel gives each tool call its `Footprint` and marks `Waiting` a call behind a conflicting one; then what is about to start (`Undecided`): hold it or decide it (`Decision`) |
| `Start` | kernel | applies the decisions as `Outcome`s, then starts what is admitted, unheld, unanswered and undecided |

**A hook is an ordinary system in a stage** that queries stage data:
`app.add_systems(AgentPass, my_gate.in_set(Stage::Gate))`, ordered by its
stage's phase and against other plugins' public sets
([ordering](#ordering-against-other-plugins)). Every effect is admitted
before it starts, wherever it was spawned, so Prepare sees each once; Gate
each pass it is ready to start; Review an outcome each pass until it settles.

**An observer is for reactions that need no order**: counting, a status
item, a span, the plugin's own entities (`On<Add<Started>>`,
`On<Add<Settled>>`, `On<Add<ToolRequest>>`). Observers of one event run in
no fixed order, so an observer never edits stage data another plugin
edits; it may decide an effect, since the first decision wins.

**The conflict rule.** A hold is any entity `HoldOf(effect)`, its `Name`
saying why (`waiting for lease on src/a.rs`) in the fronts. The first
decision wins: a gate decides through the effect's `Decision`, naming who
(`decision.decide(Decider::plugin(NAME), Outcome::refused(..))`, or
`Decider::Person` for the user's choice in a front), so no later gate
overturns a refusal; the kernel applies it as the `Outcome` at Start, and
the record keeps `decided_by`. Allowing is doing nothing. A reviewer edits
the `Outcome` in place (`Query<&mut Outcome, Unreviewed>`) once, with a
marker, since Review sees it each pass; the record keeps both. Plugins
never insert `Outcome` themselves.

**Deciding once.** A gate that only reads (a path rule) decides afresh
each time it sees a call. One whose decision asks someone or has side
effects marks the call it decided on (a `SparseSet` marker such as
`Approved`) and skips marked and held calls. A decision is applied at
Start, so until then the call is still `Undecided` for later gates: a gate
that holds or asks checks `decision.is_decided()` first.

**Failing closed.** A stage system that fails or panics closes its stage
for that pass, whatever the app's error handler: a failed Prepare system
answers what that pass admitted with an error, a failed Gate system starts
nothing, a failed Review system settles nothing, and a notice names it.

**Taking over and trying again.** A reviewer that takes a failure over
spawns a successor with the request it needs and the same `CallOf`, which
the failed effect is `SupersededBy`
(`commands.spawn(successor).add_related::<SupersededBy>(&[failed])`), or
sends the request again (`commands.queue(resend(failed))`): the failed
effect is recorded as superseded and never reaches the conversation. Any
number of effects may name one successor, as each loser of a race names
its winner. Work that takes time holds the outcome first.

**The loop's edge.** Work spawned, answered or released after `AgentLoop`
ran waits for the next frame, so a system that does so every frame runs in
`Update` `.before(AgentLoop)`. One that reacts to what the loop did that
frame (an agent it started, a call it settled) runs `.after(AgentLoop)`.

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
            decision.decide(Decider::plugin("src-only"), Outcome::refused("write only under src/"));
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
        let Ok(Answered::ToolResult { result }) = &outcome.0 else { continue };
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

Bevy reports two stage systems with no order between them that write
the same component, such as two gates that decide (`&mut Decision`):
which goes first would be chance. Debug builds log it, and a plugin's test
makes it a failure ([`assert_stages_ordered`](#testing-a-plugin)). Holds
and spawns go through `Commands` and need no order. Queries of different
kinds of effect never meet when they say so with a kind filter
(`ToolCalls`, `ModelCalls`, `IncomingMessages`, `PluginEffects`), and two
over plugin effects when one names its marker (`With<Ask>`) and the other
excludes it. Two queries of one system that write the same component must
rule each other out so, or Bevy refuses the system (error B0001).

**Prepare and Review have phases**, chained in this order, so a plugin
orders a system by what it does, not by which plugin does what:

| Phase | What goes there | The default plugins' systems there |
|---|---|---|
| `PrepareSystems::Input` | what came in, rewritten before anything is read from it: a message's text, a tool call's arguments | |
| `PrepareSystems::Compose` | the parts that come from what a request names and from its agent | the files a message names (`@path`, in `rig_coding_tools::AttachSystems`), the default system prompt, each coding tool call's handler for its place in the workspace |
| `PrepareSystems::Edit` | the request edited as it will be sent: prompt, messages, tools, options, the model it goes to, its `Adapters`, a call's `CallHandler` | |
| `PrepareSystems::Fit` | the request as sent, measured against the window of the model it goes to and fitted | compaction |
| `ReviewSystems::Edit` | an `Outcome`, a `RetryVerdict` or a proposed `TurnOutcome`, edited in place | |
| `ReviewSystems::Act` | what was reviewed, acted on: a failed call taken over, a result followed up | compaction's recovery from an overflow |

So:

- **A stage system that edits or answers data another plugin may also
  edit goes in a public `SystemSet` of its plugin**, configured in its
  phase: `app.configure_sets(AgentPass, MySystems.in_set(PrepareSystems::Edit))`.
- **Another plugin orders against the set only where it must go before
  or after that plugin**, never against a private system. A gate that
  holds or asks goes after the gates that decide the same kind of effect.
- **Two plugins of other authors never depend on each other only to order
  themselves.** Where two meant to be used together edit the same data in
  one phase, the later one depends on the earlier one's crate, orders
  against its set, and tests both, `assert_stages_ordered((Mine, Theirs))`.

The sets of Gate systems that edit shared data:

| Set | Crate | What it does | Order against it |
|---|---|---|---|
| `GateSystems::FromRecord` | `rig_ecs` | as Gate opens, before every Gate system: decides effects from a record of an earlier run (a session replay), which holds each gate's decision then | nothing; a cached or dry-run answer is no record and goes in `After` |
| `GateSystems::After` | `rig_ecs` | after every Gate system: answers that stand in for running an effect (a cached result, a dry run, an editor's buffer), and [group coordinators](#gating-and-reviewing-tool-calls); a gate's decision is applied at Start, so read `Decision` | nothing; every gate goes before it |
| `CommandSystems` | `rig_harness` | takes over a user's line that starts with `/` and runs it | `.before` it to take a line over first |
| `AskSystems` | `rig_harness` | answers the oldest open question that takes text with the user's next line, after `CommandSystems` | `.after` it to answer questions in Gate, so a line the user typed wins |
| `PrintSystems` | `rig_print` | answers a `--print` run's questions from `--answer`, after `AskSystems` | `.before` it to answer a question first |
| `SubagentSystems::Gate` | `rig_subagents` | refuses a `SpawnSubagent` deeper than the policy allows; stops the restored turn of an agent that owed reports | `.after` it to decide or hold a request it let through |

```rust,no_run
use rig_harness::prelude::*;
use rig_harness::rig_core::redact;

/// The redaction plugin's stage systems, for others to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RedactSystems;

#[derive(Default)]
pub struct RedactPlugin;

impl Plugin for RedactPlugin {
    fn build(&self, app: &mut App) {
        // After attachments and the default prompt, before compaction.
        app.configure_sets(AgentPass, RedactSystems.in_set(PrepareSystems::Edit))
            .add_systems(AgentPass, redact_messages.in_set(RedactSystems));
    }
}

/// Takes API keys out of each message as it comes in, attached files
/// included.
fn redact_messages(mut messages: Query<&mut Incoming, Added<Admitted>>) {
    for mut message in &mut messages {
        for part in &mut message.parts {
            let text = match part {
                MessagePart::Text(text) => text,
                MessagePart::Attachment(Attachment { content: UserContent::Text(it), .. }) => &mut it.text,
                MessagePart::Attachment(_) => continue,
            };
            *text = redact::text(text).into_owned();
        }
    }
}
```

# A tool

Any `PortableTool` (or `rig_core::tool::Tool`). Its rules go into the
system prompt of every agent it is offered to. What each call touches lets
calls on different things run side by side: the tool says it per call
(`Tool::footprint`, as `read` says it reads its path), else `footprint`
for every call, else its annotations (`Reads(Anything)` for one that only
reads, `Writes(Anything)` otherwise). Blocking work goes in `blocking`.
`args_schema` derives the parameters from the arguments type, whose field
doc comments describe them, and a call whose arguments do not fit is
refused with where and why (rig-core's `tool::validate`).

A tool registered with `app.add_tool_per_call(tool, options, live, make)`
is built for each call's `Place`: the app's workspace (an overlay of it
for a [preview](#gating-and-reviewing-tool-calls)), the calling agent's
`WorkDir` else the `ProjectRoot` (`base`, `confine` for a `WorkDir`), and
for a `live` tool the `Live` its output goes to, as the coding tools and
rig-coding-tools' `ListTool` (`list`, off by default) are.

```rust,no_run
use std::path::PathBuf;
use std::sync::Arc;

use rig_harness::prelude::*;
use rig_harness::rig_tools::workspace::{Local, Workspace};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Default)]
pub struct WordCountPlugin;

impl Plugin for WordCountPlugin {
    fn build(&self, app: &mut App) {
        let options = ToolOptions {
            rules: &["Use `word_count` to count a file's words."],
            footprint: Some(Touches::Reads(Scope::Anything)),
        };
        let here = WordCount { workspace: Arc::new(Local), base: None, confine: false };
        app.add_tool_per_call(here, options, false, |at| {
            WordCount { workspace: at.workspace, base: Some(at.base), confine: at.confine }
        });
    }
}

/// Reads in `workspace`, a relative path from `base`.
struct WordCount {
    workspace: Arc<dyn Workspace>,
    base: Option<PathBuf>,
    confine: bool,
}

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
        let (workspace, base, confine) = (self.workspace.clone(), self.base.clone(), self.confine);
        blocking(move || {
            let text = workspace.read_text(&workspace.place(base.as_deref(), confine, &args.path)?)?;
            Ok(format!("{} words", text.split_whitespace().count()))
        })
        .await
    }
}
```

A tool the plugin answers itself, later, is an open tool:
`app.add_open_tool(name, description, options, observer)` triggers
`ToolCalled<Args>` with the call's entity (`call`), the calling `agent`
and the parsed `args`; the call ends when the plugin decides its
`Decision`. A tool found while the agent runs, such as one an MCP server
lists, is `world.spawn_tool(definition, handler, options)` from a queued
command; despawning the entity removes it. `rig-tui` draws a tool's calls
with the first renderer the tool's plugin adds. It gets the call, its
result once there is one, where the call stands (`state`: `Streaming`
while the model writes the arguments, parsed so far; `Waiting`; `Held`
with the holds' names; `Running` with its `ToolProgress` line; `Done`),
whether it is `expanded`, a running call's `output` so far, and, while
this run knows them, what it published (`facts`), how long it ran (`took`)
and the tool and model calls it started, a subagent's included
(`children`, with their `depth`); `header`, `result_lines` (which keep a
cut output's marker line in sight by the tool's `OutputCut`) and
`default_lines` draw them. A crate that should also build without
`rig-tui` puts it behind a default `tui = ["dep:rig-tui"]` feature.

```rust,no_run
use rig_harness::prelude::*;
use rig_tui::ratatui::text::Line;
use rig_tui::{AppToolRenderersExt, RESULT_LINES};

fn build(app: &mut App) {
    app.add_tool_renderer("word_count", |call| {
        let mut lines = vec![call.header("Count", call.argument("path").unwrap_or_default())];
        lines.extend(call.result_lines(RESULT_LINES));
        for child in call.children {
            let indent = "  ".repeat(child.depth + 1);
            lines.push(Line::from(format!("{indent}↳ {} {:?}", child.name, child.state)));
        }
        lines
    });
}
```

# Gating and reviewing tool calls

A tool call is an effect with a `ToolRequest` (the model's `ToolCall`) and
the `CallHandler` that will run it. A Prepare system edits them as the
call is admitted: the arguments, or the handler, wrapped with
`ErasedHandler::layered` (to give it a secret the model never sees) or
replaced in `PrepareSystems::Edit` to run that call another way, such as
in a sandbox (below). Each time Gate opens, the kernel gives each call the
`Footprint` of the call as it stands, so a held call whose `ToolRequest`
was edited is gated again, and a call that conflicts with an earlier one
of its reply is `Waiting` until that one settles. A call that cannot run
(a missing tool, one `ToolAccess` leaves out, arguments that do not fit)
the kernel answers before every gate. A Gate system decides a call about
to run with `Outcome::refused(why)`, which the model reads, or any
`Outcome` (a cached result), or holds it; a Review system edits each
result in place before it is committed. What a tool published beside its
text, such as rig-tools' `FilesRead` (each file with the hash of the bytes
read), `FilesChanged` (for [any
call](#the-workspace-files-and-child-processes) that may write) or
`ExitStatus`, is the call's `Published`.

**What a call would change**, before it runs: a gate puts `Preview` on it,
the harness holds it, runs its tool on an overlay of the workspace that
keeps every write from the disk, and puts `Previewed(Ok(changes))` on it,
`Undecided` again. A tool not [registered per call](#a-tool), or that
starts a process (`shell`), gets `Previewed(Err(..))`.

**A restart's unfinished calls.** Before a call that may change something
starts, the kernel marks it in its agent's conversation (a `rig.call`
aside), and keeps its result there if it settles while its reply runs on.
A restart restores every unanswered call as `Restored` work, gated again,
but a marked one carries `Interrupted` (`Started`, `Settled(result)`): no
gate that holds or asks sees it, a Gate system may answer it, and after
every gate the kernel answers the rest (a started call with an error, a
settled one with its result). Removing `Interrupted` in Prepare, never in
Gate, runs it again, gated.

```rust,no_run
use rig_harness::prelude::*;
use rig_harness::rig_core::serve::ErasedHandler;
use rig_harness::rig_core::serve::adapters::ToolAdapter;
use rig_harness::rig_core::tool::FileChange;
use rig_harness::rig_ecs::effects::Handler;
use rig_tools::Shell;

/// On an agent whose `shell` calls run sandboxed.
#[derive(Component)]
pub struct Sandboxed;

fn build(app: &mut App) {
    app.add_systems(AgentPass, sandbox.in_set(PrepareSystems::Edit))
        .add_systems(AgentPass, (too_big, after_a_crash).in_set(Stage::Gate));
}

/// Refuses a call that would change more than 500 lines, once it knows.
fn too_big(
    mut calls: Query<(Entity, &Footprint, Option<&Previewed>, &mut Decision), (Undecided, ToolCalls)>,
    mut commands: Commands,
) {
    for (call, footprint, previewed, mut decision) in &mut calls {
        let lines = |changes: &Vec<FileChange>| changes.iter().map(|c| c.added + c.removed).sum::<usize>();
        if previewed.is_none() && !footprint.0.reads_only() {
            commands.entity(call).insert(Preview);
        } else if let Some(Previewed(Ok(changes))) = previewed && lines(changes) > 500 {
            decision.decide(Decider::plugin("too-big"), Outcome::refused("change less at once"));
        }
    }
}

/// A `shell` call a crash cut short: the model is told to check its work.
fn after_a_crash(mut calls: Query<(&ToolRequest, &Interrupted, &mut Decision), Without<Outcome>>) {
    for (request, interrupted, mut decision) in &mut calls {
        if request.call.function.name.as_str() == "shell" && matches!(interrupted, Interrupted::Started) {
            let answer = ToolResult::skipped("a restart cut it short: check what it did");
            decision.decide(Decider::plugin("after-a-crash"), answer);
        }
    }
}

/// A sandboxed agent's `shell` calls get a handler of their own, which
/// starts commands in the agent's directory.
fn sandbox(
    mut calls: Query<(&ToolRequest, &ForAgent, &mut CallHandler), (Added<Admitted>, ToolCalls)>,
    agents: Query<Option<&WorkDir>, With<Sandboxed>>,
    (processes, root): (Res<Processes>, Res<ProjectRoot>),
) {
    for (request, &ForAgent(agent), mut handler) in &mut calls {
        let Ok(work_dir) = agents.get(agent) else { continue };
        if request.call.function.name.as_str() == "shell" {
            let dir = work_dir.map_or(&root.0, |dir| &dir.0);
            let launcher = processes.launcher().in_dir(dir).wrapped(["firejail", "--quiet"]);
            let shell = Shell { launcher, confine: true, ..Shell::default() };
            handler.0 = Handler(ErasedHandler::new(ToolAdapter::new(shell)));
        }
    }
}
```

**Judging by what a call touches.** A gate that guards files reads a
call's `Footprint` (rig-core's `Touches`), not its tool's name, so one
rule covers `edit`, `write`, `delete`, `move` and any plugin's tool. Its
paths are absolute and normalized as written (`src/../x` is `x`), not
through links: a guard strict about links compares
`workspace.0.canonical(path)`.

**After every gate.** An answer that stands in for running a call (a
cached result, a dry run) goes in `GateSystems::After` and reads
`Decision`, so it never answers a call a gate refused. So does a group of
calls that runs all or none, such as a patch's edits: the plugin spawns a
child call per member, unheld (no gate sees a call held from its spawn),
and in `After`, without `Undecided`, holds the free ones while any member
is held or `Waiting`; then it refuses them all if one was refused or
failed, else releases them.

## A held call holds back the later calls of its reply

A call's `Footprint` says what it may read and change (`Touches`): `Reads`
or `Writes` some paths (a directory covers what is under it) or
`Anything`, or `Independent`, which changes nothing the reply's other
calls see, such as starting a subagent, and counts as reading anything. A
call waits for an earlier call of its reply when one of them writes what
the other touches, and a held call holds back like a running one. Say the
model's reply asks for five calls, and a permission gate holds the `edit`:

| # | Call | Footprint | Runs |
|---|---|---|---|
| 1 | `edit src/a.rs` | `Writes([src/a.rs])` | once the user said yes |
| 2 | `read src/b.rs` | `Reads([src/b.rs])` | at once |
| 3 | `search docs/` | `Reads([docs])` | at once |
| 4 | `read src/a.rs` | `Reads([src/a.rs])` | once 1 settled: it ran, or was refused |
| 5 | `task` | `Independent` | once 1 settled |

`shell` may change anything (`Writes(Anything)`), so every call after a
held `shell` call waits, reads too: a later call may depend on what an
earlier one changes (`mkdir x` then `ls x`), the order this keeps. So a
gate that cannot get an answer refuses at once rather than holding. A
plugin that knows better says what a call touches in Prepare, with
`Declared`, and the kernel keeps it: with
`(Footprint(Touches::Writes(Scope::Paths(vec![src]))), Declared)` on a
held `cargo fmt`, a read of `docs/` runs at once (`Anything` overlaps
every path, so a held write still holds back `Reads(Anything)`).

# Asking the user: questions and pickers

A question for the user is an `Ask` entity: a title, a body, the choices
offered and whether a line of text answers it too. It is an effect,
recorded with its answer, and any front answers it by deciding it
(`Answer::Choice(id).outcome()`, `Answer::Text(..)`), which closes it in
the others. A plugin that waits on the answer spawns the `Ask` as a hold
on what it asks about, so an interrupt takes it away everywhere. A front
that shows questions spawns an `Answerer`; a plugin asks only when
`Answerers::can_answer` for the agent, else applies its own policy. The
user's next line to the agent answers an open question that takes text.

A question may propose an answer to edit (`Ask::initial`), take several
lines (`multiline`), or wait beside the work (`inline`), which the user's
next line never answers: the terminal view puts the first two in the input
(Enter answers, Esc refuses) and an inline one in a strip above it (Alt+1
to Alt+9 pick, Alt+0 dismisses).

A choice from a list is a `Picker` entity: a title, a body, items that
each may run a command line and show a `detail`, and whether typed text
answers it (`free_text`); a new `Picker` inserted replaces it, and
despawning it closes it. A front answers it, or an open `Ask`, with
`Pick::person(entity, chosen)`; a picker then runs the chosen item's
command and gets `Picked` (`Chosen::Item`, `Chosen::Text`, or `None`).
`Picker::multiple()` (`.checked(true)` ticks the last item) and
`Ask::multiple()` take several: the terminal view ticks with Tab, and
Enter answers `Chosen::Items` (each command runs, in list order) or
`Answer::Choices`. A long body scrolls with Shift+Up/Down.

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
            decision.decide(Decider::plugin("approve-commits"), Outcome::refused("nobody can approve a commit"));
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
            decision.decide(Decider::plugin("approve-commits"), Outcome::refused("the user said no"));
        }
        commands.entity(ask).remove::<HoldOf>();
    }
}
```

# A slash command

A command is a one-shot system, `app.add_command(name, help, system)`,
taking `In<CommandArgs>`: the `agent` it was typed for, the `args` after
its name, and the `mode` its line was sent with; `RunCommand::new(agent,
line)` runs one. An error notice about the agent while it runs refuses it:
the line goes back in the input with the error, and Enter then sends it to
the model as it is. The `Gate` set `CommandSystems` runs a user's line
that starts with `/`; a plugin that takes some over orders its Gate system
`.before(CommandSystems)`.

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
    // `Queue`: once the running turn would end, or now for an idle agent
    // (`DeliveryMode` has the others).
    let reminder = Deliver::new(args.agent, format!("Reminder: {}", args.args), DeliveryMode::Queue);
    commands.trigger(reminder.with_author(Author::plugin("remind")));
}
```

# What the user types

What the user types, in any front, is a message to its agent from
`Author::user()`, and so is what an agent or a plugin sends: `Deliver`
spawns an `Incoming` effect `ForAgent` the agent. A Prepare system may
change it (`Added<Admitted>`): its `parts`, text and attachments
(`MessagePart::Text`, `MessagePart::Attachment`) in the order the model
reads them; `text()` and `text_mut()` are its first text. rig-coding-tools
attaches the `@path` files it names before the text, from the agent's
`WorkDir` else the `ProjectRoot`, in its public set `AttachSystems` (in
`PrepareSystems::Compose`; read them `.after` it). A Gate system may take
it over by deciding it (`Outcome::custom(..)`), as the harness runs a `/`
line as a command. At Start the kernel gives what is left to the agent,
which starts a turn with it or keeps it for its running turn as its `mode`
says; accepted and not sent yet, it is in the agent's inbox,
`Query<(&Incoming, &ForAgent), (With<Started>, Without<Outcome>)>`. Its
record in the effect log reads back with
`MessageRecord::read(&record.kind)`.

**With the user's next message.** What a plugin wants the agent to read
when the user next writes, such as a background check's failure, it keeps
in its own state and adds to the next `Incoming` from `AuthorKind::User`
in Prepare, a `MessagePart::Attachment(Attachment { label, content })`: no
turn of its own, no hold, and taken back by clearing that state. A message
accepted but not given yet, such as a `DeliveryMode::Note` to a busy agent
(given as its turn ends), is taken back by deciding it.

A plugin that needs a mark on its own message spawns it:
`(Incoming::new(text, mode), ForAgent(agent), MyMark)` is all `Deliver`
does. Once delivered into a turn (one it started, or a running one it
steered), the message gets `DeliveredIn(turn)`, and the turn's first model
call is recorded under it. It is despawned once it settles, so a plugin
marks the turn then (`On<Add<DeliveredIn>>`) and finds the turn's calls
from it (`CallOf(turn)`). A user's message may carry a title,
`Author::user().titled(..)`, kept as its author
(`conversation.author(at)`). What the user typed in message `at`, without
attachments or what a plugin delivered, is `typed(&conversation, at)`
(`first_typed` for the first).

```rust,no_run
use rig_harness::prelude::*;

#[derive(Default)]
pub struct ShortcutsPlugin;

impl Plugin for ShortcutsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            AgentPass,
            (
                review_on_reviewer.in_set(PrepareSystems::Edit),
                // Before the harness's systems that decide messages too.
                keep_to_self.in_set(Stage::Gate).before(CommandSystems),
            ),
        )
        .add_command("review", "Asks for a review, on the reviewing model", review)
        .add_observer(mark_review_turn);
    }
}

/// On a message `/review` sent, and on the turn it went into.
#[derive(Component)]
struct Review;

/// `/review` asks the model for a review, titled so the transcript says so.
fn review(In(args): In<CommandArgs>, mut commands: Commands) {
    let text = "Review the change you just made.";
    let author = Author::user().titled("review");
    let message = Incoming::new(text, args.mode).with_author(author);
    commands.spawn((message, ForAgent(args.agent), Review));
}

/// The review's message goes once it settles; its turn keeps the mark.
fn mark_review_turn(
    delivered: On<Add<DeliveredIn>>,
    reviews: Query<&DeliveredIn, With<Review>>,
    mut commands: Commands,
) {
    if let Ok(&DeliveredIn(turn)) = reviews.get(delivered.entity) {
        commands.entity(turn).try_insert(Review);
    }
}

/// The model that reviews.
const REVIEWER: &str = "deepseek/deepseek-flash";

/// Every call of a review's turn goes to the reviewing model.
fn review_on_reviewer(
    mut calls: Query<(&mut ModelRequest, &CallOf), (Added<Admitted>, With<TurnCall>)>,
    reviews: Query<(), With<Review>>,
    models: Res<Models>,
) {
    for (mut call, &CallOf(turn)) in &mut calls {
        if reviews.contains(turn) {
            call.connection = models.connect(REVIEWER).ok();
        }
    }
}

/// A line that starts with `#` is the user's note to self: it never
/// reaches the model.
fn keep_to_self(
    mut typed: Query<(&Incoming, &ForAgent, &mut Decision), (Undecided, IncomingMessages)>,
    mut notices: MessageWriter<Notice>,
) {
    for (typed, &ForAgent(agent), mut decision) in &mut typed {
        if typed.author.kind == AuthorKind::User && typed.text().starts_with('#') {
            decision.decide(Decider::plugin("to-self"), Outcome::custom("kept"));
            notices.write(Notice::info(agent, "Noted; not sent."));
        }
    }
}
```

# Model requests

A model call is an effect entity with a `ModelRequest`: the rig-core
`CompletionRequest` it sends (tools, options), the `connection` it goes
to, else its agent's model (`Models::connect` makes one), and, for a
request the kernel built, its system prompt in parts, `prompt` (rig-core's
`SystemPrompt`: the agent's `role` and its tagged `sections`, the tools'
rules, then the `PromptSection`s by order), and its messages, `history`: a
view of the agent's conversation, whose edits are never saved. The kernel
builds a turn's own call (`TurnCall`), or one built from an agent outside
its turns (`BuildFrom(agent)`), when it is admitted (an
`On<Add<Admitted>>` observer runs before that, a Prepare system after). A
plugin changes a request in `PrepareSystems::Edit`, so compaction measures
what is sent; it edits `history` as a [conversation](#changing-history),
each edit by its `Author`, kept entries keeping their ids, and until Start
`history.unsaved()` says who changed the request and how. Start puts the
prompt, with the rules of the tools the request carries, and then
`history` into the request's messages, and fits it to the model it goes to
(an option that model cannot take is left out, plugin edits stay).

- **Routing** a call to another model is a Prepare edit of its
  `connection`.
- **Adapting** the model, whichever it is, such as so it reads tools as
  text, is an `Adapter` pushed on the call's `Adapters`, applied at Start
  to the connection the call finally goes to, in order, `outermost()` ones
  last, so a router and an adapter never undo each other. It wraps
  `connection.handler` with rig-core's `ErasedHandler::layered` or
  `map_stream` (each streamed item rewritten, and recorded so), and says
  what the wrapped model takes on `connection.spec`.
- **What a call is sent as.** `ModelStarts::sent` gives a call's
  connection, its request as Start will send it (to measure it in
  `PrepareSystems::Fit`) and the options its model leaves out; `prompt`
  its system prompt in parts. From Start on, a call has
  `SentTo(connection)`, its `ModelRequest.request` is as sent, and
  `Fitted` lists the options left out (by the model's spec, or by a dry
  run of its handler, rig-core's `Serve::check`), for a plugin to warn. A
  fact a `map_stream` closure inserts in the call's `PublishedContext` is
  its `Published` once answered.
- **Caching.** A provider that caches by marker marks the prefixes a
  request names in `cache_points` (`Tools`, `System`, `Messages(n)`). A
  section that changes every turn breaks every prefix after it, so a
  plugin moves it in `PrepareSystems::Edit` (`prompt.take(tag)`) to the
  last message.
- **Tools.** The kernel offers an agent the tools its `ToolAccess` allows.
  A Prepare edit takes some out (`call.request.tools.retain(..)`, and a
  `prompt.sections` line on why), but a tool left out still runs if
  called: a Gate system refuses a call; `ToolAccess` is the hard limit.
- **Below every adapter,** `models.0.wrap_http(|inner|
  inner.with_middleware(..))` in `build` hands every model call's HTTP
  requests, each with the call it serves (`rig_core::serve::Serving`), to
  a rig-core `HttpMiddleware`, such as one that rotates API keys.
- **Reaching a vendor.** `models.0.provide(vendor, provider)` reaches its
  models through the plugin's provider (`ConnectOptions` with a key);
  `models.0.provider(vendor)` hands over the one it replaces.
- **Answering later, from elsewhere,** such as a batch API: the model is
  connected with rig-core's `ErasedHandler::unavailable(descriptor, why)`,
  and a system in `GateSystems::After` holds each call, submits what
  `ModelStarts::sent` gives, and keeps the batch's id in a saved component
  by turn (a restart runs the call again, `Restored`, to hold again). A
  `WakeAt` polls, and rig-core's `wire::decode_reply` turns the result's
  body into the reply it decides the call with.

A failed call carries the kernel's `RetryVerdict` (`Retry`, `GaveUp`,
`Overflow`, `Final`) in Review, which a reviewer may edit in place. A
plugin that takes the failure over spawns a new call the failed one is
`SupersededBy`, or holds it (`HoldOf`) while it works, as rig-compaction
does to summarize first; otherwise the kernel retries after a `Backoff` or
proposes the turn's end. Only a call its own model answered is judged: one
anyone else answered while it ran gets no `RetryVerdict`. A call refused
before it was sent fails the turn and keeps the user's message.

A fixed part of the system prompt is a `PromptSection` entity,
`PromptSection::new(order, tag, text)`, in every agent's prompt, or with
`SectionOf(agent)` in that agent's alone. rig-basics keeps the instruction
files and each agent's environment in sections, re-read as each turn
starts (`rig_basics::RefreshProjectContext` re-reads them now).

```rust,no_run
use std::sync::Arc;

use rig_harness::prelude::*;
use rig_harness::rig_core::message::Message;
use rig_harness::rig_ecs::effects::Handler;

/// The model a turn falls back to when its own gave up.
const FALLBACK: &str = "deepseek/deepseek-flash";

#[derive(Default)]
pub struct ModelRequestsPlugin;

impl Plugin for ModelRequestsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            AgentPass,
            (
                (example_first, tools_as_text).in_set(PrepareSystems::Edit),
                fall_back.in_set(ReviewSystems::Act),
            ),
        );
    }
}

/// Every turn's request starts with a worked example, by this plugin: the
/// request has it, the agent's conversation never does.
fn example_first(mut calls: Query<&mut ModelRequest, (Added<Admitted>, With<TurnCall>)>) {
    for mut call in &mut calls {
        if let Some(history) = &mut call.history {
            let by = Author::plugin("examples");
            let example = [Message::user("Rename x to y."), Message::assistant("Done: 3 places.")];
            history.splice(0..0, example.map(|m| Entry::new(m).by(by.clone())), by);
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
                connection.spec = Arc::new(spec);
                // Each streamed item, rewritten: here passed on, where the
                // plugin's parser reads tool calls from the text.
                let pass_on = |_published| |item| vec![item];
                connection.handler = Handler(connection.handler.0.map_stream("text-tools", pass_on));
            }
            connection
        }));
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
        let Ok(fallback) = models.connect(FALLBACK) else { continue };
        let retry = (TurnCall, CallOf(of.0), ForAgent(agent.0), Effect::child_of(effect.parent));
        commands.spawn((request.clone().on(fallback), retry)).add_related::<SupersededBy>(&[call]);
    }
}
```

# What a model streams

Every model call streams, a turn's and a plugin's own alike. Each event is
a `StreamDelta { effect, agent, event }` message carrying rig-core's
`StreamEvent`: a part's `Start` (a tool call's name), its `Text`,
`Reasoning` or `Arguments` fragments, and its `End` (a finished tool call
with its id); `part` tells parallel tool calls apart. It is written in
Arrive in order, after the call's adapters rewrote it; Bevy keeps a
message two frames, so a reader in `PostUpdate` gets a reply's last events
even when it settled that frame. A turn's call also adds its text and
reasoning up in its `Partial`, and from its first item has a `Streamed {
first, last, items }` beside its `Started { at }`. A delta moves no stage,
so a reader that spawns effects from it, such as a tool call started
before the reply ends, runs `.before(AgentLoop)` or calls `Wake::wake`.

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
        if let StreamEvent::Text { text, .. } = &delta.event {
            print!("{text}");
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
the agent's turns over the session and its `first_message`, the
`MessageId` it begins at (`conversation.start_of(turn.first_message)` is
its index now, `asked(id)` the entries it answers, `request(id)` the one
that asked). It is spawned before its first request is built, so an
`On<Add<TurnOf>>` observer's edit is in that request; once ended it loses
`TurnOf` and keeps `ForAgent`. Its end is a proposal: when the model
answers without calling tools, or the turn fails, the kernel proposes a
`TurnOutcome` (`Answered(message)`, `Failed(reason)`) in Commit, which a
Review system sees in the next pass (`Added<TurnOutcome>`) and may edit in
place, hold (`HoldOf(turn)`), or carry on: a `Queue` or `Steer` message
the agent accepts takes an `Answered` proposal away, and so does a new
`(TurnCall, CallOf(turn), ForAgent(agent))` once admitted, with its holds.
A plugin may insert `TurnOutcome::Failed(reason)` or
`Stopped(Some(reason))` at any time, such as a budget: calls not started
never start, the others are answered as stopped (or with their
`IfStopped`). Once nothing holds the outcome and every call settled, the
turn ends, `Settled` until the next frame; an `On<Add<Settled>>` observer
sees how, and may retire the agent (`Retire { entity }`). `Interrupt`
stops a turn at once.

**Ending a turn with an answer already known**, such as the JSON a
`submit` tool was called with: a Gate system decides the turn's next
model call itself, with a completion of that text. Nothing is sent, the
turn ends `Answered` with it, and `--print` prints it as the answer.

```rust,no_run
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::{CompletionResponse, Usage};
use rig_harness::rig_core::effect;
use rig_harness::rig_core::message::{AssistantContent, Origin as Made, Text};

#[derive(Default)]
pub struct TurnEndsPlugin;

impl Plugin for TurnEndsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(AgentPass, end_with.in_set(Stage::Gate))
            .add_systems(AgentPass, check_the_tests.in_set(Stage::Review));
    }
}

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
            decision.decide(Decider::plugin("end-with"), Outcome(Ok(effect::Outcome::Completion(reply))));
        }
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
            commands.trigger(ask.with_author(Author::plugin("check-the-tests")));
        }
    }
}
```

# Effects of a plugin's own

A plugin's own work gets the same stages, recording and visibility as the
kernel's when it is an effect: an `EffectRequest` with its rig-core
`EffectKind` (`Custom` for a kind of its own) and a `handler` the kernel
runs at Start, or none, an open effect the plugin decides when done.
Others can gate it, review its outcome and replay it; `CallOf(turn)` keeps
a turn open until it settles. Work off the main thread is
`Running::answering(pool, wake, future)` on the effect.

**What a running effect would answer if stopped.** Its owner keeps an
`IfStopped(outcome)` on it current, such as the output so far, which the
kernel answers with in place of "interrupted". A running tool call's owner
also keeps a `ToolProgress` line on it (`12 of 40 files`). A tool that
prints as it works reports through a rig-tools `Live`, which
`CallOutput::new(&wake)` gives with the component to insert on the call:
the harness then keeps its `ToolProgress` and `IfStopped`, as for `shell`.

**An owner follows its effect's successor.** Another plugin may take the
effect over, such as a watchdog that sends a stalled call again
(`commands.queue(resend(call))`): it settles as superseded, its
`SupersededBy` naming the successor. A plugin that waits on its effect by
a marker registers `app.add_observer(follow_successor::<Marker>)`, which
moves the marker, and the effect's hold, to each successor. A model call a
plugin makes is such an effect too: a rig-core model it calls directly is
one the kernel, the effect log and the spending never see.

```rust,no_run
use rig_harness::prelude::*;
use rig_harness::rig_core::effect::EffectKind;

/// On a deploy effect: the steps its pipeline (not shown) reported.
#[derive(Component, Default)]
struct Deploy(Vec<String>);

fn build(app: &mut App) {
    app.add_command("deploy", "Deploy a target", deploy)
        .add_observer(follow_successor::<Deploy>)
        .add_systems(Update, track.before(AgentLoop));
}

/// `/deploy`: a deploy effect any plugin can gate.
fn deploy(In(args): In<CommandArgs>, mut commands: Commands) {
    let kind = EffectKind::Custom {
        kind: "deploy".into(),
        payload: serde_json::json!({ "target": args.args }),
    };
    let request = EffectRequest { kind, handler: None };
    commands.spawn((request, ForAgent(args.agent), Deploy::default()));
}

/// Keeps what each deploy answers if it is stopped first, the steps it
/// did, and answers it once its last step is done.
fn track(
    mut deploys: Query<(Entity, &Deploy, &mut Decision), (Changed<Deploy>, PluginEffects)>,
    mut commands: Commands,
) {
    for (deploy, Deploy(steps), mut decision) in &mut deploys {
        if steps.last().is_some_and(|step| step == "live") {
            decision.decide(Decider::plugin("deploys"), Outcome::custom("deployed"));
        } else {
            let text = format!("stopped after: {}", steps.join(", "));
            commands.entity(deploy).insert(IfStopped(Outcome::custom(text)));
        }
    }
}
```

# Subagents

The `rig-subagents` plugin spawns an agent for a `SpawnSubagent` effect
`ForAgent` its parent, which the model's `task` tool makes, or a plugin
for agents of its own. Prepare systems edit it (model, tools, system
prompt) and Gate systems refuse or hold it. At Start the plugin spawns the
agent, every setting the request leaves out taken from the parent (its
`WorkDir` too), as `SpawnedAgent(agent)` on the request, and answers.

- `SpawnSubagent::task(title, brief)` works on a task and reports to its
  parent, as `task` starts it.
- `SpawnSubagent::helper(title)` works for the plugin, which talks to it:
  it has a `Helper(title)`, reports to no one, and `message` cannot reach
  it; `.brief(text)` sends it a first message. With `.once()` it retires
  when its first turn ends, whose answer an `On<Add<Settled>>` observer
  reads.

## Extending `task`

A plugin adds an argument to `task` with a property in its `ToolDef`'s
`parameters` and a line in its `ToolRules` on when to use it ([editing
another plugin's
tool](#turning-off-or-replacing-what-another-plugin-added)): `task` puts
each argument it does not take into `SpawnSubagent.extra`, for the plugin
to read in Prepare.

```rust,no_run
use rig_harness::prelude::*;
use rig_subagents::{SpawnSubagent, TASK};

#[derive(Default)]
pub struct ReviewersPlugin;

impl Plugin for ReviewersPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(AgentPass, as_reviewer.in_set(Stage::Prepare))
            .add_command("critic", "Ask a one-job critic about the last change", critic);
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
```

# The effect log: reading, redacting, replaying

Every effect that settles is an `EffectRecorded(EffectRecord)` message, in
settle order: its request (`kind`), its answer (`outcome`, a reviewer's as
`reviewed`), who decided it before it ran (`decided_by`), its times, the
agent it worked for (`scope`, an `AgentId`), its `turn` (`Turn.index`),
and the effect it was made under (`parent`). rig-telemetry writes them to
the session's effect log in `Last` (`WriteJournal`) and reads them back
through its `EffectLog`: `read()` gives every record, and
`read_matching(&filter)` those an `EffectFilter` keeps, by agent, kind
name (`completion`, `tool_call`, `message`, `subagent` or a plugin's own)
and turn. It holds what was written by the end of the last frame: to count
as a run goes, keep [saved state](#saved-state-counting-every-tool-call).

**Redacting.** A record holds what its effect really carried (tool output,
diffs, the request as sent). A plugin keeps a secret out of the log by
changing the records in place with `MessageMutator<EffectRecorded>` in
`Last` `.after(StopTurns).before(WriteJournal)`, so it sees a stop on
exit's records too. rig-core's `redact` finds keys and tokens by their
shapes: `redact::json` in a record, `redact::text` in a line.

**Replaying.** A replayer answers effects from a recorded session's log
before they start: a model call with an adapter placed outermost, every
other effect in `GateSystems::FromRecord` (rig-cassette's
`RecordedAnswers` and `Divergence`). Agent ids are new in each session, so
it pairs agents by spawn order, runs `subagent` effects again (only
running one spawns the agent), and asks for a recorded exit again where it
was, so what held it replays too.

```rust,no_run
use rig_harness::prelude::*;
use rig_harness::rig_core::redact;
use rig_harness::rig_ecs::{StopTurns, WriteJournal};
use rig_telemetry::{EffectFilter, EffectLog};

fn build(app: &mut App) {
    app.add_command("effects", "What the agent did in its turn N: /effects N", effects)
        .add_systems(Last, hide_secrets.after(StopTurns).before(WriteJournal));
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

/// Takes every key-shaped secret out of every record before it is written.
fn hide_secrets(mut records: MessageMutator<EffectRecorded>) {
    for EffectRecorded(record) in records.read() {
        let Ok(mut json) = serde_json::to_value(&*record) else { continue };
        redact::json(&mut json);
        if let Ok(clean) = serde_json::from_value(json) {
            *record = clean;
        }
    }
}
```

# Settings

A plugin's settings are its table of the user's `RIG_HOME/settings.toml`:
`app.add_setting::<T>("section")` in `build` inserts the resource `T` read
from `[section]`, else `T::default()`; a table that does not read as `T`
keeps the default and is shown at startup. A test inserts
`Settings::from_toml(..)` before adding the plugin, so the user's file is
never read. The default plugins' tables include `[shell] max_timeout`,
`[checkpoints] enabled` and `[context] file_bytes, total_bytes` (how much
of the instruction files is read). Here a [command-line
option](#fronts-options-and-commands) overrides the setting.

```rust,no_run
use rig_harness::prelude::*;
use serde::Deserialize;

/// The `[budget]` table: `max_calls = 50`; no limit by default.
#[derive(Resource, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct Budget {
    pub max_calls: Option<usize>,
}

#[derive(Default)]
pub struct BudgetPlugin;

impl Plugin for BudgetPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<Budget>("budget")
            .add_flag(Flag::new("budget", "Stop a turn after N tool calls").value("N"));
        let invoked = app.world().get_resource::<Invoked>();
        if let Some(n) = invoked.and_then(|it| it.flag_value("budget")?.parse().ok()) {
            app.insert_resource(Budget { max_calls: Some(n) });
        }
    }
}
```

# Saved state: counting every tool call

This is how to count what agents do, such as their tool calls; the [effect
log](#the-effect-log-reading-redacting-replaying) looks back at a run. An
agent component that derives `Reflect` and says `#[reflect(Component,
Saved)]` is kept with the session (a generic type once registered): each
change is logged, and a restart, `/reload` or `/resume` brings the newest
value back, inserted on the restored agent before `SessionRestored`
exists, so an `On<Insert<T>>` observer can re-arm what it says. A resource
that says `#[reflect(Resource, Saved)]` and is inserted in `build` is kept
the same way. One holding rig-core data, such as a `MessageId`, is saved
through serde ([so](#changing-history)). Work a restart carries on is
`Restored`, and gated again.

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
    let Ok((conversation, mut counts)) = agents.get_mut(added.entity) else { return };
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

# Changing history

An agent's `Conversation` is rig-core's, changed by calling its methods.
`push(entry)` adds a message and `splice(range, entries, by)` replaces
some: a cut is `from..` with nothing, a summary the range it stands for
with one message, an edit in place `entry(i)` passed back changed. Each
message is its own entry with a `MessageId` (`id(i)`) that stays across
edits and restores (`index_of(id)`; `start_of(id)` is where a span that
began at `id` begins now). A splice that reaches the end clears the halt.
A message that is not the user's own or the model's has an `Author`
(`Author::plugin(name)`, `agent(..)`, `rig()` for rig's own such as the
`STOPPED` note, or the user `titled`): `Entry::new(message).by(author)`.
Every splice names its author, so the log says who rewrote what.

**Asides** never reach a model: an `Aside { kind, after, author, value }`
follows the entry `after` and is saved, restored and forked with the
conversation unless `unsaved()` (`add_aside`, `replace_aside`,
`remove_aside`, `asides()`). The harness's `Note` is the kind fronts draw
([the transcript](#the-transcript-notes-looks-and-scrolling)).

**Saving.** A change waits in the conversation's outbox until the kernel
saves it to the agent's log: between the stages of every pass, at the end
of the frame, before a tool that may change something starts, and on an
interrupt. Each one saved is a `Committed { agent, change }`, a `Recorded`
with the `Change` (`Push`, `Splice`, `Halt`, `Aside`, `Reset` for a value
inserted over the component), its author (`by`) and time. A splice before
the end costs the provider's prompt cache from there on.
`commands.trigger(Retire { entity: agent })` despawns an agent for good,
so a restart brings back neither it nor the agents it spawned.

**Compaction** splices its summary, an entry by
`rig_compaction::summary_author()`, in place of what it summarized
(`summary_in(&change)` tells that splice from others). Its policy,
rig-memory's `CompactionPolicy` in rig-compaction's `Compaction`, cuts by
`Cut::Budget` or a plugin's `Cut::With`; `plan(..)` plans a summary call
under a plugin's own policy, and `On<Insert<Summarized>>` sees a new one.

**The session format** is rig-memory's (`rig-session/1`), read and written
without an app: `Session::read(&*store)` gives each agent's conversation,
saved values and `Stamp`s, and `session.save(&*store)` appends what
changed; `load`/`save` keep one conversation, `history`/`as_of` show how a
log got where it is, `fork(from, to, before)` copies a session as it was
before a message. A session not read whole is refused
(`Session::read_lenient` reads what it can). An importer opens
`home.new_session()` with `SessionStores::open`, makes each edit at the
source's time (`conversation.dated(at_ms, |c| ..)`), saves, and lists it
in `/resume` with rig-sessions' `write_meta(&*store, ..)`.

```rust,no_run
use rig_harness::prelude::*;

/// The messages the user bookmarked, kept with the session. A `MessageId`
/// is not `Reflect`, so the component is saved through its serde form.
#[derive(Component, Reflect, Clone, Default, serde::Serialize, serde::Deserialize)]
#[reflect(opaque, Component, Saved, Default, Clone, Serialize, Deserialize)]
struct Bookmarks(Vec<MessageId>);

/// `/undo`: the user's last message and everything after it go, unless a
/// turn runs.
fn undo(In(args): In<CommandArgs>, mut agents: Query<&mut Conversation, Without<ActiveTurn>>) {
    let Ok(mut conversation) = agents.get_mut(args.agent) else { return };
    let last = conversation.entries().rev().find(|entry| entry.is_users_own());
    if let Some(at) = last.map(|entry| entry.index) {
        conversation.splice(at.., [], Author::plugin("undo"));
    }
}

/// Says who rewrote an agent's history.
fn rewrites(mut committed: MessageReader<Committed>, mut notices: MessageWriter<Notice>) {
    for Committed { agent, change } in committed.read() {
        if let (Change::Splice { removed, .. }, Some(by)) = (&change.change, &change.by) {
            let text = format!("{} replaced {} messages", by.label(), removed.len());
            notices.write(Notice::info(*agent, text));
        }
    }
}
```

# Time

Time is Bevy's `Time`. `.run_if(on_real_timer(Duration))` runs a system
once per interval, an entity with a `KeepAwake(Duration)` keeps the
sleeping loop running at that pace, and a one-off wait is
`commands.delayed().duration(Duration)`. Bevy's clock stops while the
machine sleeps; a time of day is a `WakeAt(SystemTime)` on an entity,
which gets `Due` once it passed. A wait that may be put off or called off,
such as a debounce, is a `WakeAt::after(wait)` on an entity of the
plugin's: a new one inserted puts it off, despawning calls it off.

```rust,no_run
use rig_harness::prelude::*;

/// How often a running turn says it is still running.
const EVERY: Duration = Duration::from_secs(60);

#[derive(Default)]
pub struct StillWorkingPlugin;

impl Plugin for StillWorkingPlugin {
    fn build(&self, app: &mut App) {
        app.add_observer(keep_awake)
            .add_systems(Update, still_working.run_if(on_real_timer(EVERY)));
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
```

# The workspace, files and child processes

`ProjectRoot` is the directory the session works in. An agent with a
`WorkDir`, such as its own worktree, works there instead, and so does an
agent it spawns: the coding tools resolve its paths from it and refuse one
outside it, its commands start there, and its prompt names it. Both are
paths in the app's workspace, `ProjectWorkspace`: this machine's
(rig-tools' `Local`) unless a plugin puts another in, such as one over SSH
(a rig-tools `Workspace`), which may be swapped while the app runs, so
read it on each use. The tools, `@path` attachments, the instruction files
and `@` completion go through it, and plugins learn it is remote from
`workspace.0.host()`. `rig_tools::split_status` parts a `shell` result
from its status line, also `Published` as `ExitStatus`. `RigHome` is the
`RIG_HOME` layout (`home.0.session(..)`), and `TokioRuntime` rig's tokio
runtime. A plugin that should also run without the harness's plugins inits
the ones it reads (`app.init_resource::<ProjectRoot>()`).

**The main-thread rule.** A workspace call may wait on the network. Tools
call it inside `blocking`; a system calls it inline only while `host()` is
`None`, else as `Running::spawn` work, holding what waits for it, as
`read_todo` below does. Work a turn waits for holds the turn's first
`TurnCall` (admitted in Prepare) until it lands. A plugin's own file tool
resolves paths with `workspace.place(base, confine, path)`, and
`edit::apply(path, text, edits)` is what `edit` makes of a text.

**What a call changed.** The file tools publish the files they write as
`FilesChanged`, each change with the hash of the content it was made to
(`before`) and of what it left (`after`). The harness wraps each call that
may write anywhere (`Writes(Anything)`: `shell`, a tool that does not say)
in rig-tools' `Checkpointed`, which snapshots its agent's directory before
and after it runs and publishes the difference so. A turn's calls share
the snapshots, its `TurnCheckpoint`: `first()` is the files before its
first such call (`Snapshot::content` reads a file as it was, in git). In a
git work tree on this machine a snapshot is a tree written with a private
index and objects in the session's `Scratch`, never in the user's `.git`;
elsewhere a scan. It misses what git or the listing leave out (ignored,
hidden, `target`), counts an edit made between two calls (the user's
editor) as the next call's, and wraps no open tool, nor a call whose
footprint a plugin narrowed. `[checkpoints] enabled = false` turns it off.

**Child processes.** `Processes` is how every child process starts, a
rig-tools `Launcher` (a wrapper command, a directory, environment): the
shell tool reads it for each command, so a sandbox plugin that sets it
(`processes.set(..)`) wraps every later process. A plugin builds its own
with `processes.launcher().command(program)`, or `.spawn(program, args,
Spawn { stdin, merge_stderr })`: the `Process` hands its output over line
by line (`lines()`) or at its end (`output(timeout)`, inside `blocking`),
ends with SIGTERM then SIGKILL on `stop(grace)`, and is killed with its
process group when dropped. Every launcher it hands out drops the
session's [secrets](#sessions-their-store-secrets-and-exiting), which a
bare `std::process::Command` passes on; one that runs in the project goes
through the workspace (`workspace.0.spawn(&launcher, ..)`).

**Files and streams.** An entity with a `FileWatch` gets `FileChanges`
messages: first the files under its path when armed (read them then, so
none made in between is missed), then each batch created, changed or
removed, as the system reports them (another workspace may look every
second, rig-tools' `Workspace::watch`), leaving out what the workspace's
listing does (ignored and hidden files, `target`) unless `.all()`. A
long-lived source, such as a socket or a thread's channel, is a
`Running::stream(pool, &wake, items)` on an entity: each item is inserted
on it as a `Done<T>`, in order, waking the loop, and the `Running` goes
once the stream ended; the reader observes `Insert<Done<T>>`.

```rust,no_run
use rig_harness::prelude::*;
use rig_harness::rig_core::tool::FilesChanged;

/// Says how many files each tool call changed, `shell`'s included.
fn changed(calls: Query<(&Published, &ForAgent), Added<Settled>>, mut notices: MessageWriter<Notice>) {
    for (published, &ForAgent(agent)) in &calls {
        if let Ok(Some(FilesChanged(changes))) = published.0.get::<FilesChanged>() {
            notices.write(Notice::info(agent, format!("{} files changed", changes.len())));
        }
    }
}

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

/// The project's `TODO.md`, once read.
#[derive(Component)]
struct Todo;

/// Reads `TODO.md` where the project is: at once on this machine, off the
/// main thread in another workspace, so no frame waits on the network.
fn read_todo(
    workspace: Res<ProjectWorkspace>,
    root: Res<ProjectRoot>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let (workspace, path) = (workspace.0.clone(), root.0.join("TODO.md"));
    if workspace.host().is_none() {
        commands.spawn((Todo, Done(workspace.read_text(&path).unwrap_or_default())));
        return;
    }
    let pool = bevy_tasks::IoTaskPool::get_or_init(bevy_tasks::TaskPool::default);
    let read = blocking(move || workspace.read_text(&path));
    commands.spawn((Todo, Running::spawn(pool, &wake, async move { read.await.unwrap_or_default() })));
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

# Sessions: their store, secrets and exiting

**The store.** What a session holds goes through its store, a rig-memory
`Store` (`SessionStore`): each agent's log, its images' blobs, and files
by name (the effect log, `/resume`'s listing, a fork's parent). The
harness opens it from `SessionStores` once every plugin is built and reads
the session before writing anything: one an older rig saved, or not
readable whole, is refused unchanged. A plugin that keeps sessions
elsewhere or sealed puts its own `SessionStores::new(|dir| ..)` in while
the app is built; forks, imports, deletes and `/resume` open stores
through it too. What is named by path stays plain, under `Scratch`: cut
tool output, pasted images and the text log (`Scratch(None)` keeps none).
A copy lists a store's files with `files()`, and a command that rewrites a
session skips one that `runs_elsewhere()`.

**A plugin's own files** go in `PluginData`, there in every run:
rig-memory `Files` (`write_file`, `read_file`, `files`), names starting
with the plugin's (`vim/registers.json`). `session` is in the session's
store, copied by a fork and removed with it (in memory without a saved
session), `project` this user's for the project, whichever session works
in it (`$RIG_HOME/projects/`, never the checkout), and `user` every
session's (`$RIG_HOME/data/`, where the terminal view keeps the prompt
history).

**Secrets.** A secret a plugin needs before the session is restored, such
as a store's passphrase, is asked for in `build` with
`app.add_secret(Secret::new(name, help))` and read from `Secrets`. A
launcher run at a terminal asks for each one its environment
(`RIG_SECRET_<NAME>`) lacks, once, and passes it to the agent's process
alone: no process rig starts gets it (builds, the shell, `Processes`, the
editor). That keeps it from tools, the model's commands and helper
programs; not from the same user, nor from plugin code, which is trusted.

**Exiting.** A plugin that ends the run triggers `Exit(reason)`:
`ExitReason::Quit`, `Reload` (restart on the staged build),
`Switch(Some(id))` or `Switch(None)` (another session, or a new one),
`Signal` or `Failed(code)`; no plugin writes `AppExit` itself. The first
reason is the `ExitReason` resource. Each exit asked for is an app-level
effect, an `ExitRequest`, and the app exits when the first one settles. A
plugin delays the exit by holding it, with a deadline of its own: nothing
cancels an exit, and a SIGTERM waits as long as the hold. A second exit
meanwhile is a second effect, which a holder may take as "stop waiting".

`FirstStart` is a resource only in a session's first app (a fork's too),
not after `/reload` or `/resume`. `ForkSession { before, asker }` copies
this session into a new one (`SessionForked` names it and the `asker`; the
copy names its parent and where it was cut, `ForkedFrom::of(&*store)`),
and `DeleteSession(id)` removes another, its store wherever kept, finished
by the next start if a crash cut it short (`SessionDeleted`).
rig-sessions' `NameSession { session, name }` names a session in
`/resume`.

```rust,no_run
use std::io;
use std::sync::Arc;

use rig_harness::harness_protocol::SessionDir;
use rig_harness::prelude::*;
use rig_harness::rig_memory::DirStore;

/// Keeps every session's files on the disk mounted at `/vault`, and
/// spilled output and pasted images nowhere.
#[derive(Default)]
pub struct VaultPlugin;

impl Plugin for VaultPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Scratch(None))
            .insert_resource(SessionStores::new(|dir: &SessionDir| {
                let id = dir.id().ok_or_else(|| io::Error::other("not a session of RIG_HOME"))?;
                Ok(Arc::new(DirStore::open(format!("/vault/{id}"))?) as Arc<dyn Store>)
            }));
    }
}
```

```rust,no_run
use rig_harness::prelude::*;

#[derive(Component)]
struct NotesHold;

fn build(app: &mut App) {
    app.add_systems(AgentPass, hold_exit.in_set(Stage::Gate))
        .add_observer(give_up)
        .add_command("fork", "Continue in a copy of this session", fork)
        .add_systems(Update, switch);
}

/// Holds the first exit until the notes are written, at most 10 s.
fn hold_exit(
    exits: Query<Entity, (With<ExitRequest>, Undecided, PluginEffects)>,
    mut held: Local<bool>,
    mut commands: Commands,
) {
    // Once: a second exit asked for passes, as "do not wait".
    if let (false, Some(exit)) = (*held, exits.iter().next()) {
        *held = true;
        let deadline = WakeAt::after(Duration::from_secs(10));
        commands.spawn((Name::new("writing the exit notes"), NotesHold, HoldOf(exit), deadline));
        // ... write the notes, then despawn the hold.
    }
}

/// `WakeAt` only marks its entity `Due`: the deadline releases the hold.
fn give_up(due: On<Add<Due>>, holds: Query<(), With<NotesHold>>, mut commands: Commands) {
    if holds.contains(due.entity) {
        commands.entity(due.entity).despawn();
    }
}

/// `/fork why`: continue in a copy of this session, which keeps the reason.
fn fork(In(args): In<CommandArgs>, data: Res<PluginData>, mut commands: Commands) {
    if data.session.write_file("forks/why.txt", args.args.as_bytes()).is_ok() {
        commands.trigger(ForkSession::default());
    }
}

/// Switches to each fork once it is written.
fn switch(mut forked: MessageReader<SessionForked>, mut commands: Commands) {
    for fork in forked.read() {
        commands.trigger(Exit(ExitReason::Switch(Some(fork.to.clone()))));
    }
}
```

# Adding to another plugin's lists

A plugin that owns a list builds it from a query over entities other
plugins spawn, as the system prompt is built from `PromptSection`s: spawn
one with your text and keep it current. An empty text, or `Disabled`,
leaves it out, and despawning it takes it back.

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
`ProviderId::declare`, `Connector::provide`, `catalog_mut`) at any time,
for `--model`, `/model` and subagents to take like any other.

Spending is data too. rig-telemetry adds each settled model call's usage
to its agent's `Spending`, the session's `SessionSpending` (both saved)
and its turn's `TurnSpending`, per model and per purpose (`turn`,
`compaction`, a plugin's `Effect::with_purpose`), and shows the session's
in its `/resume` line (the `spent` label); a plugin that pays for work
that is no model request `record`s it. A `--print` run has nobody to
answer questions, so a plugin applies its policy for a run nobody answers;
`--answer=ID[,ID…]` answers each with the first of those choices.

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
        decision.decide(Decider::plugin("budget"), Outcome::refused(why));
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

# Turning off or replacing what another plugin added

Tools, slash commands, renderers, key bindings, options, fronts and prompt
sections are entities. `Disabled` on one turns it off and frees its name
for a replacement (a second one of a taken name is refused), in
`Plugin::finish`, which runs once every `build` did. What a plugin spawns
in `build`, `finish` and `cleanup` is marked `ProvidedBy` it; what it
spawns later it marks itself (`Res<PluginEntity<MyPlugin>>`). Changing how
another plugin's tool runs is rarely a replacement: a Prepare or Gate
system on its calls does it, a Review system changes its results, and a
default tool may have a switch, such as `rig_coding_tools::ShellOutput`. A
plugin's stage systems in a public set are turned off with a run
condition, such as `app.configure_sets(AgentPass, AttachSystems.run_if(||
false))` in `finish`, and another plugin's status item by its name ([the
status line](#the-status-line)).

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
        // `app.add_tool(MyShell)` would replace it. A command is found by its
        // `Name` ("/help"), a renderer by its ("renderer:shell").
    }
}
```

What a tool tells the model is data on its entity too, and another plugin
may change it, in `finish` or at any time: its `ToolDef` (the rig-core
`ToolDefinition`: name, description, the JSON schema of its `parameters`)
and its `ToolRules` (lines of the system prompt; edit only yours). A
property added to another tool's `parameters` ([extending
`task`](#extending-task) adds one) is one the model may send; the tool
still parses its arguments its own way, so the plugin that added it takes
it out of the call's `ToolRequest` in Prepare if the tool would refuse it.

# Testing a plugin

A plugin's test builds a small app: the agent core (`AgentPlugin`,
`JournalPlugin`), the plugin, and what it needs, run frame by frame with
scripted models. `rig-harness-test-support` (in the rig repository, a path
dev-dependency) has the parts: an app on a scratch session directory or a
store in memory (`app_in`, `app_on`), scripted models (`scripted_models`,
`reply`, `calls`), running frames until a condition or moving the clocks
(`run_until`, `advance`, `pass_wall_clock`), what happened (`collect`,
`log_ended`, `log_tool_outputs`), answering questions (`AnswerAsks`,
`answer_open`), and the terminal view with no terminal (`tui_view`, and
`press(&mut app, "esc 0 d w")` for one frame's keys). A plugin with stage
systems has one more test,
`rig_defaults_test_support::assert_stages_ordered(MyPlugin)`, which fails
where it and the default plugins write the same data
[unordered](#ordering-against-other-plugins). How long each system takes
is `SystemTimings` (`systems()`, `by_plugin(&plugins)`), filled once a
plugin crate depends on rig-harness with `features = ["trace"]` (Bevy's
span around each system run).

```rust,no_run
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::SessionId;
use rig_harness::prelude::*;
use rig_harness_test_support::{
    Ended, ToolOutputs, app_in, calls, first_agent, log_ended, log_tool_outputs, reply,
    run_until, scratch_home, scripted_models,
};

/// The gate under test: refuses a `write` outside `src/`.
fn src_only(mut calls: Query<(&ToolRequest, &mut Decision), (Undecided, ToolCalls)>) {
    for (request, mut decision) in &mut calls {
        let path = request.call.function.arguments["path"].as_str().unwrap_or_default();
        if request.call.function.name.as_str() == "write" && !path.starts_with("src/") {
            decision.decide(Decider::plugin("src-only"), Outcome::refused("write only under src/"));
        }
    }
}

/// A `write` that only says it wrote.
fn wrote(called: On<ToolCalled<serde_json::Value>>, mut calls: Query<&mut Decision, ToolCalls>) {
    if let Ok(mut call) = calls.get_mut(called.call) {
        call.decide(Decider::plugin("write"), ToolResult::success("wrote".into()));
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
        .add_plugins((AgentPlugin, JournalPlugin))
        .add_systems(AgentPass, src_only.in_set(Stage::Gate))
        .add_open_tool("write", "Write a file.", ToolOptions::default(), wrote)
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    app.world_mut().trigger(SetModel { entity: agent, model: "test/model".to_owned() });
    app.world_mut().trigger(Deliver::new(agent, "Fix the readme.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| !world.resource::<Ended>().0.is_empty());
    let outputs = &app.world().resource::<ToolOutputs>().0;
    let output = outputs.get("c1").map(String::as_str).unwrap_or_default();
    assert!(output.contains("only under src/"), "{output}");
}
```

# Fronts, options and commands

A front, such as the terminal view, `--print`, a window or a chat bot,
offers itself with `app.add_front(..)`; one front takes each run and alone
owns stdin and stdout, and it sets itself up in `finish` only if
`app.runs_front(name)`. It sends what the user types as `Deliver` from
`Author::user()`, shows `Notice`s (an `Error` fails a `--print` run or
refuses a command), each agent's `Conversation` (or `Committed` changes)
and [notes](#the-transcript-notes-looks-and-scrolling), its reply as it
streams ([`StreamDelta`](#what-a-model-streams)), sets `Shown` to the
agent the user looks at, and may spawn an `Answerer` and answer questions
with [`Pick`](#asking-the-user-questions-and-pickers). A front nobody sits
at drives its run with `app.run_headless(Some(prompt))`, which ends once
no turn runs and nothing is held, unless an entity has `KeepRunning`: a
plugin that sends the agent work over time, such as a file watcher, spawns
one (the run may then start with `None`) and despawns it to end the run.
`HeadlessTurnEnded` carries each answer as the work settles, and
`HeadlessEnded` comes last.

A plugin's own options are declared with `app.add_flag(Flag::new(name,
help))`, `.value("N")` for one that takes a value (`rig --name VALUE` or
`--name=VALUE`); the agent refuses an option no plugin declared. The
plugin reads them from `Invoked` (`flag(name)`, `flag_value(name)`) from
`build` on, as the [settings](#settings) example does.

A stopped turn is a `Committed` change `Change::Halt { halt:
Some(Halt::Stopped) }` (`halted()` on the conversation): a front shows a
stopped request from that, never from text. The kernel's note after it for
the model is its own entry by `Author::rig()`
(`EntryRef::is_stopped_note`).

## Commands of your own

A command is a front of its own: `Front::new("stats").on_command("stats",
"[--all]", "Count the saved sessions")` takes every run of `rig stats …`,
and the words after the name are `Invoked::command()`. A command whose
front is not `on_terminal()` runs headless: the launcher passes stdin,
stdout, stderr and the exit code through (`ExitReason::Failed(n)` exits
with `n`) and never restarts it; `headless()` is only for runs that name
no command or option. `sessionless()` leaves no session behind.

The launcher learns a build's commands, options and secrets by asking the
built agent: `rig_harness::run(app)`, which the generated `main.rs` ends
with, builds the app, writes what its plugins declared, and runs no frame,
so `build` only declares, and work goes in `finish` or systems. `rig help`
lists what the last build declared, `rig help stats` one command's usage.

```rust,no_run
use rig_harness::harness_protocol::SessionId;
use rig_harness::prelude::*;

/// `rig stats`: how many sessions there are to resume.
#[derive(Default)]
pub struct StatsPlugin;

impl Plugin for StatsPlugin {
    fn build(&self, app: &mut App) {
        let stats = Front::new("stats").sessionless();
        app.add_front(stats.on_command("stats", "", "Count the saved sessions"));
    }

    fn finish(&self, app: &mut App) {
        if app.runs_front("stats") {
            app.add_systems(Startup, stats);
        }
    }
}

/// Prints the count, then exits.
fn stats(home: Res<RigHome>, mut commands: Commands) {
    let sessions = std::fs::read_dir(home.0.sessions()).into_iter().flatten().flatten();
    let ids = sessions.filter_map(|dir| dir.file_name().to_str()?.parse::<SessionId>().ok());
    // What a session spent, with `stores: Res<SessionStores>`: `stores.open(&home.0.session(id))
    //     .ok().and_then(|s| rig_sessions::Meta::of(&*s)?.labels.get("spent").cloned())`
    println!("{} saved sessions", ids.filter(|id| home.0.session(id).is_saved()).count());
    commands.trigger(Exit(ExitReason::Quit));
}
```

# The transcript: notes, looks and scrolling

A plugin puts a note in an agent's transcript as an
[aside](#changing-history), `Note::new(title, text).aside(author, after)`,
after a message (`after`, a `MessageId`, or the summary that replaced it),
its title shown collapsed and its text expanded. Every front draws it, no
model sees it, and it comes back on `/resume`, `/reload` and in a fork,
unless `unsaved()`; `Note::of(aside)` reads one.

The terminal view draws notes, and what the conversation holds from a plugin (such
as rig-compaction's summary), with the look `add_entry_renderer(plugin, ..)` gives
that plugin's author. Its `TranscriptView` is what plugins read and set: the items
`expanded` and `selected` (Alt+Up and Alt+Down select, Ctrl+O expands), the
`highlights` lit wherever their text is drawn (`match_case` or not) or only at
their own `spans`, such as a regex's matches (where two overlap, the first placed
wins), and where it is scrolled (`top()`). `ScrollTo { agent, item, row }` scrolls
to a `TranscriptItem` (a `Message`, a tool `Call`, a `Note` by its aside's id or a
`Notice` by its index in `notices()`): to a row of it (`ItemRow::Top(n)`), its
first (`Reveal`) or that of its `n`th lit match (`Lit(n)`); `item: None` follows
the newest output. A notice's `slot(agent, conversation)` places it in a
transcript. `DrawnTranscript`, once added, holds each row the last frame drew, by
item, its `lines()` unwrapped, the matches lit (`lit_in(item)`, as `Lit(n)` counts)
and cut items (`is_cut`). `TranscriptLayout::lay_out` lays any agent's out so,
undrawn, cut or whole.

```rust,no_run
use rig_harness::prelude::*;
use rig_tui::ratatui::style::{Style, Stylize};
use rig_tui::ratatui::text::Line;
use rig_tui::{AppToolRenderersExt, Highlight, ItemRow, ScrollTo, TranscriptItem, TranscriptView};

fn build(app: &mut App) {
    app.add_command("mark", "Mark the newest message: /mark why", mark)
        .add_command("find", "Light a text and show the last message with it", find)
        .add_entry_renderer("mark", |entry| {
            let mut lines = vec![Line::from(format!("★ {}", entry.title)).yellow()];
            if entry.expanded {
                lines.extend(entry.text.lines().map(|line| Line::from(line.to_owned())));
            }
            lines
        });
}

/// `/mark why`: a note after the agent's newest message.
fn mark(In(args): In<CommandArgs>, mut agents: Query<&mut Conversation>) {
    let Ok(mut conversation) = agents.get_mut(args.agent) else { return };
    let after = conversation.messages().len().checked_sub(1).and_then(|at| conversation.id(at));
    conversation.add_aside(Note::new("marked", args.args).aside(Author::plugin("mark"), after));
}

/// `/find text`: lights it, and scrolls to it in the last message the user
/// typed it in.
fn find(
    In(args): In<CommandArgs>,
    agents: Query<&Conversation>,
    mut view: ResMut<TranscriptView>,
    mut commands: Commands,
) {
    let Ok(conversation) = agents.get(args.agent) else { return };
    let typed_it = |at: &usize| typed(conversation, *at).is_some_and(|text| text.contains(&args.args));
    let hit = (0..conversation.messages().len()).rev().find(typed_it);
    if let Some(id) = hit.and_then(|at| conversation.id(at)) {
        let item = Some(TranscriptItem::Message(id));
        commands.trigger(ScrollTo { agent: args.agent, item, row: ItemRow::Lit(0) });
    }
    let style = Style::new().on_yellow();
    view.highlights = vec![Highlight { text: args.args, style, match_case: true, ..Default::default() }];
}
```

# The status line

The terminal view's row under the transcript shows the `StatusItems` of
the agent shown and of its running turn, which every agent and turn has,
and the app's `AppStatus`. A plugin shows an item in a system in
`StatusSystems::Show` that runs when what it shows changed:
`items.show(item)` puts it at its place (side and order) in place of the
item of its name, and an empty item hides it; two at one place both show.
Another plugin replaces or hides an item by name (`items.hide(name)`) in
`StatusSystems::Edit`, as often as its owner shows it. It goes behind a
default `tui` feature, and a plugin that writes `AppStatus` inits it,
since the view may not be added. The default plugins' items, and how long
each stays when the line is too narrow (`keep`: lowest first, `u8::MAX`
never):

| side | order | name | item | keep | plugin |
|---|---|---|---|---|---|
| left | 10 | `session` | the session's name (`AppStatus`) | 2 | rig-sessions |
| left | 20 | `title` | `⤷` a spawned agent's name | 14 | rig-basics |
| left | 30 | `model` | the model | always | rig-models |
| left | 40 | `reasoning` | the reasoning setting | 12 | rig-models |
| left | 50 | `status` | idle, thinking, … | always | rig-activity |
| left | 60 | `subagents` | the agent's subagents at work | 15 | rig-basics |
| left | 70 | `others` | the other agents at work | 13 | rig-basics |
| left | 80 | `turn` | what the turn spent (on the turn) | 11 | rig-telemetry |
| left | 90 | `reload` | the rebuild of `/reload` (`AppStatus`) | 16 | rig-reload |
| right | 10 | `tokens` | tokens in and out | 4 | rig-telemetry |
| right | 20 | `cache` | cached input | 1 | rig-telemetry |
| right | 30 | `cost` | cost | 3 | rig-telemetry |
| right | 40 | `context` | the context in use; shown again when it changes | 5 | rig-telemetry |

```rust,no_run
use rig_harness::prelude::*;
use rig_tui::{Side, StatusItem, StatusItems, StatusSystems, Tone};

/// After the status; gone before the reasoning setting.
const MESSAGES: StatusItem = StatusItem::new("messages", Side::Left, 55, 10);

#[derive(Default)]
pub struct MessagesItemPlugin;

impl Plugin for MessagesItemPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PostUpdate, (show.in_set(StatusSystems::Show), no_cost.in_set(StatusSystems::Edit)));
    }
}

fn show(mut agents: Query<(&Conversation, &mut StatusItems), Changed<Conversation>>) {
    for (conversation, mut items) in &mut agents {
        let text = format!("{} messages", conversation.messages().len());
        items.show(MESSAGES.says(text, Tone::Dim));
    }
}

/// No cost in the line: rig-telemetry's `cost` item, hidden each time it shows.
fn no_cost(mut shown: Query<&mut StatusItems>) {
    for mut items in &mut shown {
        if items.iter().any(|item| item.name == "cost") {
            items.hide("cost");
        }
    }
}
```

# What agents do and say, in a terminal panel

Each agent is an entity with an `Agent`, a `Name`, an `AgentId` and a
`Conversation`; a subagent is `SpawnedBy` its parent (`Spawned`).
`rig_activity::Activity` says what an agent does now, and
`MessageReader<Committed>` sees every change of every conversation.

A `TuiPanel` entity takes a side of the transcript (`Top`, `Bottom`,
`Left`, `Right`, with a ratatui `Constraint`), rows under the input
(`BelowInput`), or a box over the screen (`Over`). The plugin draws it
into its `PanelCanvas` in `TuiSystems::Draw`, which runs only for a frame
that is drawn, and writes a `RequestRedraw` when its own state changed.
`TuiScreen` is the terminal's size, `Focused` marks the agent shown, and
all are there in any run, `--print` too.

```rust,no_run
use rig_activity::Activity;
use rig_harness::prelude::*;
use rig_tui::ratatui::layout::Constraint;
use rig_tui::ratatui::widgets::{Block, Paragraph};
use rig_tui::{PanelCanvas, Placement, TuiPanel, TuiSystems};

#[derive(Default)]
pub struct AgentsPanelPlugin;

#[derive(Component)]
struct AgentsPanel;

impl Plugin for AgentsPanelPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn)
            .add_systems(PostUpdate, draw.in_set(TuiSystems::Draw));
    }
}

fn spawn(mut commands: Commands) {
    commands.spawn((AgentsPanel, TuiPanel::new(Placement::Right(Constraint::Length(32)))));
}

/// Each agent, a subagent under its parent, with what it does and how
/// long its conversation is.
fn draw(
    mut panels: Query<&mut PanelCanvas, With<AgentsPanel>>,
    agents: Query<(&Name, &Activity, &Conversation, Has<SpawnedBy>)>,
) {
    let line = |(name, activity, conversation, sub): (&Name, &Activity, &Conversation, bool)| {
        let indent = if sub { "  " } else { "" };
        format!("{indent}{name} {}, {} messages", activity.status, conversation.messages().len())
    };
    let lines: Vec<String> = agents.iter().map(line).collect();
    for mut canvas in &mut panels {
        canvas.render(Paragraph::new(lines.join("\n")).block(Block::bordered().title("agents")));
    }
}
```

# Keys and the input in the terminal view

The terminal view (`rig_tui`) takes part in the input as data, there in
any run and inert without the view:

- **The frame's keys.** `TerminalEvents` is what the terminal reported
  this frame (`KeyInput::Event`) and the actions plugins put in
  (`KeyInput::Action`), in order. A system in `InputSystems::Filter`
  (`PreUpdate`) reads, changes, drops or adds items before any binding or
  the input sees them, with `TuiInputState` (the open completion, the
  question or picker shown) as it was before them; `Dispatch` does the
  rest. A modal input is one filter, and `/` starts a slash command
  unless a filter takes it.
- **Keys.** Every key the view acts on is a `KeyBinding` entity, its own
  editing keys included. `app.add_key_binding(binding, observer)` binds a
  key or a sequence (`ctrl+x ctrl+e`), heard as `KeyPressed`
  (`run_command("compact")` runs a slash command). The first binding of a
  key stays, a global one shadows the input's, and `Disabled` frees its
  keys. A binding with an `InputAction` (`Up`, `Send`, …) does that
  built-in action. `KeyFocus(Some(panel))` gives a `TuiPanel` the keys
  until Esc or the panel goes.
- **Completion.** An `ArgCompletion::new("/model").candidate(text,
  detail)` entity offers candidates for a command's argument (`.at(n)`).
- **The input.** `TuiInput` is the draft (`text()`, `set`, `insert`,
  `take`, …). `PromptHistory` is the prompts Up and Down browse, kept in
  `PluginData.user`: the view appends each one sent (`Sent`, which a
  plugin writes too) in `HistorySystems::Record` and saves in
  `HistorySystems::Save`; a system between the two filters what is kept.
- **The terminal.** `TerminalFocus` says whether it has the user's focus,
  a `TerminalWrite(bytes)` message goes to the terminal between frames,
  `CopyToClipboard(text)` copies (OSC 52, and this machine's clipboard,
  which rig-tools' `clipboard` reads and writes), and a
  `RunInTerminal(command)` entity runs a program with the terminal,
  without the session's secrets, and gets its `TerminalExit`.

This plugin writes the draft in `$EDITOR` on Ctrl+E, makes Ctrl+K browse
back as Up does, keeps prompts with a secret out of the history, and
opens its search box on `/` when the input is empty:

```rust,no_run
use std::path::PathBuf;
use std::process::Command;

use rig_harness::prelude::*;
use rig_harness::rig_core::redact;
use rig_tui::crossterm::event::{Event, KeyCode};
use rig_tui::{
    AppKeyBindingsExt, HistorySystems, InputAction, InputSystems, KeyBinding, KeyFocus, KeyInput,
    KeyPress, KeyPressed, PromptHistory, RunInTerminal, TerminalEvents, TerminalExit, TuiInput,
    TuiInputState,
};

#[derive(Default)]
pub struct DraftKeysPlugin;

/// A program writing the draft in this file.
#[derive(Component)]
struct Draft(PathBuf);

/// The plugin's search box, a `TuiPanel` it draws (not shown).
#[derive(Component)]
struct SearchBox;

impl Plugin for DraftKeysPlugin {
    fn build(&self, app: &mut App) {
        let scrub = no_secrets.after(HistorySystems::Record).before(HistorySystems::Save);
        app.add_systems(Update, (take_draft, scrub))
            .add_systems(PreUpdate, slash_searches.in_set(InputSystems::Filter));
        app.add_key_binding(KeyBinding::new(KeyPress::ctrl('e'), "Write the draft in $EDITOR"), edit);
        app.world_mut().spawn((KeyBinding::new(KeyPress::ctrl('k'), "Up"), InputAction::Up));
    }
}

/// `/` on an empty input gives the search box the keys; elsewhere it is typed.
fn slash_searches(
    mut events: ResMut<TerminalEvents>,
    (input, state): (Res<TuiInput>, Res<TuiInputState>),
    boxes: Query<Entity, With<SearchBox>>,
    mut focus: ResMut<KeyFocus>,
) {
    let slash = |item: &KeyInput| matches!(item, KeyInput::Event(Event::Key(key)) if key.code == KeyCode::Char('/'));
    let idle = input.is_empty() && state.completion().is_none() && state.picking().is_none();
    if idle && events.0.first().is_some_and(slash) && let Some(search) = boxes.iter().next() {
        events.0.remove(0);
        focus.0 = Some(search);
    }
}

/// A prompt with an API key in it is never saved.
fn no_secrets(mut history: ResMut<PromptHistory>) {
    history.entries.retain(|prompt| !redact::has_secret(prompt));
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
```

# A window

`rig_harness::windowed(DefaultPlugins)` is Bevy's `DefaultPlugins` without
what the agent already has; its winit plugin then runs the app's loop, and
in `finish` the plugin points the agent's `Wake` at it. The crate depends
on `bevy` at exactly the agent's version (`version = "=0.20.0",
default-features = false, features = ["ui"]`). A window that is the front
offers itself with `on_flag`.

```rust,ignore
use std::time::Duration;

use bevy::prelude::*;
use bevy::window::ExitCondition;
use bevy::winit::{EventLoopProxyWrapper, UpdateMode, WinitSettings, WinitUserEvent};
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
        let window = WindowPlugin { exit_condition: ExitCondition::DontExit, ..default() };
        // Frames when woken, and at least every second as without a window.
        let mode = UpdateMode::reactive_low_power(Duration::from_secs(1));
        app.add_plugins(rig_harness::windowed(DefaultPlugins.set(window)))
            .insert_resource(WinitSettings { focused_mode: mode, unfocused_mode: mode });
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
```
