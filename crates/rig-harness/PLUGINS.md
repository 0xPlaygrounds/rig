Writing a plugin for the rig agent: the rules every plugin follows, then a
short, copy-ready example per extension point.

A plugin is a Bevy plugin: a type implementing `Plugin + Default` in a crate
that depends on `rig-harness`. The agent's own tools, commands, subagents,
compaction, sessions and terminal view are plugins of the same kind, listed in
`RIG_HOME/plugins.toml` and built only on the same public API. A plugin that
needs another adds it unless `app.is_plugin_added::<P>()`.

Every name below comes from `use rig_harness::prelude::*;` (Bevy's preludes,
the kernel's components and stages, the harness's services) or from the crate
an example names: `rig_basics` (the default plugins, and the services some
plugins add, a module each) or `rig_tui` (with `rig_tui::ratatui` and
`rig_tui::crossterm`); `rig_harness::{rig_core, rig_memory, rig_tools}` are
re-exported. Every one of these types is reflected, so the optional
`rig-inspect` plugin's `inspect` tool lists them in the running agent
(`registry.schema` gives a type's fields, `world.query` with `ProvidedBy` what
each plugin added): ask it before reading rig's source.

| To | Read |
|---|---|
| add a tool, or draw its calls | [A tool](#a-tool) |
| refuse, approve, preview or sandbox tool calls | [Gating and reviewing tool calls](#gating-and-reviewing-tool-calls) |
| ask the user, or offer a list to pick from | [Asking the user](#asking-the-user-questions-and-pickers) |
| add a slash command | [A slash command](#a-slash-command) |
| change or take over what the user types | [What the user types](#what-the-user-types) |
| change the system prompt, route or adapt model calls, add a provider | [Model requests](#model-requests) |
| read replies as they stream | [What a model streams](#what-a-model-streams) |
| end, carry on or stop a turn | [How a turn ends](#how-a-turn-ends) |
| make a plugin's own work gated and recorded | [Effects of a plugin's own](#effects-of-a-plugins-own) |
| spawn subagents, extend `task` | [Subagents](#subagents) |
| read, redact or replay the effect log | [The effect log](#the-effect-log-reading-redacting-replaying) |
| read settings or options | [Settings](#settings) |
| keep state across restarts | [Saved state](#saved-state-counting-every-tool-call), [Data on a session](#data-on-a-session) |
| edit the conversation, review a summary, replace compaction | [Changing history](#changing-history) |
| turn off or replace a default tool, command or plugin | [Turning off or replacing](#turning-off-or-replacing-what-another-plugin-added) |
| wait, poll, debounce | [Time](#time) |
| read and watch files, start processes | [The workspace](#the-workspace-files-and-child-processes) |
| keep sessions elsewhere, ask for a secret, hold the exit, fork | [Sessions](#sessions-their-store-secrets-and-exiting) |
| add to `/model`, `/resume`, `/agents`, `/usage`, `--print` | [Adding to another plugin's lists](#adding-to-another-plugins-lists) |
| test a plugin | [Testing a plugin](#testing-a-plugin) |
| a front, `rig <command>`, a run nobody sits at | [Fronts, options and commands](#fronts-options-and-commands) |
| draw in the terminal | [The transcript](#the-transcript-notes-looks-and-scrolling), [status line](#the-status-line), [panels](#what-agents-do-and-say-in-a-terminal-panel), [keys](#keys-and-the-input-in-the-terminal-view), [a window](#a-window) |

# Words

- **Crates.** The **kernel** is rig-ecs (never "runtime" or "core"), the
  **harness** rig-harness, the **launcher** the std-only `rig` binary; a
  **plugin** is a Bevy plugin `plugins.toml` lists. Sources: `crates/<name>`
  for libraries, kernel and harness; `plugins/rig-basics/src/<module>` for
  the default plugins, `plugins/<name>` for rig-tui and the optional ones.
- **Agents.** An **agent** is an entity with `Agent`, a conversation and a
  model; a **subagent** is one another agent spawned (`SpawnedBy`). The **main
  agent** is the first unspawned one (`main_order`), the **shown agent** the one
  a front shows and sends typed text to (`ShownAgent`); "focus" is only keyboard
  focus.
- **Conversations.** A **conversation** is ordered entries, asides and a halt
  (rig-core's `Conversation`). An **entry** is one message with a `MessageId`,
  an author and a made-time. An **aside** is a typed side value after an entry,
  saved and rewritten with the conversation and never sent to a model; a
  **note** is an aside of kind `note` a front draws (`Note`). A **change** is
  one edit (`Change`), a **change record** one with who and when
  (`ChangeRecord`). The **halt** says why the last user message is unanswered.
- **Turns and forks.** A **turn** runs from a user message to the reply that
  ends it, at most one per agent; its model call is the **turn call**
  (`TurnCall`), and a **side call** is built from an agent outside any turn
  (`SideCall`). To **stop** is any early end of a turn or call: the user's
  (`StopTurn`), a plugin's, a failure's or a restart's; an **interrupted**
  call is a restored one a crash cut short (`Interrupted`). A **fork** is a
  new agent of the session that goes on from another's conversation up to a
  turn boundary (`ForkAgent`).
- **Effects.** An **effect** is a unit of work as an entity moving through the
  **stages** (Arrive, Review, Commit, Admit, Prepare, Gate, Start); a **phase**
  is an ordered part of one (`PrepareSystems`, `ReviewSystems`, `GateSystems`),
  a **hook** a plugin's system in a stage. A **call** is a model or tool call,
  never a **task**: off-thread work on an entity (`Running` becomes `Done<T>`).
  A **frame** is a Bevy update, a **pass** one run of `AgentPass`. **Ready to
  start** and **ready to settle** are Gate's and Review's filters
  (`ReadyToStart`, `ReadyToSettle`). A **decision** is a gate's first-wins
  answer, applied at Start (`Decision`); a **hold** is any entity `HoldOf` an
  effect; to **settle** is Commit's last step, recording the effect. An
  **answer** is what a handler or decision gave (`effect::Answered`), the
  **outcome** the answer or an error (`Outcome`). A **handler** serves an
  effect, a **layer** wraps one (`Intercept`: `BeforeServe`, `AfterServe`), an
  **adapter** is a system that changes the model a call goes to
  (`Adapted`). An **open tool**'s calls are
  answered by observers; a **footprint** what a tool call may read and change;
  **facts** are typed values a handler publishes with its answer (`Published`).
- **Messages.** A **message** is text sent to an agent (`Incoming`) to
  **steer**, **queue** or **inform** (no answer needed); the **inbox** holds
  accepted, unanswered ones. A **notice** is a transient line for the user
  (`Notice`), never saved.
- **Sessions.** A **session** is everything a run keeps, in a **store**
  (`Store`; `SessionStore` is the running one, `StoreOpener` opens others). A
  **log** is one agent's append-only file, or the `session` log; writing it is
  saving (`SessionLogPlugin`), and "log" alone never means the effect log or
  `agent.log`. A **record** is what is saved about a thing (`EffectRecord`,
  `TurnRecord`, `ChangeRecord`, `LogRecord`, …). A **saved value** is a
  `Saved` component or resource under its **saved key**. The **listing** is a
  session's `SessionMeta` (`meta.json`), a **listed value** a saved value
  copied into it. A session of another format is refused untouched; a saved
  value that no longer reads is skipped with a notice and replaced at its next
  save. A **rewrite** maps every saved value of a stopped session in place.
  **Scratch** is a session's plain files outside its store, a **spill** a
  tool's whole output kept when it was cut, and **plugin files** a plugin's
  own files per session, project or user (`PluginFiles`).
- **Compaction** replaces older entries by a **summary**, split at a **cut**;
  the **summary state** is what the next one builds on (`SummaryState`). A
  **checkpoint** or **snapshot** is the files as they were before a call or
  turn; rig-cassette's replay marker is a **resume point**.
- **Running.** A **front** is how the user drives the agent: the terminal view,
  `--print`, a window. A **question** is an `Ask` effect, a **picker** a
  filterable list whose items run commands. A **command** is `/name args` or a
  launcher subcommand, a **setting** a `settings.toml` value, a **secret** a
  value passed in the environment. The **reasoning setting** is what an agent
  sends for reasoning (`ReasoningSetting`); **effort** is a reasoning level
  (`rig_core::completion::Effort`).

# Making one

```sh
rig plugin new hello    # RIG_HOME/plugins/hello, listed in plugins.toml
rig plugin add viz::VizPlugin --path ~/viz   # an entry for an existing crate
rig plugin remove viz::VizPlugin             # take an entry out; its crate stays
rig plugin list                              # every entry, on or off, with its comment
rig plugin disable viz::VizPlugin            # comment the entry out; `enable` undoes it
rig plugin check --build  # build the agent as /reload would, and check its stage order
```

A commented-out entry is a disabled plugin, and the comment right above an
entry is its description, which `rig plugin list` and `/plugins` show.
`rig plugin new` writes a working, commented slash command and the entry
`crate = "hello"` (the package), `path = "plugins/hello"` (or `git`, with
`branch` or `rev`, or `version`) and `plugin = "hello::HelloPlugin"`.

Each plugin crate is its own workspace in `RIG_HOME/plugins/<name>`, depending
on `rig-harness` and the rig plugin crates whose types it uses (`rig-basics`,
`rig-tui` for a terminal panel) by version, and on `bevy_ecs` and
`bevy_reflect` (or `bevy`) at the agent's exact version. A plugin that only
uses rig-basics' services and contract types (`NeedsApproval`, `Preview`,
`CheckpointAlso`, `KeepRunning`) depends on it with
`default-features = false`: its default `tui` feature builds rig-tui and
ratatui. The agent builds from
a rig checkout (`RIG_SOURCE`), which a `[patch.crates-io]` table points at.

`/reload` (or the `reload` tool) rebuilds and restarts in the session once
no turn runs; a failed build keeps the running one, a crash rolls back. A
reload the model asked for reports the outcome to that agent in one message
that starts a turn (`rig_basics::reload`'s `ReloadAsked`).

# How a plugin steps in: effects and stages

Every step of an agent's work is an entity, an **effect**, whose
components are its request and, once answered, its `Outcome`: a model call
(`ModelRequest`), a tool call (`ToolRequest`), a message to an agent
(`Incoming`), a question for the user (`Ask`), or a plugin's own
(`EffectRequest`). Each works `ForAgent` an agent (or for the app) and is
recorded once it settles (`EffectRecorded`). Each frame, `AgentLoop` in
`Update` runs the `AgentPass` schedule, one pass through the stages in
this order, until nothing moves:

| Stage | Who | What happens there |
|---|---|---|
| `Arrive` | kernel | finished work becomes an `Outcome`; a failed model call gets a `RetryVerdict` |
| `Review` | kernel, then plugins | outcomes not settled yet (`ReadyToSettle`): read, edit in place, hold, or take a failure over; a turn's proposed end (phases: `ReviewSystems`) |
| `Commit` | kernel | redacts keys in tool answers, then settles what is answered and unheld (`Settled`), appends results, carries turns on or ends them |
| `Admit` | kernel | `Admitted` on every new effect; builds turn requests |
| `Prepare` | plugins | what was admitted this pass (`Added<Admitted>`): edit the request, once (phases: `PrepareSystems`) |
| `Gate` | plugins | as it opens, the kernel gives each tool call its `Footprint` and marks `Waiting` a call behind a conflicting one; then what is about to start (`ReadyToStart`): hold it or decide it (`Decision`) |
| `Start` | kernel | applies the decisions as `Outcome`s, then starts what is admitted, unheld, unanswered and undecided |

**The kernel's stages**, Arrive, Commit, Admit and Start, take plugin systems
too, in no order against the kernel's own there, so such a system writes only
its own data, never an `Outcome`, a `Decision` or a request. Arrive runs every
pass, even one where nothing moved, so it suits following running work: the
harness turns a running call's output into its `ToolProgress` there. In
Commit, Admit and Start a plugin reacts to what the pass did, in order with
the other stage systems where an observer would not be, such as a successor
agent spawned once a turn settled.

**A hook is an ordinary system in a stage** that queries stage data:
`app.add_systems(AgentPass, my_gate.in_set(Stage::Gate))`, ordered by its
stage's phase and against other plugins' public sets
([ordering](#ordering-against-other-plugins)). Every effect is admitted
before it starts, wherever it was spawned, so Prepare sees each once; Gate
each pass it is ready to start; Review an outcome each pass until it settles.

**An observer is for reactions that need no order**: counting, a status
item, a span, the plugin's own entities (`On<Add<Settled>>`). Observers of
one event run in no fixed order, so an observer never edits stage data
another plugin edits; it may decide an effect, since the first one wins.

**The conflict rule.** A hold is any entity `HoldOf(effect)`, its `Name`
saying why (`waiting for lease on src/a.rs`) in the fronts; holds add up. A
gate decides through the effect's `Decision`, naming who
(`decision.decide(Decider::plugin(NAME), Outcome::refused(..))`, or
`Decider::Person` for the user's choice in a front). The first decision wins,
so no later gate overturns a refusal; the kernel applies it as the `Outcome`
at Start and the record keeps `decided_by`. Allowing is doing nothing. A
reviewer edits the `Outcome` in place (`Query<&mut Outcome, ReadyToSettle>`)
once, with a marker, since Review sees it each pass. Plugins never insert
`Outcome`.

**Deciding once.** Until Start, a decided call is still `ReadyToStart` for
later gates. A gate that only reads (a path rule) may decide afresh each time.
A gate that asks someone or has side effects checks `decision.is_decided()`
first and marks the calls it handled (a `SparseSet` marker).

**Failing closed.** A stage system that fails or panics closes its stage
for that pass, whatever the app's error handler: a failed Prepare system
answers what that pass admitted with an error, a failed Gate system starts
nothing, a failed Review system settles nothing, and a notice names it.

**Taking over and trying again.** A reviewer that takes a failure over
spawns a successor with the request it needs and the same `CallOf`, which
the failed effect is `SupersededBy` (`commands.spawn(successor)
.add_related::<SupersededBy>(&[failed])`), or sends it again
(`commands.queue(resend(failed))`): the failed effect is recorded as
superseded, never in the conversation. Any number of effects may name one
successor, as a race's losers do. Slow work holds the outcome first.

**The loop's edge.** Work spawned, answered or released after `AgentLoop`
ran waits for the next frame, so a system that does so every frame runs in
`Update` `.before(AgentLoop)`. One that reacts to what the loop did that
frame (an agent it started, a call it settled) runs `.after(AgentLoop)`.

```rust,no_run
use rig_harness::prelude::*;
use rig_harness::rig_core::effect::Answered;
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
fn src_only(mut calls: Query<(&ToolRequest, &mut Decision), (ReadyToStart, ToolCalls)>) {
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
    mut calls: Query<(Entity, &ToolRequest, &mut Outcome), (ReadyToSettle, ToolCalls, Without<Shortened>)>,
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

Bevy reports two stage systems with no order between them that write the same
component, such as two gates that decide (`&mut Decision`); a plugin's test
makes it a failure ([`stage_conflicts`](#testing-a-plugin)). Holds and
spawns go through `Commands` and need no order. Queries of different kinds of
effect never meet when they say so with a kind filter (`ToolCalls`,
`ModelCalls`, `IncomingMessages`, `PluginEffects`), and two over plugin
effects when one names its marker (`With<Ask>`) and the other excludes it. Two
queries of one system that write the same component must rule each other out
so, or Bevy refuses the system (error B0001).

**Prepare and Review have phases**, chained in this order, so a plugin
orders a system by what it does, not by which plugin does what:

| Phase | What goes there | The default plugins' systems there |
|---|---|---|
| `PrepareSystems::Input` | what came in, rewritten before anything is read from it: a message's text, a tool call's arguments | |
| `PrepareSystems::Compose` | the parts that come from what a request names and from its agent | the files a message names (`@path`, in `rig_basics::coding_tools::AttachSystems`), the default system prompt, each coding tool call's place and handler (`PerCallSystems`) |
| `PrepareSystems::Edit` | the request edited as it will be sent: prompt, messages, tools, options, the model it goes to (`connection`), a call's `CallHandler` | |
| `PrepareSystems::Adapt` | the model a call goes to (`Adapted`), changed in place: its handler wrapped, its spec saying what the wrapped model takes | |
| `PrepareSystems::AdaptOutermost` | the same, after every other adapter: a change that sees what they made of the model, such as a replay that replaces it | |
| `PrepareSystems::Fit` | the request as sent, measured against the window of the model it goes to and fitted | compaction |
| `ReviewSystems::Edit` | an `Outcome`, a `RetryVerdict` or a proposed `TurnOutcome`, edited in place | compaction reading a summary or notes from its reply |
| `ReviewSystems::Act` | what was reviewed, acted on: a failed call taken over, a result followed up, a summary or notes reviewed | compaction's recovery from an overflow |

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
  against its set, and tests both in one app for
  [stage conflicts](#testing-a-plugin).

The sets of Gate systems that edit shared data:

| Set | Crate | What it does | Order against it |
|---|---|---|---|
| `GateSystems::FromRecord` | `rig_ecs` | as Gate opens, before every Gate system: decides effects from a record of an earlier run (a session replay), which holds each gate's decision then | nothing; a cached or dry-run answer is no record and goes in `After` |
| `GateSystems::After` | `rig_ecs` | after every Gate system: answers that stand in for running an effect (a cached result, a dry run, an editor's buffer), and [group coordinators](#gating-and-reviewing-tool-calls); a gate's decision is applied at Start, so read `Decision` | nothing; every gate goes before it |
| `GateSystems::Fallback` | `rig_ecs` | after `After`: the kernel's own answers (an `Interrupted` call's, an ended turn's [`Unsent`](#what-the-user-types) messages) | `.after(GateSystems::After).before` it for a fallback of your own |
| `ApprovalSystems` | `rig_basics::approval` | after `After`: one question for each effect that `NeedsApproval`, holding it, or a refusal with nobody to answer | `.before` it to hold or decide effects first, such as asking about a reply's calls together |
| `CommandSystems` | `rig_harness` | takes over a user's line that starts with `/` and runs it | `.before` it to take a line over first |
| `AskSystems` | `rig_harness` | answers the oldest open question that takes text with the user's next line, after `CommandSystems` | `.after` it to answer questions in Gate, so a line the user typed wins |
| `PrintSystems` | `rig_basics::print` | answers a `--print` run's questions from `--answer`, after `AskSystems` | `.before` it to answer a question first |
| `SubagentSystems::Gate` | `rig_basics::subagents` | refuses a `SpawnSubagent` deeper than the policy allows | `.after` it to decide or hold a request it let through |

```rust,no_run
use rig_harness::prelude::*;
use rig_harness::rig_core::redact;

/// The redaction plugin's stage systems, for others to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RedactSystems;

fn build(app: &mut App) {
    // After attachments and the default prompt, before compaction.
    app.configure_sets(AgentPass, RedactSystems.in_set(PrepareSystems::Edit))
        .add_systems(AgentPass, redact_messages.in_set(RedactSystems));
}

/// Takes API keys out of each message as it comes in, attached files
/// included.
fn redact_messages(mut messages: Query<&mut Incoming, Added<Admitted>>) {
    for mut incoming in &mut messages {
        for part in &mut incoming.parts {
            if let MessagePart::Text(text) | MessagePart::Attachment(Attachment { content: UserContent::Text(message::Text { text, .. }), .. }) = part {
                *text = redact::text(text).into_owned();
            }
        }
    }
}
```

**Saving and writing in `Last`**, on the exit frame too: the kernel queues
what the frame changed for the session log (`rig_ecs::SaveSystems`), then
writes it (`rig_ecs::WriteLogs`). A `Last` system that changes saved state, a
conversation or a [saved component](#saved-state-counting-every-tool-call),
goes `.before(SaveSystems)`; after it, the change is saved a frame late, and
on exit never. A notice sent in `Last`, such as the session log's write
failure in `WriteLogs`, is read the next frame by the systems before its
sender, so one that keeps something on exit goes `.after(WriteLogs)` or asks
`SessionLog::failure()`.

# A tool

Any `rig_core::tool::Tool`. Its description says what it does and when to pick
it, such as "Use it rather than `wc` in `shell`": the model reads it with the
tool, and the system prompt stays the same whatever tools are offered. What
each call touches lets calls on different things run side by side: the tool
says it per call (`Tool::footprint`, as `read` says it reads its path), else
`footprint` for every call, else its annotations (`Reads(Anything)` for one
that only reads, `Writes(Anything)` otherwise). Blocking work goes in
`blocking`; a stop drops the call's future at once, while that work runs on to
its end, unread (a rig-tools `Process` it started dies with its handle).
`args_schema` derives the parameters from the arguments type, whose field doc
comments describe them, and a call whose arguments do not fit is refused with
where and why (rig-core's `tool::validate`); a schema that declares no
`properties` takes any arguments.

A tool registered with `app.add_tool_per_call(tool, options)`, a
`PerCallTool`, is built for each call's `CallPlace` by its `at(place)`: the
app's workspace (an overlay of it for a
[preview](#gating-and-reviewing-tool-calls)), the calling agent's `WorkDir`
else the `ProjectRoot` (`base`, `confine` for a `WorkDir`), and for a tool
registered with `add_live_tool_per_call` the `Live` its output goes to, as
the coding tools are. Each such call gets its `CallPlace` component in
`PerCallSystems::Place` and its `CallHandler` in `PerCallSystems::Build`,
both in `PrepareSystems::Compose`: a system between them changes where the
call works, one after replaces or wraps the handler. The tool's entity has
its `PerCall`. A plugin that builds such a call somewhere else uses
`PerCallTools` (`place(agent, workspace)`, then `build(name, place)`), as
preview builds it on an overlay.

```rust,no_run
use std::{path::PathBuf, sync::Arc};

use rig_harness::prelude::*;
use rig_harness::rig_tools::workspace::{Local, Workspace};
use schemars::JsonSchema;
use serde::Deserialize;

fn build(app: &mut App) {
    let options = ToolOptions {
        footprint: Some(Touches::Reads(Scope::Anything)),
    };
    let here = WordCount { workspace: Arc::new(Local), base: None, confine: false };
    app.add_tool_per_call(here, options);
}

impl PerCallTool for WordCount {
    fn at(&self, at: CallPlace) -> Self {
        WordCount { workspace: at.workspace, base: Some(at.base), confine: at.confine }
    }
}

/// Reads in `workspace`, a relative path from `base`.
#[derive(Clone)]
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

impl Tool for WordCount {
    const NAME: &'static str = "word_count";
    type Args = Args;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Count the words of a file. Use it rather than `wc` in `shell`.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<Args>()
    }

    async fn call(&self, _: &mut ToolContext, args: Args) -> Result<String, ToolExecutionError> {
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
command; despawning the entity removes it. Its annotations are hints: a
call that only reads, and so runs again after a restart, is one that
`options.footprint` declares so.

`rig-tui` runs ordinary presentation systems in `TranscriptRender`. Its
`RenderBatch` holds owned inputs and mutable lines for items needing rendering,
including saved calls and calls with synthetic streaming ids. The
`RenderSystems::Render`, `Decorate`, and `Edit` sets run in that order. Order
competing systems within a phase using your plugin's public sets. These systems
may read agent data but write only presentation data. A crate that also builds
without `rig-tui` puts it behind a default `tui = ["dep:rig-tui"]` feature.

```rust,no_run
use rig_harness::prelude::*;
use rig_tui::ratatui::text::Line;
use rig_tui::{RenderBatch, RenderSystems, TranscriptRender};

fn build(app: &mut App) {
    app.add_systems(TranscriptRender, draw_count.in_set(RenderSystems::Render));
}

fn draw_count(mut batch: ResMut<RenderBatch>) {
    for (input, output) in batch.tools_mut("word_count") {
        let call = input.view();
        let mut lines = vec![call.header("Count", call.argument("path").unwrap_or_default())];
        lines.extend(call.result_lines());
        for child in call.children {
            let indent = "  ".repeat(child.depth + 1);
            lines.push(Line::from(format!("{indent}↳ {} {:?}", child.name, child.state)));
        }
        output.lines = lines;
    }
}
```

The batch's `pane`, `agent`, and `width` describe this layout. A system can
select a pane by its entity or query a marker on it; `pane = None` means an
export. The typed `tools_mut`, `entries_mut`, `contents_mut`, `asides_mut`,
`calls_mut`, and `messages_mut` iterators select inputs without depending on
live effect entities. `requests` exposes every request and output by its typed
`RenderKey`. Unchanged messages stay cached and do not enter the schedule.
A cached item is drawn again only when its inputs change, so after changing
anything else a look reads, or adding or removing a look system at runtime,
increment `RenderRevision.0`; this requests a frame and invalidates cached
presentation. A look that replaces another is ordered after the other's
public set: the shipped looks are in `CodingToolLooks`, `SummaryLooks`,
`SubagentLooks`, `WelcomeLooks` and `BuildFailedLooks`.


# Gating and reviewing tool calls

A tool call is an effect with a `ToolRequest` (the model's `ToolCall`) and the
`CallHandler` that will run it.

- **Prepare** edits them as the call is admitted: the arguments, or the
  handler, wrapped with `ErasedHandler::layered` (to give it a secret the
  model never sees) or replaced in `PrepareSystems::Edit` to run the call
  another way, such as in a sandbox (below).
- **Before every gate** the kernel answers a call that cannot run: a missing
  tool, one `ToolAccess` leaves out, arguments that do not fit.
- **Gate** refuses a call about to run (`Outcome::refused(why)`, which the
  model reads), answers it (a cached result), holds it, or wants the user to
  approve it, `commands.entity(call).queue(NeedsApproval::add(plugin, why))`,
  with `rig_basics::approval::ApprovalPlugin` added. Every plugin's reasons
  make one question in `ApprovalSystems`: `approval::ALLOW` makes the call
  `Approved` (its note kept in the question's record), `DENY` or a text
  alone refuses it with the note, and with nobody to answer it is refused
  at once.
- **Review** edits each result in place before it is committed. What a tool
  published beside its text, such as rig-tools' `FilesRead`, `FilesChanged`
  ([any call](#the-workspace-files-and-child-processes) that may write) or
  `ExitStatus`, is the call's `Published`.

Each time Gate opens, a call gets the `Footprint` of the call as it stands, so
a held call whose `ToolRequest` was edited is gated again ([why a call
waits](#a-held-call-holds-back-the-later-calls-of-its-reply)).

**What a call would change**, before it runs: with
`rig_basics::preview::PreviewPlugin` added, a gate puts `Preview` on it, and
the plugin holds it, runs its tool on an overlay of the workspace that keeps
every write from the disk, and puts `Previewed(Ok(changes))` on it,
`ReadyToStart` again: each `FileChange`, a deletion's too, with its hunks, the
text before and after (`after_text()`), and each side's `FileState`
(`bytes`, `lines`, `hash`). A tool not [registered per call](#a-tool), or
one that starts a process (`shell`), gets `Previewed(Err(..))`.

**A restart's unfinished calls.** A restart restores every unanswered call as
`Restored` work, gated again. One that had started or settled (the kernel
marks such calls in the conversation, `rig.call`) carries `Interrupted`
(`Started`, `Settled(result)`): no gate that holds or asks sees it, a Gate
system may answer or hold it, removing `Interrupted` in Prepare (never in
Gate) runs it again, and once nothing holds it the kernel answers it in
`GateSystems::Fallback` (an error, or its result).

```rust,no_run
use rig_basics::{approval::NeedsApproval, preview::{Preview, Previewed}};
use rig_harness::prelude::*;
use rig_harness::rig_core::{serve::{ErasedHandler, adapters::ToolAdapter}, tool::FileChange};
use rig_harness::rig_ecs::effects::Handler;
use rig_tools::Shell;

/// On an agent whose `shell` calls run sandboxed.
#[derive(Component)]
pub struct Sandboxed;

fn build(app: &mut App) {
    app.add_systems(AgentPass, sandbox.in_set(PrepareSystems::Edit))
        .add_systems(AgentPass, (too_big, commits).in_set(Stage::Gate));
}

/// Wants the user to approve each `git commit`.
fn commits(calls: Query<(Entity, &ToolRequest), (ReadyToStart, ToolCalls)>, mut commands: Commands) {
    for (call, request) in &calls {
        let command = request.call.function.arguments["command"].as_str().unwrap_or_default();
        if command.contains("git commit") {
            commands.entity(call).queue(NeedsApproval::add("commits", format!("It commits: {command}")));
        }
    }
}

/// Refuses a call that would change more than 500 lines, or leave a file
/// over 1 MiB, once it knows.
fn too_big(
    mut calls: Query<(Entity, &Footprint, Option<&Previewed>, &mut Decision), (ReadyToStart, ToolCalls)>,
    mut commands: Commands,
) {
    for (call, footprint, previewed, mut decision) in &mut calls {
        let lines = |changes: &Vec<FileChange>| changes.iter().map(|c| c.added + c.removed).sum::<usize>();
        let huge = |c: &FileChange| c.after.as_ref().is_some_and(|after| after.bytes > 1 << 20);
        if previewed.is_none() && !footprint.0.reads_only() {
            commands.entity(call).insert(Preview);
        } else if let Some(Previewed(Ok(changes))) = previewed
            && (lines(changes) > 500 || changes.iter().any(huge))
        {
            decision.decide(Decider::plugin("too-big"), Outcome::refused("change less at once"));
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

**After every gate.** An answer that stands in for running a call (a
cached result, a dry run) goes in `GateSystems::After` and reads
`Decision`, so it never answers a call a gate refused. So does a group of
calls that runs all or none, such as a patch's edits: the plugin spawns a
child call per member, unheld (no gate sees a call held at spawn), and in
`After`, without `ReadyToStart`, holds the free ones while any member is held
or `Waiting`; then refuses all if one was refused or failed, else releases
them.

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
earlier one changes (`mkdir x` then `ls x`). So a gate that cannot get an
answer refuses at once rather than holding. A plugin that knows better
says what a call touches in Prepare, with `DeclaredFootprint`: with
`(Footprint(Touches::Writes(Scope::Paths(vec![src]))), DeclaredFootprint)` on a
held `cargo fmt`, a read of `docs/` runs at once (a held write still
holds back `Reads(Anything)`, which overlaps every path).

A gate that guards files reads a call's `Footprint`, not its tool's name, so
one rule covers `edit`, `write`, `delete`, `move` and any plugin's tool. Its
paths are absolute and normalized as written (`src/../x` is `x`), not
through links: a strict guard compares `workspace.0.canonical(path)`.

# Asking the user: questions and pickers

A question for the user is an `Ask` entity: a title, a body, the choices
offered and whether a line of text answers it too. It is an effect,
recorded with its answer, and any front answers it by deciding it with an
`Answer { choices, text }` (`Answer::choice(id).outcome()`,
`Answer::text(..)`, a choice with a note `.with_text(..)`), which closes
it in the others; `answer.is(id)` says whether it takes that choice. A
plugin that waits on the answer spawns the `Ask` as a hold on what it asks
about, and asks only when `Answerers::can_answer` for the agent (a front
spawns an `Answerer`). The user's next line answers an open text question.

A question may propose an answer to edit (`Ask::initial`), take several
lines (`multiline`), or wait beside the work (`inline`), which the user's
next line never answers: the terminal view puts the first two in the input
(Enter answers, Esc refuses) and an inline one in a strip above it (Alt+1
to Alt+9 pick, Alt+0 dismisses).

A choice from a list is a `Picker` entity: a title, a body, items that
each may run a command line and show a `detail`, and whether typed text
answers it (`free_text`); a new `Picker` inserted replaces it, and
despawning it closes it. A front answers it, or an open `Ask`, with
`Pick::person(entity, chosen)`, a `Chosen { items, text }`: a picker runs
each chosen item's command, in list order, and gets `Picked` (`None` when
closed), a question the chosen choices with the text as their note.
`Picker::multiple()` (`.checked(true)`) and `Ask::multiple()` (with
`.with(Choice::new(id, label).checked().more(text))`) take several. The
terminal view ticks with Tab, shows a choice's `more` on Right, and once
ticks are in play Enter answers just the ticked; a `ChoiceKeys` component
adds keys of its own (`ChoiceAction::TickAll`, …). A changed `Ask` is a
new one, with `SupersededBy(new)` on the old: the view keeps the user's
ticks.

```rust,no_run
use rig_basics::compaction::{Summarizing, Uncompact};
use rig_harness::prelude::*;

/// Asks the user about each compaction's summary before it replaces
/// anything; `/uncompact` puts back what the last one replaced.
fn build(app: &mut App) {
    app.add_systems(AgentPass, (answered, ask).chain().in_set(ReviewSystems::Act))
        .add_command("uncompact", "Undo the last compaction", |In(args): In<CommandArgs>, mut commands: Commands| {
            commands.trigger(Uncompact { entity: args.agent });
        });
}

/// On a summary call the user was asked about.
#[derive(Component)]
struct Asked;

/// Holds each summary, once read from the reply, with a question.
fn ask(
    calls: Query<(Entity, &Summarizing, &ForAgent), (ReadyToSettle, Without<Asked>)>,
    answerers: Answerers,
    mut commands: Commands,
) {
    for (call, summarizing, &ForAgent(agent)) in &calls {
        let Some(summary) = &summarizing.summary else { continue };
        if answerers.can_answer(Some(agent)) {
            let ask = Ask::new("Use this summary?").body(summary.clone()).free_text();
            let ask = ask.choice("use", "Use it").choice("cancel", "Keep the messages");
            commands.spawn((ask, ForAgent(agent), HoldOf(call)));
            commands.entity(call).insert(Asked);
        }
    }
}

/// `cancel` replaces nothing, a text typed alone is the summary; the
/// question lets go of the call either way.
fn answered(
    asks: Query<(Entity, &Outcome, &HoldOf), (With<Ask>, Added<Outcome>)>,
    mut calls: Query<&mut Summarizing>,
    mut commands: Commands,
) {
    for (ask, outcome, &HoldOf(call)) in &asks {
        let answer = Answer::of(outcome).unwrap_or_default();
        if let Ok(mut summarizing) = calls.get_mut(call) {
            if answer.is("cancel") {
                summarizing.summary = None;
            } else if answer.choices.is_empty() && answer.text.is_some() {
                summarizing.summary = answer.text;
            }
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
the model as it is. A plugin that takes some `/` lines over orders its Gate
system `.before(CommandSystems)`.

```rust,no_run
use rig_harness::prelude::*;

fn build(app: &mut App) {
    app.add_command("remind", "Remind the agent of something after its turn", remind);
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
`Author::user()`, and so is what an agent or a plugin sends: `Deliver` spawns
an `Incoming` effect `ForAgent` the agent. A Prepare system may change it
(`Added<Admitted>`): its `parts`, text and attachments in the order the model
reads them (`text()`, `text_mut()` the first text); `rig_basics::coding_tools`
attaches the `@path` files it names in its public set `AttachSystems` (in
`PrepareSystems::Compose`). A Gate system may take it over by deciding it
(`Outcome::custom(..)`), as the harness runs a `/` line as a command. At Start
the kernel gives the rest to the agent, which starts a turn with it or keeps
it for its running turn as its `mode` says: accepted and not sent yet, it is
in the agent's inbox,
`Query<(&Incoming, &ForAgent), (With<Started>, Without<Outcome>)>`. Its
effect-log record reads with `MessageRecord::read(&record.kind)`.

**The inbox over time.** It is saved (a `rig.inbox` aside per message
until it is answered), and a restart brings it back `Started` and
`Restored`, admitted again, so a Prepare system that adds to messages
skips `Started` ones. A message taken back (decided) leaves it at once.
Each message a turn ended without sending gets `Unsent { ended }`, how the
turn ended (`None` when a restart brings it back to an idle agent): a Gate
system keeps it for the agent's next model call by removing `Unsent`, or
takes it over by deciding it; else `GateSystems::Fallback` hands the
user's back to the views as one `Recalled` (the terminal view's input) and
puts the others in the conversation. A failed turn takes the users'
messages after the last reply that counts out of the conversation, with
the failed replies among them, and keeps an agent's or a plugin's entry
in place; that `Recalled` brings the texts of the local user's own back
first, in order, as the conversation holds them after Prepare: what is
not text, such as an image or a file attached, does not come back, nor
does a titled user's message, such as a chat bridge's or a skill's. A
plugin that sends a task as `Author::user()` gets it back too. Nothing
else keeps the text, so it does not outlive a restart unless sent
again. An exit recalls nothing.

**With the user's next message.** What a plugin wants the agent to read
when the user next writes, such as a background check's failure, it keeps
in its own state and adds to the next `Incoming` from `AuthorKind::User`
in Prepare, a `MessagePart::Attachment(Attachment { label, content })`: no
turn of its own, no hold, and taken back by clearing that state.

A plugin that needs a mark on its own message spawns it,
`(Incoming::new(text, mode), ForAgent(agent), MyMark)`, as `Deliver` does.
Delivered into a turn, the message gets `DeliveredIn(turn)` and the turn's
first model call is recorded under it; it is despawned once it settles, so a
plugin marks the turn then (`On<Add<DeliveredIn>>`) and finds its calls by
`CallOf(turn)`. A user's message may carry a title,
`Author::user().titled(..)` (`conversation.author(at)`). What the user typed
in message `at`, without attachments or what a plugin delivered, is
`typed(&conversation, at)` (`first_typed` for the first).

```rust,no_run
use rig_harness::prelude::*;

fn build(app: &mut App) {
    // Before the harness's systems that decide messages too.
    let gate = keep_to_self.before(CommandSystems).in_set(Stage::Gate);
    app.add_systems(AgentPass, gate)
        .add_command("review", "Asks for a review, on the reviewing model", review)
        .add_observer(mark_review_turn);
}

/// On a message `/review` sent, and on the turn it went into, for a
/// plugin to route its calls.
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

/// A line that starts with `#` is the user's note to self: it never
/// reaches the model.
fn keep_to_self(
    mut typed: Query<(&Incoming, &ForAgent, &mut Decision), (ReadyToStart, IncomingMessages)>,
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
`CompletionRequest` it sends (tools, options), the `connection` it goes to,
else its agent's model (`Models::connect` makes one), and, for a request the
kernel built, its system prompt in parts, `prompt` (the agent's `role`,
then the `PromptSection`s by order), and its messages, `history`, a
view of the agent's conversation whose edits are never saved.

The kernel builds a turn's own call (`TurnCall`), or one from an agent
outside its turns (`SideCall(agent)`), when it is admitted (an
`On<Add<Admitted>>` observer runs before, a Prepare system after). A plugin
changes it in `PrepareSystems::Edit`, editing `history` as a
[conversation](#changing-history) by its `Author` (`history.unsaved()` says
who changed what), so compaction measures what is sent.

Start puts the prompt and `history` into the request's messages and fits it
to the model it goes to: an unset option takes that model's default (its
caching), one it cannot take is left out.

- **Routing** a call to another model is an edit of its `connection` in
  `PrepareSystems::Edit` or before; it then asks for that model's caching.
  The kernel reads `connection` once, after Edit, so a later change is a
  change of the call's `Adapted` instead.
- **Adapting** the model, whichever it is, such as so it reads tools as
  text, is a system in `PrepareSystems::Adapt` (or `AdaptOutermost`, after
  the others) that changes the call's `Adapted(connection)` in place,
  `Query<&mut Adapted, Added<Admitted>>`. After Edit, the kernel puts on
  each model call the connection it goes to, its own or else its agent's
  model at that moment, so a router and an adapter never undo each other,
  and a call keeps that model while it is held or waits. A call admitted
  while it has neither gets none, and Start fails it with "No model is
  connected", even if its agent was connected meanwhile. An adapter wraps
  `connection.handler` with rig-core's `ErasedHandler::layered` or
  `map_stream` (each streamed item rewritten, and recorded so), and says
  what the wrapped model takes on `connection.spec`. A successor is
  routed and adapted anew, so no call is adapted twice.
- **What a call is sent as.** `Adapted::sent` gives a call's request as
  Start will send it to that connection (to measure in
  `PrepareSystems::Fit`), `dropped` the options its model leaves out; its
  system prompt in parts is `ModelRequest::prompt`. Start sends the request
  to the `Adapted` connection and moves it into `SentTo(connection)`; from
  then on a call has its request as sent, and `DroppedOptions`, the options
  left out (by the model's spec or a dry run, rig-core's `Serve::check`).
  A fact a `map_stream` closure inserts in the call's `PublishedContext`
  is its `Published` once answered.
- **Caching.** A provider that caches by marker marks the prefixes a
  request names in `cache_points` (`Tools`, `System`, `Messages(n)`). A
  section that changes every turn breaks every prefix after it, so a
  plugin moves it in `PrepareSystems::Edit` (`prompt.take(tag)`) to the
  last message.
- **Tools.** The kernel offers an agent the tools its `ToolAccess` allows.
  A Prepare edit takes some out (`call.request_mut().tools.retain(..)`, and a
  `prompt.sections` line on why), but a tool left out still runs if
  called: a Gate system refuses a call; `ToolAccess` is the hard limit.
- **Below every adapter,** `models.0.wrap_http(|inner|
  inner.with_middleware(..))` in `build` hands every model call's HTTP
  requests to a rig-core `HttpMiddleware`, such as one that rotates API
  keys: each hook gets the request's `Exchange` (method, URI, and
  extensions holding the call it serves, `rig_core::serve::Serving`), and
  `after_response` sees every reply's status and headers, an error
  reply's (a 429's) too.
- **What the provider said** is rig-core's: a reply's `unfinished_calls()`
  (cut on `Length`), `spec.compat.prefill`, `request.foreign_reasoning`,
  `Native::reasoning_blob`, an `Origin`'s `model`.
- **Reaching a vendor.** `models.0.provide(vendor, provider)` reaches its
  models through the plugin's provider (`ConnectOptions` with a key);
  `models.0.provider(vendor)` hands over the one it replaces.
- **Answering later, from elsewhere,** such as a batch API: the model is
  connected with rig-core's `ErasedHandler::unavailable(descriptor, why)`,
  and a system in `GateSystems::After` holds each call, submits what
  `Adapted::sent` gives, and keeps the batch's id in a saved component
  by turn (a restart runs the call again, `Restored`, to hold again). A
  `WakeAt` polls, and rig-core's `wire::decode_reply` turns the result's
  body into the reply it decides the call with.

A failed call carries the kernel's `RetryVerdict` (`Retry`, `GaveUp`,
`Overflow`, `Final`) in Review: a reviewer edits it in place, takes the
failure over with a successor, or holds the call (`HoldOf`) while it works, as
`rig_basics::compaction` does to summarize first; otherwise the kernel retries
after a `Backoff` or proposes the turn's end. Only a call its own model
answered is judged. A call refused before it was sent fails the turn.

A fixed part of the system prompt is a `PromptSection` entity,
`PromptSection::new(order, tag, text)`, in every agent's prompt, or with
`SectionOf(agent)` in that agent's alone. The provider caches a conversation
from its start, so a section whose text changes costs the whole conversation's
cache once per change: rig's own prompt (rig-basics' `DEFAULT_SYSTEM_PROMPT`)
never changes, and has no date, paths or instruction files in it. Put what
changes in a message instead,
`Deliver::new(agent, text, DeliveryMode::Inform)` with
`.with_author(Author::plugin(name))` (a note without one is the user's, and
plugins that react to the user's messages react to it): it reaches the model
with its next request and stays in the conversation, cached.

```rust,no_run
use std::sync::Arc;

use rig_harness::prelude::*;
use rig_harness::rig_ecs::effects::Handler;

/// The model a turn falls back to when its own gave up.
const FALLBACK: &str = "deepseek/deepseek-flash";

fn build(app: &mut App) {
    let adapt = tools_as_text.in_set(PrepareSystems::Adapt);
    app.add_systems(AgentPass, (adapt, fall_back.in_set(ReviewSystems::Act)));
}

/// Tools as text for a model that calls none, whichever a router picks.
fn tools_as_text(mut calls: Query<&mut Adapted, (Added<Admitted>, With<TurnCall>)>) {
    for mut adapted in &mut calls {
        let Adapted(connection) = &mut *adapted;
        if !connection.spec.tools {
            let mut spec = (*connection.spec).clone();
            spec.tools = true;
            connection.spec = Arc::new(spec);
            // Each streamed item, rewritten: here passed on, where the
            // plugin's parser reads tool calls from the text.
            let pass_on = |_published| |item| vec![item];
            let handler = connection.handler.0.clone().map_stream("text-tools", pass_on);
            connection.handler = Handler(handler);
        }
    }
}

/// A turn's call whose retries gave up is sent again at once, on the
/// fallback model, unless it went there already.
fn fall_back(
    failed: Query<
        (Entity, &RetryVerdict, &ModelRequest, &Effect, &CallOf, &ForAgent),
        (ReadyToSettle, With<TurnCall>, Without<SupersededBy>),
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
        let retry = (TurnCall, CallOf(of.0), ForAgent(agent.0), Effect::with_parent_id(effect.parent));
        commands.spawn((request.clone().on(fallback), retry)).add_related::<SupersededBy>(&[call]);
    }
}
```

# What a model streams

Every model call streams, a turn's and a plugin's own alike. Each event is a
`StreamDelta { effect, agent, event }` message carrying rig-core's
`StreamEvent`: a part's `Start` (a tool call's name), its `Text`,
`Reasoning` or `Arguments` fragments, and its `End` (a finished tool call
with its id); `part` tells parallel tool calls apart. It is written in
Arrive in order, after the call's adapted handler rewrote it, and kept two frames,
so a reader in `PostUpdate` gets a reply's last events even when it settled
that frame. A turn's call also adds its text and reasoning up in its
`Partial` (a successor continuing a reply may start it with the text so
far), and from its first item has a `StreamTiming { first, last, items }`
beside its `Started { at }`. A delta moves no stage: a reader spawning
effects runs `.before(AgentLoop)` or calls `Wake::wake`.

```rust,no_run
use rig_harness::prelude::*;

fn build(app: &mut App) {
    app.add_systems(PostUpdate, print_text).add_observer(time_to_first);
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
    streamed: On<Add<StreamTiming>>,
    calls: Query<(&StreamTiming, &Started, Option<&ForAgent>)>,
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

A turn is an entity `TurnOf` its agent, with a `Turn`: its `index` among the
agent's turns over the session, and `start`, the `MessageId` it begins at
(`conversation.start_of(id)` is its index now, `asked(id)` the entries it
answers, `request(id)` the one that asked). It is spawned before its first
request is built, so an `On<Add<TurnOf>>` observer's edit is in that request;
once ended it loses `TurnOf` and keeps `ForAgent`.

Its end is a proposal. When the model answers without calling tools, or the
turn fails, the kernel proposes a `TurnOutcome` (`Answered(message)`,
`Failed(reason)`) in Commit. A Review system sees it in the next pass
(`Added<TurnOutcome>`) and may:

- edit it in place;
- hold it (`HoldOf(turn)`);
- carry the turn on: a `Queue` or `Steer` message the agent accepts takes an
  `Answered` proposal away, and so does a new
  `(TurnCall, CallOf(turn), ForAgent(agent))` once admitted.

A plugin may insert `TurnOutcome::Failed(reason)` or `Stopped(Some(reason))`
at any time, such as a budget: calls not started never start, the others are
answered as stopped (or with their `IfStopped`). `StopTurn` stops a turn at
once, and the turns of every agent below it in `SpawnedBy`, even when its own
agent is idle: stopping follows ancestry, so work meant to outlive its parent
is spawned without `SpawnedBy`. A call that did not finish reads rig-core's
`tool::feedback::not_finished`: "Not run: …" when it never started, else
"Stopped: …" and whether it only read or may have partly run (an `Independent`
footprint may have). Once nothing holds the outcome and every call settled,
the turn ends, `Settled` until the next frame (an `On<Add<Settled>>` observer
may `Retire` the agent). A turn with no `TurnCall` left stays open until
something inserts a `TurnOutcome`.

**Turn records.** Each turn is saved in its agent's conversation as a
`TurnRecord` aside by rig (`TURN`), and `Turn` is filled from it, a
restarted turn's too: its `index`, `start`, and how it ended,
`TurnEnd::Answered { answer }`, `Failed { why }` or `Stopped { why }`, or
`None` while it runs or once a crash or an exit cut it short;
`conversation.turns()` lists them, for a program without the app too. A stop
mid-reply saves what streamed as an aborted reply (`StopReason::Aborted`) no
model reads again, its request `halted()` `Stopped`, and the kernel's note
after it by `Author::rig()` (`EntryRef::is_stop_marker`); a
plugin-stopped turn's request, and an agent's or a plugin's message a
failed turn left, is halted `Kept`. A front shows a stop from
the halt, never from text.

**Ending a turn with an answer already known**, such as the JSON a
`submit` tool was called with: a Gate system decides the turn's next
model call itself, with a completion of that text. Nothing is sent, the
turn ends `Answered` with it, and `--print` prints it as the answer.

```rust,no_run
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::{CompletionResponse, Usage};
use rig_harness::rig_core::effect;
use rig_harness::rig_core::message::{AssistantContent, Origin, Text};

fn build(app: &mut App) {
    app.add_systems(AgentPass, end_with.in_set(Stage::Gate))
        .add_systems(AgentPass, check_the_tests.in_set(Stage::Review));
}

/// On a turn whose answer is known: the text it ends with.
#[derive(Component)]
pub struct EndWith(pub String);

/// Answers the turn's next model call with the known text.
fn end_with(
    mut calls: Query<(&CallOf, &mut Decision), (ReadyToStart, ModelCalls, With<TurnCall>)>,
    turns: Query<&EndWith>,
) {
    for (&CallOf(turn), mut decision) in &mut calls {
        if let Ok(EndWith(text)) = turns.get(turn) {
            let choice = vec![AssistantContent::Text(Text::new(text.clone()))];
            let origin = Origin::new("plugin", "submit", "submit");
            let reply = CompletionResponse::new(choice, Usage::default(), origin, serde_json::Value::Null);
            decision.decide(Decider::plugin("end-with"), Outcome(Ok(effect::Answered::Completion(reply.into()))));
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
Others can gate it, review its outcome and its `Published` facts (an open
effect inserts its own as it decides: `Published(context.result_context())`
of a rig-core `ToolContext` it `insert_result`s them into) and replay it;
`CallOf(turn)` keeps a turn open until it settles. Work off the main
thread is `Running::answering(pool, wake, future)` on the effect.

**What a running effect would answer if stopped.** Its owner keeps an
`IfStopped(outcome)` on it current (the output so far), which the kernel
answers with in place of "stopped", and on a tool call a
`ToolProgress` line (`12 of 40 files`); a tool that prints as it works
reports through a rig-tools `Live` from `CallOutput::new(&wake)`.

**An owner follows its effect's successor.** Another plugin may take the
effect over, such as a watchdog that sends a stalled call again
(`commands.queue(resend(call))`). A plugin that waits on its effect by a
marker registers `app.add_observer(follow_successor::<Marker>)`, which
moves the marker and the effect's hold to each successor. Make a plugin's
model call an effect (a `ModelRequest`) so the effect log and spending see
it; a call made directly, they never see.

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
    let kind = EffectKind::Custom { kind: "deploy".into(), payload: serde_json::json!({ "target": args.args }) };
    commands.spawn((EffectRequest { kind, handler: None }, ForAgent(args.agent), Deploy::default()));
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

The `rig_basics::subagents` plugin spawns an agent for a `SpawnSubagent`
effect `ForAgent` its parent, made by the model's `task` tool or a plugin.
Prepare systems edit it (model, tools, system prompt), Gate systems refuse or
hold it, and at Start the plugin spawns the agent, with what the request
leaves out from the parent, as `SpawnedAgent(agent)` on it.

- `SpawnSubagent::task(title, brief)` works on a task and reports to its
  parent, as `task` starts it.
- `SpawnSubagent::helper(title)` works for the plugin, which talks to it:
  it has a `Helper` and an `AgentTitle`, reports to no one, and `message`
  cannot reach it; `.brief(text)` sends it a first message, and with
  `.once()` it retires when its first turn ends.

## Extending `task`

A plugin adds an argument to `task` with a property in its `ToolDef`'s
`parameters` and a sentence in its description on when to use it ([editing
another plugin's tool](#turning-off-or-replacing-what-another-plugin-added)):
`task` puts each argument it does not take into `SpawnSubagent.extra`.

```rust,no_run
use rig_harness::prelude::*;
use rig_basics::subagents::{SpawnSubagent, TASK};

#[derive(Default)]
pub struct ReviewersPlugin;

impl Plugin for ReviewersPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(AgentPass, as_reviewer.in_set(Stage::Prepare));
    }

    /// In `finish`, so `task` exists whatever the plugins' order.
    fn finish(&self, app: &mut App) {
        let world = app.world_mut();
        let mut tools = world.query::<&mut ToolDef>();
        for mut def in tools.iter_mut(world) {
            if def.0.name.as_str() != TASK {
                continue;
            }
            let def = std::sync::Arc::make_mut(&mut def.0);
            if let Some(properties) = def.parameters.get_mut("properties")
                && let Some(properties) = properties.as_object_mut()
            {
                let profile = serde_json::json!({ "type": "string", "enum": ["reviewer"] });
                properties.insert("profile".to_owned(), profile);
            }
            def.description.push_str(" Give `task` the `reviewer` profile for a review.");
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
```

# The effect log: reading, redacting, replaying

Every effect that settles is an `EffectRecorded(EffectRecord)` message, in
settle order: its request (`kind`), its answer (`outcome`, a reviewer's as
`reviewed`), who decided it before it ran (`decided_by`), its times, the agent
it worked for (`scope`, an `AgentId`), its `turn` (`Turn.index`), and the
effect it was made under (`parent`). `rig_basics::telemetry` writes them to
the session's effect log in `Last` (`WriteLogs`) and reads them back through
its `EffectLog`: `read()`, or `read_matching(&filter)` by agent, kind name
(`completion`, `tool_call`, `message`, `subagent` or a plugin's) and turn
(rig-cassette's `EffectFilter::default().agent(id).turn(n)`), as written by
the end of the last frame.

**Redacting.** A record holds what its effect really carried (tool output,
file changes with the text before, the request as sent). A plugin changes
records before they are written (`MessageMutator<EffectRecorded>`, below) and
[rewrites](#rewriting-saved-history-a-redaction) those written once the
session stopped. rig-core's `redact` finds keys and tokens by their shapes
(`json`, `text`).

**Replaying.** A replayer answers effects from a recorded log before
they start: a model call by an adapter in
`PrepareSystems::AdaptOutermost`, the others in `GateSystems::FromRecord` (rig-cassette's `RecordedAnswers` pairs agents).

```rust,no_run
use rig_harness::prelude::*;
use rig_harness::rig_core::redact;
use rig_harness::rig_ecs::{StopTurnsOnExit, WriteLogs};

fn build(app: &mut App) {
    app.add_systems(Last, hide_secrets.after(StopTurnsOnExit).before(WriteLogs));
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
never read. `add_project_setting::<T>(section)` also merges the table of
the project's `.rig/settings.toml` over the user's, key by key, at
startup (a cloned repository changes only what a plugin lets it): each
such table is a `ProjectSetting` entity, the project's file is the
`ProjectSettings` resource, and both happen in `PreStartup`'s
`ProjectSettingsSystems::{Read, Apply}`. Here a
[command-line option](#fronts-options-and-commands) overrides the setting.

```rust,no_run
use rig_harness::prelude::*;
use serde::Deserialize;

/// The `[budget]` table: `max_calls = 50`; no limit by default.
#[derive(Resource, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct Budget {
    pub max_calls: Option<usize>,
}

fn build(app: &mut App) {
    app.add_setting::<Budget>("budget")
        .add_flag(Flag::new("budget", "Stop a turn after N tool calls").value("N"));
    let invoked = app.world().get_resource::<Invoked>();
    if let Some(n) = invoked.and_then(|it| it.flag_value("budget")?.parse().ok()) {
        app.insert_resource(Budget { max_calls: Some(n) });
    }
}
```

# Saved state: counting every tool call

An agent component that derives `Reflect` and says
`#[reflect(Component, Saved)]` is kept with the session (a generic type only
once registered): each change is logged, and a restart, `/reload` or
`/resume` inserts the newest value on the restored agent before
`SessionRestored` exists, so an `On<Insert<T>>` observer can re-arm what it
says. A resource that says `#[reflect(Resource, Saved)]` and is inserted in
`build` is kept the same way, and one holding rig-core data, such as a
`MessageId`, is saved through serde. A plugin says what it keeps
(`keeps_saved::<T>()`, `keeps_files("vim")`) as `Keeps` on its entity. A
changed type needs no upgrade step ([saved value](#words)).

```rust,no_run
use std::collections::HashMap;

use rig_harness::prelude::*;

/// How often each agent called each tool, kept across restarts.
#[derive(Component, Reflect, Default)]
#[reflect(Component, Saved)]
struct ToolCounts(HashMap<String, u32>);

fn build(app: &mut App) {
    app.register_required_components::<Agent, ToolCounts>()
        .keeps_saved::<ToolCounts>()
        .add_observer(count);
}

/// Counts each call as the model asks for it, not again on a restart.
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

An agent's `Conversation` is rig-core's, changed by calling its methods. Each
message is an entry with a `MessageId` (`id(i)`, `index_of(id)`) that stays
across edits and restores, the time it was made (`EntryRef::made`) and, when
it is not the user's own or the model's, an `Author` (`Author::plugin(name)`,
`agent(..)`, `rig()`, or the user `titled`). Every change names its author.

| To | Call |
|---|---|
| add a message | `push(Entry::new(message).by(author))` |
| replace a range: a cut (`from..`, nothing), a summary, an edit (`entry(i)` passed back changed) | `splice(range, entries, by)`; one that reaches the end clears the halt |
| put back entries it held (a summary's originals, an undone edit), ids, times and asides kept | `restore(range, entries, by)` |
| make it hold a whole list, such as a transcript edited as text | `set_entries(entries, by)` |
| merge adjacent replies and repair tool-call pairs | `normalize(by)` |
| find a message's copy in another conversation | `key(i)` (a `MessageKey`), `index_of_key` |
| where a span that began at `id` begins now | `start_of(id)` |

**Asides** never reach a model: an `Aside { kind, after, author, value }`
follows the entry `after` and is saved and restored with the
conversation unless `unsaved()` (`add_aside`, `replace_aside`,
`remove_aside`, `asides()`). The harness's `Note` is the kind fronts draw
([the transcript](#the-transcript-notes-looks-and-scrolling)).

**Saving.** A change waits in the conversation's outbox until the kernel saves
it to the agent's log: between the stages of every pass, at the end of the
frame, before a tool that may change something starts, and on a stop. Each one
saved is a `ConversationChanged { agent, change }`, a `ChangeRecord` with the
`Change` (`Push`, `Splice`, `Halt`, `Aside`, `Reset` for a value inserted over
the component), its author (`by`) and time. A splice before the end costs the
provider's prompt cache from there on. `Retire { entity }` retires an agent
for good, with the agents it spawned.

**Compaction** (`rig_basics::compaction`) splices its summary, by
`summary_author()`, in place of what it summarized (`summary_in(&change)`
tells that splice apart). Its policy comes only from `[compaction]` in
`settings.toml` (`CompactionDefault`). Other compaction is a plugin listed in
its place, not an extension of it: rig's
`corpus/ct/src/keep_last_compaction.rs` drops the oldest turns instead, on
public API alone. A request that does not fit takes one recovery path, carried
through retries (`Recovering`): old tool outputs cleared, then compacted, then
the turn fails; a model reply starts it again. A summary that fails or is cut
is never used: old tool outputs are cleared instead, and the call goes if it
then fits (`Recovering::NotSummarized`, as for a cancelled summary or notes
call); an overflow at that step fails the turn. A notes (`Noting`) or summary
call is the agent's request cut where its reading ends, plus one instruction,
so the provider reads it from its cache; a Prepare edit of it costs that from
the edit on. In `ReviewSystems::Act` `Summarizing.summary` and `Noting.notes`
hold the text: a reviewer holds the call, replaces the text or sets `None` to
cancel. `Uncompact { entity }` puts the newest compaction's messages back.

**The session format** is rig-memory's (`rig-session/4`), read and written
without an app: `Session::read(&*store)`/`session.save`, `load`/`save` for one
conversation, and `read_conversation` for one agent's alone. rig's saved
values have stable keys (`SavedAgent::model()`, `title()`,
`session.main_agent()`). An importer opens `home.new_session()` with
`StoreOpener::open`, edits at the source's time
(`conversation.dated(at_ms, |c| ..)`), lists it with
`SessionMeta { title, .. }.write(&*store)` and marks it saved
(`dir.mark_saved()`), so `/resume` lists it.

```rust,no_run
use rig_harness::prelude::*;

/// `/undo`: the user's last message and everything after it go, unless a
/// turn runs.
fn undo(In(args): In<CommandArgs>, mut agents: Query<&mut Conversation, Without<ActiveTurn>>) {
    let Ok(mut conversation) = agents.get_mut(args.agent) else { return };
    let last = conversation.entries().rev().find(|entry| entry.is_users_own());
    if let Some(at) = last.map(|entry| entry.index) {
        conversation.splice(at.., [], Author::plugin("undo"));
    }
}
```

## Rewriting saved history: a redaction

Records are only appended. A session no rig runs is rewritten in place with
`rewrite_stopped(&stores, &paths, &dir, by, &with)`, under its lock: every
saved record keeps its place and stamp, and the store's files, the scratch
files and the name are mapped too. The running session is refused; keys are
already redacted as its logs are written. `with` is a `Rewrite`: `Strings`
maps every string (a redaction, a path remap), `rig_memory::Messages` only
messages, or its own hooks per message, aside, saved value, text and file. A
program without an app calls `rig_memory::rewrite(&store, by, &with)` on a
session not `runs_elsewhere()`.

```rust,no_run
use rig_harness::harness_protocol::SessionId;
use rig_harness::prelude::*;

fn build(app: &mut App) {
    app.add_command("redact", "Take a text out of a stopped session: /redact <id> <text>", redact);
}

/// `/redact <id> <text>`: every saved copy of it in that session becomes `[REDACTED]`.
fn redact(
    In(args): In<CommandArgs>,
    (home, stores, paths): (Res<RigHome>, Res<StoreOpener>, Res<SessionPaths>),
    mut notices: MessageWriter<Notice>,
) {
    let Some((id, secret)) = args.args.split_once(' ') else { return };
    let Ok(id) = id.parse::<SessionId>() else { return };
    let secret = secret.to_owned();
    let with = Strings(move |text: &mut String| {
        let hit = text.contains(&secret);
        if hit { *text = text.replace(&secret, "[REDACTED]") }
        hit
    });
    let dir = home.0.session(&id);
    let done = rewrite_stopped(&stores, &paths, &dir, Author::plugin("redact"), &with);
    let text = done.map_or_else(|why| why, |it| format!("{} logs rewritten", it.logs.len()));
    notices.write(Notice::info(None, text));
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

fn build(app: &mut App) {
    app.add_observer(keep_awake)
        .add_systems(Update, still_working.run_if(on_real_timer(EVERY)));
}

/// A turn keeps the loop awake while it runs (until its entity goes).
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
(rig-tools' `Local`) unless a plugin puts another in (a rig-tools
`Workspace`, such as one over SSH, `workspace.0.host()`), which may be
swapped while the app runs, so read it on each use. `RigHome` is the
`RIG_HOME` layout, `TokioRuntime` rig's tokio runtime; a plugin that
should also run without the harness's plugins inits what it reads.

**The main-thread rule.** A workspace call may wait on the network. Tools
call it inside `blocking`; a system calls it inline only while `host()` is
`None`, else as `Running::spawn` work, holding what waits for it (a turn's
first `TurnCall`, admitted in Prepare), as `read_todo` below does. A file
tool resolves paths with `workspace.place(base, confine, path)`;
`edit::apply(path, text, edits)` is what `edit` makes of a text, and
`rig_tools::split_status` parts a `shell` result from its status line.

**What a call changed.** The file tools publish the files they write as
`FilesChanged`, each change with the `FileState` before and after, its
`hunks` (`unified()`, `changed_lines()`) and, for a text file read whole,
`before_text` (`after_text()` applies the hunks to it). With
`rig_basics::checkpoint::CheckpointPlugin` added, each call that may write
anywhere (`Writes(Anything)`: `shell`, a tool that does not say) is wrapped
in rig-tools' `Checkpointed`, which publishes the difference
between snapshots just before and after it runs; `CallCheckpoint::before()`
on the call and `TurnCheckpoint::first()` on the turn keep the files before,
and `CheckpointAlso` names ignored paths to include. It says what changed
while the call ran, not who changed it. A plugin's own snapshots keep in a
directory of their own under the scratch: two with one `Taking::keep` of one
work tree share a git index and race on its lock.

**Child processes.** `Processes` is how every child process starts, a
rig-tools `Launcher` (a wrapper command, a directory, environment): the
shell tool reads it for each command, so a sandbox plugin that sets it
(`processes.set(..)`) wraps every later process. A plugin builds its own
with `processes.launcher().command(program)`, or `.spawn(program, args,
Spawn { stdin, merge_stderr })`, a `Process`. Every launcher it hands out
drops the session's [secrets](#sessions-their-store-secrets-and-exiting),
which a bare `std::process::Command` passes on; one that runs in the
project goes through the workspace (`workspace.0.spawn(&launcher, ..)`).

**Files and streams.** An entity with a `FileWatch` gets `FileChanges`
messages: first the files under its path when armed (read them then, so
none made in between is missed), then each batch created, changed or
removed (another workspace may look every second), leaving out what the
workspace's listing does (ignored and hidden files, `target`) unless
`.listing(..)` says otherwise: rig-tools' `Listing::Hidden` adds the
hidden files the ignore rules keep, `All` every file, as for
`Workspace::files` and the `list` and `search` tools' `hidden` argument. A
long-lived source, such as a socket, is a `Running::stream(pool, &wake,
items)` on an entity: each item is inserted on it as a `Done<T>`, in
order, waking the loop, and the `Running` goes once the stream ended. A
`Done` on a hold or a plugin's own entity moves no effect, so no stage
after Arrive runs for it: react with `On<Insert<Done<T>>>` (releasing a
hold or deciding there is a move), never a stage system.

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
```

# Sessions: their store, secrets and exiting

**The store.** What a session holds goes through its store, a rig-memory
`Store` (`SessionStore`): each agent's log, and files by name, its images'
blobs among them (under `blobs/`, named by their content). The harness opens
it once every plugin is built and the run's front is chosen, and reads it
before writing: one not readable whole is refused unchanged. A plugin that
keeps sessions elsewhere or sealed puts its own `StoreOpener::new(|dir| ..)`
in while the app is built (imports, deletes and `/resume` open stores through
it); a front whose runs keep their session elsewhere, such as in a chat
thread, says so, `Front::session_store(open)`, for the running session alone.
Reading or writing a store may block the frame: keep it to commands and
startup, as `/resume` does. What is named by path stays plain, under
`Scratch`: cut tool output, pasted images and the text log (`Scratch(None)`
keeps none). A copy lists a store's files with `files()`, and a command that
rewrites a session skips one that `runs_elsewhere()`.

**A plugin's own files** go in `PluginData`, there in every run: rig-memory
`Files` (`write_file`, `read_file`, `files`), names led by the plugin's
(`vim/registers.json`). `session` is in the session's store and removed with
it, `project` this user's for the project (`$RIG_HOME/projects/`), and `user`
every session's (`$RIG_HOME/data/`).

**Secrets.** A secret a plugin needs before the session is restored, such as a
store's passphrase, is asked for in `build` with
`app.add_secret(Secret::new(name, help))` and read from `Secrets`. A launcher
at a terminal asks for each one its environment (`RIG_SECRET_<NAME>`) lacks,
once, and passes it to the agent's process alone, never to a process rig
starts: it is kept from tools and the model's commands, not from plugin code,
which is trusted.

**Exiting.** A plugin that ends the run triggers `Exit(reason)`
(`ExitReason::Quit`, `Reload`, `Switch(Some(id))` or `Switch(None)`,
`Signal`, `Failed(code)`), never `AppExit`; the first reason is the
`ExitReason` resource. Each exit asked for is an app-level effect, an
`ExitRequest`, and the app exits when the first one settles. A plugin
delays it by holding it, with a deadline of its own: nothing cancels an
exit, and a SIGTERM waits as long as the hold. A second exit is a second
effect, which a holder may take as "stop waiting".

- **Start:** `AppStart` says why this app started (`StartReason`): `New`,
  `Resumed`, `Reloaded` after `/reload`, or `Restarted` on the
  previous build after a rollback; a switch is `New` or `Resumed`.
- **Fork an agent:** `ForkAgent { agent, through }` makes a new agent in this
  session, no restart: its log is a copy of `agent`'s up to a turn boundary at
  or before `through` (`None` for the end), so a compaction made before the
  cut is undone in it too; it has every saved component `agent` has now
  (model, reasoning, tools, prompt, working directory), its connection and its
  prompt-cache key (`CacheKey`). `AgentForked { from, fork, through }` names
  it. `rig_basics::sessions`' `/fork`, `/agents` and rewind are built on it.
- **Edit a stopped one:** `edit_stopped(&stores, &paths.0, &dir, |session|
  ..)` under its lock; `PluginData::of(&stores, &dir)` opens its plugin files.
- **Delete:** `DeleteSession(id)`, finished by the next start if a crash cut
  it short (`SessionDeleted`, else `SessionDeleteFailed`).
- **Name:** a plain file in its directory (`SessionDir::name`), written by
  `rig_basics::sessions`' `/name` and `NameSession { session, name }`, looked
  up by `rig --resume <name>`, `/resume <name>` and `Home::find`.

```rust,no_run
use std::io;
use std::sync::Arc;

use rig_harness::harness_protocol::SessionDir;
use rig_harness::prelude::*;
use rig_harness::rig_memory::{DirStore, SessionError};

/// Keeps every session's files on the disk mounted at `/vault`, `rig bot`
/// runs' in `/vault/bot`, and spilled output and pasted images nowhere.
fn build(app: &mut App) {
    app.insert_resource(Scratch(None))
        .insert_resource(StoreOpener::new(|dir: &SessionDir| vault("/vault", dir)));
    // The bot's front sets itself up in `finish` (not shown).
    let bot = Front::new("bot").on_command("bot", "", "Answer in the chat");
    app.add_front(bot.session_store(|dir: &SessionDir| vault("/vault/bot", dir)))
        .add_systems(AgentPass, hold_exit.in_set(Stage::Gate))
        .add_observer(give_up);
}

fn vault(root: &str, dir: &SessionDir) -> Result<Arc<dyn Store>, SessionError> {
    let id = dir.id().ok_or_else(|| io::Error::other("not a session of RIG_HOME"))?;
    Ok(Arc::new(DirStore::open(format!("{root}/{id}"))?))
}

#[derive(Component)]
struct NotesHold;

/// Holds the first exit until the notes are written, at most 10 s.
fn hold_exit(
    exits: Query<Entity, (With<ExitRequest>, ReadyToStart, PluginEffects)>,
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
```

## The native session views

A `SessionPicker` on a `Picker` (`rig_basics::session_picker`, with its
`SessionPickerPlugin` added) is the native view of the saved sessions
(`SessionSource::Saved`, `Sessions(ids)`) or of one agent's entries
(`Entries`): filled in place from tasks as they are read, filtered by
`SessionFilter` (running ones, a project, a text), the highlighted item
previewed (`PickPreview`, which any picker sets as its `Highlighted` moves).
Picking runs its `command` with `{session}`, `{agent}` and `{entry}` filled
in, or, with none, puts `SessionPicked` on the entity before `Picked`. The
items are filled in `SessionPickerSystems`; a system after it may relabel
them, but not add, remove or reorder them, as a pick names the session at
the item's position. `/resume` and `/fork at` are this view; a program
reads the same with `Home::saved_sessions()`, `SessionMeta::read(&store)`
and `read_conversation`.

```rust,no_run
use rig_basics::session_picker::{SessionPicked, SessionPicker, SessionSource};
use rig_harness::prelude::*;

/// `/fork-at`: the entries of the agent's conversation, previewed; the one
/// picked ends a fork of the agent (`rig_basics::sessions`' `fork {entry}` as a
/// `command` does the same).
fn fork_at(In(args): In<CommandArgs>, mut commands: Commands) {
    let source = SessionSource::Entries { session: None, agent: None };
    let picker = Picker::new("Fork through which message?").for_agent(args.agent);
    let agent = args.agent;
    let fork = move |picked: On<Picked>, chosen: Query<&SessionPicked>, mut commands: Commands| {
        if let Ok(SessionPicked { entry: Some(through), .. }) = chosen.get(picked.entity) {
            commands.trigger(ForkAgent { agent, through: Some(*through) });
        }
    };
    commands.spawn((picker, SessionPicker { source, ..Default::default() })).observe(fork);
}
```

## Data on a session

A plugin keeps any serde value with the session under a key of its own:
`app.saved_as::<T>("goals.goal")` logs a saved resource or agent component
under that key, not its type's path. A program without Bevy reads it typed,
`Session::value::<T>(key)` (an agent's, `SavedAgent::value`), and writes a
stopped session's through `edit_stopped` (which keeps its listing).
`.listed()` also copies it into the session's listing (`SessionMeta.listed`;
keep it small), so `/resume`, `SessionMeta::read` and a `SessionPicker`'s
filter show and find it without reading a log.

```rust,no_run
use rig_harness::harness_protocol::SessionDir;
use rig_harness::prelude::*;
use rig_harness::rig_memory::{DirStore, Session};

/// What the session is for, shown in `/resume`.
#[derive(Resource, Reflect, Clone, Default, serde::Serialize, serde::Deserialize)]
#[reflect(opaque, Resource, Saved, Default, Clone, Serialize, Deserialize)]
struct Goal(String);

fn build(app: &mut App) {
    app.init_resource::<Goal>().saved_as::<Goal>("goals.goal").listed();
}

/// A program's read of a stopped session's goal, without Bevy.
fn goal_of(dir: &SessionDir) -> Option<String> {
    let store = DirStore::open(dir.path()).ok()?;
    let session = Session::read(&store).ok()?;
    session.value::<String>("goals.goal")?.ok()
}
```

# Adding to another plugin's lists

A plugin that owns a list builds it from entities other plugins spawn, as
the system prompt is built from `PromptSection`s: spawn one with your text
and keep it current; an empty text or `Disabled` leaves it out.

| List | Spawn | Crate |
|---|---|---|
| `/model` and its picker: a choice that is no catalog model, run by name | `ModelItem::new(name, label, command)`; an agent's `ModelLabel` says what its model is in the status line | `rig_basics::models` |
| `/resume`: a part of this session's line, kept with it | `ResumeLabel::new(tag, text)` | `rig_basics::sessions` |
| `/agents`: a column of one agent's line | `(AgentColumn::new(tag, text), ColumnOf(agent))`, despawned with the agent | `rig_basics::sessions::agents` |
| `/usage`: a line after the usage | `UsageRow::new(tag, text)` | `rig_basics::telemetry::usage` |
| `--print`: text printed after the answer | `PrintSection::new(tag, text)` | `rig_basics::print` |

A provider plugin's models are catalog models: it declares its vendor and
adds them to the catalog in `Models` (rig-core's `ProviderId::declare`,
`Connector::provide`, `catalog_mut`) at any time.

Spending is data too. `rig_basics::telemetry` adds each settled model call's
usage to its agent's `AgentUsage`, the session's `SessionUsage` (both saved)
and its turn's `TurnUsage`, per model and per purpose (`turn`, `compaction`, a
plugin's `Effect::with_purpose`), and shows the session's in its `/resume`
line (the `spent` label); a plugin that pays for work that is no model request
`record`s it, and one whose model call settles without a reply (superseded,
cancelled) puts its usage on it, `CallUsage`.

```rust,no_run
use rig_harness::prelude::*;
use rig_basics::print::PrintSection;
use rig_basics::telemetry::usage::{SessionUsage, UsageRow};

/// What the session may spend, in USD.
const CAP: f64 = 5.0;

/// Refuses the turns' model calls once the session spent `CAP`, and says
/// what is left in `/usage` and after a `--print` answer.
fn build(app: &mut App) {
    app.world_mut().spawn((UsageRow::new("budget", ""), PrintSection::new("budget", "")));
    let show = show_budget.run_if(resource_exists_and_changed::<SessionUsage>);
    app.add_systems(AgentPass, refuse_over_budget.in_set(Stage::Gate)).add_systems(Update, show);
}

fn refuse_over_budget(
    mut calls: Query<&mut Decision, (ReadyToStart, ModelCalls, With<TurnCall>)>,
    spent: Option<Res<SessionUsage>>,
) {
    if spent.is_none_or(|spent| spent.0.total.cost < CAP) {
        return;
    }
    for mut decision in &mut calls {
        let why = format!("The session spent its ${CAP:.2} budget.");
        decision.decide(Decider::plugin("budget"), Outcome::refused(why));
    }
}

fn show_budget(spent: Res<SessionUsage>, mut shown: Query<(&mut UsageRow, &mut PrintSection)>) {
    let left = (CAP - spent.0.total.cost).max(0.0);
    let text = format!("Budget: ${left:.2} of ${CAP:.2} left.");
    for (mut row, mut section) in &mut shown {
        row.text.clone_from(&text);
        section.text.clone_from(&text);
    }
}
```

# Turning off or replacing what another plugin added

A whole plugin is replaced in `plugins.toml`: disable its entry and list
yours, as for [compaction](#changing-history). Tools, slash commands,
key bindings, options, fronts and prompt sections are entities.
`Disabled` on one turns it off and frees its name for a replacement (a second
one of a taken name is refused), in `Plugin::finish`, which runs once every
`build` did. What a plugin spawns in `build`, `finish` and `cleanup` is marked
`ProvidedBy` it; what it spawns later it marks itself
(`Res<PluginEntity<MyPlugin>>`). Changing how another plugin's tool runs is
rarely a replacement: a Prepare or Gate system on its calls does it, a Review
system changes its results, and a default tool may have a switch, such as
`rig_basics::coding_tools::ShellOutput`. A plugin's stage systems in a public
set are turned off with a run condition
(`configure_sets(AgentPass, AttachSystems.run_if(|| false))` in `finish`).
A shipped transcript look is replaced by a render system ordered after its
public set (`CodingToolLooks`, `SummaryLooks`, `SubagentLooks`,
`WelcomeLooks`, `BuildFailedLooks`) that writes the same output.

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
        // `Name` ("/help").
    }
}
```

What a tool tells the model is data on its entity too, and another plugin may
change it, in `finish` or at any time: its `ToolDef` (the rig-core
`ToolDefinition`: name, description, the JSON schema of its `parameters`; edit
only what you add). A property added to another tool's `parameters` ([as for
`task`](#extending-task)) is one the model may send; its plugin takes it out
of the call's `ToolRequest` in Prepare if the tool would refuse it.

# Testing a plugin

A plugin's test builds a small app: the kernel (`AgentPlugin`,
`SessionLogPlugin` on a rig-memory `MemoryStore`), the plugin, and the
`rig_basics` services it needs (`ApprovalPlugin`, `PreviewPlugin`,
`CheckpointPlugin`), run frame by frame with `app.update()`. An effect spawned
for an agent stands in for what its model would ask, as below. A whole turn
runs on rig-core's scripted `MockCompletionModel` (its `test-utils` feature),
put in `Models` as a provider plugin's models are, and Bevy's
`TaskPoolPlugin`, as model calls run off the main thread.

A plugin with stage systems also checks they are ordered:
`rig_harness::stage_conflicts(app.world_mut())` lists each pair of systems of
one stage that write the same data [unordered](#ordering-against-other-plugins),
naming their plugins, and is empty for an app with the plugin and those it is
used with. `rig plugin check --build` runs the same check on the user's own
`plugins.toml`, the default plugins included. How long each system takes is
`PluginTimings` (`totals()`, `last_frame()`), filled once a plugin crate
depends on rig-harness with `features = ["trace"]`.

```rust
use rig_harness::prelude::*;
use rig_harness::rig_core::message::{ToolFunction, ToolName};
use rig_harness::rig_ecs::session_log::SessionStore;
use rig_harness::rig_memory::MemoryStore;

/// The plugin under test: refuses a `write` outside `src/`.
struct SrcOnlyPlugin;

impl Plugin for SrcOnlyPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(AgentPass, src_only.in_set(Stage::Gate));
    }
}

fn src_only(mut calls: Query<(&ToolRequest, &mut Decision), (ReadyToStart, ToolCalls)>) {
    for (request, mut decision) in &mut calls {
        let path = request.call.function.arguments["path"].as_str().unwrap_or_default();
        if request.call.function.name.as_str() == "write" && !path.starts_with("src/") {
            decision.decide(Decider::plugin("src-only"), Outcome::refused("write only under src/"));
        }
    }
}

/// A test: `#[test]` in the plugin's `tests.rs`.
fn a_write_outside_src_is_refused() {
    let mut app = App::new();
    app.insert_resource(SessionStore::new(MemoryStore::default()))
        .add_plugins((AgentPlugin, SessionLogPlugin, SrcOnlyPlugin))
        .add_open_tool("write", "Write a file.", ToolOptions::default(), |_: On<ToolCalled<serde_json::Value>>| {});
    app.finish();
    let conflicts = rig_harness::stage_conflicts(app.world_mut());
    assert!(conflicts.is_empty(), "{}", conflicts.join("\n"));
    // The first frame restores the session and spawns the first agent.
    app.update();
    let agent = app.world_mut().query_filtered::<Entity, With<Agent>>().single(app.world());
    let (Ok(agent), Ok(name)) = (agent, ToolName::new("write")) else {
        panic!("the kernel spawns an agent");
    };
    // The call its model would ask for.
    let args = serde_json::json!({ "path": "README.md" });
    let call = ToolCall::from_wire("c1", ToolFunction::new(name, args));
    let call = app.world_mut().spawn((ToolRequest { call }, ForAgent(agent))).id();
    app.update();
    let outcome = app.world().get::<Outcome>(call);
    let refused = outcome.and_then(|outcome| outcome.0.as_ref().err());
    assert!(refused.is_some_and(|why| why.to_string().contains("only under src/")), "{outcome:?}");
}

a_write_outside_src_is_refused();
```

# Fronts, options and commands

A front (the terminal view, `--print`, a window, a chat bot) offers itself
with `app.add_front(..)`; one takes each run, alone owns stdin and stdout, and
sets itself up in `finish` only if `app.runs_front(name)`. It sends what the
user types as `Deliver` from `Author::user()`, shows `Notice`s (an `Error`
fails a `--print` run or refuses a command), each agent's `Conversation` (or
`ConversationChanged` changes) and
[notes](#the-transcript-notes-looks-and-scrolling), its reply as it streams
([`StreamDelta`](#what-a-model-streams)), sets `ShownAgent` to the agent the
user looks at, and may spawn an `Answerer` and answer questions with
[`Pick`](#asking-the-user-questions-and-pickers). A front nobody sits at
drives its run with `rig_basics::print`'s `app.run_headless(Some(prompt))`,
the loop of the `-p` and `--json` fronts too, which ends once no turn runs
and nothing is held, unless an entity has `KeepRunning` (a file watcher's,
say; the run may then start with `None`). `HeadlessTurnEnded` carries each
answer, and `HeadlessEnded` comes last.

A plugin's own options are declared with `app.add_flag(Flag::new(name,
help))`, `.value("N")` for one that takes a value (`rig --name VALUE` or
`--name=VALUE`), and read from `Invoked` (`flag(name)`, `flag_value(name)`)
from `build` on, as the [settings](#settings) example does.

## Commands of your own

A command is a front of its own: `Front::new("stats").on_command("stats",
"[--all]", "Count the saved sessions")` takes every run of `rig stats …`,
and the words after the name are `Invoked::command()`. A command whose
front is not `on_terminal()` runs headless: the launcher passes stdin,
stdout, stderr and the exit code through (`ExitReason::Failed(n)` exits
with `n`) and never restarts it; `headless()` is only for runs that name
no command or option. `rig -r <id or name> stats` runs it in that session;
`sessionless()` leaves no session behind and refuses a resumed one.

The launcher learns a build's commands, options and secrets from
`rig_harness::run(app)`, which the generated `main.rs` ends with: it
builds the app, writes what its plugins declared and runs no frame, so
`build` only declares, and work goes in `finish` or systems (`rig help`).

```rust,no_run
use rig_harness::prelude::*;
use rig_harness::rig_memory::SessionMeta;

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

/// Prints the count and their messages, from each one's listing, then exits.
fn stats(home: Res<RigHome>, stores: Res<StoreOpener>, mut commands: Commands) {
    let sessions = home.0.saved_sessions();
    let listings = sessions.iter().filter_map(|(_, dir)| SessionMeta::read(&*stores.open(dir).ok()?));
    println!("{} saved sessions, {} messages", sessions.len(), listings.map(|it| it.messages).sum::<usize>());
    commands.trigger(Exit(ExitReason::Quit));
}
```

# The transcript: notes, looks and scrolling

A plugin puts a note in an agent's transcript as an
[aside](#changing-history), `Note::new(title, text).aside(author, after)`,
after a message (`after`, a `MessageId`, or the summary that replaced it),
with its title and its text. Every front draws it, it is kept as any aside
is, and `Note::of(aside)` reads one.

The terminal view draws notes, and what the conversation holds from a plugin
(such as `rig_basics::compaction`'s summary), with systems selecting
`RenderBatch::entries_mut(plugin)`; other asides use `asides_mut(kind)`. The transcript puts each item on the band of
what it is, from the `Theme` resource (mixed from the terminal's background):
a user message on `user`, a tool call and what a plugin added on `tool`, a
reply on none; a look's line with a background of its own keeps it, as a
diff's `added` and `removed` lines do
(`rig_tui::diff::unified_lines(text, &theme)`, also `ToolCallView::theme`). A
look draws code in the terminal's palette with
`rig_tui::highlight::code_rows(code, Language::File(path))` (or `highlight`
for the spans of each line; `Language::Named("rust")` for a language by name).
The transcript is a `TranscriptPane`
([another](#what-agents-do-and-say-in-a-terminal-panel)), the default screen's
marked `MainPane`. Plugins read and set a pane's `TranscriptView` component
(`MainTranscript` for the main one): the item `selected` (Alt+Up and Alt+Down
select), the `highlights` lit, the `marks` (a `TextMark` label inside a
reply's text or a streaming call's) and where it is scrolled (`top()`, placed
in `TuiSystems::Layout`; a panel fitted to it changes in `TuiSystems::Fit`,
drawn the same frame). `ScrollTo { pane, item, row }` scrolls to a
`TranscriptItem` (a `Message`, a tool `Call`, an `Aside`, a `Notice`), at a
row of it or its `n`th lit match; `item: None` follows the newest output. A
`DrawnTranscript` added to a pane holds the rows it drew by item;
`TranscriptLayout::lay_out(world, view, agent, width)` lays any agent's out
so, undrawn. It takes `&mut World` and runs the same render schedule, so call it
from an exclusive system or an exclusive command.

A message's images, audio, videos and documents, the user's or the
model's, are drawn by systems selecting `contents_mut(ContentKind::Image)`.
The input's `view()` gives a `ContentView` (`default_lines()` is `[image]`),
each block an item `TranscriptItem::Content(id, index)` with its rows.
`RenderOutput::decorate(name, lines)` appends named lines after a call, an item
`Decoration(call, name)`. For messages, use `messages_mut()` and the same
method to produce `MessageDecoration(message, name)`. `TuiFrame`, written in `TuiSystems::Render`,
says where the frame drew each panel (`area`) and row (`screen_row`) and
what an overlay `covered`, for a plugin that writes to the terminal
itself, such as an image over rows its render system left blank, and
`RedrawArea(rect)` writes cells it drew over again.

```rust,no_run
use rig_harness::prelude::*;
use rig_tui::ratatui::layout::Rect;
use rig_tui::ratatui::style::{Style, Stylize};
use rig_tui::ratatui::text::Line;
use rig_tui::{
    RenderBatch, RenderSystems, TranscriptRender, ContentKind, DrawnTranscript, Highlight, ItemRow, MainPane, MainTranscript,
    ScrollTo, TerminalWrite, TranscriptItem, TuiFrame, TuiSystems,
};

fn build(app: &mut App) {
    app.add_command("mark", "Mark the newest message: /mark why", mark)
        .add_command("find", "Light a text and show the last message with it", find)
        .add_systems(TranscriptRender, draw_marks.in_set(RenderSystems::Render))
        .add_systems(TranscriptRender, show_duration.in_set(RenderSystems::Decorate))
        .add_systems(PostUpdate, place_images.after(TuiSystems::Render))
        .register_required_components::<MainPane, DrawnTranscript>();
}

fn draw_marks(mut batch: ResMut<RenderBatch>) {
    for (entry, output) in batch.entries_mut("mark") {
        let mut lines = vec![Line::from(format!("★ {}", entry.title)).yellow()];
        lines.extend(entry.text.lines().map(|line| Line::from(line.to_owned())));
        output.lines = lines;
    }
    for (image, output) in batch.contents_mut(ContentKind::Image) {
        output.lines = [image.view().default_lines(), vec![Line::default(); 3]].concat();
    }
}

fn show_duration(mut batch: ResMut<RenderBatch>) {
    for (call, output) in batch.calls_mut() {
        output.decorate("took", call.took.map(|it| Line::from(format!("  {it:?}"))).into_iter().collect());
    }
}

/// Puts the cursor on each image's first row, unless an overlay covers its
/// rows, where a real plugin writes the picture's escape codes.
fn place_images(frame: Res<TuiFrame>, drawn: Single<&DrawnTranscript, With<MainPane>>, mut out: MessageWriter<TerminalWrite>) {
    for (item, rows) in drawn.items() {
        let Some(top) = frame.screen_row(&drawn, rows.start) else { continue };
        let area = Rect { y: top, height: 4, ..drawn.area() };
        if matches!(item, TranscriptItem::Content(..)) && !frame.covered(area) {
            out.write(TerminalWrite(format!("\x1b[{};{}H", top + 1, area.x + 3).into_bytes()));
        }
    }
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
    (mut view, pane): (MainTranscript, Single<Entity, With<MainPane>>),
    mut commands: Commands,
) {
    let Ok(conversation) = agents.get(args.agent) else { return };
    let typed_it = |at: &usize| typed(conversation, *at).is_some_and(|text| text.contains(&args.args));
    let hit = (0..conversation.messages().len()).rev().find(typed_it);
    if let Some(id) = hit.and_then(|at| conversation.id(at)) {
        let item = Some(TranscriptItem::Message(id));
        commands.trigger(ScrollTo { pane: *pane, item, row: ItemRow::Lit(0) });
    }
    let style = Style::new().on_yellow();
    view.highlights = vec![Highlight { text: args.args, style, match_case: true, ..Default::default() }];
}
```

# The status line

The terminal view's row under the transcript shows the `StatusItems` of the
shown agent and of its running turn, which every agent and turn has, and the
app's `AppStatus`. A plugin shows an item in a system in `StatusSystems::Show`
that runs when what it shows changed: `items.show(item)` puts it at its place
(side and order) in place of the item of its name, and an empty item hides it.
Another plugin replaces or hides an item by name (`items.hide(name)`) in
`StatusSystems::Edit`. A plugin that writes `AppStatus` inits it. The default
plugins' items, and how long each stays when the line is too narrow (`keep`:
lowest first, `u8::MAX` never):

| side | order | name | item | keep | plugin |
|---|---|---|---|---|---|
| left | 10 | `session` | the session's name (`AppStatus`) | 2 | `rig_basics::sessions` |
| left | 20 | `title` | `⤷` a spawned agent's name | 14 | `rig_basics::subagents` |
| left | 30 | `model` | the model | always | `rig_basics::models` |
| left | 40 | `reasoning` | the reasoning setting | 12 | `rig_basics::models` |
| left | 50 | `status` | idle, thinking, … | always | `rig_basics::activity` |
| left | 60 | `subagents` | the agent's subagents at work | 15 | `rig_basics::subagents` |
| left | 70 | `others` | the other agents at work | 13 | `rig_basics::subagents` |
| left | 80 | `turn` | what the turn spent (on the turn) | 11 | `rig_basics::telemetry` |
| left | 90 | `reload` | the rebuild of `/reload` (`AppStatus`) | 16 | `rig_basics::reload` |
| right | 10 | `tokens` | tokens in and out | 4 | `rig_basics::telemetry` |
| right | 20 | `cache` | cached input | 1 | `rig_basics::telemetry` |
| right | 30 | `cost` | cost | 3 | `rig_basics::telemetry` |
| right | 40 | `context` | the context in use; shown again when it changes | 5 | `rig_basics::telemetry` |

```rust,no_run
use rig_harness::prelude::*;
use rig_tui::{Side, StatusItem, StatusItems, StatusSystems, Tone};

/// After the status; gone before the reasoning setting.
const MESSAGES: StatusItem = StatusItem::new("messages", Side::Left, 55, 10);

fn build(app: &mut App) {
    app.add_systems(PostUpdate, show.in_set(StatusSystems::Show));
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
`Conversation`; a subagent is `SpawnedBy` its parent (`Spawned`).
`rig_basics::activity::Activity` says what an agent does now, and
`MessageReader<ConversationChanged>` sees every change of every conversation.

A `TuiPanel` entity takes a side of the transcript (`Top`, `Bottom`,
`Left`, `Right`, with a ratatui `Constraint`), rows under the input
(`BelowInput`), or a box over the screen (`Over`). The plugin draws it
into its `PanelCanvas` in `TuiSystems::Draw`, run only for a frame that is
drawn, and writes a `RequestRedraw` when its own state changed;
`TerminalSize` is the terminal's size, `ShownAgent` names the shown agent;
`Within { panel, part, of }` is a column of another panel.

```rust,no_run
use rig_basics::activity::Activity;
use rig_harness::prelude::*;
use rig_tui::ratatui::layout::Constraint;
use rig_tui::ratatui::widgets::Paragraph;
use rig_tui::{PanelCanvas, Placement, Theme, TuiPanel, TuiSystems};

#[derive(Component)]
struct AgentsPanel;

fn build(app: &mut App) {
    app.add_systems(Startup, spawn).add_systems(PostUpdate, draw.in_set(TuiSystems::Draw));
}

fn spawn(mut commands: Commands) {
    commands.spawn((AgentsPanel, TuiPanel::new(Placement::Right(Constraint::Length(32)))));
}

/// Each agent, a subagent under its parent, with what it does and how
/// long its conversation is.
fn draw(
    mut panels: Query<&mut PanelCanvas, With<AgentsPanel>>,
    agents: Query<(&Name, &Activity, &Conversation, Has<SpawnedBy>)>,
    theme: Res<Theme>,
) {
    let line = |(name, activity, conversation, sub): (&Name, &Activity, &Conversation, bool)| {
        let indent = if sub { "  " } else { "" };
        format!("{indent}{name} {}, {} messages", activity.status, conversation.messages().len())
    };
    let lines: Vec<_> = agents.iter().map(line).collect();
    for mut canvas in &mut panels {
        canvas.render(Paragraph::new(lines.join("\n")).block(theme.panel("agents")));
    }
}
```

The default screen (`DefaultScreenPlugin` in `TuiPlugin`) is made of
widgets a plugin places too, each in a `TuiPanel`: a `TranscriptPane` of
the shown agent (`TranscriptSource::Shown`), one live (`Agent(entity)`),
or a conversation read from a saved session (`Value`), which
scrolls and selects with `PaneKeys` while it has the `KeyFocus`; an
`Editor` (the main one `MainEditor`); a `StatusLine { agent }`, in a
panel of its own or, as the default screen has it, on the editor, which
draws it in its bottom row, with the keys in its top row, both while
nothing is typed.

```rust,no_run
use rig_harness::prelude::*;
use rig_tui::ratatui::layout::Constraint;
use rig_tui::{PaneKeys, Placement, TranscriptPane, TranscriptSource, TuiPanel};

/// `/watch <name>`: that agent's conversation, live, beside the main one.
fn watch(In(args): In<CommandArgs>, agents: Query<(Entity, &Name), With<Agent>>, mut commands: Commands) {
    let Some((agent, name)) = agents.iter().find(|(_, name)| name.as_str() == args.args) else { return };
    commands.spawn((
        TranscriptPane::new(TranscriptSource::Agent(agent)).titled(name.as_str()),
        TuiPanel::new(Placement::Right(Constraint::Percentage(40))),
        PaneKeys,
    ));
}
```

# Keys and the input in the terminal view

The terminal view (`rig_tui`) takes part in the input as data, there in
any run and inert without the view:

- **The frame's keys.** `PendingInput` is what the terminal reported
  (`KeyInput::Event`) and the actions plugins put in (`KeyInput::Action`),
  in order, done one at a time in the `TuiKeys` schedule: a system in its
  `InputSystems::Filter` reads, changes, drops or replaces the item before
  any binding or the input sees it, with the main `Editor` and
  `InputContext` (the open completion, the question or picker shown, its
  `selection()`) as the keys before it left them; `Dispatch` does the
  rest. A modal input is one filter, and `/` starts a slash command unless
  a filter or binding takes it.
- **Keys.** Every key the view acts on is a `KeyBinding` entity, its own
  editing keys included. `app.add_key_binding(binding, observer)` binds a
  key or a sequence (`ctrl+x ctrl+e`), heard as `KeyPressed`
  (`run_command("compact")` runs a slash command). The first binding of a
  key stays, a global one shadows the others, and `Disabled` frees its
  keys. While a transcript item is selected (Alt+Up), bindings
  `in_scope(KeyScope::Transcript)` shadow the input's. A binding with an
  `InputAction` (`Up`, `Send`, …) does that built-in action.
  `KeyFocus(Some(panel))` gives a `TuiPanel` the keys until Esc.
- **Completion.** A `CompletionSource` entity offers candidates
  (`.candidate(text, detail)`) for a word starting with a marker anywhere
  in the input, `CompletionSource::marker('#')` (`/` and `@` are the
  view's), or for a command's argument, `CompletionSource::arg("model")`
  (`.at(n)`). One that looks its candidates up reads what is typed,
  `InputContext::typed()`, and is inserted again: an open list follows.
- **The input.** An `Editor` holds a draft (`text()`, `set`, `insert`,
  `take`, …), the main one's read as `MainInput`; the keys type into a
  panel's own while it has the `KeyFocus`. `PromptHistory` is the prompts
  Up and Down browse, kept in `PluginData.user`: the view appends each one
  sent (`Sent`, which a plugin writes too) in `HistorySystems::Record` and
  saves in `HistorySystems::Save`; a system between the two filters what
  is kept.
- **The terminal.** `TerminalFocus` says whether it has the user's focus,
  a `TerminalWrite(bytes)` message goes to the terminal between frames,
  `CopyToClipboard(text)` copies (OSC 52 and this machine's clipboard),
  and a `RunInTerminal(command)` entity runs a program with the terminal,
  without the session's secrets, and gets its `TerminalExit`.

This plugin writes the draft in `$EDITOR` on Ctrl+E, makes Ctrl+K browse
back as Up does, and opens its search box on `/` when the input is empty:

```rust,no_run
use std::path::PathBuf;
use std::process::Command;

use rig_harness::prelude::*;
use rig_tui::crossterm::event::{Event, KeyCode};
use rig_tui::{AppKeyBindingsExt, Editor, InputAction, InputContext, InputSystems, KeyBinding, KeyFocus};
use rig_tui::{KeyInput, KeyPress, KeyPressed, MainEditor, MainInput, PendingInput, RunInTerminal, TerminalExit, TuiKeys};

/// A program writing the draft in this file.
#[derive(Component)]
struct Draft(PathBuf);

/// The plugin's search box, a `TuiPanel` it draws (not shown).
#[derive(Component)]
struct SearchBox;

fn build(app: &mut App) {
    app.add_systems(Update, take_draft)
        .add_systems(TuiKeys, slash_searches.in_set(InputSystems::Filter))
        .add_key_binding(KeyBinding::new(KeyPress::ctrl('e'), "Write the draft in $EDITOR"), edit);
    app.world_mut().spawn((KeyBinding::new(KeyPress::ctrl('k'), "Up"), InputAction::Up));
}

/// `/` on an empty input gives the search box the keys; elsewhere it is typed.
fn slash_searches(
    mut events: ResMut<PendingInput>,
    (input, state): (Single<&Editor, With<MainEditor>>, Res<InputContext>),
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

fn edit(_: On<KeyPressed>, input: Single<&Editor, With<MainEditor>>, mut commands: Commands) {
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
    mut input: MainInput,
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
what the agent already has; its winit plugin then runs the app's loop. The
crate depends on `bevy` at exactly the agent's version (`"=0.20.0"`,
without default features, with `ui`). A window beside the terminal view
(`Invoked::interactive`) sets `ExitCondition::DontExit`, so closing it
leaves the agent running, and `WinitSettings` with
`UpdateMode::reactive_low_power(Duration::from_secs(1))`; in `finish` it
points the agent's `Wake` at the `EventLoopProxyWrapper`, sending
`WinitUserEvent::WakeUp`. A window that is the front offers itself
`on_flag`.
