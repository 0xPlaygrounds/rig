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
- [Fronts, options and commands](#fronts-options-and-commands), with [commands of your own](#commands-of-your-own); [the transcript](#the-transcript-entries-looks-and-scrolling), [the status line](#the-status-line), [a terminal panel](#what-agents-do-and-say-in-a-terminal-panel), [keys and the input](#keys-and-the-input-in-the-terminal-view), [a window](#a-window)

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

Each plugin crate is its own workspace in `RIG_HOME/plugins/<name>`. It
depends on `rig-harness` and the rig plugin crates it uses (`rig-tui` for
a terminal panel) by version, and on `bevy_ecs` and `bevy_reflect` (or
`bevy`) at the agent's exact version. The agent builds from a rig checkout
(`RIG_SOURCE`), which a `[patch.crates-io]` table points the rig crates at.

`/reload` in the agent, or its `reload` tool (both from `rig-reload`),
rebuilds and restarts in the same session once no turn runs. A failed
build leaves the running one; one that crashes at startup is rolled back.

# Finding names

The prelude holds Bevy's app, ECS, reflection and time preludes, the agent
runtime's components, stages and registries (`rig_harness::rig_ecs`), and
the harness's services, which rig-harness's crate docs list. rig-core is
`rig_harness::rig_core` and rig-tools `rig_harness::rig_tools`; each
default plugin's types are in its crate, such as the terminal view's in
`rig_tui` (with `rig_tui::ratatui` and `rig_tui::crossterm`). What every
plugin shares lives in rig-harness, never in another plugin. What hosts
without Bevy need too is in the libraries: rig-core's `redact` and schema
check, rig-cassette's session format (`journal`), rig-tools' workspace and
process runner.

Every one of these types is reflected, so the running agent lists them
with the `inspect` tool of the optional `rig-inspect` plugin: ask it
before reading rig's source. `registry.schema` gives a type's fields
(`type_limit: {with: ["Saved"]}`: what is saved with the session),
`world.query` with `ProvidedBy` what each plugin added, and the
`Diagnostics` resource the warnings logged.

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
before it starts, wherever it was spawned, so every Prepare system sees
it. Prepare sees an effect once; Gate each pass it is ready to start, so
again after each hold is released; Review an outcome each pass until it
settles.

**An observer is for reactions that need no order**: counting, a status
item, a span, the plugin's own entities (`On<Add<Started>>`,
`On<Add<Settled>>`, `On<Add<ToolRequest>>`). Observers of one event run in
no fixed order, so an observer never edits stage data another plugin
edits; it may decide an effect, since the first decision wins.

**The conflict rule.** A hold is any entity `HoldOf(effect)`, saying why
with its `Name` (`waiting for lease on src/a.rs`), which the fronts show.
The first decision wins: a gate decides through the effect's `Decision`,
naming who decided (`decision.decide(Decider::plugin(NAME),
Outcome::refused(..))`, or `Decider::Person` for what the user chose
through a front), so a later gate can never overturn a refusal; the
kernel applies it as the `Outcome` at Start, and the record keeps
`decided_by`. Allowing is doing nothing. A reviewer edits the `Outcome`
in place (`Query<&mut Outcome, Unreviewed>`), once (a marker), since
Review sees it each pass; the record keeps both outcomes. Plugins never
insert `Outcome` themselves, so Bevy sees every decision and edit.

**Deciding once.** A gate that only reads (a path rule) decides afresh
each time it sees a call. One whose decision asks someone or has side
effects marks the call it decided on (a `SparseSet` marker such as
`Approved`) and skips marked and held calls. A decision is applied at
Start, so until then the call is still `Undecided` for later gates: a gate
that holds or asks checks `decision.is_decided()` first.

**Failing closed.** A stage system that fails or panics closes its stage
for that pass, whatever the app's error handler: a failed Prepare system
answers what that pass admitted with an error, a failed Gate system starts
nothing, a failed Review system settles nothing. A notice names the
system. So a broken guard never lets a call through.

**Taking over and trying again.** A reviewer that takes a failure over
spawns a successor that `Supersedes(failed)`, with the request it needs
and the same `CallOf`, or sends the same request again with
`commands.queue(resend(failed))`: the failed effect is recorded as
superseded and never reaches the conversation. Work that takes time holds
the outcome first, then supersedes it.

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
            .add_systems(AgentPass, (redact_messages, redact_prompts).in_set(RedactSystems));
    }
}

/// Takes API keys out of each message as it comes in, attached files
/// included.
fn redact_messages(mut messages: Query<&mut Incoming, Added<Admitted>>) {
    for mut message in &mut messages {
        message.text = redact::text(&message.text).into_owned();
        for attachment in &mut message.attachments {
            if let UserContent::Text(text) = &mut attachment.content {
                text.text = redact::text(&text.text).into_owned();
            }
        }
    }
}

/// Takes API keys out of each request's system prompt.
fn redact_prompts(mut calls: Query<&mut ModelRequest, Added<Admitted>>) {
    for mut call in &mut calls {
        let Some(prompt) = &mut call.prompt else { continue };
        prompt.role = redact::text(&prompt.role).into_owned();
        for section in &mut prompt.sections {
            section.text = redact::text(&section.text).into_owned();
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
whose field doc comments describe them. Before a call runs, its arguments
are checked against them (rig-core's `tool::validate`): a call that does
not fit is refused with where and why, which the model reads.

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
`app.add_open_tool(name, description, options, observer)` triggers
`ToolCalled<Args>` with the call's entity (`call`), the calling `agent`
and the parsed `args`; the call ends when the plugin decides its
`Decision`, such as with `ToolResult::success("Done.".into())`. A tool
found while the agent runs, such as one an MCP server lists, is
`world.spawn_tool(definition, handler, options)` from a queued command,
with a rig-core `ErasedHandler`; despawning the entity removes it. How
`rig-tui` draws a tool's calls is a renderer the tool's plugin adds; the
first one of a tool stays. It gets the call, its result once there is one,
where the call stands (`state`: `Streaming` while the model still writes
the arguments, parsed so far; `Waiting`; `Held` with the holds' names;
`Running` with its `ToolProgress` line; `Done`), whether it is `expanded`,
a running call's `output` so far, and, while this run knows them, what it
published (`facts`) and how long it ran (`took`); `header`, `result_lines`
and `default_lines` draw them. A crate that should also build without
`rig-tui` puts it behind a default cargo feature, as `rig-coding-tools`
does with `tui = ["dep:rig-tui"]`.

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

A tool call is an effect with a `ToolRequest` (the model's `ToolCall`) and
the `CallHandler` that will run it, on the call's entity. A Prepare system
edits them as the call is admitted: the arguments, or the handler, wrapped
with `ErasedHandler::layered` (to give it a secret the model never sees,
say) or replaced in `PrepareSystems::Edit` to run that one call another
way, such as in a sandbox (below). Each time Gate opens, the kernel gives
each call the `Footprint` of the call as it stands, so a held call whose
`ToolRequest` was edited is gated again (Prepare does not run again for
it), and a call that conflicts with an earlier one of its reply is
`Waiting` until that one settles. Before every gate, the kernel answers a
call that cannot run: of a missing tool, of one the agent's `ToolAccess`
leaves out, or with arguments that do not fit; no gate sees it, and
`After` reads its `Outcome`. A Gate system decides a call about to run
with `Outcome::refused(why)`, which the model reads as the tool's refusal,
or any `Outcome` (a cached result), or holds it; a Review system edits
each result in place before it is committed. What a tool published beside
its text, such as rig-tools' `FilesRead` (each file with the hash of the
bytes read and, for `read`, the `LinesShown`), `FilesChanged` or
`ExitStatus`, is the call's `Published`.

```rust,no_run
use rig_harness::prelude::*;
use rig_harness::rig_core::serve::ErasedHandler;
use rig_harness::rig_core::serve::adapters::ToolAdapter;
use rig_harness::rig_ecs::effects::Handler;
use rig_tools::Shell;

/// On an agent whose `shell` calls run sandboxed.
#[derive(Component)]
pub struct Sandboxed;

fn build(app: &mut App) {
    app.add_systems(AgentPass, sandbox.in_set(PrepareSystems::Edit));
}

/// A sandboxed agent's `shell` calls get a handler of their own, which
/// starts commands in the agent's directory.
fn sandbox(
    mut calls: Query<(&ToolRequest, &ForAgent, &mut CallHandler), (Added<Admitted>, ToolCalls)>,
    agents: Query<Option<&WorkDir>, With<Sandboxed>>,
    (processes, root): (Res<Processes>, Res<ProjectRoot>),
) {
    for (request, &ForAgent(agent), mut handler) in &mut calls {
        let Ok(work_dir) = agents.get(agent) else {
            continue;
        };
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
call's `Footprint` (the paths it reads and writes, rig-core's `Touches`),
not its tool's name, so one rule covers `edit`, `write`, rig-coding-tools'
`delete` and `move` (off by default) and any plugin's tool.

**After every gate.** `GateSystems::After` runs once every gate had its
say in the pass. An answer that stands in for running a call (a cached
result, a dry run) goes there and reads `Decision`, so it never answers a
call a gate refused (one the kernel refused already has its `Outcome`). So
does a group of calls that runs all or none, such as a patch's edits and
deletes. The plugin spawns a child call per member, unheld: no gate sees a
call held from its spawn. In `After` it queries them without `Undecided`:
while any member is held by another or `Waiting`, it holds the free ones
with holds of its own; once none is, it refuses them all if one was
refused or answered with an error, else releases its holds.

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

`shell` may change anything (`Writes(Anything)`), so every call after a
held `shell` call waits, reads too: a later call may depend on what an
earlier one changes (`mkdir x` then `ls x`). So a gate that cannot get an
answer refuses at once rather than holding: a refused call settles, and
the next ones run.

# Asking the user: questions and pickers

A question for the user is an `Ask` entity: a title, a body, the choices
offered and whether a line of text answers it too. It is an effect,
recorded with its answer, and any front answers it by deciding it
(`Answer::Choice(id).outcome()`, `Answer::Text(..)`); the first decision
closes it in every other front. A plugin that waits on the answer spawns
the `Ask` as a hold on what it asks about, so an interrupt takes the
question away everywhere. A front that shows questions spawns an
`Answerer`; a plugin asks only when `Answerers::can_answer` for the agent,
and applies its own policy otherwise. While a question that takes text is
open, the user's next line to its agent answers it.

A question may propose an answer to edit (`Ask::initial`), take several
lines (`multiline`), or wait beside the work (`inline`), which the user's
next line never answers. The terminal view puts the first two in the input
(Enter answers, Esc refuses, the draft comes back) and an inline one in a
strip above it while the user keeps typing: Alt+1 to Alt+9 pick, Alt+0
dismisses, unless a plugin's global binding has those keys.

A choice from a list is a `Picker` entity: a title, a body, items that
each may run a command line and show a `detail`, and whether typed text
answers it (`free_text`). Inserting a new `Picker` replaces what it shows,
and despawning it closes it. A front answers it, or an open `Ask`, by
triggering `Pick::person(entity, chosen)`; a picker then runs the chosen
item's command and gets `Picked` (`Chosen::Item`, `Chosen::Text`, or
`None` when closed).

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
its name, and the `mode` its line was sent with. `RunCommand::new(agent,
line)` runs one. An error notice about the agent while the command runs
refuses it: the line goes back in the input with the error, and Enter
then sends it to the model as it is. A `Gate` system in the public set
`CommandSystems` takes over a user's line that starts with `/` and runs
it; a plugin that takes some `/` lines over itself orders its `Gate`
system `.before(CommandSystems)`.

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
    commands.trigger(reminder.with_origin(Origin::plugin("remind")));
}
```

# What the user types

What the user types, in any front, is a message to its agent from
`Origin::user()`, and so is what an agent or a plugin sends. `Deliver`
spawns it as an effect, an `Incoming` that is `ForAgent` the agent, and it
moves through the stages like a tool call. A `Prepare` system may change
it before it is sent (`Added<Admitted>`): rewrite its `text`, add
`attachments`. rig-coding-tools attaches the `@path` files it names this
way, from the agent's `WorkDir` else the `ProjectRoot`, warning of a
missing one, in its public set `AttachSystems` (in
`PrepareSystems::Compose`): a plugin that reads or changes them orders
`.after(AttachSystems)`. A `Gate` system may take it over by deciding it,
such as with `Outcome::custom(..)`, and then it never reaches the agent, as
the harness does to run a user's `/` line as a command. At `Start` the
kernel accepts what is left: it goes to the agent, which starts a turn
with it or keeps it for its running turn as its `mode` says. The messages
an agent accepted and has not sent yet are its inbox,
`Query<(&Incoming, &ForAgent), (With<Started>, Without<Outcome>)>`. A
message's record in the effect log is the custom kind `message`, and
`MessageRecord::read(&record.kind)` reads it back.

**With the user's next message.** What a plugin wants the agent to read
when the user next writes, such as a background check's failure, is a
Prepare edit of that message: the plugin keeps it in its own state and
adds it to the next `Incoming` from `OriginKind::User` as an `Attachment {
label, content }`. It starts no turn of its own, needs no hold, and is
taken back by clearing that state; `Deliver` it to have the agent act at
once. A message accepted but not given yet, such as a `Note` to a busy
agent, is taken back by deciding it (`Outcome::custom("withdrawn")`). A
busy agent is given its notes when its turn ends.

A plugin that needs a mark on its own message spawns it:
`(Incoming::new(text, mode), ForAgent(agent), MyMark)` is all `Deliver`
does. Once the kernel delivers a message into a turn, the turn it started
or the running turn it steered or carried on, the message gets
`DeliveredIn(turn)`, and a turn's first model call is recorded under the
message that started it. The message is despawned once it settles, so a
plugin that tags its message marks the turn then (`On<Add<DeliveredIn>>`)
and finds the turn's calls from it (`CallOf(turn)`). A user's message may
carry a title, `Origin::user().titled(..)`: it still goes to the model as
the user's words, and the conversation keeps the origin
(`Conversation::origin`). What the user typed in message `at`, without
the files attached or what a plugin delivered, is
`typed(&conversation, at)` (`first_typed` for the first).

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
                keep_asides.in_set(Stage::Gate).before(CommandSystems),
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
    let origin = Origin::user().titled("review");
    let message = Incoming::new(text, args.mode).with_origin(origin);
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

/// A line that starts with `#` is the user's own aside: it never reaches
/// the model.
fn keep_asides(
    mut typed: Query<(&Incoming, &ForAgent, &mut Decision), (Undecided, IncomingMessages)>,
    mut notices: MessageWriter<Notice>,
) {
    for (typed, &ForAgent(agent), mut decision) in &mut typed {
        if typed.origin.kind == OriginKind::User && typed.text.starts_with('#') {
            decision.decide(Decider::plugin("asides"), Outcome::custom("aside"));
            notices.write(Notice::info(agent, "Noted; not sent."));
        }
    }
}
```

# Model requests

A model call is an effect entity with a `ModelRequest`: the rig-core
`CompletionRequest` it sends (the messages, the tools offered, the
options), the `connection` it goes to, else its agent's model
(`Models::connect` makes one), and, for a request the kernel built, its
system prompt in parts, `prompt`: rig-core's `SystemPrompt`, the agent's
`role` and its tagged `sections` (the offered tools' rules, then the
`PromptSection`s by order). A turn's own call is a `TurnCall`, whose
request the kernel builds from its agent when it is admitted (an
`On<Add<Admitted>>` observer runs before that, a Prepare system after); a
call built from an agent outside its turns, such as a preview, is
`BuildFrom(agent)`. A plugin changes a request in `Prepare`, which sees
each call once (`Added<Admitted>`), in `PrepareSystems::Edit`, so
compaction measures what is sent. Start then sets the tool rules to the
tools the request carries, renders the prompt as its first message
(`SystemPrompt::text`), and fits the request to the model it goes to (an
option that model cannot take is left out, plugin edits stay).

- **Routing** a call to another model is a Prepare edit of its
  `connection`.
- **Adapting** the model, whichever it is, such as so it reads tools as
  text, is an `Adapter` pushed on the call's `Adapters`. Start applies them
  to the connection the call finally goes to, in the order pushed, those
  placed `outermost()` last, so a router and an adapter never undo each
  other. An adapter wraps `connection.handler` with rig-core's
  `ErasedHandler::layered` or `map_stream` (rewrite each streamed item,
  recorded as rewritten), and says what the wrapped model takes on
  `connection.spec`.
- **What a call is sent as.** `ModelStarts::sent` gives a call's
  connection and request exactly as Start will send them, such as to
  measure it in `PrepareSystems::Fit`, and `prompt` its system prompt in
  parts. From `Start` on, a model call has `SentTo(connection)`, every
  adapter applied, and its `ModelRequest.request` is the request as sent.
  A fact a `map_stream` closure inserts in the call's `PublishedContext`
  is on the call as `Published` once it is answered, and recorded with it.
- **Caching.** A provider that caches prompts by marker, such as
  Anthropic, marks the prefixes a request names in `cache_points`
  (`CachePoint::Tools`, `System`, `Messages(n)`); others ignore them. A
  section that changes every turn breaks every prefix after it, so a
  plugin moves it out in `PrepareSystems::Edit`: it takes it
  (`prompt.take(tag)`) and adds its text to the request's last message.
- **Tools.** The kernel offers an agent the tools its `ToolAccess` allows
  and leaves taking them away to the model the call goes to. A tool left
  out of a request still runs if called: a `Gate` system refuses a call,
  and `ToolAccess` on the agent is the hard limit.
- **Below every adapter: the HTTP client.** `models.0.wrap_http(|inner|
  ..)`, in `build`, wraps the HTTP client of every model call: a rig-core
  `HttpMiddleware` (`inner.with_middleware(..)`) sees each request's
  headers and body, the place to rotate API keys. A provider plugin's own
  client is wrapped with the `Transport` it is handed. Every request
  carries the call it serves, `rig_core::serve::Serving`; a wrapper runs
  on the IO task, and tells the app through a channel read as a
  [`Running::stream`](#the-workspace-files-and-child-processes).
- **Reaching a vendor.** `models.0.provide(vendor, provider)` reaches
  every model of a vendor through a provider of the plugin's own, such as
  `ConnectOptions` with a key; `models.0.provider(vendor)` hands over the
  one it replaces. A model it has no key for falls back to the sign-ins.
- **Answering later, from elsewhere,** such as from a provider's batch
  API: the model is connected with a handler that serves nothing,
  rig-core's `ErasedHandler::unavailable(descriptor, why)`. A system in
  `GateSystems::After` holds each of its calls, submits the request
  `ModelStarts::sent` gives, and keeps the batch's id in a saved component
  on the agent, by turn: a restart runs the call again as `Restored`
  work, with a new effect id, which the plugin holds again. A `WakeAt`
  polls, and rig-core's `wire::decode_reply` turns the result's body into
  the reply it decides the call with.

A failed call carries the kernel's `RetryVerdict` (`Retry`, `GaveUp`,
`Overflow`, `Final`) in `Review`, which a reviewer may edit in place. A
plugin that takes the failure over spawns a new call that `Supersedes` the
failed one (the old one is recorded as superseded and never reaches the
conversation), or holds it (`HoldOf`) while it works, as `rig-compaction`
does to summarize first; otherwise the kernel retries after a `Backoff` or
proposes the turn's end ([How a turn ends](#how-a-turn-ends)). Only a call
its own model answered is judged: one that anyone else answered while it
ran, such as a watchdog, gets no `RetryVerdict`. A call refused before it
was sent fails the turn and keeps the user's message.

A fixed part of the system prompt is a `PromptSection` entity,
`PromptSection::new(order, tag, text)`, in every agent's prompt, or with a
`SectionOf(agent)` in that agent's alone, despawned with it. rig-basics
keeps the project's instruction files and each agent's environment in
sections, re-read as each turn starts; a plugin that changes an
instruction file triggers `rig_basics::RefreshProjectContext`.

```rust,no_run
use std::sync::Arc;

use rig_harness::prelude::*;
use rig_harness::rig_core::completion::Section;
use rig_harness::rig_core::http_client::{self, HeaderMap, HeaderValue, HttpMiddleware, Method, Uri};
use rig_harness::rig_core::wasm_compat::WasmBoxedFuture;
use rig_harness::rig_ecs::effects::Handler;

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
                (plan_only, tools_as_text).in_set(PrepareSystems::Edit),
                fall_back.in_set(ReviewSystems::Act),
            ),
        );
        // Every model request, whoever sends it, carries the team's header.
        let mut models = app.world_mut().get_resource_or_init::<Models>();
        models.0.wrap_http(|inner| inner.with_middleware(TeamHeader));
    }
}

/// Sends the team's id with every model request.
struct TeamHeader;

impl HttpMiddleware for TeamHeader {
    fn before_request_headers<'a>(
        &'a self,
        _: &'a Method,
        _: &'a Uri,
        headers: &'a mut HeaderMap,
    ) -> WasmBoxedFuture<'a, http_client::Result<()>> {
        headers.insert("x-team", HeaderValue::from_static("rig"));
        Box::pin(async { Ok(()) })
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

/// An agent in plan mode is offered only the tools that read, and told so.
fn plan_only(
    mut calls: Query<(&mut ModelRequest, &ForAgent), (Added<Admitted>, With<TurnCall>)>,
    planning: Query<(), With<PlanMode>>,
) {
    for (mut call, &ForAgent(agent)) in &mut calls {
        if planning.contains(agent) {
            let reads = |name: &str| matches!(name, "read" | "search");
            call.request.tools.retain(|tool| reads(tool.name.as_str()));
            if let Some(prompt) = &mut call.prompt {
                prompt.sections.push(Section::new("plan_mode", "Plan only: change nothing yet."));
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

Every model call streams, a turn's and a plugin's own alike. Each event is
a `StreamDelta { effect, agent, event }` message carrying rig-core's
`StreamEvent`: a part's `Start` (a tool call's name), its `Text`,
`Reasoning` or `Arguments` fragments, and its `End` (a finished tool call
with its id); the event's `part` tells parallel tool calls apart. It is
written in `Arrive` in the order it came, after the call's adapters
rewrote it; Bevy keeps a message two frames, so a reader in `PostUpdate`
gets the last events of a reply that settled in the same frame. A turn's
call also adds its text and reasoning up in its `Partial`, gone once the
reply is in the conversation. From its stream's first item, the call has a
`Streamed { first, last, items }` beside its `Started { at }`. A delta
moves no stage, so a reader that spawns effects from it, such as a tool
call started before the reply ends, runs in `Update` `.before(AgentLoop)`
or calls `Wake::wake` once it spawned, else they wait for another wake.

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
the agent's turns over the whole session and its `first_message`, the
`MessageId` it begins at (`conversation.start_of(turn.first_message)` is
its index now). It is spawned before its first request is built, so an
`On<Add<TurnOf>>` observer's edit, such as a prompt section's new text,
is in that request; once the turn ends it loses `TurnOf` and keeps
`ForAgent`. A turn's end is a proposal: when the model answers without
calling tools, or the turn fails, the kernel proposes a `TurnOutcome`
(`Answered(message)`, `Failed(reason)`) in `Commit`, and a `Review`
system sees it in the next pass (`Added<TurnOutcome>`). It may edit it in
place, hold it (an entity `HoldOf(turn)`), or carry the turn on: a
`Queue` or `Steer` message the agent accepts takes an `Answered` proposal
away, and so does a new `(TurnCall, CallOf(turn), ForAgent(agent))` once
it is admitted; a proposal taken away takes its holds with it. A plugin
may insert `TurnOutcome::Failed(reason)` or `Stopped(Some(reason))` at any
time, such as a budget: the turn's calls that have not started never
start, the others are answered as stopped (or with their `IfStopped`).
Once nothing holds the outcome and every call of the turn settled, the
turn ends, `Settled` until the next frame; an `On<Add<Settled>>` observer
sees how, and may retire the agent (`Commit::retire`). `Interrupt` stops
a turn at once; it cannot be held.

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
            commands.trigger(ask.with_origin(Origin::plugin("check-the-tests")));
        }
    }
}
```

# Effects of a plugin's own

A plugin's own work gets the same stages, recording and visibility as the
kernel's when it is an effect: an `EffectRequest` with its rig-core
`EffectKind` (`EffectKind::Custom` for a kind of its own) and a `handler`
the kernel runs at Start, or none, an open effect the plugin decides
whenever it is done. Other plugins can gate it, review its outcome and
replay it; `CallOf(turn)` keeps a turn open until it settles. Work off the
main thread is `Running::answering(pool, wake, future)` on the effect,
whose output is its outcome.

**What a running effect would answer if stopped.** Its owner keeps an
`IfStopped(outcome)` on it current, such as the output so far, which the
kernel answers with in place of "interrupted" when the user interrupts,
the turn ends or the app exits. The owner of a running tool call also
keeps a `ToolProgress` line on it, such as `12 of 40 files`, drawn under
the call. A tool that prints as it works reports through a rig-tools
`Live`, which `CallOutput::new(&wake)` gives with the component to insert
on the call: the harness then keeps the call's `ToolProgress` (the last
line) and `IfStopped` (the output so far), as for `shell`.

**An owner follows its effect's successor.** Another plugin may take the
effect over, such as a watchdog that sends a stalled call again with
`commands.queue(resend(call))`: the effect settles as superseded, and its
`SupersededBy` names the successor. A plugin that waits for its own
effect's outcome by a marker registers
`app.add_observer(follow_successor::<Marker>)`, which moves the marker,
and the hold the effect keeps, to each successor. A model call a plugin
makes is such an effect too: a rig-core model it calls directly is one the
kernel, the effect log and the spending never see.

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
(`rig_subagents`) `ForAgent` its parent: the model's `task` tool makes one,
and a plugin spawns one itself for agents of its own. Prepare systems edit
it (its model, tools, system prompt) and Gate systems refuse or hold it.
When it starts, the plugin spawns the agent with every setting the request
leaves out taken from the parent (its `WorkDir` too), puts it on the
request as `SpawnedAgent(agent)`, and answers with the new agent's id.

- `SpawnSubagent::task(title, brief)` works on a task and reports to its
  parent, as `task` starts it.
- `SpawnSubagent::helper(title)` works for the plugin, which talks to it:
  it has a `Helper(title)`, reports to no one, and `message` cannot reach
  it; `.brief(text)` sends it a first message. With `.once()` it retires
  when its first turn ends, whose answer an `On<Add<Settled>>` observer
  reads.

## Extending `task`

A plugin adds an argument to `task` by adding a property to its
`ToolDef`'s `parameters`, and a line to its `ToolRules` on when to use it
([editing another plugin's
tool](#turning-off-or-replacing-what-another-plugin-added)). `task` puts
each argument it does not take into the request's `SpawnSubagent.extra` by
name, recorded with the rest, for the plugin to read in Prepare.

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
replacement as `reviewed`), who decided that answer before it ran
(`decided_by`), its times, the agent it worked for (`scope`, the agent's
`AgentId`), the agent's `turn` it was made in (`Turn.index`), and the
effect it was made under (`parent`). The `rig-telemetry` plugin writes
them to the session's effect log, a file of the session's store, in
`Last`, in `WriteJournal`, and reads them back through its `EffectLog`
resource: `read()` gives every record, and `read_matching(&filter)` those
an `EffectFilter` keeps, by agent, kind name (`completion`, `tool_call`,
`message`, `subagent` or a plugin's own) and turn, as
`io::Result<Vec<rig_core::effect::EffectRecord>>` (a record's `kind`:
`EffectKind::ToolCall { name, args }`, `Completion { request, .. }`, …).
It holds what was written by the end of the last frame: to count as a run
goes, keep [saved state](#saved-state-counting-every-tool-call).

**Redacting.** A record holds what its effect really carried: tool output
and the diffs of changed files, the request as the model got it. A plugin
that keeps a secret out of the log changes the records in place with
Bevy's `MessageMutator<Recorded>`, in `Last`
`.after(StopTurns).before(WriteJournal)` (both in `rig_harness::rig_ecs`),
so it sees the records a stop on exit makes too, and the log gets only
what it left. rig-core's `redact` finds API keys and tokens by their
shapes: `redact::json` hides every one in a record, and `redact::text` in
a line.

**Replaying.** A replayer answers effects from a recorded session's log
before they start, a model call with an adapter placed outermost (so the
recording stands for the whole adapted model, and no `map_stream`
rewrite runs twice) and every other effect in `GateSystems::FromRecord`.
rig-cassette's `RecordedAnswers` takes each agent's recorded answers by
kind, in order; its `Divergence` carries the record it expected. Agent ids
are new in each session, so a replayer pairs the first agent with the
first recorded one, and the n-th spawn of a paired parent with that
parent's n-th recorded `subagent` effect, whose answer names the recorded
agent. It runs `subagent` effects again, since only running one spawns
the agent, and asks for a recorded exit that ended the session, a quit,
again where it was, so what held it replays too.

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
fn hide_secrets(mut records: MessageMutator<Recorded>) {
    for Recorded(record) in records.read() {
        let Ok(mut json) = serde_json::to_value(&*record) else {
            continue;
        };
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
keeps the default and is shown at startup, naming the file. So that it
never reads the user's file, a test inserts `Settings::from_toml(..)`
before adding the plugin (the test support's apps insert
`Settings::default()`). Here a [command-line option](#fronts-options-and-commands)
overrides the setting.

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

This is how to count what agents do, such as their tool calls; the
[effect log](#the-effect-log-reading-redacting-replaying) looks back at a
run. An agent component that derives `Reflect` and says
`#[reflect(Component, Saved)]` is kept with the session (a generic type
once registered, `app.register_type::<T>()`): each change is logged, and a
restart, `/reload` or `/resume` brings the newest value back, inserted on
the restored agent before `SessionRestored` exists, so an `On<Insert<T>>`
observer can re-arm what it says. A resource that says
`#[reflect(Resource, Saved)]` is kept the same way; the plugin inserts it
in `build`, and the saved value replaces it. Work a restart carries on is
`Restored`, and gated again (holds and approvals are not saved).

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

# Changing history

A conversation changes only through `Commit`, a system parameter, so the
session log records every change and a restore makes it again.
`commit.message(agent, &mut conversation, message)` adds a message, and
`commit.splice(agent, &mut conversation, range, entries)` replaces
messages: a cut is `from..` with nothing, a summary is the range it stands
for with one message, and an edit in place passes `conversation.entry(i)`
back changed. Every message has a `MessageId` (`conversation.id(i)`) that
stays while messages around it are added, cut or replaced, and across
restores: remember messages by id, found again with `index_of(id)`;
`start_of(id)` is where a span that began at `id` begins now. A splice
that reaches the end clears the halt; one before it keeps it.

Each splice is announced as `Committed::Splice { agent, removed, changed }`,
the ids of the messages gone and of those it made or changed; who spliced
is read from the changed messages' item origins. A splice before the end
costs the provider's prompt cache from there on: that is the splicing
plugin's choice. `commit.retire(agent)` despawns an agent for good, so a
restart brings back neither it nor the agents it spawned.

**Compaction** splices its summary, one user item of origin
`rig_compaction::summary_origin()`, in place of the messages it
summarized. Where it cuts is the policy's in rig-compaction's `Compaction`
resource, rig-memory's `CompactionPolicy`: `cut` is `Cut::Budget` (keep
the newest messages the budget allows) or a plugin's rule, `Cut::With`,
and `per_turn` caps the summaries a turn asks for.
`rig_compaction::unsummarized(&conversation)` is what the next summary
covers, `plan(..)` plans one summary call under a policy of the plugin's,
and `On<Insert<Summarized>>` sees the conversation with a new one.

**The session format** is rig-cassette's `journal`, which reads and writes
sessions without an app. `read_session(store)` reads one: each agent's
conversation and saved values, and each message's `Stamp`, its record
number and when it was logged. `write_session(store, &session)` writes one
from data, appending only what differs, each new message at its stamp's
time. An importer takes a session with `home.new_session()`, opens its
store with `SessionStores::open`, writes it, and lists it in `/resume`
with rig-sessions' `write_meta(store, ..)` (`title_of` makes a title as
`/resume` does). `fork_session(from, to, before)` copies a session as it
was when a message was made, `before` being its agent's id and the
message's id.

```rust,no_run
use rig_harness::prelude::*;

/// `/undo`: takes the last exchange back, from the last prompt the user
/// typed on, unless a turn runs.
fn undo(
    In(args): In<CommandArgs>,
    mut agents: Query<&mut Conversation, Without<ActiveTurn>>,
    mut commit: Commit,
) {
    let Ok(mut conversation) = agents.get_mut(args.agent) else {
        return;
    };
    let len = conversation.messages().len();
    if let Some(at) = (0..len).rev().find(|&at| typed(&conversation, at).is_some()) {
        commit.splice(args.agent, &mut conversation, at.., []);
    }
}
```

# Time

Time is Bevy's `Time`. `.run_if(on_real_timer(Duration))` runs a system
once per interval, while an entity with a `KeepAwake(Duration)` keeps the
sleeping loop running at that pace, and a one-off wait is a delayed
command, `commands.delayed().duration(Duration)`. Bevy's clock stops while
the machine sleeps; a time of day is a `WakeAt(SystemTime)` on an entity,
which gets `Due` once that time passed. A wait that may be put off or
called off, such as a debounce, is a `WakeAt::after(wait)` on an entity
of the plugin's: inserting a new one puts it off, and despawning the
entity calls it off.

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
outside it (what a command does once started is not confined), its
commands start there, and its prompt names it. Both are paths in the app's
workspace, `ProjectWorkspace`: this machine's (rig-tools' `Local`) unless
a plugin puts another in, such as one reached over SSH (a rig-tools
`Workspace`), with the `ProjectRoot` there. It may be swapped while the
app runs (a reconnect), so read it on each use. Every file tool, the
shell, `@path` attachments, the instruction files and `@` completion go
through it (rig-tools' `context::instruction_file_in`,
`attach::is_path_reference` and `attach::read_image` are their rules), and
other plugins learn it is remote from `workspace.0.host()`. A workspace
learns how a call's command ended in Review, from its `Published`
`ExitStatus` (`rig_tools::split_status` parts shell's output from its
status line). `RigHome` is the `RIG_HOME` layout (`home.0.session(..)`,
`home.0.auth(..)`), and `TokioRuntime` rig's tokio runtime. A plugin that
should also run without the harness's plugins inits the ones it reads
(`app.init_resource::<ProjectRoot>()`).

**The main-thread rule.** A workspace call may wait on the network. Tools
call it inside `blocking`; a system calls it inline only while `host()` is
`None`, else as `Running::spawn` work, holding what waits for it, as
`read_todo` below reads. Work a turn waits for, such as a snapshot as it
starts, holds the turn's first `TurnCall` (admitted in Prepare) until it
lands, as rig-basics' instruction refresh does. A plugin's own file tool
resolves paths as the coding tools do with `workspace.place(base, confine,
path)`, and `edit::apply(path, text, edits)` is what `edit` makes of a
text.

**Child processes.** `Processes` is how every child process starts, a
rig-tools `Launcher` (a wrapper command, a directory, environment): the
shell tool reads it for each command, so a sandbox plugin that sets it
(`processes.set(..)`) wraps every later process. A plugin builds its own
with `processes.launcher().command(program)`, or starts one whose output
it reads as it comes with `.spawn(program, args, Spawn { stdin,
merge_stderr })`: the `Process` hands its output over line by line
(`lines()`) or all at its end (`output(timeout)`, inside `blocking`), ends
with SIGTERM then SIGKILL on `stop(grace)`, and is killed with its process
group when dropped (`rig_tools::process`). Every launcher it hands out
drops the session's [secrets](#sessions-their-store-secrets-and-exiting):
a bare `std::process::Command` passes them on, so start processes with
`Processes`, and one that runs in the project through the workspace
(`workspace.0.spawn(&launcher, ..)`).

**Files and streams.** An entity with a `FileWatch` gets `FileChanges`
messages: first one listing the files under its path when armed (read them
then, so none made in between is missed), then one about the files changed
under it. A long-lived source, such as a socket, a server's notifications
or a thread's channel, is a `Running::stream(pool, &wake, items)` on an
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

**The store.** Every file of a session that holds conversation data goes
through its store, a rig-cassette `JournalStore` (`SessionStore`): each
agent's log and blobs, the effect log, `/resume`'s listing, a fork's
parent, the terminal view's notices and draft, and a plugin's own files,
under a directory of its own (`store.0.write_file("<plugin>/<name>",
..)`). The harness opens it from `SessionStores` once every plugin is
built; a plugin that keeps sessions elsewhere or sealed puts its own
`SessionStores::new(|dir| ..)` in while the app is built, and forks,
imports and `/resume` open their stores through it too. What is named by
path stays plain, under `Scratch`: tool output cut short, pasted images
and the text log; `Scratch(None)` keeps none. A copy to another store
lists its files with `files()`, which a wrapping store forwards, and a
command that rewrites a session skips one that `runs_elsewhere()`.

**Secrets.** A secret a plugin needs before the session is restored, such
as a store's passphrase, is asked for in `build` with
`app.add_secret(Secret::new(name, help))` and read from `Secrets`. A
launcher run at a terminal, a command's too, asks for each one its
environment (`RIG_SECRET_<NAME>`) lacks, once, and passes it to the
agent's process alone: no process rig starts gets it, not builds, the
shell, anything started through `Processes` or the terminal view's editor.
That keeps it from tools, the model's commands, build scripts and helper
programs; not from the same user reading the process's environment, nor
from plugin code, which is trusted.

**Exiting.** A plugin that ends the run triggers `Exit(reason)`:
`ExitReason::Quit`, `Reload` (restart on the staged build),
`Switch(Some(id))` or `Switch(None)` (another session, or a new one),
`Signal` or `Failed(code)`, such as a plugin's refusal to start. `Exit` is
the one door: no plugin writes `AppExit` itself. The first reason is the
`ExitReason` resource, for code in Bevy's `OnAppExitSystems`. Each exit
asked for is an app-level effect, an `ExitRequest`, and the app exits
when the first one settles. A plugin delays the exit by holding that
effect, with a deadline of its own: nothing cancels an exit, and a single
SIGTERM waits as long as the hold. A second exit asked for meanwhile is a
second effect, which a holder may take as "stop waiting".

`FirstStart` is a resource only in a session's first app (a fork's too),
not after `/reload` or `/resume`: the place for what a new session or fork
does once. `ForkSession { before, asker }` copies this session into a new
one (`SessionForked` names it and the request's `asker` entity, so a
plugin knows its own fork; the copy names its parent and where it was cut,
with the user's text of the message it stopped before,
`ForkedFrom::of(&*store)`), and `DeleteSession(id)` removes another
(`SessionDeleted`). rig-sessions' `NameSession { session, name }` names a
session in `/resume`.

```rust,no_run
use std::io;
use std::sync::Arc;

use rig_cassette::journal::JsonlDirStore;
use rig_harness::harness_protocol::SessionDir;
use rig_harness::prelude::*;

/// Keeps every session's files on the disk mounted at `/vault`, and
/// spilled output and pasted images nowhere.
#[derive(Default)]
pub struct VaultPlugin;

impl Plugin for VaultPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Scratch(None))
            .insert_resource(SessionStores::new(|dir: &SessionDir| {
                let id = dir.id().ok_or_else(|| io::Error::other("not a session of RIG_HOME"))?;
                Ok(Arc::new(JsonlDirStore::new(format!("/vault/{id}"))) as Arc<dyn JournalStore>)
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
    // Once: a second exit asked for passes, as "do not wait", and the app
    // exits when it settles; a holder that still has to write something
    // holds that one too.
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
```

# Adding to another plugin's lists

A plugin that owns a list builds it from a query over entities other
plugins spawn, the way the system prompt is built from `PromptSection`s:
spawn one with your text and keep its text current. An empty text, or
Bevy's `Disabled`, leaves it out, and despawning it takes it back.

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
`ProviderId::declare`, `Connector::provide` and `catalog_mut`), at any
time, so `--model`, `/model` and subagents take them like any other, and
an agent whose chosen model was not usable connects once it is there.

Spending is data too. rig-telemetry adds each settled model call's usage
to its agent's `Spending`, the session's `SessionSpending` (both saved)
and its turn's `TurnSpending`, split per model and per purpose (`turn`,
`compaction`, a plugin's `Effect::with_purpose`); a plugin that pays for
work that is no model request `record`s it. A `--print` run has nobody to
answer questions, so a plugin applies its policy for a run nobody
answers; `--answer=ID[,ID…]` answers each with the first of those choices.

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
sections are entities. Bevy's `Disabled` on one turns it off, and its name
is free for a replacement (a second one of a taken name is refused). Do it
in `Plugin::finish`, which runs once every plugin's `build` did. What a
plugin spawns in `build`, `finish` and `cleanup` is marked `ProvidedBy`
it; what it spawns later it marks itself, with its entity from
`Res<PluginEntity<MyPlugin>>`. Changing how another plugin's tool runs is
rarely a replacement: a Prepare or Gate system on its calls does it, a
Review system changes its results, and a default tool may have a switch of
its own, such as `rig_coding_tools::ShellOutput`. A plugin's stage systems
in a public set are turned off with a run condition, such as
`app.configure_sets(AgentPass, AttachSystems.run_if(|| false))` in
`finish` for rig-coding-tools' `@path` attachments.

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
may change it, in `finish` or at any time: its `ToolDef` (the rig-core
`ToolDefinition`: name, description, the JSON schema of its `parameters`)
and its `ToolRules` (lines of the system prompt). A property added to
another tool's `parameters` ([extending `task`](#extending-task) adds one)
is one the model may send, since calls are checked against the schema;
the tool still parses its arguments its own way, so the plugin that added
it takes it out of the call's `ToolRequest` in Prepare if the tool would
refuse it (`task` hands such arguments on). Edit only the `ToolRules`
lines you added.

# Testing a plugin

A plugin's test builds a small app: the agent core (`AgentPlugin`,
`JournalPlugin`), the plugin, and what it needs, run frame by frame with
scripted models. The rig repository's `rig-harness-test-support` crate
(`test-support/rig-harness-test-support`, a path dev-dependency) has the
parts: an app on a scratch session directory or a store in memory
(`app_in`, `app_on`), scripted models (`scripted_models`, `reply`,
`calls`), running frames until a condition or moving the clocks
(`run_until`, `advance`, `pass_wall_clock`), what happened (`collect`,
`log_ended`, `log_tool_outputs`), and answering questions (`AnswerAsks`,
`answer_open`). A plugin with stage systems has one more test,
`rig_defaults_test_support::assert_stages_ordered(MyPlugin)`, which fails
where it and the default plugins write the same data
[unordered](#ordering-against-other-plugins).

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

/// What `write` is called with: its schema declares `path`.
#[derive(serde::Deserialize, schemars::JsonSchema)]
struct WriteArgs {
    path: String,
}

/// A `write` that only says it wrote.
fn wrote(called: On<ToolCalled<WriteArgs>>, mut calls: Query<&mut Decision, ToolCalls>) {
    if let Ok(mut call) = calls.get_mut(called.call) {
        call.decide(Decider::plugin("write"), ToolResult::success(format!("wrote {}", called.args.path).into()));
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
    let model = "test/model".to_owned();
    app.world_mut().trigger(SetModel { entity: agent, model });
    app.world_mut().trigger(Deliver::new(agent, "Fix the readme.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| !world.resource::<Ended>().0.is_empty());
    let outputs = &app.world().resource::<ToolOutputs>().0;
    let output = outputs.get("c1").map(String::as_str).unwrap_or_default();
    assert!(output.contains("only under src/"), "{output}");
}
```

# Fronts, options and commands

A front, such as the terminal view, `--print`, a window or a chat bot,
offers itself with `app.add_front(..)`; one front takes each run (as
`Front` says) and alone owns stdin and stdout, and it sets itself up in
`finish` only if `app.runs_front(name)`. It sends what the user types as
`Deliver` from `Origin::user()`, shows `Notice`s (only an `Error` fails a
`--print` run or refuses a command), each agent's `Conversation` (or each
`Committed` change), its reply as it streams
([`StreamDelta`](#what-a-model-streams)) and its [transcript
entries](#the-transcript-entries-looks-and-scrolling), and sets `Shown` to
the agent the user looks at. If it can show questions, it spawns an
`Answerer` and answers them with
[`Pick`](#asking-the-user-questions-and-pickers). A front nobody sits at
drives its run with `app.run_headless(prompt)`, which ends once no turn
runs and nothing is held.

A plugin's own options are declared with `app.add_flag(Flag::new(name,
help))`, `.value("N")` for one that takes a value: `rig --name`, and
`rig --name VALUE` or `--name=VALUE`, reach the agent, which refuses an
option no plugin of its build declared. The plugin reads them from
`Invoked` (`flag(name)`, `flag_value(name)`) from `build` on, as the
[settings](#settings) example does, and `rig help` lists them.

A stopped turn is `Committed::Halt { agent, reason: Some(Halt::Stopped) }`
(`halted()` on the conversation says it now): a front shows a stopped
request from that, never by reading text, and knows the text the kernel
puts after it for the model by the constant `STOPPED`.

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
with, builds the app, writes what its plugins declared, and runs no frame.
So `build` only declares, and work goes in `finish` or systems. `rig help`
lists what the last build declared, `rig help stats` one command's usage;
an unknown command is refused, after a build, with the list.

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
    // A session's cost in USD (`None` unpriced), with `stores: Res<SessionStores>`:
    // `stores.open(&home.0.session(id)).ok().and_then(|s| rig_sessions::Meta::of(&*s)?.cost)`
    println!("{} saved sessions", ids.filter(|id| home.0.session(id).is_saved()).count());
    commands.trigger(Exit(ExitReason::Quit));
}
```

# The transcript: entries, looks and scrolling

A plugin puts an entry in an agent's transcript with a `TranscriptEntry`
entity `EntryOf` the agent: from an origin, after a message (`after`, a
`MessageId`; once a compaction took that message out, after its
summary), with a title shown collapsed and a text shown expanded. Every
front draws it, no model sees it, and the harness keeps it with the
session, so it comes back on `/resume`, `/reload` and in a fork, unless
`saved` is false.

The terminal view draws entries, and what the conversation holds from a
plugin's origin (such as rig-compaction's summary), with the look
`add_entry_renderer(origin, ..)` gives that origin. Its `TranscriptView`
is what plugins read and set: the items `expanded` and `selected` (Alt+Up
and Alt+Down select, Ctrl+O expands), the `highlights` lit wherever their
text is drawn (`match_case` or not; where two overlap, the first placed
wins), and where it is scrolled (`top()`). `ScrollTo { agent, item, row }`
scrolls to a `TranscriptItem` (a `Message`, a tool `Call`, an `Entry` or a
`Notice` by its index in `notices()`): to a row of it (`ItemRow::Top(n)`),
its first (`Reveal`) or the row of its `n`th lit match (`Lit(n)`); `item:
None` follows the newest output. A notice's `slot(agent, conversation)` is
where that agent's transcript draws it (a `TranscriptSlot`, ordered as drawn).
`DrawnTranscript`, once a plugin adds it, holds each row the last frame drew,
by item, with the matches lit (`lit_in(item)` is what `Lit(n)` counts).

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

/// `/mark why`: an entry after the agent's newest message.
fn mark(In(args): In<CommandArgs>, agents: Query<&Conversation>, mut commands: Commands) {
    let Ok(conversation) = agents.get(args.agent) else { return };
    let after = conversation.messages().len().checked_sub(1).and_then(|at| conversation.id(at));
    commands.spawn((TranscriptEntry::new("mark", after, "marked", args.args), EntryOf(args.agent)));
}

/// `/find text`: lights it, and scrolls to it in the last message the user
/// typed it in.
fn find(
    In(args): In<CommandArgs>,
    agents: Query<&Conversation>,
    mut view: ResMut<TranscriptView>,
    mut commands: Commands,
) {
    let Ok(conversation) = agents.get(args.agent) else {
        return;
    };
    let typed_it = |at: &usize| typed(conversation, *at).is_some_and(|text| text.contains(&args.args));
    let hit = (0..conversation.messages().len()).rev().find(typed_it);
    if let Some(id) = hit.and_then(|at| conversation.id(at)) {
        let item = Some(TranscriptItem::Message(id));
        commands.trigger(ScrollTo { agent: args.agent, item, row: ItemRow::Lit(0) });
    }
    let style = Style::new().on_yellow();
    view.highlights = vec![Highlight { text: args.args, style, match_case: true }];
}
```

# The status line

The terminal view's row under the transcript shows the `StatusItems` of
the agent shown and of its running turn, which every agent and turn has,
and the app's `AppStatus`, all `rig_tui`'s. A plugin shows an item there
in a system in `StatusSystems` that runs when what it shows changed:
`items.show(item)` puts the item at its place (its side and order) in
place of the one there, and an empty item clears the place. It goes behind
a default `tui` cargo feature, as a [renderer](#a-tool) does, and a plugin
that writes `AppStatus` registers it with
`app.init_resource::<AppStatus>()`, since the view may not be added. The
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
`Spawned`. `rig_activity::Activity` says what an agent does now, and
`MessageReader<Committed>` sees every change of every conversation.

A `TuiPanel` entity takes a side of the transcript (`Top`, `Bottom`,
`Left`, `Right`, with a ratatui `Constraint`), rows under the input
(`BelowInput`), or a box over the screen (`Over`). The plugin's own system
draws it into its `PanelCanvas` in `TuiSystems::Draw`, which runs only for
a frame that is drawn; a plugin whose own state changed writes a
`RequestRedraw`. `TuiScreen` is the terminal's size and `Focused` marks
the agent shown. These are there in any run, `--print` too.

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
    let lines: Vec<String> = agents
        .iter()
        .map(|(name, activity, conversation, sub)| {
            let indent = if sub { "  " } else { "" };
            let messages = conversation.messages().len();
            format!("{indent}{name} {}, {messages} messages", activity.status)
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
  editing keys included. A plugin binds a key or a sequence
  (`ctrl+x ctrl+e`) with `app.add_key_binding(binding, observer)`, which
  hears `KeyPressed`; `run_command("compact")` is an observer that runs a
  slash command. The first binding of a key stays, a global binding
  shadows the input's, and `Disabled` frees a binding's keys (rig-tui's
  `keys` docs have the rules). A binding with an `InputAction` (`Up`,
  `Send`, `Clear`, …) does that built-in action, as the view's own keys
  do. `KeyFocus(Some(panel))` gives a `TuiPanel` the keys until Esc or
  the panel goes.
- **Completion.** An `ArgCompletion::new("/model").candidate(text,
  detail)` entity offers candidates for a command's first argument
  (`.at(n)` for another).
- **The input.** `TuiInput` is the draft (`text()`, `set`, `insert`,
  `take`, …); Enter sends it to the agent shown (`Focused`).
  `PromptHistory` is the prompts Up and Down browse: the view appends each
  one sent in `HistorySystems::Record` and saves in `HistorySystems::Save`,
  and a system between the two filters what is kept.
- **The terminal.** `TerminalEvent` messages are what the terminal
  reported, keys and pastes, and `TerminalFocus` whether it has the user's
  focus. A `TerminalWrite(bytes)` message goes to the terminal between
  frames, such as a notification's escape code, `CopyToClipboard(text)`
  puts text on the clipboard (OSC 52), and a `RunInTerminal(command)`
  entity runs a program with the terminal, without the session's secrets,
  and gets its `TerminalExit`. `read_clipboard_image` saves the
  clipboard's image, off the main thread.

These are there in any run, inert without the terminal view. This plugin
writes the draft in `$EDITOR` on Ctrl+E, which shadows the view's end of
the line, makes Ctrl+K browse back as Up does, and keeps prompts with a
secret out of the history:

```rust,no_run
use std::path::PathBuf;
use std::process::Command;

use rig_harness::prelude::*;
use rig_harness::rig_core::redact;
use rig_tui::{
    AppKeyBindingsExt, HistorySystems, InputAction, KeyBinding, KeyPress, KeyPressed, PromptHistory,
    RunInTerminal, TerminalExit, TuiInput,
};

#[derive(Default)]
pub struct DraftKeysPlugin;

/// A program writing the draft in this file.
#[derive(Component)]
struct Draft(PathBuf);

impl Plugin for DraftKeysPlugin {
    fn build(&self, app: &mut App) {
        let scrub = no_secrets.after(HistorySystems::Record).before(HistorySystems::Save);
        app.add_systems(Update, (take_draft, scrub));
        app.add_key_binding(KeyBinding::new(KeyPress::ctrl('e'), "Write the draft in $EDITOR"), edit);
        app.world_mut().spawn((KeyBinding::new(KeyPress::ctrl('k'), "Up"), InputAction::Up));
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
what the agent already has; its winit plugin then runs the app's loop. In
`finish`, the plugin points the agent's `Wake` at winit's event loop. The
crate depends on `bevy` at exactly the agent's version, such as
`bevy = { version = "=0.20.0", default-features = false, features =
["ui"] }`. A window that is the front offers itself with `on_flag`.

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
        let window = WindowPlugin {
            exit_condition: ExitCondition::DontExit,
            ..default()
        };
        // Frames when woken, and at least every second as without a window.
        let mode = UpdateMode::reactive_low_power(Duration::from_secs(1));
        app.add_plugins(rig_harness::windowed(DefaultPlugins.set(window)))
            .insert_resource(WinitSettings { focused_mode: mode, unfocused_mode: mode });
        // Systems in `Update` draw the agents' `Activity` into the window.
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
