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
- [The effect log](#the-effect-log-reading-redacting-replaying): reading, redacting, replaying; [settings and command-line options](#settings-and-command-line-options)
- [Saved state](#saved-state-counting-every-tool-call), [changing history](#changing-history), [time](#time), [the process: project, files and child processes](#the-process-project-files-and-child-processes), [sessions and exiting](#sessions-and-exiting)
- [Adding to another plugin's lists](#adding-to-another-plugins-lists): `/model`, `/resume`, `/context`, `/agents`, `/usage`, `--print`; spending and budgets
- [Turning off or replacing what another plugin added](#turning-off-or-replacing-what-another-plugin-added), and [testing a plugin](#testing-a-plugin)
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
the same way (`rig-tui` for a terminal panel), and on `bevy_ecs` and
`bevy_reflect` (or `bevy`) at the agent's exact version, as Bevy's derives
expand to the Bevy crate the plugin's `Cargo.toml` names. The agent builds
from a rig checkout (`RIG_SOURCE`): a `[patch.crates-io]` table points the
rig crates at it, so plugins use the agent's own crates.

`/reload` in the agent, or its `reload` tool (both from `rig-reload`),
rebuilds and restarts in the same session once no turn runs. A failed
build leaves the running one; one that crashes at startup is rolled back.

# Finding names

The prelude holds Bevy's app, ECS, reflection and time preludes, the agent
runtime's components, stages and registries (`rig_harness::rig_ecs`), and
the harness's services. rig-core is `rig_harness::rig_core`, and each
default plugin's types are in its crate: the terminal view's panels, tool
renderers, status line, key bindings, input and pickers in `rig_tui`
(ratatui is `rig_tui::ratatui`, crossterm `rig_tui::crossterm`), what
agents do in `rig_activity`, `/reload`'s state in `rig_reload`, and so on
for each crate `plugins.toml` names.

What every plugin shares lives in rig-harness, never in another plugin:
how the agent was started (`Invoked`, with the plugins' options), the
user's settings (`Settings`, `add_setting`), the fronts (`add_front`), the
questions any front answers (`Ask`, `Answerer`), how the app exits
(`Exit`), the project, an agent's own directory and `RIG_HOME`
(`ProjectRoot`, `WorkDir`, `RigHome`), how child processes start
(`Processes`), file watches (`FileWatch`), wall-clock wakes (`WakeAt`), a
running call's progress (`ToolProgress`) and rig's tokio runtime
(`TokioRuntime`). What hosts without Bevy need too is in the libraries,
such as rig-core's `redact` and rig-tools' process runner.

Every one of these types is reflected, so the running agent can list
them with the `inspect` tool of the optional `rig-inspect` plugin
(uncomment its entry in `plugins.toml`): ask it before reading rig's
source. `world.list_components` and `world.list_resources` name what
exists, `registry.schema` gives a type's fields (`type_limit: {with:
["Saved"]}`: what is saved with the session), `world.query` with
`ProvidedBy` shows what each plugin added, and the `Diagnostics`
resource holds the warnings logged. A plugin's own Bevy Remote method is
sent once its system's entity has `rig_inspect::ReadOnlyMethod`.

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
`app.add_systems(AgentPass, my_gate.in_set(Stage::Gate))`. There are no
hook events. A system orders itself by its stage's phase and against other
plugins' public sets ([ordering](#ordering-against-other-plugins)). Every
effect is admitted before it starts, wherever and whenever it was spawned,
so every Prepare and Gate system sees it, in order. Prepare sees an effect
once; Gate sees it each pass it is ready to start, so once, and again
after each hold is released; Review an outcome each pass until it settles.

**An observer is for reactions that need no order**: counting, a status
item, a span, the plugin's own entities (`On<Add<Started>>`,
`On<Add<Settled>>`, `On<Add<ToolRequest>>`). Observers of one event run in
no fixed order, so an observer never edits stage data another plugin
edits; it may decide an effect, since the first decision wins.

**The conflict rule.** A hold is any entity `HoldOf(effect)`: it keeps the
effect from starting, or its outcome from settling, until the hold is
despawned or loses its `HoldOf`. Holds add up: releasing yours never
releases another plugin's, and "allow" means doing nothing. A hold says
why with its `Name`, such as `waiting for lease on src/a.rs`, which the
fronts show on the held call. The first decision wins: a gate decides
through the effect's `Decision` (`decision.decide(Outcome::refused(..))`),
so a later gate can never overturn a refusal; the kernel applies it as the
`Outcome` when the pass reaches Start. A reviewer edits the `Outcome` in
place (`Query<&mut Outcome, Unreviewed>`), once (a marker), since Review
sees it each pass; a reviewer ordered after it sees the edit, and the
record keeps both the first and the reviewed outcome. Plugins never insert
`Outcome` themselves, so Bevy sees every decision and edit
([ordering](#ordering-against-other-plugins)).

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

Bevy reports two stage systems with no order between them that write
the same component through their queries, such as two
`Query<&mut ModelRequest>` in Prepare, two gates that decide
(`&mut Decision`) or two reviewers that edit (`&mut Outcome`): which goes
first would be chance. Debug builds log it as a warning, and a plugin's
test makes it a failure (`assert_stages_ordered`,
[testing](#testing-a-plugin)). Holds and spawns go through `Commands`,
which it does not see, and need no order. Queries of different kinds of
effect never meet when they say so with a kind filter (`ToolCalls`,
`ModelCalls`, `IncomingMessages`, `PluginEffects`), and two over plugin
effects when one names its marker (`With<Ask>`) and the other excludes
it. Inside one system, two queries that write the same component must
rule each other out this way, or Bevy refuses the system when the app
starts (error B0001).

**Prepare and Review have phases**, chained in this order, so a plugin
orders a system by what it does, not by which plugin does what:

| Phase | What goes there | The default plugins' systems there |
|---|---|---|
| `PrepareSystems::Input` | what came in, rewritten before anything is read from it: a message's text, a tool call's arguments | |
| `PrepareSystems::Compose` | the parts that come from what a request names and from its agent | the files a message names (`@path`), the default system prompt, the coding tools' handler for the agent's `WorkDir` |
| `PrepareSystems::Edit` | the request edited as it will be sent: prompt, messages, tools, options, the model it goes to, its `Adapters`, a call's `CallHandler` | |
| `PrepareSystems::Fit` | the request as sent, measured against the window of the model it goes to and fitted | compaction |
| `ReviewSystems::Edit` | an `Outcome`, a `RetryVerdict` or a proposed `TurnOutcome`, edited in place | |
| `ReviewSystems::Act` | what was reviewed, acted on: a failed call taken over, a result followed up | compaction's recovery from an overflow |

So:

- **A stage system that edits or answers data another plugin may also
  edit goes in a public `SystemSet` of its plugin**, configured in its
  phase, or in its stage before or after one (`.after(ReviewSystems::Act)`):
  `app.configure_sets(AgentPass, MySystems.in_set(PrepareSystems::Edit))`.
- **Another plugin orders against the set only where it must go before
  or after that plugin**, never against a private system:
  `.before(MySystems)` or `.after(MySystems)`. A gate that holds or asks
  goes after the gates that decide the same kind of effect, so it sees
  their decisions.
- **Two plugins of other authors never depend on each other only to order
  themselves.** Where two meant to be used together edit the same data in
  one phase, the later one depends on the earlier one's crate and orders
  against its set, and its test checks both together,
  `assert_stages_ordered((Mine, Theirs))`.

The sets of Gate systems that edit shared data:

| Set | Crate | What it does | Order against it |
|---|---|---|---|
| `AnswerFromRecord` | `rig_ecs` | as Gate opens, before every Gate system: decides effects from a record of an earlier run (a session replay), which holds each gate's decision then | nothing; a cached or dry-run answer is no record and goes after the gates that refuse |
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
        if let Some(prompt) = call.system_prompt_mut() {
            *prompt = redact::text(prompt).into_owned();
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
and the `args`. The call ends when the plugin decides its `Decision`
(`Query<&mut Decision, ToolCalls>`), such as with
`ToolResult::success("Done.".into())`: the first decision counts. A tool
found while the agent runs, such as one an MCP server lists, is registered
from a system with `commands.queue(move |world: &mut World| {
world.spawn_tool(definition, handler, options); })`: its `ToolDefinition`,
and a rig-core `ErasedHandler` that answers its calls; despawning the
entity it returns removes it. How `rig-tui` draws a tool's calls is a
renderer the tool's plugin adds; the first one of a tool stays. It gets
the call, its result once there is one, and where the call stands
(`state`: `Waiting`, `Held` with the holds' names, `Running` with its
`ToolProgress` line, `Done`), which `header` and `result_lines` draw. A
crate that should also build without `rig-tui` puts it behind a default
cargo feature, as `rig-coding-tools` does with `tui = ["dep:rig-tui"]`.

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
with `ErasedHandler::layered` so the call runs through it (to give it a
secret the model never sees, say). Replacing a call's `CallHandler`, in
`PrepareSystems::Edit`, runs that one call another way, such as one
agent's `shell` calls in a sandbox (below), while `Processes` stays the
way every other call starts. When Gate opens, the kernel gives each call
the `Footprint` of the call as Prepare left it. A Gate system sees each
call when it is about to run: a call that conflicts with an earlier one of
the same reply (`Touches::conflicts_with`) is `Waiting` until that one
settles, so a gate judges it with the earlier calls' effects done. It
decides it with `Outcome::refused(why)`, which the model reads as the
tool's refusal, or with any `Outcome` (a cached result), or holds it. A
Review system edits each result in place before it is committed. The
reply's results go back in the order the model asked for them, refused
ones included. What a tool published beside its text, such as rig-tools'
`FilesRead` and `FilesChanged` (each file with the hash of the bytes read
or written) or `ExitStatus`, is the call's `Published` component. A call
of a tool the agent's `ToolAccess` leaves out is refused at Start (a
`Denied` refusal in its record): the model reads it as a tool it lacks.

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
(`mkdir x` then `ls x`), and only the tool knows what its calls touch. So
a gate that cannot get an answer refuses at once rather than holding: a
refused call settles, and the next ones run.

# Asking the user: questions and pickers

A question for the user is an `Ask` entity: a title, a body, the choices
offered and whether a line of text answers it too. It is an effect like
any, recorded with its answer, and any front answers it by deciding it
(`Decision`, with `Answer::Choice(id).outcome()` or `Answer::Text(..)`);
the first decision counts and closes it in every other front. A plugin
that waits on the answer spawns the `Ask` as a hold on what it asks about,
and an interrupt, which despawns its turn's holds, takes the question away
everywhere. A front that shows questions spawns an `Answerer` (for one
agent, or `Answerer::default()` for all); a plugin asks only when
`Answerers::can_answer` for the agent, and applies its own policy
otherwise, such as refusing. While a question that takes text is open, the
user's next line to its agent answers it instead of reaching the model.

A choice from a list is a `Picker` entity of the plugin's
(`rig_harness::pick`, in the prelude): a title, a body, items that each
may run a command line and show a `detail` that is kept whole while the
label is shortened to fit (a session's age), and whether typed text
answers it too (`free_text`). Inserting a new `Picker` replaces what it
shows, and despawning it closes it. Any front shows pickers its own way
and answers one, or an open `Ask`, by triggering `Pick { entity, chosen
}`: on a picker, the harness runs the chosen item's command, if any, for
the picker's agent or else the agent the user looks at (`Shown`), triggers
`Picked` on it (`Chosen::Item(index)`, `Chosen::Text` for typed text,
`None` when closed) and despawns it; on an `Ask`, it decides the answer.

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
`Outcome::custom(..)`, and then it never reaches the agent, as the
harness does to run a user's `/` line as a command. At `Start` the kernel
accepts what is left: it goes to the agent, which starts a turn with it
or keeps it for its running turn as its `mode` says. The messages an
agent accepted and has not sent yet are its inbox,
`Query<(&Incoming, &ForAgent), (With<Started>, Without<Outcome>)>`, and
`On<Add<Started>>` sees each message as it is accepted. A message's record
in the effect log is the custom kind `message`, and
`MessageRecord::read(&record.kind)` reads its text, origin and mode back.

**With the user's next message.** What a plugin wants the agent to read
when the user next writes, such as a background check's failure, is a
Prepare edit of that message: the plugin keeps it in its own state and
adds it to the next `Incoming` from `OriginKind::User` as an `Attachment {
label, content }` (`Added<Admitted>`). It then starts no turn of its own,
needs no hold, and is taken back by clearing that state. To have the agent
act on it at once, `Deliver` it instead. A message the agent accepted but
was not given yet, such as a `Note` to a busy agent, is taken back by
deciding it (`decision.decide(Outcome::custom( "withdrawn"))`): it leaves
the inbox and never reaches the conversation. A busy agent is given its
notes when its turn ends, so after that a later note says what changed.

A plugin that needs a mark on its own message spawns it:
`(Incoming::new(text, mode), ForAgent(agent), MyMark)` is all `Deliver`
does (the kernel records it under the agent's work and wakes the loop).
Once the kernel delivers a message into a turn, the turn it started or
the running turn it steered or carried on, the message gets
`DeliveredIn(turn)`, and a turn's first model call is recorded under the
message that started it. The message is despawned once it settles, so a
plugin that tags its message marks the turn then (`On<Add<DeliveredIn>>`)
and finds the turn's calls from it (`CallOf(turn)`), such as to route
them. A user's message may carry a title, `Origin::user().titled(..)`: it
still goes to the model as the user's words, and the conversation keeps
the origin (`Conversation::origin`), so a view can show where it came
from. What the user typed in message `at` of a conversation, without the
files attached or what a plugin delivered, is `typed(&conversation, at)`
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
            decision.decide(Outcome::custom("aside"));
            notices.write(Notice::info(agent, "Noted; not sent."));
        }
    }
}
```

# Model requests

A model call is an effect entity with a `ModelRequest`: the rig-core
`CompletionRequest` it sends (system prompt as its first message, the
messages, the tools offered, the options) and the `connection` it goes to,
else its agent's model (`Models::connect` makes one). A turn's own call is
a `TurnCall`, whose request the kernel builds from its agent when it is
admitted (an `On<Add<Admitted>>` observer runs before that, a Prepare
system after); a call built from an agent outside its turns, such as a
preview, is `BuildFrom(agent)`. A plugin changes a request in `Prepare`,
which sees each call once (`Added<Admitted>`), in `PrepareSystems::Edit`,
so it sees the whole prompt and compaction measures what is sent. Start
then fits the request to the model it goes to (an option that model cannot
take is left out, plugin edits stay) and makes the tool rules in the
prompt follow the tools it carries. Changing the system prompt or the
tools misses the provider's prompt cache.

- **Routing** a call to another model is a Prepare edit of its
  `connection`.
- **Adapting** the model, whichever it is, such as so it reads tools as
  text, is an `Adapter` pushed on the call's `Adapters`, a component
  beside the request (`Query<&mut Adapters, Added<Admitted>>`). Start
  applies them to the connection the call finally goes to, after every
  Prepare edit, in the order pushed, those placed `outermost()` last, then
  fits the request to the adapted spec: a router and an adapter never undo
  each other. An outermost adapter sees what every other one made of the
  model. An adapter wraps `connection.handler` with rig-core's
  `ErasedHandler::layered` (see each request and answer) or `map_stream`
  (rewrite each streamed item and the reply, recorded as rewritten), and
  says what the wrapped model takes on `connection.spec`.
- **What a call was sent as.** From `Start` on, a model call has
  `SentTo(connection)`, the model it went to with every adapter applied,
  and its `ModelRequest.request` is the request as sent: a plugin reads
  these in `Review` (or on `Add<Settled>`) to know what the other adapters
  made of it. A fact a `map_stream` closure inserts in the call's
  `PublishedContext`, such as that it read tool calls from text, is on the
  call as `Published` once it is answered, and recorded with it.
- **Tools.** The kernel offers an agent the tools its `ToolAccess` allows
  and leaves taking them away to the model the call goes to: a call routed
  from a model that calls no tools to one that does is sent its tools. A
  tool left out of a request is not offered but still runs if called: a
  `Gate` system refuses a call (`Outcome::refused`), and `ToolAccess` on
  the agent is the hard limit.
- **Below every adapter: the HTTP client.** `models.0.wrap_http(|inner|
  ..)`, in `build` before an agent connects, wraps the HTTP client of
  every model call, a turn's and a plugin's own, on the shared client, a
  provider's own and a sign-in's alike. A rig-core `HttpMiddleware`
  (`inner.with_middleware(..)`) sees each request's headers and body and
  each response's status and headers, the place to rotate API keys; a
  wrapper of the whole `HttpClientExt` also sees the streamed bytes, such
  as a raw log of the wire (its methods are written in
  `rig_core::http_client::Bytes`). A provider plugin that builds its own
  client wraps it with the `Transport` its `Provide` or `SignIn` is handed
  (`transport.wrap(client)`). The effect log never holds the wire.
- **Which call a request serves.** Every request a model call sends
  carries a `rig_core::serve::Serving` in its extensions
  (`request.extensions().get::<Serving>()`): the call's effect id, the
  same as its entity's `Effect::id()`, the handler it went to, and the
  provider it is reached through. A wrapper keeps what it learns per
  call by that id, such as which key served it, and a system reads it
  back when the call settles; a request without one serves no call, such
  as a model listing. A wrapper runs on the IO task, not in a frame: to
  tell the app at once, it sends into a channel whose receiver is a
  `Running::stream` ([Files and
  streams](#the-process-project-files-and-child-processes)), which wakes
  the loop.
- **Reaching a vendor.** `models.0.provide(vendor, provider)` reaches
  every model of a vendor through a provider of the plugin's own, such as
  `ConnectOptions` with a key: it replaces one provided before, which
  `models.0.provider(vendor)` hands over first, for a provider that wraps
  it. A model the provider has no key for (`ConnectError::MissingKey`)
  falls back to the sign-ins, as one the environment has no key for does.
- `rig_ecs::prompt::system_prompt` and `tool_rules` put a prompt together
  exactly as the kernel does, such as for a view of what an agent is told.
  `rig_memory::HeuristicTokenCounter` estimates tokens as rig-compaction
  does.

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
`SectionOf(agent)` in that agent's alone, despawned with it.

```rust,no_run
use std::sync::Arc;

use rig_harness::prelude::*;
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

Every model call streams, a turn's and a plugin's own alike. Each event
is a `StreamDelta { effect, agent, event }` message carrying rig-core's
`StreamEvent`: a part's `Start` (a tool call's name), its `Text`,
`Reasoning` or `Arguments` fragments, and its `End` (a finished tool
call with its id); the event's `part` tells parallel tool calls apart.
It is written in `Arrive` in the order it came, before the call's
outcome is applied; Bevy keeps a message at least two frames, so a reader
in `PostUpdate` gets the last events of a reply that settled in the
same frame. They are what the call's adapters left: a `map_stream` rewrite
comes before them. A turn's call also adds its text and reasoning up in
its `Partial`, which is gone once the reply is in the conversation. From
its stream's first item of any kind, text, a tool call or the end, the
call has a `Streamed { first, last, items }` beside its `Started { at }`:
`last` is its latest item, the reply's end once it ended.

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
the agent's turns over the whole session (saved with the agent) and its
`first_message`, the `MessageId` of the message it begins at
(`conversation.start_of(turn.first_message)` is its index now, as splices
move messages). The kernel makes it `ForAgent` its agent when it is
spawned, so it goes with the agent. It is spawned before its first request
is built, so an `On<Add<TurnOf>>` observer's edit, such as a prompt
section's new text, is in that request; once the turn ends it loses
`TurnOf` and keeps `ForAgent`. A turn's end is a proposal: when the model
answers without calling tools, or the turn fails, the kernel proposes a
`TurnOutcome` on the turn in `Commit` (`Answered(message)`,
`Failed(reason)`), and a `Review` system sees it in the next pass
(`Added<TurnOutcome>`). It may edit it in place (`Query<&mut TurnOutcome,
With<TurnOf>>`), hold it (spawn an entity `HoldOf(turn)`, despawn it when
done), or carry the turn on: a `Queue` or `Steer` message the agent
accepts takes an `Answered` proposal away, one on its way keeps the turn
from ending meanwhile, and a new `(TurnCall, CallOf(turn),
ForAgent(agent))` takes any away once it is admitted; a proposal taken
away takes its holds with it (they are despawned), and the next proposal
is reviewed afresh. A plugin may insert `TurnOutcome::Failed(reason)` or
`Stopped(Some(reason))` at any time, with a plain `insert`, such as a
budget: the turn's calls that have not started never start, the others are
answered as stopped (or with their `IfStopped`), and the turn ends. Once
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

#[derive(Default)]
pub struct TurnEndsPlugin;

impl Plugin for TurnEndsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(AgentPass, end_with.in_set(Stage::Gate))
            .add_systems(AgentPass, check_the_tests.in_set(Stage::Review))
            .add_observer(announce);
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
            decision.decide(Outcome(Ok(effect::Outcome::Completion(reply))));
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
`EffectKind` (`EffectKind::Custom` for a kind of its own) and a `handler`,
which the kernel runs at Start, or none, an open effect the plugin decides
(`Decision`) whenever it is done. Other plugins' Gate systems can refuse
it, Review systems edit its outcome, and a replayer may answer it from the
effect log; `CallOf(turn)` keeps a turn open until it settles. Work off
the main thread is a `Running` task on the effect's entity,
`Running::answering(pool, wake, future)`, whose output is its outcome.

**What a running effect would answer if stopped.** Its owner keeps an
`IfStopped(outcome)` on it current, such as the output so far, which the
kernel answers with in place of "interrupted" when the user interrupts,
the turn ends or the app exits (and logs for the restart). The owner of a
running tool call also keeps a `ToolProgress` line on it, such as `12 of
40 files`, drawn under the call and written to stderr by `--print`.

**An owner follows its effect's successor.** Another plugin may take the
effect over ([taking over](#how-a-plugin-steps-in-effects-and-stages)),
such as a watchdog that sends a stalled call again with
`commands.queue(resend(call))`: the effect then settles as superseded,
with no outcome applied, and its `SupersededBy` names the successor. A
plugin that waits for its own effect's outcome by a marker registers
`app.add_observer(follow_successor::<Marker>)`, which moves the marker,
and the hold the effect keeps, to each successor; the kernel follows its
own turn and tool calls. A model call a plugin makes is such an effect
too: a rig-core model it calls directly is one the kernel, the effect log
and the spending never see.

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
            decision.decide(Outcome::custom("deployed"));
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
and a plugin spawns one itself for agents of its own, such as the sides of a
debate. Like any effect, Prepare systems edit it (`Query<&mut SpawnSubagent,
Added<Admitted>>`: its model, tools, system prompt) and Gate systems refuse
or hold it. When it starts, the plugin spawns the agent with every setting
the request leaves out taken from the parent (its `WorkDir` too), puts it on
the request as `SpawnedAgent(agent)`, and answers with the new agent's id,
`{"agent": "<AgentId>"}`, which the record keeps.

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
replacement as `reviewed`), its times, the agent it worked for (`scope`,
the agent's `AgentId`), the agent's `turn` it was made in (`Turn.index`),
and the effect it was made under (`parent`). The `rig-telemetry` plugin
writes them to the session's effect log in `Last`, in `WriteJournal`, and
reads them back through its `EffectLog` resource, there when the session
has a directory: `read()` gives every record, and `read_matching(&filter)`
those an `EffectFilter` keeps, by agent, kind name (`completion`,
`tool_call`, `message`, `subagent` or a plugin's own) and turn. It holds
what was written by the end of the last frame.

**Redacting.** A record holds what its effect really carried: tool output
and the diffs of changed files, the request as the model got it. A plugin
that keeps a secret out of the log changes the records in place with
Bevy's `MessageMutator<Recorded>`, in `Last`
`.after(StopTurns).before(WriteJournal)` (both in `rig_harness::rig_ecs`),
so it sees the records a stop on exit makes too, and the log gets only
what it left. A reader earlier in the frame sees the record unchanged.
rig-core's `redact` finds API keys and tokens by their shapes:
`redact::json` hides every one in a record, and `redact::text` in a line.

**Replaying.** A replayer answers effects from a recorded session's log
before they start, a model call with an adapter placed outermost (so the
recording stands for the whole adapted model, and no `map_stream`
rewrite runs twice) and every other effect with a Gate system that
decides. rig-cassette's `RecordedAnswers` takes each agent's recorded
answers by kind, in order; its `Divergence` carries the record it
expected. Agent ids are new in each session, so a replayer pairs each
agent with the recorded one it stands for: the first agent with the
first recorded agent, and the n-th spawn of a paired parent with that
parent's n-th recorded `subagent` effect, whose recorded answer names the
recorded agent, whatever order the subagents start in. It runs `subagent`
effects again, since only running one spawns the agent, and leaves the
app's `exit` effect alone.

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

# Settings and command-line options

A plugin's settings are its table of the user's `RIG_HOME/settings.toml`:
`app.add_setting::<T>("section")` in `build` inserts the resource `T` read
from `[section]`, else `T::default()`; a table that does not read as `T`
keeps the default and is shown at startup. So that it never reads the
user's file, a test inserts `Settings::from_toml(..)` before adding the
plugin (the test support's apps insert `Settings::default()`).

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
`#[reflect(Component, Saved)]` is kept with the session (a generic type
once registered, `app.register_type::<T>()`): each change is logged, and a
restart, `/reload` or `/resume` brings the newest value back, inserted on
the restored agent before `SessionRestored` exists, so an `On<Insert<T>>`
observer can re-arm what it says. A resource that says
`#[reflect(Resource, Saved)]` is kept the same way, for what belongs to
the whole session; the plugin inserts it in `build`, and the saved value
replaces it. `On<Add<ToolRequest>>` sees every tool call as the model asks
for it; one a restart runs again is `Restored`, like all the work a
restart carries on, and is gated again (holds and approvals are not
saved). Calls made before the plugin was added are in the conversation a
restored agent is spawned with: the observer of the plugin's component
counts them, and a saved count replaces that right after. An assistant
message is the tuple variant `Message::Assistant(AssistantMessage)`.

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
back changed. A user message spliced beside a user message joins it. Every
message has a `MessageId` (`conversation.id(i)`) that stays while messages
around it are added, cut or replaced, while any of its items survives, and
across restores: remember messages by id, found again with `index_of(id)`.
`start_of(id)` is where a span that began at `id` begins now, such as a
turn (`Turn::first_message`). A splice that reaches the end clears the
halt (`conversation.halted()`); one before it keeps it.

Each splice is announced as `Committed::Splice { agent, removed, changed }`,
the ids of the messages gone and of those it made or changed; who spliced
is read from the changed messages' item origins. A compaction splices its
summary, one user item of origin `rig_compaction::summary_origin()`, in
place of the messages it summarized, and requests send the conversation
as it is. A splice before the end costs the provider's prompt cache from
there on: that is the splicing plugin's choice. `commit.retire(agent)`
despawns an agent for good, so a restart brings back neither it nor the
agents it spawned. `rig_ecs::journal::read_session(store)` reads a
session's logs, another session's too, without restoring or writing
anything: each agent's conversation as it is and its saved components.
`fork_session(from, to, before)` copies a session as it was when a
message was made, `before` being its agent's id and the message's id:
every log up to the records made before that message's first record.

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

fn build(app: &mut App) {
    app.add_command("undo", "Take the last exchange back", undo);
}
```

# Time

Time is Bevy's `Time`. `.run_if(on_real_timer(Duration))` runs a system
once per interval, and a one-off wait is a delayed command,
`commands.delayed().duration(Duration)`. The loop sleeps when nothing
happens and wakes on its own for a delayed command; for an interval, an
entity with a `KeepAwake(Duration)` keeps it running at that pace while it
lives. None needs a thread. Bevy's clock stops while the machine sleeps; a
time of day is a `WakeAt(SystemTime)` on an entity, which gets `Due` in
`First` once that time passed (the loop wakes for it), and is armed again
by inserting a new `WakeAt`. A delayed command cannot be taken back, so a
wait that may be put off or called off, such as a debounce, is a
`WakeAt::after(wait)` on an entity of the plugin's: inserting a new one
puts it off, and removing it or despawning the entity calls it off (the
exit hold in [sessions and exiting](#sessions-and-exiting) is one).

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

# The process: project, files and child processes

`ProjectRoot` is the directory the session works in. An agent with a
`WorkDir`, such as its own worktree, works there instead, and so does an
agent it spawns: the coding tools resolve its paths from it and refuse one
outside it (what a command does once started is not confined), its
commands start there, and its prompt names it. `RigHome` is the `RIG_HOME`
layout (`home.0.session(..)`, `home.0.auth(..)`), and `TokioRuntime` rig's
tokio runtime. A plugin that should also run without the harness's plugins
inits the ones it reads (`app.init_resource::<ProjectRoot>()`).

**Child processes.** `Processes` is how every child process starts, a
rig-tools `Launcher` (a wrapper command, a directory, environment): the
shell tool reads it for each command, so a sandbox plugin that sets it
(`processes.set(..)`) wraps every later process. A plugin builds its own
with `processes.launcher().command(program)`, or starts one whose output
it reads as it comes with `.spawn(program, args, Spawn { stdin,
merge_stderr })`: the `Process` hands its output over line by line
(`lines()`), ends with SIGTERM then SIGKILL on `stop(grace)`, and is
killed with its process group when dropped (rig-tools' types, as
`rig_harness::rig_tools::process`). A plugin's own file tool follows the
coding tools' rules with rig-tools: `fs::resolve(base, path)` makes a path
absolute as they and their footprints do, their `base` and `confine` say
where they work, `fs::place(base, confine, path)` refuses a path outside
`base` as they do, `edit::apply(path, text, edits)` is what `edit` makes
of a text, as for a preview, and `ReadArgs::show(text)` what `read` shows.

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
or a new one), `Signal` or `Failed(code)`, such as a plugin's refusal to
start (`Failed(2)` in `Startup`, with a notice). `Exit` is the one door:
no plugin writes `AppExit` itself. The first reason of a run is the
`ExitReason` resource from the moment it is asked, for code on the way
out in Bevy's `OnAppExitSystems` (a background job kept across a reload,
stopped on a quit). Each exit asked for is an app-level effect, an
`ExitRequest`, and the app exits when the first one settles, whatever
answered it. A plugin delays the exit by holding that effect, with a
deadline of its own: nothing cancels an exit, the agents keep working
while it is held (hold their work too to freeze it), and a single SIGTERM
waits as long as the hold. A second exit asked for meanwhile is a second
effect, which a holder may take as "stop waiting". An `AppExit` Bevy
writes itself, such as a window closing, cannot be held.

`FirstStart` is a resource only in a session's first app (a fork's too),
not after `/reload` or `/resume`: the place for what a new session or fork
does once. `ForkSession { before, asker }` copies this session into a new
one (`SessionForked` names it and the request's `asker` entity, so a
plugin knows its own fork; the copy names its parent and where it was cut,
with the user's text of the message it stopped before,
`ForkedFrom::of(&dir)`, so a view of a session's family reads no other
session's log), and `DeleteSession(id)` removes another
(`SessionDeleted`). A plugin's files go in its own `SessionPaths`
directory: each top-level `*.jsonl` there is an agent's log. rig-sessions'
`NameSession { session, name }` names a session in `/resume`.

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

Spending is data too. rig-telemetry adds each settled model call's usage
to its agent's `Spending`, to the session's `SessionSpending` (both saved
with the session) and its turn's `TurnSpending`; a `Spending` keeps the
total and its split per model and per purpose (`turn`, `compaction`, a
plugin's `Effect::with_purpose`). A budget reads it; a plugin that pays
for work that is no model request `record`s it in both.

A `--print` run has nobody to answer questions: it is no `Answerer`, so a
plugin applies its policy for a run nobody answers, and a question asked
anyway is refused with why on stderr. `--answer=ID[,ID…]` answers each one
with the first of those choices it offers.

```rust,no_run
use rig_basics::{AgentColumn, ColumnOf};
use rig_harness::prelude::*;
use rig_models::{ModelItem, ModelLabel};
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

/// A choice in `/model` that is no catalog model: `/model router` (or its
/// picker item) runs `/route on`, and what an agent on it shows.
fn add_router(app: &mut App) {
    let item = ModelItem::new("router", "router  the cheapest model that fits", "route on");
    app.world_mut().spawn(item);
    app.add_command("route", "Pick the model per request: /route on", route);
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
tool's `parameters` ([extending `task`](#extending-task) adds one) is one
the model may send, since the kernel refuses only arguments the schema
does not declare; the tool still parses its arguments its own way, so the
plugin that added the property reads it, and takes it out if the tool
would refuse it, from the call's `ToolRequest` in Prepare (`task` hands
such arguments on instead). Edit lines of `ToolRules` you added, found by
their text, and leave the others' alone.

# Testing a plugin

A plugin's test builds a small app: the agent core (`AgentPlugin`,
`JournalPlugin`), the plugin, and what it needs, run frame by frame with
scripted models. The rig repository's `rig-harness-test-support` crate
(`test-support/rig-harness-test-support` in the rig checkout, a path
dev-dependency) has the parts every plugin test needs:

- **Apps.** `app_in(&home, &id)` is an app on a session directory of
  `scratch_home(name)`, as the harness keeps one, so `SessionPaths`, forks
  and the effect log (`rig_telemetry::EffectLogPlugin`, read back with
  `EffectLog`) work; its failed assertions still print. `app_on(&store)`
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
  due (add `rig_harness::clock::ClockPlugin`). `first_agent(world)` finds
  the agent; `start_call` spawns a tool call as a model would.
- **What happened.** `collect::<M>(&mut app)` keeps every message `M`
  (`Notice`, `Recorded`, `StreamDelta`) in `Collected<M>`, read in `Last`;
  the observers `log_ended` and `log_tool_outputs` keep how each turn
  ended (`Ended`) and each tool call's output by call id (`ToolOutputs`).
- **Questions.** The `AnswerAsks` front answers every question from an
  `Answers` script; `answer_open(world, title, &answer)` answers one
  later, as the user would.
- **Order.** A plugin with stage systems has one test,
  `rig_defaults_test_support::assert_stages_ordered(MyPlugin)`, from the
  `test-support/rig-defaults-test-support` crate: it builds the harness
  with the default plugins and the plugin, runs no frame, and fails,
  naming the systems, where two stage systems write the same data
  unordered ([ordering](#ordering-against-other-plugins)).

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
            decision.decide(Outcome::refused("write only under src/"));
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
        call.decide(ToolResult::success(format!("wrote {}", called.args.path).into()));
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

# A front

A front shows the agents to someone and takes what they type: the terminal
view (`rig-tui`), `--print` (`rig-print`), a window, a web server, a chat
bot. It offers itself in `build` with `app.add_front(..)`, saying which
runs it takes (`on_terminal()`, `headless()`, or `on_flag(name, help)` for
a run given its option), and one front takes each run: the one whose
option was given, else the first offered for the kind of run. It asks
`app.runs_front(name)` in `finish` and sets itself up only then. It sends
what the user types as `Deliver` from `Origin::user()`, shows `Notice`s
(`Info`, `Warn` or `Error`; only an error fails a `--print` run or refuses
a command) and each agent's `Conversation` (or each change as `Committed`)
and its reply as it streams ([`StreamDelta`](#what-a-model-streams)), and
sets `Shown` to the agent the user looks at. If it can show questions, it
spawns an `Answerer` and answers [`Ask`s and
`Picker`s](#asking-the-user-questions-and-pickers) with `Pick`. A front
nobody sits at drives its run with `app.run_headless(prompt)`, which ends
once no turn runs and nothing is held, and writes out `HeadlessEnded` (the
answer, and the `ExitReason`); one whose runs leave nothing to resume is
`sessionless()`. Several fronts may run side by side (a web server beside
the terminal view); only the one that takes the run owns stdin and stdout.

When the user stops a turn before it answered, `Committed::Halt { agent,
reason }` says so with `Some(Halt::Stopped)` (`rig_ecs::agent::Halt`;
`Kept` when the message was left for the next turn, such as after its turn
failed, and `None` when it was asked for again; `halted()` on the
conversation says it now): a front shows a stopped request from that,
never by reading text. For the model, the kernel puts the `STOPPED` text
between the stopped request and the next message that joins it; a view
that draws a whole conversation, such as a restored one, recognizes that
item by the constant `rig_ecs::agent::STOPPED`, not by its words.

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
`Spawned`. The activity plugin's `rig_activity::Activity` says what an
agent does now (its status and running tools), and its `MessageFeed`
resource holds the latest deliveries. `MessageReader<Committed>` sees
every change of every conversation as the session log records it, and
`On<Add<Settled>>` on a turn with a `TurnOutcome` each turn's end ([How a
turn ends](#how-a-turn-ends)).

A `TuiPanel` entity takes a side of the transcript (`Top`, `Bottom`,
`Left`, `Right`, with a ratatui `Constraint`) or a box over the screen
(`Over`). The plugin's own system draws it into its `PanelCanvas` in
`TuiSystems::Draw`, which runs only for a frame that is drawn. Frames are
drawn when agents or panels change; a plugin whose own state changed
writes a `RequestRedraw`. `TuiScreen` is the terminal's size and `Focused`
marks the agent shown. The crate depends on `rig-tui` and `rig-activity`
for these, there in any run, `--print` too, which draws no frame.

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
  from the user's settings. A binding that carries an `InputAction`
  (`Up`, `Send`, `Clear`, …) does that built-in action in place of
  `KeyPressed`, in order with the frame's other keys, as the view's own
  keys do; an `InputAction` message does it after the frame's keys.
- **A panel's keys.** `KeyFocus(Some(panel))` gives a `TuiPanel` the keys:
  its bindings in `KeyScope::Panel(panel)` and the global ones apply, not
  the input's, and Esc it does not bind, or the panel going, gives the
  keys back. While it has them, a key it does not bind edits nothing, so
  a panel with a text field of its own, such as a palette's filter, reads
  the typed keys in order from `TerminalEvent` (below).
- **Completion.** An `ArgCompletion::new("/model").candidate(text,
  detail)` entity offers candidates for a command's first argument
  (`.at(n)` for another), which the input completes as it does commands
  and `@` paths.
- **The input.** `TuiInput` is the draft: `text()`, `cursor()`, `set`,
  `insert`, `take`, `set_cursor`; `is_changed()` says it was edited. Enter
  sends it to the agent shown (`Focused`). `PromptHistory` is the prompts
  Up and Down browse and their file: the view appends each one sent in
  `HistorySystems::Record` and saves in `HistorySystems::Save`, both in
  `Update`, and a system between the two filters what is kept.
- **The terminal.** `TerminalEvent` messages are what the terminal
  reported, keys and pastes; a plugin may read them, and write some.
  `TerminalFocus` says whether the terminal has the user's focus (`None`
  until it reports). A `TerminalWrite(bytes)` message goes to the
  terminal between frames and on the way out, such as a notification's
  or the window title's escape code, and `CopyToClipboard(text)` puts
  text on the clipboard (OSC 52). A `RunInTerminal(command)` entity runs
  a program with the terminal (the view steps aside and draws again
  after), and gets its `TerminalExit`.
- **Pickers and questions.** The view shows each
  [`Picker` and open `Ask`](#asking-the-user-questions-and-pickers) over
  the transcript, one at a time in the order they came, and answers it
  with `Pick`.

These are there in any run, inert without the terminal view. This plugin
writes the draft in `$EDITOR` on Ctrl+E, which shadows the view's end of
the line, offers a few prompts on Ctrl+T, makes Ctrl+K browse back as Up
does, and keeps prompts with a secret out of the history:

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

const PROMPTS: [&str; 2] = ["Explain this code: ", "Write a test for "];

impl Plugin for DraftKeysPlugin {
    fn build(&self, app: &mut App) {
        let scrub = no_secrets.after(HistorySystems::Record).before(HistorySystems::Save);
        app.add_systems(Update, (take_draft, scrub));
        app.add_key_binding(KeyBinding::new(KeyPress::ctrl('e'), "Write the draft in $EDITOR"), edit);
        app.add_key_binding(KeyBinding::new(KeyPress::ctrl('t'), "Start from a prompt"), prompts);
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

`rig_harness::windowed(DefaultPlugins)` is Bevy's `DefaultPlugins` without
the log, task pools, clock, signal handler and loop the agent already has;
its winit plugin then runs the app's loop. In `finish`, where winit's
event loop exists, the plugin points the agent's `Wake` at it, so agent
activity and terminal input wake the window instead of a poll. The crate
depends on `bevy` at exactly the agent's version, with the features it
draws with, such as `bevy = { version = "=0.20.0", default-features =
false, features = ["ui"] }` for Bevy UI. A window beside the terminal view
is no front; one that is the front offers itself with `on_flag` (`--gui`).

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
