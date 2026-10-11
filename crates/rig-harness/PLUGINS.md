Writing a plugin for the rig agent: the rules every plugin follows, then
one short, copy-ready example per extension point. Every name they use
comes from `use rig_harness::prelude::*;`, except a plugin crate's own
types, such as the terminal view's panels from `rig_tui`.

- [Making one](#making-one) and [finding names](#finding-names)
- [How a plugin steps in: effects and stages](#how-a-plugin-steps-in-effects-and-stages): the rules
- [A tool](#a-tool) and [gating and reviewing its calls](#gating-and-reviewing-tool-calls)
- [Asking the user](#asking-the-user)
- [A slash command](#a-slash-command) and [what the user types](#what-the-user-types)
- [Model requests](#model-requests) and [how a turn ends](#how-a-turn-ends)
- [Effects of a plugin's own](#effects-of-a-plugins-own)
- [Settings and command-line options](#settings-and-command-line-options)
- [Saved state](#saved-state-counting-every-tool-call), [time](#time), [the process: project, files and child processes](#the-process-project-files-and-child-processes), [sessions and exiting](#sessions-and-exiting)
- [Adding to another plugin's lists](#adding-to-another-plugins-lists): `/model`, `/resume`, `/context`, `/agents`, `/usage`, `--print`; spending and budgets
- [Turning off or replacing what another plugin added](#turning-off-or-replacing-what-another-plugin-added)
- [A front](#a-front), [the status line](#the-status-line), [a terminal panel](#what-agents-do-and-say-in-a-terminal-panel), [keys, the input and pickers](#keys-the-input-and-pickers-in-the-terminal-view), [a window](#a-window)

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
| `Review` | plugins | outcomes not settled yet (`Unreviewed`): read, replace, hold, or take a failure over; a turn's proposed end |
| `Commit` | kernel | settles what is answered and unheld (`Settled`), appends results, carries turns on or ends them |
| `Admit` | kernel | `Admitted` on every new effect; builds turn requests; `Waiting` on a call behind a conflicting one |
| `Prepare` | plugins | what was admitted this pass (`Added<Admitted>`): edit the request, once |
| `Gate` | plugins | what is about to start (`Undecided`): hold it or answer it |
| `Start` | kernel | starts what is admitted, unheld and unanswered |

**A hook is an ordinary system in a stage** that queries stage data:
`app.add_systems(AgentPass, my_gate.in_set(Stage::Gate))`. There are no
hook events. Order between plugins is Bevy's `.before` and `.after` on the
systems. Every effect is admitted before it starts, wherever and whenever
it was spawned, so every Prepare and Gate system sees it, in their order.
Prepare sees an effect once; Gate sees it each pass it is ready to start,
so once, and again after each hold on it is released; Review sees an
outcome each pass until it settles.

**An observer is for reactions that need no order**: counting, a status
item, a span, the plugin's own entities (`On<Add<Started>>`,
`On<Add<Settled>>`, `On<Add<ToolRequest>>`). Observers of one event run in
no fixed order, so an observer never edits stage data another plugin
edits; it may answer an effect, since the first answer wins.

**The conflict rule.** A hold is any entity `HoldOf(effect)`: it keeps the
effect from starting, or its outcome from settling, until the hold is
despawned or loses its `HoldOf`. Holds add up: releasing yours never
releases another plugin's, and "allow" means doing nothing. The first
answer wins: a gate filters `Without<Outcome>` and answers with
`insert_if_new`, so a later gate can never overturn a refusal. A reviewer
replaces an outcome with `insert`, ordered against other reviewers, and
the record keeps both the first and the reviewed outcome. Edits nobody
ordered must commute; debug builds warn about two unordered stage systems
writing the same request or outcome. Two unordered gates in one pass may
hold and answer the same call: that is harmless, the hold goes when the
call settles.

**Deciding once.** A gate that only reads (a path rule) decides afresh
each time it sees a call. One whose decision asks someone or has side
effects marks the call it decided on (a `SparseSet` marker such as
`Approved`), skips marked calls, and leaves held calls alone, so a gate
ordered after it decides only once the first released the call.

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
fn src_only(calls: Query<(Entity, &ToolRequest), Undecided>, mut commands: Commands) {
    for (call, request) in &calls {
        let path = request.call.function.arguments["path"].as_str().unwrap_or_default();
        if request.call.function.name.as_str() == "write" && !path.starts_with("src/") {
            commands.entity(call).insert_if_new(Outcome::refused("write only under src/"));
        }
    }
}

/// Keeps the first line of a failed `shell` call's output, before it is
/// committed.
fn shorten_errors(
    calls: Query<(Entity, &ToolRequest, &Outcome), (Unreviewed, Without<Shortened>)>,
    mut commands: Commands,
) {
    for (call, request, outcome) in &calls {
        let Ok(Answered::ToolResult { result }) = &outcome.0 else {
            continue;
        };
        let text = result.output().as_text().unwrap_or_default();
        if request.call.function.name.as_str() == "shell" && !text.contains("[exit code 0]") {
            let first = text.lines().next().unwrap_or_default().to_owned();
            let shortened = ToolResult::success(ToolOutput::text(first));
            commands.entity(call).insert((Outcome::from(shortened), Shortened));
        }
    }
}

/// On a call whose outcome was shortened: once is enough.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Shortened;
```

# A tool

Any `PortableTool` (or `rig_core::tool::Tool`). Its rules go into the
system prompt of every agent it is offered to; `Footprint::ReadOnly` lets
its calls run beside others (a tool whose annotations say it only reads
is read-only unless `footprint` says otherwise). Blocking work goes in `blocking`, on a thread
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
                footprint: Some(Footprint::ReadOnly),
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
and the `args`. The call ends when the plugin inserts its `Outcome`,
such as `Outcome::from(ToolResult::success("Done.".into()))`, with
`insert_if_new`: the first answer counts. A tool
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

A tool call is an effect with a `ToolRequest` (the model's `ToolCall`),
its tool's `Footprint` and the `CallHandler` that will run it, all on the
call's entity. A Prepare system edits them as the call is admitted: the
arguments, the footprint, or the handler, wrapped with
`ErasedHandler::layered` so the call runs through it (to give it a secret
the model never sees, say). A Gate system sees each call when it is about
to run: a call that conflicts with an earlier one of the same reply
(`Footprint::waits_for`) is `Waiting` until that one settles, so a gate
judges it with the earlier calls' effects done. It refuses it with
`Outcome::refused(why)`, which the model reads as the tool's refusal, or
answers it with any `Outcome` (a cached result), or holds it. A Review
system reads or replaces each result before it is committed, as
[above](#how-a-plugin-steps-in-effects-and-stages). The reply's results go
back in the order the model asked for them, refused ones included. What a
tool published beside its text, such as rig-tools' `FilesChanged` or
`ExitStatus`, is the call's `Published` component.

# Asking the user

A question for the user is an `Ask` entity: a title, a body, the choices
offered and whether a line of text answers it too. It is an effect like
any, recorded with its answer, and any front answers it by inserting its
`Outcome` (`Answer::Choice(id).outcome()` or `Answer::Text(..)`); the first
answer counts and closes it in every other front. A plugin that waits on
the answer spawns the `Ask` as a hold on what it asks about, so the call
waits for it, and an interrupt, which despawns its turn's holds, takes the
question away everywhere. A front that shows questions spawns an
`Answerer` (for one agent, or `Answerer::default()` for all); a plugin
asks only when `Answerers::can_answer` for the agent, and applies its own
policy otherwise, such as refusing. While a question that takes text is
open, the user's next line to its agent answers it instead of reaching the
model. Tests answer with `rig_harness_test_support::AnswerAsks`.

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
    calls: Query<(Entity, &ToolRequest, &ForAgent), (Undecided, Without<Approved>)>,
    answerers: Answerers,
    mut commands: Commands,
) {
    for (call, request, &ForAgent(agent)) in &calls {
        let command = request.call.function.arguments["command"].as_str().unwrap_or_default();
        if !command.contains("git commit") {
            continue;
        }
        if !answerers.can_answer(Some(agent)) {
            commands.entity(call).insert_if_new(Outcome::refused("nobody can approve a commit"));
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
    mut commands: Commands,
) {
    for (ask, outcome, &HoldOf(call)) in &asks {
        if Answer::of(outcome).is_some_and(|answer| answer.is("yes")) {
            commands.entity(call).insert(Approved);
        } else {
            commands.entity(call).insert_if_new(Outcome::refused("the user said no"));
        }
        commands.entity(ask).remove::<HoldOf>();
    }
}
```

# A slash command

A command is a one-shot system, `app.add_command(name, help, system)`,
taking `In<CommandArgs>`: the `agent` it was typed for and the `args`
after its name. An error notice about the agent while the command runs,
from it or from what it triggers, refuses it: the line goes back in the
input with the error, and Enter then sends it to the model as it is (a
`literal` `Deliver`).

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
    // needs no answer: it goes with the next call and starts no turn.
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
way). A `Gate` system may take it over by answering it, such as with
`Outcome::custom(..)`, and then it never reaches the agent. At `Start` the
kernel accepts what is left: a user's text that starts with `/` runs as a
command (`incoming.command()` is its line, unless it is `literal`);
anything else goes to the agent, which starts a turn with it or keeps it
for its running turn as its `mode` says. The messages an agent accepted
and has not sent yet are its inbox,
`Query<(&Incoming, &ForAgent), (With<Started>, Without<Outcome>)>`, and
`On<Add<Started>>` sees each message as it is accepted.

```rust,no_run
use rig_harness::prelude::*;

#[derive(Default)]
pub struct ShortcutsPlugin;

impl Plugin for ShortcutsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            AgentPass,
            (expand.in_set(Stage::Prepare), keep_asides.in_set(Stage::Gate)),
        );
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
    typed: Query<(Entity, &Incoming, &ForAgent), Undecided>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (message, typed, &ForAgent(agent)) in &typed {
        if typed.origin.kind == OriginKind::User && typed.text.starts_with('#') {
            commands.entity(message).insert_if_new(Outcome::custom("aside"));
            notices.write(Notice::info(agent, "Noted; not sent."));
        }
    }
}
```

# Model requests

A model call is an effect entity with a `ModelRequest`: the rig-core
`CompletionRequest` it sends (system prompt as its first message, the
messages, the tools offered, the options) and the `connection` it goes
to, else its agent's model; `Models::connect` makes one. A turn's own call
is a `TurnCall`, whose request the kernel builds from its agent when it
is admitted; a call built from an agent outside its turns, such as a
preview, is `BuildFrom(agent)`. A plugin changes a request with a system
in the `Prepare` stage, which sees each call once (`Added<Admitted>`);
Start then fits it to the model it goes to (an option that model cannot
take is left out, plugin edits stay) and makes the tool rules in the
prompt follow the tools it carries. Routing a call to another model is a
Prepare edit of its `connection`. A tool left out is not offered but still
runs if called: a `Gate` system refuses a call (`Outcome::refused`), and
`ToolAccess` on the agent is the hard limit. Changing the system prompt or
the tools misses the provider's prompt cache.

A failed call carries the kernel's `RetryVerdict` (`Retry`, `GaveUp`,
`Overflow`, `Final`) in `Review`. A plugin that takes the failure over
spawns a new call that `Supersedes` the failed one (the old one is
recorded as superseded and never reaches the conversation), or holds it
(`HoldOf`) while it works, as compaction (the `rig-compaction` crate)
does to summarize first; otherwise the kernel retries after a `Backoff`
or proposes the turn's end ([How a turn ends](#how-a-turn-ends)). A
call refused before it was sent fails the turn and keeps the user's
message. Before any of it, what the user typed can be
changed as it comes in ([What the user types](#what-the-user-types)).
A part of the system prompt that does not change from request to request
is a `PromptSection` entity, `PromptSection::new(order, tag, text)`, in
every agent's prompt, or with a `SectionOf(agent)` in that agent's
alone, despawned with it.

```rust,no_run
use rig_harness::prelude::*;

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
                (add_the_time, plan_only).in_set(Stage::Prepare),
                fall_back.in_set(Stage::Review),
            ),
        );
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
        let retried = request.connection.is_some();
        if !matches!(verdict.0, Verdict::GaveUp(_)) || retried {
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

# How a turn ends

A turn is an entity `TurnOf` its agent, with a `Turn`: its `index` among
the agent's turns and its `first_message` in the conversation. A turn's
end is a proposal: when the model answers without calling tools, or the
turn fails, the kernel proposes a `TurnOutcome` on the turn in `Commit`
(`Answered(message)`, `Failed(reason)`), and a `Review` system sees it in
the next pass (`Added<TurnOutcome>`). It may replace it (`insert`), hold
it (spawn an entity `HoldOf(turn)`, despawn it when done), or carry the
turn on: a `Queue` or `Steer` message the agent accepts takes an
`Answered` proposal away, one on its way keeps the turn from ending
meanwhile, and a new `(TurnCall, CallOf(turn), ForAgent(agent))` takes
any away once it is admitted; a proposal taken away takes its holds with
it (they are despawned), and the next proposal is reviewed afresh. A
plugin may insert `TurnOutcome::Failed(reason)` or `Stopped(Some(reason))`
at any time, such as a budget: the turn's calls that have not started
never start, the others are answered as stopped, and the turn ends. Once
nothing holds the outcome and every call of the turn settled, the turn
ends: the agent is idle, and the turn entity is `Settled`, `ForAgent` its
agent, until the next frame. `Interrupt` stops a turn at once; it cannot
be held.

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
plugin (or anyone) answers with an `Outcome` whenever it is done. Other
plugins' Gate systems can refuse it, Review systems replace its outcome,
and a replay answers it from the effect log; `CallOf(turn)` keeps a turn
open until it settles. Work off the main thread is a `Running` task on the
effect's entity, `Running::answering(pool, wake, future)`, whose output
becomes its outcome.

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
fn report(deploys: Query<(Entity, &EffectRequest), Added<Started>>, mut commands: Commands) {
    for (deploy, request) in &deploys {
        if request.kind.name() == "deploy" {
            commands.entity(deploy).insert_if_new(Outcome::custom("deployed"));
        }
    }
}

fn build(app: &mut App) {
    app.add_command("deploy", "Deploy a target", deploy)
        .add_systems(AgentPass, report.in_set(Stage::Gate));
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
change: `commit.truncate(agent, &mut conversation, len)` cuts it back,
such as to rewind it, and `commit.retire(agent)` despawns an agent for
good, so a restart brings back neither it nor the agents it spawned.
`rig_ecs::journal::read_session(store)` reads a session's logs, another
session's too, without restoring or writing anything: each agent's whole
conversation, summary and saved components. `fork_session(from, to,
before)` copies a session as of a message, `before` being its agent's id
and `conversation.seq(index)`.

# Time

Time is Bevy's `Time`. `.run_if(on_real_timer(Duration))` runs a system
once per interval, and a one-off wait is a delayed command,
`commands.delayed().duration(Duration)`. The loop sleeps when nothing
happens and wakes on its own for a delayed command; for an interval, an
entity with a `KeepAwake(Duration)` keeps it running at that pace while it
lives. None needs a thread. Bevy's clock stops while the machine sleeps; a
time of day is a `WakeAt(SystemTime)` on an entity, which gets `Due` in
`First` once that time passed (the loop wakes for it), and is armed again
by inserting a new `WakeAt`.

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
`FileWatch` gets `FileChanges` messages about the files changed under its
path. `TokioRuntime` is rig's own tokio runtime, for a library that needs
tokio. A plugin that should also run in an app without the harness's
plugins inits the ones it reads (`app.init_resource::<ProjectRoot>()`).

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
```

# Sessions and exiting

A plugin that ends the run triggers `Exit(reason)`: `ExitReason::Quit`,
`Reload` (the launcher restarts the session on its staged build),
`Switch(Some(id))` or `Switch(None)` (the launcher runs another session,
or a new one). The harness writes the exit code and tells the launcher
what to run; any exit's reason is the `ExitReason` resource from then on,
for code on the way out in Bevy's `OnAppExitSystems` (a background job
kept across a reload, stopped on a quit). `ForkSession { before }` copies
this session into a new one (`SessionForked` names it), and
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
take them like any other.

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
    calls: Query<Entity, (With<TurnCall>, Undecided)>,
    spent: Option<Res<SessionSpending>>,
    mut commands: Commands,
) {
    if spent.is_none_or(|spent| spent.0.total.cost < CAP) {
        return;
    }
    for call in &calls {
        let why = format!("The session spent its ${CAP:.2} budget.");
        commands.entity(call).insert_if_new(Outcome::refused(why));
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
without taking the tool over.

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

# A front

A front shows the agents to someone and takes what they type: the
terminal view (`rig-tui`), `--print` (`rig-print`), a window, a web
server, a chat bot. It offers itself in `build` with `app.add_front(..)`,
saying which runs it takes (`on_terminal()`, `headless()`, or
`on_flag(name, help)` for a run given its option), and one front takes
each run: the one whose option was given, else the first offered for the
kind of run. It asks `app.runs_front(name)` in `finish` and sets itself
up only then. It sends what the user types as `Deliver` from
`Origin::user()`, shows `Notice`s and each agent's `Conversation` and
`Partial` reply, and, if it can show questions, spawns an `Answerer` and
answers `Ask`s. Several fronts may run side by side (a web server beside
the terminal view); only the one that takes the run owns stdin and stdout.

```rust,no_run
use rig_harness::prelude::*;

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
        if let Committed::Message { message, .. } = change {
            println!("{}", serde_json::to_string(message).unwrap_or_default());
        }
    }
}

/// Answers every question with its first choice, as `--yes` would.
fn answer_yes(asks: Query<(Entity, &Ask), Without<Outcome>>, mut commands: Commands) {
    for (ask, question) in &asks {
        if let Some(first) = question.choices.first() {
            let answer = Answer::Choice(first.id.clone());
            commands.entity(ask).insert_if_new(answer.outcome());
        }
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
`AppStatus` or a `PickRequest` (the short form of a
[picker](#keys-the-input-and-pickers-in-the-terminal-view)) registers it
with `app.init_resource::<AppStatus>()` or
`app.add_message::<PickRequest>()`, since the view may not be added. The
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

# Keys, the input and pickers in the terminal view

The terminal view (`rig_tui`) takes part in the input as data a plugin
reads and changes, as it does for panels:

- **Keys.** Every key the view acts on is a `KeyBinding` entity, its own
  editing keys included, in the input's `KeyScope`. A plugin binds a key
  or a sequence (`ctrl+x ctrl+e`) with `app.add_key_binding(binding,
  observer)`: the observer hears `KeyPressed` on the binding (and, for one
  `with_releases()`, the key let go), and `run_command("compact")` is an
  observer that runs a slash command. As with commands, the first binding
  of a key stays: a binding that another of an overlapping scope takes,
  starts, or is started by, is refused with a warning, and `Disabled` on a
  binding frees its keys, in `Plugin::finish`. `Keys` parse from text
  (`"alt+m".parse()`), for keys from the user's settings.
- **A panel's keys.** `KeyFocus(Some(panel))` gives a panel the keys: its
  bindings in `KeyScope::Panel(panel)` and the global ones apply, not the
  input's, and Esc it does not bind gives the keys back.
- **The input.** `TuiInput` is the draft: `text()`, `cursor()`, `set`,
  `insert`, `take`, `set_cursor`; `is_changed()` says it was edited. Enter
  sends it to the agent shown (`Focused`).
- **The terminal.** `TerminalEvent` messages are what the terminal
  reported, keys and pastes; a plugin may read them, and write some. A
  `RunInTerminal(command)` entity runs a program with the terminal (the
  view steps aside and draws again after), and gets its `TerminalExit`.
- **Pickers and questions.** A `Picker` entity is a filterable list the
  view shows over the transcript, one at a time in the order they came,
  with every agent's open [`Ask`](#asking-the-user); the view answers those
  too. Inserting a new `Picker` replaces what it shows; despawning it
  closes it. When the user picks, the view runs the item's command, if it
  has one, triggers `Picked` on the entity (`Chosen::Item(index)`, or
  `Chosen::Text` for a `free_text` picker; `None` when closed), and
  despawns it.

These are there in any run, and inert without the terminal view's run.
This plugin writes the draft in `$EDITOR` on Ctrl+E, which the view binds
to the line's end, and offers a few prompts on Ctrl+T:

```rust,no_run
use std::path::PathBuf;
use std::process::Command;

use rig_harness::prelude::*;
use rig_tui::{
    AppKeyBindingsExt, Chosen, KeyBinding, KeyPress, KeyPressed, Keys, Picked, Picker,
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
        app.add_systems(Update, take_draft);
    }

    /// Once every plugin was built, the view's Ctrl+E is there to free.
    fn finish(&self, app: &mut App) {
        let ctrl_e = Keys::from(KeyPress::ctrl('e'));
        let world = app.world_mut();
        let mut bindings = world.query::<(Entity, &KeyBinding)>();
        let taken: Vec<Entity> = bindings
            .iter(world)
            .filter(|(_, binding)| binding.keys == ctrl_e)
            .map(|(entity, _)| entity)
            .collect();
        for binding in taken {
            world.entity_mut(binding).insert(Disabled);
        }
        app.add_key_binding(KeyBinding::new(ctrl_e, "Write the draft in $EDITOR"), edit)
            .add_key_binding(KeyBinding::new(KeyPress::ctrl('t'), "Start from a prompt"), prompts);
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
