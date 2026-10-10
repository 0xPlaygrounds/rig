Writing a plugin for the rig agent: one short, copy-ready example per
extension point. Every name they use comes from
`use rig_harness::prelude::*;`, except a plugin crate's own types, such as
the terminal view's panels from `rig_tui`.

- [Making one](#making-one) and [finding names](#finding-names)
- [A tool](#a-tool) and how its calls look
- [A slash command](#a-slash-command)
- [Saved state: counting every tool call](#saved-state-counting-every-tool-call)
- [Turn hooks](#turn-hooks), [timers](#timers)
- [Turning off or replacing what another plugin added](#turning-off-or-replacing-what-another-plugin-added)
- [The status line](#the-status-line)
- [What agents do and say, in a terminal panel](#what-agents-do-and-say-in-a-terminal-panel)
- [A window](#a-window)

A plugin is a Bevy plugin: a type implementing `Plugin + Default` in a
crate that depends on `rig-harness`. The agent is a Bevy app, and its
tools, commands, subagents, compaction, sessions and terminal view are
plugins of the same kind, crates in the rig repository's `plugins/`
folder, listed in `RIG_HOME/plugins.toml` beside any other and built only
on the same public API. A plugin never edits
rig-harness, rig-ecs or rig-tools. One that needs another adds it unless
it is there, with Bevy's `app.is_plugin_added::<P>()`.

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

`/reload` in the agent, or the agent's `reload` tool, rebuilds and
restarts in the same session once no turn runs. A build that fails leaves
the running one, and one that crashes at startup is rolled back.

# Finding names

The prelude holds Bevy's app, ECS, reflection and time preludes, and the
agent runtime's components, events and registries (`rig_harness::rig_ecs`).
rig-core is `rig_harness::rig_core`, and each default plugin's types are
in its crate: the terminal view's panels and tool renderers in `rig_tui`
(ratatui is `rig_tui::ratatui`), what agents do in `rig_activity`, and so
on for each crate `plugins.toml` names.

Every one of these types is reflected, so the running agent can list
them with the `inspect` tool of the optional `rig-inspect` plugin (enable it
by uncommenting its entry in `plugins.toml`): ask it before reading rig's
source. `world.list_components` and `world.list_resources` name what exists,
`registry.schema` gives a type's fields (with `type_limit: {with:
["Saved"]}`, the state saved with the session), `world.query` with
`ProvidedBy` shows what each plugin added, and the `Diagnostics` resource
holds the warnings logged. A plugin's own Bevy Remote method is sent by
`inspect` once its system's entity has `rig_inspect::ReadOnlyMethod`.

# A tool

Any `PortableTool` (or `rig_core::tool::Tool`). Its rules go into the
system prompt of every agent it is offered to; `Footprint::ReadOnly` lets
its calls run beside others. Blocking work goes in `blocking`, on a thread
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
                footprint: Footprint::ReadOnly,
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
and the `args`. The call ends when the plugin inserts a `ToolOutput` on
it, such as `ToolOutput(ToolResult::success("Done.".into()))`. A tool
found while the agent runs, such as one an MCP server lists, is
registered from a system with
`commands.queue(move |world: &mut World| { world.spawn_tool(definition, handler, options); })`:
its `ToolDefinition`, and a rig-core `ErasedHandler` that answers its
calls; despawning the entity it returns removes it. How the
terminal view (`rig-tui`) draws a tool's calls:

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

# A slash command

A command is a one-shot system, `app.add_command(name, help, system)`,
taking `In<CommandArgs>`: the `agent` it was typed for and the `args`
after its name. An error notice about the agent while the command runs,
from it or from what it triggers, refuses it: the line goes back in the
input with the error, and Enter then sends it to the model as it is.

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

# Saved state: counting every tool call

An agent component that derives `Reflect` and says
`#[reflect(Component, Saved)]` is kept with the session: each change is
logged, and a restart, `/reload` or `/resume` brings the newest value
back, before `Restored` is triggered on the agent. A resource that says
`#[reflect(Resource, Saved)]` is kept the same way, for what belongs to
the whole session; the plugin inserts it in `build`, and the saved value
replaces it. `On<Add<CallOf>>` sees
every model and tool call of every turn as it starts; a tool call also
has a `ToolCallRun`. The calls made before the plugin was added are in
the conversation, which `Restored` lets it count once: an assistant
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

/// On the first restart with the plugin, counts the calls already in the
/// conversation; a saved count is never redone.
fn count_the_past(restored: On<Restored>, mut agents: Query<(&Conversation, &mut ToolCounts)>) {
    let Ok((conversation, mut counts)) = agents.get_mut(restored.entity) else {
        return;
    };
    if !counts.0.is_empty() {
        return;
    }
    for message in conversation.messages() {
        if let Message::Assistant(assistant) = message {
            for call in assistant.tool_calls() {
                *counts.0.entry(call.function.name.as_str().to_owned()).or_default() += 1;
            }
        }
    }
}

fn count(
    call: On<Add<CallOf>>,
    calls: Query<(&ToolCallRun, &CallOf)>,
    turns: Query<&TurnOf>,
    mut counts: Query<&mut ToolCounts>,
) {
    // A model call has no `ToolCallRun`.
    let Ok((run, CallOf(turn))) = calls.get(call.entity) else {
        return;
    };
    if let Ok(TurnOf(agent)) = turns.get(*turn)
        && let Ok(mut counts) = counts.get_mut(*agent)
    {
        let name = run.call.function.name.as_str().to_owned();
        *counts.0.entry(name).or_default() += 1;
    }
}
```

A generic type is saved only once registered (`app.register_type::<T>()`).

# Turn hooks

`PrepareRequest` is triggered on a turn right before its model request,
with everything it sends, which an observer may change: the system prompt
(`preamble`), the `messages`, the `tools` offered and the generation
`options`. The request is checked against the model afterwards. Changing
the preamble or the tools misses the provider's prompt cache, and a tool
left out is not offered but still runs if called: `ToolAccess` on the
agent is the hard limit. A `Connection` on a turn sends its requests to
that model instead of the agent's, and one on a plugin's `ModelRequest`
call that call; `Models::connect` makes one. `ModelFailed` is triggered on
a turn whose model call failed for good; an observer that takes the turn
over sets `handled` and triggers `CallModel` once it is ready. Observers
of one event run in no set order. The compaction plugin (the
`rig-compaction` crate) is the full example: it summarizes on
`PrepareRequest` with a `ModelRequest` call of its own, and on a
`ModelFailed` overflow. Before any of it, `Input` is triggered on an
agent with what the user typed, in any front: an observer may rewrite its
`text`, such as expanding a template, or set `handled` and deal with it
itself; a text that starts with `/` then runs as a command. A part of
the system prompt that does not change from request to request is a
`PromptSection` entity, `PromptSection::new(order, tag, text)`, in every
agent's prompt, or with a `SectionOf(agent)` in that agent's alone,
despawned with it.

```rust,no_run
use rig_harness::prelude::*;

/// The model a turn falls back to when its own fails for good.
const FALLBACK: &str = "deepseek/deepseek-flash";

/// On an agent that only plans.
#[derive(Component)]
pub struct PlanMode;

#[derive(Default)]
pub struct TurnHooksPlugin;

impl Plugin for TurnHooksPlugin {
    fn build(&self, app: &mut App) {
        app.add_observer(add_the_time)
            .add_observer(plan_only)
            .add_observer(fall_back);
    }
}

/// Every request tells the model how long the agent has run, at its end,
/// where the prompt cache is not disturbed.
fn add_the_time(mut prepare: On<PrepareRequest>, time: Res<Time<Real>>) {
    let note = format!("(The agent has run for {} s.)", time.elapsed().as_secs());
    if let Some(message::Message::User { content }) = prepare.messages.last_mut() {
        content.push(UserContent::text(note));
    }
}

/// An agent in plan mode is offered only the tools that read, and told so.
fn plan_only(mut prepare: On<PrepareRequest>, planning: Query<(), With<PlanMode>>) {
    if planning.contains(prepare.agent) {
        prepare.tools.retain(|tool| matches!(tool.name.as_str(), "read" | "search"));
        prepare.preamble.push_str("\n\nPlan only: change nothing yet.");
    }
}

/// A model call that failed for good, other than on a conversation too
/// long, is sent again once, on the fallback model, for this turn only.
fn fall_back(
    mut failed: On<ModelFailed>,
    routed: Query<(), With<Connection>>,
    models: Res<Models>,
    mut commands: Commands,
) {
    let turn = failed.entity;
    if failed.handled || failed.report.is_context_overflow() || routed.contains(turn) {
        return;
    }
    let Ok(fallback) = models.connect(FALLBACK) else {
        return;
    };
    failed.handled = true;
    commands.entity(turn).insert(fallback);
    commands.trigger(CallModel { entity: turn });
}
```

# Timers

Time is Bevy's `Time`. `.run_if(on_real_timer(Duration))` runs a system
once per interval, and a one-off wait is a delayed command,
`commands.delayed().duration(Duration)`. The loop sleeps when nothing
happens and wakes on its own for a delayed command; for an interval, an
entity with a `KeepAwake(Duration)` keeps it running at that pace while it
lives. None needs a thread.

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

# Turning off or replacing what another plugin added

Tools, slash commands and prompt sections are entities. Bevy's `Disabled`
on one turns it off: no query finds it, so agents are not offered the
tool, the command is unknown and the section is left out, and its name is
free for a replacement (a second tool or command of a taken name is
refused). Do it in `Plugin::finish`, which runs once every plugin's
`build` did, so what they added exists whatever their order. What a
plugin spawns in `finish` is not marked `ProvidedBy` it.

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
        // `SlashCommand`; a section by its `PromptSection`'s `tag`.
    }
}
```

# The status line

The row under the transcript shows the `StatusItems` of the agent shown
and of its running turn, which every agent and turn has, and the app's
`AppStatus`. A plugin shows an item there in a system in `StatusSystems`
that runs when what it shows changed: `items.show(item)` puts the item at
its place (its side and order) in place of the one there, and an empty
item clears the place. The default plugins' places, and how long each
item stays when the line is too narrow (`keep`: the lowest goes first,
`u8::MAX` never):

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
| left | 90 | the rebuild of `/reload` (`AppStatus`) | 16 | rig-harness |
| right | 10 | tokens in and out | 4 | rig-telemetry |
| right | 20 | cached input | 1 | rig-telemetry |
| right | 30 | cost | 3 | rig-telemetry |
| right | 40 | the context in use | 5 | rig-telemetry |

```rust,no_run
use rig_harness::prelude::*;

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
of every conversation as the session log records it, and `On<TurnEnded>`
each turn's end.

A `TuiPanel` entity takes a side of the transcript (`Top`, `Bottom`,
`Left`, `Right`, with a ratatui `Constraint`) or a box over the screen
(`Over`). The plugin's own system draws it into its `PanelCanvas` in
`TuiSystems::Draw`, which runs only for a frame that is drawn. Frames are
drawn when agents or panels change; a plugin whose own state changed
writes a `RequestRedraw`. `TuiScreen` is the terminal's size and `Focused`
marks the agent shown. These are `rig_tui`'s, so the crate depends on
`rig-tui` and `rig-activity` beside `rig-harness`.

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
        // A print run has no terminal view.
        if app.world().get_resource::<RunMode>().is_some_and(RunMode::is_headless) {
            return;
        }
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

# A window

`rig_harness::windowed(DefaultPlugins)` is Bevy's `DefaultPlugins`
without the log, task pools, clock, signal handler and loop the agent
already has; its winit plugin then runs the app's loop. In `finish`, where
winit's event loop exists, the plugin points the agent's `Wake` at it, so
agent activity and terminal input wake the window instead of a poll. The
crate depends on `bevy` at exactly the agent's version, with the features
it draws with, such as
`bevy = { version = "=0.20.0", default-features = false, features = ["ui"] }`
for Bevy UI. (This example is not built with the agent's docs, which have
no window.)

```rust,ignore
use std::time::Duration;

use bevy::prelude::*;
use bevy::window::ExitCondition;
use bevy::winit::{EventLoopProxyWrapper, UpdateMode, WinitSettings, WinitUserEvent};
use rig_activity::Activity;
use rig_harness::prelude::{RunMode, Wake};

#[derive(Default)]
pub struct DashboardPlugin;

impl Plugin for DashboardPlugin {
    fn build(&self, app: &mut App) {
        if app.world().get_resource::<RunMode>().is_some_and(RunMode::is_headless) {
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
