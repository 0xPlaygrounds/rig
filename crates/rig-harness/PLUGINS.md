Writing a plugin for the rig agent: one short, copy-ready example per
extension point. Every name they use comes from
`use rig_harness::prelude::*;`.

- [Making one](#making-one) and [finding names](#finding-names)
- [A tool](#a-tool) and how its calls look
- [A slash command](#a-slash-command)
- [Saved state: counting every tool call](#saved-state-counting-every-tool-call)
- [Turn hooks](#turn-hooks), [timers](#timers)
- [What agents do and say, in a terminal panel](#what-agents-do-and-say-in-a-terminal-panel)
- [A window](#a-window)

A plugin is a Bevy plugin: a type implementing `Plugin + Default` in a
crate that depends on `rig-harness`. The agent is a Bevy app, and its
tools, commands, subagents, compaction, sessions and terminal view are
plugins of the same kind, listed in `RIG_HOME/plugins.toml` beside any
other and built only on the same public API. A plugin never edits
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
depends on `rig-harness` by version, and on `bevy_ecs` and `bevy_reflect`
(or `bevy`) at the agent's exact version, because Bevy's derives expand to
the Bevy crate the plugin's own `Cargo.toml` names. When the agent is built from a rig
checkout (`RIG_SOURCE`), a `[patch.crates-io]` table points the rig crates
at it, so every plugin uses the agent's own crates.

`/reload` in the agent, or the agent's `reload` tool, rebuilds and
restarts in the same session once no turn runs. A build that fails leaves
the running one, and one that crashes at startup is rolled back.

# Finding names

The prelude holds Bevy's app, ECS, reflection and time preludes, the agent
runtime's components, events and registries (`rig_harness::rig_ecs`), and
the terminal view's panels and tool renderers. ratatui is
`rig_harness::tui::ratatui`, rig-core is `rig_harness::rig_core`, and each
default plugin's types are in `rig_harness::plugins::<name>`.

Every one of these types is reflected, so the running agent can list
them: ask its `inspect` tool before reading rig's source.
`world.list_components` and `world.list_resources` name what exists,
`registry.schema` gives a type's fields (with `type_limit: {with:
["Saved"]}`, the state saved with the session), `world.query` with
`ProvidedBy` shows what each plugin added, and the `Diagnostics` resource
holds the warnings logged. A plugin's own Bevy Remote method is sent by
`inspect` once its system's entity has `ReadOnlyMethod`
(`rig_harness::plugins::inspect`).

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
it, such as `ToolOutput(ToolResult::success("Done.".into()))`. How the
terminal draws a tool's calls:

```rust,no_run
use rig_harness::prelude::*;

fn build(app: &mut App) {
    app.add_tool_renderer("word_count", |call| {
        let mut lines = vec![call.header("Count", call.argument("path").unwrap_or_default())];
        lines.extend(call.result_lines(RESULT_LINES));
        lines
    });
}
```

# A slash command

A command that is one event is added as that event: the agent fills its
`Entity` field and the text after the name its `String` field. An error
notice about the agent while the command runs refuses it: the line goes
back in the input with the error, and Enter then sends it to the model as
it is. A command that needs more is a one-shot system,
`app.add_command(name, help, system)`, taking `In<CommandArgs>` (`agent`,
`args`).

```rust,no_run
use rig_harness::prelude::*;

#[derive(Default)]
pub struct RemindPlugin;

impl Plugin for RemindPlugin {
    fn build(&self, app: &mut App) {
        app.add_command_event::<Remind>("remind", "Remind the agent of something after its turn")
            .add_observer(remind);
    }
}

/// `/remind <text>`.
#[derive(EntityEvent, Reflect)]
struct Remind {
    /// The agent it was typed for.
    entity: Entity,
    /// The text after `/remind`.
    text: String,
}

fn remind(remind: On<Remind>, mut commands: Commands, mut notices: MessageWriter<Notice>) {
    if remind.text.is_empty() {
        notices.write(Notice::error(remind.entity, "/remind needs a text"));
        return;
    }
    // A message in the agent's conversation: `Steer` goes with the running
    // turn's next model call, `Queue` once that turn would end, together
    // with everything else queued; an idle agent starts a turn. A `Note`
    // needs no answer: it goes with the next call and starts no turn.
    commands.trigger(Deliver {
        entity: remind.entity,
        text: format!("Reminder: {}", remind.text),
        origin: Origin {
            kind: OriginKind::Plugin("remind".to_owned()),
            ..Origin::default()
        },
        mode: DeliveryMode::Queue,
        attachments: Vec::new(),
    });
}
```

# Saved state: counting every tool call

An agent component that derives `Reflect` and says
`#[reflect(Component, Saved)]` is kept with the session: each change is
logged, and a restart, `/reload` or `/resume` brings the newest value
back, before `Restored` is triggered on the agent. `On<Add<CallOf>>` sees
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
with the messages it sends, which an observer may change. `ModelFailed` is
triggered on a turn whose model call failed for good; an observer that
takes the turn over sets `handled` and triggers `CallModel` once it is
ready. Observers of one event run in no set order. The compaction plugin
(`rig_harness::plugins::compaction`) is the full example: it summarizes on
`PrepareRequest` with a `ModelRequest` call of its own, and on a
`ModelFailed` overflow.

```rust,no_run
use rig_harness::prelude::*;
use rig_harness::rig_core::message::{Message, UserContent};

/// The model a turn falls back to when its own fails for good.
const FALLBACK: &str = "deepseek/deepseek-flash";

#[derive(Default)]
pub struct FallbackPlugin;

impl Plugin for FallbackPlugin {
    fn build(&self, app: &mut App) {
        app.add_observer(add_the_time).add_observer(fall_back);
    }
}

/// Every request tells the model how long the agent has run.
fn add_the_time(mut prepare: On<PrepareRequest>, time: Res<Time<Real>>) {
    let note = format!("(The agent has run for {} s.)", time.elapsed().as_secs());
    if let Some(Message::User { content }) = prepare.messages.last_mut() {
        content.push(UserContent::text(note));
    }
}

/// A model call that failed for good, other than on a conversation too
/// long, is sent again once, on the fallback model.
fn fall_back(mut failed: On<ModelFailed>, choices: Query<&ModelChoice>, mut commands: Commands) {
    let on_fallback = choices.get(failed.agent).is_ok_and(|choice| choice.0 == FALLBACK);
    if failed.handled || failed.report.is_context_overflow() || on_fallback {
        return;
    }
    failed.handled = true;
    commands.entity(failed.agent).insert(ModelChoice(FALLBACK.to_owned()));
    commands.trigger(CallModel { entity: failed.entity });
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

# What agents do and say, in a terminal panel

Each agent is an entity with an `Agent`, a `Name`, an `AgentId` and a
`Conversation`; a subagent is `SpawnedBy` its parent, which lists it in
`Spawned`. The activity plugin's `Activity` says what an agent does now
(status, running tools, streamed preview), and its `MessageFeed` resource
holds the latest deliveries. `MessageReader<Committed>` sees every change
of every conversation as the session log records it, and `On<TurnEnded>`
each turn's end.

A `TuiPanel` entity takes a side of the transcript (`Top`, `Bottom`,
`Left`, `Right`, with a ratatui `Constraint`) or a box over the screen
(`Over`). The plugin's own system draws it into its `PanelCanvas` in
`TuiSystems::Draw`, which runs only for a frame that is drawn. Frames are
drawn when agents or panels change; a plugin whose own state changed
writes a `RequestRedraw`. `TuiScreen` is the terminal's size and `Focused`
marks the agent shown.

```rust,no_run
use std::collections::HashMap;

use rig_harness::prelude::*;
use rig_harness::tui::ratatui::widgets::{Block, Paragraph};

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
use rig_harness::prelude::{Activity, RunMode, Wake};

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
