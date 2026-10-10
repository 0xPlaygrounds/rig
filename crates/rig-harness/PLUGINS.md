Writing a plugin for the rig agent: a cookbook of short, copy-ready
examples, one per extension point, and the names they use. Every name here
comes from `use rig_harness::prelude::*;`, so a typical plugin needs no
other import and no reading of rig's source.

- [Making one](#making-one): `rig plugin new`, plugins.toml, `/reload`
- [Names](#names): what the common types hold
- [A slash command](#a-slash-command), [a tool](#a-tool) and how its calls look
- [Reading what agents do and say](#reading-what-agents-do-and-say)
- [Turns, state and time](#turns-state-and-time): `TurnEnded`, saved
  components, timers
- [A terminal panel](#a-terminal-panel), [a window](#a-window)

A plugin is a Bevy plugin: a type implementing `Plugin + Default` in a
crate that depends on `rig-harness`. The agent is a Bevy app; its built-in
tools, commands, subagents and terminal view are plugins of the same kind,
listed in `RIG_HOME/plugins.toml` beside any other. A plugin never edits
rig-harness, rig-ecs or rig-tools: it builds on what they export.

# Making one

```sh
rig plugin new hello    # RIG_HOME/plugins/hello, listed in plugins.toml
rig plugin add viz::VizPlugin --path ~/viz   # an entry for an existing crate
rig plugin remove viz::VizPlugin             # take an entry out; its crate stays
rig plugin check        # plugins.toml and the crates it names, without a build
rig plugin check --build  # and build the agent with them, as /reload would
```

`rig plugin new` writes a crate with one plugin, `hello::HelloPlugin`,
whose `src/lib.rs` is a working, commented example: `/hello` counts the
words of the agent's last answer. It adds this entry:

```toml
[[plugin]]
crate = "hello"           # the package name
path = "plugins/hello"    # or git = "…" (branch, rev), or version = "…"
plugin = "hello::HelloPlugin"
# bevy_features = []      # Bevy features it needs, such as a window's
```

Plugin crates live in `RIG_HOME/plugins/<name>`, each its own workspace,
so they never touch the rig repository or a project. A plugin crate
depends on `rig-harness` by version. When the agent is built from a rig
checkout (`RIG_SOURCE`), the generated project's `[patch.crates-io]`
points the rig crates (`rig`, `rig-core`, `rig-ecs`, `rig-tools`,
`rig-harness`) at the checkout, so every plugin uses the agent's own
crates; the crate made by `rig plugin new` has the same table so it also
checks on its own, and the build refuses a second copy of a rig crate or
another Bevy than the agent's. Bevy's derives (`Component`, `Resource`,
`Message`, `SystemSet`) expand to the Bevy crate the plugin's own
`Cargo.toml` names, so the crate depends on `bevy_ecs` (or `bevy`) at the
agent's version; `rig plugin new` adds it.

`/reload` in the agent (or `rig build`) rebuilds and restarts in the same
session; typed during a turn, it waits until no turn runs (`/reload
cancel` drops it). The agent's model can ask for the same with the
`reload` tool, so it applies its own plugin changes once its turn ends.
A build that fails leaves the running one, and one that crashes at
startup is rolled back.

What a plugin uses comes from `rig_harness::prelude::*`: Bevy's app and
ECS preludes, the agent runtime's components, events and registries
(`rig_harness::rig_ecs`), `blocking`, `Duration`, and the terminal view's
panels and tool renderers. ratatui's widgets are
`rig_harness::tui::ratatui`, and rig-core (such as its `message::Message`)
is `rig_harness::rig_core`. For anything this guide does not show, ask the
running agent's `inspect` tool before reading rig's source:
`world.list_components` and `registry.schema` name the types and their
fields, and `world.query` with `ProvidedBy` shows what each plugin added. A
plugin's own Bevy Remote method is sent by `inspect` once its system's
entity has `ReadOnlyMethod` (`rig_harness::plugins::inspect`).

# Names

- `CommandArgs { agent: Entity, args: String }`: what a slash command's
  system receives, `In<CommandArgs>`; `args` is the text after the name,
  trimmed.
- `Notice::info(agent, text)`, `Notice::error(agent, text)`: a line for the
  user, written with `MessageWriter<Notice>`. `write` returns an id, so
  end it with `;` in a `match` arm.
- `Conversation` (on each agent): `messages()`, oldest first, of rig-core's
  `Message`. `final_answer(&message)` is the text of a final answer, `None`
  for any other message.
- `Activity` (on each agent): `status: Status` (`Idle`, `Thinking`,
  `RunningTools`, `Busy(name)` for a plugin's call such as `compacting`,
  `Retrying { attempt, seconds }`; it
  implements `Display`), `tools: Vec<ToolActivity>` (`name`, `queued`) and
  `preview: Option<Preview>` (`kind: PreviewKind` of `Text`, `Reasoning` or
  `LastReply`, and `text`, the last 2000 characters); `is_busy()`.
- `MessageFeed` (a resource): `iter()` of the last 64 deliveries, oldest
  first, each a `FedMessage { to: Entity, from: Option<Entity>, origin:
  Origin, text: String }`.
- `Agent` marks an agent, `AgentId(String)` names it, `SpawnedBy(Entity)`
  is on a subagent and `Spawned` lists an agent's subagents. A system
  taking `agents: PrimaryQuery` finds the user's agent with
  `primary(&agents)`; `Focused` marks the one the terminal shows.
- `Deliver { entity, text, origin, mode, attachments }`: triggered, puts a
  message in an agent's conversation (see [a slash command](#a-slash-command)).
- `TurnEnded { entity, outcome }`: an agent's turn ended;
  `outcome` is a `TurnOutcome`: `Answered(Message)`, `Failed(String)` or
  `Stopped`.
- `RunMode::is_headless`: a `--print` run, without the terminal view.

# A tool

Any `PortableTool` (or `rig_core::tool::Tool`). Its rules go into the
system prompt of every agent it is offered to; `Footprint::ReadOnly` lets
its calls run beside others. Blocking work goes in `blocking`, on a thread
of its own. `args_schema` derives its parameters from its arguments type,
whose field doc comments describe them; `deny_unknown_fields` on every
struct refuses a misspelled argument at any depth.

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

A tool answered later, by the plugin rather than a future, is an open
tool: `app.add_open_tool(name, description, options, observer)` derives
its parameters from the arguments type the observer takes, parses each
call's arguments into it, refusing those that do not fit, and triggers
`ToolCalled<Args>` on the tool's entity with the call (`call`, and the
model's `run.call`), the calling agent (`agent`, `caller`), the call's
`effect` and the parsed `args`. The call ends when a `ToolOutput` is
inserted on it, such as `ToolOutput(ToolResult::success("Done.".into()))`. How a tool's calls look in the terminal is
`add_tool_renderer`:

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

A one-shot system that receives the agent it was typed for and the text
after the name:

```rust,no_run
use rig_harness::prelude::*;

#[derive(Default)]
pub struct RemindPlugin;

impl Plugin for RemindPlugin {
    fn build(&self, app: &mut App) {
        app.add_command("remind", "Remind the agent of something after its turn", remind);
    }
}

fn remind(In(args): In<CommandArgs>, mut commands: Commands, mut notices: MessageWriter<Notice>) {
    if args.args.is_empty() {
        notices.write(Notice::error(args.agent, "/remind needs a text"));
        return;
    }
    // A message in the agent's conversation: `Steer` goes with the running
    // turn's next model call, `Queue` once that turn would end, together
    // with everything else queued; an idle agent starts a turn. A `Note`
    // needs no answer: it goes with the next call and starts no turn.
    commands.trigger(Deliver {
        entity: args.agent,
        text: format!("Reminder: {}", args.args),
        origin: Origin {
            kind: OriginKind::Plugin("remind".to_owned()),
            ..Origin::default()
        },
        mode: DeliveryMode::Queue,
        attachments: Vec::new(),
    });
}
```

An error notice about the agent while the command runs refuses it: the
line goes back in the input with the error, and Enter sends it to the
model as it is. A command that is one event is added as that event,
`app.add_command_event::<Compact>("compact", "help")`: the arguments are
parsed into its reflected fields (its `Entity` is the agent, a unit enum
takes a word naming a variant, a `String` the rest of the line), and
arguments it cannot take are refused the same way.

# Reading what agents do and say

An ordinary system reads the agents' components and the `MessageFeed`;
this one is a slash command, and a panel or an observer reads them the
same way.

```rust,no_run
use rig_harness::prelude::*;

#[derive(Default)]
pub struct RecentPlugin;

impl Plugin for RecentPlugin {
    fn build(&self, app: &mut App) {
        app.add_command("recent", "Each agent's status and the last messages", recent);
    }
}

fn recent(
    In(args): In<CommandArgs>,
    agents: Query<(&AgentId, &Activity, &Conversation)>,
    feed: Res<MessageFeed>,
    mut notices: MessageWriter<Notice>,
) {
    let mut lines = Vec::new();
    for (id, activity, conversation) in &agents {
        let answer = conversation.messages().iter().rev().find_map(final_answer);
        let words = answer.map_or(0, |answer| answer.split_whitespace().count());
        lines.push(format!("{}: {}, last answer {words} words", id.0, activity.status));
    }
    for message in feed.iter().rev().take(3) {
        let to = agents.get(message.to).map_or("?", |(id, ..)| id.0.as_str());
        lines.push(format!("to {to}: {}", message.text));
    }
    notices.write(Notice::info(args.agent, lines.join("\n")));
}
```

# Turns, state and time

- `TurnEnded` is triggered on an agent when its turn ends.
- An agent component that derives `Reflect` and says
  `#[reflect(Component, Saved)]` is kept with the session: each change is
  logged by reflection, and a restart, `/reload` or `/resume` brings the
  newest value back. `Restored` is triggered on each agent after a restart,
  where a plugin re-arms what it owes. The core saves `ModelChoice`,
  `Effort`, `SystemPrompt`, `ToolAccess` and `LastUsage`, and the usage
  plugin `Spending`, the same way, so a plugin that changes them has
  nothing to log. A generic type is saved only once registered
  (`app.register_type::<T>()`).
- `On<Add<CallOf>>` sees every model and tool call of a turn;
  `On<Add<ToolCallRun>>` the tool calls. `MessageReader<Committed>` sees
  every change of every conversation, as the session log records it.
- A `PromptSection` entity adds to every agent's system prompt.
- Time is Bevy's `Time`. `.run_if(on_real_timer(Duration))` runs a system
  once per interval; the loop sleeps when idle, so an entity with a
  `KeepAwake(Duration)` keeps it running at that pace while the timer
  matters. A one-off wait is a delayed command,
  `commands.delayed().duration(Duration)`; the loop wakes for it on its own.
  None needs a thread.

```rust,no_run
use std::collections::HashMap;

use rig_harness::prelude::*;

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
            .add_observer(count);
    }
}

fn count(
    call: On<Add<ToolCallRun>>,
    calls: Query<(&ToolCallRun, &CallOf)>,
    turns: Query<&TurnOf>,
    mut counts: Query<&mut ToolCounts>,
) {
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

# A terminal panel

A `TuiPanel` entity takes a side of the transcript (`Top`, `Bottom`,
`Left`, `Right`, with a ratatui `Constraint`) or a box over the screen
(`Over`). The plugin's own system draws it into its `PanelCanvas` in
`TuiSystems::Draw`, which runs only for a frame that is drawn. Frames are
drawn when agents or panels change; a plugin whose own state changed
writes a `RequestRedraw`. What agents do is their `Activity` component
(status, running tools, streamed preview), the `MessageFeed` resource of
delivered messages, and the tree through `SpawnedBy` and `Spawned`.
`TuiScreen` is the terminal's size and `Focused` marks the agent shown.

```rust,no_run
use rig_harness::prelude::*;
use rig_harness::tui::ratatui::widgets::{Block, Paragraph};

#[derive(Default)]
pub struct AgentsPanelPlugin;

#[derive(Component)]
struct AgentsPanel;

/// The spinner's step.
#[derive(Resource, Default)]
struct Step(usize);

/// How often the spinner turns.
const TURN: Duration = Duration::from_millis(200);

impl Plugin for AgentsPanelPlugin {
    fn build(&self, app: &mut App) {
        // A print run has no terminal view.
        if app.world().get_resource::<RunMode>().is_some_and(RunMode::is_headless) {
            return;
        }
        app.init_resource::<Step>()
            .add_systems(Startup, spawn)
            .add_systems(Update, (keep_awake, spin.run_if(busy.and_then(on_real_timer(TURN)))))
            .add_systems(PostUpdate, draw.in_set(TuiSystems::Draw));
    }
}

fn spawn(mut commands: Commands) {
    commands.spawn((AgentsPanel, TuiPanel::new(Placement::Right(Constraint::Length(32)))));
}

fn busy(agents: Query<&Activity>) -> bool {
    agents.iter().any(Activity::is_busy)
}

/// While an agent works the panel keeps the loop awake, so the spinner
/// turns; once all are idle the loop sleeps again.
fn keep_awake(
    agents: Query<&Activity>,
    panels: Query<(Entity, Has<KeepAwake>), With<AgentsPanel>>,
    mut commands: Commands,
) {
    let busy = busy(agents);
    for (panel, awake) in &panels {
        if busy && !awake {
            commands.entity(panel).insert(KeepAwake(TURN));
        } else if !busy && awake {
            commands.entity(panel).remove::<KeepAwake>();
        }
    }
}

fn spin(mut step: ResMut<Step>, mut redraw: MessageWriter<RequestRedraw>) {
    step.0 = step.0.wrapping_add(1);
    redraw.write(RequestRedraw);
}

fn draw(
    mut panels: Query<&mut PanelCanvas, With<AgentsPanel>>,
    agents: Query<(&AgentId, &Activity, Has<SpawnedBy>)>,
    step: Res<Step>,
) {
    let spinner = ['|', '/', '-', '\\'].get(step.0 % 4).copied().unwrap_or(' ');
    let lines: Vec<String> = agents
        .iter()
        .map(|(id, activity, sub)| {
            let mark = if activity.is_busy() { spinner } else { ' ' };
            let indent = if sub { "  " } else { "" };
            format!("{indent}{mark} {} {}", id.0, activity.status)
        })
        .collect();
    for mut canvas in &mut panels {
        canvas.render(Paragraph::new(lines.join("\n")).block(Block::bordered().title("agents")));
    }
}
```

# A window

`rig_harness::windowed(DefaultPlugins)` is Bevy's `DefaultPlugins`
without the log, task pools, signal handler and loop the agent already
has; its winit plugin then runs the app's loop. In `finish`, where
winit's event loop exists, the plugin points the agent's `Wake` at it, so
agent activity and terminal input wake the window instead of a poll. The
crate depends on `bevy` at exactly the agent's version, with the features
it draws with, such as
`bevy = { version = "=0.20.0", default-features = false, features = ["ui"] }`
for Bevy UI; the build says so when the version differs.

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
