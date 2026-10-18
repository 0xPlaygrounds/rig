//! Plugin corpus: 34. Todo tracker (spec `34-todo-tracker.md`).
//!
//! The `todo` tool keeps a todo list per agent ([`Todos`], saved with the
//! session), shown as `todo 3/7` in the status line. An agent whose turn
//! ends with an answer while items are still open is nudged once per
//! user message to carry on. `/todo` shows the list and `/todo clear`
//! clears it. Each change of a list is published as a [`TodoList`] event,
//! from which the view half ([`TodoView`]) shows the shown agent's list in
//! a panel (Alt+T or `/todo-panel` toggles it).

use rig_harness::channel::{ChannelAppExt, ChannelEvent};
use rig_harness::prelude::*;
use rig_harness::view::AppViewCommandsExt;
use rig_harness::view::{Applied, Mirror};
use rig_harness::worker::{Publish, WorkerOutbox};
use rig_tui::ratatui::layout::Constraint;
use rig_tui::ratatui::widgets::{Block, Paragraph};
use rig_tui::{
    AppKeyBindingsExt, KeyBinding, KeyPress, PanelCanvas, Placement, RequestRedraw, Side,
    StatusItem, StatusItems, StatusSystems, Tone, TuiPanel, TuiSystems, run_command,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The tool's name.
pub const TODO_TOOL: &str = "todo";

const DESCRIPTION: &str = "Keep your todo list for the current work. `set` replaces the list \
    with `items`, `check` ticks off the item `id`, `add` appends an item of `text`. The result \
    is the list, with each item's id. For a task of 3 or more steps, first `set` the steps, \
    `check` each off as soon as it is done, and `add` steps you discover. Do not end your turn \
    while items are open unless you say why.";

/// After the subagents at work; goes before the other agents at work.
const STATUS: StatusItem = StatusItem::new("todo-tracker", Side::Left, 65, 9);

/// The panel's width.
const PANEL_WIDTH: u16 = 32;

/// Adds the `todo` tool and command, the nudge, the status item and the
/// lists' events.
#[derive(Default)]
pub struct TodoPlugin;

/// The plugin's stage system, in [`Stage::Review`]: it nudges an agent
/// whose turn would end with items open.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TodoSystems;

impl Plugin for TodoPlugin {
    fn build(&self, app: &mut App) {
        let options = ToolOptions {
            footprint: Some(Touches::Independent),
        };
        app.register_required_components::<Agent, Todos>()
            .register_required_components::<Agent, Nudged>()
            .add_channel_event::<TodoList>()
            .add_open_tool(TODO_TOOL, DESCRIPTION, options, on_todo)
            .add_command(
                "todo",
                "Show the todo list; `/todo clear` clears it",
                todo_command,
            )
            .configure_sets(AgentPass, TodoSystems.in_set(Stage::Review))
            .add_systems(AgentPass, nudge.in_set(TodoSystems))
            .add_observer(reset_nudge)
            .add_systems(PostUpdate, show_status.in_set(StatusSystems::Show))
            .add_systems(Last, publish.run_if(resource_exists::<WorkerOutbox>));
    }
}

/// The view half: the shown agent's list in a panel, from the
/// [`TodoList`] events; named in `plugins.toml` as the entry's `view`.
#[derive(Default)]
pub struct TodoView;

impl Plugin for TodoView {
    fn build(&self, app: &mut App) {
        app.init_resource::<PanelShown>()
            .add_message::<RequestRedraw>()
            .add_message::<Applied>()
            .add_view_command(
                "todo-panel",
                "Show or hide the todo list's panel",
                toggle_panel,
            )
            .add_systems(Startup, spawn_panel)
            .add_systems(Update, (mirror_lists, redraw).chain())
            .add_systems(PostUpdate, place_panel.in_set(TuiSystems::Prepare))
            .add_systems(PostUpdate, draw_panel.in_set(TuiSystems::Draw))
            .add_key_binding(
                KeyBinding::new(KeyPress::alt('t'), "Show or hide the todo list"),
                run_command("todo-panel"),
            );
    }
}

/// An agent's list after a change: the event [`TodoPlugin`] publishes and
/// [`TodoView`] shows.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct TodoList {
    /// The agent's id.
    pub agent_id: String,
    /// Its items, in order.
    pub items: Vec<Todo>,
}

impl ChannelEvent for TodoList {
    const KIND: &'static str = "x/todo/list";
}

/// An agent's todo list, in order. Saved with the session.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Saved, Default, Clone, Debug)]
pub struct Todos(pub Vec<Todo>);

/// One item of a [`Todos`] list.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Default, Clone, Debug)]
pub struct Todo {
    /// The id the model checks it off by.
    pub id: u32,
    /// What to do.
    pub text: String,
    /// Whether it is done.
    pub done: bool,
}

impl Todos {
    /// The items not done yet.
    pub fn open(&self) -> impl Iterator<Item = &Todo> {
        self.0.iter().filter(|todo| !todo.done)
    }

    /// `done/total`, such as `3/7`.
    pub fn tally(&self) -> String {
        let done = self.0.len() - self.open().count();
        format!("{done}/{}", self.0.len())
    }

    /// The list, one item a line.
    pub fn render(&self) -> String {
        if self.0.is_empty() {
            return "The todo list is empty.".to_owned();
        }
        let lines: Vec<String> = self.0.iter().map(Todo::line).collect();
        lines.join("\n")
    }

    /// The list after `args`, or why they do not apply.
    fn applied(&self, args: &TodoArgs) -> Result<Vec<Todo>, String> {
        let mut list = self.0.clone();
        match args.action {
            Action::Set => {
                list = (1..)
                    .zip(args.items.iter().cloned())
                    .map(Todo::new)
                    .collect();
            }
            Action::Add => {
                let text = args.text.clone().filter(|text| !text.trim().is_empty());
                let text = text.ok_or("`add` needs a `text`")?;
                let id = list.iter().map(|todo| todo.id).max().unwrap_or(0) + 1;
                list.push(Todo::new((id, text)));
            }
            Action::Check => {
                let id = args.id.ok_or("`check` needs an `id`")?;
                let todo = list.iter_mut().find(|todo| todo.id == id);
                todo.ok_or_else(|| format!("no item has id {id}"))?.done = true;
            }
        }
        Ok(list)
    }
}

impl Todo {
    fn new((id, text): (u32, String)) -> Self {
        Self {
            id,
            text: text.trim().to_owned(),
            done: false,
        }
    }

    /// Such as `[x] 1. Write the parser`.
    fn line(&self) -> String {
        let mark = if self.done { "x" } else { " " };
        format!("[{mark}] {}. {}", self.id, self.text)
    }
}

/// Whether the agent was nudged since the user's last message.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct Nudged(pub bool);

/// What a `todo` call does.
#[derive(Deserialize, JsonSchema, Clone, Copy, Debug)]
#[serde(rename_all = "snake_case")]
enum Action {
    Set,
    Check,
    Add,
}

/// The arguments of a `todo` call.
#[derive(Deserialize, JsonSchema, Debug)]
#[serde(deny_unknown_fields)]
struct TodoArgs {
    /// `set` replaces the list with `items`, `check` ticks off the item
    /// `id`, `add` appends an item of `text`.
    action: Action,
    /// For `set`: the new list's items, in order.
    #[serde(default)]
    items: Vec<String>,
    /// For `check`: the id of the item done.
    id: Option<u32>,
    /// For `add`: what to do.
    text: Option<String>,
}

/// The `todo` tool: changes the calling agent's list and answers with it.
fn on_todo(
    called: On<ToolCalled<TodoArgs>>,
    mut agents: Query<&mut Todos>,
    mut decisions: Query<&mut Decision, ToolCalls>,
) {
    let result = match agents.get_mut(called.agent) {
        Ok(mut todos) => todos.applied(&called.args).map(|list| {
            todos.set_if_neq(Todos(list));
            todos.render()
        }),
        Err(_) => Err("this agent keeps no todo list".to_owned()),
    };
    let output = match result {
        Ok(list) => ToolResult::success(list.into()),
        Err(why) => ToolResult::failed(ToolExecutionError::other(why)),
    };
    if let Ok(mut decision) = decisions.get_mut(called.call) {
        decision.decide(Decider::plugin("todo-tracker"), output);
    }
}

/// A turn the model would end with an answer while items are open carries
/// on, with one nudge per user message as its next step: a `Queue` message
/// delivered while the answer is proposed, in Review.
fn nudge(
    proposed: Query<(&TurnOf, &TurnOutcome), Added<TurnOutcome>>,
    mut agents: Query<(&Todos, &mut Nudged)>,
    mut commands: Commands,
) {
    for (&TurnOf(agent), outcome) in &proposed {
        if !matches!(outcome, TurnOutcome::Answered(_)) {
            continue;
        }
        let Ok((todos, mut nudged)) = agents.get_mut(agent) else {
            continue;
        };
        let open: Vec<String> = todos
            .open()
            .map(|todo| format!("{}. {}", todo.id, todo.text))
            .collect();
        if nudged.0 || open.is_empty() {
            continue;
        }
        nudged.0 = true;
        let text = format!(
            "You have {} open todo items: {}; continue, or explain why you stopped.",
            open.len(),
            open.join(", ")
        );
        let nudge = Deliver::new(agent, text, DeliveryMode::Queue);
        commands.trigger(nudge.with_author(Author::plugin(TODO_TOOL)));
    }
}

/// A user's message the agent accepted allows one more nudge.
fn reset_nudge(
    accepted: On<Add<Started>>,
    messages: Query<(&Incoming, &ForAgent)>,
    mut agents: Query<&mut Nudged>,
) {
    if let Ok((typed, &ForAgent(agent))) = messages.get(accepted.entity)
        && typed.author.kind == AuthorKind::User
        && let Ok(mut nudged) = agents.get_mut(agent)
    {
        nudged.0 = false;
    }
}

/// Whether the panel is shown while the agent shown has a list.
#[derive(Resource)]
struct PanelShown(bool);

impl Default for PanelShown {
    fn default() -> Self {
        Self(true)
    }
}

/// `/todo` and `/todo clear`.
fn todo_command(
    In(args): In<CommandArgs>,
    mut agents: Query<&mut Todos>,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let Ok(mut todos) = agents.get_mut(agent) else {
        return;
    };
    let notice = match args.args.as_str() {
        "" => Notice::info(agent, todos.render()),
        "clear" => {
            todos.set_if_neq(Todos::default());
            Notice::info(agent, "The todo list is cleared.")
        }
        _ => Notice::error(agent, "/todo takes nothing or `clear`"),
    };
    notices.write(notice);
}

/// Publishes each list that changed this frame.
fn publish(agents: Query<(&AgentId, &Todos), Changed<Todos>>, mut out: Publish) {
    for (id, todos) in &agents {
        out.event(&TodoList {
            agent_id: id.0.clone(),
            items: todos.0.clone(),
        });
    }
}

/// `/todo-panel`, in the view.
fn toggle_panel(
    In(args): In<CommandArgs>,
    mut shown: ResMut<PanelShown>,
    mut notices: MessageWriter<Notice>,
) {
    shown.0 = !shown.0;
    let now = if shown.0 { "shown" } else { "hidden" };
    notices.write(Notice::info(
        args.agent,
        format!("The todo panel is {now}."),
    ));
}

/// Keeps each mirrored agent's [`Todos`] as its events say, in the view.
fn mirror_lists(
    mut applied: MessageReader<Applied>,
    mirror: Option<Res<Mirror>>,
    mut commands: Commands,
) {
    let Some(mirror) = mirror else {
        return;
    };
    for Applied(record) in applied.read() {
        if record.kind != TodoList::KIND {
            continue;
        }
        let Ok(list) = serde_json::from_value::<TodoList>(record.data.clone()) else {
            continue;
        };
        if let Some(agent) = mirror.agent(&list.agent_id) {
            commands.entity(agent).insert(Todos(list.items));
        }
    }
}

/// The status line's `todo 3/7` of each agent with a list.
fn show_status(mut agents: Query<(&Todos, &mut StatusItems), Changed<Todos>>) {
    for (todos, mut items) in &mut agents {
        let item = match (todos.0.is_empty(), todos.open().next().is_some()) {
            (true, _) => STATUS,
            (false, open) => {
                let tone = if open { Tone::Yellow } else { Tone::Green };
                STATUS.says(format!("todo {}", todos.tally()), tone)
            }
        };
        items.show(item);
    }
}

/// The entity of the panel, which has a [`TuiPanel`] while it is shown.
#[derive(Component)]
struct TodoPanel;

fn spawn_panel(mut commands: Commands) {
    commands.spawn((Name::new("todo panel"), TodoPanel));
}

/// Shows the panel while it is on and the agent shown has a list.
fn place_panel(
    shown: Res<PanelShown>,
    (focused, todos): (Option<Res<ShownAgent>>, Query<&Todos>),
    panels: Query<(Entity, Has<TuiPanel>), With<TodoPanel>>,
    mut commands: Commands,
) {
    let focused = focused.and_then(|focused| todos.get(focused.0?).ok());
    let wanted = shown.0 && focused.is_some_and(|todos| !todos.0.is_empty());
    for (panel, placed) in &panels {
        match (wanted, placed) {
            (true, false) => {
                let placement = Placement::Right(Constraint::Length(PANEL_WIDTH));
                commands.entity(panel).insert(TuiPanel::new(placement));
            }
            (false, true) => {
                commands.entity(panel).remove::<TuiPanel>();
            }
            _ => {}
        }
    }
}

/// A frame for a changed list; the view does not watch it.
fn redraw(changed: Query<(), Changed<Todos>>, mut redraw: MessageWriter<RequestRedraw>) {
    if !changed.is_empty() {
        redraw.write(RequestRedraw);
    }
}

fn draw_panel(
    mut panels: Query<&mut PanelCanvas, With<TodoPanel>>,
    (focused, todos): (Option<Res<ShownAgent>>, Query<&Todos>),
) {
    let Some(todos) = focused.and_then(|focused| todos.get(focused.0?).ok()) else {
        return;
    };
    let title = format!("todo {}", todos.tally());
    for mut canvas in &mut panels {
        let list = Paragraph::new(todos.render()).block(Block::bordered().title(title.clone()));
        canvas.render(list);
    }
}

#[cfg(test)]
mod tests;
