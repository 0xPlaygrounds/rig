//! Plugin corpus: 34. Todo tracker (spec `34-todo-tracker.md`).
//!
//! The `todo` tool keeps a todo list per agent ([`Todos`], saved with the
//! session), which the terminal view shows in a panel (`/todo panel`
//! toggles it) and as `todo 3/7` in the status line. An agent whose turn
//! ends with an answer while items are still open is nudged once per
//! user message to carry on. `/todo` shows the list and `/todo clear`
//! clears it; a `--print` run prints the final list after the answer.

use rig_harness::prelude::*;
use rig_tui::ratatui::layout::Constraint;
use rig_tui::ratatui::widgets::{Block, Paragraph};
use rig_tui::{
    Focused, PanelCanvas, Placement, RequestRedraw, Side, StatusItem, StatusItems, StatusSystems,
    Tone, TuiPanel, TuiSystems,
};
use schemars::JsonSchema;
use serde::Deserialize;

/// The tool's name.
pub const TODO_TOOL: &str = "todo";

const DESCRIPTION: &str = "Keep your todo list for the current work. `set` replaces the list \
    with `items`, `check` ticks off the item `id`, `add` appends an item of `text`. The result \
    is the list, with each item's id.";

const RULES: &[&str] = &[
    "For a task of 3 or more steps, first `todo` `set` the steps, check each off with \
     `todo` `check` as soon as it is done, and `add` steps you discover. Do not end your turn \
     while items are open unless you say why.",
];

/// After the subagents at work; goes before the other agents at work.
const STATUS: StatusItem = StatusItem::at(Side::Left, 65, 9);

/// The panel's width.
const PANEL_WIDTH: u16 = 32;

/// Adds the `todo` tool and command, the nudge, the panel, the status
/// item and the list printed after a `--print` answer.
#[derive(Default)]
pub struct TodoPlugin;

impl Plugin for TodoPlugin {
    fn build(&self, app: &mut App) {
        let options = ToolOptions {
            rules: RULES,
            footprint: Footprint::Independent,
        };
        app.register_required_components::<Agent, Todos>()
            .register_required_components::<Agent, Nudged>()
            .init_resource::<PanelShown>()
            .add_message::<RequestRedraw>()
            .add_open_tool(TODO_TOOL, DESCRIPTION, options, on_todo)
            .add_command(
                "todo",
                "Show the todo list; `/todo clear` clears it, `/todo panel` toggles its panel",
                todo_command,
            )
            .add_observer(nudge)
            .add_observer(reset_nudge)
            .add_systems(Startup, spawn_panel)
            .add_systems(Update, redraw)
            .add_systems(PostUpdate, show_status.in_set(StatusSystems))
            .add_systems(PostUpdate, place_panel.in_set(TuiSystems::Prepare))
            .add_systems(PostUpdate, draw_panel.in_set(TuiSystems::Draw));
        let printing = app
            .world()
            .get_resource::<Invoked>()
            .is_some_and(|invoked| !invoked.interactive());
        if printing {
            app.add_systems(PostUpdate, print_list);
        }
    }
}

/// An agent's todo list, in order. Saved with the session.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Saved, Default, Clone, Debug)]
pub struct Todos(pub Vec<Todo>);

/// One item of a [`Todos`] list.
#[derive(Reflect, Clone, Debug, Default, PartialEq, Eq)]
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
    mut commands: Commands,
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
    commands
        .entity(called.call)
        .insert_if_new(Outcome::from(output));
}

/// A turn the model ended with an answer while items are open carries on
/// with one nudge per user message.
fn nudge(ended: On<TurnEnded>, mut agents: Query<(&Todos, &mut Nudged)>, mut commands: Commands) {
    if !matches!(ended.outcome, TurnOutcome::Answered(_)) {
        return;
    }
    let agent = ended.entity;
    let Ok((todos, mut nudged)) = agents.get_mut(agent) else {
        return;
    };
    let open: Vec<String> = todos
        .open()
        .map(|todo| format!("{}. {}", todo.id, todo.text))
        .collect();
    if nudged.0 || open.is_empty() {
        return;
    }
    nudged.0 = true;
    let text = format!(
        "You have {} open todo items: {}; continue, or explain why you stopped.",
        open.len(),
        open.join(", ")
    );
    let nudge = Deliver::new(agent, text, DeliveryMode::Queue);
    commands.trigger(nudge.with_origin(Origin::plugin(TODO_TOOL)));
}

/// A user's message allows one more nudge.
fn reset_nudge(typed: On<Deliver>, mut agents: Query<&mut Nudged>) {
    let user = typed.origin.kind == OriginKind::User && typed.command().is_none();
    if user
        && !typed.text.trim().is_empty()
        && let Ok(mut nudged) = agents.get_mut(typed.entity)
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

/// `/todo`, `/todo clear` and `/todo panel`.
fn todo_command(
    In(args): In<CommandArgs>,
    mut agents: Query<&mut Todos>,
    mut shown: ResMut<PanelShown>,
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
        "panel" => {
            shown.0 = !shown.0;
            let now = if shown.0 { "shown" } else { "hidden" };
            Notice::info(agent, format!("The todo panel is {now}."))
        }
        _ => Notice::error(agent, "/todo takes nothing, `clear` or `panel`"),
    };
    notices.write(notice);
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
    focused: Query<&Todos, With<Focused>>,
    panels: Query<(Entity, Has<TuiPanel>), With<TodoPanel>>,
    mut commands: Commands,
) {
    let wanted = shown.0 && focused.iter().any(|todos| !todos.0.is_empty());
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
    focused: Query<&Todos, With<Focused>>,
) {
    let Some(todos) = focused.iter().next() else {
        return;
    };
    let title = format!("todo {}", todos.tally());
    for mut canvas in &mut panels {
        let list = Paragraph::new(todos.render()).block(Block::bordered().title(title.clone()));
        canvas.render(list);
    }
}

/// After a `--print` answer, which the print plugin writes as it asks to
/// exit, the primary agent's list.
fn print_list(mut exits: MessageReader<AppExit>, agents: PrimaryQuery, lists: Query<&Todos>) {
    if exits.read().count() == 0 {
        return;
    }
    if let Some(todos) = primary(&agents).and_then(|agent| lists.get(agent).ok())
        && !todos.0.is_empty()
    {
        println!("\n{}", todos.render());
    }
}

#[cfg(test)]
mod tests;
