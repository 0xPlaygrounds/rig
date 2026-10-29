//! `rig-todo`: a goals and checklist plugin whose state survives exit,
//! `/reload` and a crash.
//!
//! Each agent keeps its own [`Checklist`], which the model changes with
//! the `todo` tool (`add`, `done`, `list`, `clear`) and the user with
//! `/todo`; the session keeps one [`Goal`] line, set with `/todo goal`.
//! Both are saved values under stable keys ([`LIST_KEY`], [`GOAL_KEY`]),
//! the goal listed so `/resume` shows it, and plain serde, so a program
//! without an app reads a stopped session's ([`read_stopped`]). The
//! status line shows `3/5 done`, and a reminder about open items goes to
//! the model a while after the list last changed; both are re-armed from
//! the list whenever it changes, a restore's insert included.

use rig_harness::prelude::*;
use rig_harness::rig_memory::SessionMeta;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The tool's name.
pub const TODO_TOOL: &str = "todo";

/// The saved key of each agent's [`Checklist`].
pub const LIST_KEY: &str = "todo.list";

/// The saved key of the session's [`Goal`], listed with the session.
pub const GOAL_KEY: &str = "todo.goal";

/// How long after the list last changed, with items open, the model is
/// reminded of them.
pub const REMIND_AFTER: Duration = Duration::from_secs(15 * 60);

const DESCRIPTION: &str = "Keep your checklist for the current work. `add` appends an item of \
    `text`, `done` ticks off the item `id`, `list` shows the list, `clear` empties it. The \
    result is the list with each item's id. For work of 3 or more steps, `add` each step first \
    and tick it off as soon as it is done.";

/// The `todo` tool, the `/todo` command, the status item and the
/// reminder.
#[derive(Default)]
pub struct TodoPlugin;

impl Plugin for TodoPlugin {
    fn build(&self, app: &mut App) {
        let options = ToolOptions {
            footprint: Some(Touches::Independent),
        };
        app.register_required_components::<Agent, Checklist>()
            .init_resource::<Goal>()
            .add_open_tool(TODO_TOOL, DESCRIPTION, options, on_todo)
            .add_command(
                "todo",
                "Show the checklist and goal; `/todo add <text>`, `/todo done <id>`, \
                 `/todo clear`, `/todo goal <text>`",
                todo_command,
            )
            .add_systems(Update, arm_reminders)
            .add_observer(remind);
        // Both under stable keys, so a renamed type keeps its values and a
        // program reads them by key; the goal is listed for `/resume`.
        // An unlisted key: `SavedAs` has no way back to the app but `listed`.
        let _unlisted = app.saved_as::<Checklist>(LIST_KEY);
        app.saved_as::<Goal>(GOAL_KEY).listed();
        #[cfg(feature = "tui")]
        app.add_systems(PostUpdate, show_status.in_set(rig_tui::StatusSystems::Show));
    }
}

/// An agent's checklist, in order. Saved with the session under
/// [`LIST_KEY`], through serde: a field added later takes its default, and
/// a value that no longer reads is skipped with a notice.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[reflect(opaque, Component, Saved, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Checklist {
    /// The items, in the order they were added.
    pub items: Vec<Item>,
}

/// One item of a [`Checklist`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Item {
    /// The id it is ticked off by.
    pub id: u32,
    /// What to do.
    pub text: String,
    /// Whether it is done.
    pub done: bool,
}

/// The session's goal: one line, empty when none was set. Saved under
/// [`GOAL_KEY`] and listed with the session.
#[derive(Resource, Reflect, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[reflect(opaque, Resource, Saved, Default, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Goal(pub String);

impl Checklist {
    /// The items not done yet.
    pub fn open(&self) -> impl Iterator<Item = &Item> {
        self.items.iter().filter(|item| !item.done)
    }

    /// Such as `1/3 done`.
    pub fn tally(&self) -> String {
        let done = self.items.iter().filter(|item| item.done).count();
        format!("{done}/{} done", self.items.len())
    }

    /// The list, one item a line.
    pub fn render(&self) -> String {
        if self.items.is_empty() {
            return "The checklist is empty.".to_owned();
        }
        let lines: Vec<String> = self.items.iter().map(Item::line).collect();
        lines.join("\n")
    }

    /// Adds an item of `text`.
    fn add(&mut self, text: &str) -> Result<(), String> {
        let text = text.trim();
        if text.is_empty() {
            return Err("`add` needs a `text`".to_owned());
        }
        let id = self.items.iter().map(|item| item.id).max().unwrap_or(0) + 1;
        self.items.push(Item {
            id,
            text: text.to_owned(),
            done: false,
        });
        Ok(())
    }

    /// Ticks off the item `id`.
    fn done(&mut self, id: u32) -> Result<(), String> {
        let item = self.items.iter_mut().find(|item| item.id == id);
        item.ok_or_else(|| format!("no item has id {id}"))?.done = true;
        Ok(())
    }
}

impl Item {
    /// Such as `[x] 1. Write the parser`.
    fn line(&self) -> String {
        let mark = if self.done { "x" } else { " " };
        format!("[{mark}] {}. {}", self.id, self.text)
    }
}

/// What a `todo` call does.
#[derive(Deserialize, JsonSchema, Clone, Copy, Debug)]
#[serde(rename_all = "snake_case")]
enum Action {
    Add,
    Done,
    List,
    Clear,
}

/// The arguments of a `todo` call.
#[derive(Deserialize, JsonSchema, Debug)]
#[serde(deny_unknown_fields)]
struct TodoArgs {
    /// `add` appends an item of `text`, `done` ticks off the item `id`,
    /// `list` shows the list, `clear` empties it.
    action: Action,
    /// For `add`: what to do.
    text: Option<String>,
    /// For `done`: the id of the item done.
    id: Option<u32>,
}

/// The `todo` tool: changes the calling agent's list and answers with it.
fn on_todo(
    called: On<ToolCalled<TodoArgs>>,
    mut lists: Query<&mut Checklist>,
    mut decisions: Query<&mut Decision, ToolCalls>,
) {
    let result = match lists.get_mut(called.agent) {
        Ok(mut list) => {
            let mut changed = list.clone();
            let applied = match called.args.action {
                Action::Add => changed.add(called.args.text.as_deref().unwrap_or_default()),
                Action::Done => match called.args.id {
                    Some(id) => changed.done(id),
                    None => Err("`done` needs an `id`".to_owned()),
                },
                Action::List => Ok(()),
                Action::Clear => {
                    changed.items.clear();
                    Ok(())
                }
            };
            applied.map(|()| {
                let rendered = changed.render();
                list.set_if_neq(changed);
                rendered
            })
        }
        Err(_) => Err("this agent keeps no checklist".to_owned()),
    };
    let output = match result {
        Ok(list) => ToolResult::success(list.into()),
        Err(why) => ToolResult::failed(ToolExecutionError::other(why)),
    };
    if let Ok(mut decision) = decisions.get_mut(called.call) {
        decision.decide(Decider::plugin(TODO_TOOL), output);
    }
}

/// `/todo`, `/todo add <text>`, `/todo done <id>`, `/todo clear` and
/// `/todo goal <text>` (an empty text clears the goal).
fn todo_command(
    In(args): In<CommandArgs>,
    mut lists: Query<&mut Checklist>,
    mut goal: ResMut<Goal>,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let Ok(mut list) = lists.get_mut(agent) else {
        notices.write(Notice::error(agent, "this agent keeps no checklist"));
        return;
    };
    let (word, rest) = args
        .args
        .split_once(' ')
        .unwrap_or((args.args.as_str(), ""));
    let mut changed = list.clone();
    let applied = match word {
        "" => Ok(()),
        "add" => changed.add(rest),
        "done" => rest
            .trim()
            .parse()
            .map_err(|_| "/todo done takes an item's id".to_owned())
            .and_then(|id| changed.done(id)),
        "clear" => {
            changed.items.clear();
            Ok(())
        }
        "goal" => {
            goal.set_if_neq(Goal(rest.trim().to_owned()));
            Ok(())
        }
        _ => Err("/todo takes nothing, `add`, `done`, `clear` or `goal`".to_owned()),
    };
    let notice = match applied {
        Ok(()) => {
            list.set_if_neq(changed);
            let goal = match goal.0.as_str() {
                "" => String::new(),
                text => format!("Goal: {text}\n"),
            };
            Notice::info(agent, format!("{goal}{}", list.render()))
        }
        Err(why) => Notice::error(agent, why),
    };
    notices.write(notice);
}

/// The status line's `1/3 done` of each agent with a list, shown again
/// whenever the list changes, as when a restore inserts it.
#[cfg(feature = "tui")]
fn show_status(mut agents: Query<(&Checklist, &mut rig_tui::StatusItems), Changed<Checklist>>) {
    use rig_tui::{Side, StatusItem, Tone};
    /// After the status; goes before the other agents at work.
    const STATUS: StatusItem = StatusItem::new("todo", Side::Left, 65, 9);
    for (list, mut items) in &mut agents {
        let item = match (list.items.is_empty(), list.open().next().is_some()) {
            (true, _) => STATUS,
            (false, true) => STATUS.says(list.tally(), Tone::Yellow),
            (false, false) => STATUS.says(list.tally(), Tone::Green),
        };
        items.show(item);
    }
}

/// The wait before the model of `agent` is reminded of its open items.
#[derive(Component, Debug)]
pub struct Reminder {
    /// The agent reminded.
    pub agent: Entity,
}

/// Arms an agent's reminder whenever its list changes with items open
/// (a restore's insert included), putting off one armed already, and
/// calls it off once nothing is open.
fn arm_reminders(
    lists: Query<(Entity, &Checklist), Changed<Checklist>>,
    reminders: Query<(Entity, &Reminder)>,
    mut commands: Commands,
) {
    for (agent, list) in &lists {
        let armed = reminders
            .iter()
            .find(|(_, reminder)| reminder.agent == agent);
        match (armed, list.open().next().is_some()) {
            (Some((reminder, _)), true) => {
                commands
                    .entity(reminder)
                    .insert(WakeAt::after(REMIND_AFTER));
            }
            (None, true) => {
                let name = Name::new("checklist reminder");
                commands.spawn((name, Reminder { agent }, WakeAt::after(REMIND_AFTER)));
            }
            (Some((reminder, _)), false) => commands.entity(reminder).despawn(),
            (None, false) => {}
        }
    }
}

/// Reminds the model of its open items, with its next model call, once
/// the wait is over.
fn remind(
    due: On<Add<Due>>,
    reminders: Query<&Reminder>,
    lists: Query<&Checklist>,
    mut commands: Commands,
) {
    let Ok(reminder) = reminders.get(due.entity) else {
        return;
    };
    commands.entity(due.entity).despawn();
    let Ok(list) = lists.get(reminder.agent) else {
        return;
    };
    let open: Vec<String> = list.open().map(Item::line).collect();
    if open.is_empty() {
        return;
    }
    let text = format!("Your checklist has open items:\n{}", open.join("\n"));
    let message = Deliver::new(reminder.agent, text, DeliveryMode::Inform);
    commands.trigger(message.with_author(Author::plugin(TODO_TOOL)));
}

/// What a stopped session holds of this plugin, read without an app.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Stopped {
    /// The session's goal, empty when none was set.
    pub goal: String,
    /// Each agent's checklist, by agent id, in the order their logs were
    /// started; an agent that never changed its list has none here.
    pub lists: Vec<(String, Checklist)>,
}

/// The goal and checklists of the stopped session kept in `store`, as a
/// program without Bevy reads them: rig-memory's session read and its
/// typed saved values. A value that no longer reads is left out.
///
/// # Errors
///
/// When the session cannot be read.
pub fn read_stopped(store: &dyn Store) -> Result<Stopped, String> {
    let session = Session::read(store).map_err(|failure| failure.to_string())?;
    let goal = session.value::<Goal>(GOAL_KEY).and_then(Result::ok);
    let lists = session.agents.iter().filter_map(|agent| {
        let list = agent.value::<Checklist>(LIST_KEY)?.ok()?;
        Some((agent.id.clone(), list))
    });
    Ok(Stopped {
        goal: goal.map(|goal| goal.0).unwrap_or_default(),
        lists: lists.collect(),
    })
}

/// The goal as the session's listing holds it, as `/resume` shows it,
/// without opening a log.
pub fn listed_goal(store: &dyn Store) -> Option<String> {
    let meta = SessionMeta::read(store)?;
    let goal = meta.listed.get(GOAL_KEY)?;
    goal.as_str().map(str::to_owned)
}

#[cfg(test)]
mod tests;
