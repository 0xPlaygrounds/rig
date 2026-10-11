//! Plugin corpus: 44. Session goal (saved custom state) (spec `44-session-goal.md`).
//!
//! `/goal <text>` sets the session's goal, `/goal` shows it and
//! `/goal clear` removes it. The goal is a saved, immutable resource: each
//! change inserts a new value, which the session log records and a
//! restart, `/reload` and `/resume` bring back, and whose insertion
//! rewrites a [`PromptSection`] every agent's system prompt carries, so a
//! subagent sees it too and a change counts from the next request. A new
//! session starts without one. The status line shows it, and `/resume`'s
//! picker lists each session's goal beside its title, from a `goal.txt`
//! this plugin keeps in each session's directory. The model has no tool to
//! change it.

use std::fs;
use std::io::ErrorKind;
use std::path::PathBuf;

use bevy_log::warn;
use rig_harness::harness_protocol::{Home, SessionDir};
use rig_harness::prelude::*;
use rig_sessions::{SessionsPlugin, SwitchSession, list};
use rig_tools::shorten;
use rig_tui::{AppStatus, PickItem, PickRequest, Side, StatusItem, StatusSystems, Tone};

/// Where the goal goes in the system prompt: after the project's
/// instructions, which change less often, and before the environment.
pub const ORDER_GOAL: i32 = 260;

/// The tag around the goal in the system prompt.
const TAG: &str = "session_goal";

/// Most characters of the goal in the status line and the picker.
const SHOWN_CHARS: usize = 40;

/// Where the goal is in the status line: right after the session's name,
/// and gone before the model's reasoning setting when the line is narrow.
const GOAL_ITEM: StatusItem = StatusItem::at(Side::Left, 15, 6);

/// The file in a session's directory that holds its goal, for `/resume`.
const GOAL_FILE: &str = "goal.txt";

/// `/goal`, the goal's prompt section and status item, and a `/resume`
/// picker that shows each session's goal.
#[derive(Default)]
pub struct SessionGoalPlugin;

impl Plugin for SessionGoalPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<SessionsPlugin>() {
            app.add_plugins(SessionsPlugin);
        }
        app.init_resource::<AppStatus>()
            .add_message::<PickRequest>()
            .add_observer(on_goal)
            .init_resource::<SessionGoal>()
            .add_command(
                "goal",
                "Set the session's goal (/goal <text>), show it (/goal), or remove it (/goal clear)",
                goal,
            )
            .add_systems(
                PostUpdate,
                show_goal
                    .in_set(StatusSystems)
                    .run_if(resource_changed::<SessionGoal>),
            );
    }

    /// Replaces `rig-sessions`' `/resume` with one whose picker shows each
    /// session's goal.
    fn finish(&self, app: &mut App) {
        let world = app.world_mut();
        let mut commands = world.query_filtered::<(Entity, &Name), With<SlashCommand>>();
        let theirs = commands
            .iter(world)
            .find(|(_, name)| name.as_str() == "/resume")
            .map(|(entity, _)| entity);
        if let Some(theirs) = theirs {
            world.entity_mut(theirs).insert(Disabled);
        }
        app.add_command(
            "resume",
            "Pick an earlier session to resume, or /resume <id>",
            resume,
        );
    }
}

/// The session's goal; empty for none. Saved with the session; each change
/// is a new value inserted.
#[derive(Resource, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[component(immutable)]
#[reflect(Resource, Saved, Default, Clone, Debug)]
pub struct SessionGoal(pub String);

impl SessionGoal {
    /// The prompt section's text; empty without a goal.
    fn section(&self) -> String {
        if self.0.is_empty() {
            return String::new();
        }
        format!(
            "Session goal: {}\nThe user set this goal for the whole session; work towards it. \
             Only the user can change it.",
            self.0
        )
    }
}

/// On the [`PromptSection`] of the goal.
#[derive(Component)]
struct GoalSection;

/// Keeps the goal's prompt section and its `goal.txt` the goal, each time a
/// value is inserted: at startup, on a change, and when it is restored.
fn on_goal(
    _insert: On<Insert<SessionGoal>>,
    goal: Res<SessionGoal>,
    mut sections: Query<&mut PromptSection, With<GoalSection>>,
    paths: Option<Res<SessionPaths>>,
    mut commands: Commands,
) {
    let section = PromptSection::new(ORDER_GOAL, TAG, goal.section());
    match sections.single_mut() {
        Ok(mut mine) => {
            mine.set_if_neq(section);
        }
        Err(_) => {
            commands.spawn((GoalSection, section));
        }
    }
    if let Some(paths) = paths
        && let Err(failure) = write_goal_file(goal_file(&paths), &goal.0)
    {
        warn!("could not write the session goal for /resume: {failure}");
    }
}

/// The goal file of the session in `dir`.
fn goal_file(dir: &SessionDir) -> PathBuf {
    dir.path().join(GOAL_FILE)
}

/// Writes `goal` to `file`, or removes it for no goal.
fn write_goal_file(file: PathBuf, goal: &str) -> std::io::Result<()> {
    if !goal.is_empty() {
        return fs::write(file, goal);
    }
    match fs::remove_file(file) {
        Err(failure) if failure.kind() != ErrorKind::NotFound => Err(failure),
        _ => Ok(()),
    }
}

/// `text` on one line.
fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// `/goal <text>`, `/goal`, `/goal clear`.
fn goal(
    In(args): In<CommandArgs>,
    goal: Res<SessionGoal>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let reply = match args.args.as_str() {
        "" if goal.0.is_empty() => "No session goal; /goal <text> sets one.".to_owned(),
        "" => format!("Session goal: {}", goal.0),
        "clear" => {
            commands.insert_resource(SessionGoal::default());
            "Session goal cleared.".to_owned()
        }
        text => {
            let text = one_line(text);
            let reply = format!("Session goal set: {text}");
            commands.insert_resource(SessionGoal(text));
            reply
        }
    };
    notices.write(Notice::info(args.agent, reply));
}

fn show_goal(goal: Res<SessionGoal>, mut status: ResMut<AppStatus>) {
    let text = match goal.0.as_str() {
        "" => String::new(),
        goal => format!("◎ {}", shorten(goal, SHOWN_CHARS)),
    };
    status.0.show(GOAL_ITEM.says(text, Tone::Cyan));
}

/// `/resume <id>` runs that session; without arguments, a picker lists the
/// earlier sessions with their goals.
fn resume(
    In(args): In<CommandArgs>,
    paths: Option<Res<SessionPaths>>,
    mut picks: MessageWriter<PickRequest>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    if !args.args.is_empty() {
        commands.trigger(SwitchSession {
            session: Some(args.args),
        });
        return;
    }
    let Some(paths) = paths else {
        return;
    };
    let home = Home::from_env();
    let items: Vec<PickItem> = list(&home, paths.path())
        .into_iter()
        .map(|session| {
            let mut label = session.label();
            let goal = fs::read_to_string(goal_file(&home.session(&session.id)));
            if let Ok(goal) = goal
                && !goal.trim().is_empty()
            {
                label = format!("{label}  · ◎ {}", shorten(goal.trim(), SHOWN_CHARS));
            }
            PickItem {
                label,
                command: format!("resume {}", session.id),
            }
        })
        .collect();
    if items.is_empty() {
        notices.write(Notice::info(args.agent, "No earlier session to resume."));
        return;
    }
    picks.write(PickRequest {
        agent: args.agent,
        title: "Resume a session".to_owned(),
        items,
        selected: 0,
    });
}

#[cfg(test)]
mod tests;
