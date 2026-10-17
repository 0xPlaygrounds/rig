//! Plugin corpus: 44. Session goal (saved custom state) (spec `44-session-goal.md`).
//!
//! `/goal <text>` sets the session's goal, `/goal` shows it and
//! `/goal clear` removes it. The goal is a saved, immutable resource: each
//! change inserts a new value, which the session log records and a
//! restart, `/reload` and `/resume` bring back. Each insertion, restored
//! ones too, rewrites one entity's [`PromptSection`], in every agent's
//! system prompt (subagents too) from the next request on, and its
//! [`ResumeLabel`], the goal in this session's `/resume` line, and the
//! status line's goal item. A new session starts without one. The model
//! has no tool to change it.

use rig_basics::sessions::ResumeLabel;
use rig_harness::prelude::*;
use rig_tools::shorten;
use rig_tui::{AppStatus, Side, StatusItem, Tone};

/// Where the goal goes in the system prompt: after the project's
/// instructions, which change less often, and before the environment.
pub const ORDER_GOAL: i32 = 260;

/// The tag of the goal's prompt section and `/resume` label.
const TAG: &str = "goal";

/// Most characters of the goal in a session's `/resume` line, which goes
/// on with the session's cost and directory.
const LABEL_CHARS: usize = 40;

/// Where the goal is in the status line: right after the session's name,
/// cut down to 12 columns when the line is narrow, and gone before the
/// model's reasoning setting.
const GOAL_ITEM: StatusItem = StatusItem::new("session-goal", Side::Left, 15, 6).shrinks_to(12);

/// `/goal`, and the goal's prompt section, `/resume` label and status item.
#[derive(Default)]
pub struct SessionGoalPlugin;

impl Plugin for SessionGoalPlugin {
    fn build(&self, app: &mut App) {
        app.world_mut().spawn((
            GoalShown,
            PromptSection::new(ORDER_GOAL, TAG, ""),
            ResumeLabel::new(TAG, ""),
        ));
        app.init_resource::<AppStatus>()
            .add_observer(show_goal)
            .init_resource::<SessionGoal>()
            .add_command(
                "goal",
                "Set the session's goal (/goal <text>), show it (/goal), or remove it (/goal clear)",
                goal,
            );
    }
}

/// The session's goal; empty for none. Saved with the session; each change
/// is a new value inserted.
#[derive(Resource, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[component(immutable)]
#[reflect(Resource, Saved, Default, Clone, Debug)]
pub struct SessionGoal(pub String);

/// On the entity whose prompt section and `/resume` label show the goal.
#[derive(Component)]
struct GoalShown;

/// Shows the goal each time a value is inserted: at startup, on a change,
/// and when it is restored.
fn show_goal(
    _insert: On<Insert<SessionGoal>>,
    goal: Res<SessionGoal>,
    mut shown: Query<(&mut PromptSection, &mut ResumeLabel), With<GoalShown>>,
    mut status: ResMut<AppStatus>,
) {
    let (section, label, item) = match goal.0.as_str() {
        "" => Default::default(),
        goal => (
            format!(
                "Session goal: {goal}\nThe user set this goal for the whole session; work \
                 towards it. Only the user can change it."
            ),
            format!("◎ {}", shorten(goal, LABEL_CHARS)),
            format!("◎ {goal}"),
        ),
    };
    for (mut prompt, mut resume) in &mut shown {
        prompt.set_if_neq(PromptSection::new(ORDER_GOAL, TAG, section.clone()));
        resume.set_if_neq(ResumeLabel::new(TAG, label.clone()));
    }
    status.0.show(GOAL_ITEM.says(item, Tone::Cyan));
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
            let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
            let reply = format!("Session goal set: {text}");
            commands.insert_resource(SessionGoal(text));
            reply
        }
    };
    notices.write(Notice::info(args.agent, reply));
}

#[cfg(test)]
mod tests;
