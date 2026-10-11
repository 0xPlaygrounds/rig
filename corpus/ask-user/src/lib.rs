//! Plugin corpus: 20. Ask the user (spec `20-ask-user.md`).
//!
//! The `ask_user` tool stops the turn on a question for the user. It is an
//! open tool: each call spawns an [`Ask`] that holds the call (`HoldOf`), so
//! every front that answers questions shows it (the terminal view as a
//! picker of the choices, with a typed answer for free text) and answers
//! it with a `Pick`, and its answer becomes the call's result. Nobody to answer (`--print`, no
//! front) answers at once [`NO_USER`]. An interrupt despawns the turn's
//! holds, which closes the question everywhere, and the kernel answers the
//! call as interrupted. The `[ask_user] timeout` setting refuses a question
//! left waiting, answered [`NO_ANSWER`]. The tool is exclusive, so the
//! calls of one reply are asked one after another.

use rig_harness::prelude::*;
use schemars::JsonSchema;
use serde::Deserialize;

/// The tool's result without anyone to ask.
pub const NO_USER: &str = "No user is available; make a reasonable assumption and state it.";

/// The tool's result for a question that waited past the timeout.
pub const NO_ANSWER: &str =
    "No answer: the user did not answer in time; make a reasonable assumption and state it.";

/// What `ask_user` says it does.
const DESCRIPTION: &str = "Ask the user a question and wait for the answer, which is the \
    result. Give `choices` when the answer is one of a few options.";

/// When the model should ask.
const RULE: &str = "Use `ask_user` when a choice is the user's to make and guessing would be \
    costly; otherwise decide and say what you assumed.";

/// The `[ask_user]` table of the user's settings.
#[derive(Resource, Deserialize, Clone, Debug, Default)]
#[serde(default, deny_unknown_fields)]
pub struct AskUserSettings {
    /// Seconds a question waits before it is answered [`NO_ANSWER`]; none
    /// waits however long it takes.
    pub timeout: Option<u64>,
}

/// The arguments of `ask_user`.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Args {
    /// The question.
    question: String,
    /// The answers to pick from, if any.
    #[serde(default)]
    choices: Vec<String>,
    /// Whether the user may answer in words of their own besides the
    /// choices; always so without choices.
    #[serde(default)]
    allow_free_text: bool,
}

/// On the [`Ask`] of an `ask_user` call: its answer is the call's result.
#[derive(Component)]
struct AskedByTool;

#[derive(Default)]
pub struct AskUserPlugin;

impl Plugin for AskUserPlugin {
    fn build(&self, app: &mut App) {
        let options = ToolOptions {
            rules: &[RULE],
            footprint: Some(Footprint::Exclusive),
        };
        app.add_setting::<AskUserSettings>("ask_user")
            .add_open_tool("ask_user", DESCRIPTION, options, ask)
            .add_systems(AgentPass, answered.in_set(Stage::Review));
        #[cfg(feature = "tui")]
        {
            use rig_tui::{AppToolRenderersExt, RESULT_LINES};
            app.add_tool_renderer("ask_user", |call| {
                let question = call.argument("question").unwrap_or_default();
                let mut lines = vec![call.header("Ask", question)];
                lines.extend(call.result_lines(RESULT_LINES));
                lines
            });
        }
    }
}

/// Answers at once when nobody can answer; otherwise holds the call with
/// its question, named after a subagent that asks it, and refuses the
/// question once it waited past the timeout.
fn ask(
    called: On<ToolCalled<Args>>,
    answerers: Answerers,
    settings: Res<AskUserSettings>,
    subagents: Query<&Name, With<SpawnedBy>>,
    mut commands: Commands,
) {
    let call = called.call;
    if !answerers.can_answer(Some(called.agent)) {
        let result = ToolResult::success(NO_USER.to_owned().into());
        commands.entity(call).insert_if_new(Outcome::from(result));
        return;
    }
    let args = &called.args;
    let title = match subagents.get(called.agent) {
        Ok(name) => format!("{name} asks: {}", args.question.trim()),
        Err(_) => args.question.trim().to_owned(),
    };
    let mut question = args
        .choices
        .iter()
        .fold(Ask::new(title), |ask, choice| ask.choice(choice, choice));
    if args.allow_free_text || args.choices.is_empty() {
        question = question.free_text();
    }
    let asked = commands
        .spawn((
            question,
            AskedByTool,
            ForAgent(called.agent),
            HoldOf(call),
            Effect::child_of(Some(called.effect)),
        ))
        .id();
    if let Some(seconds) = settings.timeout {
        commands
            .delayed()
            .duration(Duration::from_secs(seconds))
            .queue(move |world: &mut World| {
                if let Ok(mut asked) = world.get_entity_mut(asked) {
                    asked.insert_if_new(Outcome::refused(NO_ANSWER));
                }
            });
    }
}

/// Gives each answered question's call its answer, the choice or the text,
/// or why it was not answered (closed, timed out), and lets go of the call.
fn answered(
    asks: Query<(Entity, &Outcome, &HoldOf), (With<AskedByTool>, Added<Outcome>)>,
    mut commands: Commands,
) {
    for (asked, outcome, &HoldOf(call)) in &asks {
        let text = match (Answer::of(outcome), &outcome.0) {
            (Some(Answer::Choice(text) | Answer::Text(text)), _) => text,
            (None, Err(report)) => report.message.clone(),
            (None, Ok(_)) => NO_ANSWER.to_owned(),
        };
        let result = ToolResult::success(text.into());
        commands.entity(call).insert_if_new(Outcome::from(result));
        commands.entity(asked).remove::<HoldOf>();
    }
}

#[cfg(test)]
mod tests;
