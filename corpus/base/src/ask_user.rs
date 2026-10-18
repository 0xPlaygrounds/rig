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
    result. Give `choices` when the answer is one of a few options. Use it when a choice is \
    the user's to make and guessing would be costly; otherwise decide and say what you assumed.";

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

/// The Review system that gives an answered question's call its answer,
/// to order against.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AskUserSystems;

#[derive(Default)]
pub struct AskUserPlugin;

impl Plugin for AskUserPlugin {
    fn build(&self, app: &mut App) {
        let options = ToolOptions {
            footprint: Some(Touches::Writes(Scope::Anything)),
        };
        app.add_setting::<AskUserSettings>("ask_user")
            .add_open_tool("ask_user", DESCRIPTION, options, ask)
            .configure_sets(AgentPass, AskUserSystems.in_set(Stage::Review))
            .add_systems(AgentPass, answered.in_set(AskUserSystems));
    }
}

/// The view half: how the terminal view draws an `ask_user` call, named
/// in `plugins.toml` as the entry's `view`.
#[cfg(feature = "tui")]
#[derive(Default)]
pub struct AskUserView;

#[cfg(feature = "tui")]
impl Plugin for AskUserView {
    fn build(&self, app: &mut App) {
        use rig_tui::AppToolRenderersExt;
        app.add_tool_renderer("ask_user", |call| {
            let question = call.argument("question").unwrap_or_default();
            let mut lines = vec![call.header("Ask", question)];
            lines.extend(call.result_lines());
            lines
        });
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
    mut calls: Query<&mut Decision, ToolCalls>,
    mut commands: Commands,
) {
    let call = called.call;
    if !answerers.can_answer(Some(called.agent)) {
        if let Ok(mut decision) = calls.get_mut(call) {
            decision.decide(
                Decider::plugin("ask-user"),
                ToolResult::success(NO_USER.to_owned().into()),
            );
        }
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
            Effect::with_parent_id(Some(called.effect)),
        ))
        .id();
    if let Some(seconds) = settings.timeout {
        commands
            .delayed()
            .duration(Duration::from_secs(seconds))
            .queue(move |world: &mut World| {
                if let Some(mut decision) = world.get_mut::<Decision>(asked) {
                    decision.decide(Decider::plugin("ask-user"), Outcome::refused(NO_ANSWER));
                }
            });
    }
}

/// Gives each answered question's call its answer, the choice or the text,
/// or why it was not answered (closed, timed out), and lets go of the call.
fn answered(
    asks: Query<(Entity, &Outcome, &HoldOf), (PluginEffects, With<AskedByTool>, Added<Outcome>)>,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut commands: Commands,
) {
    for (asked, outcome, &HoldOf(call)) in &asks {
        let text = match (Answer::of(outcome), &outcome.0) {
            (Some(Answer { choices, text }), _) => {
                let choices = choices.join(", ");
                match text {
                    Some(text) if !choices.is_empty() => format!("{choices}: {text}"),
                    Some(text) => text,
                    None => choices,
                }
            }
            (None, Err(report)) => report.message.clone(),
            (None, Ok(_)) => NO_ANSWER.to_owned(),
        };
        if let Ok(mut decision) = calls.get_mut(call) {
            decision.decide(
                Decider::plugin("ask-user"),
                ToolResult::success(text.into()),
            );
        }
        commands.entity(asked).remove::<HoldOf>();
    }
}

#[cfg(test)]
mod tests;
