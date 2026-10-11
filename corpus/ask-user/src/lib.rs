//! Plugin corpus: 20. Ask the user (spec `20-ask-user.md`).
//!
//! The `ask_user` tool stops the turn on a question for the user. It is an
//! open tool: its call stays open with a [`Question`] on the call's
//! entity until the user answers, and the answer is its result. One
//! question is shown at a time, the oldest first, as [`AskedUser`], which
//! the terminal view turns into a picker of the choices (plus "Other…"
//! for free text). The user answers by picking, by typing a message while
//! a free-text question waits, or with `/answer`. Without someone at the
//! terminal the tool answers at once that no user is available, and an
//! [`AskUserSettings::timeout`] answers "no answer" to a question left
//! waiting. Interrupting the turn despawns the call, which the kernel
//! answers as interrupted by the user, and the next question shows.
//! The calls of one reply are exclusive, so the kernel asks them one after
//! another.

use rig_harness::prelude::*;
use schemars::JsonSchema;
use serde::Deserialize;

#[cfg(feature = "tui")]
mod tui;

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

/// How the plugin waits for answers; insert it to change it.
#[derive(Resource, Reflect, Clone, Debug, Default)]
#[reflect(Resource, Default)]
pub struct AskUserSettings {
    /// How long a question waits before it is answered [`NO_ANSWER`];
    /// `None` waits for the user however long it takes.
    pub timeout: Option<Duration>,
}

/// The arguments of `ask_user`.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Ask {
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

/// A question waiting for the user, on its `ask_user` call's entity.
#[derive(Component, Clone, Debug)]
pub struct Question {
    /// Its number, in the order asked; the oldest is shown first.
    pub id: u64,
    /// The agent that asks.
    pub agent: Entity,
    /// The question, naming a subagent that asks it.
    pub text: String,
    /// The answers to pick from.
    pub choices: Vec<String>,
    /// Whether any text answers it.
    pub free_text: bool,
}

/// Show `question` to the user: the question of call `call` is the one
/// waiting now. A front (the terminal view's picker) observes it.
#[derive(Event, Clone, Debug)]
pub struct AskedUser {
    /// The `ask_user` call.
    pub call: Entity,
    /// The question.
    pub question: Question,
}

/// The question shown, and the number of the last one asked.
#[derive(Resource, Default)]
struct Shown {
    call: Option<Entity>,
    last_id: u64,
}

#[derive(Default)]
pub struct AskUserPlugin;

impl Plugin for AskUserPlugin {
    fn build(&self, app: &mut App) {
        let options = ToolOptions {
            rules: &[RULE],
            footprint: Footprint::Exclusive,
        };
        app.init_resource::<AskUserSettings>()
            .init_resource::<Shown>()
            .add_open_tool("ask_user", DESCRIPTION, options, ask)
            .add_command(
                "answer",
                "Answer the agent's question: /answer <text>, or alone to show it again",
                answer,
            )
            .add_observer(answer_typed)
            .add_observer(show_next);
        #[cfg(feature = "tui")]
        tui::build(app);
    }
}

/// Answers at once without a user at the terminal; otherwise keeps the
/// call open with its question, shown now unless another one is.
fn ask(
    called: On<ToolCalled<Ask>>,
    invoked: Option<Res<Invoked>>,
    settings: Res<AskUserSettings>,
    subagents: Query<&Name, With<SpawnedBy>>,
    mut shown: ResMut<Shown>,
    mut commands: Commands,
) {
    let call = called.call;
    if !invoked.is_some_and(|invoked| invoked.interactive()) {
        commands
            .entity(call)
            .insert(ToolOutput(ToolResult::success(NO_USER.to_owned().into())));
        return;
    }
    let args = &called.args;
    let question = args.question.trim();
    shown.last_id += 1;
    let question = Question {
        id: shown.last_id,
        agent: called.agent,
        text: match subagents.get(called.agent) {
            Ok(name) => format!("{name} asks: {question}"),
            Err(_) => question.to_owned(),
        },
        choices: args.choices.clone(),
        free_text: args.allow_free_text || args.choices.is_empty(),
    };
    commands.entity(call).insert(question.clone());
    if shown.call.is_none() {
        shown.call = Some(call);
        commands.trigger(AskedUser { call, question });
    }
    if let Some(timeout) = settings.timeout {
        commands
            .delayed()
            .duration(timeout)
            .queue(move |world: &mut World| settle(world, call, NO_ANSWER));
    }
}

/// Answers the question of `call` with `answer`, unless it was answered.
fn settle(world: &mut World, call: Entity, answer: &str) {
    let Ok(mut call) = world.get_entity_mut(call) else {
        return;
    };
    if call.take::<Question>().is_some() {
        call.insert(ToolOutput(ToolResult::success(answer.to_owned().into())));
    }
}

/// Once the shown question is answered or its call is gone, shows the
/// oldest one still waiting.
fn show_next(
    removed: On<Remove<Question>>,
    questions: Query<(Entity, &Question)>,
    mut shown: ResMut<Shown>,
    mut commands: Commands,
) {
    if shown.call != Some(removed.entity) {
        return;
    }
    let next = questions
        .iter()
        .filter(|(call, _)| *call != removed.entity)
        .min_by_key(|(_, question)| question.id);
    shown.call = next.map(|(call, _)| call);
    if let Some((call, question)) = next {
        let question = question.clone();
        commands.trigger(AskedUser { call, question });
    }
}

/// `/answer [#id] [text]`: answers the question numbered `id`, or the one
/// shown. Without text it shows the question again, or for a free-text
/// one picked as "Other…", asks for the text.
fn answer(
    In(args): In<CommandArgs>,
    shown: Res<Shown>,
    questions: Query<(Entity, &Question)>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let (id, text) = match args.args.strip_prefix('#') {
        Some(rest) => {
            let (id, text) = rest.split_once(' ').unwrap_or((rest, ""));
            (id.parse::<u64>().ok(), text.trim())
        }
        None => (None, args.args.as_str()),
    };
    let asked = match id {
        Some(id) => questions.iter().find(|(_, question)| question.id == id),
        None => shown.call.and_then(|call| questions.get(call).ok()),
    };
    let Some((call, question)) = asked else {
        notices.write(Notice::info(args.agent, "No such question is waiting."));
        return;
    };
    if text.is_empty() {
        if id.is_some() && question.free_text {
            notices.write(Notice::info(
                args.agent,
                format!("{} Type your answer and press Enter.", question.text),
            ));
        } else {
            let question = question.clone();
            commands.trigger(AskedUser { call, question });
        }
        return;
    }
    if !question.free_text && !question.choices.iter().any(|choice| choice == text) {
        let choices = question.choices.join(", ");
        notices.write(Notice::error(
            args.agent,
            format!("Answer one of: {choices}."),
        ));
        return;
    }
    let text = text.to_owned();
    commands.queue(move |world: &mut World| settle(world, call, &text));
}

/// What the user sends to the asking agent, or to an agent it was spawned
/// by, while a free-text question is shown answers it instead of going to
/// the model.
fn answer_typed(
    mut typed: On<Deliver>,
    shown: Res<Shown>,
    questions: Query<&Question>,
    parents: Query<&SpawnedBy>,
    mut commands: Commands,
) {
    if typed.origin.kind != OriginKind::User
        || typed.command().is_some()
        || typed.text.trim().is_empty()
    {
        return;
    }
    let Some(call) = shown.call else {
        return;
    };
    let Ok(question) = questions.get(call) else {
        return;
    };
    let to_asker = question.agent == typed.entity
        || parents
            .iter_ancestors(question.agent)
            .any(|agent| agent == typed.entity);
    if !question.free_text || !to_asker {
        return;
    }
    let text = std::mem::take(&mut typed.text);
    commands.queue(move |world: &mut World| settle(world, call, text.trim()));
}

#[cfg(test)]
mod tests;
