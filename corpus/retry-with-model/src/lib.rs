//! Plugin corpus: 17. Retry with another model (spec `17-retry-with-model.md`).
//!
//! `/retry-with [--force] <vendor/model>` rewinds the agent's conversation
//! to the last turn's user message, keeps the answer it takes out as an
//! alternative, and runs that turn again on the model named, for this turn
//! only: the agent's own [`ModelChoice`] stays. `/alternatives` lists the
//! kept answers and `/pick N` swaps the conversation to one of them. The
//! conversation changes through [`Commit`], so the session log keeps them,
//! and the alternatives are saved with the session.

use rig_harness::prelude::*;
use rig_harness::rig_core::message::Message;
use rig_harness::rig_core::tool::FilesChanged;
use rig_tui::Picker;
use serde::{Deserialize, Serialize};

/// How long an answer's excerpt in `/alternatives` is, in characters.
const EXCERPT: usize = 80;

#[derive(Default)]
pub struct RetryWithModelPlugin;

impl Plugin for RetryWithModelPlugin {
    fn build(&self, app: &mut App) {
        app.register_required_components::<Agent, Alternatives>()
            .register_required_components::<TurnOf, TurnChanges>()
            .add_observer(note_change)
            .add_observer(note_turn)
            .add_systems(AgentPass, route.in_set(Stage::Prepare))
            .add_command(
                "retry-with",
                "Run the last turn again on another model: /retry-with [--force] vendor/model",
                retry_with,
            )
            .add_command(
                "alternatives",
                "List the answers /retry-with replaced",
                alternatives,
            )
            .add_command(
                "pick",
                "Swap the conversation to an answer /alternatives lists: /pick N",
                pick,
            );
    }
}

/// The agent's last turn, the answer the conversation holds to it, and the
/// answers `/retry-with` replaced. Saved with the session.
#[derive(Component, Reflect, Clone, Default, Serialize, Deserialize)]
#[reflect(opaque, Component, Saved, Default, Clone, Serialize, Deserialize)]
pub struct Alternatives {
    /// The index of the user message the last turn answered.
    pub turn: Option<usize>,
    /// The model of the answer the conversation holds.
    pub model: String,
    /// What that answer's tool calls changed.
    pub changed: Vec<String>,
    /// The replaced answers, oldest first.
    pub kept: Vec<Alternative>,
}

impl Alternatives {
    /// The answer to the user message at `turn` that `conversation` holds,
    /// taken out of these as an alternative.
    fn take_current(&mut self, conversation: &Conversation, turn: usize) -> Alternative {
        Alternative {
            model: std::mem::take(&mut self.model),
            changed: std::mem::take(&mut self.changed),
            messages: conversation
                .messages()
                .get(turn + 1..)
                .unwrap_or_default()
                .to_vec(),
        }
    }
}

/// One answer to the last turn: every message after its user message.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Alternative {
    /// The model that answered, `vendor/model`.
    pub model: String,
    /// What its tool calls changed.
    pub changed: Vec<String>,
    /// The answer's messages, the tool calls and results included.
    pub messages: Vec<Message>,
}

impl Alternative {
    /// The start of the answer's last text, on one line.
    fn excerpt(&self) -> String {
        let text = self.messages.iter().rev().find_map(final_answer);
        let line = text
            .unwrap_or_default()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        let mut excerpt: String = line.chars().take(EXCERPT).collect();
        if line.chars().count() > EXCERPT {
            excerpt.push('…');
        }
        excerpt
    }
}

/// On a turn: what its tool calls changed, files by path, or a tool that
/// may change anything and named no files.
#[derive(Component, Default)]
struct TurnChanges(Vec<String>);

/// On a turn `/retry-with` started: the model its calls go to.
#[derive(Component)]
struct RetryOn(Connection);

/// Notes on its turn what each tool call that ran changed.
fn note_change(
    settled: On<Add<Settled>>,
    calls: Query<(&CallOf, &ToolRequest, &Footprint, Option<&Published>), With<Started>>,
    mut turns: Query<&mut TurnChanges>,
) {
    let Ok((&CallOf(turn), request, footprint, published)) = calls.get(settled.entity) else {
        return;
    };
    let Ok(mut changed) = turns.get_mut(turn) else {
        return;
    };
    let files = published.and_then(|published| published.0.get::<FilesChanged>().ok().flatten());
    match files {
        Some(files) => changed.0.extend(files.paths().map(str::to_owned)),
        None if *footprint == Footprint::Exclusive => {
            changed.0.push(format!(
                "whatever `{}` changed",
                request.call.function.name.as_str()
            ));
        }
        None => {}
    }
}

/// Notes each answered turn on its agent: which message it answered, on
/// which model, and what it changed. A turn that answers another message
/// than the last drops the last one's alternatives.
fn note_turn(
    settled: On<Add<Settled>>,
    turns: Query<(
        &Turn,
        &TurnOutcome,
        &ForAgent,
        &TurnChanges,
        Option<&RetryOn>,
    )>,
    mut agents: Query<(&Conversation, &mut Alternatives, Option<&Connection>)>,
) {
    let Ok((turn, outcome, &ForAgent(agent), changed, retry)) = turns.get(settled.entity) else {
        return;
    };
    let Ok((conversation, mut alternatives, connection)) = agents.get_mut(agent) else {
        return;
    };
    let asked = conversation.messages().get(turn.first_message);
    if !matches!(outcome, TurnOutcome::Answered(_)) || !matches!(asked, Some(Message::User { .. }))
    {
        return;
    }
    if retry.is_none() && alternatives.turn != Some(turn.first_message) {
        alternatives.kept.clear();
    }
    alternatives.turn = Some(turn.first_message);
    alternatives.model = retry
        .map(|RetryOn(connection)| connection)
        .or(connection)
        .map(|connection| connection.spec.reference())
        .unwrap_or_default();
    alternatives.changed.clone_from(&changed.0);
}

/// Sends the calls of a turn `/retry-with` started to its model.
fn route(
    mut calls: Query<(&mut ModelRequest, &CallOf), (Added<Admitted>, With<TurnCall>)>,
    turns: Query<&RetryOn>,
) {
    for (mut call, &CallOf(turn)) in &mut calls {
        if let Ok(RetryOn(connection)) = turns.get(turn) {
            call.connection = Some(connection.clone());
        }
    }
}

/// The index of the last turn's user message, when the conversation still
/// holds it.
fn last_turn(conversation: &Conversation, alternatives: &Alternatives) -> Option<usize> {
    let turn = alternatives.turn?;
    matches!(
        conversation.messages().get(turn),
        Some(Message::User { .. })
    )
    .then_some(turn)
}

/// A picker of `models`, each running `/retry-with` on it.
fn models_picker(agent: Entity, force: bool, models: &[String]) -> Picker {
    let flag = if force { "--force " } else { "" };
    models.iter().fold(
        Picker::new("Retry the last turn on").for_agent(agent),
        |picker, model| picker.item(model, format!("retry-with {flag}{model}")),
    )
}

type Agents<'w, 's> = Query<
    'w,
    's,
    (
        &'static mut Conversation,
        &'static mut Alternatives,
        Has<ActiveTurn>,
    ),
>;

/// `/retry-with [--force] <vendor/model>`.
fn retry_with(
    In(args): In<CommandArgs>,
    mut agents: Agents,
    models: Res<Models>,
    mut commit: Commit,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let words: Vec<&str> = args.args.split_whitespace().collect();
    let force = words.contains(&"--force");
    let named: Vec<&str> = words
        .into_iter()
        .filter(|word| *word != "--force")
        .collect();
    let Ok((mut conversation, mut alternatives, busy)) = agents.get_mut(agent) else {
        return;
    };
    if busy {
        notices.write(Notice::error(agent, "A turn is running; interrupt first."));
        return;
    }
    let [model] = named.as_slice() else {
        let reachable: Vec<String> = models
            .0
            .reachable()
            .iter()
            .map(|spec| spec.reference())
            .collect();
        commands.spawn(models_picker(agent, force, &reachable));
        notices.write(Notice::error(
            agent,
            "Usage: /retry-with [--force] vendor/model",
        ));
        return;
    };
    if let Err(missed) = models.0.catalog().resolve(model) {
        commands.spawn(models_picker(agent, force, &missed.suggestions));
        notices.write(Notice::error(agent, format!("Cannot retry: {missed}.")));
        return;
    }
    let connection = match models.connect(model) {
        Ok(connection) => connection,
        Err(why) => {
            notices.write(Notice::error(agent, why));
            return;
        }
    };
    let Some(turn) = last_turn(&conversation, &alternatives) else {
        notices.write(Notice::error(agent, "There is no last turn to retry."));
        return;
    };
    if !alternatives.changed.is_empty() {
        let changed = alternatives.changed.join(", ");
        if !force {
            notices.write(Notice::error(
                agent,
                format!(
                    "The last turn changed files ({changed}), which a retry does not roll back; \
                     add --force to retry anyway."
                ),
            ));
            return;
        }
        notices.write(Notice::info(agent, format!("Not rolled back: {changed}.")));
    }
    let current = alternatives.take_current(&conversation, turn);
    if !current.messages.is_empty() {
        alternatives.kept.push(current);
    }
    commit.truncate(agent, &mut conversation, turn + 1);
    notices.write(Notice::info(
        agent,
        format!("Retrying the last turn on {}.", connection.spec.reference()),
    ));
    let retry = commands.spawn((TurnOf(agent), RetryOn(connection))).id();
    commands.spawn((TurnCall, CallOf(retry), ForAgent(agent)));
}

/// `/alternatives`.
fn alternatives(
    In(args): In<CommandArgs>,
    agents: Query<&Alternatives>,
    mut notices: MessageWriter<Notice>,
) {
    let kept = agents
        .get(args.agent)
        .map(|alternatives| alternatives.kept.as_slice());
    let lines: Vec<String> = kept
        .unwrap_or_default()
        .iter()
        .enumerate()
        .map(|(at, kept)| format!("{}. {}: {}", at + 1, kept.model, kept.excerpt()))
        .collect();
    let text = if lines.is_empty() {
        "No alternatives to the last turn.".to_owned()
    } else {
        format!(
            "Alternatives to the last turn (/pick N):\n{}",
            lines.join("\n")
        )
    };
    notices.write(Notice::info(args.agent, text));
}

/// `/pick N`: the conversation takes alternative N, which takes the
/// answer it replaces.
fn pick(
    In(args): In<CommandArgs>,
    mut agents: Agents,
    mut commit: Commit,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let Ok((mut conversation, mut alternatives, busy)) = agents.get_mut(agent) else {
        return;
    };
    if busy {
        notices.write(Notice::error(agent, "A turn is running; interrupt first."));
        return;
    }
    let count = alternatives.kept.len();
    let index = args
        .args
        .trim()
        .parse::<usize>()
        .ok()
        .and_then(|n| n.checked_sub(1));
    let Some(index) = index.filter(|&index| index < count) else {
        notices.write(Notice::error(
            agent,
            format!("Pick an alternative from 1 to {count}."),
        ));
        return;
    };
    let Some(turn) = last_turn(&conversation, &alternatives) else {
        notices.write(Notice::error(agent, "The last turn is gone."));
        return;
    };
    let current = alternatives.take_current(&conversation, turn);
    let Some(slot) = alternatives.kept.get_mut(index) else {
        return;
    };
    let chosen = std::mem::replace(slot, current);
    commit.truncate(agent, &mut conversation, turn + 1);
    for message in chosen.messages {
        commit.message(agent, &mut conversation, message);
    }
    notices.write(Notice::info(
        agent,
        format!("Picked the answer of {}.", chosen.model),
    ));
    alternatives.model = chosen.model;
    alternatives.changed = chosen.changed;
}

#[cfg(test)]
mod tests;
