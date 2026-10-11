//! Plugin corpus: 17. Retry with another model (spec `17-retry-with-model.md`).
//!
//! `/retry-with [--force] <vendor/model>` takes the last turn's replies out
//! of the agent's conversation, keeps them as an alternative, and runs the
//! turn again on that model, for this turn only: the agent's own
//! [`ModelChoice`] stays. `/alternatives` lists the kept answers and
//! `/pick N` swaps the conversation to one of them. The alternatives are
//! saved with the session.

use rig_harness::prelude::*;
use rig_harness::rig_core::message::Message;
use serde::{Deserialize, Serialize};

/// How long an answer's excerpt in `/alternatives` is, in characters.
const EXCERPT: usize = 80;

#[derive(Default)]
pub struct RetryWithModelPlugin;

impl Plugin for RetryWithModelPlugin {
    fn build(&self, app: &mut App) {
        app.register_required_components::<Agent, Alternatives>()
            .add_observer(track_turn)
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

/// The agent's last turn and the answers to it that `/retry-with`
/// replaced. Saved with the session.
#[derive(Component, Reflect, Clone, Default, Serialize, Deserialize)]
#[reflect(opaque, Component, Saved, Default, Clone, Serialize, Deserialize)]
pub struct Alternatives {
    /// The index of the user message the last turn answered.
    pub turn: Option<usize>,
    /// The model of the answer the conversation holds now.
    pub model: String,
    /// The replaced answers, oldest first.
    pub kept: Vec<Alternative>,
}

/// One answer to the last turn: every message after its user message.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Alternative {
    /// The model that answered, `vendor/model`.
    pub model: String,
    /// The answer's messages, the tool calls and results included.
    pub messages: Vec<Message>,
}

impl Alternative {
    /// The start of the answer's last text, on one line.
    fn excerpt(&self) -> String {
        let text = self
            .messages
            .iter()
            .rev()
            .find_map(answer_text)
            .unwrap_or_default();
        let line = text.split_whitespace().collect::<Vec<_>>().join(" ");
        let mut excerpt: String = line.chars().take(EXCERPT).collect();
        if line.chars().count() > EXCERPT {
            excerpt.push('…');
        }
        excerpt
    }
}

/// The text of an assistant message, when it has any.
fn answer_text(message: &Message) -> Option<String> {
    let Message::Assistant(assistant) = message else {
        return None;
    };
    let text: String = assistant
        .content
        .iter()
        .filter_map(|item| match item {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect();
    (!text.trim().is_empty()).then_some(text)
}

/// On an agent whose next turn runs on another model: that model.
#[derive(Component)]
struct RetryOn(Connection);

/// Notes which user message each turn answers, and gives a retried turn
/// its model. A turn that answers a new message drops the alternatives of
/// the one before.
fn track_turn(
    added: On<Add<TurnOf>>,
    turns: Query<&TurnOf>,
    mut agents: Query<(
        &Conversation,
        &mut Alternatives,
        Option<&RetryOn>,
        Option<&ModelChoice>,
    )>,
    mut commands: Commands,
) {
    let Ok(&TurnOf(agent)) = turns.get(added.entity) else {
        return;
    };
    let Ok((conversation, mut alternatives, retry, choice)) = agents.get_mut(agent) else {
        return;
    };
    if let Some(RetryOn(connection)) = retry {
        alternatives.model = connection.spec.reference();
        commands.entity(added.entity).insert(connection.clone());
        commands.entity(agent).remove::<RetryOn>();
        return;
    }
    alternatives.model = choice.map(|choice| choice.0.clone()).unwrap_or_default();
    let messages = conversation.messages();
    let asked = messages.len().checked_sub(1).filter(|&last| {
        messages.get(last).is_some_and(|message| {
            matches!(message, Message::User { content }
                if content.iter().any(|item| !matches!(item, UserContent::ToolResult(_))))
        })
    });
    if asked.is_some() && asked != alternatives.turn {
        alternatives.turn = asked;
        alternatives.kept.clear();
    }
}

/// The last turn's user message index and its answer, when the
/// conversation still ends with that turn.
fn last_answer(conversation: &Conversation, alternatives: &Alternatives) -> Option<usize> {
    let turn = alternatives.turn?;
    let asked = conversation.messages().get(turn)?;
    matches!(asked, Message::User { .. }).then_some(turn)
}

/// `conversation` with its messages after `turn` replaced by `answer`.
///
/// The workaround: the kernel has no public way to take messages out of a
/// conversation (`Commit::retract` is crate-private), so this rebuilds the
/// component through its serde form. The change is not logged: a restored
/// session folds its log back to every answer in a row.
fn rewritten(
    conversation: &Conversation,
    turn: usize,
    answer: Vec<Message>,
) -> Option<Conversation> {
    let mut value = serde_json::to_value(conversation).ok()?;
    let object = value.as_object_mut()?;
    let mut messages: Vec<Message> = conversation.messages().get(..=turn)?.to_vec();
    messages.extend(answer);
    let kept = turn + 1;
    let origins = object
        .get("origins")
        .and_then(serde_json::Value::as_array)
        .map(|origins| {
            origins
                .iter()
                .filter(|origin| {
                    origin
                        .get(0)
                        .and_then(serde_json::Value::as_u64)
                        .is_some_and(|at| at < kept as u64)
                })
                .cloned()
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    object.insert("messages".to_owned(), serde_json::to_value(messages).ok()?);
    object.insert("origins".to_owned(), serde_json::Value::Array(origins));
    object.insert("halted".to_owned(), serde_json::Value::Null);
    serde_json::from_value(value).ok()
}

/// Whether a tool of `name` may change files: any tool not registered as
/// changing nothing.
fn may_change(name: &str, tools: &Query<(&ToolDef, &Footprint)>) -> bool {
    tools
        .iter()
        .find(|(def, _)| def.0.name.as_str() == name)
        .is_none_or(|(_, footprint)| *footprint == Footprint::Exclusive)
}

/// `/retry-with [--force] <vendor/model>`.
fn retry_with(
    In(args): In<CommandArgs>,
    mut agents: Query<(&mut Conversation, &mut Alternatives, Has<ActiveTurn>)>,
    tools: Query<(&ToolDef, &Footprint)>,
    models: Res<Models>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let words: Vec<&str> = args.args.split_whitespace().collect();
    let force = words.contains(&"--force");
    let mut named = words.iter().filter(|word| **word != "--force");
    let (Some(model), None) = (named.next(), named.next()) else {
        notices.write(Notice::error(
            agent,
            "Usage: /retry-with [--force] vendor/model",
        ));
        return;
    };
    let Ok((mut conversation, mut alternatives, busy)) = agents.get_mut(agent) else {
        return;
    };
    if busy {
        notices.write(Notice::error(agent, "A turn is running; interrupt first."));
        return;
    }
    let connection = match models.connect(model) {
        Ok(connection) => connection,
        Err(why) => {
            notices.write(Notice::error(agent, why));
            return;
        }
    };
    let Some(turn) = last_answer(&conversation, &alternatives) else {
        notices.write(Notice::error(agent, "There is no last turn to retry."));
        return;
    };
    let answer: Vec<Message> = conversation
        .messages()
        .get(turn + 1..)
        .unwrap_or_default()
        .to_vec();
    let changing: Vec<String> = answer
        .iter()
        .filter_map(|message| match message {
            Message::Assistant(assistant) => Some(assistant.tool_calls()),
            _ => None,
        })
        .flatten()
        .map(|call| call.function.name.as_str().to_owned())
        .filter(|name| may_change(name, &tools))
        .collect();
    if !changing.is_empty() && !force {
        notices.write(Notice::error(
            agent,
            format!(
                "The last turn ran {} and may have changed files, which a retry does not \
                 roll back; add --force to retry anyway.",
                changing.join(", ")
            ),
        ));
        return;
    }
    let Some(rewound) = rewritten(&conversation, turn, Vec::new()) else {
        notices.write(Notice::error(agent, "Could not rewind the conversation."));
        return;
    };
    if !changing.is_empty() {
        notices.write(Notice::info(
            agent,
            "Files the last turn changed are not rolled back.",
        ));
    }
    if !answer.is_empty() {
        let model = std::mem::take(&mut alternatives.model);
        alternatives.kept.push(Alternative {
            model,
            messages: answer,
        });
    }
    *conversation = rewound;
    notices.write(Notice::info(
        agent,
        format!("Retrying the last turn on {}.", connection.spec.reference()),
    ));
    commands.entity(agent).insert(RetryOn(connection));
    commands.trigger(Retry { entity: agent });
}

/// `/alternatives`.
fn alternatives(
    In(args): In<CommandArgs>,
    agents: Query<&Alternatives>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(alternatives) = agents.get(args.agent) else {
        return;
    };
    if alternatives.kept.is_empty() {
        notices.write(Notice::info(
            args.agent,
            "No alternatives to the last turn.",
        ));
        return;
    }
    let lines: Vec<String> = alternatives
        .kept
        .iter()
        .enumerate()
        .map(|(at, kept)| format!("{}. {}: {}", at + 1, kept.model, kept.excerpt()))
        .collect();
    notices.write(Notice::info(
        args.agent,
        format!(
            "Alternatives to the last turn (/pick N):\n{}",
            lines.join("\n")
        ),
    ));
}

/// `/pick N`: the conversation takes alternative N, which takes the
/// answer it replaces.
fn pick(
    In(args): In<CommandArgs>,
    mut agents: Query<(&mut Conversation, &mut Alternatives, Has<ActiveTurn>)>,
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
    let chosen = args
        .args
        .parse::<usize>()
        .ok()
        .and_then(|n| n.checked_sub(1));
    let Some(index) = chosen.filter(|&index| index < count) else {
        notices.write(Notice::error(
            agent,
            format!("Pick an alternative from 1 to {count}."),
        ));
        return;
    };
    let Some(turn) = last_answer(&conversation, &alternatives) else {
        notices.write(Notice::error(agent, "The last turn is gone."));
        return;
    };
    let Some(slot) = alternatives.kept.get(index).cloned() else {
        return;
    };
    let Some(swapped) = rewritten(&conversation, turn, slot.messages) else {
        notices.write(Notice::error(agent, "Could not swap the conversation."));
        return;
    };
    let current = Alternative {
        model: std::mem::replace(&mut alternatives.model, slot.model.clone()),
        messages: conversation
            .messages()
            .get(turn + 1..)
            .unwrap_or_default()
            .to_vec(),
    };
    if let Some(kept) = alternatives.kept.get_mut(index) {
        *kept = current;
    }
    *conversation = swapped;
    notices.write(Notice::info(
        agent,
        format!("Picked the answer of {}.", slot.model),
    ));
}

#[cfg(test)]
mod tests;
