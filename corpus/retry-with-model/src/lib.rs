//! Plugin corpus: 17. Retry with another model (spec `17-retry-with-model.md`).
//!
//! `/retry-with [--force] <vendor/model>` rewinds the agent's conversation
//! to the last turn's user message, keeps the answer it takes out as an
//! alternative, and runs that turn again on the model named, for this turn
//! only: the agent's own [`ModelChoice`] stays. `/alternatives` lists the
//! kept answers and `/pick N` swaps the conversation to one of them. The
//! conversation changes are splices by this plugin, so the session log
//! keeps them, and the alternatives are saved with the session. The input completes
//! `/retry-with`'s argument to the models the agent can reach.

use rig_harness::prelude::*;
use rig_harness::rig_core::tool::FilesChanged;
use rig_tui::CompletionSource;
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
            .configure_sets(
                AgentPass,
                RetryWithModelSystems.in_set(PrepareSystems::Edit),
            )
            .add_systems(AgentPass, route.in_set(RetryWithModelSystems))
            .add_systems(Update, complete.run_if(resource_changed::<Models>))
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

/// The plugin's stage system, in [`Stage::Prepare`]: it routes a retried
/// turn's calls, after the default prompt went in and before compaction
/// measures them.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RetryWithModelSystems;

/// The agent's last turn, the answer the conversation holds to it, and the
/// answers `/retry-with` replaced. Saved with the session.
#[derive(Component, Reflect, Clone, Default, Serialize, Deserialize)]
#[reflect(opaque, Component, Saved, Default, Clone, Serialize, Deserialize)]
pub struct Alternatives {
    /// The user message the last turn answered.
    pub turn: Option<MessageId>,
    /// The model of the answer the conversation holds.
    pub model: String,
    /// What that answer's tool calls changed.
    pub changed: Vec<String>,
    /// The replaced answers, oldest first.
    pub kept: Vec<Alternative>,
}

impl Alternatives {
    /// The answer that `conversation` holds from `answer` on, taken out of
    /// these as an alternative.
    fn take_current(&mut self, conversation: &Conversation, answer: usize) -> Alternative {
        Alternative {
            model: std::mem::take(&mut self.model),
            changed: std::mem::take(&mut self.changed),
            messages: (answer..conversation.messages().len())
                .filter_map(|at| conversation.entry(at))
                .collect(),
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
    /// The answer's messages, the tool calls and results included, with
    /// their authors and ids.
    pub messages: Vec<Entry>,
}

impl Alternative {
    /// The start of the answer's last text, on one line.
    fn excerpt(&self) -> String {
        let text = self
            .messages
            .iter()
            .rev()
            .find_map(|entry| entry.message.final_answer());
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
        None if matches!(footprint.0, Touches::Writes(_)) => {
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
    let asked = conversation.request(turn.start).is_some();
    if !matches!(outcome, TurnOutcome::Answered(_)) || !asked {
        return;
    }
    if retry.is_none() && alternatives.turn != Some(turn.start) {
        alternatives.kept.clear();
    }
    alternatives.turn = Some(turn.start);
    alternatives.model = retry
        .map(|RetryOn(connection)| connection)
        .or(connection)
        .map(|connection| connection.spec.reference())
        .unwrap_or_default();
    alternatives.changed.clone_from(&changed.0);
}

/// Sends the calls of a turn `/retry-with` started to its model.
fn route(
    mut calls: Query<(&mut ModelRequest, &CallOf), (Added<Admitted>, ModelCalls, With<TurnCall>)>,
    turns: Query<&RetryOn>,
) {
    for (mut call, &CallOf(turn)) in &mut calls {
        if let Ok(RetryOn(connection)) = turns.get(turn) {
            call.connection = Some(connection.clone());
        }
    }
}

/// On the entity of `/retry-with`'s completions.
#[derive(Component)]
struct RetryCompletions;

/// Offers `--force` and the models the agent can reach as `/retry-with`'s
/// argument.
fn complete(
    models: Res<Models>,
    completions: Query<Entity, With<RetryCompletions>>,
    mut commands: Commands,
) {
    let force = CompletionSource::arg("retry-with").candidate("--force", "even if files changed");
    let completion = models
        .0
        .reachable()
        .into_iter()
        .fold(force, |completion, spec| {
            completion.candidate(spec.reference(), spec.display_name.clone())
        });
    match completions.iter().next() {
        Some(entity) => {
            commands.entity(entity).insert(completion);
        }
        None => {
            commands.spawn((
                Name::new("completions:retry-with"),
                RetryCompletions,
                completion,
            ));
        }
    }
}

/// Where the answer to the last turn starts: after the user entries the
/// turn answered, when the conversation still holds them.
fn last_turn(conversation: &Conversation, alternatives: &Alternatives) -> Option<usize> {
    let turn = alternatives.turn?;
    conversation.index_of(turn)?;
    let asked = conversation.asked(turn).last()?;
    asked.index.checked_add(1)
}

/// The author of this plugin's edits.
fn author() -> Author {
    Author::plugin("retry-with-model")
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
        notices.write(Notice::error(
            agent,
            "Usage: /retry-with [--force] vendor/model",
        ));
        return;
    };
    if let Err(missed) = models.0.catalog().resolve(model) {
        notices.write(Notice::error(agent, format!("Cannot retry: {missed}")));
        return;
    }
    let connection = match models.connect(model) {
        Ok(connection) => connection,
        Err(why) => {
            notices.write(Notice::error(agent, why));
            return;
        }
    };
    let Some(answer) = last_turn(&conversation, &alternatives) else {
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
    let current = alternatives.take_current(&conversation, answer);
    if !current.messages.is_empty() {
        alternatives.kept.push(current);
    }
    conversation.splice(answer.., [], author());
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
fn pick(In(args): In<CommandArgs>, mut agents: Agents, mut notices: MessageWriter<Notice>) {
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
    let Some(answer) = last_turn(&conversation, &alternatives) else {
        notices.write(Notice::error(agent, "The last turn is gone."));
        return;
    };
    let current = alternatives.take_current(&conversation, answer);
    let Some(slot) = alternatives.kept.get_mut(index) else {
        return;
    };
    let chosen = std::mem::replace(slot, current);
    conversation.splice(answer.., chosen.messages, author());
    notices.write(Notice::info(
        agent,
        format!("Picked the answer of {}.", chosen.model),
    ));
    alternatives.model = chosen.model;
    alternatives.changed = chosen.changed;
}

#[cfg(test)]
mod tests;
