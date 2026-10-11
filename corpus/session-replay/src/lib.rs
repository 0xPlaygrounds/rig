//! Replays a recorded session: the user's messages are sent again in
//! order, and every model call, tool call and plugin effect of a kind the
//! recording has is answered from the recorded effect log before it starts,
//! in Gate, instead of by the provider or the tool. Each model request is
//! checked against the recorded one at the same position of the same
//! agent; the first difference stops the replay, or with `continue_live`
//! is reported and the run goes on live, its calls left to start as usual.
//!
//! The replay is started by a [`ReplaySource`] resource inserted before the
//! plugin is added, or by the `RIG_REPLAY` environment variable naming a
//! session directory (with `RIG_REPLAY_FLAGS`, a comma list of
//! `continue-live`, `live-tools` and `speed`). Without either the plugin
//! does nothing. Nothing recorded is written to.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::io;
use std::path::Path;
use std::sync::Arc;

use rig_cassette::effect_log::{EffectLog, RecordedAnswers, RequestCheck, jsonl};
use rig_cassette::journal::EFFECT_LOG;
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::CompletionRequest;
use rig_harness::rig_core::effect::{self, EffectKind, EffectRecord, HandlerKey};
use rig_harness::rig_core::error::{ErrorKind, ErrorReport};
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// How a replay runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReplayFlags {
    /// After the first difference, report it and go on with the live model
    /// and tools instead of stopping.
    pub continue_live: bool,
    /// Run the tools for real; the model replies still come from the record.
    pub live_tools: bool,
    /// Send the user messages with their original gaps instead of at once.
    pub original_timing: bool,
}

impl ReplayFlags {
    /// The flags of a comma list such as `continue-live,live-tools,speed`.
    pub fn parse(list: &str) -> Self {
        let words: Vec<&str> = list.split(',').map(str::trim).collect();
        Self {
            continue_live: words.contains(&"continue-live"),
            live_tools: words.contains(&"live-tools"),
            original_timing: words.contains(&"speed"),
        }
    }
}

/// The recorded session to replay: its effect log, which holds every
/// effect its agents settled, the user's messages included.
#[derive(Resource, Clone)]
pub struct ReplaySource {
    /// Every recorded effect.
    pub effects: EffectLog,
    /// How to replay.
    pub flags: ReplayFlags,
}

impl ReplaySource {
    /// The session kept in the directory `dir`, read only.
    pub fn from_dir(dir: &Path, flags: ReplayFlags) -> io::Result<Self> {
        Ok(Self {
            effects: jsonl::read(dir.join(EFFECT_LOG))?,
            flags,
        })
    }

    /// The source `RIG_REPLAY` and `RIG_REPLAY_FLAGS` name, if any.
    fn from_env() -> Option<io::Result<Self>> {
        let dir = std::env::var_os("RIG_REPLAY")?;
        let flags = std::env::var("RIG_REPLAY_FLAGS").unwrap_or_default();
        Some(Self::from_dir(Path::new(&dir), ReplayFlags::parse(&flags)))
    }
}

/// The first difference between a replayed effect and the recorded one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Divergence {
    /// The recorded agent's id.
    pub agent: String,
    /// The position of the model call among the agent's, from 0: the
    /// request that differed, or the one whose reply asked for the effect.
    pub position: usize,
    /// What differs, one line each, such as `system prompt: + line`.
    pub differences: Vec<String>,
}

impl fmt::Display for Divergence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Replay diverged at model call {} of agent {}:",
            self.position + 1,
            self.agent.get(..8).unwrap_or(&self.agent)
        )?;
        for difference in &self.differences {
            write!(f, "\n  {difference}")?;
        }
        Ok(())
    }
}

/// Where a replay is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReplayStatus {
    /// Waiting for the session to be restored.
    #[default]
    Waiting,
    /// Sending the user's messages.
    Running,
    /// Every user message was replayed.
    Finished,
    /// Stopped at the first divergence.
    Stopped,
}

/// A user message of the recording, and when it was accepted.
#[derive(Clone, Debug)]
struct Said {
    text: String,
    at_ms: u64,
}

/// A recorded message's payload.
#[derive(Deserialize)]
struct Delivered {
    text: String,
    origin: Origin,
}

/// The running replay.
#[derive(Resource)]
pub struct Replay {
    flags: ReplayFlags,
    /// The model calls, answered in recorded order per agent and compared
    /// here: requests carry values of the session, such as cache keys.
    models: RecordedAnswers,
    /// Every other effect, answered by the record its request matches.
    others: RecordedAnswers,
    /// The kinds of effect the recording answers, other than model calls.
    kinds: HashSet<String>,
    /// The recorded agents, in order of their first effect.
    scopes: Vec<Option<Arc<str>>>,
    /// Each replaying agent's recorded agent, mapped at its first effect.
    agents: HashMap<Entity, Option<Arc<str>>>,
    /// How many model calls of each recorded agent were taken.
    positions: HashMap<Option<Arc<str>>, usize>,
    /// The user messages not sent yet.
    messages: VecDeque<Said>,
    primary: Option<Entity>,
    /// When the last message was sent: originally, and now.
    last_sent: Option<(u64, Duration)>,
    status: ReplayStatus,
    divergences: Vec<Divergence>,
    /// Whether the run went live after a divergence.
    live: bool,
    /// Model calls and other effects answered from the record.
    answered: (usize, usize),
}

impl Replay {
    fn new(source: ReplaySource) -> Self {
        let ReplaySource { mut effects, flags } = source;
        effects.records.sort_by_key(|record| record.id);
        let mut scopes = Vec::new();
        for record in &effects.records {
            if !scopes.contains(&record.scope) {
                scopes.push(record.scope.clone());
            }
        }
        let primary = scopes.first().cloned().flatten();
        let messages = effects
            .records
            .iter()
            .filter(|record| record.scope == primary)
            .filter_map(said)
            .collect();
        let (models, others): (Vec<EffectRecord>, Vec<EffectRecord>) = effects
            .records
            .into_iter()
            .filter(|record| !is_message(&record.kind))
            .partition(|record| matches!(record.kind, EffectKind::Completion { .. }));
        let kinds = others
            .iter()
            .map(|record| record.kind.name().to_owned())
            .collect();
        let answers = |records| RecordedAnswers::new(&EffectLog::from_records(records));
        Self {
            flags,
            models: answers(models).checking(RequestCheck::Skip),
            others: answers(others),
            kinds,
            scopes,
            agents: HashMap::new(),
            positions: HashMap::new(),
            messages,
            primary: None,
            last_sent: None,
            status: ReplayStatus::Waiting,
            divergences: Vec::new(),
            live: false,
            answered: (0, 0),
        }
    }

    /// Where the replay is.
    pub fn status(&self) -> ReplayStatus {
        self.status
    }

    /// Every divergence found, first first.
    pub fn divergences(&self) -> Vec<Divergence> {
        self.divergences.clone()
    }

    /// The model calls and the other effects answered from the record.
    pub fn answered(&self) -> (usize, usize) {
        self.answered
    }

    /// The recorded agent `agent` replays, mapped at its first effect to
    /// the first recorded agent not mapped yet.
    fn scope_of(&mut self, agent: Entity) -> Option<Arc<str>> {
        if let Some(scope) = self.agents.get(&agent) {
            return scope.clone();
        }
        let taken: HashSet<&Option<Arc<str>>> = self.agents.values().collect();
        let scope = self
            .scopes
            .iter()
            .find(|scope| !taken.contains(scope))
            .cloned()
            .flatten();
        self.agents.insert(agent, scope.clone());
        scope
    }

    /// The answer to the model call `request`, sent to `key`, of the
    /// recorded agent `scope`, or why it diverged.
    fn model(
        &mut self,
        scope: Option<&str>,
        (key, request): (&HandlerKey, &CompletionRequest),
    ) -> Result<EffectRecord, Divergence> {
        let kind = EffectKind::Completion {
            request: request.clone(),
            stream: true,
        };
        let position = self.positions.entry(scope.map(Arc::from)).or_default();
        let at = *position;
        *position += 1;
        let differences = match self.models.answer(scope, &kind) {
            Ok(record) => match &record.kind {
                EffectKind::Completion { request: was, .. } => {
                    let found = differences((&record.key, was), (key, request));
                    if found.is_empty() {
                        return Ok(record);
                    }
                    found
                }
                _ => vec!["a model call arrived where another effect was recorded".to_owned()],
            },
            Err(_) => vec!["the recording has no more model calls for this agent".to_owned()],
        };
        Err(Divergence {
            agent: scope.unwrap_or_default().to_owned(),
            position: at,
            differences,
        })
    }

    /// The answer to `kind`, an effect of the recorded agent `scope` other
    /// than a model call, or why it diverged.
    fn other(
        &mut self,
        scope: Option<&str>,
        kind: &EffectKind,
    ) -> Result<EffectRecord, Divergence> {
        self.others.answer(scope, kind).map_err(|report| {
            let made = self.positions.get(&scope.map(Arc::from)).copied();
            Divergence {
                agent: scope.unwrap_or_default().to_owned(),
                position: made.unwrap_or_default().saturating_sub(1),
                differences: vec![report.message],
            }
        })
    }
}

/// Whether `kind` is a delivered message's.
fn is_message(kind: &EffectKind) -> bool {
    matches!(kind, EffectKind::Custom { kind, .. } if &**kind == "message")
}

/// The user's message `record` holds, if it was delivered.
fn said(record: &EffectRecord) -> Option<Said> {
    let EffectKind::Custom { kind, payload } = &record.kind else {
        return None;
    };
    let delivered = matches!(&record.outcome,
        Ok(effect::Outcome::Custom { payload }) if payload == "delivered");
    if &**kind != "message" || !delivered {
        return None;
    }
    let message: Delivered = serde_json::from_value(payload.clone()).ok()?;
    (message.origin.kind == OriginKind::User).then(|| Said {
        text: message.text,
        at_ms: record.started_ms.unwrap_or_default(),
    })
}

/// What differs between the recorded request and the new one, each sent
/// to the handler of its key; empty when they match.
fn differences(
    (was_key, was): (&HandlerKey, &CompletionRequest),
    (key, now): (&HandlerKey, &CompletionRequest),
) -> Vec<String> {
    let mut found = Vec::new();
    if was_key != key || was.model != now.model {
        found.push(format!("model: recorded `{was_key}`, now `{key}`"));
    }
    let (was_prompt, was_messages) = split_system(&was.chat_history);
    let (now_prompt, now_messages) = split_system(&now.chat_history);
    if was_prompt != now_prompt {
        let was_lines: HashSet<&str> = was_prompt.lines().collect();
        let now_lines: HashSet<&str> = now_prompt.lines().collect();
        found.push("system prompt differs:".to_owned());
        let removed = was_prompt.lines().filter(|line| !now_lines.contains(line));
        found.extend(removed.map(|line| format!("  - {line}")));
        let added = now_prompt.lines().filter(|line| !was_lines.contains(line));
        found.extend(added.map(|line| format!("  + {line}")));
    }
    let names = |request: &CompletionRequest| -> Vec<String> {
        request
            .tools
            .iter()
            .map(|tool| tool.name.to_string())
            .collect()
    };
    if names(was) != names(now) {
        found.push(format!(
            "tools offered: recorded {:?}, now {:?}",
            names(was),
            names(now)
        ));
    }
    if was_messages.len() != now_messages.len() {
        found.push(format!(
            "message count: recorded {}, now {}",
            was_messages.len(),
            now_messages.len()
        ));
    }
    let as_json = |message: &message::Message| serde_json::to_value(message).ok();
    let first = was_messages
        .iter()
        .zip(now_messages)
        .position(|(was, now)| as_json(was) != as_json(now));
    if let Some(at) = first {
        found.push(format!("message {} differs", at + 1));
    }
    found
}

/// A request's history split into its system prompt and the rest.
fn split_system(history: &[message::Message]) -> (String, &[message::Message]) {
    let count = history
        .iter()
        .take_while(|message| matches!(message, message::Message::System { .. }))
        .count();
    let (system, rest) = history.split_at(count);
    let prompt = system
        .iter()
        .filter_map(|message| match message {
            message::Message::System { content } => Some(content.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    (prompt, rest)
}

/// Replays the session a [`ReplaySource`] names.
#[derive(Default)]
pub struct SessionReplayPlugin;

impl Plugin for SessionReplayPlugin {
    fn build(&self, app: &mut App) {
        let source = match app.world_mut().remove_resource::<ReplaySource>() {
            Some(source) => source,
            None => match ReplaySource::from_env() {
                Some(Ok(source)) => source,
                Some(Err(failure)) => {
                    let why = format!("Cannot replay the session in RIG_REPLAY: {failure}.");
                    app.add_systems(Startup, move |mut notices: MessageWriter<Notice>| {
                        notices.write(Notice::error(None, why.clone()));
                    });
                    return;
                }
                None => return,
            },
        };
        app.insert_resource(Replay::new(source))
            .add_systems(
                Update,
                start
                    .run_if(resource_exists::<SessionRestored>)
                    .before(AgentLoop),
            )
            .add_systems(AgentPass, answer_from_record.in_set(Stage::Gate))
            .add_observer(next_message);
    }
}

/// Sends the first user message once the session is restored.
fn start(
    mut replay: ResMut<Replay>,
    agents: PrimaryQuery,
    time: Res<Time<Real>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if replay.status != ReplayStatus::Waiting {
        return;
    }
    let Some(agent) = primary(&agents) else {
        return;
    };
    replay.primary = Some(agent);
    replay.status = ReplayStatus::Running;
    notices.write(Notice::info(
        agent,
        format!("Replaying {} user messages.", replay.messages.len()),
    ));
    send_next(
        &mut replay,
        agent,
        time.elapsed(),
        &mut commands,
        &mut notices,
    );
}

/// Sends the next user message to `agent`, with its original gap after the
/// last one when timing is kept; or finishes.
fn send_next(
    replay: &mut Replay,
    agent: Entity,
    now: Duration,
    commands: &mut Commands,
    notices: &mut MessageWriter<Notice>,
) {
    let Some(said) = replay.messages.pop_front() else {
        replay.status = ReplayStatus::Finished;
        let (models, others) = replay.answered();
        notices.write(Notice::info(
            agent,
            format!("Replay finished: {models} model calls and {others} other effects replayed."),
        ));
        return;
    };
    let deliver = Deliver::new(agent, said.text, DeliveryMode::Queue).literal();
    let wait = match (replay.flags.original_timing, replay.last_sent) {
        (true, Some((then_ms, sent))) => {
            let gap = Duration::from_millis(said.at_ms.saturating_sub(then_ms));
            (sent + gap).saturating_sub(now)
        }
        _ => Duration::ZERO,
    };
    replay.last_sent = Some((said.at_ms, now + wait));
    if wait.is_zero() {
        commands.trigger(deliver);
    } else {
        commands.delayed().duration(wait).trigger(deliver);
    }
}

/// An effect as the replay answers it.
type Answerable<'a> = (
    Entity,
    &'a Effect,
    &'a ForAgent,
    Option<&'a ModelRequest>,
    Option<&'a ToolRequest>,
    Option<&'a EffectRequest>,
);

/// Answers each effect about to start from the record, in id order: a
/// model call with the recorded reply when its request matches, a tool
/// call (unless tools run live) and a plugin's effect of a recorded kind
/// with the record its request matches. At a divergence the call is
/// refused, which stops the replay, or with `continue_live` left to start
/// live, as is everything after it.
fn answer_from_record(
    effects: Query<Answerable, (Undecided, Without<Incoming>)>,
    mut replay: ResMut<Replay>,
    mut commands: Commands,
) {
    let mut ready: Vec<Answerable> = effects.iter().collect();
    ready.sort_by_key(|(_, effect, ..)| effect.id());
    for (entity, _, &ForAgent(agent), model, tool, request) in ready {
        if replay.live {
            return;
        }
        let scope = replay.scope_of(agent);
        let scope = scope.as_deref();
        let answer = match (model, tool, request) {
            (Some(model), ..) => {
                let key = model.connection.as_ref().map_or_else(
                    || HandlerKey::new("completion"),
                    |connection| connection.handler.0.descriptor().key,
                );
                let answer = replay.model(scope, (&key, &model.request));
                replay.answered.0 += usize::from(answer.is_ok());
                answer
            }
            (None, Some(_), _) if replay.flags.live_tools => continue,
            (None, Some(ToolRequest { call }), _) => {
                let kind = EffectKind::ToolCall {
                    name: call.function.name.as_str().to_owned(),
                    args: call.function.raw_arguments(),
                };
                let answer = replay.other(scope, &kind);
                replay.answered.1 += usize::from(answer.is_ok());
                answer
            }
            (None, None, Some(request)) if replay.kinds.contains(request.kind.name()) => {
                let answer = replay.other(scope, &request.kind);
                replay.answered.1 += usize::from(answer.is_ok());
                answer
            }
            _ => continue,
        };
        let mut answering = commands.entity(entity);
        match answer {
            Ok(record) => {
                if let Some(published) = record.tool_output {
                    answering.insert(Published(published));
                }
                answering.insert_if_new(Outcome(record.outcome));
            }
            Err(divergence) => {
                let report = ErrorReport::new(ErrorKind::Divergence, divergence.to_string());
                replay.divergences.push(divergence);
                if replay.flags.continue_live {
                    replay.live = true;
                    return;
                }
                answering.insert_if_new(Outcome(Err(report)));
            }
        }
    }
}

/// After each turn of the replayed agent, stops at a divergence or sends
/// the next user message.
fn next_message(
    settled: On<Add<Settled>>,
    turns: Query<&ForAgent, With<TurnOutcome>>,
    mut replay: ResMut<Replay>,
    time: Res<Time<Real>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(&ForAgent(agent)) = turns.get(settled.entity) else {
        return;
    };
    if replay.primary != Some(agent) || replay.status != ReplayStatus::Running {
        return;
    }
    if let Some(first) = replay.divergences.first() {
        notices.write(Notice::error(agent, first.to_string()));
        if !replay.flags.continue_live {
            replay.status = ReplayStatus::Stopped;
            return;
        }
    }
    send_next(
        &mut replay,
        agent,
        time.elapsed(),
        &mut commands,
        &mut notices,
    );
}
