//! Replays a recorded session: the original user messages are sent again
//! in order, and every model call and tool call is answered from the
//! recorded effect log instead of the provider or the tool. Each model
//! request is checked against the recorded one at the same position of the
//! same agent; the first difference stops the replay, or with
//! `continue_live` is reported and the run goes on live.
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
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use futures::channel::oneshot;
use rig_cassette::effect_log::{EffectLog, jsonl};
use rig_cassette::journal::{EFFECT_LOG, JournalStore, JsonlDirStore};
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::{CompletionRequest, ModelRef, ProviderCapabilities};
use rig_harness::rig_core::effect::{
    EffectId, EffectKind, EffectRecord, FamilyDescriptor, HandlerDescriptor, HandlerKey,
    family::Dynamic,
};
use rig_harness::rig_core::error::{ErrorKind, ErrorReport};
use rig_harness::rig_core::serve::{Dispatch, ErasedHandler, Reply, Serve};

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

/// The recorded session to replay: its effect log and its journal, which
/// holds the user's messages.
#[derive(Resource, Clone)]
pub struct ReplaySource {
    /// Every recorded model and tool call.
    pub effects: EffectLog,
    /// The recorded session's agent logs.
    pub journal: Arc<dyn JournalStore>,
    /// How to replay.
    pub flags: ReplayFlags,
}

impl ReplaySource {
    /// The session kept in the directory `dir`, read only.
    pub fn from_dir(dir: &Path, flags: ReplayFlags) -> io::Result<Self> {
        Ok(Self {
            effects: jsonl::read(dir.join(EFFECT_LOG))?,
            journal: Arc::new(JsonlDirStore::new(dir)),
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

/// The first difference between a replayed request and the recorded one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Divergence {
    /// The recorded agent's id.
    pub agent: String,
    /// The position of the request among the agent's model calls, from 0.
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

/// The replay's state, shared with the handlers that answer calls.
#[derive(Default)]
struct Script {
    /// Each recorded agent's model calls, oldest first.
    models: HashMap<String, VecDeque<EffectRecord>>,
    /// How many model calls of each recorded agent were taken.
    positions: HashMap<String, usize>,
    /// The recorded agents in order of their first effect.
    scopes: Vec<String>,
    /// The recorded tool calls not replayed yet.
    tools: Vec<EffectRecord>,
    /// The recorded model calls replayed, whose tool calls may be replayed.
    replayed: HashSet<EffectId>,
    /// Every divergence, first first.
    divergences: Vec<Divergence>,
    /// Whether the run went live after a divergence.
    live: bool,
    /// Model and tool calls answered from the record.
    answered: (usize, usize),
}

/// What a replayed model call does.
enum Step {
    /// Answer from the record.
    Answer(Reply),
    /// Fail the call: the replay stops.
    Refuse(ErrorReport),
    /// Call the live model; `true` when the run just went live.
    Live(bool),
}

impl Script {
    fn new(effects: &EffectLog) -> Self {
        let mut script = Self::default();
        let mut records: Vec<&EffectRecord> = effects.iter().collect();
        records.sort_by_key(|record| record.id);
        for record in records {
            let scope = record.scope.as_deref().unwrap_or_default().to_owned();
            if !script.scopes.contains(&scope) {
                script.scopes.push(scope.clone());
            }
            match &record.kind {
                EffectKind::Completion { .. } => {
                    let calls = script.models.entry(scope).or_default();
                    calls.push_back(record.clone());
                }
                EffectKind::ToolCall { .. } => script.tools.push(record.clone()),
                _ => {}
            }
        }
        script
    }

    /// The next model call of `scope` under `key`, asked as `kind`.
    fn model(&mut self, scope: &str, key: &HandlerKey, kind: &EffectKind, live: bool) -> Step {
        if self.live {
            return Step::Live(false);
        }
        let position = self.positions.entry(scope.to_owned()).or_default();
        let at = *position;
        *position += 1;
        let next = self.models.get_mut(scope).and_then(VecDeque::pop_front);
        let differences = match (&next, kind) {
            (None, _) => vec!["the recording has no more model calls for this agent".to_owned()],
            (
                Some(EffectRecord {
                    kind: EffectKind::Completion { request: was, .. },
                    key: was_key,
                    ..
                }),
                EffectKind::Completion { request: now, .. },
            ) => differences((was_key, was), (key, now)),
            (Some(_), _) => vec!["a model call arrived where another effect was recorded".into()],
        };
        if let Some(record) = next.filter(|_| differences.is_empty()) {
            self.replayed.insert(record.id);
            self.answered.0 += 1;
            return Step::Answer(Reply::Outcome(record.outcome));
        }
        let divergence = Divergence {
            agent: scope.to_owned(),
            position: at,
            differences,
        };
        let report = ErrorReport::new(ErrorKind::Divergence, divergence.to_string());
        self.divergences.push(divergence);
        if live {
            self.live = true;
            return Step::Live(true);
        }
        Step::Refuse(report)
    }

    /// The recorded call of the tool `name` with `args` that a replayed
    /// model call asked for.
    fn tool(&mut self, name: &str, args: &str) -> Result<Reply, ErrorReport> {
        let parse = |args: &str| serde_json::from_str::<serde_json::Value>(args).ok();
        let wanted = parse(args);
        let found = self.tools.iter().position(|record| {
            matches!(&record.kind, EffectKind::ToolCall { name: was, args: was_args }
                if was == name && parse(was_args) == wanted)
                && record
                    .parent
                    .is_some_and(|parent| self.replayed.contains(&parent))
        });
        match found {
            Some(at) => {
                let record = self.tools.remove(at);
                self.answered.1 += 1;
                Ok(Reply::Outcome(record.outcome))
            }
            None => Err(ErrorReport::new(
                ErrorKind::Divergence,
                format!("replay divergence: no recorded call of `{name}` with {args}"),
            )),
        }
    }
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

/// Locks the script; a panic elsewhere leaves it usable.
fn lock(script: &Mutex<Script>) -> MutexGuard<'_, Script> {
    script.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A request from a handler to the app to put the live tools back, and
/// the answer once it did.
type GoLive = oneshot::Sender<()>;

/// Answers one agent's model calls from the record, or from the live
/// model once the run went live.
struct ReplayModel {
    scope: String,
    script: Arc<Mutex<Script>>,
    descriptor: HandlerDescriptor,
    /// The agent's own model, with `continue_live`.
    live: Option<ErasedHandler>,
    /// Asks the app to put the live tools back, with replayed tools.
    go_live: Option<(Sender<GoLive>, Wake)>,
}

impl Serve for ReplayModel {
    type Family = Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        self.descriptor.clone()
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        let step = lock(&self.script).model(
            &self.scope,
            &self.descriptor.key,
            &kind,
            self.live.is_some(),
        );
        let fresh = match step {
            Step::Answer(reply) => return reply,
            Step::Refuse(report) => return Reply::Outcome(Err(report)),
            Step::Live(fresh) => fresh,
        };
        if fresh && let Some((sender, wake)) = &self.go_live {
            let (done, put_back) = oneshot::channel();
            if sender.send(done).is_ok() {
                wake.wake();
                put_back.await.ok();
            }
        }
        match &self.live {
            Some(live) => live.handle(kind, dispatch).await,
            None => Reply::Outcome(Err(ErrorReport::new(
                ErrorKind::Divergence,
                "replay: no live model to continue with",
            ))),
        }
    }
}

/// Answers a tool's calls from the record.
struct ReplayTool {
    name: String,
    script: Arc<Mutex<Script>>,
    descriptor: HandlerDescriptor,
}

impl Serve for ReplayTool {
    type Family = Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        self.descriptor.clone()
    }

    async fn serve(&self, kind: EffectKind, _dispatch: Dispatch) -> Reply {
        let EffectKind::ToolCall { args, .. } = &kind else {
            let report = ErrorReport::new(ErrorKind::Divergence, "not a tool call");
            return Reply::Outcome(Err(report));
        };
        let answered = lock(&self.script).tool(&self.name, args);
        answered.unwrap_or_else(|report| Reply::Outcome(Err(report)))
    }
}

/// A user message of the recording, and when it was written.
#[derive(Clone, Debug)]
struct Said {
    text: String,
    at_ms: u64,
}

/// The running replay.
#[derive(Resource)]
pub struct Replay {
    script: Arc<Mutex<Script>>,
    flags: ReplayFlags,
    /// The user messages not sent yet.
    messages: VecDeque<Said>,
    /// Each replaying agent's recorded id.
    agents: HashMap<Entity, String>,
    primary: Option<Entity>,
    /// When the last message was sent: originally, and now.
    last_sent: Option<(u64, Duration)>,
    status: ReplayStatus,
    go_live: Sender<GoLive>,
    requests: Mutex<Receiver<GoLive>>,
}

impl Replay {
    /// Where the replay is.
    pub fn status(&self) -> ReplayStatus {
        self.status
    }

    /// Every divergence found, first first.
    pub fn divergences(&self) -> Vec<Divergence> {
        lock(&self.script).divergences.clone()
    }

    /// The model calls and tool calls answered from the record.
    pub fn answered(&self) -> (usize, usize) {
        lock(&self.script).answered
    }

    /// The recorded agent `agent` replays, mapped at its first turn.
    fn scope_of(&mut self, agent: Entity) -> Option<String> {
        if let Some(scope) = self.agents.get(&agent) {
            return Some(scope.clone());
        }
        let taken: HashSet<&String> = self.agents.values().collect();
        let scope = lock(&self.script)
            .scopes
            .iter()
            .find(|scope| !taken.contains(scope))
            .cloned()?;
        self.agents.insert(agent, scope.clone());
        Some(scope)
    }
}

/// On a tool replaced by its replay.
#[derive(Component)]
struct Replaced;

/// On a tool answered from the record.
#[derive(Component)]
struct Replaying;

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
        let script = Script::new(&source.effects);
        let messages = script
            .scopes
            .first()
            .map(|primary| user_messages(source.journal.as_ref(), primary))
            .unwrap_or_default();
        let (go_live, requests) = channel();
        app.insert_resource(Replay {
            script: Arc::new(Mutex::new(script)),
            flags: source.flags,
            messages: messages.into(),
            agents: HashMap::new(),
            primary: None,
            last_sent: None,
            status: ReplayStatus::Waiting,
            go_live,
            requests: Mutex::new(requests),
        })
        .add_systems(
            Update,
            (
                start.run_if(resource_exists::<SessionRestored>),
                put_live_tools_back.before(PollCalls),
            ),
        )
        .add_systems(AgentPass, route_calls.in_set(Stage::Prepare))
        .add_observer(next_message);
    }

    /// Replaces every tool with one answering from the record, unless the
    /// tools run live.
    fn finish(&self, app: &mut App) {
        let world = app.world_mut();
        let Some(replay) = world.get_resource::<Replay>() else {
            return;
        };
        if replay.flags.live_tools {
            return;
        }
        let script = replay.script.clone();
        let mut tools = world.query::<(Entity, &ToolDef, &ToolRules, &Footprint)>();
        let tools: Vec<_> = tools
            .iter(world)
            .map(|(tool, def, rules, footprint)| (tool, def.0.clone(), rules.clone(), *footprint))
            .collect();
        for (tool, definition, rules, footprint) in tools {
            world.entity_mut(tool).insert((Disabled, Replaced));
            let name = definition.name.to_string();
            let handler = ReplayTool {
                descriptor: HandlerDescriptor::tool(
                    &name,
                    definition.description.clone(),
                    definition.parameters.clone(),
                ),
                name,
                script: script.clone(),
            };
            let rules: Vec<&str> = rules.0.iter().map(String::as_str).collect();
            let options = ToolOptions {
                rules: &rules,
                footprint: Some(footprint),
            };
            if let Some(replaying) =
                world.spawn_tool(definition, ErasedHandler::new(handler), options)
            {
                world.entity_mut(replaying).insert(Replaying);
            }
        }
    }
}

/// The user's own messages in the journal of the recorded agent `scope`,
/// in order.
fn user_messages(journal: &dyn JournalStore, scope: &str) -> Vec<Said> {
    let Ok(log) = journal.read(scope) else {
        return Vec::new();
    };
    let lines = log.split(|byte| *byte == b'\n');
    let lines = lines.filter_map(|line| serde_json::from_slice::<serde_json::Value>(line).ok());
    lines
        .filter(|line| line.get("type").and_then(|kind| kind.as_str()) == Some("message"))
        .filter(|line| line.get("origin").is_none_or(serde_json::Value::is_null))
        .filter_map(|line| {
            let at_ms = line.get("t").and_then(serde_json::Value::as_u64)?;
            let message = serde_json::from_value(line.get("message")?.clone()).ok()?;
            let message::Message::User { content } = message else {
                return None;
            };
            let mut texts = Vec::new();
            for item in content {
                match item {
                    UserContent::Text(text) => texts.push(text.text),
                    UserContent::ToolResult(_) => return None,
                    _ => {}
                }
            }
            Some(Said {
                text: texts.join("\n"),
                at_ms,
            })
        })
        .collect()
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
        let (models, tools) = replay.answered();
        notices.write(Notice::info(
            agent,
            format!("Replay finished: {models} model calls and {tools} tool calls replayed."),
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

/// Sends each turn's model call to its agent's replay.
fn route_calls(
    mut calls: Query<(&mut ModelRequest, &ForAgent), (Added<Admitted>, With<TurnCall>)>,
    agents: Query<(Option<&Connection>, Option<&ModelChoice>)>,
    models: Res<Models>,
    wake: Res<Wake>,
    mut replay: ResMut<Replay>,
) {
    for (mut call, &ForAgent(agent)) in &mut calls {
        let Ok((connection, choice)) = agents.get(agent) else {
            continue;
        };
        if let Some(routed) = replayed(&mut replay, agent, (connection, choice), &models, &wake) {
            call.connection = Some(routed);
        }
    }
}

/// The connection that answers `agent`'s model calls from the recording,
/// with its own model, `connection`, or else its `choice`'s.
fn replayed(
    replay: &mut Replay,
    agent: Entity,
    (connection, choice): (Option<&Connection>, Option<&ModelChoice>),
    models: &Models,
    wake: &Wake,
) -> Option<Connection> {
    let scope = replay.scope_of(agent)?;
    let recorded_key = lock(&replay.script)
        .models
        .get(&scope)
        .and_then(|calls| calls.front().map(|record| record.key.clone()));
    let (spec, descriptor) = match (connection, choice) {
        (Some(connection), _) => (connection.spec.clone(), connection.handler.0.descriptor()),
        (None, Some(choice)) => {
            let spec = models.0.catalog().resolve(&choice.0).ok()?.shared();
            let descriptor = HandlerDescriptor {
                key: recorded_key?,
                family: FamilyDescriptor::Completion {
                    model: ModelRef::new(spec.reference()),
                    capabilities: ProviderCapabilities::default(),
                },
                layers: Vec::new(),
            };
            (spec, descriptor)
        }
        (None, None) => return None,
    };
    let continue_live = replay.flags.continue_live;
    let handler = ReplayModel {
        scope,
        script: replay.script.clone(),
        descriptor,
        live: connection
            .filter(|_| continue_live)
            .map(|connection| connection.handler.0.clone()),
        go_live: (continue_live && !replay.flags.live_tools)
            .then(|| (replay.go_live.clone(), wake.clone())),
    };
    let handler = rig_harness::rig_ecs::effects::Handler(ErasedHandler::new(handler));
    Some(Connection { spec, handler })
}

/// After each turn of the replayed agent, stops at a divergence or sends
/// the next user message.
fn next_message(
    ended: On<TurnEnded>,
    mut replay: ResMut<Replay>,
    time: Res<Time<Real>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = ended.entity;
    if replay.primary != Some(agent) || replay.status != ReplayStatus::Running {
        return;
    }
    let divergences = replay.divergences();
    if let Some(first) = divergences.first() {
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

/// Puts the live tools back once a replay goes live, before the live
/// model's reply can call them.
fn put_live_tools_back(
    replay: Res<Replay>,
    replaying: Query<Entity, With<Replaying>>,
    replaced: Query<Entity, (With<Replaced>, With<Disabled>)>,
    mut commands: Commands,
) {
    let asked: Vec<GoLive> = replay
        .requests
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .try_iter()
        .collect();
    if asked.is_empty() {
        return;
    }
    for tool in &replaying {
        commands.entity(tool).despawn();
    }
    for tool in &replaced {
        commands.entity(tool).remove::<(Disabled, Replaced)>();
    }
    // The model call waits for this before it is sent; the tools are back
    // when this system's commands apply, before any reply is polled.
    for done in asked {
        done.send(()).ok();
    }
}
