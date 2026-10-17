//! Replays a recorded session: the user's messages are sent again in
//! order, and every model call, tool call and replayable plugin effect is
//! answered from the recorded effect log instead of by the provider or the
//! tool. A model call is answered by the replay's adapter, placed
//! outermost: it replaces the call's adapted model with the record, so
//! Start fits the request as it would live, the request compared is the
//! one recorded as sent, and no other adapter rewrites the recorded reply
//! again. A recorded model the environment has no key for, such as one
//! whose credentials are gone, is reached through the replay's sign-in,
//! which answers from the record only. Tool calls and replayable plugin
//! effects are decided as Gate opens, before any gate, but for the calls
//! of a tool that [`RunsAgain`], such as `rig_basics::subagents`' `task`, `message`
//! and `wait`, whose answers come from the agents the replay runs again: a
//! spawned subagent gets its recorded id, so what names it reads as it was
//! recorded, and a `wait` ends when its report comes, as it did. Each request is checked against the
//! recorded one at the same place of the same agent; the first difference
//! stops the replay, or with `--continue-live` is reported and the run goes
//! on live. With `--speed` each answer settles after its recorded duration
//! and each user message after its recorded gap. A replay that ends with
//! recorded model calls not replayed (a turn failed before its call reached
//! the replay) stops with why, as at a divergence, and `--replay` then
//! exits with 1. A recorded exit that ended the session for good, such as
//! `/quit`, is asked for again where it was, so what held it, such as a
//! summary call, replays too, and the replay ends as the exit goes; one
//! that restarted the session, a reload or a switch, is passed over. Once
//! the replay ended, every new effect but the exit, such as the summary
//! call of a killed session's exit, is refused before it starts: the record
//! does not have it.
//!
//! `rig --replay=DIR` gives the run to the replay's own front, which keeps
//! no session: it replays the session kept in `DIR`, prints the report as
//! the app exits, and exits with 0, or 1 when the replay diverged. A
//! [`ReplaySource`] inserted before the plugin is added replays inside
//! whatever front runs. Nothing recorded is written to.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::io;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use rig_basics::subagents::{MESSAGE, SUBAGENT, SpawnSubagent, SpawnedAgent, TASK, WAIT};
use rig_cassette::effect_log::jsonl::{self, SESSION_EFFECT_LOG};
use rig_cassette::effect_log::{EffectLog, EffectLogReplayer, RecordedAnswers, RequestCheck};
use rig_harness::harness_protocol::SessionDir;
use rig_harness::prelude::*;
use rig_harness::rig_core::catalog::{ModelSpec, SignIn, Transport};
use rig_harness::rig_core::completion::{CompletionRequest, ModelRef, ProviderCapabilities};
use rig_harness::rig_core::effect::{
    self, EffectId, EffectKind, EffectRecord, FamilyDescriptor, HandlerDescriptor, HandlerKey,
    family, model_key,
};
use rig_harness::rig_core::error::{ErrorKind, ErrorReport};
use rig_harness::rig_core::providers::registry::ConnectError;
use rig_harness::rig_core::serve::{Dispatch, ErasedHandler, Reply, Serve};
use rig_harness::rig_ecs::effects::Handler;

#[cfg(test)]
mod tests;

/// The option naming the session directory to replay; its front.
pub const REPLAY: &str = "replay";
/// The option: after the first difference, go on live.
pub const CONTINUE_LIVE: &str = "continue-live";
/// The option: run the tools for real.
pub const LIVE_TOOLS: &str = "live-tools";
/// The option: keep the recorded timing.
pub const SPEED: &str = "speed";

/// How a replay runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReplayFlags {
    /// After the first difference, report it and go on with the live model
    /// and tools instead of stopping.
    pub continue_live: bool,
    /// Run the tools for real; the model replies still come from the record.
    pub live_tools: bool,
    /// Settle each answer after its recorded duration, and send each user
    /// message after its recorded gap, instead of at once.
    pub original_timing: bool,
}

/// The recorded session to replay: its effect log, which holds every
/// effect its agents settled, the user's messages and the app's exits
/// included.
#[derive(Resource, Clone)]
pub struct ReplaySource {
    /// Every recorded effect.
    pub effects: EffectLog,
    /// How to replay.
    pub flags: ReplayFlags,
}

impl ReplaySource {
    /// The session kept in `store`, read only.
    pub fn from_store(store: &dyn Files, flags: ReplayFlags) -> io::Result<Self> {
        Ok(Self {
            effects: jsonl::read_from(store.read_file(SESSION_EFFECT_LOG)?.as_slice())?,
            flags,
        })
    }

    /// The source the options name, kept in the directory they name, read
    /// through `stores`.
    fn from_invocation(invoked: &Invoked, stores: &StoreOpener) -> io::Result<Self> {
        let dir = invoked.flag_value(REPLAY).unwrap_or_default();
        let flags = ReplayFlags {
            continue_live: invoked.flag(CONTINUE_LIVE),
            live_tools: invoked.flag(LIVE_TOOLS),
            original_timing: invoked.flag(SPEED),
        };
        let store = stores.open(&SessionDir::at(dir))?;
        Self::from_store(&*store, flags)
    }
}

/// A kind of plugin effect ([`EffectKind::name`]) a replay answers from the
/// record, as an entity: a plugin whose effects only report what happened
/// outside the agent spawns one; one whose effects change the agent's own
/// world, such as a subagent's spawn, does not, and its effects run again.
/// Questions to the user (`ask`) are replayable.
#[derive(Component, Clone, Debug)]
pub struct Replayable(pub Cow<'static, str>);

/// A tool, by name, whose calls a replay runs again instead of answering
/// them from the record, as an entity: a tool that acts only in the
/// agents' own world, so its answer depends on what the agents the replay
/// runs do, such as a subagent's report. The replay spawns one for
/// `rig_basics::subagents`' `task`, `message` and `wait`.
#[derive(Component, Clone, Debug)]
pub struct RunsAgain(pub Cow<'static, str>);

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
    /// Sending the user's messages, or exiting as recorded.
    Running,
    /// Every user message was replayed, and the recorded exit asked for.
    Finished,
    /// Stopped at the first divergence, or finished with recorded model
    /// calls not replayed.
    Stopped,
}

/// The replay finished or stopped; [`Replay`] has its report.
#[derive(Event, Clone, Copy, Debug)]
pub struct ReplayEnded(pub ReplayStatus);

/// What the user did in the recording, and when it was accepted.
#[derive(Clone, Debug)]
struct Step {
    act: Act,
    at_ms: u64,
}

/// A user's message, or an exit that ended the session for good.
#[derive(Clone, Debug)]
enum Act {
    Say(MessageRecord),
    Exit(ExitReason),
}

/// What the replay's systems and its model handlers, which answer inside
/// the calls' tasks, share.
#[derive(Default)]
struct Answers {
    /// Every recorded effect but the messages, answered by scope and kind
    /// in recorded order, each request checked as data.
    record: RecordedAnswers,
    /// How many model calls of each recorded agent were taken.
    positions: HashMap<Option<Arc<str>>, usize>,
    divergences: Vec<Divergence>,
    /// Model calls and other effects answered from the record.
    answered: (usize, usize),
    /// How long each effect answered from the record ran when recorded,
    /// kept until Review holds it that long.
    took: HashMap<EffectId, Duration>,
    /// The replay ended: every effect after is past the record.
    ended: bool,
}

impl Answers {
    /// The answer to the model call `kind`, sent as is to the model `key`
    /// by the recorded agent `scope`, or why it diverged.
    fn model(
        &mut self,
        scope: Option<&str>,
        key: &HandlerKey,
        kind: &EffectKind,
    ) -> Result<EffectRecord, Divergence> {
        let position = self.positions.entry(scope.map(Arc::from)).or_default();
        let at = *position;
        *position += 1;
        let differences = match self.record.answer(scope, kind) {
            Ok(record) if record.key == *key => {
                self.answered.0 += 1;
                return Ok(record);
            }
            Ok(record) => vec![format!("model: recorded `{}`, now `{key}`", record.key)],
            Err(diverged) => {
                // The recorded call next in order, for a readable diff.
                let mut found = vec![diverged.report.message];
                if let (
                    Some(EffectRecord {
                        kind: EffectKind::Completion { request: was, .. },
                        ..
                    }),
                    EffectKind::Completion { request: now, .. },
                ) = (diverged.expected, kind)
                {
                    found.extend(differences(&was, now));
                }
                found
            }
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
        let answer = self.record.answer(scope, kind).map_err(|diverged| {
            let made = self.positions.get(&scope.map(Arc::from)).copied();
            Divergence {
                agent: scope.unwrap_or_default().to_owned(),
                position: made.unwrap_or_default().saturating_sub(1),
                differences: vec![diverged.report.message],
            }
        });
        self.answered.1 += usize::from(answer.is_ok());
        answer
    }

    /// Why no effect is answered from the record any more: the replay
    /// stopped at a divergence, or it ended.
    fn past(&self) -> Option<&'static str> {
        match (self.divergences.is_empty(), self.ended) {
            (false, _) => Some(STOPPED),
            (true, true) => Some(ENDED),
            (true, false) => None,
        }
    }

    /// Notes how long `record`, the answer to the effect `id`, ran.
    fn time(&mut self, id: EffectId, record: &EffectRecord) {
        if let (Some(started), Some(finished)) = (record.started_ms, record.finished_ms) {
            let took = Duration::from_millis(finished.saturating_sub(started));
            self.took.insert(id, took);
        }
    }
}

/// The shared answers, locked; a poisoned lock is taken as it is.
fn lock(answers: &Mutex<Answers>) -> MutexGuard<'_, Answers> {
    answers.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The running replay.
#[derive(Resource)]
pub struct Replay {
    flags: ReplayFlags,
    answers: Arc<Mutex<Answers>>,
    /// The recorded agent the primary agent stands for.
    first: Option<Arc<str>>,
    /// The agents each recorded agent spawned, in order.
    spawned: HashMap<Arc<str>, VecDeque<Arc<str>>>,
    /// The recorded agent each replaying agent stands for.
    paired: HashMap<Entity, Arc<str>>,
    /// The recorded agent each spawn request given its id stands for,
    /// until it spawned.
    spawning: HashMap<Entity, Arc<str>>,
    /// The user messages not sent yet, and the recorded exit.
    steps: VecDeque<Step>,
    primary: Option<Entity>,
    /// When the last message was sent: originally, and now.
    last_sent: Option<(u64, Duration)>,
    /// The model calls the record holds.
    recorded_models: usize,
    /// Why the primary agent's last failed turn failed.
    failed: Option<String>,
    status: ReplayStatus,
}

impl Replay {
    fn new(source: ReplaySource) -> Self {
        let ReplaySource { mut effects, flags } = source;
        effects.records.sort_by_key(|record| record.id);
        // The first agent's; effects of the app itself have no scope.
        let first = effects
            .records
            .iter()
            .find_map(|record| record.scope.clone());
        let mut spawned: HashMap<Arc<str>, VecDeque<Arc<str>>> = HashMap::new();
        for record in &effects.records {
            if let (Some(parent), Some(agent)) = (&record.scope, spawned_agent(record)) {
                spawned.entry(parent.clone()).or_default().push_back(agent);
            }
        }
        let steps = effects
            .records
            .iter()
            .filter_map(|record| match &record.scope {
                Some(_) if record.scope == first => said(record),
                Some(_) => None,
                None => exited(record),
            })
            .collect();
        effects.records.retain(|record| {
            MessageRecord::read(&record.kind).is_none() && exit_reason(record).is_none()
        });
        let recorded_models = effects
            .records
            .iter()
            .filter(|record| matches!(record.kind, EffectKind::Completion { .. }))
            .count();
        let answers = Answers {
            record: RecordedAnswers::new(&effects),
            ..Answers::default()
        };
        Self {
            flags,
            answers: Arc::new(Mutex::new(answers)),
            first,
            spawned,
            paired: HashMap::new(),
            spawning: HashMap::new(),
            steps,
            primary: None,
            last_sent: None,
            recorded_models,
            failed: None,
            status: ReplayStatus::Waiting,
        }
    }

    /// Why the replay, which found no divergence, is not complete: some
    /// recorded model calls were never made.
    fn missed(&self) -> Option<Divergence> {
        let answers = lock(&self.answers);
        let made = answers.answered.0;
        if !answers.divergences.is_empty() || made >= self.recorded_models {
            return None;
        }
        let recorded = self.recorded_models;
        let mut differences = vec![format!(
            "{} of the {recorded} recorded model calls were not replayed",
            recorded - made
        )];
        differences.extend(
            self.failed
                .iter()
                .map(|why| format!("a turn failed: {why}")),
        );
        let first = self.first.as_deref();
        Some(Divergence {
            agent: first.unwrap_or_default().to_owned(),
            position: answers
                .positions
                .get(&self.first)
                .copied()
                .unwrap_or_default(),
            differences,
        })
    }

    /// Where the replay is.
    pub fn status(&self) -> ReplayStatus {
        self.status
    }

    /// Every divergence found, first first.
    pub fn divergences(&self) -> Vec<Divergence> {
        lock(&self.answers).divergences.clone()
    }

    /// The model calls and the other effects answered from the record.
    pub fn answered(&self) -> (usize, usize) {
        lock(&self.answers).answered
    }

    /// The recorded agent an effect for `agent` stands for (none for the
    /// app's own); an agent the recording does not have goes by its own
    /// name, which no record answers.
    fn scope(&self, agent: Option<&ForAgent>) -> Option<Arc<str>> {
        let &ForAgent(agent) = agent?;
        let unrecorded = || Arc::from(format!("unrecorded agent {agent}"));
        Some(self.paired.get(&agent).cloned().unwrap_or_else(unrecorded))
    }
}

/// The agent a `subagent` effect's answer names.
fn spawned_agent(record: &EffectRecord) -> Option<Arc<str>> {
    let Ok(effect::Answered::Custom { payload }) = &record.outcome else {
        return None;
    };
    (record.kind.name() == SUBAGENT).then_some(())?;
    payload["agent"].as_str().map(Arc::from)
}

/// The user's message `record` holds, if it was delivered.
fn said(record: &EffectRecord) -> Option<Step> {
    let message = MessageRecord::read(&record.kind)?;
    let delivered = matches!(&record.outcome,
        Ok(effect::Answered::Custom { payload }) if payload == "delivered");
    (delivered && message.author.kind == AuthorKind::User).then(|| Step {
        act: Act::Say(message),
        at_ms: recorded_at(record),
    })
}

/// The exit `record` holds, if it ended the session for good: a reload or
/// a switch restarted it, and what the session did after is replayed in
/// the same run.
fn exited(record: &EffectRecord) -> Option<Step> {
    match exit_reason(record)? {
        ExitReason::Reload | ExitReason::Switch(_) => None,
        reason => Some(Step {
            act: Act::Exit(reason),
            at_ms: recorded_at(record),
        }),
    }
}

/// The reason of the app's exit `record` holds, if it holds one. The
/// harness records it as the custom kind `exit` with the reason's debug
/// form; a switch's target is not needed, as a switch is not replayed.
fn exit_reason(record: &EffectRecord) -> Option<ExitReason> {
    let EffectKind::Custom { kind, payload } = &record.kind else {
        return None;
    };
    let reason = payload.as_str().filter(|_| &**kind == "exit")?;
    match reason {
        "Quit" => Some(ExitReason::Quit),
        "Reload" => Some(ExitReason::Reload),
        "Signal" => Some(ExitReason::Signal),
        _ if reason.starts_with("Switch(") => Some(ExitReason::Switch(None)),
        _ => reason
            .strip_prefix("Failed(")?
            .strip_suffix(')')?
            .parse()
            .ok()
            .map(ExitReason::Failed),
    }
}

/// When `record` started, else when it finished.
fn recorded_at(record: &EffectRecord) -> u64 {
    record.started_ms.or(record.finished_ms).unwrap_or_default()
}

/// What differs between the recorded request and the new one, as lines.
fn differences(was: &CompletionRequest, now: &CompletionRequest) -> Vec<String> {
    let mut found = Vec::new();
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
        let tools = request.tools.iter();
        tools.map(|tool| tool.name.to_string()).collect()
    };
    if names(was) != names(now) {
        let (was, now) = (names(was), names(now));
        found.push(format!("tools offered: recorded {was:?}, now {now:?}"));
    }
    if was_messages.len() != now_messages.len() {
        let (was, now) = (was_messages.len(), now_messages.len());
        found.push(format!("message count: recorded {was}, now {now}"));
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
fn split_system(
    history: &[std::sync::Arc<message::Message>],
) -> (String, &[std::sync::Arc<message::Message>]) {
    let count = history
        .iter()
        .take_while(|message| matches!(&***message, message::Message::System { .. }))
        .count();
    let (system, rest) = history.split_at(count);
    let prompt = system
        .iter()
        .filter_map(|message| match &**message {
            message::Message::System { content } => Some(content.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    (prompt, rest)
}

/// The adapter's name on a replayed model call.
const ADAPTER: &str = "replay";

/// Why an effect after a divergence is refused.
const STOPPED: &str = "the replay stopped at a divergence";

/// Why an effect after the replay ended is refused.
const ENDED: &str = "not in the record: the replay ended";

/// A model of a replayed agent: answers each request it is served from the
/// recorded agent's calls, in order; after a divergence or the replay's
/// end it refuses, or with `--continue-live` passes the request to the
/// live model.
struct FromRecord {
    answers: Arc<Mutex<Answers>>,
    scope: Option<Arc<str>>,
    live: ErasedHandler,
    flags: ReplayFlags,
}

impl Serve for FromRecord {
    type Family = family::Dynamic;

    /// The live model's, so the call is recorded as sent to it.
    fn descriptor(&self) -> HandlerDescriptor {
        self.live.descriptor()
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        let answer = {
            let mut answers = lock(&self.answers);
            if let Some(why) = answers.past() {
                let stopped = ErrorReport::new(ErrorKind::Divergence, why);
                (!self.flags.continue_live).then_some(Err(stopped))
            } else {
                let key = self.live.descriptor().key;
                let answer = answers.model(self.scope.as_deref(), &key, &kind);
                match answer {
                    Ok(record) => {
                        if self.flags.original_timing {
                            answers.time(dispatch.id(), &record);
                        }
                        Some(Ok(record))
                    }
                    Err(divergence) => {
                        let report =
                            ErrorReport::new(ErrorKind::Divergence, divergence.to_string());
                        answers.divergences.push(divergence);
                        (!self.flags.continue_live).then_some(Err(report))
                    }
                }
            }
        };
        match answer {
            // A replayer of this one record streams it as recorded and
            // publishes its facts.
            Some(Ok(record)) => {
                let key = record.key.clone();
                let log = EffectLog::from_records(vec![record]);
                match EffectLogReplayer::for_key(&log, &key) {
                    Ok(replayer) => {
                        let replayer = replayer.checking(RequestCheck::Skip);
                        replayer.serve(kind, dispatch).await
                    }
                    Err(report) => Reply::Outcome(Err(report)),
                }
            }
            Some(Err(report)) => Reply::Outcome(Err(report)),
            None => self.live.handle(kind, dispatch).await,
        }
    }
}

/// Replays the session a [`ReplaySource`] or `--replay=DIR` names.
#[derive(Default)]
pub struct SessionReplayPlugin;

/// The replay's stage systems, for others to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReplaySystems {
    /// Puts the record outermost on each model call, before compaction
    /// measures the request.
    Prepare,
    /// Decides tool calls and replayable plugin effects from the record,
    /// before any gate refuses or holds them or a front answers a question.
    Gate,
    /// Holds each answer for its recorded duration, with `--speed`.
    Review,
}

impl Plugin for SessionReplayPlugin {
    fn build(&self, app: &mut App) {
        let flag = Flag::new(REPLAY, "Replay the session kept in DIR, then exit").value("DIR");
        app.add_front(Front {
            flag: Some(flag),
            ..Front::new(REPLAY).sessionless()
        })
        .add_flag(Flag::new(
            CONTINUE_LIVE,
            "Go on live after --replay's first difference",
        ))
        .add_flag(Flag::new(LIVE_TOOLS, "Run tools for real in --replay"))
        .add_flag(Flag::new(SPEED, "Keep the recorded timing in --replay"));
        app.world_mut().spawn(Replayable("ask".into()));
        for tool in [TASK, MESSAGE, WAIT] {
            app.world_mut().spawn(RunsAgain(tool.into()));
        }
        app.configure_sets(
            AgentPass,
            (
                ReplaySystems::Prepare.in_set(PrepareSystems::Edit),
                ReplaySystems::Gate.in_set(GateSystems::FromRecord),
                ReplaySystems::Review.in_set(Stage::Review),
            ),
        );
    }

    fn finish(&self, app: &mut App) {
        let source = if app.runs_front(REPLAY) {
            let invoked = app.world().get_resource::<Invoked>().cloned();
            let stores = app.world().get_resource::<StoreOpener>().cloned();
            let stores = stores.unwrap_or_default();
            let source = ReplaySource::from_invocation(&invoked.unwrap_or_default(), &stores);
            app.add_observer(report_and_exit);
            match source {
                Ok(source) => source,
                Err(failure) => {
                    eprintln!("rig --replay: cannot read the session: {failure}");
                    app.add_systems(Startup, |mut commands: Commands| {
                        commands.trigger(Exit(ExitReason::Failed(1)));
                    });
                    return;
                }
            }
        } else {
            match app.world_mut().remove_resource::<ReplaySource>() {
                Some(source) => source,
                None => return,
            }
        };
        // The replay's own front starts with no agent: its primary agent is
        // the recorded first one again, id and all, so what names it, such
        // as a subagent's brief, reads as recorded.
        let first = source.effects.records.iter().find_map(|r| r.scope.clone());
        if let (true, Some(first)) = (app.runs_front(REPLAY), first) {
            let first = AgentId(first.to_string());
            app.world_mut().spawn((Name::new("agent"), Agent, first));
        }
        let timed = source.flags.original_timing;
        let recorded = source.effects.records.iter();
        let models = recorded.filter(|record| matches!(record.kind, EffectKind::Completion { .. }));
        let models = models.map(|record| record.key.clone()).collect();
        app.world_mut()
            .get_resource_or_init::<Models>()
            .0
            .add_sign_in(RecordedModels(models));
        app.insert_resource(Replay::new(source))
            .add_systems(
                Update,
                start
                    .run_if(resource_exists::<SessionRestored>)
                    .before(AgentLoop),
            )
            .add_systems(
                AgentPass,
                (
                    (recorded_ids, models_from_record)
                        .chain()
                        .in_set(ReplaySystems::Prepare),
                    answer_from_record.in_set(ReplaySystems::Gate),
                    keep_recorded_timing
                        .in_set(ReplaySystems::Review)
                        .run_if(move || timed),
                ),
            )
            .add_observer(next_message)
            .add_observer(exit_started)
            .add_observer(pair_spawned);
    }
}

/// The replay front: prints the report once the replay is over, and exits,
/// with 1 when it found a divergence. When the replay ended as the app
/// exits, the exit leaves with that code instead of the recorded one.
fn report_and_exit(
    ended: On<ReplayEnded>,
    replay: Res<Replay>,
    exiting: Option<Res<ExitReason>>,
    mut commands: Commands,
) {
    for divergence in replay.divergences() {
        println!("{divergence}");
    }
    let (models, others) = replay.answered();
    let how = match ended.0 {
        ReplayStatus::Stopped => "stopped",
        _ => "finished",
    };
    println!("Replay {how}: {models} model calls and {others} other effects replayed.");
    let exit = match replay.divergences().is_empty() {
        true => ExitReason::Quit,
        false => ExitReason::Failed(1),
    };
    match exiting {
        Some(_) => commands.insert_resource(exit),
        None => commands.trigger(Exit(exit)),
    }
}

/// Pairs the primary agent with the recorded first, and sends the first
/// user message, once the session is restored.
fn start(
    mut replay: ResMut<Replay>,
    agents: MainAgentQuery,
    time: Res<Time<Real>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if replay.status != ReplayStatus::Waiting {
        return;
    }
    let Some(agent) = main_agent(&agents) else {
        return;
    };
    replay.primary = Some(agent);
    if let Some(first) = replay.first.clone() {
        replay.paired.insert(agent, first);
    }
    replay.status = ReplayStatus::Running;
    let count = replay
        .steps
        .iter()
        .filter(|step| matches!(step.act, Act::Say(_)))
        .count();
    notices.write(Notice::info(
        agent,
        format!("Replaying {count} user messages."),
    ));
    send_next(
        &mut replay,
        agent,
        time.elapsed(),
        &mut commands,
        &mut notices,
    );
}

/// Gives each spawn request of a replayed agent the id of the agent its
/// recorded spawn at the same place made, so the subagent is that agent
/// again: the answers and the messages that name it read as recorded.
fn recorded_ids(
    mut spawns: Query<(Entity, &mut SpawnSubagent, &ForAgent), Added<Admitted>>,
    mut replay: ResMut<Replay>,
) {
    for (entity, mut spawn, &ForAgent(parent)) in &mut spawns {
        let Some(recorded_parent) = replay.paired.get(&parent).cloned() else {
            continue;
        };
        let recorded = replay
            .spawned
            .get_mut(&recorded_parent)
            .and_then(VecDeque::pop_front);
        if let Some(recorded) = recorded {
            spawn.id = Some(AgentId(recorded.to_string()));
            replay.spawning.insert(entity, recorded);
        }
    }
}

/// Pairs each subagent the replay spawns with the recorded one its
/// request was given the id of.
fn pair_spawned(
    spawned: On<Add<SpawnedAgent>>,
    spawns: Query<&SpawnedAgent>,
    mut replay: ResMut<Replay>,
) {
    let Ok(&SpawnedAgent(child)) = spawns.get(spawned.entity) else {
        return;
    };
    if let Some(recorded) = replay.spawning.remove(&spawned.entity) {
        replay.paired.insert(child, recorded);
    }
}

/// Sends the next user message to `agent`, or asks for the recorded exit,
/// with its original gap after the last one when timing is kept; or
/// finishes.
fn send_next(
    replay: &mut Replay,
    agent: Entity,
    now: Duration,
    commands: &mut Commands,
    notices: &mut MessageWriter<Notice>,
) {
    let Some(step) = replay.steps.pop_front() else {
        finish(replay, agent, commands, notices);
        return;
    };
    let wait = match (replay.flags.original_timing, replay.last_sent) {
        (true, Some((then_ms, sent))) => {
            let gap = Duration::from_millis(step.at_ms.saturating_sub(then_ms));
            (sent + gap).saturating_sub(now)
        }
        _ => Duration::ZERO,
    };
    replay.last_sent = Some((step.at_ms, now + wait));
    match step.act {
        Act::Say(message) => {
            let deliver = Deliver::new(agent, message.text, message.mode).literal();
            trigger_after(commands, wait, deliver);
        }
        Act::Exit(reason) => {
            notices.write(Notice::info(agent, "Replaying the recorded exit."));
            trigger_after(commands, wait, Exit(reason));
        }
    }
}

/// Triggers `event` after `wait`, or now when there is none.
fn trigger_after<'a>(
    commands: &mut Commands,
    wait: Duration,
    event: impl Event<Trigger<'a>: Default>,
) {
    if wait.is_zero() {
        commands.trigger(event);
    } else {
        commands.delayed().duration(wait).trigger(event);
    }
}

/// Ends the replay: stopped when recorded model calls were not replayed,
/// else finished, and says so to `agent`.
fn finish(
    replay: &mut Replay,
    agent: Entity,
    commands: &mut Commands,
    notices: &mut MessageWriter<Notice>,
) {
    let (models, others) = replay.answered();
    if let Some(missed) = replay.missed() {
        notices.write(Notice::error(agent, missed.to_string()));
        lock(&replay.answers).divergences.push(missed);
        end(replay, ReplayStatus::Stopped, commands);
        return;
    }
    notices.write(Notice::info(
        agent,
        format!("Replay finished: {models} model calls and {others} other effects replayed."),
    ));
    end(replay, ReplayStatus::Finished, commands);
}

/// Ends the replay with `status`, and says so.
fn end(replay: &mut Replay, status: ReplayStatus, commands: &mut Commands) {
    replay.status = status;
    lock(&replay.answers).ended = true;
    commands.trigger(ReplayEnded(status));
}

/// Puts the record outermost on each model call, whichever model the call
/// goes to: it replaces the model as every other adapter made it.
fn models_from_record(
    mut calls: Query<(&mut Adapters, Option<&ForAgent>), Added<Admitted>>,
    replay: Res<Replay>,
) {
    for (mut adapters, agent) in &mut calls {
        // A successor cloned from a replayed call has the adapter already.
        if adapters.0.iter().any(|adapter| adapter.name == ADAPTER) {
            continue;
        }
        let scope = replay.scope(agent);
        let (answers, flags) = (replay.answers.clone(), replay.flags);
        let adapter = Adapter::new(ADAPTER, move |mut connection: Connection| {
            let model = FromRecord {
                answers: answers.clone(),
                scope: scope.clone(),
                live: connection.handler.0,
                flags,
            };
            connection.handler = Handler(ErasedHandler::new(model));
            connection
        });
        adapters.0.push(adapter.outermost());
    }
}

/// The recorded models, for the ones the environment has no key for: each
/// is reached as a model that has the key a connection to it would have,
/// so the record answers it, and that refuses what goes on live.
struct RecordedModels(HashSet<HandlerKey>);

impl SignIn for RecordedModels {
    fn handler(
        &self,
        spec: &ModelSpec,
        _: &Transport,
    ) -> Option<Result<ErasedHandler, ConnectError>> {
        let label = spec.reference();
        let key = model_key(&label);
        self.0.contains(&key).then(|| {
            let why =
                format!("{label} is not connected: the replay answers it from the record only");
            let descriptor = HandlerDescriptor {
                key,
                family: FamilyDescriptor::Completion {
                    model: ModelRef::from(label),
                    capabilities: ProviderCapabilities::default(),
                },
                layers: Vec::new(),
            };
            Ok(ErasedHandler::new(Unreachable { descriptor, why }))
        })
    }

    fn plan(&self, _: &ModelSpec) -> Option<&'static str> {
        None
    }
}

/// A model the replay cannot reach, and why.
struct Unreachable {
    descriptor: HandlerDescriptor,
    why: String,
}

impl Serve for Unreachable {
    type Family = family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        self.descriptor.clone()
    }

    async fn serve(&self, _: EffectKind, _: Dispatch) -> Reply {
        Reply::Outcome(Err(ErrorReport::new(
            ErrorKind::HandlerUnavailable,
            self.why.clone(),
        )))
    }
}

/// An effect as the replay decides it in Gate.
type Answerable<'a> = (
    Entity,
    &'a Effect,
    &'a mut Decision,
    Option<&'a ForAgent>,
    Option<&'a ToolRequest>,
    Option<&'a EffectRequest>,
);

/// Decides each tool call (unless tools run live, or its tool [`RunsAgain`]) and [`Replayable`]
/// plugin effect about to start, and not decided yet, with the record its
/// request matches, in id order. At a divergence the effect is refused,
/// which stops the replay, as is everything after it but the exit, model
/// calls included; so is everything after the replay ended, as no record
/// has it. With `continue_live` they are left to start live.
fn answer_from_record(
    mut effects: Query<Answerable, (ReadyToStart, Without<Incoming>, Without<ExitRequest>)>,
    (replayable, again): (Query<&Replayable>, Query<&RunsAgain>),
    replay: Res<Replay>,
    mut commands: Commands,
) {
    let mut ready: Vec<_> = effects.iter_mut().collect();
    ready.sort_by_key(|(_, effect, ..)| effect.id());
    let mut answers = lock(&replay.answers);
    for (entity, effect, mut decision, agent, tool, request) in ready {
        if decision.is_decided() {
            continue;
        }
        if let Some(why) = answers.past() {
            if replay.flags.continue_live {
                return;
            }
            decision.decide(Decider::plugin("session-replay"), Outcome::refused(why));
            continue;
        }
        let replays = |kind: &EffectKind| {
            let name = kind.name();
            replayable.iter().any(|replayable| replayable.0 == name)
        };
        let runs_again = |call: &ToolCall| {
            let name = call.function.name.as_str();
            again.iter().any(|again| again.0 == name)
        };
        let kind = match (tool, request) {
            (Some(ToolRequest { call }), _) if replay.flags.live_tools || runs_again(call) => {
                continue;
            }
            (Some(ToolRequest { call }), _) => EffectKind::from(call),
            (None, Some(request)) if replays(&request.kind) => request.kind.clone(),
            _ => continue,
        };
        match answers.other(replay.scope(agent).as_deref(), &kind) {
            Ok(record) => {
                if replay.flags.original_timing {
                    answers.time(effect.id(), &record);
                }
                decision.decide(Decider::plugin("session-replay"), Outcome(record.outcome));
                if let Some(published) = record.published {
                    commands.entity(entity).insert(Published(published));
                }
            }
            Err(divergence) => {
                let report = ErrorReport::new(ErrorKind::Divergence, divergence.to_string());
                answers.divergences.push(divergence);
                if replay.flags.continue_live {
                    return;
                }
                decision.decide(Decider::plugin("session-replay"), Outcome(Err(report)));
            }
        }
    }
}

/// With `--speed`, holds each answer from the record as long as the
/// recorded effect ran.
fn keep_recorded_timing(
    answered: Query<(Entity, &Effect), ReadyToSettle>,
    replay: Res<Replay>,
    mut commands: Commands,
) {
    let mut answers = lock(&replay.answers);
    for (entity, effect) in &answered {
        if let Some(took) = answers.took.remove(&effect.id()) {
            let hold = commands.spawn(HoldOf(entity)).id();
            commands.delayed().duration(took).entity(hold).try_despawn();
        }
    }
}

/// The replay ends as the app's exit goes, once whatever held it, such as a
/// recorded summary call, is done.
fn exit_started(
    started: On<Add<Started>>,
    exits: Query<(), With<ExitRequest>>,
    mut replay: ResMut<Replay>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let running = replay.status == ReplayStatus::Running;
    if let (true, true, Some(agent)) = (exits.contains(started.entity), running, replay.primary) {
        finish(&mut replay, agent, &mut commands, &mut notices);
    }
}

/// After each turn of the replayed agent, stops at a divergence or sends
/// the next user message.
fn next_message(
    settled: On<Add<Settled>>,
    turns: Query<(&ForAgent, &TurnOutcome)>,
    mut replay: ResMut<Replay>,
    time: Res<Time<Real>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((&ForAgent(agent), outcome)) = turns.get(settled.entity) else {
        return;
    };
    if replay.primary != Some(agent) || replay.status != ReplayStatus::Running {
        return;
    }
    if let TurnOutcome::Failed(why) = outcome {
        replay.failed = Some(why.clone());
    }
    if let Some(first) = replay.divergences().first() {
        notices.write(Notice::error(agent, first.to_string()));
        if !replay.flags.continue_live {
            end(&mut replay, ReplayStatus::Stopped, &mut commands);
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
