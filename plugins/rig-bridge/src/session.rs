//! A harness kept running for one conversation, and the handler that makes
//! it a model. A session answers one request at a time: it streams what the
//! model says until the model calls a tool or its turn ends. A tool call is
//! handed to Rig as an ordinary tool call and its MCP request stays open; the
//! request that carries the results answers it, and the same harness turn
//! goes on. What the user said meanwhile is given to the harness as a
//! message of the user's, never as tool output: for its running turn if it
//! takes one, and else as its next prompt, answered in the same reply.
//!
//! Rig owns the conversation. A request continues a session only when its
//! history is the one the session answered plus what came since. Any other
//! request starts a new harness: on the session a harness [saved](crate::saved)
//! when its history stood where the request goes on from, and else with the
//! history written out as text.
//!
//! A reply that is dropped stops the harness's turn, not the harness: the
//! session waits for what the user says next.

use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use base64::Engine;
use rig_harness::prelude::Processes;
use rig_harness::rig_core::completion::{
    AssistantContent, AssistantMessage, CompletionRequest, Effort, FinishReason, Message,
    Reasoning, ToolDefinition, UnsupportedOption, Usage,
};
use rig_harness::rig_core::conversation::STOPPED;
use rig_harness::rig_core::effect::{self, EffectKind, FamilyDescriptor, HandlerDescriptor};
use rig_harness::rig_core::error::{ErrorKind, ErrorReport};
use rig_harness::rig_core::message::{
    Api, CallId, DocumentSourceKind, Image, MimeType, Origin, ToolResult, ToolResultContent,
    UserContent,
};
use rig_harness::rig_core::operation::Finish;
use rig_harness::rig_core::serve::{Dispatch, Reply, Serve};
use serde_json::{Value, json};
use tokio::sync::{Notify, OnceCell, mpsc, oneshot};

use crate::commands::BridgeSignal;
use crate::harness::{
    ContextSnapshot, Event, Harness, Launch, Part, Prompt, Running, Steer, Vendor,
};
use crate::mcp::{Call, Entry, Outcome, Server};
use crate::saved::{self, Record, Reply as Replied, SessionDir, Stand};

#[cfg(test)]
mod tests;

/// How long a reply stays open after a tool call for the calls made beside
/// it. A harness makes parallel calls one request each, so the first cannot
/// say how many follow; a call that arrives later starts the next reply. A
/// reply also stays open until the inference that made the call is counted.
const SETTLE: Duration = Duration::from_millis(50);

/// Maximum time a pending `Began` counter may delay settling an open tool call.
const COUNT_SETTLE: Duration = Duration::from_secs(2);

/// How long a harness told that the tools changed has to list them again.
const RELIST: Duration = Duration::from_secs(5);

/// How long a harness has to say it took a message for its running turn
/// before the message waits to be its next prompt instead.
const STEER: Duration = Duration::from_secs(10);

/// How long a harness has to end a turn it was asked to stop. One that does
/// not is ended itself.
const STOP: Duration = Duration::from_secs(10);

/// Maximum bytes of prior conversation history serialized into a cold
/// session's opening replay. Kept well below harness tail-truncation limits
/// so the latest user turn is never cut off.
pub(crate) const REPLAY_BUDGET: usize = 160 * 1024;

/// Bytes kept from the head and tail when eliding an old tool output.
const TOOL_PREVIEW_BYTES: usize = 1024;

/// Minimum input tokens before consecutive warm prompts with zero cached tokens
/// trigger a cache-drop warning (below this, providers may skip caching).
const CACHE_WARN_MIN_INPUT: u64 = 2048;

/// Prefix of per-session scratch directories created under the bridge root.
const DIR_PREFIX: &str = "rig-bridge-";

/// Per-vendor runtime counters and diagnostics shown by `/bridge status` and
/// `/bridge usage`.
#[derive(Debug, Clone, Default)]
pub(crate) struct VendorStats {
    pub calls: u64,
    pub input_tokens: u64,
    pub cached_input_tokens: u64,
    pub output_tokens: u64,
    pub last_model: Option<String>,
    pub last_context_tokens: Option<u64>,
    pub last_context_limit: Option<u32>,
    pub last_note: Option<String>,
}

/// Summary of one warm idle harness process.
#[derive(Debug, Clone)]
pub(crate) struct WarmSession {
    pub vendor: Vendor,
    pub model: String,
    pub effort: Option<Effort>,
    pub turns: u32,
    pub relists: bool,
    pub context_tokens: Option<u64>,
    pub context_limit: Option<u32>,
}

/// Point-in-time snapshot of the bridge for `/bridge status` and `/bridge usage`.
#[derive(Debug, Clone)]
pub(crate) struct BridgeSnapshot {
    /// Where each conversation's directory is kept.
    pub kept_root: PathBuf,
    pub warm: Vec<WarmSession>,
    pub vendors: BTreeMap<Vendor, VendorStats>,
}

/// What the bridge's models share: where harnesses run and the sessions
/// waiting for their next request.
pub(crate) struct Bridge {
    runtime: tokio::runtime::Handle,
    processes: Processes,
    server: OnceCell<Server>,
    /// Waiting sessions, the most recently used last.
    waiting: Mutex<VecDeque<Session>>,
    /// How many sessions may wait. The least recently used is ended first.
    limit: usize,
    /// The project Rig works in, when it is known.
    project: Option<PathBuf>,
    /// Disk-backed root directory where the probes' scratch dirs live.
    scratch_root: PathBuf,
    /// Where each session's directory is, and stays once its conversation
    /// can go on from it.
    kept_root: PathBuf,
    /// The requests being answered, each under a number: what a session
    /// for it is set up with, its history, and where its reply is written.
    flights: Mutex<(u64, Vec<Flown>)>,
    /// Told when one of them is answered or given up.
    landed: Notify,
    /// Per-vendor usage and diagnostic state.
    stats: Mutex<BTreeMap<Vendor, VendorStats>>,
    /// Optional channel for notices and runtime context window updates.
    signals: Option<mpsc::UnboundedSender<BridgeSignal>>,
}

type Flown = (u64, Setup, Vec<Arc<Message>>, mpsc::UnboundedSender<Wrote>);

/// A request being answered, until it is dropped.
struct Flight<'a>(&'a Bridge, u64);

impl Drop for Flight<'_> {
    fn drop(&mut self) {
        self.0.flights().1.retain(|(number, ..)| *number != self.1);
        self.0.landed.notify_waiters();
    }
}

impl Bridge {
    pub(crate) fn new(
        runtime: tokio::runtime::Handle,
        processes: Processes,
        limit: usize,
        project: Option<PathBuf>,
        scratch_root: PathBuf,
        kept_root: PathBuf,
    ) -> Self {
        Self {
            project,
            runtime,
            processes,
            server: OnceCell::new(),
            waiting: Mutex::default(),
            limit,
            scratch_root,
            kept_root,
            flights: Mutex::default(),
            landed: Notify::new(),
            stats: Mutex::default(),
            signals: None,
        }
    }

    pub(crate) fn with_signals(mut self, signals: mpsc::UnboundedSender<BridgeSignal>) -> Self {
        self.signals = Some(signals);
        self
    }

    pub(crate) fn scratch_root(&self) -> &Path {
        &self.scratch_root
    }

    pub(crate) fn runtime(&self) -> &tokio::runtime::Handle {
        &self.runtime
    }

    fn waiting(&self) -> std::sync::MutexGuard<'_, VecDeque<Session>> {
        self.waiting.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn stats(&self) -> std::sync::MutexGuard<'_, BTreeMap<Vendor, VendorStats>> {
        self.stats.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn flights(&self) -> std::sync::MutexGuard<'_, (u64, Vec<Flown>)> {
        self.flights.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Marks a request for `history` under `setup`, answered into `out`, as
    /// being answered, once the requests it goes on from whose replies were
    /// dropped are given up: the turn of one is being stopped, and its
    /// session is the one to go on in once it waits.
    async fn fly(
        &self,
        setup: &Setup,
        history: &[Arc<Message>],
        out: &mpsc::UnboundedSender<Wrote>,
    ) -> Flight<'_> {
        let earlier = |(_, set_up, flown, written): &Flown| {
            written.is_closed()
                && set_up.fits(setup, true)
                && flown.len() < history.len()
                && flown
                    .iter()
                    .zip(history)
                    .all(|(then, now)| Arc::ptr_eq(then, now) || then == now)
        };
        let landed = async {
            loop {
                let landed = self.landed.notified();
                tokio::pin!(landed);
                landed.as_mut().enable();
                if !self.flights().1.iter().any(earlier) {
                    return;
                }
                landed.await;
            }
        };
        let _ = tokio::time::timeout(STOP.saturating_mul(2), landed).await;
        let mut flights = self.flights();
        flights.0 = flights.0.wrapping_add(1);
        let number = flights.0;
        flights
            .1
            .push((number, setup.clone(), history.to_vec(), out.clone()));
        Flight(self, number)
    }

    /// The kept directory of a saved session that `history` goes on from,
    /// with the program's name for the session and what to send it. The
    /// session is one of the same model and system prompt, and of the same
    /// tools unless a saved session takes others (`retools`).
    fn saved(
        &self,
        vendor: Vendor,
        retools: bool,
        setup: &Setup,
        history: &[Arc<Message>],
    ) -> Option<(SessionDir, String, Vec<Part>)> {
        let system = saved::digest(&setup.system);
        let tools = setup.tools_digest();
        let keys = std::cell::OnceCell::new();
        let asked = |stands: &Stand| {
            let keys = keys.get_or_init(|| saved::prefixes(history.iter().map(|m| digested(m))));
            if keys.get(stands.seen) != Some(&stands.key) {
                return None;
            }
            let after = history.get(stands.seen..)?;
            match &stands.said {
                Replied::Whole(said) => asked(after, false, |reply| {
                    Said::of(&reply.content).digest() == *said
                }),
                Replied::Cut(said) => {
                    asked(after, true, |reply| Said::of(&reply.content).starts(said))
                }
            }
        };
        let (dir, record) = saved::find(&self.kept_root, |record, stands| {
            (record.vendor.as_str(), &record.model, &record.system)
                == (vendor.as_str(), &setup.model, &system)
                && (retools || record.tools == tools)
                && asked(stands).is_some()
        })?;
        let parts = asked(record.stands.as_ref()?)?;
        Some((dir, record.session, parts))
    }

    /// Stops every warm idle harness process. What each saved stays.
    pub(crate) fn reset(&self) -> usize {
        let ended: Vec<Session> = {
            let mut waiting = self.waiting();
            waiting.drain(..).collect()
        };
        let count = ended.len();
        drop(ended);
        count
    }

    /// Removes idle saved sessions from disk, leaving any held by a live
    /// session intact.
    pub(crate) fn prune(&self) -> usize {
        saved::prune(&self.kept_root)
    }

    /// Current warm sessions and per-vendor counters.
    pub(crate) fn snapshot(&self) -> BridgeSnapshot {
        let warm = {
            let waiting = self.waiting();
            waiting
                .iter()
                .map(|session| WarmSession {
                    vendor: session.vendor,
                    model: session.setup.model.clone(),
                    effort: session.setup.effort,
                    turns: session.turns,
                    relists: session.relists,
                    context_tokens: session.last_context_tokens,
                    context_limit: session.context_limit,
                })
                .collect()
        };
        let vendors = self.stats().clone();
        BridgeSnapshot {
            kept_root: self.kept_root.clone(),
            warm,
            vendors,
        }
    }

    fn record_usage(&self, vendor: Vendor, model: &str, usage: &Usage, context_limit: Option<u32>) {
        let mut stats = self.stats();
        let entry = stats.entry(vendor).or_default();
        entry.calls = entry.calls.saturating_add(1);
        entry.last_model = Some(model.to_owned());
        if let Some(input) = usage.input_tokens {
            entry.input_tokens = entry.input_tokens.saturating_add(input);
        }
        if let Some(cached) = usage.cached_input_tokens {
            entry.cached_input_tokens = entry.cached_input_tokens.saturating_add(cached);
        }
        if let Some(output) = usage.output_tokens {
            entry.output_tokens = entry.output_tokens.saturating_add(output);
        }
        if let Some(ctx) = usage.context_tokens() {
            entry.last_context_tokens = Some(ctx);
            entry.last_context_limit = context_limit;
        }
    }

    fn record_note(&self, vendor: Vendor, note: String) {
        let mut stats = self.stats();
        stats.entry(vendor).or_default().last_note = Some(note);
    }

    fn warn(&self, text: String) {
        if let Some(signals) = &self.signals {
            let _ = signals.send(BridgeSignal::Notice {
                agent: None,
                warn: true,
                text,
                release: None,
            });
        }
    }

    fn report_window(&self, vendor: Vendor, model: &str, limit: u32) {
        if let Some(signals) = &self.signals {
            let _ = signals.send(BridgeSignal::Window {
                vendor,
                model: model.to_owned(),
                limit,
            });
        }
    }

    /// The waiting session `history` continues, with what to send it.
    /// Any idle session whose harness has already exited is pruned so the
    /// conversation transparently starts a fresh process.
    fn resume(&self, setup: &Setup, history: &[Arc<Message>]) -> Option<(Session, Next)> {
        let (found, dead) = {
            let mut waiting = self.waiting();
            let (alive, dead): (VecDeque<_>, VecDeque<_>) = waiting
                .drain(..)
                .partition(|session| !session.running.prompts.is_closed());
            *waiting = alive;
            let found = waiting
                .iter()
                .enumerate()
                .rev()
                .find_map(|(index, session)| {
                    (session
                        .setup
                        .fits(setup, session.changes_effort && session.open.is_empty()))
                    .then(|| session.next(history))
                    .flatten()
                    .map(|next| (index, next))
                })
                .and_then(|(index, next)| waiting.remove(index).map(|session| (session, next)));
            (found, dead)
        };
        drop(dead);
        found
    }

    /// Keeps `session` for its next request, ending those past the limit.
    fn keep(&self, session: Session) {
        let ended: Vec<Session> = {
            let mut waiting = self.waiting();
            waiting.push_back(session);
            let over = waiting.len().saturating_sub(self.limit);
            waiting.drain(..over).collect()
        };
        drop(ended);
    }
}

/// What a session was started with. A request with another setup needs
/// another session.
#[derive(Clone, PartialEq)]
struct Setup {
    program: PathBuf,
    model: String,
    effort: Option<Effort>,
    system: String,
    tools: Arc<[Arc<ToolDefinition>]>,
}

impl Setup {
    /// Whether a session with this setup can serve `other`: the same in
    /// all but the tools (which a running harness can be given anew) and,
    /// when `changes_effort` is true, the reasoning effort.
    fn fits(&self, other: &Self, changes_effort: bool) -> bool {
        (&self.program, &self.model, &self.system) == (&other.program, &other.model, &other.system)
            && (changes_effort || self.effort == other.effort)
    }

    /// The digest of the tools' names, for the record of a saved session.
    fn tools_digest(&self) -> String {
        let names: Vec<&str> = self.tools.iter().map(|tool| tool.name.as_str()).collect();
        saved::digest(&names.join("\n"))
    }
}

/// A harness's model as an effect handler.
#[derive(Clone)]
pub(crate) struct Model {
    pub bridge: Arc<Bridge>,
    pub harness: Arc<dyn Harness>,
    /// The harness's program.
    pub program: PathBuf,
    /// The catalog vendor the model is filed under.
    pub vendor: Vendor,
    /// The model's id, as the harness names it.
    pub model: String,
    /// The model's catalog input limit, when known.
    pub context_limit: Option<u32>,
}

impl Serve for Model {
    type Family = effect::family::Completion;

    fn descriptor(&self) -> HandlerDescriptor {
        let label = format!("{}/{}", self.vendor, self.model);
        HandlerDescriptor {
            key: effect::model_key(&label),
            family: FamilyDescriptor::Completion {
                model: label.into(),
                capabilities: Default::default(),
            },
            layers: Vec::new(),
        }
    }

    /// Answers a completion on the session its history continues, or a new
    /// one. Dropping the reply ends the harness.
    async fn serve(&self, kind: EffectKind, _dispatch: Dispatch) -> Reply {
        let EffectKind::Completion { request, .. } = kind else {
            return Reply::Outcome(Err(ErrorReport::new(
                ErrorKind::Internal,
                format!("a model was sent {}", kind.name()),
            )));
        };
        let (out, mut written) = mpsc::unbounded_channel();
        self.bridge.runtime.spawn(self.clone().run(request, out));
        let origin = Origin::new(
            Api::from_static("bridge"),
            self.vendor.as_str(),
            self.model.clone(),
        );
        Reply::written(origin, |mut writer| async move {
            while let Some(wrote) = written.recv().await {
                let sent = match wrote {
                    Wrote::Text(text) => writer.text(text).await,
                    Wrote::Reasoning(text) => writer.reasoning(text).await,
                    Wrote::Call { name, arguments } => writer.tool_call(name, arguments).await,
                    Wrote::Finish(finish) => {
                        let _ = writer.finish(*finish).await;
                        return;
                    }
                    Wrote::Failed(why) => {
                        let _ = writer
                            .error(ErrorReport::new(ErrorKind::Provider, why))
                            .await;
                        return;
                    }
                };
                if sent.is_err() {
                    return;
                }
            }
        })
    }

    /// A harness takes a prompt, a model and an effort: the sampling,
    /// length, tool choice, output schema and reasoning budget of a request
    /// are left out.
    fn check(&self, kind: &EffectKind) -> Vec<UnsupportedOption> {
        let EffectKind::Completion { request, .. } = kind else {
            return Vec::new();
        };
        left_out(request)
            .into_iter()
            .map(|option| {
                UnsupportedOption::new(
                    option,
                    self.vendor.as_str(),
                    self.model.clone(),
                    "the program takes no such option",
                )
            })
            .collect()
    }
}

/// The options of `request` a harness has no way to take.
fn left_out(request: &CompletionRequest) -> Vec<&'static str> {
    let budgeted = matches!(request.options.reasoning, Some(Reasoning::Budget { .. }));
    [
        ("temperature", request.temperature.is_some()),
        ("max_tokens", request.max_tokens.is_some()),
        ("tool_choice", request.tool_choice.is_some()),
        ("output_schema", request.output_schema.is_some()),
        ("reasoning", budgeted),
    ]
    .into_iter()
    .filter_map(|(option, set)| set.then_some(option))
    .collect()
}

/// What a step writes into its reply.
enum Wrote {
    Text(String),
    Reasoning(String),
    Call { name: String, arguments: Value },
    Finish(Box<Finish>),
    Failed(String),
}

impl Model {
    async fn run(self, request: Arc<CompletionRequest>, out: mpsc::UnboundedSender<Wrote>) {
        let _ = out.send(match self.answer(&request, &out).await {
            Some(Ok(finish)) => Wrote::Finish(Box::new(finish)),
            Some(Err(why)) => Wrote::Failed(why),
            None => return,
        });
    }

    /// Streams one reply into `out` and keeps the session for the next.
    /// `None` when the reply was dropped: the turn it answered is stopped,
    /// and its session waits for what the user says next.
    async fn answer(
        &self,
        request: &CompletionRequest,
        out: &mpsc::UnboundedSender<Wrote>,
    ) -> Option<Result<Finish, String>> {
        let (system, history) = split(&request.chat_history);
        let setup = Setup {
            program: self.program.clone(),
            model: self.model.clone(),
            effort: effort(request),
            system,
            tools: request.tools.iter().cloned().collect(),
        };
        let _flight = self.bridge.fly(&setup, &history, out).await;
        // A reply dropped meanwhile is seen below, with a session to stop:
        // one dropped here would be ended in the middle of its turn.
        let (mut session, mut warm) = match self.open(&setup, &history).await {
            Ok(opened) => opened,
            Err(why) => return Some(Err(why)),
        };
        // Nothing is said of this reply yet, also if it is stopped at once.
        session.said = Said::default();
        let mut emitted = false;
        let finish = loop {
            // A reply that is dropped is stopped, whatever of it is ready.
            let streamed = tokio::select! {
                biased;
                () = out.closed() => None,
                streamed = session.answer(out, &self.bridge, &mut emitted) => Some(streamed),
            };
            match streamed {
                None => {
                    let stopped = session.stop().await;
                    let note = match stopped {
                        true => "stopped a turn: its session waits for what is said next",
                        false => "a turn could not be stopped: its session was ended",
                    };
                    self.bridge.record_note(self.vendor, note.to_owned());
                    if stopped {
                        session.seen = history;
                        session.cut = true;
                        session.stand();
                        self.bridge.keep(session);
                    }
                    return None;
                }
                Some(Ok(finish)) => break finish,
                // Transparently recover once if a session that was gone on
                // in fails before streaming any output (e.g. corrupt session
                // state or sudden exit).
                Some(Err(why)) if warm && !emitted && !is_non_recoverable(&why) => {
                    self.bridge.record_note(
                        self.vendor,
                        format!("restarted warm session after error before output: {why}"),
                    );
                    drop(session);
                    warm = false;
                    session = match self.start(setup.clone(), &history).await {
                        Ok(session) => session,
                        Err(why) => return Some(Err(why)),
                    };
                }
                Some(Err(why)) => return Some(Err(why)),
            }
        };
        session.turns = session.turns.saturating_add(1);
        if let Some(ctx) = finish.usage.context_tokens() {
            session.last_context_tokens = Some(ctx);
        }
        self.bridge.record_usage(
            self.vendor,
            &self.model,
            &finish.usage,
            session.context_limit,
        );
        if session.prompts >= 2
            && let Some(cached) = finish.usage.cached_input_tokens
        {
            if cached > 0 {
                session.consecutive_zero_cache = 0;
                session.last_zero_cache_prompt = 0;
            } else if finish.usage.input_tokens.unwrap_or(0) >= CACHE_WARN_MIN_INPUT
                && session.last_zero_cache_prompt != session.prompts
            {
                session.last_zero_cache_prompt = session.prompts;
                session.consecutive_zero_cache = session.consecutive_zero_cache.saturating_add(1);
                if session.consecutive_zero_cache == 2 && !session.warned_cache_drop {
                    session.warned_cache_drop = true;
                    let input = finish.usage.input_tokens.unwrap_or(0);
                    let note = format!(
                        "2 consecutive warm prompts reported 0 cached tokens ({input} input); prompt prefix may have changed"
                    );
                    self.bridge.record_note(self.vendor, note.clone());
                    self.bridge.warn(format!(
                        "rig-bridge ({}/{}): {note}",
                        self.vendor, self.model
                    ));
                }
            }
        }
        session.seen = history;
        session.cut = false;
        // With a call open the harness is in a turn: nothing goes on from
        // what it saved until the turn ends.
        if session.open.is_empty() {
            session.stand();
        }
        self.bridge.keep(session);
        Some(Ok(finish))
    }

    /// A session sent what `history` goes on with, and whether it went on
    /// in one that was there: a waiting session, else one a harness saved,
    /// else a new one, sent the whole of `history`.
    async fn open(
        &self,
        setup: &Setup,
        history: &[Arc<Message>],
    ) -> Result<(Session, bool), String> {
        // A session that cannot take what is next, its harness gone or a
        // call given up, is ended: the conversation goes on in a new one.
        if let Some((mut session, next)) = self.bridge.resume(setup, history) {
            let is_prompt = matches!(next, Next::Prompt(_));
            if is_prompt {
                session.setup.effort = setup.effort;
            }
            // The harness is told of other tools before it goes on, or is
            // ended when it cannot take them.
            if session.retool(&setup.tools).await && session.send(next).await.is_ok() {
                if is_prompt {
                    session.prompts = session.prompts.saturating_add(1);
                }
                return Ok((session, is_prompt));
            }
        }
        let retools = self.harness.retools_saved();
        if let Some((dir, name, prompt)) = self.bridge.saved(self.vendor, retools, setup, history)
            && let Ok(session) = self.launch(setup.clone(), dir, Some(name), prompt).await
        {
            let note = "went on from a saved session".to_owned();
            self.bridge.record_note(self.vendor, note);
            return Ok((session, true));
        }
        Ok((self.start(setup.clone(), history).await?, false))
    }

    /// A new session, sent `history` as its first prompt.
    async fn start(&self, setup: Setup, history: &[Arc<Message>]) -> Result<Session, String> {
        let dir = SessionDir::create(&self.bridge.kept_root)
            .map_err(|why| format!("could not make a directory: {why}"))?;
        let (prompt, trimmed) = budgeted_opening(history, REPLAY_BUDGET);
        if let Some(summary) = trimmed {
            self.bridge
                .record_note(self.vendor, format!("replay trimmed: {summary}"));
            self.bridge.warn(format!(
                "rig-bridge ({}/{}): history replay trimmed ({summary}); latest turn kept intact",
                self.vendor, self.model
            ));
        }
        self.launch(setup, dir, None, prompt).await
    }

    /// A session in `dir`, sent `prompt`: on the session the harness saved
    /// there as `saved`, or on a new one.
    async fn launch(
        &self,
        setup: Setup,
        dir: SessionDir,
        saved: Option<String>,
        prompt: Vec<Part>,
    ) -> Result<Session, String> {
        let server = self
            .bridge
            .server
            .get_or_try_init(Server::start)
            .await
            .map_err(|why| format!("could not serve the tools: {why}"))?;
        let (entry, calls) = server.open(setup.tools.clone());
        let note = self.harness.note(self.bridge.project.as_deref());
        let system = match (setup.system.as_str(), note.as_str()) {
            (system, "") => system.to_owned(),
            ("", note) => note.to_owned(),
            (system, note) => format!("{system}\n\n{note}"),
        };
        let running = self.harness.start(Launch {
            command: self.bridge.processes.launcher().command(&setup.program),
            dir: dir.path(),
            resume: saved.as_deref(),
            system: &system,
            model: &setup.model,
            effort: setup.effort,
            mcp: &entry,
            tools: &setup.tools,
        })?;
        let mut session = Session {
            vendor: self.vendor,
            setup,
            relists: self.harness.relists(),
            changes_effort: self.harness.changes_effort(),
            turns: 0,
            prompts: 1,
            sent: 0,
            consecutive_zero_cache: 0,
            last_zero_cache_prompt: 0,
            warned_cache_drop: false,
            last_context_tokens: None,
            context_limit: self.context_limit,
            seen: Vec::new(),
            said: Said::default(),
            cut: false,
            open: Vec::new(),
            queued: Vec::new(),
            calls,
            running,
            entry,
            saved,
            dir,
        };
        session.send(Next::Prompt(prompt)).await?;
        Ok(session)
    }
}

/// The reasoning effort a request asks for.
fn effort(request: &CompletionRequest) -> Option<Effort> {
    match request.options.reasoning {
        Some(Reasoning::Effort(effort)) => Some(effort),
        Some(Reasoning::Off | Reasoning::Budget { .. }) | None => None,
        Some(_) => None,
    }
}

/// A request's system prompt and the conversation after it.
fn split(messages: &[Arc<Message>]) -> (String, Vec<Arc<Message>>) {
    let mut system = String::new();
    let mut history = Vec::with_capacity(messages.len());
    for message in messages {
        match &**message {
            Message::System { content } => {
                if !system.is_empty() {
                    system.push_str("\n\n");
                }
                system.push_str(content);
            }
            Message::User { .. } | Message::Assistant(_) => history.push(message.clone()),
        }
    }
    (system, history)
}

/// A probe's private directory, removed with it.
pub(crate) struct Dir(PathBuf);

impl Dir {
    pub(crate) fn create(root: &Path) -> std::io::Result<Self> {
        let mut builder = std::fs::DirBuilder::new();
        builder.recursive(true);
        #[cfg(unix)]
        std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
        builder.create(root)?;

        let path = root.join(format!(
            "{DIR_PREFIX}{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        builder.recursive(false);
        builder.create(&path)?;
        Ok(Self(path))
    }

    pub(crate) fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Removes leftover `rig-bridge-<pid>-*` directories whose owning process is
/// no longer alive.
pub(crate) fn reap_stale_dirs(root: &Path) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    let me = std::process::id();
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some(rest) = name.strip_prefix(DIR_PREFIX) else {
            continue;
        };
        let Some((pid_str, _)) = rest.split_once('-') else {
            continue;
        };
        let Ok(pid) = pid_str.parse::<u32>() else {
            continue;
        };
        if pid == me || process_alive(pid) {
            continue;
        }
        let _ = std::fs::remove_dir_all(&path);
    }
}

#[cfg(target_os = "linux")]
pub(crate) fn process_alive(pid: u32) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}

#[cfg(target_os = "macos")]
pub(crate) fn process_alive(pid: u32) -> bool {
    std::process::Command::new("/bin/kill")
        .env_clear()
        .args(["-0", &pid.to_string()])
        .output()
        .map_or(true, |output| {
            output.status.success()
                || String::from_utf8_lossy(&output.stderr).contains("Operation not permitted")
        })
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(crate) fn process_alive(_pid: u32) -> bool {
    true
}

/// Errors on a warm prompt that represent deterministic rejections or
/// cancellation rather than stale-session corruption and should not be retried.
fn is_non_recoverable(why: &str) -> bool {
    matches!(
        why,
        "the model refused to answer" | STOPPED_TURN | "the tool server closed"
    )
}

/// Why a reply failed whose turn the harness says was stopped, not by Rig.
const STOPPED_TURN: &str = "the turn was stopped";

/// What a waiting session is sent to go on.
#[derive(Debug, PartialEq)]
enum Next {
    /// The next prompt.
    Prompt(Vec<Part>),
    /// The results of its open calls, in the calls' order, and what the
    /// user said beside them.
    Results {
        outcomes: Vec<Outcome>,
        said: Vec<Part>,
    },
}

/// The text and tool calls of a reply.
#[derive(Default, PartialEq)]
struct Said {
    text: String,
    calls: Vec<(String, serde_json::Map<String, Value>)>,
}

impl Said {
    /// What an assistant message in a history says.
    fn of(content: &[AssistantContent]) -> Self {
        let mut said = Self::default();
        for content in content {
            match content {
                AssistantContent::Text(text) => said.text.push_str(&text.text),
                AssistantContent::ToolCall(call) => said.calls.push((
                    call.function.name.as_str().to_owned(),
                    call.function.arguments.clone(),
                )),
                _ => {}
            }
        }
        said
    }

    /// Whether this is what Rig kept of a reply the user stopped after
    /// `text` was said: the start of that text, and no tool call, which Rig
    /// keeps none of.
    fn starts(&self, text: &str) -> bool {
        self.calls.is_empty() && text.starts_with(&self.text)
    }

    /// The digest of it, for the record of a saved session.
    fn digest(&self) -> String {
        saved::digest(&json!([self.text, self.calls]).to_string())
    }
}

/// A running harness and where its conversation stands.
struct Session {
    vendor: Vendor,
    setup: Setup,
    /// Whether the harness takes a changed tool list while it runs.
    relists: bool,
    /// Whether the harness can change reasoning effort between prompts.
    changes_effort: bool,
    /// How many steps this process has completed.
    turns: u32,
    /// How many user prompts (`Next::Prompt`) have been sent to this process.
    prompts: u32,
    /// How many prompts the harness was sent, the session's own among
    /// them: the number of the one it answers, by which its turn is stopped.
    sent: u64,
    /// Consecutive warm user prompts reporting 0 cached tokens.
    consecutive_zero_cache: u8,
    /// Prompt number most recently counted toward `consecutive_zero_cache`.
    last_zero_cache_prompt: u32,
    /// Whether we already warned about repeated warm prompts reporting 0 cached tokens.
    warned_cache_drop: bool,
    /// Context tokens reported on the last completed reply.
    last_context_tokens: Option<u64>,
    /// Input/context limit known from the catalog or reported by the harness.
    context_limit: Option<u32>,
    /// The history of the last request this session answered. The reply to
    /// it is the message after.
    seen: Vec<Arc<Message>>,
    /// What the last reply said: its text, and each tool call's name and
    /// arguments.
    said: Said,
    /// Whether the user stopped the last reply: what Rig has of it, if
    /// anything, is then not what the harness has.
    cut: bool,
    /// The tool calls Rig is running, in the order the reply made them.
    open: Vec<oneshot::Sender<Outcome>>,
    /// What the user said beside tool results that the harness did not take
    /// for its running turn: its next prompt, once that turn ends.
    queued: Vec<Part>,
    calls: mpsc::UnboundedReceiver<Call>,
    running: Running,
    entry: Entry,
    /// The harness's name for the session it saves, once it said it.
    saved: Option<String>,
    dir: SessionDir,
}

impl Session {
    /// What to send this session for `history`, if `history` is the one it
    /// answered, its reply, and then only what answers that reply: the
    /// results of its open calls, or else the next user messages.
    fn next(&self, history: &[Arc<Message>]) -> Option<Next> {
        let (seen, after) = history.split_at_checked(self.seen.len())?;
        let same = seen
            .iter()
            .zip(&self.seen)
            .all(|(now, then)| Arc::ptr_eq(now, then) || now == then);
        // Two conversations can share a history: the reply tells them apart.
        // Of a reply the user stopped, Rig has the start of what was said.
        let replied = |reply: &AssistantMessage| {
            let reply = Said::of(&reply.content);
            match self.cut {
                true => reply.starts(&self.said.text),
                false => reply == self.said,
            }
        };
        if !same {
            return None;
        }
        if self.open.is_empty() {
            return asked(after, self.cut, replied).map(Next::Prompt);
        }
        let (reply, since) = after.split_first()?;
        let Message::Assistant(reply) = &**reply else {
            return None;
        };
        if since.is_empty() || !replied(reply) {
            return None;
        }
        let mut contents = Vec::new();
        for message in since {
            let Message::User { content } = &**message else {
                return None;
            };
            contents.extend(content);
        }
        let calls: Vec<&CallId> = reply
            .content
            .iter()
            .filter_map(|content| match content {
                AssistantContent::ToolCall(call) => Some(&call.id),
                _ => None,
            })
            .collect();
        if calls.len() != self.open.len() {
            return None;
        }
        let mut outcomes: Vec<Option<Outcome>> = vec![None; calls.len()];
        let mut said = Vec::new();
        for content in contents {
            match content {
                UserContent::ToolResult(result) => {
                    let index = calls.iter().position(|call| **call == result.call)?;
                    let slot = outcomes.get_mut(index)?;
                    if slot.is_some() {
                        return None;
                    }
                    *slot = Some(outcome(result));
                }
                other => said.push(other),
            }
        }
        let outcomes = outcomes.into_iter().collect::<Option<Vec<_>>>()?;
        Some(Next::Results {
            outcomes,
            said: parts(said),
        })
    }

    /// Offers the harness `said` as a message for its running turn: what
    /// tells whether it took it, if it takes such messages at all.
    fn steer(&self, said: &[Part]) -> Option<oneshot::Receiver<Result<(), String>>> {
        let (taken, held) = oneshot::channel();
        let steer = Steer {
            parts: said.to_vec(),
            taken,
        };
        self.running.steers.as_ref()?.send(steer).ok()?;
        Some(held)
    }

    /// Offers `tools` in place of the session's, if they differ: whether
    /// the harness now has them. It has once it listed the tools again
    /// after being told they changed; one that is not told, or does not
    /// list them in time, has not.
    async fn retool(&mut self, tools: &Arc<[Arc<ToolDefinition>]>) -> bool {
        if self.setup.tools == *tools {
            return true;
        }
        if !self.relists {
            return false;
        }
        let Some(mut listed) = self.entry.retool(tools.clone()) else {
            return false;
        };
        let relisted = tokio::time::timeout(RELIST, listed.changed()).await;
        if !matches!(relisted, Ok(Ok(()))) {
            return false;
        }
        self.setup.tools = tools.clone();
        true
    }

    /// Writes where the conversation stands beside what the harness saved
    /// of it, so a later process can go on from there: at the history this
    /// session saw and its reply, or nowhere (`stands` false) while the
    /// harness answers, and while it has not been given all the user said.
    /// Whether it is written, where there is something to write: nothing is
    /// before the harness named what it saves.
    fn record(&mut self, stands: bool) -> bool {
        let Some(session) = self.saved.clone() else {
            return true;
        };
        let stands = (stands && self.queued.is_empty()).then(|| Stand {
            seen: self.seen.len(),
            key: saved::prefixes(self.seen.iter().map(|message| digested(message)))
                .pop()
                .unwrap_or_default(),
            said: match self.cut {
                true => Replied::Cut(self.said.text.clone()),
                false => Replied::Whole(self.said.digest()),
            },
        });
        if stands.is_none() && !self.dir.is_kept() {
            return true;
        }
        self.dir.keep(&Record {
            vendor: self.vendor.as_str().to_owned(),
            model: self.setup.model.clone(),
            system: saved::digest(&self.setup.system),
            tools: self.setup.tools_digest(),
            session,
            stands,
        })
    }

    /// Records that the conversation stands where this session last saw
    /// it, its turn ended.
    fn stand(&mut self) {
        self.record(true);
    }

    /// Has the harness end the turn it answers, the reply to which was
    /// dropped: whether the harness can go on from there. It can when the
    /// turn was stopped in time, or was over before it could be; not when
    /// the turn failed, or the harness is gone. The calls it has open are
    /// given up, and so is one it makes meanwhile.
    async fn stop(&mut self) -> bool {
        self.open.clear();
        if self.running.stops.send(self.sent).is_err() {
            return false;
        }
        let ended = async {
            loop {
                tokio::select! {
                    event = self.running.events.recv() => match event {
                        Some(Event::Stopped | Event::Done(Ok(_))) => return true,
                        Some(Event::Done(Err(_))) | None => return false,
                        Some(Event::Saved(name)) => self.saved = Some(name),
                        Some(_) => {}
                    },
                    Some(_) = self.calls.recv() => {}
                }
            }
        };
        let ended = tokio::time::timeout(STOP, ended).await.unwrap_or(false);
        while self.calls.try_recv().is_ok() {}
        ended && !self.running.prompts.is_closed()
    }

    /// Sends the session what is next. What the user said beside tool
    /// results goes first, as a message for the running turn, so the model
    /// reads it with the results. A harness that takes no such message, or
    /// does not say in time that it took this one, gets it as its next
    /// prompt when the turn ends.
    async fn send(&mut self, next: Next) -> Result<(), String> {
        // What the harness saved is ahead of Rig's history from here on: a
        // record that still said otherwise would have the user's next
        // message sent to it twice.
        if !self.record(false) {
            return Err("where the session stands could not be recorded".to_owned());
        }
        match next {
            Next::Prompt(parts) => {
                // What the user said that the harness has not read is first.
                let mut asked = std::mem::take(&mut self.queued);
                asked.extend(parts);
                self.sent += 1;
                // A call that comes before the prompt is of a turn that is
                // over: nothing waits for its answer.
                while self.calls.try_recv().is_ok() {}
                self.running
                    .prompts
                    .send(Prompt {
                        parts: asked,
                        effort: self.setup.effort,
                    })
                    .map_err(|_| "the harness is gone".to_owned())
            }
            Next::Results { outcomes, said } => {
                if !said.is_empty() {
                    let taken = match self.steer(&said) {
                        Some(held) => tokio::time::timeout(STEER, held).await,
                        None => Ok(Ok(Err(String::new()))),
                    };
                    if !matches!(taken, Ok(Ok(Ok(())))) {
                        self.queued.extend(said);
                    }
                }
                let mut given_up = false;
                for (answer, outcome) in self.open.drain(..).zip(outcomes) {
                    given_up |= answer.send(outcome).is_err();
                }
                if given_up {
                    return Err("the harness gave up a tool call".to_owned());
                }
                Ok(())
            }
        }
    }

    /// Streams the harness into `out` until its turn ends or it calls
    /// tools, which stay open. A turn that ends with a message of the
    /// user's waiting goes on into the turn that answers it. Sets `emitted`
    /// once any output or tool call is forwarded to `out`.
    async fn answer(
        &mut self,
        out: &mpsc::UnboundedSender<Wrote>,
        bridge: &Bridge,
        emitted: &mut bool,
    ) -> Result<Finish, String> {
        self.said = Said::default();
        let mut usage = Usage::new();
        let mut usage_reports: u32 = 0;
        let mut last_inference_context: Option<u64> = None;
        let mut context_used: Option<u64> = None;
        let mut settled = None;
        let mut count_deadline = None;
        // Whether an inference began whose counters have not arrived: a
        // harness may call a tool before it finishes reading the message.
        let mut counting = false;
        // Whether the next text starts the answer to a waiting message,
        // after text of the turn before it.
        let mut apart = false;
        loop {
            tokio::select! {
                event = self.running.events.recv() => match event {
                    Some(Event::Text(mut text)) => {
                        if std::mem::take(&mut apart) {
                            text.insert_str(0, "\n\n");
                        }
                        *emitted = true;
                        self.said.text.push_str(&text);
                        let _ = out.send(Wrote::Text(text));
                    }
                    Some(Event::Reasoning(text)) => {
                        *emitted = true;
                        drop(out.send(Wrote::Reasoning(text)));
                    }
                    Some(Event::Began) => counting = true,
                    Some(Event::Saved(name)) => self.saved = Some(name),
                    Some(Event::Usage(counted)) => {
                        if let Some(ctx) = counted.context_tokens().filter(|tokens| *tokens > 0) {
                            last_inference_context = Some(ctx);
                        }
                        usage_reports = usage_reports.saturating_add(1);
                        usage += counted;
                        counting = false;
                        count_deadline = None;
                    }
                    Some(Event::Context(ContextSnapshot { used, window })) => {
                        if let Some(tokens) = window
                            && self.context_limit != Some(tokens)
                        {
                            self.context_limit = Some(tokens);
                            bridge.report_window(self.vendor, &self.setup.model, tokens);
                        }
                        if let Some(tokens) = used {
                            context_used = Some(tokens);
                        }
                    }
                    // A turn that ends with calls open gave them up: Rig
                    // must not run them for a harness that no longer waits.
                    Some(Event::Done(Ok(_))) if !self.open.is_empty() => {
                        return Err("the harness ended its turn before its tools answered".to_owned());
                    }
                    Some(Event::Done(Ok(_))) if !self.queued.is_empty() => {
                        self.send(Next::Prompt(Vec::new())).await?;
                        self.prompts = self.prompts.saturating_add(1);
                        apart = !self.said.text.is_empty();
                    }
                    Some(Event::Done(Ok(reason))) => {
                        return Ok(finish(
                            apply_context(usage, context_used, last_inference_context, usage_reports),
                            reason,
                        ));
                    }
                    Some(Event::Done(Err(why))) => return Err(why),
                    Some(Event::Stopped) => return Err(STOPPED_TURN.to_owned()),
                    None => return Err("the harness exited before it answered".to_owned()),
                },
                call = self.calls.recv() => {
                    let Some(Call { name, arguments, answer }) = call else {
                        return Err("the tool server closed".to_owned());
                    };
                    *emitted = true;
                    self.said.calls.push((name.clone(), arguments.clone()));
                    let _ = out.send(Wrote::Call { name, arguments: Value::Object(arguments) });
                    self.open.push(answer);
                    let now = tokio::time::Instant::now();
                    settled.get_or_insert(now + SETTLE);
                    if counting {
                        count_deadline.get_or_insert(now + COUNT_SETTLE);
                    }
                }
                () = async {
                    match if counting { count_deadline } else { settled } {
                        Some(at) => tokio::time::sleep_until(at).await,
                        None => std::future::pending().await,
                    }
                } => {
                    return Ok(finish(
                        apply_context(usage, context_used, last_inference_context, usage_reports),
                        FinishReason::ToolCalls,
                    ));
                }
            }
        }
    }
}

/// What the user said after a reply, as the next prompt: `after` is the
/// reply, which `replied` knows, and then the user's messages, with no tool
/// result among them. Where the user `cut` the reply off, Rig may have kept
/// none of it, and its note that the reply was stopped comes next: a
/// history without the note is of a conversation whose reply was whole,
/// which a harness that has part of it cannot go on.
fn asked(
    after: &[Arc<Message>],
    cut: bool,
    replied: impl FnOnce(&AssistantMessage) -> bool,
) -> Option<Vec<Part>> {
    let (first, since) = after.split_first()?;
    let since = match &**first {
        Message::Assistant(reply) if replied(reply) => since,
        Message::User { .. } if cut => after,
        _ => return None,
    };
    let stopped = |message: &Arc<Message>| {
        matches!(&**message, Message::User { content }
            if matches!(content.as_slice(), [UserContent::Text(text)] if text.text == STOPPED))
    };
    if cut && !since.first().is_some_and(stopped) {
        return None;
    }
    let mut contents = Vec::new();
    for message in since {
        let Message::User { content } = &**message else {
            return None;
        };
        contents.extend(content);
    }
    let results = |content: &&UserContent| matches!(content, UserContent::ToolResult(_));
    let asks = !contents.is_empty() && !contents.iter().any(results);
    asks.then(|| parts(contents))
}

/// A message as text for the digest of a history: all of it, each part in
/// its place, as Rig saves it.
fn digested(message: &Message) -> String {
    serde_json::to_string(message).unwrap_or_else(|_| format!("{message:?}"))
}

/// Ensures `usage.total_tokens` (read by Rig's `LastUsage::context()` and
/// compaction) reflects the single latest inference's context fill rather than
/// a sum across multiple inferences or tool-call rounds.
fn apply_context(
    mut usage: Usage,
    context_used: Option<u64>,
    last_inference_context: Option<u64>,
    usage_reports: u32,
) -> Usage {
    if let Some(used) = context_used {
        usage.total_tokens = Some(used);
    } else if usage_reports > 1
        && let Some(ctx) = last_inference_context
    {
        usage.total_tokens = Some(ctx);
    }
    usage
}

fn finish(usage: Usage, reason: FinishReason) -> Finish {
    Finish {
        usage,
        reason: Some(reason),
        ..Finish::default()
    }
}

/// A tool result as MCP content.
fn outcome(result: &ToolResult) -> Outcome {
    let content = result
        .content
        .iter()
        .map(|content| match content {
            ToolResultContent::Text(text) => block(&Part::Text(text.text.clone())),
            ToolResultContent::Image(image) => block(&picture(image)),
            ToolResultContent::Json { value } => block(&Part::Text(value.to_string())),
        })
        .collect();
    Outcome {
        content,
        is_error: result.is_error,
    }
}

/// A part as an MCP content block.
fn block(part: &Part) -> Value {
    match part {
        Part::Text(text) => json!({ "type": "text", "text": text }),
        Part::Image { media_type, data } => {
            json!({ "type": "image", "data": data, "mimeType": media_type })
        }
    }
}

/// An image as a part, or a note in its place when its bytes are elsewhere.
fn picture(image: &Image) -> Part {
    let data = match &image.data {
        DocumentSourceKind::Base64(data) => Some(data.clone()),
        DocumentSourceKind::Raw(bytes) => {
            Some(base64::engine::general_purpose::STANDARD.encode(bytes))
        }
        _ => None,
    };
    match (data, &image.media_type) {
        (Some(data), Some(media_type)) => Part::Image {
            media_type: media_type.to_mime_type().to_owned(),
            data,
        },
        _ => Part::Text("[an image that could not be sent]".to_owned()),
    }
}

/// User content as prompt parts. Content with no text form is named.
fn parts<'a>(contents: impl IntoIterator<Item = &'a UserContent>) -> Vec<Part> {
    contents
        .into_iter()
        .map(|content| match content {
            UserContent::Text(text) => Part::Text(text.text.clone()),
            UserContent::Image(image) => picture(image),
            UserContent::ToolResult(result) => Part::Text(told(result)),
            UserContent::Audio(_) => Part::Text("[audio that could not be sent]".to_owned()),
            UserContent::Video(_) => Part::Text("[a video that could not be sent]".to_owned()),
            UserContent::Document(_) => {
                Part::Text("[a document that could not be sent]".to_owned())
            }
        })
        .collect()
}

/// A tool result as transcript text.
fn told(result: &ToolResult) -> String {
    format!("[{}]\n{}", result_title(result), returned(result))
}

/// What a tool result is a result of, and whether it is an error.
fn result_title(result: &ToolResult) -> String {
    let error = if result.is_error { ", an error" } else { "" };
    format!("result of `{}`{error}", result.name.as_str())
}

/// What a tool returned, as text.
fn returned(result: &ToolResult) -> String {
    let mut text = String::new();
    for content in &result.content {
        match content {
            ToolResultContent::Text(said) => text.push_str(&said.text),
            ToolResultContent::Image(_) => text.push_str("[an image]"),
            ToolResultContent::Json { value } => text.push_str(&value.to_string()),
        }
        text.push('\n');
    }
    text
}

/// One atomic replay unit: pieces of `(text, can_elide_as_tool_result)`.
struct ReplayUnit(Vec<(String, bool)>);

impl ReplayUnit {
    fn byte_len(&self) -> usize {
        self.0.iter().map(|(text, _)| text.len()).sum()
    }
}

/// Elides the middle of a large tool output while keeping a UTF-8-safe head
/// and tail preview plus the original byte count.
fn elide_tool_output(text: &str) -> String {
    if text.len() <= TOOL_PREVIEW_BYTES.saturating_mul(2).saturating_add(128) {
        return text.to_owned();
    }
    let head = text
        .get(..text.floor_char_boundary(TOOL_PREVIEW_BYTES))
        .unwrap_or_default();
    let tail_start = text.ceil_char_boundary(text.len().saturating_sub(TOOL_PREVIEW_BYTES));
    let tail = text.get(tail_start..).unwrap_or_default();
    format!(
        "{head}\n... [tool output elided: {} bytes] ...\n{tail}",
        text.len()
    )
}

/// A replayed message as pieces of `(text, can_elide_as_tool_result)`. A
/// tool's output has a heading of its own: it is not the user's.
fn render_message_pieces(message: &Message) -> Vec<(String, bool)> {
    match message {
        Message::User { content } => {
            let mut out = Vec::new();
            let mut users = false;
            for item in content {
                match item {
                    UserContent::ToolResult(result) => {
                        users = false;
                        out.push((format!("## Tool {}\n", result_title(result)), false));
                        out.push((format!("{}\n", returned(result)), true));
                    }
                    other => {
                        if !std::mem::replace(&mut users, true) {
                            out.push(("## User\n".to_owned(), false));
                        }
                        for part in parts([other]) {
                            let line = match part {
                                Part::Text(said) => format!("{said}\n"),
                                Part::Image { .. } => "[an image]\n".to_owned(),
                            };
                            out.push((line, false));
                        }
                    }
                }
            }
            out.push(("\n".to_owned(), false));
            out
        }
        Message::Assistant(reply) => {
            let mut text = String::from("## Assistant\n");
            for content in &reply.content {
                match content {
                    AssistantContent::Text(said) => text.push_str(&said.text),
                    AssistantContent::ToolCall(call) => text.push_str(&format!(
                        "[called `{}` with {}]",
                        call.function.name.as_str(),
                        Value::Object(call.function.arguments.clone()),
                    )),
                    _ => continue,
                }
                text.push('\n');
            }
            text.push('\n');
            vec![(text, false)]
        }
        Message::System { .. } => Vec::new(),
    }
}

fn group_replay_units(messages: &[Arc<Message>]) -> Vec<ReplayUnit> {
    let is_tool_call = |m: &Message| {
        matches!(
            m,
            Message::Assistant(r) if r.content.iter().any(|c| matches!(c, AssistantContent::ToolCall(_)))
        )
    };
    let is_tool_result = |m: &Message| {
        matches!(
            m,
            Message::User { content } if content.iter().any(|c| matches!(c, UserContent::ToolResult(_)))
        )
    };
    let mut units = Vec::new();
    let mut idx = 0;
    while let Some(message) = messages.get(idx) {
        let mut pieces = render_message_pieces(message);
        if is_tool_call(message) {
            while let Some(next) = messages.get(idx + 1)
                && is_tool_result(next)
            {
                idx += 1;
                pieces.extend(render_message_pieces(next));
            }
        }
        if !pieces.is_empty() {
            units.push(ReplayUnit(pieces));
        }
        idx += 1;
    }
    units
}

/// The first prompt of a new session for `history`: the user's last
/// messages, after the conversation before them written out as text under
/// `budget`. The latest user turn is always preserved intact at the end.
pub(crate) fn budgeted_opening(
    history: &[Arc<Message>],
    budget: usize,
) -> (Vec<Part>, Option<String>) {
    let asked = |message: &Arc<Message>| match &**message {
        Message::User { content } => !content
            .iter()
            .any(|content| matches!(content, UserContent::ToolResult(_))),
        _ => false,
    };
    let latest_count = history
        .iter()
        .rev()
        .take_while(|message| asked(message))
        .count();
    let (before, latest) = history.split_at(history.len().saturating_sub(latest_count));

    let mut latest_parts = Vec::new();
    let mut latest_bytes = 0usize;
    for message in latest {
        if let Message::User { content } = &**message {
            for part in parts(content) {
                let bytes = match &part {
                    Part::Text(text) => text.len(),
                    Part::Image { data, .. } => data.len(),
                };
                latest_bytes = latest_bytes.saturating_add(bytes);
                latest_parts.push(part);
            }
        }
    }

    let mut prompt = Vec::new();
    let mut trim_notes = Vec::new();
    if !before.is_empty() {
        let header = "The conversation so far follows between the markers. Continue it: the tool calls \
             in it were yours.\n\n<conversation>\n";
        let footer = if latest.is_empty() {
            "</conversation>\n\nGo on from where the conversation stops."
        } else {
            "</conversation>\n"
        };
        let framing = header.len().saturating_add(footer.len()).saturating_add(96);
        let history_budget = budget.saturating_sub(latest_bytes).saturating_sub(framing);

        let mut units = group_replay_units(before);
        let mut total: usize = units.iter().map(ReplayUnit::byte_len).sum();

        // Pass 1: elide oldest tool results first.
        let mut elided_tools = 0usize;
        if total > history_budget {
            'elide: for unit in &mut units {
                for (piece, can_elide) in &mut unit.0 {
                    if !*can_elide {
                        continue;
                    }
                    let shrunk = elide_tool_output(piece);
                    if shrunk.len() < piece.len() {
                        total = total.saturating_sub(piece.len() - shrunk.len());
                        *piece = shrunk;
                        *can_elide = false;
                        elided_tools += 1;
                        if total <= history_budget {
                            break 'elide;
                        }
                    }
                }
            }
        }
        if elided_tools > 0 {
            trim_notes.push(format!("elided {elided_tools} old tool output(s)"));
        }

        // Pass 2: drop oldest turns while pinning a leading compaction summary.
        let pin_summary = units.first().is_some_and(|unit| {
            unit.0.iter().any(|(piece, _)| {
                piece.contains("The conversation history before this point was compacted")
                    || piece.contains("<summary>")
            })
        });
        let mut kept_summary = pin_summary;
        let mut omitted_turns = 0usize;
        let mut start = usize::from(pin_summary);
        while total > history_budget && start < units.len() {
            if let Some(unit) = units.get(start) {
                total = total.saturating_sub(unit.byte_len());
            }
            start += 1;
            omitted_turns += 1;
        }
        if total > history_budget && kept_summary {
            kept_summary = false;
            omitted_turns += 1;
        }
        if omitted_turns > 0 {
            trim_notes.push(format!("omitted {omitted_turns} earlier turn(s)"));
        }

        let mut text = String::from(header);
        let push_unit = |out: &mut String, unit: &ReplayUnit| {
            for (piece, _) in &unit.0 {
                out.push_str(piece);
            }
        };
        if kept_summary && let Some(summary) = units.first() {
            push_unit(&mut text, summary);
        }
        if omitted_turns > 0 {
            text.push_str(&format!(
                "[{omitted_turns} earlier turn(s) omitted due to replay budget]\n\n"
            ));
        }
        for unit in units.get(start..).unwrap_or_default() {
            push_unit(&mut text, unit);
        }
        text.push_str(footer);
        prompt.push(Part::Text(text));
    }

    prompt.extend(latest_parts);
    let summary = (!trim_notes.is_empty()).then(|| trim_notes.join("; "));
    (prompt, summary)
}
