//! FH-23: turn timeline. Records, for every turn of every agent, when each
//! piece of its work ran, and shows it as a chart (`/timeline`) or writes
//! it in Chrome's trace event format for Perfetto (`/timeline export`).
//!
//! Recording is a handful of observers on the kernel's own markers, so it
//! costs a few map lookups per step of a turn:
//!
//! - a turn is a [`TurnOf`] until it is [`Settled`];
//! - each model or tool call of it ([`CallOf`]) and each plugin effect of
//!   it is a [`Span`] from the moment it is [`Admitted`] to its
//!   [`Outcome`], cut into [`PhaseKind`]s: [`Waiting`] behind an earlier
//!   call, held ([`Holds`]), [`Started`], and for a model call the time to
//!   its first streamed item ([`StreamTiming`]) and the streaming after it;
//! - each hold ([`HoldOf`]) on a call or on the turn is a span of its own,
//!   nested under what it holds: a question for the user ([`Ask`]) is a
//!   wait on the user, any other hold is plugin work;
//! - an agent whose work is recorded under an effect ([`EffectParent`]),
//!   such as the subagent of a `task` call, has its turns nested under that
//!   effect's span.
//!
//! Times are taken from the monotonic clock, as offsets from the turn's
//! start; a frame that finds the wall clock ran ahead of it marks the gap
//! as a suspension instead of stretching what was running. Each finished
//! turn is appended to `turn-timeline/turns.jsonl` in the session's
//! directory and read back when the session is resumed.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use rig_harness::prelude::*;
use rig_harness::rig_core::effect::EffectId;
use serde::{Deserialize, Serialize};

mod view;

pub use view::{Summary, chrome_trace, render};

/// Wall-clock time that ran ahead of the monotonic clock between two
/// frames by more than this was the machine sleeping.
const SUSPEND_MIN: Duration = Duration::from_secs(5);

/// Records every turn's timeline and adds `/timeline`.
#[derive(Default)]
pub struct TurnTimelinePlugin;

impl Plugin for TurnTimelinePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Recorder>()
            .init_resource::<Timelines>()
            .add_systems(Startup, load)
            .add_systems(First, notice_suspension)
            .add_observer(turn_began)
            .add_observer(turn_settled)
            .add_observer(turn_gone)
            .add_observer(call_admitted)
            .add_observer(hold_added)
            .add_observer(question_asked)
            .add_observer(hold_released)
            .add_observer(held)
            .add_observer(unheld)
            .add_observer(waiting)
            .add_observer(unblocked)
            .add_observer(started)
            .add_observer(first_item)
            .add_observer(answered)
            .add_observer(settled)
            .add_observer(nest_agent)
            .add_command(
                "timeline",
                "Where a turn's time went: /timeline [N|all] [expand], \
                 /timeline export FILE [N] [--session]",
                view::timeline,
            );
    }
}

/// What a span stands for.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum SpanKind {
    /// A model call.
    Model,
    /// A tool call.
    Tool,
    /// A question waiting on the user.
    User,
    /// A plugin's hold or effect.
    Plugin,
}

/// What a part of a span was doing.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum PhaseKind {
    /// Admitted, not started, nothing in its way known.
    Queued,
    /// Behind an earlier call of its reply that it must not run beside.
    Waiting,
    /// Held by a plugin or a question.
    Held,
    /// Sent, no item streamed back yet.
    FirstToken,
    /// The model streaming its reply.
    Streaming,
    /// Running.
    Running,
}

/// A part of a span, in milliseconds from its turn's start.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Phase {
    /// What it was doing.
    pub kind: PhaseKind,
    /// Its start.
    pub start_ms: f64,
    /// Its end.
    pub end_ms: f64,
}

/// One piece of a turn's work, in milliseconds from the turn's start.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Span {
    /// Its id in its turn.
    pub id: u32,
    /// The span it is part of, such as the call a question holds.
    pub parent: Option<u32>,
    /// What it is.
    pub kind: SpanKind,
    /// What the chart calls it, such as `shell cargo build`.
    pub label: String,
    /// When it was admitted or began.
    pub start_ms: f64,
    /// When it was answered or ended.
    pub end_ms: f64,
    /// Its parts, in order.
    pub phases: Vec<Phase>,
}

impl Span {
    /// How long it lasted.
    pub fn duration_ms(&self) -> f64 {
        self.end_ms - self.start_ms
    }

    /// How long it spent in phases of `kind`.
    pub fn in_phase_ms(&self, kind: PhaseKind) -> f64 {
        let phases = self.phases.iter().filter(|phase| phase.kind == kind);
        phases.map(|phase| phase.end_ms - phase.start_ms).sum()
    }
}

/// Where a subagent's turn sits in the turn that started it.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Nesting {
    /// The parent's [`AgentId`].
    pub agent: String,
    /// The parent's turn, by [`Turn::index`].
    pub turn: u64,
    /// The span of the call that started the subagent.
    pub span: u32,
    /// How long after the parent's turn began this turn began.
    pub offset_ms: f64,
}

/// A stretch the machine slept through, which no span is stretched by.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Suspension {
    /// When it began, in milliseconds from the turn's start.
    pub at_ms: f64,
    /// How long it slept.
    pub for_ms: f64,
}

/// One turn of one agent, as it ran.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct TurnTimeline {
    /// The agent's [`AgentId`].
    pub agent: String,
    /// The agent's name.
    pub name: String,
    /// The turn's [`Turn::index`].
    pub index: u64,
    /// When it began, in milliseconds since the Unix epoch.
    pub began_unix_ms: u64,
    /// How long it ran, on the monotonic clock.
    pub wall_ms: f64,
    /// How it ended: `answered`, `failed`, `stopped`, or `running` for a
    /// turn still running.
    pub outcome: String,
    /// The call it is nested under, for a subagent's turn.
    pub under: Option<Nesting>,
    /// Its work, in the order it began.
    pub spans: Vec<Span>,
    /// The stretches the machine slept through.
    pub suspended: Vec<Suspension>,
}

/// Every finished turn's timeline, oldest first: those of this run and
/// those saved with the session.
#[derive(Resource, Default, Debug)]
pub struct Timelines {
    /// The timelines.
    pub turns: Vec<TurnTimeline>,
}

impl Timelines {
    /// The finished turns of the agent `agent`, oldest first.
    pub fn of_agent<'a>(&'a self, agent: &'a str) -> impl Iterator<Item = &'a TurnTimeline> {
        self.turns.iter().filter(move |turn| turn.agent == agent)
    }

    /// The subagent turns nested under span `span` of `turn`.
    pub fn nested_under<'a>(
        &'a self,
        turn: &'a TurnTimeline,
        span: u32,
    ) -> impl Iterator<Item = &'a TurnTimeline> {
        self.turns.iter().filter(move |sub| {
            sub.under.as_ref().is_some_and(|under| {
                under.agent == turn.agent && under.turn == turn.index && under.span == span
            })
        })
    }
}

/// A span being recorded.
struct OpenSpan {
    span: Span,
    effect: Option<EffectId>,
    start: Instant,
    end: Option<Instant>,
    phases: Vec<(PhaseKind, Instant, Option<Instant>)>,
}

impl OpenSpan {
    fn new(id: u32, parent: Option<u32>, kind: SpanKind, label: String, now: Instant) -> Self {
        let span = Span {
            id,
            parent,
            kind,
            label,
            start_ms: 0.0,
            end_ms: 0.0,
            phases: Vec::new(),
        };
        Self {
            span,
            effect: None,
            start: now,
            end: None,
            phases: Vec::new(),
        }
    }

    /// Ends the current phase at `at` and begins `kind`, unless the span
    /// ended.
    fn enter(&mut self, kind: PhaseKind, at: Instant) {
        if self.end.is_some() {
            return;
        }
        self.close_phase(at);
        self.phases.push((kind, at, None));
    }

    fn close_phase(&mut self, at: Instant) {
        if let Some((_, _, end @ None)) = self.phases.last_mut() {
            *end = Some(at);
        }
    }

    fn finish(&mut self, at: Instant) {
        if self.end.is_none() {
            self.close_phase(at);
            self.end = Some(at);
        }
    }

    fn current(&self) -> Option<PhaseKind> {
        self.phases.last().map(|(kind, ..)| *kind)
    }
}

/// A turn being recorded.
struct OpenTurn {
    agent: String,
    name: String,
    index: u64,
    began: Instant,
    began_unix_ms: u64,
    under: Option<Nesting>,
    spans: Vec<OpenSpan>,
    suspended: Vec<(Instant, Duration)>,
}

impl OpenTurn {
    fn span_mut(&mut self, id: u32) -> Option<&mut OpenSpan> {
        self.spans.iter_mut().find(|open| open.span.id == id)
    }

    fn add(&mut self, mut span: OpenSpan) -> u32 {
        let id = u32::try_from(self.spans.len()).unwrap_or(u32::MAX);
        span.span.id = id;
        self.spans.push(span);
        id
    }

    /// The turn as it stands at `now`: what still runs ends there.
    fn freeze(&self, now: Instant, outcome: &str) -> TurnTimeline {
        let ms = |at: Instant| at.saturating_duration_since(self.began).as_secs_f64() * 1000.0;
        let spans = self.spans.iter().map(|open| {
            let mut span = open.span.clone();
            span.start_ms = ms(open.start);
            span.end_ms = ms(open.end.unwrap_or(now));
            span.phases = open
                .phases
                .iter()
                .map(|&(kind, start, end)| Phase {
                    kind,
                    start_ms: ms(start),
                    end_ms: ms(end.unwrap_or(now)),
                })
                .filter(|phase| phase.end_ms - phase.start_ms >= 0.5 || kind_matters(phase.kind))
                .collect();
            span
        });
        let suspended = self.suspended.iter().map(|&(at, slept)| Suspension {
            at_ms: ms(at),
            for_ms: slept.as_secs_f64() * 1000.0,
        });
        TurnTimeline {
            agent: self.agent.clone(),
            name: self.name.clone(),
            index: self.index,
            began_unix_ms: self.began_unix_ms,
            wall_ms: ms(now),
            outcome: outcome.to_owned(),
            under: self.under.clone(),
            spans: spans.collect(),
            suspended: suspended.collect(),
        }
    }
}

/// Phases kept even when they took no time.
fn kind_matters(kind: PhaseKind) -> bool {
    matches!(
        kind,
        PhaseKind::Running | PhaseKind::FirstToken | PhaseKind::Streaming
    )
}

/// What is being recorded now.
#[derive(Resource, Default)]
pub struct Recorder {
    /// The running turns, by turn entity.
    turns: HashMap<Entity, OpenTurn>,
    /// The span each effect or hold entity is, by turn entity and id.
    spans: HashMap<Entity, (Entity, u32)>,
    /// The span a subagent's turns go under, by agent entity, and when
    /// that span's turn began.
    nests: HashMap<Entity, (Nesting, Instant)>,
    /// The monotonic and wall-clock times the last frame began.
    last_tick: Option<(Instant, SystemTime)>,
    /// Where finished turns are appended, when the session has a directory.
    file: Option<PathBuf>,
}

impl Recorder {
    /// The running turns' timelines as they stand now.
    pub fn running(&self) -> Vec<TurnTimeline> {
        let now = Instant::now();
        self.turns
            .values()
            .map(|turn| turn.freeze(now, "running"))
            .collect()
    }

    fn open_mut(&mut self, entity: Entity) -> Option<&mut OpenSpan> {
        let &(turn, id) = self.spans.get(&entity)?;
        self.turns.get_mut(&turn)?.span_mut(id)
    }

    /// Opens `span` for `entity` in `turn`.
    fn open(&mut self, entity: Entity, turn: Entity, span: OpenSpan) {
        if self.spans.contains_key(&entity) {
            return;
        }
        if let Some(open) = self.turns.get_mut(&turn) {
            let id = open.add(span);
            self.spans.insert(entity, (turn, id));
        }
    }

    /// The span of effect `id`, or of the nearest effect it was asked for
    /// by that has one, among the effects in flight.
    fn span_of_effect(
        &self,
        id: EffectId,
        effects: &Query<(Entity, &Effect)>,
    ) -> Option<(Entity, u32)> {
        let mut next = Some(id);
        // Effect parents form a tree; the bound only guards against a cycle.
        for _ in 0..32 {
            let wanted = next?;
            let found = self.turns.iter().find_map(|(&turn, open)| {
                let span = open.spans.iter().find(|span| span.effect == Some(wanted))?;
                Some((turn, span.span.id))
            });
            if found.is_some() {
                return found;
            }
            next = effects
                .iter()
                .find(|(_, effect)| effect.id() == wanted)
                .and_then(|(_, effect)| effect.parent);
        }
        None
    }
}

/// Reads the timelines saved with the session.
fn load(
    paths: Option<Res<SessionPaths>>,
    mut recorder: ResMut<Recorder>,
    mut timelines: ResMut<Timelines>,
) {
    let Some(paths) = paths else {
        return;
    };
    let file = paths.path().join("turn-timeline").join("turns.jsonl");
    if let Ok(text) = std::fs::read_to_string(&file) {
        let saved = text
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok());
        timelines.turns.splice(0..0, saved);
    }
    recorder.file = Some(file);
}

/// Marks the stretch the machine slept through in every running turn: the
/// wall clock runs on through a suspension, the monotonic clock does not.
fn notice_suspension(mut recorder: ResMut<Recorder>) {
    let now = (Instant::now(), SystemTime::now());
    if let Some((mono, wall)) = recorder.last_tick {
        let awake = now.0.saturating_duration_since(mono);
        let passed = now.1.duration_since(wall).unwrap_or_default();
        if let Some(slept) = passed
            .checked_sub(awake)
            .filter(|slept| *slept > SUSPEND_MIN)
        {
            for turn in recorder.turns.values_mut() {
                turn.suspended.push((mono, slept));
            }
        }
    }
    recorder.last_tick = Some(now);
}

fn unix_ms(at: SystemTime) -> u64 {
    let since = at.duration_since(UNIX_EPOCH).unwrap_or_default();
    u64::try_from(since.as_millis()).unwrap_or(u64::MAX)
}

/// Opens a turn's timeline, nested under the call that started its agent,
/// if one did.
fn turn_began(
    add: On<Add<TurnOf>>,
    turns: Query<(&TurnOf, &Turn)>,
    agents: Query<(&AgentId, Option<&Name>)>,
    mut recorder: ResMut<Recorder>,
) {
    let Ok((&TurnOf(agent), turn)) = turns.get(add.entity) else {
        return;
    };
    let Ok((id, name)) = agents.get(agent) else {
        return;
    };
    let now = Instant::now();
    let under = recorder.nests.get(&agent).map(|(nesting, began)| Nesting {
        offset_ms: now.saturating_duration_since(*began).as_secs_f64() * 1000.0,
        ..nesting.clone()
    });
    let name = name.map_or_else(|| id.short().to_owned(), |name| name.as_str().to_owned());
    let open = OpenTurn {
        agent: id.0.clone(),
        name,
        index: turn.index,
        began: now,
        began_unix_ms: unix_ms(SystemTime::now()),
        under,
        spans: Vec::new(),
        suspended: Vec::new(),
    };
    recorder.turns.insert(add.entity, open);
}

/// Ends a turn's timeline as the turn settles.
fn turn_settled(
    settled: On<Add<Settled>>,
    outcomes: Query<&TurnOutcome>,
    mut recorder: ResMut<Recorder>,
    mut timelines: ResMut<Timelines>,
) {
    let outcome = match outcomes.get(settled.entity) {
        Ok(TurnOutcome::Answered(_)) => "answered",
        Ok(TurnOutcome::Failed(_)) => "failed",
        Ok(TurnOutcome::Stopped(_)) => "stopped",
        Err(_) => return,
    };
    finish_turn(settled.entity, outcome, &mut recorder, &mut timelines);
}

/// Ends the timeline of a turn despawned before it settled, such as one
/// the app's exit stopped.
fn turn_gone(
    removed: On<Remove<Turn>>,
    mut recorder: ResMut<Recorder>,
    mut timelines: ResMut<Timelines>,
) {
    finish_turn(removed.entity, "stopped", &mut recorder, &mut timelines);
}

fn finish_turn(turn: Entity, outcome: &str, recorder: &mut Recorder, timelines: &mut Timelines) {
    let Some(open) = recorder.turns.remove(&turn) else {
        return;
    };
    recorder.spans.retain(|_, (of, _)| *of != turn);
    let timeline = open.freeze(Instant::now(), outcome);
    if let Some(file) = &recorder.file {
        save(file, &timeline);
    }
    timelines.turns.push(timeline);
}

/// Appends `timeline` to the session's file; a turn that cannot be saved
/// is still shown in this run.
fn save(file: &PathBuf, timeline: &TurnTimeline) {
    let Ok(line) = serde_json::to_string(timeline) else {
        return;
    };
    if let Some(dir) = file.parent() {
        std::fs::create_dir_all(dir).ok();
    }
    let opened = OpenOptions::new().create(true).append(true).open(file);
    if let Ok(mut opened) = opened {
        writeln!(opened, "{line}").ok();
    }
}

/// What a tool call is about, in a few words: its command or path.
fn tool_label(request: &ToolRequest) -> String {
    let function = &request.call.function;
    let arguments = &function.arguments;
    let keys = [
        "command",
        "path",
        "description",
        "pattern",
        "query",
        "agent",
    ];
    let shown = |value: &serde_json::Value| match value {
        serde_json::Value::String(text) => {
            Some(text.split_whitespace().collect::<Vec<_>>().join(" "))
        }
        serde_json::Value::Number(number) => Some(number.to_string()),
        _ => None,
    };
    let named = keys.iter().find_map(|key| shown(arguments.get(*key)?));
    let about = named.or_else(|| arguments.values().find_map(shown));
    let about = about.unwrap_or_default();
    format!("{} {about}", function.name.as_str())
        .trim_end()
        .to_owned()
}

/// Opens a span for each call or plugin effect of a turn as it is
/// admitted, so a call refused before it starts has one too.
fn call_admitted(
    admitted: On<Add<Admitted>>,
    calls: Query<(
        &Effect,
        &CallOf,
        Option<&ToolRequest>,
        Has<EffectRequest>,
        Option<&Name>,
    )>,
    mut recorder: ResMut<Recorder>,
) {
    let Ok((effect, &CallOf(turn), tool, plugin, name)) = calls.get(admitted.entity) else {
        return;
    };
    let (kind, label) = match (tool, plugin) {
        (Some(request), _) => (SpanKind::Tool, tool_label(request)),
        (None, true) => {
            let label = name.map_or("effect", Name::as_str);
            (SpanKind::Plugin, label.to_owned())
        }
        (None, false) => {
            let label = effect.purpose.as_deref().unwrap_or("model");
            (SpanKind::Model, label.to_owned())
        }
    };
    let now = Instant::now();
    let mut span = OpenSpan::new(0, None, kind, label, now);
    span.effect = Some(effect.id());
    span.enter(PhaseKind::Queued, now);
    recorder.open(admitted.entity, turn, span);
}

/// Opens a span for each hold on a call or a turn: a wait on the user for
/// a question, plugin work otherwise.
fn hold_added(
    add: On<Add<HoldOf>>,
    holds: Query<(&HoldOf, Option<&Name>, Option<&Ask>)>,
    mut recorder: ResMut<Recorder>,
) {
    let Ok((&HoldOf(target), name, ask)) = holds.get(add.entity) else {
        return;
    };
    let (turn, parent) = match recorder.spans.get(&target) {
        Some(&(turn, id)) => (turn, Some(id)),
        None if recorder.turns.contains_key(&target) => (target, None),
        None => return,
    };
    let (kind, label) = match ask {
        Some(ask) => (SpanKind::User, format!("you: {}", ask.title)),
        None => {
            let name = name.map_or("hold", Name::as_str);
            (SpanKind::Plugin, name.to_owned())
        }
    };
    let span = OpenSpan::new(0, parent, kind, label, Instant::now());
    recorder.open(add.entity, turn, span);
}

/// Opens a wait on the user for a question that holds nothing, in its
/// agent's running turn.
fn question_asked(
    add: On<Add<Ask>>,
    asks: Query<(&Ask, &ForAgent), Without<HoldOf>>,
    agents: Query<&ActiveTurn>,
    mut recorder: ResMut<Recorder>,
) {
    let Ok((ask, &ForAgent(agent))) = asks.get(add.entity) else {
        return;
    };
    let Ok(turn) = agents.get(agent) else {
        return;
    };
    let label = format!("you: {}", ask.title);
    let span = OpenSpan::new(0, None, SpanKind::User, label, Instant::now());
    recorder.open(add.entity, turn.turn(), span);
}

/// Ends a hold's span as it lets go.
fn hold_released(removed: On<Remove<HoldOf>>, mut recorder: ResMut<Recorder>) {
    if let Some(open) = recorder.open_mut(removed.entity) {
        open.finish(Instant::now());
    }
}

fn held(add: On<Add<Holds>>, mut recorder: ResMut<Recorder>) {
    if let Some(open) = recorder.open_mut(add.entity) {
        open.enter(PhaseKind::Held, Instant::now());
    }
}

fn unheld(removed: On<Remove<Holds>>, mut recorder: ResMut<Recorder>) {
    if let Some(open) = recorder.open_mut(removed.entity)
        && open.current() == Some(PhaseKind::Held)
    {
        open.enter(PhaseKind::Queued, Instant::now());
    }
}

fn waiting(add: On<Add<Waiting>>, mut recorder: ResMut<Recorder>) {
    if let Some(open) = recorder.open_mut(add.entity) {
        open.enter(PhaseKind::Waiting, Instant::now());
    }
}

fn unblocked(removed: On<Remove<Waiting>>, mut recorder: ResMut<Recorder>) {
    if let Some(open) = recorder.open_mut(removed.entity)
        && open.current() == Some(PhaseKind::Waiting)
    {
        open.enter(PhaseKind::Queued, Instant::now());
    }
}

/// A model call starts waiting for its first item; anything else runs.
fn started(add: On<Add<Started>>, sent: Query<&SentTo>, mut recorder: ResMut<Recorder>) {
    let model = sent.get(add.entity).ok();
    let Some(open) = recorder.open_mut(add.entity) else {
        return;
    };
    let now = Instant::now();
    match open.span.kind {
        SpanKind::Model => {
            if let Some(SentTo(connection)) = model
                && open.span.label == "model"
            {
                open.span.label = format!("model {}", connection.spec.id);
            }
            open.enter(PhaseKind::FirstToken, now);
        }
        _ => open.enter(PhaseKind::Running, now),
    }
}

/// A model call streams from its first item on, timed where it came.
fn first_item(
    add: On<Add<StreamTiming>>,
    streams: Query<&StreamTiming>,
    mut recorder: ResMut<Recorder>,
) {
    let Ok(streamed) = streams.get(add.entity) else {
        return;
    };
    let Some(open) = recorder.open_mut(add.entity) else {
        return;
    };
    // The kernel stamps the item on the wall clock as it came; on the
    // monotonic clock it came that long before now.
    let now = Instant::now();
    let ago = SystemTime::now()
        .duration_since(streamed.first)
        .unwrap_or_default();
    let at = now.checked_sub(ago).unwrap_or(now).max(open.start);
    open.enter(PhaseKind::Streaming, at);
}

/// An effect or a question is answered.
fn answered(add: On<Add<Outcome>>, mut recorder: ResMut<Recorder>) {
    if let Some(open) = recorder.open_mut(add.entity) {
        open.finish(Instant::now());
    }
}

/// An effect settled without an answer, such as one superseded.
fn settled(add: On<Add<Settled>>, mut recorder: ResMut<Recorder>) {
    if let Some(open) = recorder.open_mut(add.entity) {
        open.finish(Instant::now());
    }
}

/// Puts an agent's turns under the span of the effect its work is
/// recorded under, such as the `task` call that started a subagent.
fn nest_agent(
    insert: On<Insert<EffectParent>>,
    parents: Query<&EffectParent>,
    effects: Query<(Entity, &Effect)>,
    mut recorder: ResMut<Recorder>,
) {
    let agent = insert.entity;
    let Ok(&EffectParent(id)) = parents.get(agent) else {
        return;
    };
    let Some((turn, span)) = recorder.span_of_effect(id, &effects) else {
        recorder.nests.remove(&agent);
        return;
    };
    let Some(open) = recorder.turns.get(&turn) else {
        return;
    };
    let nesting = Nesting {
        agent: open.agent.clone(),
        turn: open.index,
        span,
        offset_ms: 0.0,
    };
    let began = open.began;
    recorder.nests.insert(agent, (nesting, began));
}

#[cfg(test)]
mod tests;
