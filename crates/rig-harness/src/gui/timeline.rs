//! What the window's timeline draws: a span for every turn and every call
//! of every agent since the window opened. Nothing here drives the agents:
//! observers of the core's own lifecycle components (a call entity's
//! [`CallOf`], [`Running`], [`Done`], [`ToolCallRun`], [`Backoff`],
//! [`Summarizing`], [`AwaitingApproval`]) note when each began and ended,
//! as the terminal view and the JSON event stream read the same
//! components.

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use bevy::prelude::*;
use rig_core::completion::{AssistantContent, Usage};
use rig_core::effect::EffectId;
use rig_core::message::ToolResult;
use serde_json::{Map, Value};

use crate::core::agent::{CallOf, ModelChoice, ToolCallRun, TurnOf};
use crate::core::approval::AwaitingApproval;
use crate::core::calls::{Done, Running};
use crate::core::compaction::{Summarizing, Summary, summary_text};
use crate::core::recovery::Backoff;
use crate::core::turn::{ModelCall, ModelReply};

/// Spans kept; older ones are dropped first.
const MAX_SPANS: usize = 5000;

/// What a span stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    /// A turn, from the user's message to the reply that ends it.
    Turn,
    /// A model call.
    Model,
    /// A tool call.
    Tool,
    /// A compaction's summary call.
    Summary,
    /// The wait before a retried model call.
    Retry,
    /// Any other call of a turn, such as restoring files.
    Other,
}

/// How a span ended, or that it has not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Outcome {
    Running,
    Done,
    Failed,
    /// Its entity went away before it finished: Esc, a failed turn, exit.
    Stopped,
}

/// What a span knows beyond its times.
pub(super) enum Detail {
    None,
    Model {
        model: String,
        effect: Option<EffectId>,
        usage: Option<Usage>,
        text: String,
        reasoning: usize,
        tool_calls: Vec<String>,
        error: Option<String>,
    },
    Tool {
        name: String,
        arguments: Map<String, Value>,
        result: Option<String>,
        failed: bool,
    },
    Summary {
        messages: usize,
        tokens: u64,
        text: Option<String>,
        error: Option<String>,
    },
    Retry {
        attempt: u32,
        why: String,
    },
}

/// One bar of the timeline.
pub(super) struct Span {
    /// Its id, increasing; the view selects spans by it.
    pub id: u64,
    /// The turn or call entity, while it exists.
    pub entity: Entity,
    /// The agent whose lane it is drawn in.
    pub agent: Entity,
    pub kind: Kind,
    pub label: String,
    pub start: Instant,
    /// When a queued tool call started running.
    pub begun: Option<Instant>,
    pub end: Option<Instant>,
    pub outcome: Outcome,
    /// When the call waited for the user's approval, and until when.
    pub approval: Option<(Instant, Option<Instant>)>,
    pub detail: Detail,
}

impl Span {
    /// When it ended, or `now` while it runs.
    pub fn end_or(&self, now: Instant) -> Instant {
        self.end.unwrap_or(now)
    }
}

/// Every span, oldest first.
#[derive(Resource)]
pub(super) struct Timeline {
    spans: VecDeque<Span>,
    /// The spans still running, by entity.
    open: HashMap<Entity, u64>,
    next: u64,
    /// Set by every change, cleared by the view once drawn.
    pub changed: bool,
    /// When the window opened: the left end of the whole timeline.
    pub opened: Instant,
}

impl Default for Timeline {
    fn default() -> Self {
        Self {
            spans: VecDeque::new(),
            open: HashMap::new(),
            next: 0,
            changed: true,
            opened: Instant::now(),
        }
    }
}

impl Timeline {
    /// Every span, oldest first.
    pub fn spans(&self) -> impl DoubleEndedIterator<Item = &Span> {
        self.spans.iter()
    }

    /// The span `id`, unless it was dropped.
    pub fn get(&self, id: u64) -> Option<&Span> {
        let first = self.spans.front()?.id;
        self.spans
            .get(usize::try_from(id.checked_sub(first)?).ok()?)
    }

    fn get_mut(&mut self, id: u64) -> Option<&mut Span> {
        let first = self.spans.front()?.id;
        self.spans
            .get_mut(usize::try_from(id.checked_sub(first)?).ok()?)
    }

    /// Whether any span is still running, so the bars still grow.
    pub fn any_running(&self) -> bool {
        !self.open.is_empty()
    }

    /// The running span of `entity`.
    fn open_mut(&mut self, entity: Entity) -> Option<&mut Span> {
        let id = *self.open.get(&entity)?;
        self.changed = true;
        self.get_mut(id)
    }

    /// The running span of `entity`, begun now as `kind` when it has none;
    /// a span begun as [`Kind::Other`] takes `kind`.
    fn begin(
        &mut self,
        entity: Entity,
        agent: Entity,
        kind: Kind,
        label: &str,
    ) -> Option<&mut Span> {
        self.changed = true;
        if let Some(&id) = self.open.get(&entity) {
            let span = self.get_mut(id)?;
            if span.kind == Kind::Other {
                span.kind = kind;
            }
            return Some(span);
        }
        if self.spans.len() >= MAX_SPANS
            && let Some(dropped) = self.spans.pop_front()
        {
            self.open.remove(&dropped.entity);
        }
        let id = self.next;
        self.next += 1;
        self.open.insert(entity, id);
        self.spans.push_back(Span {
            id,
            entity,
            agent,
            kind,
            label: label.to_owned(),
            start: Instant::now(),
            begun: None,
            end: None,
            outcome: Outcome::Running,
            approval: None,
            detail: Detail::None,
        });
        self.spans.back_mut()
    }

    /// Ends the running span of `entity` with `outcome`.
    fn finish(&mut self, entity: Entity, outcome: Outcome) {
        let Some(id) = self.open.remove(&entity) else {
            return;
        };
        self.changed = true;
        if let Some(span) = self.get_mut(id) {
            let now = Instant::now();
            span.end = Some(now);
            span.outcome = outcome;
            if let Some((_, answered @ None)) = &mut span.approval {
                *answered = Some(now);
            }
        }
    }
}

/// The agent whose turn the call `call` belongs to.
fn agent_of(call: Entity, calls: &Query<&CallOf>, turns: &Query<&TurnOf>) -> Option<Entity> {
    let &CallOf(turn) = calls.get(call).ok()?;
    turns.get(turn).ok().map(|&TurnOf(agent)| agent)
}

pub(super) fn on_turn_start(
    add: On<Add<TurnOf>>,
    turns: Query<(&TurnOf, Option<&Name>)>,
    mut timeline: ResMut<Timeline>,
) {
    let Ok((&TurnOf(agent), name)) = turns.get(add.entity) else {
        return;
    };
    let label = name.map_or("turn", Name::as_str);
    timeline.begin(add.entity, agent, Kind::Turn, label);
}

pub(super) fn on_turn_end(remove: On<Remove<TurnOf>>, mut timeline: ResMut<Timeline>) {
    timeline.finish(remove.entity, Outcome::Done);
}

/// Any call of a turn: a span named after the call entity, which the
/// observers of its own components then fill in.
pub(super) fn on_call_start(
    add: On<Add<CallOf>>,
    names: Query<&Name>,
    calls: Query<&CallOf>,
    turns: Query<&TurnOf>,
    mut timeline: ResMut<Timeline>,
) {
    let Some(agent) = agent_of(add.entity, &calls, &turns) else {
        return;
    };
    let label = names.get(add.entity).map_or("call", Name::as_str);
    timeline.begin(add.entity, agent, Kind::Other, label);
}

/// A call entity going away: a call that never finished was stopped; a
/// retry's wait ends by going away.
pub(super) fn on_call_end(remove: On<Remove<CallOf>>, mut timeline: ResMut<Timeline>) {
    let outcome = match timeline.open_mut(remove.entity) {
        Some(span) if span.kind == Kind::Retry => Outcome::Done,
        Some(_) => Outcome::Stopped,
        None => return,
    };
    timeline.finish(remove.entity, outcome);
}

pub(super) fn on_model_call(
    add: On<Add<Running<ModelReply>>>,
    calls: Query<&CallOf>,
    turns: Query<&TurnOf>,
    model_calls: Query<&ModelCall>,
    choices: Query<&ModelChoice>,
    mut timeline: ResMut<Timeline>,
) {
    let Some(agent) = agent_of(add.entity, &calls, &turns) else {
        return;
    };
    let model = choices
        .get(agent)
        .map_or_else(|_| "model".to_owned(), |choice| choice.0.clone());
    let effect = model_calls.get(add.entity).ok().map(ModelCall::effect);
    let label = model.rsplit('/').next().unwrap_or(&model).to_owned();
    if let Some(span) = timeline.begin(add.entity, agent, Kind::Model, &label) {
        span.label = label;
        span.detail = Detail::Model {
            model,
            effect,
            usage: None,
            text: String::new(),
            reasoning: 0,
            tool_calls: Vec::new(),
            error: None,
        };
    }
}

pub(super) fn on_model_done(
    add: On<Add<Done<ModelReply>>>,
    replies: Query<&Done<ModelReply>>,
    mut timeline: ResMut<Timeline>,
) {
    let Ok(Done(reply)) = replies.get(add.entity) else {
        return;
    };
    let outcome = match timeline.open_mut(add.entity) {
        Some(Span {
            detail:
                Detail::Model {
                    usage,
                    text,
                    reasoning,
                    tool_calls,
                    error,
                    ..
                },
            ..
        }) => match reply {
            Ok(response) => {
                *usage = Some(response.usage);
                for content in &response.choice {
                    match content {
                        AssistantContent::Text(said) => {
                            if !text.is_empty() {
                                text.push('\n');
                            }
                            text.push_str(&said.text);
                        }
                        AssistantContent::ToolCall(call) => {
                            tool_calls.push(call.function.name.as_str().to_owned());
                        }
                        AssistantContent::Reasoning(thought) => {
                            *reasoning += thought.text.len();
                        }
                        _ => {}
                    }
                }
                *error = response.error.clone();
                if response.error.is_some() {
                    Outcome::Failed
                } else {
                    Outcome::Done
                }
            }
            Err(report) => {
                *error = Some(report.to_string());
                Outcome::Failed
            }
        },
        Some(_) => match reply {
            Ok(_) => Outcome::Done,
            Err(_) => Outcome::Failed,
        },
        None => return,
    };
    timeline.finish(add.entity, outcome);
}

pub(super) fn on_tool_call(
    add: On<Add<ToolCallRun>>,
    runs: Query<(&ToolCallRun, Has<Running<ToolResult>>)>,
    calls: Query<&CallOf>,
    turns: Query<&TurnOf>,
    mut timeline: ResMut<Timeline>,
) {
    let Ok((run, running)) = runs.get(add.entity) else {
        return;
    };
    let Some(agent) = agent_of(add.entity, &calls, &turns) else {
        return;
    };
    let name = run.call.function.name.as_str().to_owned();
    let arguments = run.call.function.arguments.clone();
    let label = match subject(&arguments) {
        Some(subject) => format!("{name} {subject}"),
        None => name.clone(),
    };
    if let Some(span) = timeline.begin(add.entity, agent, Kind::Tool, &label) {
        span.label = label;
        if running {
            span.begun = Some(span.start);
        }
        span.detail = Detail::Tool {
            name,
            arguments,
            result: None,
            failed: false,
        };
    }
}

/// What a tool call is about, for its bar: the path, command, pattern or
/// task its arguments name.
pub(super) fn subject(arguments: &Map<String, Value>) -> Option<String> {
    ["path", "command", "pattern", "title", "query", "url"]
        .iter()
        .find_map(|key| arguments.get(*key)?.as_str())
        .map(|subject| {
            let line = subject.lines().next().unwrap_or_default();
            let mut short: String = line.chars().take(60).collect();
            if short.len() < subject.len() {
                short.push('…');
            }
            short
        })
}

pub(super) fn on_tool_running(add: On<Add<Running<ToolResult>>>, mut timeline: ResMut<Timeline>) {
    if let Some(span) = timeline.open_mut(add.entity)
        && span.begun.is_none()
    {
        span.begun = Some(Instant::now());
    }
}

pub(super) fn on_tool_done(
    add: On<Add<Done<ToolResult>>>,
    results: Query<&Done<ToolResult>>,
    mut timeline: ResMut<Timeline>,
) {
    let Ok(Done(done)) = results.get(add.entity) else {
        return;
    };
    let text = done
        .content
        .iter()
        .filter_map(|content| content.as_text())
        .collect::<Vec<_>>()
        .join("\n");
    match timeline.open_mut(add.entity) {
        Some(Span {
            detail: Detail::Tool { result, failed, .. },
            ..
        }) => {
            *result = Some(text);
            *failed = done.is_error;
        }
        Some(_) => {}
        None => return,
    }
    let outcome = if done.is_error {
        Outcome::Failed
    } else {
        Outcome::Done
    };
    timeline.finish(add.entity, outcome);
}

pub(super) fn on_approval_asked(add: On<Add<AwaitingApproval>>, mut timeline: ResMut<Timeline>) {
    if let Some(span) = timeline.open_mut(add.entity) {
        span.approval = Some((Instant::now(), None));
    }
}

pub(super) fn on_approval_answered(
    remove: On<Remove<AwaitingApproval>>,
    mut timeline: ResMut<Timeline>,
) {
    if let Some(span) = timeline.open_mut(remove.entity)
        && let Some((_, answered @ None)) = &mut span.approval
    {
        *answered = Some(Instant::now());
    }
}

pub(super) fn on_summary_start(
    add: On<Add<Summarizing>>,
    summaries: Query<&Summarizing>,
    calls: Query<&CallOf>,
    turns: Query<&TurnOf>,
    mut timeline: ResMut<Timeline>,
) {
    let (Ok(summarizing), Some(agent)) = (
        summaries.get(add.entity),
        agent_of(add.entity, &calls, &turns),
    ) else {
        return;
    };
    if let Some(span) = timeline.begin(add.entity, agent, Kind::Summary, "summary") {
        span.label = format!("summary of {} messages", summarizing.messages);
        span.detail = Detail::Summary {
            messages: summarizing.messages,
            tokens: summarizing.tokens,
            text: None,
            error: None,
        };
    }
}

pub(super) fn on_summary_done(
    add: On<Add<Done<Summary>>>,
    summaries: Query<&Done<Summary>>,
    mut timeline: ResMut<Timeline>,
) {
    let Ok(Done(Summary(reply))) = summaries.get(add.entity) else {
        return;
    };
    let written = match reply {
        Ok(response) => summary_text(response),
        Err(report) => Err(report.to_string()),
    };
    let failed = written.is_err();
    match timeline.open_mut(add.entity) {
        Some(Span {
            detail: Detail::Summary { text, error, .. },
            ..
        }) => match written {
            Ok(written) => *text = Some(written),
            Err(why) => *error = Some(why),
        },
        Some(_) => {}
        None => return,
    }
    let outcome = if failed {
        Outcome::Failed
    } else {
        Outcome::Done
    };
    timeline.finish(add.entity, outcome);
}

pub(super) fn on_backoff(
    add: On<Add<Backoff>>,
    backoffs: Query<&Backoff>,
    calls: Query<&CallOf>,
    turns: Query<&TurnOf>,
    mut timeline: ResMut<Timeline>,
) {
    let (Ok(backoff), Some(agent)) = (
        backoffs.get(add.entity),
        agent_of(add.entity, &calls, &turns),
    ) else {
        return;
    };
    if let Some(span) = timeline.begin(add.entity, agent, Kind::Retry, "retry") {
        span.label = format!("retry {}", backoff.attempt);
        span.detail = Detail::Retry {
            attempt: backoff.attempt,
            why: backoff.why.clone(),
        };
    }
}
