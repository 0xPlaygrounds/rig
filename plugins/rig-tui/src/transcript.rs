//! The transcript's wrapped rows, kept per message. A frame re-renders and
//! re-wraps only the messages that changed (a message's look also depends
//! on the next message's results and its notices),
//! re-wraps without re-rendering when the width changed (but for a reply
//! with a table, which is fit to the width), and copies out
//! only the rows on screen. Each message keeps where its items are (the
//! message, its calls), so the view can scroll to one, select it and say
//! which is at the top.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::ops::Range;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use bevy_ecs::prelude::*;
use ratatui::style::{Color, Style, Stylize};
use ratatui::text::Line;
use rig_core::completion::{AssistantContent, Message};
use rig_core::conversation;
use rig_core::error::ErrorReport;
use rig_core::message::{AssistantMessage, ToolFunction, ToolName, ToolResult, UserContent};

use super::markdown;
use super::render::{Lights, light, lit};
use super::renderers::{
    AsideView, CallState, ChildCall, ContentBlock, ContentView, DecoratedMessage, EntryView,
    LookLine, MessageView, Pass, Place, Renderers, Role, ToolCallView, plugin_name, result_style,
    result_text, text_rows,
};
use super::theme::Theme;
use super::view::{
    CallItem, DrawnRow, DrawnTranscript, Highlight, ItemRow, MarkIn, RowWrap, ShownNotice,
    TextMark, TranscriptItem,
};
use super::wrap::wrap;
use rig_core::conversation::pairing::answered_by;
use rig_core::conversation::{CallRef, ResultFacts, TurnEnd, attached_file};
use rig_core::effect::{self, EffectId, EffectKind, EffectRecord};
use rig_ecs::agent::{AgentId, NoticeLevel};
use rig_ecs::fork::ForkedFrom;
use rig_ecs::prelude::{Aside, AsideId, Author, AuthorKind, MessageId};
use rig_ecs::stage::{EffectRecorded, Outcome};
use rig_ecs::tools::ToolRequest;
use rig_ecs::turn::{ModelRequest, SentTo};

/// An item's rows (or lines) within a part.
pub(crate) type Marks = Vec<(TranscriptItem, Range<usize>)>;

/// What the view knows of the effects that settled, in this run and, once
/// [`PastEffects`] brought them, before it: how long each tool call ran,
/// by its agent and [`CallRef`]; and which calls each effect started,
/// through any effect between, for [`ToolCallView::children`]. All of it
/// comes from the kernel's effect records. What a call published comes
/// from its result's entry ([`ShownEntry::facts`]), which the session
/// keeps.
#[derive(Resource, Default)]
pub(crate) struct CallFacts {
    /// This run's effects, noted as they settle.
    run: Settled,
    /// The effects before this run, as [`PastEffects`] brought them.
    past: Arc<Settled>,
    /// Each agent's name by its id, as last named: a record names its
    /// agent by id, and a helper may be gone by the time it is drawn.
    names: HashMap<String, String>,
    /// How many times what it knows changed, so a look tells a change by
    /// this alone.
    changes: u64,
    /// The calls under each tool call of an agent that started any, by the
    /// agent's id, as the settled effects say: worked out when first asked
    /// for after a change, not every frame.
    children: Mutex<HashMap<String, Arc<CallsUnder>>>,
}

/// The calls under each tool call that started any, by the call.
pub(crate) type CallsUnder = BTreeMap<CallRef, Vec<ChildCall>>;

impl CallFacts {
    /// How long the call `at` of the agent with the id `agent` ran, as far
    /// as known.
    fn took(&self, agent: Option<&str>, at: CallRef) -> Option<Duration> {
        let key = (Arc::from(agent?), at);
        let mut known = [&self.run, &*self.past].into_iter();
        known.find_map(|settled| settled.ran.get(&key).copied().flatten())
    }

    /// Notes the effect `record` settled in this run.
    fn note(&mut self, record: &EffectRecord) {
        if self.run.note(record) {
            self.changed();
        }
    }

    /// Takes over the effects before this run; a later batch replaces it.
    fn take_past(&mut self, past: Arc<Settled>) {
        self.past = past;
        self.changed();
    }

    /// Notes that the agent with the id `agent` is called `name`; whether
    /// that is new.
    pub(crate) fn name(&mut self, agent: &str, name: &str) -> bool {
        if self.names.get(agent).is_some_and(|known| known == name) {
            return false;
        }
        self.names.insert(agent.to_owned(), name.to_owned());
        self.changed();
        true
    }

    /// Notes a change: what was worked out from before it is gone.
    fn changed(&mut self) {
        self.changes = self.changes.wrapping_add(1);
        self.children
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
    }

    /// The calls under each tool call of the agent with the id `agent`
    /// that started any, as the settled effects and the `live` ones (not
    /// settled) say: each call after the one that started it, in the order
    /// they were made, through any effect between that is no call. What
    /// the settled effects say is worked out once per change; over it, the
    /// tool calls a live effect is under, which change as it runs. Also
    /// what the settled effects say of each agent it was forked from,
    /// `forked_from`.
    pub(crate) fn children(
        &self,
        live: &CallTree,
        agent: &str,
        forked_from: Option<&ForkedFrom>,
    ) -> ChildCalls {
        let settled = self.settled_children(agent);
        let inherited = forked_from.into_iter().flat_map(ForkedFrom::agents);
        let inherited = inherited.map(|from| {
            let children = self.settled_children(from);
            (from.to_owned(), children)
        });
        let inherited = inherited.collect();
        let trees = [&self.past.calls, &self.run.calls, live];
        // Every live effect, and every effect it is under.
        let mut above = HashSet::new();
        for &effect in live.parents.keys() {
            let mut at = Some(effect);
            while let Some(effect) = at.filter(|&effect| above.insert(effect)) {
                at = trees
                    .iter()
                    .find_map(|tree| tree.parents.get(&effect).copied());
            }
        }
        let mut live = CallsUnder::new();
        for parent in above {
            let starts_any = trees.iter().any(|tree| tree.children.contains_key(&parent));
            match tool_call(&trees, parent) {
                Some((of, call)) if **of == *agent && starts_any && !live.contains_key(call) => {
                    live.insert(*call, self.below(&trees, parent));
                }
                _ => {}
            }
        }
        ChildCalls {
            settled,
            live,
            inherited,
        }
    }

    /// What [`Self::children`] works out from the settled effects alone.
    fn settled_children(&self, agent: &str) -> Arc<CallsUnder> {
        let mut known = self.children.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(found) = known.get(agent) {
            return Arc::clone(found);
        }
        let trees = [&self.past.calls, &self.run.calls];
        let mut found = CallsUnder::new();
        for parent in trees.iter().flat_map(|tree| tree.children.keys()) {
            match tool_call(&trees, *parent) {
                Some((of, call)) if **of == *agent && !found.contains_key(call) => {
                    found.insert(*call, self.below(&trees, *parent));
                }
                _ => {}
            }
        }
        let found = Arc::new(found);
        known.insert(agent.to_owned(), Arc::clone(&found));
        found
    }

    /// The calls under `parent`, as `trees` say.
    fn below(&self, trees: &[&CallTree], parent: EffectId) -> Vec<ChildCall> {
        let mut out = Vec::new();
        self.add_below(trees, parent, 0, &mut HashSet::from([parent]), &mut out);
        out
    }

    /// Adds the calls under `parent`, `depth` below the tool call, to
    /// `out`, each followed by those under it; an effect that is no call
    /// adds those under it at its own depth. `seen` guards against an
    /// effect named as its own ancestor, and an effect two trees hold
    /// counts once.
    fn add_below(
        &self,
        trees: &[&CallTree],
        parent: EffectId,
        depth: usize,
        seen: &mut HashSet<EffectId>,
        out: &mut Vec<ChildCall>,
    ) {
        let children = trees.iter().filter_map(|tree| tree.children.get(&parent));
        let mut children: Vec<&Child> = children.flatten().collect();
        children.sort_by_key(|child| child.id);
        for child in children {
            if !seen.insert(child.id) {
                continue;
            }
            let Some(call) = &child.call else {
                self.add_below(trees, child.id, depth, seen, out);
                continue;
            };
            let name = child
                .agent
                .as_deref()
                .and_then(|agent| self.names.get(agent));
            out.push(ChildCall {
                depth,
                agent: name.cloned().or_else(|| call.agent.clone()),
                ..call.clone()
            });
            self.add_below(trees, child.id, depth + 1, seen, out);
        }
    }
}

/// The agent (by id) and the call of the tool call effect `id` is, as the
/// first of `trees` that knows says.
fn tool_call<'a>(trees: &[&'a CallTree], id: EffectId) -> Option<&'a (Arc<str>, CallRef)> {
    trees.iter().find_map(|tree| tree.tool_calls.get(&id))
}

/// The calls under each tool call of an agent that started any, as
/// [`CallFacts::children`] works them out: the settled effects' and, over
/// them, those of the tool calls a live effect is under.
#[derive(Default)]
pub(crate) struct ChildCalls {
    settled: Arc<CallsUnder>,
    live: CallsUnder,
    /// What the settled effects say of each agent the drawn one was
    /// forked from, by its id.
    inherited: Vec<(String, Arc<CallsUnder>)>,
}

impl ChildCalls {
    /// The calls under the tool call `at`, if it started any: the drawn
    /// agent's own, or with `from`, those of the agent with that id it was
    /// forked from.
    pub(crate) fn get(&self, at: Option<CallRef>, from: Option<&str>) -> Option<&Vec<ChildCall>> {
        let at = at?;
        let Some(from) = from else {
            return self.live.get(&at).or_else(|| self.settled.get(&at));
        };
        let inherited = self.inherited.iter().find(|(of, _)| of == from);
        inherited.and_then(|(_, children)| children.get(&at))
    }
}

/// Effects that settled, as the view keeps them: each tool call of a
/// reply, by its agent's id and the call, with how long it ran when
/// known; and which calls each effect started.
#[derive(Clone, Default)]
pub(crate) struct Settled {
    ran: HashMap<(Arc<str>, CallRef), Option<Duration>>,
    calls: CallTree,
}

impl Settled {
    /// Notes the effect `record` settled: under the effect that started
    /// it, as a call or as an effect calls go through; and, for a tool
    /// call of a reply, how long it ran. A record read twice counts once:
    /// whether it is new.
    fn note(&mut self, record: &EffectRecord) -> bool {
        if !self.calls.known.insert(record.id) {
            return false;
        }
        let outcome = record.reviewed.as_ref().unwrap_or(&record.outcome);
        let child = match &record.kind {
            EffectKind::Completion { .. } => {
                let model = record.key.as_str().strip_prefix("model:");
                let (failed, answer) = answer(Some(outcome), false);
                Some(ChildCall {
                    depth: 0,
                    call: None,
                    name: model.unwrap_or("model").to_owned(),
                    agent: None,
                    state: CallState::Done,
                    failed,
                    answer,
                })
            }
            EffectKind::ToolCall { name: tool, args } => {
                if let (Some(at), Some(agent)) = (record.call, &record.scope) {
                    self.calls.tool_call(record.id, agent, at);
                    let took = record.finished_ms.zip(record.started_ms);
                    let took = took.map(|(end, start)| end.saturating_sub(start));
                    let took = took.map(Duration::from_millis);
                    self.ran.insert((Arc::clone(agent), at), took);
                }
                let args = serde_json::from_str(args).unwrap_or_default();
                let call = ToolName::new(tool.as_str()).ok();
                let call = call.map(|name| ToolFunction::new(name, args));
                let (failed, answer) = answer(Some(outcome), true);
                Some(ChildCall {
                    depth: 0,
                    call,
                    name: tool.clone(),
                    agent: None,
                    state: CallState::Done,
                    failed,
                    answer,
                })
            }
            _ => None,
        };
        let agent = record.scope.clone();
        self.calls.add(record.parent, record.id, child, agent);
        true
    }
}

/// Effects by the effect that started them (its `parent`), each with its
/// own effect id and, for a tool or model call, the call; the effect each
/// one was started by; and each reply's tool call's agent (by id) and
/// call, by its effect.
#[derive(Clone, Default)]
pub(crate) struct CallTree {
    children: HashMap<EffectId, Vec<Child>>,
    parents: HashMap<EffectId, EffectId>,
    tool_calls: HashMap<EffectId, (Arc<str>, CallRef)>,
    /// Every effect noted, so none is noted twice.
    known: HashSet<EffectId>,
}

/// An effect under the one that started it.
#[derive(Clone)]
struct Child {
    id: EffectId,
    /// The call it is; `None` when calls only go through it.
    call: Option<ChildCall>,
    /// The id of the agent it is for, whose name is looked up when it is
    /// drawn; `None` when the call names its agent itself.
    agent: Option<Arc<str>>,
}

impl CallTree {
    /// Notes the effect `id` as one `parent` started: `call` when it is a
    /// call, `None` when calls only go through it; for the agent with the
    /// id `agent`, when the call does not name it.
    pub(crate) fn add(
        &mut self,
        parent: Option<EffectId>,
        id: EffectId,
        call: Option<ChildCall>,
        agent: Option<Arc<str>>,
    ) {
        if let Some(parent) = parent {
            self.parents.insert(id, parent);
            let child = Child { id, call, agent };
            self.children.entry(parent).or_default().push(child);
        }
    }

    /// Notes that effect `id` is the tool call `at` of the agent with the
    /// id `agent`.
    pub(crate) fn tool_call(&mut self, id: EffectId, agent: &str, at: CallRef) {
        self.tool_calls.insert(id, (Arc::from(agent), at));
    }
}

/// Whether an answer is a failure, and its text: a tool's result
/// (`tool`), else a model's reply, or why it failed; neither before it
/// has one.
pub(crate) fn answer(
    outcome: Option<&Result<effect::Answered, ErrorReport>>,
    tool: bool,
) -> (bool, Option<String>) {
    match (outcome, tool) {
        (None, _) => (false, None),
        (Some(outcome), true) => {
            let result = Outcome(outcome.clone()).tool_result();
            let failed = result.is_error() || result.is_refused();
            (failed, Some(result.output().render()))
        }
        (Some(Err(report)), false) => (true, Some(report.message.clone())),
        (Some(Ok(effect::Answered::Completion(reply))), false) => (false, Some(reply.text())),
        (Some(Ok(_)), false) => (false, None),
    }
}

/// A tool or model call not settled yet as one of
/// [`ToolCallView::children`], standing as `state`, `depth` 0; `None` for
/// any other effect.
pub(crate) fn live_call(
    (request, model, sent): (Option<&ToolRequest>, Option<&ModelRequest>, Option<&SentTo>),
    (agent, outcome): (Option<&Name>, Option<&Outcome>),
    state: CallState,
) -> Option<ChildCall> {
    let connection = sent.map(|SentTo(connection)| connection);
    let connection = connection.or(model.and_then(|model| model.connection.as_ref()));
    let name = match (request, model) {
        (Some(request), _) => request.call.function.name.to_string(),
        (None, Some(_)) => connection.map_or("model".to_owned(), |c| c.spec.reference()),
        (None, None) => return None,
    };
    let (failed, answer) = answer(outcome.map(|Outcome(outcome)| outcome), request.is_some());
    Some(ChildCall {
        depth: 0,
        call: request.map(|request| request.call.function.clone()),
        name,
        agent: agent.map(|name| name.as_str().to_owned()),
        state,
        failed,
        answer,
    })
}

/// Effects that settled before this run, such as a resumed session's, for
/// [`ToolCallView::children`] and each call's running time: read as the
/// kernel's [`EffectRecorded`] records are. `rig_basics::telemetry` sends
/// the session's effect log once it was restored. [`PastEffects::new`]
/// does all the work per record, so make it where the records are read,
/// off the main thread for a long log: the view only takes it over. A
/// later one replaces it.
#[derive(Message, Clone)]
pub struct PastEffects(Arc<Settled>);

impl PastEffects {
    /// The effects `records` settled, in the order they settled.
    #[must_use]
    pub fn new(records: impl IntoIterator<Item = EffectRecord>) -> Self {
        let mut settled = Settled::default();
        for record in records {
            settled.note(&record);
        }
        Self(Arc::new(settled))
    }

    /// How many effects it holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.calls.known.len()
    }

    /// Whether it holds no effect.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.calls.known.is_empty()
    }
}

/// Notes every effect that settled since the last frame, and takes over
/// those [`PastEffects`] brought.
pub(crate) fn note_effects(
    mut recorded: MessageReader<EffectRecorded>,
    mut past: MessageReader<PastEffects>,
    mut facts: ResMut<CallFacts>,
) {
    if recorded.is_empty() && past.is_empty() {
        return;
    }
    for PastEffects(settled) in past.read() {
        facts.take_past(Arc::clone(settled));
    }
    for EffectRecorded(record) in recorded.read() {
        facts.note(record);
    }
}

/// Notes each agent's name by its id, for the records that name it so.
pub(crate) fn note_agent_name(
    named: On<Insert<Name>>,
    agents: Query<(&AgentId, &Name)>,
    mut facts: ResMut<CallFacts>,
) {
    if let Ok((id, name)) = agents.get(named.entity)
        && facts
            .bypass_change_detection()
            .name(id.0.as_str(), name.as_str())
    {
        facts.set_changed();
    }
}

/// What the messages' look depends on besides the messages.
pub(crate) struct Look<'a> {
    /// The looks plugins added.
    pub(crate) renderers: &'a Renderers,
    /// The id of the agent drawn, when it is one of this app's.
    pub(crate) agent: Option<&'a str>,
    /// The agents it was forked from, if it is a fork.
    pub(crate) forked_from: Option<&'a ForkedFrom>,
    /// The state of each call of the running turn.
    pub(crate) calls: &'a BTreeMap<CallRef, CallState>,
    /// The output so far of each of its running calls that reports it.
    pub(crate) outputs: &'a BTreeMap<CallRef, String>,
    /// The notices and notes drawn inside a message, by its index, each
    /// with the content item it goes before.
    pub(crate) inside: HashMap<usize, Vec<(usize, Inside<'a>)>>,
    /// How long settled calls ran, and the calls they started.
    pub(crate) facts: &'a CallFacts,
    /// The transcript's colors.
    pub(crate) theme: &'a Theme,
    /// The calls under each tool call that started any.
    pub(crate) children: &'a ChildCalls,
    /// The labels drawn inside replies' text.
    pub(crate) marks: &'a [TextMark],
    /// Where stops cut turns off.
    pub(crate) stops: &'a Stops,
}

/// Where stops cut the turns of a conversation off, read from its turn
/// records: the last entry of each stopped turn, which the stop is drawn
/// under, and the calls each stop answered. A turn that ended as stopped
/// in its own course, with no last entry recorded, has none.
#[derive(Default, Hash)]
pub(crate) struct Stops {
    at: BTreeSet<MessageId>,
    calls: BTreeSet<CallRef>,
}

impl Stops {
    /// The stops of `conversation`'s turns, among the `entries` drawn, so
    /// a turn a compaction took out keeps its stop.
    pub(crate) fn of(
        conversation: &conversation::Conversation,
        entries: &ShownEntries<'_>,
    ) -> Self {
        let mut stops = Self::default();
        for (_, record) in conversation.turns() {
            let Some(TurnEnd::Stopped {
                at: Some(at),
                calls,
                ..
            }) = record.end
            else {
                continue;
            };
            if entries.of_id(at).is_none() {
                continue;
            }
            stops.at.insert(at);
            stops.calls.extend(calls);
        }
        stops
    }

    /// Whether a stop is drawn under the entry `id`.
    fn under(&self, id: Option<MessageId>) -> bool {
        id.is_some_and(|id| self.at.contains(&id))
    }

    /// Whether a stop answered the call `at`.
    fn answered(&self, at: Option<CallRef>) -> bool {
        at.is_some_and(|at| self.calls.contains(&at))
    }
}

/// What is drawn inside a message, before one of its items.
#[derive(Clone, Copy)]
pub(crate) enum Inside<'a> {
    /// A notice written there, and its index in the notices shown.
    Notice(usize, &'a ShownNotice),
    /// An aside whose entry was taken out, after the start of what stands
    /// for it.
    Aside(&'a ShownAside<'a>),
}

/// An aside a look draws in the conversation of the shown agent: a
/// plugin's note, or one of a kind a render system draws.
pub(crate) struct ShownAside<'a> {
    /// Its id.
    pub(crate) id: AsideId,
    /// The aside.
    pub(crate) aside: &'a Aside,
}

impl ShownAside<'_> {
    /// The aside's item.
    pub(crate) fn item(&self) -> TranscriptItem {
        TranscriptItem::Aside(self.id)
    }

    /// The aside as a render system sees it, in `theme`'s colors.
    pub(crate) fn view(&self, theme: Theme) -> AsideView<'_> {
        AsideView {
            id: self.id,
            aside: self.aside,
            theme,
        }
    }

    /// Its lines in `theme`'s colors; none for a kind nothing draws.
    pub(crate) fn lines(&self, renderers: &Renderers, theme: &Theme) -> Vec<LookLine> {
        renderers.aside(&self.view(*theme)).unwrap_or_default()
    }
}

impl<'a> Look<'a> {
    /// The marks in `text`, each with the span it draws.
    pub(crate) fn marks_in(&self, text: MarkIn) -> Vec<(usize, ratatui::text::Span<'static>)> {
        let marks = self.marks.iter().filter(|mark| mark.r#in == text);
        let span = |mark: &TextMark| ratatui::text::Span::styled(mark.label.clone(), mark.style);
        marks.map(|mark| (mark.at, span(mark))).collect()
    }

    /// The notices and notes inside message `index`.
    fn inside(&self, index: usize) -> &[(usize, Inside<'_>)] {
        self.inside.get(&index).map_or(&[], Vec::as_slice)
    }

    /// Hashes the notices and notes inside message `index`.
    fn hash_inside(&self, index: usize, hasher: &mut DefaultHasher) {
        for (item, inside) in self.inside(index) {
            item.hash(hasher);
            match inside {
                Inside::Notice(at, notice) => {
                    at.hash(hasher);
                    level_code(notice.level).hash(hasher);
                    notice.text.hash(hasher);
                }
                Inside::Aside(shown) => {
                    shown.id.hash(hasher);
                    (&shown.aside.kind, plugin_name(&shown.aside.author)).hash(hasher);
                    serde_json::to_vec(&shown.aside.value)
                        .unwrap_or_default()
                        .hash(hasher);
                }
            }
        }
    }

    /// The id of the agent the drawn one was forked from whose effect
    /// records say how the call `at` ran, its running time and the calls
    /// it started: the one that made its reply, for a call the fork took
    /// over ([`ForkedFrom::made_by`]). `None` for the drawn agent's own.
    fn forked_call(&self, at: CallRef) -> Option<&'a str> {
        self.forked_from?.made_by(at.reply)
    }

    /// Where the call `at` stands: stopped when a stop answered it, else
    /// as its effect says while it has one, else done once it has
    /// `result`.
    fn state(&self, at: Option<CallRef>, result: Option<&ToolResult>) -> CallState {
        if self.stops.answered(at) {
            return CallState::Stopped;
        }
        match at.and_then(|at| self.calls.get(&at)) {
            Some(state) => state.clone(),
            None if result.is_some() => CallState::Done,
            None => CallState::Running(None),
        }
    }

    /// A hash of what entry `index` looks like besides its message: its
    /// notices, a stop under it, and its calls' states, output so far and
    /// running times.
    fn hash_message(&self, index: usize, entry: &ShownEntry<'_>) -> u64 {
        let (id, message) = (entry.id, &**entry.message);
        let hasher = &mut DefaultHasher::new();
        (id, entry.author, entry.archived).hash(hasher);
        self.hash_inside(index, hasher);
        self.stops.under(id).hash(hasher);
        let marks = self.marks.iter();
        marks
            .filter(|mark| matches!(mark.r#in, MarkIn::Message { id: of, .. } if Some(of) == id))
            .for_each(|mark| mark.hash(hasher));
        if let (Message::Assistant(assistant), Some(id)) = (message, id) {
            for (at, _) in CallRef::calls(id, assistant) {
                self.stops.answered(Some(at)).hash(hasher);
                self.calls.get(&at).hash(hasher);
                self.outputs.get(&at).hash(hasher);
                let from = self.forked_call(at);
                self.facts.took(from.or(self.agent), at).hash(hasher);
                hash_children(self.children.get(Some(at), from), hasher);
            }
        }
        hasher.finish()
    }

    /// A hash of all of it but the render systems, whose changes
    /// [`Transcript::clear`] handles.
    fn digest(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        let mut indices: Vec<&usize> = self.inside.keys().collect();
        indices.sort();
        for index in indices {
            index.hash(&mut hasher);
            self.hash_inside(*index, &mut hasher);
        }
        self.calls.hash(&mut hasher);
        self.outputs.hash(&mut hasher);
        self.marks.hash(&mut hasher);
        self.stops.hash(&mut hasher);
        self.theme.hash(&mut hasher);
        // The settled calls' running times and children change only with it.
        self.facts.changes.hash(&mut hasher);
        for (call, children) in &self.children.live {
            call.hash(&mut hasher);
            hash_children(Some(children), &mut hasher);
        }
        hasher.finish()
    }
}

/// Hashes what a call's look draws of `children`.
fn hash_children(children: Option<&Vec<ChildCall>>, hasher: &mut DefaultHasher) {
    for child in children.into_iter().flatten() {
        (child.depth, &child.name, &child.agent, &child.state).hash(hasher);
        (child.failed, &child.answer).hash(hasher);
        let call = child.call.as_ref();
        call.map(|call| (&call.name, call.raw_arguments()))
            .hash(hasher);
    }
}

/// A number for each notice level.
fn level_code(level: NoticeLevel) -> u8 {
    match level {
        NoticeLevel::Info => 0,
        NoticeLevel::Warn => 1,
        NoticeLevel::Error => 2,
    }
}

/// Entries a plugin took out of an agent's conversation that its
/// transcript still draws, by the entry that replaced them (a compaction's
/// summary), each set just above that entry. Drawing starts from the
/// entries in the conversation and follows what each replaced, so a set
/// whose entry is gone, as after a rewind or an undo, draws nothing.
/// Display only: requests never read it.
#[derive(Component, Clone, Debug, Default)]
pub struct Replaced(pub BTreeMap<MessageId, Vec<conversation::Entry>>);

/// An entry the transcript draws: one of the conversation's, or an
/// archived one, which an entry of it replaced ([`Replaced`]).
#[derive(Clone, Copy)]
pub(crate) struct ShownEntry<'a> {
    pub(crate) message: &'a Arc<Message>,
    pub(crate) id: Option<MessageId>,
    pub(crate) author: Option<&'a Author>,
    /// What the calls its results answer published: all a call's row
    /// draws of its facts, live and after a resume alike.
    pub(crate) facts: &'a ResultFacts,
    /// Whether it is the note rig puts after a request the user stopped.
    pub(crate) stop: bool,
    /// Whether it is archived: drawn, but no item to select, rewind to or
    /// fork from.
    pub(crate) archived: bool,
    /// How many archived entries it stands for, drawn above it; an
    /// archived entry that stands for others counts as those.
    pub(crate) stands_for: usize,
}

/// The entries the transcript draws, in order: each entry of the
/// conversation after the archived entries it stands for, each of those
/// after the ones it stands for in turn.
#[derive(Default)]
pub(crate) struct ShownEntries<'a> {
    entries: Vec<ShownEntry<'a>>,
    /// Where each entry of the conversation is among them.
    live: Vec<usize>,
    /// Where each entry with an id is among them.
    ids: HashMap<MessageId, usize>,
}

impl<'a> ShownEntries<'a> {
    /// The entries of `conversation`, with what `replaced` says they stand
    /// for.
    pub(crate) fn new(
        conversation: &'a conversation::Conversation,
        replaced: Option<&'a Replaced>,
    ) -> Self {
        let mut shown = Self::default();
        let mut expanded = HashSet::new();
        for (message, entry) in conversation.messages().iter().zip(conversation.entries()) {
            let stands_for = shown.archive(replaced, entry.id, &mut expanded);
            shown.live.push(shown.entries.len());
            shown.ids.insert(entry.id, shown.entries.len());
            shown.entries.push(ShownEntry {
                message,
                id: Some(entry.id),
                author: entry.author,
                facts: entry.facts,
                stop: entry.is_stop_marker(),
                archived: false,
                stands_for,
            });
        }
        shown
    }

    /// Adds the archived entries `id` stands for, each after those it
    /// stands for in turn; how many. Each id is followed once, so a cycle
    /// ends.
    fn archive(
        &mut self,
        replaced: Option<&'a Replaced>,
        id: MessageId,
        expanded: &mut HashSet<MessageId>,
    ) -> usize {
        let taken = replaced.and_then(|replaced| replaced.0.get(&id));
        let Some(taken) = taken.filter(|_| expanded.insert(id)) else {
            return 0;
        };
        let mut count = 0;
        for entry in taken {
            let below = entry
                .id
                .map_or(0, |id| self.archive(replaced, id, expanded));
            count += below.max(1);
            let stop = entry.is_stop_marker();
            if let Some(id) = entry.id {
                self.ids.insert(id, self.entries.len());
            }
            self.entries.push(ShownEntry {
                message: &entry.message,
                id: entry.id,
                author: entry.author.as_ref(),
                facts: &entry.facts,
                stop,
                archived: true,
                stands_for: below,
            });
        }
        count
    }

    /// How many entries it draws.
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Entry `at` of them.
    pub(crate) fn get(&self, at: usize) -> Option<&ShownEntry<'a>> {
        self.entries.get(at)
    }

    /// Where entry `index` of the conversation is among them.
    pub(crate) fn of_live(&self, index: usize) -> Option<usize> {
        self.live.get(index).copied()
    }

    /// Where the entry `id` is among them, live or archived.
    pub(crate) fn of_id(&self, id: MessageId) -> Option<usize> {
        self.ids.get(&id).copied()
    }

    /// Where the archived entry `id` is among them.
    pub(crate) fn of_archived(&self, id: MessageId) -> Option<usize> {
        let at = self.of_id(id)?;
        self.entries.get(at)?.archived.then_some(at)
    }
}

/// One message's look: its lines and where its items are in them, and
/// those lines wrapped to the width.
struct Cached {
    /// The message as it was laid out.
    message: Arc<Message>,
    /// Its facts then, which the calls of the message before it draw.
    facts: ResultFacts,
    /// [`Look::hash_message`] of it then.
    look: u64,
    lines: Vec<LookLine>,
    line_marks: Marks,
    wrapped: Wrapped,
    /// Whether its lines depend on the width, as a reply's table does, so
    /// a new width renders it again rather than only wrapping it.
    fitted: bool,
}

impl Cached {
    /// Entry `index` of `entries` laid out in `look` at `usable` columns;
    /// `None` past their end.
    fn new(
        entries: &ShownEntries<'_>,
        index: usize,
        look_hash: u64,
        (look, usable): (&Look<'_>, usize),
    ) -> Option<Self> {
        let entry = entries.get(index)?;
        let message = entry.message;
        let previous = index.checked_sub(1).and_then(|at| entries.get(at));
        let previous = previous.map(|previous| &**previous.message);
        let next = entries.get(index + 1);
        let (lines, line_marks) = message_lines((entry, index), (previous, next), (look, usable));
        let wrapped = wrap_marked(&lines, &line_marks, usable);
        Some(Self {
            message: message.clone(),
            facts: entry.facts.clone(),
            look: look_hash,
            lines,
            line_marks,
            wrapped,
            fitted: fitted(message),
        })
    }
}

/// Lines wrapped to a width: the rows, whether each continues the row
/// before it (a line wrapped), and where the items are in the rows. Code
/// is plain in the rows, and colored when they are drawn
/// ([`drawn`](Self::drawn)).
#[derive(Default)]
pub(crate) struct Wrapped {
    rows: Vec<Line<'static>>,
    continues: Vec<Option<RowWrap>>,
    marks: Marks,
    /// The lines with code, by the rows they wrapped to, top to bottom.
    code: Vec<(Range<usize>, LookLine)>,
    width: usize,
}

impl Wrapped {
    /// Rows `range` as drawn: those of a line with code wrapped again from
    /// the line colored, which wraps to the same rows, their text the same.
    fn drawn(&self, range: Range<usize>) -> Vec<Line<'static>> {
        let mut rows = self.rows.get(range.clone()).unwrap_or_default().to_vec();
        let first = self
            .code
            .partition_point(|(rows, _)| rows.end <= range.start);
        let code = self.code.get(first..).unwrap_or_default().iter();
        for (lines, line) in code.take_while(|(lines, _)| lines.start < range.end) {
            let colored = wrap(&line.colored(), self.width);
            for (at, (row, _)) in lines.clone().zip(colored) {
                if let Some(slot) = at
                    .checked_sub(range.start)
                    .and_then(|index| rows.get_mut(index))
                {
                    *slot = row;
                }
            }
        }
        rows
    }
}

/// The wrapped rows of the shown agent's messages.
#[derive(Default)]
pub(crate) struct Transcript {
    agent: Option<Entity>,
    width: u16,
    /// The [`Look::digest`] the rows were made with.
    look: u64,
    messages: Vec<Cached>,
}

/// A part of the transcript, in order: a message's cached rows, or rows
/// made for this frame (notices, a note, the reply streaming in), with
/// where the items among them are.
pub(crate) enum Part {
    Message(usize),
    Rows(Wrapped),
}

impl Part {
    /// `lines`, with the items `marks` says are in them, wrapped to
    /// `width`.
    pub(crate) fn rows(
        lines: &[LookLine],
        marks: &[(TranscriptItem, Range<usize>)],
        width: usize,
    ) -> Self {
        Self::Rows(wrap_marked(lines, marks, width))
    }
}

/// The rows of the transcript on screen, and where it is.
pub(crate) struct Visible {
    /// The rows shown.
    pub(crate) rows: Vec<Line<'static>>,
    /// What is below them.
    pub(crate) below: Below,
    /// The first one's row in the whole transcript.
    pub(crate) first: usize,
    /// The item at the top and the row of it there, while the view is
    /// scrolled up.
    pub(crate) top: Option<(TranscriptItem, usize)>,
}

impl Transcript {
    /// Brings the rows up to date with `entries`, those `agent`'s
    /// transcript draws, and `look` at `width`.
    /// `changed` says whether the entries may have changed since the last
    /// call; when neither they nor the look did, only a new agent or width
    /// costs anything.
    pub(crate) fn update(
        &mut self,
        agent: Entity,
        entries: &ShownEntries<'_>,
        changed: bool,
        look: &Look<'_>,
        width: u16,
    ) {
        let usable = usize::from(width.max(1));
        if look.renderers.pass == Pass::Finish {
            self.finish(entries, look, usable);
            return;
        }
        if self.agent != Some(agent) {
            self.agent = Some(agent);
            self.messages.clear();
        }
        let resized = self.width != width;
        if resized {
            self.width = width;
            for cached in &mut self.messages {
                cached.wrapped = wrap_marked(&cached.lines, &cached.line_marks, usable);
            }
        }
        let refit = resized && self.messages.iter().any(|cached| cached.fitted);
        let digest = look.digest();
        let looks_alike = std::mem::replace(&mut self.look, digest) == digest;
        if !changed && !refit && looks_alike && self.messages.len() == entries.len() {
            return;
        }
        // A message's look depends on it and its neighbours, so each is
        // compared once with the one laid out last time, before any is
        // replaced; a message past either end matches only another.
        let same: Vec<bool> = (0..=entries.len())
            .map(|at| match (self.messages.get(at), entries.get(at)) {
                (Some(cached), Some(entry)) => {
                    Arc::ptr_eq(&cached.message, entry.message) && cached.facts == *entry.facts
                }
                (cached, entry) => cached.is_none() && entry.is_none(),
            })
            .collect();
        self.messages.truncate(entries.len());
        for (index, entry) in entries.entries.iter().enumerate() {
            let look_hash = look.hash_message(index, entry);
            if self
                .messages
                .get(index)
                .is_some_and(|cached| cached.look == look_hash && !(resized && cached.fitted))
                && (index.saturating_sub(1)..=index + 1).all(|at| same.get(at) == Some(&true))
            {
                continue;
            }
            let Some(cached) = Cached::new(entries, index, look_hash, (look, usable)) else {
                continue;
            };
            match self.messages.get_mut(index) {
                Some(slot) => *slot = cached,
                None => self.messages.push(cached),
            }
        }
    }

    /// Lays out again the messages whose presentation the render systems
    /// changed, from their outputs; the rest stand as the collecting pass
    /// left them.
    fn finish(&mut self, entries: &ShownEntries<'_>, look: &Look<'_>, usable: usize) {
        for &index in look.renderers.messages.borrow().iter() {
            let Some(slot) = self.messages.get_mut(index) else {
                continue;
            };
            let look_hash = slot.look;
            if let Some(cached) = Cached::new(entries, index, look_hash, (look, usable)) {
                *slot = cached;
            }
        }
    }

    /// Drops every cached row, after the renderers changed.
    pub(crate) fn clear(&mut self) {
        self.messages.clear();
    }

    /// A part's wrapped rows; `None` for a message not cached.
    fn wrapped<'a>(&'a self, part: &'a Part) -> Option<&'a Wrapped> {
        match part {
            Part::Message(index) => self.messages.get(*index).map(|cached| &cached.wrapped),
            Part::Rows(wrapped) => Some(wrapped),
        }
    }

    /// A part's rows.
    fn rows<'a>(&'a self, part: &'a Part) -> &'a [Line<'static>] {
        let wrapped = self.wrapped(part);
        wrapped
            .map(|wrapped| wrapped.rows.as_slice())
            .unwrap_or_default()
    }

    /// Where a part's items are in its rows.
    fn marks<'a>(&'a self, part: &'a Part) -> &'a [(TranscriptItem, Range<usize>)] {
        let wrapped = self.wrapped(part);
        wrapped
            .map(|wrapped| wrapped.marks.as_slice())
            .unwrap_or_default()
    }

    /// Every item's rows in the whole transcript `parts` make, top to
    /// bottom; a call's are within its message's.
    pub(crate) fn marks_in<'a>(
        &'a self,
        parts: &'a [Part],
    ) -> impl Iterator<Item = (&'a TranscriptItem, Range<usize>)> + 'a {
        let mut start = 0;
        parts.iter().flat_map(move |part| {
            let at = start;
            start += self.rows(part).len();
            let marks = self.marks(part).iter();
            marks.map(move |(item, rows)| (item, rows.start + at..rows.end + at))
        })
    }

    /// The rows of `item` in the transcript `parts` make.
    pub(crate) fn rows_of(&self, parts: &[Part], item: &TranscriptItem) -> Option<Range<usize>> {
        let mut marks = self.marks_in(parts);
        marks
            .find(|(marked, _)| *marked == item)
            .map(|(_, rows)| rows)
    }

    /// The rows of `parts` that fit `height` rows at `width`, where
    /// `scroll` places them, what is below them, and where the items are.
    /// A view scrolled up shows [`Below::text_rows`] rows.
    /// A jump to an item's match counts the matches of `highlights`.
    pub(crate) fn visible(
        &self,
        parts: &[Part],
        (width, height): (u16, usize),
        scroll: &mut Scroll,
        highlights: &[Highlight],
    ) -> Visible {
        let counts: Vec<usize> = parts.iter().map(|part| self.rows(part).len()).collect();
        let lights = self.lights(parts, highlights);
        scroll.find(|item| self.rows_of(parts, item));
        let (mut top, mut below) = scroll.place(&counts, height, width);
        if let Some((item, to)) = scroll.jump.take() {
            let row = self.rows_of(parts, &item).map(|rows| {
                let lit = match to {
                    ItemRow::Lit(n) => self.lit_row(parts, &rows, &lights, n),
                    ItemRow::Top(_) | ItemRow::Reveal => None,
                };
                lit.unwrap_or(rows.start)
            });
            let shown = top..top + below.text_rows(height);
            if let Some(row) = row.filter(|row| !shown.contains(row)) {
                scroll.at = Some(row);
                (top, below) = scroll.place(&counts, height, width);
            }
        }
        let rows = below.text_rows(height);
        let mut shown = Vec::with_capacity(rows);
        let mut start = 0;
        for (part, count) in parts.iter().zip(&counts) {
            let end = start + count;
            if end > top && start < top + rows {
                let from = top.saturating_sub(start);
                let to = (top + rows - start).min(*count);
                shown.extend(self.lit_rows(part, start, from..to, &lights));
            }
            start = end;
            if start >= top + rows {
                break;
            }
        }
        let at_top = scroll.held.is_some().then(|| {
            // The innermost item there: the one that starts last.
            let holding = self.marks_in(parts).filter(|(_, rows)| rows.contains(&top));
            let item = holding.max_by_key(|(_, rows)| rows.start);
            item.map(|(item, rows)| (item.clone(), top - rows.start))
        });
        Visible {
            rows: shown,
            below,
            first: top,
            top: at_top.flatten(),
        }
    }

    /// The transcript of `agent` as `parts` lay it out, with the matches of
    /// `highlights` in each row, counted only when one is set, and
    /// `on_screen` on screen.
    pub(crate) fn laid_out(
        &self,
        agent: Option<Entity>,
        parts: &[Part],
        highlights: &[Highlight],
        on_screen: Range<usize>,
    ) -> DrawnTranscript {
        let items = self.marks_in(parts);
        let items = items.map(|(item, rows)| (item.clone(), rows)).collect();
        let lights = self.lights(parts, highlights);
        let counts = if lights.any() {
            lit(self.joined(parts), 0, &lights)
        } else {
            Vec::new()
        };
        let mut counts = counts.into_iter().map(|row| row.matches);
        let rows = self.joined(parts).map(|(row, continues)| DrawnRow {
            text: row.to_string(),
            continues: continues.cloned(),
            lit: counts.next().unwrap_or_default(),
        });
        DrawnTranscript::new(agent, rows.collect(), items, on_screen)
    }

    /// `highlights`, with where in `parts` their spans are.
    fn lights<'a>(&self, parts: &[Part], highlights: &'a [Highlight]) -> Lights<'a> {
        let mut lights = Lights {
            highlights,
            placed: HashMap::new(),
        };
        if highlights
            .iter()
            .all(|highlight| highlight.spans.is_empty())
        {
            return lights;
        }
        let starts: Vec<usize> = self
            .joined(parts)
            .enumerate()
            .filter(|(_, (_, joint))| joint.is_none())
            .map(|(row, _)| row)
            .collect();
        let items: HashMap<&TranscriptItem, Range<usize>> = self.marks_in(parts).collect();
        for (index, highlight) in highlights.iter().enumerate() {
            for span in &highlight.spans {
                let rows = match &span.item {
                    Some(item) => items.get(item).cloned(),
                    None => Some(0..usize::MAX),
                };
                let Some(rows) = rows else { continue };
                let first = starts.partition_point(|&start| start < rows.start);
                let line = starts
                    .get(first.saturating_add(span.line))
                    .filter(|row| rows.contains(row));
                if let Some(&row) = line {
                    let placed = lights.placed.entry((index, row)).or_default();
                    placed.push(span.bytes.clone());
                }
            }
        }
        lights
    }

    /// The row among `rows` of `parts` with match `n` of `lights`.
    fn lit_row(
        &self,
        parts: &[Part],
        rows: &Range<usize>,
        lights: &Lights<'_>,
        n: usize,
    ) -> Option<usize> {
        // An item's rows are whole lines, so its matches are all in them.
        let drawn = self.joined(parts).skip(rows.start).take(rows.len());
        let mut before = 0;
        for (at, row) in rows.clone().zip(lit(drawn, rows.start, lights)) {
            before += row.matches.iter().sum::<usize>();
            if before > n {
                return Some(at);
            }
        }
        None
    }

    /// Every row of `parts`, with how it continues the row before it.
    fn joined<'a>(
        &'a self,
        parts: &'a [Part],
    ) -> impl Iterator<Item = (&'a Line<'static>, Option<&'a RowWrap>)> + 'a {
        let wrapped = parts.iter().filter_map(|part| self.wrapped(part));
        let rows = wrapped.flat_map(|wrapped| wrapped.rows.iter().zip(&wrapped.continues));
        rows.map(|(row, continues)| (row, continues.as_ref()))
    }

    /// Rows `range` of `part`, which starts at row `start` of the
    /// transcript, with the matches of `lights` lit; each is sought in the
    /// whole line it is in, even where the line goes on past `range`.
    fn lit_rows(
        &self,
        part: &Part,
        start: usize,
        range: Range<usize>,
        lights: &Lights<'_>,
    ) -> Vec<Line<'static>> {
        let Some(wrapped) = self.wrapped(part) else {
            return Vec::new();
        };
        if !lights.any() {
            return wrapped.drawn(range);
        }
        let continues = |row: usize| wrapped.continues.get(row).is_some_and(Option::is_some);
        let mut first = range.start;
        while first > 0 && continues(first) {
            first -= 1;
        }
        let mut end = range.end;
        while continues(end) {
            end += 1;
        }
        let rows = wrapped.drawn(first..end);
        let joints = wrapped.continues.get(first..end).unwrap_or_default();
        let lit = light(&rows, joints, start + first, lights);
        let shown = lit.into_iter().skip(range.start - first);
        shown.take(range.len()).collect()
    }
}

/// `lines` wrapped to `width`, and `marks` of lines moved to the rows
/// they wrap to.
fn wrap_marked(
    lines: &[LookLine],
    marks: &[(TranscriptItem, Range<usize>)],
    width: usize,
) -> Wrapped {
    let mut rows = Vec::with_capacity(lines.len());
    let mut continues = Vec::with_capacity(lines.len());
    let mut code = Vec::new();
    // The first row of each line, and past the last.
    let mut starts = Vec::with_capacity(lines.len() + 1);
    for line in lines {
        let start = rows.len();
        starts.push(start);
        for (row, joint) in wrap(&line.line, width) {
            rows.push(row);
            continues.push(joint);
        }
        if line.code.is_some() {
            code.push((start..rows.len(), line.clone()));
        }
    }
    starts.push(rows.len());
    let row = |line: usize| starts.get(line).copied().unwrap_or(rows.len());
    let marks = marks
        .iter()
        .map(|(item, lines)| (item.clone(), row(lines.start)..row(lines.end)))
        .collect();
    Wrapped {
        rows,
        continues,
        marks,
        code,
        width,
    }
}

/// Where the transcript is scrolled: following its bottom, or held at the
/// row the user scrolled to, which then stays put as rows arrive below.
#[derive(Default)]
pub(crate) struct Scroll {
    /// The row held at the top of the screen; `None` follows the bottom.
    held: Option<Held>,
    /// Rows the keys moved the view since the last frame, up positive.
    moved: isize,
    /// An item to go to at the next frame, and which row of it.
    jump: Option<(TranscriptItem, ItemRow)>,
    /// The row to put at the top at the next frame, where a jump found it.
    at: Option<usize>,
}

/// The top row of a view scrolled up.
struct Held {
    /// Its index in the whole transcript.
    top: usize,
    /// The part it is in, its row there and that part's rows, to find it
    /// again after a resize re-wraps everything above it.
    part: usize,
    row: usize,
    rows: usize,
    /// The transcript's width and rows at the last frame.
    width: u16,
    total: usize,
    /// Whether rows arrived below since the view was held.
    grew: bool,
}

/// What is below a view scrolled up.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Below {
    /// Nothing: the view follows the bottom.
    #[default]
    Nothing,
    /// Rows that were there when the view was scrolled up.
    More,
    /// Rows that arrived since.
    New,
}

impl Below {
    /// The text rows of a view `height` rows high with this below it: a
    /// view scrolled up leaves its last row to the marker of what is below,
    /// unless that would leave it no text row.
    pub(crate) fn text_rows(self, height: usize) -> usize {
        match self {
            Self::More | Self::New if height > 1 => height - 1,
            _ => height,
        }
    }
}

impl Scroll {
    /// Moves the view `rows` up.
    pub(crate) fn up(&mut self, rows: usize) {
        self.moved = self
            .moved
            .saturating_add(isize::try_from(rows).unwrap_or(isize::MAX));
    }

    /// Moves the view `rows` down; reaching the bottom follows it again.
    pub(crate) fn down(&mut self, rows: usize) {
        self.moved = self
            .moved
            .saturating_sub(isize::try_from(rows).unwrap_or(isize::MAX));
    }

    /// Follows the bottom again.
    pub(crate) fn follow(&mut self) {
        *self = Self::default();
    }

    /// Shows `row` of `item` at the next frame.
    pub(crate) fn to(&mut self, item: TranscriptItem, row: ItemRow) {
        self.jump = Some((item, row));
    }

    /// Finds the row a jump to put a row of an item at the top goes to,
    /// by the item's rows: the item's first row and the row asked for, within
    /// the item. An item not there leaves the view where it is.
    fn find(&mut self, rows_of: impl Fn(&TranscriptItem) -> Option<Range<usize>>) {
        if let Some((item, ItemRow::Top(line))) = &self.jump {
            self.at = rows_of(item).map(|rows| {
                let last = rows.end.saturating_sub(1).max(rows.start);
                (rows.start + line).min(last)
            });
            self.jump = None;
        }
    }

    /// The top row of a view `height` rows high over parts of `counts`
    /// rows at `width`, after the keys' moves or a jump; holds it while it
    /// is above the bottom.
    fn place(&mut self, counts: &[usize], height: usize, width: u16) -> (usize, Below) {
        let total: usize = counts.iter().sum();
        let bottom = total.saturating_sub(height);
        let moved = std::mem::take(&mut self.moved);
        let (top, grew) = match (self.at.take(), &self.held) {
            (Some(at), _) => (at, false),
            (None, None) => (bottom, false),
            (None, Some(held)) if held.width == width => {
                (held.top, held.grew || total > held.total)
            }
            (None, Some(held)) => {
                let start: usize = counts.iter().take(held.part).sum();
                let rows = counts.get(held.part).copied().unwrap_or(0);
                let row = (held.row * rows / held.rows.max(1)).min(rows.saturating_sub(1));
                (start + row, held.grew)
            }
        };
        let top = top.saturating_add_signed(moved.saturating_neg());
        if top >= bottom {
            self.held = None;
            return (bottom, Below::Nothing);
        }
        let (mut part, mut start) = (0, 0);
        for (index, count) in counts.iter().enumerate() {
            part = index;
            if start + count > top {
                break;
            }
            start += count;
        }
        self.held = Some(Held {
            top,
            part,
            row: top - start,
            rows: counts.get(part).copied().unwrap_or(0),
            width,
            total,
            grew,
        });
        (top, if grew { Below::New } else { Below::More })
    }
}

/// The tool results a message carries.
fn tool_results(message: Option<&Message>) -> impl Iterator<Item = &ToolResult> {
    let content = match message {
        Some(Message::User { content }) => Some(content.iter()),
        _ => None,
    };
    content.into_iter().flatten().filter_map(|item| match item {
        UserContent::ToolResult(result) => Some(result),
        _ => None,
    })
}

/// The result of each call of `reply` among those of `results`, in call
/// order ([`answered_by`]).
fn answers<'a>(
    reply: &AssistantMessage,
    results: Option<&'a Message>,
) -> Vec<Option<&'a ToolResult>> {
    let calls = reply.tool_calls().map(|call| &call.id);
    answered_by(
        calls,
        tool_results(results).map(|result| (&result.call, result)),
    )
}

/// Draws `entry`, the `at`-th of those shown, with the notices and notes
/// inside it before their items, and says which lines are the message and
/// which each of its calls and notes; an archived entry has no items. A
/// tool call's result, which comes in the `next` message, is drawn with the
/// call, so the results of a reply's calls are not drawn after all of its
/// calls.
fn message_lines(
    (entry, at): (&ShownEntry<'_>, usize),
    (previous, next): (Option<&Message>, Option<&ShownEntry<'_>>),
    (look, width): (&Look<'_>, usize),
) -> (Vec<LookLine>, Marks) {
    let mut lines = Vec::new();
    let mut marks = Marks::new();
    look.renderers.at(Place::Message {
        index: at,
        block: 0,
    });
    let (message, id, author) = (&**entry.message, entry.id, entry.author);
    // The note a stop leaves for the model; the stop is drawn under the
    // turn it cut off.
    let note = entry.stop;
    let inside = look.inside(at);
    let inside_before = |item: usize, lines: &mut Vec<LookLine>, marks: &mut Marks| {
        for (_, inside) in inside.iter().filter(|(before, _)| *before == item) {
            match inside {
                Inside::Notice(at, notice) => notice_lines((*at, notice), lines, marks),
                Inside::Aside(shown) => {
                    let drawn = shown.lines(look.renderers, look.theme);
                    if drawn.is_empty() {
                        continue;
                    }
                    let start = lines.len();
                    lines.push(LookLine::default());
                    lines.extend(on_band(drawn, look.theme.tool));
                    marks.push((shown.item(), start..lines.len()));
                }
            }
        }
    };
    match message {
        Message::System { .. } => {}
        Message::User { content } => {
            let drawn = match previous {
                Some(Message::Assistant(reply)) => answers(reply, Some(message)),
                _ => Vec::new(),
            };
            let drawn: Vec<&ToolResult> = drawn.into_iter().flatten().collect();
            for (item_at, item) in content.iter().enumerate() {
                look.renderers.at(Place::Message {
                    index: at,
                    block: item_at,
                });
                inside_before(item_at, &mut lines, &mut marks);
                match item {
                    UserContent::Text(text) => {
                        if let Some((label, body)) = attached_file(&text.text) {
                            let count = body.lines().count();
                            let header = Line::from(format!("  [{label}, {count} lines]")).dim();
                            let body = text_rows(body, Style::new().dim());
                            lines.extend(on_band(
                                std::iter::once(header).chain(body),
                                look.theme.user,
                            ));
                            continue;
                        }
                        if note {
                            continue;
                        }
                        lines.push(LookLine::default());
                        // The user's own text has no author, or the user with no title.
                        let other = |author: &&Author| {
                            author.kind != AuthorKind::User || author.title.is_some()
                        };
                        // Text beside tool results is no one's typing, such
                        // as compaction's notes on outputs it cleared.
                        let results = content
                            .iter()
                            .any(|part| matches!(part, UserContent::ToolResult(_)));
                        match author.filter(other) {
                            Some(author) => lines.extend(on_band(
                                delivered_lines(id, author, &text.text, look),
                                look.theme.tool,
                            )),
                            None if results => lines.extend(on_band(
                                text_rows(&text.text, Style::new().dim()),
                                look.theme.tool,
                            )),
                            None => lines.extend(on_band(
                                MessageView {
                                    role: Role::User,
                                    message: id,
                                    block: item_at,
                                    text: &text.text,
                                    marks: &[],
                                    width,
                                    theme: *look.theme,
                                }
                                .default_lines(),
                                look.theme.user,
                            )),
                        }
                    }
                    // Drawn under its call already.
                    UserContent::ToolResult(result)
                        if drawn.iter().any(|drawn| std::ptr::eq(*drawn, result)) => {}
                    UserContent::ToolResult(result) => {
                        let style = result_style(result.is_error);
                        let rows = text_rows(&result_text(result), style);
                        lines.extend(on_band(rows, look.theme.tool));
                    }
                    other => {
                        if let Some(block) = ContentBlock::of_user(other) {
                            let view = (id, item_at, block, false);
                            content_lines(view, look, (&mut lines, &mut marks));
                        }
                    }
                }
            }
        }
        Message::Assistant(assistant) => {
            // Each call's result and facts, by its place in the reply.
            let results = answers(assistant, next.map(|next| &**next.message));
            let facts = next.zip(id).map(|(next, id)| {
                let calls = CallRef::calls(id, assistant).map(|(at, call)| (at, &call.id));
                next.facts.of(calls)
            });
            let facts = facts.into_iter().flatten().chain(std::iter::repeat(None));
            let mut calls = (0..).zip(results).zip(facts);
            for (item_at, item) in assistant.content.iter().enumerate() {
                look.renderers.at(Place::Message {
                    index: at,
                    block: item_at,
                });
                inside_before(item_at, &mut lines, &mut marks);
                match item {
                    AssistantContent::Text(text) => {
                        lines.push(LookLine::default());
                        let marks =
                            id.map(|id| look.marks_in(MarkIn::Message { id, block: item_at }));
                        lines.extend(
                            MessageView {
                                role: Role::Assistant,
                                message: id,
                                block: item_at,
                                text: &text.text,
                                marks: &marks.unwrap_or_default(),
                                width,
                                theme: *look.theme,
                            }
                            .default_lines(),
                        );
                    }
                    AssistantContent::Reasoning(reasoning) => {
                        lines.extend(
                            MessageView {
                                role: Role::Reasoning,
                                message: id,
                                block: item_at,
                                text: &reasoning.text,
                                marks: &[],
                                width,
                                theme: *look.theme,
                            }
                            .default_lines(),
                        );
                    }
                    AssistantContent::ToolCall(call) => {
                        let ((index, result), facts) = calls.next().unwrap_or_default();
                        let at = id.map(|reply| CallRef { reply, index });
                        let from = at.and_then(|at| look.forked_call(at));
                        let view = ToolCallView {
                            call,
                            result,
                            state: look.state(at, result),
                            theme: *look.theme,
                            facts,
                            took: at.and_then(|at| look.facts.took(from.or(look.agent), at)),
                            output: at.and_then(|at| look.outputs.get(&at)).map(String::as_str),
                            children: look.children.get(at, from).map_or(&[], Vec::as_slice),
                        };
                        let (drawn, decorated) = look.renderers.call_decorated(&view);
                        let start = lines.len();
                        lines.extend(on_band(drawn, look.theme.tool));
                        // Only an archived entry has no id, and no items.
                        if let Some(at) = at.map(CallItem::Reply) {
                            marks.push((TranscriptItem::Call(at), start..lines.len()));
                            for (name, rows) in decorated {
                                let decoration = TranscriptItem::Decoration(at, name);
                                marks.push((decoration, start + rows.start..start + rows.end));
                            }
                        }
                    }
                    // A provider item with no meaning here shows nothing.
                    other => {
                        if let Some(block) = ContentBlock::of_assistant(other) {
                            let view = (id, item_at, block, true);
                            content_lines(view, look, (&mut lines, &mut marks));
                        }
                    }
                }
            }
        }
    }
    if look.stops.under(id) {
        lines.extend(stop_lines(message, look.theme));
    }
    if !matches!(message, Message::System { .. }) {
        let decorated = DecoratedMessage {
            message,
            id,
            author,
        };
        let (drawn, decorations) = look.renderers.message_decorations(&decorated);
        let start = lines.len();
        lines.extend(drawn);
        for (name, rows) in decorations {
            if let Some(id) = id {
                let decoration = TranscriptItem::MessageDecoration(id, name);
                marks.push((decoration, start + rows.start..start + rows.end));
            }
        }
    }
    if let Some(id) = id {
        marks.insert(0, (TranscriptItem::Message(id), 0..lines.len()));
    }
    if entry.archived {
        marks.clear();
    }
    (lines, marks)
}

/// The stop drawn under `message`, the last of a turn a stop cut off:
/// within a reply when it is one, else under the request or the results
/// nothing answered.
fn stop_lines(message: &Message, theme: &Theme) -> Vec<LookLine> {
    match message {
        Message::Assistant(_) => vec![Line::from("[stopped]").dim().into()],
        Message::User { content } => {
            let results = content
                .iter()
                .any(|item| matches!(item, UserContent::ToolResult(_)));
            let band = if results { theme.tool } else { theme.user };
            let stop = Line::from("  [stopped before it was answered]").dim();
            on_band([stop], band).collect()
        }
        Message::System { .. } => Vec::new(),
    }
}

/// Draws the content block `block`, the `index`-th of the message `id`
/// (by `assistant` or not), by its kind's look, and marks its rows when
/// the message has an id.
fn content_lines(
    (id, index, block, assistant): (Option<MessageId>, usize, ContentBlock<'_>, bool),
    look: &Look<'_>,
    (lines, marks): (&mut Vec<LookLine>, &mut Marks),
) {
    let view = ContentView {
        message: id,
        index,
        block,
        assistant,
    };
    let start = lines.len();
    lines.extend(look.renderers.content(&view));
    if let Some(id) = id {
        marks.push((TranscriptItem::Content(id, index), start..lines.len()));
    }
}

/// Text an agent or a plugin sent, without the header the model reads:
/// drawn by the render system of the plugin that sent it, if it has one,
/// else as where it came from and its start.
fn delivered_lines(
    id: Option<MessageId>,
    author: &Author,
    text: &str,
    look: &Look<'_>,
) -> Vec<LookLine> {
    let body = author
        .header()
        .and_then(|header| text.strip_prefix(&header))
        .map_or(text, |body| body.trim_start_matches('\n'));
    let label = author.label();
    let view = EntryView {
        id,
        plugin: plugin_name(author),
        title: &label,
        text: body,
    };
    look.renderers.entry(&view)
}

/// The lines of the notice `at` in the notices shown, coloured by its
/// level, marked as that item.
pub(crate) fn notice_lines(
    (at, notice): (usize, &ShownNotice),
    lines: &mut Vec<LookLine>,
    marks: &mut Marks,
) {
    let style = match notice.level {
        NoticeLevel::Info => Style::new().fg(Color::Magenta),
        NoticeLevel::Warn => Style::new().fg(Color::Yellow),
        NoticeLevel::Error => Style::new().fg(Color::Red),
    };
    let start = lines.len();
    lines.extend(
        plain_lines(&notice.text, style)
            .into_iter()
            .map(LookLine::from),
    );
    marks.push((TranscriptItem::Notice(at), start..lines.len()));
}

/// `lines` on the band `band`, each where it has no background of its
/// own, such as a diff's line: the background of every row it wraps to.
pub(crate) fn on_band(
    lines: impl IntoIterator<Item = impl Into<LookLine>>,
    band: Color,
) -> impl Iterator<Item = LookLine> {
    lines.into_iter().map(move |line| {
        let mut line = line.into();
        line.line.style.bg = line.line.style.bg.or(Some(band));
        line
    })
}

/// Whether `message`'s lines depend on the width: a reply or its reasoning
/// with a table.
fn fitted(message: &Message) -> bool {
    let Message::Assistant(assistant) = message else {
        return false;
    };
    let texts = assistant.content.iter().filter_map(|item| match item {
        AssistantContent::Text(text) => Some(text.text.as_str()),
        AssistantContent::Reasoning(reasoning) => Some(reasoning.text.as_str()),
        _ => None,
    });
    texts.into_iter().any(markdown::has_table)
}

/// `text`'s lines in one style.
pub(crate) fn plain_lines(text: &str, style: Style) -> Vec<Line<'static>> {
    text.lines()
        .map(|line| Line::styled(line.replace('\t', "    "), style))
        .collect()
}

#[cfg(test)]
mod tests;
