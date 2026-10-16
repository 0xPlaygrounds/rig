//! SP-17: two branches of the session side by side. `/compare` shows the
//! active line and the most recently active kept branch from the entry
//! where they part, each in a transcript pane drawn as the main one is;
//! `/compare <a> <b>` picks two by label, aside id (`a3`) or leaf id
//! (`m12`). The active line's pane is live, so a turn streaming there shows
//! as it streams.
//!
//! Each side has a header (label, models, turns, tokens, files changed),
//! and a look of its own pane draws a header row over each turn, padded so
//! the k-th turn after the divergence starts at the same row on both
//! sides. Up and Down scroll both a turn, `s` lets the focused side scroll
//! alone, Tab moves the focus, `p` continues on the focused side's branch
//! and Esc closes. Calls only one side made, by tool and target, are lit.
//! Under 100 columns the sides are tabs.
//!
//! What a tool call touched and changed is kept as an aside after the
//! reply that made it ([`CALL_KIND`]), so it goes with the branch it is
//! in; tokens come from the effect records of each turn.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use rig_harness::prelude::*;
use rig_harness::rig_core::conversation::{Conversation as Line, shared_prefix};
use rig_harness::rig_core::effect::{Answered, EffectId, EffectKind, EffectRecord};
use rig_harness::rig_core::message::{CallId, Message as ChatMessage};
use rig_harness::rig_core::tool::FilesChanged;
use rig_telemetry::EffectLog;
use rig_tui::crossterm::event::KeyCode;
use rig_tui::ratatui::layout::{Constraint, Rect};
use rig_tui::ratatui::style::{Style, Stylize};
use rig_tui::ratatui::text::{Line as Row, Span};
use rig_tui::ratatui::widgets::{Paragraph, Widget};
use rig_tui::{
    DrawnTranscript, Highlight, ItemRow, KeyBinding, KeyFocus, KeyPress, KeyPressed, KeyScope,
    LineSpan, MessageDecorator, PanelCanvas, Placement, RequestRedraw, ScrollTo, TranscriptItem,
    TranscriptPane, TranscriptSource, TranscriptView, TuiPanel, TuiSystems,
};
use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests;

/// The plugin's name, as it authors asides and its looks.
const NAME: &str = "branch-compare";

/// The kind of the aside a [`CallFacts`] is kept in.
pub const CALL_KIND: &str = "branch-compare.call";

/// Rows over the sides for their headers, and as many under them.
const HEADER_ROWS: u16 = 3;

/// Narrower than this, the sides are tabs.
const NARROW: u16 = 100;

/// `/compare` and what it needs.
#[derive(Default)]
pub struct BranchComparePlugin;

impl Plugin for BranchComparePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Spent>()
            .add_command(
                "compare",
                "Compare two branches side by side: /compare [<a>] [<b>], by label or leaf id",
                compare,
            )
            .add_observer(note_call)
            .add_systems(Update, note_spent.after(AgentLoop))
            .add_systems(
                PostUpdate,
                (
                    read_sides.in_set(TuiSystems::Prepare),
                    fit.in_set(TuiSystems::Fit),
                    draw.in_set(TuiSystems::Draw),
                ),
            );
    }
}

/// What one tool call touched and changed, kept as an aside of kind
/// [`CALL_KIND`] after the reply that made it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallFacts {
    /// The call.
    pub call: CallId,
    /// Its tool.
    pub tool: String,
    /// The paths its footprint named, absolute; none for one that may
    /// touch anything.
    pub targets: Vec<String>,
    /// The files it changed, as it published them.
    pub changed: Vec<String>,
}

/// Keeps what each settled tool call touched and changed in its agent's
/// conversation, after the reply that made it.
fn note_call(
    settled: On<Add<Settled>>,
    calls: Query<(
        &ToolRequest,
        &ForAgent,
        Option<&Footprint>,
        Option<&Published>,
    )>,
    mut agents: Query<&mut Conversation>,
) {
    let Ok((request, &ForAgent(agent), footprint, published)) = calls.get(settled.entity) else {
        return;
    };
    let Ok(mut conversation) = agents.get_mut(agent) else {
        return;
    };
    let call = &request.call;
    let made = conversation
        .entries()
        .rev()
        .find(|entry| calls_of(entry.message).any(|made| made.id == call.id))
        .map(|entry| entry.id);
    let Some(after) = made else {
        return;
    };
    let targets = match footprint.map(|footprint| &footprint.0) {
        Some(Touches::Reads(Scope::Paths(paths)) | Touches::Writes(Scope::Paths(paths))) => paths
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
        _ => Vec::new(),
    };
    let changed = published
        .and_then(|published| published.0.get::<FilesChanged>().ok().flatten())
        .map(|FilesChanged(changes)| changes.into_iter().map(|change| change.path).collect())
        .unwrap_or_default();
    let facts = CallFacts {
        call: call.id.clone(),
        tool: call.function.name.as_str().to_owned(),
        targets,
        changed,
    };
    if let Ok(value) = serde_json::to_value(&facts) {
        let aside = Aside::new(CALL_KIND, Author::plugin(NAME), value).after(Some(after));
        conversation.add_aside(aside);
    }
}

/// The tool calls of `message`, a reply.
fn calls_of(message: &ChatMessage) -> impl Iterator<Item = &ToolCall> {
    let content = match message {
        ChatMessage::Assistant(reply) => reply.content.as_slice(),
        _ => &[],
    };
    content.iter().filter_map(|item| match item {
        AssistantContent::ToolCall(call) => Some(call),
        _ => None,
    })
}

/// The tokens each model call spent, by its effect: its agent's id and
/// its turn. From the effect records of this run, and of the earlier ones
/// once a comparison read the effect log.
#[derive(Resource, Default)]
struct Spent {
    by_effect: HashMap<EffectId, (Option<String>, Option<u64>, u64)>,
    read_log: bool,
}

impl Spent {
    /// Notes `record`, if it is an answered model call.
    fn note(&mut self, record: &EffectRecord) {
        let outcome = record.reviewed.as_ref().unwrap_or(&record.outcome);
        if let (EffectKind::Completion { .. }, Ok(Answered::Completion(reply))) =
            (&record.kind, outcome)
        {
            let usage = &reply.usage;
            let parts = usage.input_tokens.unwrap_or(0) + usage.output_tokens.unwrap_or(0);
            let tokens = usage.total_tokens.unwrap_or(parts);
            let agent = record.scope.as_deref().map(str::to_owned);
            self.by_effect
                .insert(record.id, (agent, record.turn, tokens));
        }
    }

    /// What the agent `agent` spent in `turns`; `None` when nothing is known.
    fn of(&self, agent: &str, turns: &[u64]) -> Option<u64> {
        let spent = self.by_effect.values().filter(|(by, turn, _)| {
            by.as_deref() == Some(agent) && turn.is_some_and(|turn| turns.contains(&turn))
        });
        spent.map(|(.., tokens)| *tokens).reduce(|a, b| a + b)
    }
}

/// Notes each model call's tokens as it settles.
fn note_spent(mut recorded: MessageReader<EffectRecorded>, mut spent: ResMut<Spent>) {
    for done in recorded.read() {
        spent.note(&done.0);
    }
}

/// The comparison, on its box over the screen, which takes the keys.
#[derive(Component)]
struct CompareView {
    /// The agent whose branches it compares.
    agent: Entity,
    /// Left and right.
    sides: [Side; 2],
    /// The side the keys act on.
    focus: usize,
    /// Whether the sides scroll together.
    together: bool,
    /// The box the sides are columns of.
    inner: Entity,
    /// The row each turn's header is drawn at on both sides, from the
    /// first turn after the divergence.
    rows: Vec<usize>,
    /// Whether the panes are to be scrolled to their turns.
    scroll: bool,
}

/// One side of a comparison.
struct Side {
    /// Its branch's label.
    label: String,
    /// The kept branch it shows; `None` for the active line, shown live.
    line: Option<AsideId>,
    /// Its line as it was when the view opened; `None` for the active line.
    value: Option<Line>,
    /// Its transcript pane.
    pane: Entity,
    /// Its pane's own look: the turn headers and the padding.
    look: Entity,
    /// What it holds from the divergence on.
    facts: SideFacts,
    /// The padding its look draws now.
    pads: Pads,
    /// The turn at its top, from 0.
    turn: usize,
}

/// Blank rows a side's look draws before each turn's header and after its
/// last message.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Pads {
    before: Vec<usize>,
    tail: usize,
}

/// What a side holds from the divergence on.
#[derive(Clone, Debug, Default)]
struct SideFacts {
    turns: Vec<TurnFacts>,
    models: Vec<String>,
    tokens: Option<u64>,
    files: Vec<String>,
    /// Its last message.
    last: Option<MessageId>,
}

/// One turn of a side.
#[derive(Clone, Debug)]
struct TurnFacts {
    /// The message before its first, under which its header is drawn.
    anchor: Option<MessageId>,
    /// Its first message.
    first: Option<MessageId>,
    /// The first line of its last answer.
    title: String,
    /// Its tool calls, each with its tool and target.
    calls: Vec<(CallId, CallKey)>,
}

/// A call as both sides are compared by: its tool and what it targets.
type CallKey = (String, Vec<String>);

/// Marks a side's pane.
#[derive(Component)]
struct ComparePane;

/// Marks the box the sides are columns of.
#[derive(Component)]
struct CompareInner;

/// Marks a side's look.
#[derive(Component)]
struct CompareLook;

/// A line `/compare` can name.
struct Candidate {
    line: Option<AsideId>,
    title: String,
    leaf: Option<MessageId>,
    /// When its newest own entry was made.
    recent: u64,
}

/// `/compare [<a>] [<b>]`.
fn compare(In(args): In<CommandArgs>, world: &mut World) {
    if let Err(notice) = open(world, &args) {
        world.write_message(notice);
    }
}

/// Opens the comparison `args` asks for, or says why not.
fn open(world: &mut World, args: &CommandArgs) -> Result<(), Notice> {
    let agent = args.agent;
    let error = |text: &str| Notice::error(agent, text.to_owned());
    let conversation = world
        .get::<Conversation>(agent)
        .map(|conversation| conversation.view())
        .ok_or_else(|| error("The agent has no conversation."))?;
    let candidates = candidates(&conversation);
    let find = |name: &str| {
        let aside = name.parse::<AsideId>().ok();
        let leaf = name.parse::<MessageId>().ok();
        let found = candidates.iter().position(|candidate| {
            candidate.title == name
                || (aside.is_some() && candidate.line == aside)
                || (leaf.is_some() && candidate.leaf == leaf)
        });
        found.ok_or_else(|| Notice::error(agent, format!("There is no branch `{name}`.")))
    };
    let recent = (1..candidates.len()).max_by_key(|&at| {
        let recent = candidates.get(at).map_or(0, |candidate| candidate.recent);
        (recent, at)
    });
    let words: Vec<&str> = args.args.split_whitespace().collect();
    let (left, right) = match words.as_slice() {
        [] => (0, recent),
        [one] => (0, Some(find(one)?)),
        [one, two, ..] => (find(one)?, Some(find(two)?)),
    };
    let none = "There is no other branch to compare with: /split makes one.";
    let right = right.ok_or_else(|| Notice::info(agent, none.to_owned()))?;
    if left == right {
        return Err(error("Name two different branches."));
    }
    let mut closing = world.query_filtered::<Entity, With<CompareView>>();
    let open: Vec<Entity> = closing.iter(world).collect();
    for view in open {
        world.entity_mut(view).despawn();
    }
    let chosen = [left, right].map(|at| candidates.get(at));
    let [Some(left), Some(right)] = chosen else {
        return Err(error("There is no such branch."));
    };
    spawn(world, agent, &conversation, [left, right]);
    if let Some(log) = world.get_resource::<EffectLog>().cloned()
        && !world.resource::<Spent>().read_log
    {
        let records = log.read_outcomes().unwrap_or_default();
        let mut spent = world.resource_mut::<Spent>();
        records.iter().for_each(|record| spent.note(record));
        spent.read_log = true;
    }
    Ok(())
}

/// The active line, then every kept branch.
fn candidates(conversation: &Line) -> Vec<Candidate> {
    let active = Candidate {
        line: None,
        title: conversation.line_title().to_owned(),
        leaf: conversation
            .messages()
            .len()
            .checked_sub(1)
            .and_then(|at| conversation.id(at)),
        recent: u64::MAX,
    };
    let kept = conversation.branches().into_iter().map(|kept| Candidate {
        line: Some(kept.id),
        title: kept.branch.title.clone(),
        leaf: kept.branch.entries.last().and_then(|entry| entry.id),
        recent: (kept.branch.entries.iter().filter_map(|entry| entry.made))
            .max()
            .unwrap_or(0),
    });
    std::iter::once(active).chain(kept).collect()
}

/// Spawns the view of `chosen`, lines of `agent`'s `conversation`, and
/// gives it the keys.
fn spawn(world: &mut World, agent: Entity, conversation: &Line, [left, right]: [&Candidate; 2]) {
    let whole = Placement::Over {
        width: Constraint::Percentage(100),
        height: Constraint::Percentage(100),
    };
    let host = world
        .spawn((Name::new("branch compare"), TuiPanel::new(whole)))
        .id();
    let inner = world
        .spawn((
            Name::new("branch compare: sides"),
            CompareInner,
            TuiPanel::new(whole),
            ChildOf(host),
        ))
        .id();
    let sides = [0u16, 1].map(|part| {
        let candidate = if part == 0 { left } else { right };
        let value = candidate.line.and_then(|line| {
            let length = conversation.line(Some(line)).map_or(0, |line| line.len());
            conversation.line_conversation(Some(line), length)
        });
        let source = match &value {
            Some(value) => TranscriptSource::Value(value.clone()),
            None => TranscriptSource::Agent(agent),
        };
        let within = Placement::Within {
            panel: inner,
            part,
            of: 2,
        };
        let pane = world
            .spawn((
                ComparePane,
                TranscriptPane::new(source).titled(candidate.title.clone()),
                TuiPanel::new(within),
                DrawnTranscript::default(),
                ChildOf(host),
            ))
            .id();
        let look = world
            .spawn((
                CompareLook,
                MessageDecorator::new(NAME, |_| Vec::new()),
                ChildOf(pane),
            ))
            .id();
        Side {
            label: candidate.title.clone(),
            line: candidate.line,
            value,
            pane,
            look,
            facts: SideFacts::default(),
            pads: Pads::default(),
            turn: 0,
        }
    });
    world.entity_mut(host).insert(CompareView {
        agent,
        sides,
        focus: 0,
        together: true,
        inner,
        rows: Vec::new(),
        scroll: true,
    });
    bind(world, host);
    world.insert_resource(KeyFocus(Some(host)));
}

/// Binds the view's keys in its box's scope.
fn bind(world: &mut World, host: Entity) {
    let key = |code, label: &str| {
        let binding = KeyBinding::new(KeyPress::key(code), label);
        (binding.in_scope(KeyScope::Panel(host)), ChildOf(host))
    };
    world
        .spawn(key(KeyCode::Esc, "Close the comparison"))
        .observe(move |_: On<KeyPressed>, mut commands: Commands| {
            commands.entity(host).try_despawn();
        });
    world.spawn(key(KeyCode::Tab, "The other side")).observe(
        move |_: On<KeyPressed>,
              views: Query<&mut CompareView>,
              redraw: MessageWriter<RequestRedraw>| {
            change(host, views, redraw, |view| {
                view.focus = 1 - view.focus.min(1)
            });
        },
    );
    world.spawn(key(KeyCode::Down, "The next turn")).observe(
        move |_: On<KeyPressed>,
              views: Query<&mut CompareView>,
              redraw: MessageWriter<RequestRedraw>| {
            change(host, views, redraw, |view| move_turn(view, true));
        },
    );
    world.spawn(key(KeyCode::Up, "The turn before")).observe(
        move |_: On<KeyPressed>,
              views: Query<&mut CompareView>,
              redraw: MessageWriter<RequestRedraw>| {
            change(host, views, redraw, |view| move_turn(view, false));
        },
    );
    world
        .spawn(key(
            KeyCode::Char('s'),
            "Scroll the focused side alone, or both",
        ))
        .observe(
            move |_: On<KeyPressed>,
                  views: Query<&mut CompareView>,
                  redraw: MessageWriter<RequestRedraw>| {
                change(host, views, redraw, |view| {
                    view.together = !view.together;
                    let turn = view.sides.get(view.focus).map_or(0, |side| side.turn);
                    view.sides.iter_mut().for_each(|side| side.turn = turn);
                    view.scroll = true;
                });
            },
        );
    world
        .spawn(key(
            KeyCode::Char('p'),
            "Continue on the focused side's branch",
        ))
        .observe(move |_: On<KeyPressed>, mut commands: Commands| {
            commands.queue(move |world: &mut World| pick(world, host));
        });
}

/// Changes the view on `host` with `edit` and draws it again.
fn change(
    host: Entity,
    mut views: Query<&mut CompareView>,
    mut redraw: MessageWriter<RequestRedraw>,
    edit: impl FnOnce(&mut CompareView),
) {
    if let Ok(mut view) = views.get_mut(host) {
        edit(&mut view);
        redraw.write(RequestRedraw);
    }
}

/// Moves the sides that scroll a turn down or up.
fn move_turn(view: &mut CompareView, down: bool) {
    let step = |turn: usize, last: usize| {
        if down {
            (turn + 1).min(last)
        } else {
            turn.saturating_sub(1)
        }
    };
    let last = |side: &Side| side.facts.turns.len().saturating_sub(1);
    if view.together {
        let most = view.sides.iter().map(last).max().unwrap_or(0);
        let turn = view.sides.get(view.focus).map_or(0, |side| side.turn);
        let turn = step(turn, most);
        view.sides.iter_mut().for_each(|side| side.turn = turn);
    } else if let Some(side) = view.sides.get_mut(view.focus) {
        side.turn = step(side.turn, last(side));
    }
    view.scroll = true;
}

/// `p`: makes the focused side's branch the active line and closes the
/// view; a running turn keeps it open.
fn pick(world: &mut World, host: Entity) {
    let Some(view) = world.get::<CompareView>(host) else {
        return;
    };
    let agent = view.agent;
    let Some(side) = view.sides.get(view.focus) else {
        return;
    };
    let (line, label) = (side.line, side.label.clone());
    let notice = match line {
        None => Notice::info(agent, format!("Staying on “{label}”.")),
        Some(_) if world.get::<ActiveTurn>(agent).is_some() => {
            let busy = "A turn is running. Press Esc to stop it first.";
            world.write_message(Notice::error(agent, busy.to_owned()));
            return;
        }
        Some(line) => match world.get_mut::<Conversation>(agent) {
            Some(mut conversation) => {
                match conversation.switch_branch(line, Author::plugin(NAME)) {
                    Ok(_) => Notice::info(agent, format!("Now on “{label}”.")),
                    Err(failure) => Notice::error(agent, failure.to_string()),
                }
            }
            None => Notice::error(agent, "The agent has no conversation.".to_owned()),
        },
    };
    world.write_message(notice);
    world.entity_mut(host).despawn();
}

/// Reads what each side holds from the divergence on, when the view
/// opened, the active line changed or a model call settled.
fn read_sides(
    mut views: Query<&mut CompareView>,
    agents: Query<(Ref<Conversation>, &AgentId)>,
    spent: Res<Spent>,
    root: Option<Res<ProjectRoot>>,
) {
    let root = root.map(|root| root.0.clone());
    for mut view in &mut views {
        let Ok((conversation, id)) = agents.get(view.agent) else {
            continue;
        };
        if !(view.is_added() || conversation.is_changed() || spent.is_changed()) {
            continue;
        }
        let live: &Line = &conversation;
        let [left, right] = &view.sides;
        let lines = [left, right].map(|side| side.value.as_ref().unwrap_or(live));
        let [ours, theirs] = lines.map(|line| line.line(None).unwrap_or_default());
        let parted = shared_prefix(&ours, &theirs);
        let facts = lines.map(|line| side_facts(line, parted, &id.0, &spent, root.as_deref()));
        for (side, facts) in view.sides.iter_mut().zip(facts) {
            side.facts = facts;
        }
    }
}

/// What `line` holds from entry `parted` on: its turns, the models that
/// answered, what the turns of `agent` spent and the files its calls
/// changed, relative to `root`.
fn side_facts(
    line: &Line,
    parted: usize,
    agent: &str,
    spent: &Spent,
    root: Option<&Path>,
) -> SideFacts {
    let known: HashMap<CallId, CallFacts> = line
        .asides()
        .filter(|(_, aside)| aside.kind == CALL_KIND)
        .filter_map(|(_, aside)| serde_json::from_value::<CallFacts>(aside.value.clone()).ok())
        .map(|facts| (facts.call.clone(), facts))
        .collect();
    let entries: Vec<EntryRef<'_>> = line.entries().collect();
    let own = entries.get(parted..).unwrap_or_default();
    let mut starts: Vec<(usize, Option<u64>)> = line
        .turns()
        .filter_map(|(_, record)| {
            let at = line.index_of(record.start)?;
            (at >= parted).then_some((at, Some(record.index)))
        })
        .collect();
    if starts.is_empty() {
        let user = |at: usize| {
            entries
                .get(at)
                .is_some_and(|entry| matches!(entry.message, ChatMessage::User { .. }))
        };
        let asks = |at: &usize| {
            entries.get(*at).is_some_and(EntryRef::is_request)
                && (*at == parted || !user(at.saturating_sub(1)))
        };
        starts = (parted..entries.len())
            .filter(asks)
            .map(|at| (at, None))
            .collect();
    }
    starts.sort_unstable();
    starts.dedup_by_key(|(at, _)| *at);
    let key = |call: &ToolCall| {
        let targets = known.get(&call.id).map(|facts| facts.targets.clone());
        (
            call.function.name.as_str().to_owned(),
            targets.unwrap_or_default(),
        )
    };
    let turns = starts
        .iter()
        .enumerate()
        .map(|(k, &(start, _))| {
            let end = starts.get(k + 1).map_or(entries.len(), |(end, _)| *end);
            let span = entries.get(start..end).unwrap_or_default();
            let title = span.iter().rev().find_map(|entry| answer(entry.message));
            let calls = span.iter().flat_map(|entry| calls_of(entry.message));
            TurnFacts {
                anchor: start.checked_sub(1).and_then(|at| line.id(at)),
                first: line.id(start),
                title: title.unwrap_or_else(|| "…".to_owned()),
                calls: calls.map(|call| (call.id.clone(), key(call))).collect(),
            }
        })
        .collect();
    let mut models = Vec::new();
    for entry in own {
        if let ChatMessage::Assistant(reply) = entry.message
            && let Some(origin) = &reply.origin
            && !models.contains(&origin.model)
        {
            models.push(origin.model.clone());
        }
    }
    let indices: Vec<u64> = starts.iter().filter_map(|(_, index)| *index).collect();
    let relative =
        |path: &String| match root.and_then(|root| Path::new(path).strip_prefix(root).ok()) {
            Some(relative) => relative.display().to_string(),
            None => path.clone(),
        };
    let files: BTreeSet<String> = own
        .iter()
        .flat_map(|entry| calls_of(entry.message))
        .filter_map(|call| known.get(&call.id))
        .flat_map(|facts| facts.changed.iter().map(relative))
        .collect();
    SideFacts {
        turns,
        models,
        tokens: spent.of(agent, &indices),
        files: files.into_iter().collect(),
        last: entries.last().map(|entry| entry.id),
    }
}

/// The first line of `message`'s text, if it is a reply with any.
fn answer(message: &ChatMessage) -> Option<String> {
    let ChatMessage::Assistant(reply) = message else {
        return None;
    };
    let text = reply.content.iter().find_map(|item| match item {
        AssistantContent::Text(text) => text.text.lines().find(|line| !line.trim().is_empty()),
        _ => None,
    })?;
    Some(text.trim().chars().take(60).collect())
}

/// Fits the view to the screen and to what its panes laid out: the sides
/// as columns or tabs, each side padded so its turns start at the rows of
/// the other's, the calls only one side made lit, and the panes scrolled
/// to their turns. A change lays the frame out again before it is drawn.
#[allow(clippy::type_complexity)]
fn fit(
    mut views: Query<(&mut CompareView, &PanelCanvas)>,
    mut panes: Query<
        (&DrawnTranscript, &mut TranscriptView, &mut TuiPanel),
        (With<ComparePane>, Without<CompareView>),
    >,
    mut inners: Query<&mut TuiPanel, (With<CompareInner>, Without<ComparePane>)>,
    mut looks: Query<&mut MessageDecorator, With<CompareLook>>,
    mut commands: Commands,
) {
    for (mut view, canvas) in &mut views {
        let screen = canvas.area();
        let Ok(mut inner) = inners.get_mut(view.inner) else {
            continue;
        };
        if screen.is_empty() {
            continue;
        }
        let rows = screen.height.saturating_sub(2 * HEADER_ROWS);
        let placed = TuiPanel::new(Placement::Over {
            width: Constraint::Percentage(100),
            height: Constraint::Length(rows),
        });
        let mut again = inner.set_if_neq(placed);
        let narrow = screen.width < NARROW;
        let (host, focus) = (view.inner, view.focus);
        let [left, right] = &view.sides;
        let Ok([left, right]) = panes.get_many_mut([left.pane, right.pane]) else {
            continue;
        };
        let mut drawn = [left.0.clone(), right.0.clone()];
        for (at, mut panel) in [left.2, right.2].into_iter().enumerate() {
            let placement = match (narrow, at == focus) {
                (false, _) => Placement::Within {
                    panel: host,
                    part: u16::try_from(at).unwrap_or(0),
                    of: 2,
                },
                (true, true) => Placement::Within {
                    panel: host,
                    part: 0,
                    of: 1,
                },
                (true, false) => Placement::Over {
                    width: Constraint::Length(0),
                    height: Constraint::Length(0),
                },
            };
            again |= panel.set_if_neq(TuiPanel::new(placement));
        }
        let naturals = [0, 1].map(|at| match (view.sides.get(at), drawn.get(at)) {
            (Some(side), Some(drawn)) => naturals(drawn, &side.facts, &side.pads),
            _ => (Vec::new(), 0),
        });
        let (pads, header_rows) = align(&naturals);
        let mut padded = false;
        for (side, pads) in view.sides.iter_mut().zip(pads) {
            if side.pads != pads
                && let Ok(mut look) = looks.get_mut(side.look)
            {
                *look = turn_look(&side.facts, &pads);
                side.pads = pads;
                padded = true;
            }
        }
        view.rows = header_rows;
        if padded {
            view.scroll = true;
            again = true;
        } else if view.scroll {
            view.scroll = false;
            for (side, drawn) in view.sides.iter().zip(&drawn) {
                let Some(row) = view.rows.get(side.turn).copied() else {
                    continue;
                };
                let Some(item) = drawn.item_at(row).cloned() else {
                    continue;
                };
                let start = drawn.rows_of(&item).map_or(row, |rows| rows.start);
                commands.trigger(ScrollTo {
                    pane: side.pane,
                    item: Some(item),
                    row: ItemRow::Top(row.saturating_sub(start)),
                });
            }
            again = true;
        }
        let lit = only_here(&view.sides);
        let [mut left_view, mut right_view] = [left.1, right.1];
        for ((pane_view, only), drawn) in [&mut left_view, &mut right_view]
            .into_iter()
            .zip(lit)
            .zip(&mut drawn)
        {
            let highlights = highlights(drawn, only);
            if pane_view.highlights != highlights {
                pane_view.highlights = highlights;
                again = true;
            }
        }
        if again {
            inner.set_changed();
        }
    }
}

/// The highlight of the calls `only` in `drawn`, by their header lines;
/// none for none.
fn highlights(drawn: &DrawnTranscript, only: Vec<CallId>) -> Vec<Highlight> {
    let spans: Vec<LineSpan> = only
        .into_iter()
        .filter_map(|call| {
            let item = TranscriptItem::Call(call);
            let header = drawn.item(&item)?.lines().next()?;
            Some(LineSpan {
                item: Some(item),
                line: 0,
                bytes: 0..header.text.len(),
            })
        })
        .collect();
    if spans.is_empty() {
        return Vec::new();
    }
    vec![Highlight {
        style: Style::new().black().on_yellow(),
        spans,
        ..Highlight::default()
    }]
}

/// Where each turn's header of a side would start without the padding,
/// and how many rows it would have, as `drawn` laid it out with `pads`.
fn naturals(drawn: &DrawnTranscript, facts: &SideFacts, pads: &Pads) -> (Vec<usize>, usize) {
    let mut before = 0;
    let mut heads = Vec::with_capacity(facts.turns.len());
    for (k, turn) in facts.turns.iter().enumerate() {
        let pad = pads.before.get(k).copied().unwrap_or(0);
        let decorated = turn.anchor.and_then(|anchor| {
            let item = TranscriptItem::MessageDecoration(anchor, NAME.to_owned());
            drawn.rows_of(&item)
        });
        let at = match decorated {
            Some(rows) => rows.start.saturating_sub(before),
            None => turn
                .first
                .and_then(|first| drawn.rows_of(&TranscriptItem::Message(first)))
                .map_or(0, |rows| rows.start.saturating_sub(before + pad)),
        };
        heads.push(at);
        before += pad;
    }
    let total = drawn.rows().len().saturating_sub(before + pads.tail);
    (heads, total)
}

/// The padding of each side that starts the k-th turn of both at one row,
/// and makes both as long, with the row of each turn's header.
fn align(naturals: &[(Vec<usize>, usize); 2]) -> ([Pads; 2], Vec<usize>) {
    let turns = naturals
        .iter()
        .map(|(heads, _)| heads.len())
        .max()
        .unwrap_or(0);
    let mut pads = [Pads::default(), Pads::default()];
    let mut before = [0usize; 2];
    let mut rows: Vec<usize> = Vec::with_capacity(turns);
    for k in 0..turns {
        let previous = k.checked_sub(1).and_then(|k| rows.get(k).copied());
        let target = (0..2)
            .map(|at| {
                let (heads, total) = naturals.get(at).cloned().unwrap_or_default();
                match (heads.get(k), k.checked_sub(1).and_then(|k| heads.get(k))) {
                    (Some(head), _) => head + before.get(at).copied().unwrap_or(0),
                    (None, Some(last)) if heads.len() == k => {
                        previous.unwrap_or(0) + total.saturating_sub(*last)
                    }
                    _ => 0,
                }
            })
            .max()
            .unwrap_or(0);
        for (at, (heads, _)) in naturals.iter().enumerate() {
            if let (Some(head), Some(pads), Some(before)) =
                (heads.get(k), pads.get_mut(at), before.get_mut(at))
            {
                let pad = target.saturating_sub(head + *before);
                pads.before.push(pad);
                *before += pad;
            }
        }
        rows.push(target);
    }
    let ends = naturals.iter().enumerate().map(|(at, (heads, total))| {
        let last = heads.len().checked_sub(1);
        match last.and_then(|last| Some((rows.get(last)?, heads.get(last)?))) {
            Some((row, head)) => row + total.saturating_sub(*head),
            None => *total + before.get(at).copied().unwrap_or(0),
        }
    });
    let end = ends.max().unwrap_or(0);
    for (at, (_, total)) in naturals.iter().enumerate() {
        if let (Some(pads), Some(before)) = (pads.get_mut(at), before.get(at)) {
            pads.tail = end.saturating_sub(total + before);
        }
    }
    (pads, rows)
}

/// A side's look: under the message before each turn, its padding and a
/// header row; under its last message, the padding that makes it as long
/// as the other side.
fn turn_look(facts: &SideFacts, pads: &Pads) -> MessageDecorator {
    let mut rows: HashMap<MessageId, Vec<Row<'static>>> = HashMap::new();
    let style = Style::new().magenta().bold();
    for (k, turn) in facts.turns.iter().enumerate() {
        let Some(anchor) = turn.anchor else {
            continue;
        };
        let pad = pads.before.get(k).copied().unwrap_or(0);
        let under = rows.entry(anchor).or_default();
        under.extend(std::iter::repeat_n(Row::default(), pad));
        under.push(Row::styled(
            format!("▸ turn {} · {}", k + 1, turn.title),
            style,
        ));
    }
    if let Some(last) = facts.last {
        let under = rows.entry(last).or_default();
        under.extend(std::iter::repeat_n(Row::default(), pads.tail));
    }
    let rows = Arc::new(rows);
    MessageDecorator::new(NAME, move |message| {
        let drawn = message.id.and_then(|id| rows.get(&id));
        drawn.cloned().unwrap_or_default()
    })
}

/// The calls of each side whose tool and target the other side's calls
/// have none of.
fn only_here([left, right]: &[Side; 2]) -> [Vec<CallId>; 2] {
    let calls = |side: &Side| -> Vec<(CallId, CallKey)> {
        let turns = side.facts.turns.iter();
        turns.flat_map(|turn| turn.calls.iter().cloned()).collect()
    };
    let (ours, theirs) = (calls(left), calls(right));
    let only = |mine: &[(CallId, CallKey)], other: &[(CallId, CallKey)]| {
        let keys: HashSet<&CallKey> = other.iter().map(|(_, key)| key).collect();
        let only = mine.iter().filter(|(_, key)| !keys.contains(key));
        only.map(|(call, _)| call.clone()).collect()
    };
    [only(&ours, &theirs), only(&theirs, &ours)]
}

/// Draws each side's header over its column (a tab bar when the sides are
/// tabs) and the keys under them.
fn draw(
    mut views: Query<(&CompareView, &mut PanelCanvas), Without<ComparePane>>,
    panes: Query<&PanelCanvas, With<ComparePane>>,
) {
    for (view, mut canvas) in &mut views {
        let screen = canvas.area();
        let buffer = canvas.buffer_mut();
        let narrow = screen.width < NARROW;
        for (at, side) in view.sides.iter().enumerate() {
            let column = panes
                .get(side.pane)
                .map(PanelCanvas::area)
                .unwrap_or_default();
            if column.is_empty() {
                continue;
            }
            let area = Rect::new(column.x, screen.y, column.width, HEADER_ROWS);
            let focused = at == view.focus;
            let first = match narrow {
                true => tabs(view),
                false => {
                    let marker = if focused { "▶ " } else { "  " };
                    let current = if side.line.is_none() {
                        " (current)"
                    } else {
                        ""
                    };
                    let style = Style::new().bold();
                    Row::styled(format!("{marker}{}{current}", side.label), style)
                }
            };
            let facts = &side.facts;
            let models = match facts.models.is_empty() {
                true => "no model".to_owned(),
                false => facts.models.join(", "),
            };
            let turns = facts.turns.len();
            let tokens = facts.tokens.map_or_else(
                || "? tokens".to_owned(),
                |tokens| format!("{tokens} tokens"),
            );
            let plural = if turns == 1 { "" } else { "s" };
            let stats = format!("  {models} · {turns} turn{plural} · {tokens}");
            let files = match facts.files.is_empty() {
                true => "none".to_owned(),
                false => facts.files.join(", "),
            };
            let rows = vec![
                first,
                Row::from(stats).dim(),
                Row::from(format!("  files changed: {files}")),
            ];
            Paragraph::new(rows).render(area, buffer);
        }
        let scrolling = if view.together {
            "together"
        } else {
            "this side alone"
        };
        let help = format!(
            " Tab other side · ↑↓ turn · s scroll {scrolling} · p continue on this branch · Esc close"
        );
        let legend = Row::from(vec![
            Span::styled(" call ", Style::new().black().on_yellow()),
            Span::from(" only on this side"),
        ]);
        let bottom = screen.bottom().saturating_sub(2);
        let area = Rect::new(screen.x, bottom, screen.width, 2);
        Paragraph::new(vec![legend, Row::from(help).dim()]).render(area, buffer);
    }
}

/// The tab bar of sides shown one at a time: the focused one lit.
fn tabs(view: &CompareView) -> Row<'static> {
    let tabs = view.sides.iter().enumerate().map(|(at, side)| {
        let style = match at == view.focus {
            true => Style::new().bold().reversed(),
            false => Style::new().dim(),
        };
        Span::styled(format!(" {} ", side.label), style)
    });
    let mut spans = vec![Span::from(" ")];
    for tab in tabs {
        spans.push(tab);
        spans.push(Span::from(" "));
    }
    Row::from(spans)
}
