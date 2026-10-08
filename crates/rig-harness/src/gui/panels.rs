//! The panels, built again from the agents' components and the
//! [`Timeline`] whenever what they show changed: the agent graph, the
//! cost, the timeline, the notices, the details of the selected call or
//! agent, and the header and status lines. A panel is a [`Region`] node
//! whose children are despawned and spawned again; the scroll position
//! lives on the region, so it stays.

use std::collections::{HashSet, VecDeque};
use std::time::{Duration, Instant};

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use serde_json::{Map, Value};

use super::style::{
    self, BLUE, BODY, BORDER, CYAN, DIM, GREEN, MAGENTA, ORANGE, PANEL, RAISED, RED, SELECTED,
    SMALL, TEXT, TITLE, TURN, YELLOW, label, text,
};
use super::timeline::{Detail, Kind, Outcome, Span, Timeline};
use super::{Action, Dirty, GuiView, REBUILD_EVERY, Region, Zoom, status_text};
use crate::core::agent::{ActiveTurn, Agent, AgentId, CallOf, Connection, ModelChoice, TurnOf};
use crate::core::approval::{ApprovalAnswer, AwaitingApproval, Policy};
use crate::core::recovery::Backoff;
use crate::core::rewind::Forked;
use crate::core::subagents::{Delegated, RosterQuery, roster};
use crate::core::usage::{self, ContextUse, Spending, TurnSpending};
use crate::host::sessions::SessionName;

/// Notices kept for the notice panel.
const NOTICES: usize = 60;
/// Notices shown.
const NOTICES_SHOWN: usize = 6;
/// Bars drawn per lane, the newest.
const BARS_PER_LANE: usize = 400;
/// Rows of call bars per lane before they overlap.
const ROWS: usize = 6;
/// A bar row's height and the gap under it, in pixels.
const ROW: f32 = 14.0;
/// The lane label's width.
const LANE_LABEL: f32 = 150.0;
/// Model calls in the per-call cost chart.
const COST_BARS: usize = 60;

/// The last notices, for the notice panel.
#[derive(Resource, Default)]
pub(super) struct NoticeLog(VecDeque<NoticeEntry>);

struct NoticeEntry {
    at: Instant,
    agent: Option<Entity>,
    text: String,
    error: bool,
}

impl NoticeLog {
    pub(super) fn push(&mut self, agent: Option<Entity>, text: String, error: bool) {
        if self.0.len() >= NOTICES {
            self.0.pop_front();
        }
        self.0.push_back(NoticeEntry {
            at: Instant::now(),
            agent,
            text,
            error,
        });
    }
}

/// What the panels read.
type AgentQuery<'w, 's> = Query<
    'w,
    's,
    (
        &'static AgentId,
        Option<&'static Delegated>,
        Option<&'static Forked>,
        Option<&'static ModelChoice>,
        Has<ActiveTurn>,
        &'static Spending,
        Option<&'static Connection>,
        Option<&'static Policy>,
        Option<&'static ActiveTurn>,
    ),
    With<Agent>,
>;

#[derive(SystemParam)]
pub(super) struct Sources<'w, 's> {
    view: Res<'w, GuiView>,
    timeline: Res<'w, Timeline>,
    log: Res<'w, NoticeLog>,
    session: Option<Res<'w, SessionName>>,
    roster: RosterQuery<'w, 's>,
    agents: AgentQuery<'w, 's>,
    waiting: Query<'w, 's, (Entity, &'static CallOf, &'static AwaitingApproval)>,
    backoffs: Query<'w, 's, &'static CallOf, With<Backoff>>,
    turns: Query<'w, 's, (&'static TurnOf, &'static TurnSpending)>,
    regions: Query<'w, 's, (Entity, &'static Region)>,
}

/// One agent as the panels show it.
struct AgentInfo {
    entity: Entity,
    depth: usize,
    title: String,
    model: Option<String>,
    busy: bool,
    retrying: bool,
    /// Its calls waiting for approval: the call, the tool and the subject.
    waiting: Vec<(Entity, String, String)>,
    spending: Spending,
    turn: Option<Spending>,
    context: Option<ContextUse>,
    approvals: Option<&'static str>,
}

impl AgentInfo {
    fn color(&self) -> Color {
        if !self.waiting.is_empty() {
            YELLOW
        } else if self.retrying {
            ORANGE
        } else if self.busy {
            BLUE
        } else {
            GREEN
        }
    }

    fn state(&self) -> &'static str {
        if !self.waiting.is_empty() {
            "waiting for approval"
        } else if self.retrying {
            "retrying"
        } else if self.busy {
            "working"
        } else {
            "idle"
        }
    }
}

impl Sources<'_, '_> {
    /// Every agent in the graph's order: the roots by id, each followed by
    /// its subagents.
    fn agents(&self) -> Vec<AgentInfo> {
        let entries = roster(&self.roster);
        let several = entries.iter().filter(|entry| entry.depth == 0).count() > 1;
        entries
            .into_iter()
            .filter_map(|entry| {
                let (id, delegated, forked, model, busy, spending, connection, policy, turn) =
                    self.agents.get(entry.agent).ok()?;
                let short = |id: &str| id.get(..8).unwrap_or(id).to_owned();
                let title = match (delegated, forked) {
                    (Some(delegated), _) => cut(&delegated.task, 48),
                    (None, Some(forked)) => format!("fork of {}", short(&forked.from)),
                    (None, None) if several => format!("agent {}", short(&id.0)),
                    (None, None) => "main agent".to_owned(),
                };
                let waiting = self
                    .waiting
                    .iter()
                    .filter(|(_, CallOf(turn), _)| {
                        self.turns
                            .get(*turn)
                            .is_ok_and(|(&TurnOf(agent), _)| agent == entry.agent)
                    })
                    .map(|(call, _, ask)| (call, ask.tool.clone(), ask.subject.clone()))
                    .collect();
                let retrying = self.backoffs.iter().any(|&CallOf(turn)| {
                    self.turns
                        .get(turn)
                        .is_ok_and(|(&TurnOf(agent), _)| agent == entry.agent)
                });
                let turn = turn
                    .and_then(|turn| self.turns.get(turn.turn()).ok())
                    .map(|(_, spent)| spent.0);
                Some(AgentInfo {
                    entity: entry.agent,
                    depth: entry.depth,
                    title,
                    model: model.map(|model| model.0.clone()),
                    busy,
                    retrying,
                    waiting,
                    spending: *spending,
                    turn,
                    context: spending.context_use(connection.map(|connection| connection.spec)),
                    approvals: policy.map(|policy| policy.mode.name()),
                })
            })
            .collect()
    }

    fn region(&self, wanted: Region) -> Option<Entity> {
        self.regions
            .iter()
            .find(|(_, region)| **region == wanted)
            .map(|(entity, _)| entity)
    }
}

/// Builds the panels marked [`Dirty`], at most every [`REBUILD_EVERY`].
pub(super) fn rebuild(mut dirty: ResMut<Dirty>, sources: Sources, mut commands: Commands) {
    let any = dirty.graph || dirty.timeline || dirty.details || dirty.cost || dirty.notices;
    if !any || dirty.built.elapsed() < REBUILD_EVERY {
        return;
    }
    let agents = sources.agents();
    let shown = sources
        .view
        .agent
        .and_then(|agent| agents.iter().find(|info| info.entity == agent));
    let mut build = |region: Region, wanted: bool, fill: &dyn Fn(&mut ChildSpawnerCommands)| {
        if !wanted {
            return;
        }
        if let Some(entity) = sources.region(region) {
            commands
                .entity(entity)
                .despawn_children()
                .with_children(|parent| fill(parent));
        }
    };
    build(Region::Header, dirty.graph || dirty.cost, &|parent| {
        header(parent, &agents, sources.session.as_deref());
    });
    build(Region::Status, dirty.graph, &|parent| {
        if let Some(info) = shown {
            let (line, color) = status_text(&info.title, info.busy, info.waiting.len());
            parent.spawn(label(line, BODY, color));
        }
    });
    build(Region::Graph, dirty.graph, &|parent| {
        graph(parent, &agents, sources.view.agent);
    });
    build(Region::Cost, dirty.cost, &|parent| {
        cost(parent, &agents, shown, &sources.timeline);
    });
    build(Region::Timeline, dirty.timeline, &|parent| {
        timeline(parent, &agents, &sources.view, &sources.timeline);
    });
    build(Region::Notices, dirty.notices, &|parent| {
        notices(parent, &agents, &sources.log);
    });
    build(Region::Details, dirty.details, &|parent| match sources
        .view
        .span
        .and_then(|id| sources.timeline.get(id))
    {
        Some(span) => span_details(parent, span, &agents, &sources.waiting),
        None => match shown {
            Some(info) => overview(parent, info, &sources.timeline),
            None => {
                parent.spawn(text("No agent yet.", BODY, DIM));
            }
        },
    });
    dirty.graph = false;
    dirty.timeline = false;
    dirty.details = false;
    dirty.cost = false;
    dirty.notices = false;
    dirty.built = Instant::now();
}

/// `text` cut to `width` characters, with an ellipsis when cut.
fn cut(text: &str, width: usize) -> String {
    let line = text.lines().next().unwrap_or_default();
    let mut short: String = line.chars().take(width).collect();
    if short.len() < text.len() {
        short.push('…');
    }
    short
}

/// The title of `agent`, or a placeholder when it is gone.
fn title_of(agents: &[AgentInfo], agent: Entity) -> String {
    agents
        .iter()
        .find(|info| info.entity == agent)
        .map_or_else(|| "a finished agent".to_owned(), |info| info.title.clone())
}

fn header(parent: &mut ChildSpawnerCommands, agents: &[AgentInfo], session: Option<&SessionName>) {
    parent.spawn(label("rig", TITLE, TEXT));
    if let Some(name) = session.and_then(|session| session.0.as_deref()) {
        parent.spawn(label(name.to_owned(), BODY, DIM));
    }
    let working = agents.iter().filter(|info| info.busy).count();
    let waiting: usize = agents.iter().map(|info| info.waiting.len()).sum();
    parent.spawn(label(
        format!(
            "{} agent{} · {working} working",
            agents.len(),
            if agents.len() == 1 { "" } else { "s" }
        ),
        BODY,
        DIM,
    ));
    if waiting > 0 {
        parent.spawn(label(
            format!("{waiting} waiting for approval"),
            BODY,
            YELLOW,
        ));
    }
    if let Some(cost) = session_cost(agents) {
        parent.spawn(label(format!("{cost} this session"), BODY, DIM));
    }
}

/// Every agent's cost summed, as `$1.23`, with `+` when some calls had no
/// price.
fn session_cost(agents: &[AgentInfo]) -> Option<String> {
    let mut total = Spending::default();
    for info in agents {
        total.cost += info.spending.cost;
        total.calls += info.spending.calls;
        total.unpriced += info.spending.unpriced;
    }
    total.cost_label()
}

/// The agent graph: each agent under the one that started it, with its
/// state, model, cost and context. A click shows the agent in every view.
fn graph(parent: &mut ChildSpawnerCommands, agents: &[AgentInfo], selected: Option<Entity>) {
    if agents.is_empty() {
        parent.spawn(text("No agents.", BODY, DIM));
    }
    for info in agents {
        let mut detail = vec![info.model.clone().unwrap_or_else(|| "no model".to_owned())];
        if let Some(cost) = info.spending.cost_label() {
            detail.push(cost);
        }
        if let Some(context) = info.context {
            detail.push(format!("ctx {}", context.label()));
        }
        let branch = if info.depth > 0 { "└ " } else { "" };
        parent.spawn((
            Action::Agent(info.entity),
            Node {
                padding: UiRect {
                    left: px(6.0 + 16.0 * info.depth as f32),
                    right: px(6),
                    top: px(4),
                    bottom: px(4),
                },
                margin: UiRect::bottom(px(2)),
                column_gap: px(6),
                align_items: AlignItems::Center,
                border_radius: BorderRadius::all(px(4)),
                overflow: Overflow::clip(),
                ..default()
            },
            BackgroundColor(if selected == Some(info.entity) {
                SELECTED
            } else {
                PANEL
            }),
            children![
                label(format!("{branch}●"), BODY, info.color()),
                (
                    Node {
                        flex_direction: FlexDirection::Column,
                        min_width: px(0),
                        ..default()
                    },
                    children![
                        label(info.title.clone(), BODY, TEXT),
                        label(
                            format!("{} · {}", info.state(), detail.join(" · ")),
                            SMALL,
                            DIM
                        ),
                    ],
                ),
            ],
        ));
    }
}

/// The cost: the session's, each agent's as a bar, and the shown agent's
/// per model call, its tokens and its context.
fn cost(
    parent: &mut ChildSpawnerCommands,
    agents: &[AgentInfo],
    shown: Option<&AgentInfo>,
    timeline: &Timeline,
) {
    parent.spawn(style::heading("Cost"));
    let calls: u64 = agents.iter().map(|info| info.spending.calls).sum();
    parent.spawn(label(
        format!(
            "{} · {calls} model call{}",
            session_cost(agents).unwrap_or_else(|| "no priced calls".to_owned()),
            if calls == 1 { "" } else { "s" }
        ),
        BODY,
        TEXT,
    ));
    let most = agents
        .iter()
        .map(|info| info.spending.cost)
        .fold(0.0_f64, f64::max);
    if most > 0.0 {
        for info in agents.iter().filter(|info| info.spending.cost > 0.0) {
            parent.spawn((
                Node {
                    column_gap: px(6),
                    align_items: AlignItems::Center,
                    ..default()
                },
                children![
                    (
                        Node {
                            width: px(110),
                            overflow: Overflow::clip(),
                            ..default()
                        },
                        children![label(info.title.clone(), SMALL, DIM)],
                    ),
                    style::meter((info.spending.cost / most) as f32, BLUE, px(100)),
                    label(usage::dollars(info.spending.cost), SMALL, TEXT),
                ],
            ));
        }
    }
    let Some(info) = shown else {
        return;
    };
    // The shown agent's model calls, newest last, by cost.
    let costs: Vec<(f64, bool, u64)> = timeline
        .spans()
        .rev()
        .filter(|span| span.agent == info.entity && span.kind == Kind::Model)
        .filter_map(|span| match &span.detail {
            Detail::Model { usage, .. } => Some((
                usage
                    .as_ref()
                    .and_then(|usage| usage.cost.as_ref())
                    .map_or(0.0, |cost| cost.total),
                span.outcome == Outcome::Failed,
                span.id,
            )),
            _ => None,
        })
        .take(COST_BARS)
        .collect();
    if !costs.is_empty() {
        let highest = costs.iter().map(|(cost, ..)| *cost).fold(0.0_f64, f64::max);
        parent.spawn(label(
            format!("{}: cost per model call", info.title),
            SMALL,
            DIM,
        ));
        parent
            .spawn(Node {
                height: px(44),
                align_items: AlignItems::FlexEnd,
                column_gap: px(1),
                padding: UiRect::bottom(px(2)),
                border: UiRect::bottom(px(1)),
                ..default()
            })
            .insert(BorderColor::all(BORDER))
            .with_children(|chart| {
                for (cost, failed, id) in costs.iter().rev() {
                    let height = if highest > 0.0 {
                        (cost / highest * 100.0).max(4.0) as f32
                    } else {
                        4.0
                    };
                    chart.spawn((
                        Action::Span(*id),
                        Node {
                            width: px(4),
                            height: percent(height),
                            ..default()
                        },
                        BackgroundColor(if *failed { RED } else { BLUE }),
                    ));
                }
            });
    }
    let spent = &info.spending;
    parent.spawn(label(
        format!(
            "↑{} ↓{} R{} W{}",
            usage::tokens(spent.uncached_input()),
            usage::tokens(spent.tokens.output_tokens.unwrap_or(0)),
            usage::tokens(spent.tokens.cached_input_tokens.unwrap_or(0)),
            usage::tokens(spent.tokens.cache_creation_input_tokens.unwrap_or(0)),
        ),
        SMALL,
        DIM,
    ));
    if let Some(context) = info.context {
        let share = context.percent();
        parent.spawn(label(
            format!("context {}", context.label()),
            SMALL,
            share.map_or(DIM, style::context_color),
        ));
        if let Some(share) = share {
            parent.spawn(style::meter(
                share as f32 / 100.0,
                style::context_color(share),
                percent(100),
            ));
        }
    }
}

/// The colour of a span's bar.
fn bar_color(span: &Span) -> Color {
    match (span.outcome, span.kind) {
        (Outcome::Failed, _) => RED,
        (Outcome::Stopped, Kind::Turn) => TURN,
        (Outcome::Stopped, _) => DIM,
        (_, Kind::Turn) => TURN,
        (_, Kind::Model) => BLUE,
        (_, Kind::Tool) => GREEN,
        (_, Kind::Summary) => MAGENTA,
        (_, Kind::Retry) => ORANGE,
        (_, Kind::Other) => CYAN,
    }
}

/// The timeline: one lane per agent, in the graph's order, with its
/// turns as a thin band, its model calls on the first row and its tool
/// calls packed into the rows under it. Queued time is drawn dim and time
/// waiting for approval yellow. A click shows the call's details.
fn timeline(
    parent: &mut ChildSpawnerCommands,
    agents: &[AgentInfo],
    view: &GuiView,
    timeline: &Timeline,
) {
    parent
        .spawn((
            Node {
                column_gap: px(6),
                align_items: AlignItems::Center,
                margin: UiRect::bottom(px(6)),
                ..default()
            },
            children![style::label("Timeline", TITLE, TEXT)],
        ))
        .with_children(|bar| {
            for zoom in Zoom::ALL {
                bar.spawn((
                    Action::Zoom(zoom),
                    style::button_node(view.zoom == zoom),
                    children![label(zoom.label(), SMALL, TEXT)],
                ));
            }
            for (name, color) in [
                ("model", BLUE),
                ("tool", GREEN),
                ("summary", MAGENTA),
                ("retry", ORANGE),
                ("approval", YELLOW),
                ("failed", RED),
            ] {
                bar.spawn(label(format!("■ {name}"), SMALL, color));
            }
        });
    let now = Instant::now();
    let lanes: HashSet<Entity> = agents.iter().map(|info| info.entity).collect();
    let from = match view.zoom.window() {
        Some(window) => now.checked_sub(window).unwrap_or(timeline.opened),
        None => timeline
            .spans()
            .filter(|span| lanes.contains(&span.agent))
            .map(|span| span.start)
            .min()
            .unwrap_or(timeline.opened),
    };
    let total = now
        .saturating_duration_since(from)
        .max(Duration::from_secs(1))
        .as_secs_f32();
    let at = |instant: Instant| {
        (instant.saturating_duration_since(from).as_secs_f32() / total * 100.0).clamp(0.0, 100.0)
    };
    // The time axis.
    parent
        .spawn(Node {
            height: px(16),
            margin: UiRect::left(px(LANE_LABEL)),
            ..default()
        })
        .with_children(|axis| {
            for step in 0..=4_u8 {
                let fraction = f32::from(step) / 4.0;
                let ago = Duration::from_secs_f32(total * (1.0 - fraction));
                let tick = if step == 4 {
                    "now".to_owned()
                } else {
                    format!("-{}", style::duration(ago))
                };
                let mut node = Node {
                    position_type: PositionType::Absolute,
                    ..default()
                };
                if step == 4 {
                    node.right = px(0);
                } else {
                    node.left = percent(fraction * 100.0);
                }
                axis.spawn((node, children![label(tick, SMALL, DIM)]));
            }
        });
    let mut empty = true;
    for info in agents {
        let spans: Vec<&Span> = timeline
            .spans()
            .rev()
            .filter(|span| span.agent == info.entity && span.end_or(now) >= from)
            .take(BARS_PER_LANE)
            .collect();
        if !spans.is_empty() {
            empty = false;
        }
        // Calls into rows: model calls and summaries on row 0, the others
        // on the first row under it that is free when they start.
        let mut rows: Vec<Instant> = Vec::new();
        let mut placed: Vec<(&Span, Option<usize>)> = Vec::new();
        for span in spans.iter().rev() {
            let row = match span.kind {
                Kind::Turn => None,
                Kind::Model | Kind::Summary => Some(0),
                Kind::Tool | Kind::Retry | Kind::Other => {
                    let free = rows.iter().position(|end| *end <= span.start);
                    let row = match free {
                        Some(row) => row,
                        None if rows.len() < ROWS => {
                            rows.push(span.start);
                            rows.len() - 1
                        }
                        None => ROWS - 1,
                    };
                    if let Some(end) = rows.get_mut(row) {
                        *end = span.end_or(now);
                    }
                    Some(row + 1)
                }
            };
            placed.push((span, row));
        }
        let height = 8.0 + ROW * (rows.len() + 1) as f32 + 2.0;
        parent
            .spawn(Node {
                margin: UiRect::bottom(px(4)),
                min_height: px(height),
                ..default()
            })
            .with_children(|lane| {
                lane.spawn((
                    Action::Agent(info.entity),
                    Node {
                        width: px(LANE_LABEL),
                        flex_direction: FlexDirection::Column,
                        padding: UiRect::axes(px(4), px(2)),
                        overflow: Overflow::clip(),
                        ..default()
                    },
                    BackgroundColor(if view.agent == Some(info.entity) {
                        SELECTED
                    } else {
                        Color::NONE
                    }),
                    children![
                        label(
                            format!("{}{}", "  ".repeat(info.depth), info.title),
                            SMALL,
                            TEXT
                        ),
                        label(info.state(), SMALL, info.color()),
                    ],
                ));
                lane.spawn((
                    Node {
                        flex_grow: 1.0,
                        height: px(height),
                        border_radius: BorderRadius::all(px(3)),
                        overflow: Overflow::clip(),
                        ..default()
                    },
                    BackgroundColor(PANEL),
                ))
                .with_children(|track| {
                    for (span, row) in &placed {
                        let (top, tall) = match row {
                            None => (1.0, 5.0),
                            Some(row) => (8.0 + ROW * *row as f32, ROW - 2.0),
                        };
                        let start = at(span.start);
                        let end = at(span.end_or(now));
                        let selected = view.span == Some(span.id);
                        let mut bar = track.spawn((
                            Action::Span(span.id),
                            Node {
                                position_type: PositionType::Absolute,
                                left: percent(start),
                                width: percent((end - start).max(0.25)),
                                top: px(top),
                                height: px(tall),
                                border: UiRect::all(px(if selected { 1 } else { 0 })),
                                border_radius: BorderRadius::all(px(2)),
                                overflow: Overflow::clip(),
                                ..default()
                            },
                            BorderColor::all(TEXT),
                            BackgroundColor(bar_color(span)),
                        ));
                        if row.is_some() {
                            bar.with_children(|inside| {
                                // Queued time, then approval time, over the bar.
                                let width = (end - start).max(0.25);
                                let part = |from: Instant, to: Instant| {
                                    let left = (at(from) - start) / width * 100.0;
                                    let right = (at(to) - start) / width * 100.0;
                                    (left.clamp(0.0, 100.0), (right - left).clamp(0.0, 100.0))
                                };
                                if let Some(begun) = span.begun.filter(|begun| *begun > span.start)
                                {
                                    let (left, wide) = part(span.start, begun);
                                    inside.spawn(overlay(left, wide, TURN));
                                }
                                if let Some((asked, answered)) = span.approval {
                                    let (left, wide) =
                                        part(asked, answered.unwrap_or_else(|| span.end_or(now)));
                                    inside.spawn(overlay(left, wide, YELLOW));
                                }
                                inside.spawn((
                                    label(cut(&span.label, 40), SMALL - 1.0, Color::BLACK),
                                    Node {
                                        margin: UiRect::left(px(3)),
                                        ..default()
                                    },
                                ));
                            });
                        }
                    }
                });
            });
    }
    if empty {
        parent.spawn(text("No calls yet. Type below and press Enter.", BODY, DIM));
    }
}

/// A part of a bar, from `left` percent of it and `wide` percent wide.
fn overlay(left: f32, wide: f32, color: Color) -> impl Bundle {
    (
        Node {
            position_type: PositionType::Absolute,
            left: percent(left),
            width: percent(wide),
            height: percent(100),
            ..default()
        },
        BackgroundColor(color),
        bevy::picking::Pickable::IGNORE,
    )
}

fn notices(parent: &mut ChildSpawnerCommands, agents: &[AgentInfo], log: &NoticeLog) {
    let shown: Vec<&NoticeEntry> = log.0.iter().rev().take(NOTICES_SHOWN).collect();
    if shown.is_empty() {
        parent.spawn(text("Notices appear here.", SMALL, DIM));
    }
    for notice in shown.into_iter().rev() {
        let who = notice
            .agent
            .map(|agent| format!("{}: ", title_of(agents, agent)))
            .unwrap_or_default();
        parent.spawn(label(
            format!(
                "{} ago  {who}{}",
                style::duration(notice.at.elapsed()),
                cut(&notice.text, 160)
            ),
            SMALL,
            if notice.error { RED } else { DIM },
        ));
    }
}

/// A row of buttons answering the approval of `call`.
fn approval_buttons(parent: &mut ChildSpawnerCommands, call: Entity) {
    parent.spawn((
        Node {
            column_gap: px(6),
            margin: UiRect::vertical(px(4)),
            ..default()
        },
        children![
            (
                Action::Approve(call, ApprovalAnswer::Allow),
                style::button_node(false),
                children![label("Allow", BODY, GREEN)],
            ),
            (
                Action::Approve(call, ApprovalAnswer::AllowAlways),
                style::button_node(false),
                children![label("Always allow this tool", BODY, GREEN)],
            ),
            (
                Action::Approve(
                    call,
                    ApprovalAnswer::Deny {
                        reason: "the user refused it in the window".to_owned()
                    }
                ),
                style::button_node(false),
                children![label("Deny", BODY, RED)],
            ),
        ],
    ));
}

/// A section heading inside the details.
fn section(parent: &mut ChildSpawnerCommands, title: impl Into<String>) {
    parent.spawn((
        label(title, BODY, TEXT),
        Node {
            margin: UiRect::top(px(8)),
            ..default()
        },
    ));
}

/// Lines of `body`, at most `lines` of them, wrapped to the panel, in
/// `color`, with how many were left out.
fn body(parent: &mut ChildSpawnerCommands, content: &str, lines: usize, color: Color) {
    let (shown, more) = style::excerpt(content, lines, 400);
    if !shown.is_empty() {
        parent.spawn(text(shown.join("\n"), SMALL, color));
    }
    if more > 0 {
        parent.spawn(label(format!("… {more} more lines"), SMALL, DIM));
    }
}

/// A unified diff, a line per node coloured by its sign.
fn diff(parent: &mut ChildSpawnerCommands, unified: &str, lines: usize) {
    let (shown, more) = style::excerpt(unified, lines, 300);
    for line in shown {
        let color = if line.starts_with("@@") {
            CYAN
        } else if line.starts_with("+++") || line.starts_with("---") {
            DIM
        } else if line.starts_with('+') {
            GREEN
        } else if line.starts_with('-') {
            RED
        } else {
            TEXT
        };
        parent.spawn(text(line, SMALL, color));
    }
    if more > 0 {
        parent.spawn(label(format!("… {more} more lines"), SMALL, DIM));
    }
}

/// Whether `text` holds a unified diff's hunk.
fn has_hunk(text: &str) -> bool {
    text.lines().any(|line| line.starts_with("@@"))
}

/// The diff an `edit` call's arguments ask for, one hunk group per edit.
fn edit_diff(arguments: &Map<String, Value>) -> Option<String> {
    let edits = arguments.get("edits")?.as_array()?;
    let mut out = String::new();
    for edit in edits {
        let old = edit
            .get("old_text")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let new = edit
            .get("new_text")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let diff = similar::TextDiff::from_lines(old, new);
        out.push_str(&diff.unified_diff().context_radius(2).to_string());
    }
    Some(out)
}

/// The selected span: what it was, how long it took and what it said.
fn span_details(
    parent: &mut ChildSpawnerCommands,
    span: &Span,
    agents: &[AgentInfo],
    waiting: &Query<(Entity, &CallOf, &AwaitingApproval)>,
) {
    parent.spawn((
        Node {
            margin: UiRect::bottom(px(6)),
            ..default()
        },
        children![(
            Action::Overview,
            style::button_node(false),
            children![label("← agent", SMALL, TEXT)],
        )],
    ));
    let kind = match span.kind {
        Kind::Turn => "Turn",
        Kind::Model => "Model call",
        Kind::Tool => "Tool call",
        Kind::Summary => "Compaction summary",
        Kind::Retry => "Retry wait",
        Kind::Other => "Call",
    };
    parent.spawn(text(format!("{kind}: {}", span.label), TITLE, TEXT));
    let now = Instant::now();
    let took = span.end_or(now).saturating_duration_since(span.start);
    let (state, color) = match span.outcome {
        Outcome::Running => (format!("running for {}", style::duration(took)), BLUE),
        Outcome::Done => (format!("took {}", style::duration(took)), GREEN),
        Outcome::Failed => (format!("failed after {}", style::duration(took)), RED),
        Outcome::Stopped => (format!("stopped after {}", style::duration(took)), DIM),
    };
    parent.spawn(label(
        format!(
            "{} · started {} ago",
            title_of(agents, span.agent),
            style::duration(now.saturating_duration_since(span.start))
        ),
        SMALL,
        DIM,
    ));
    parent.spawn(label(state, SMALL, color));
    if let Some(begun) = span.begun.filter(|begun| *begun > span.start) {
        parent.spawn(label(
            format!(
                "queued {} behind calls touching the same files",
                style::duration(begun.saturating_duration_since(span.start))
            ),
            SMALL,
            DIM,
        ));
    }
    if let Some((asked, answered)) = span.approval {
        let line = match answered {
            Some(answered) => format!(
                "waited {} for approval",
                style::duration(answered.saturating_duration_since(asked))
            ),
            None => "waiting for approval".to_owned(),
        };
        parent.spawn(label(line, SMALL, YELLOW));
    }
    if let Ok((call, _, ask)) = waiting.get(span.entity) {
        parent.spawn(text(
            format!("{} wants to run on {}", ask.tool, cut(&ask.subject, 200)),
            BODY,
            YELLOW,
        ));
        approval_buttons(parent, call);
    }
    match &span.detail {
        Detail::None => {}
        Detail::Model {
            model,
            effect,
            usage,
            text: said,
            reasoning,
            tool_calls,
            error,
        } => {
            let mut line = model.clone();
            if let Some(effect) = effect {
                line.push_str(&format!(" · {effect} in effects.jsonl"));
            }
            parent.spawn(label(line, SMALL, DIM));
            if let Some(usage) = usage {
                let mut spent = Spending::default();
                spent.record(usage);
                parent.spawn(label(spent.summary(), SMALL, TEXT));
            }
            if *reasoning > 0 {
                parent.spawn(label(
                    format!("{reasoning} characters of reasoning"),
                    SMALL,
                    DIM,
                ));
            }
            if let Some(error) = error {
                section(parent, "Error");
                body(parent, error, 40, RED);
            }
            if !tool_calls.is_empty() {
                section(parent, "Asked for");
                parent.spawn(text(tool_calls.join(", "), SMALL, GREEN));
            }
            if !said.is_empty() {
                section(parent, "Reply");
                body(parent, said, 80, TEXT);
            }
        }
        Detail::Tool {
            name,
            arguments,
            result,
            failed,
        } => {
            section(parent, "Arguments");
            match name.as_str() {
                "edit" => match edit_diff(arguments) {
                    Some(asked) if result.as_deref().is_none_or(|result| !has_hunk(result)) => {
                        diff(parent, &asked, 200);
                    }
                    _ => {
                        if let Some(path) = arguments.get("path").and_then(Value::as_str) {
                            parent.spawn(text(path.to_owned(), SMALL, TEXT));
                        }
                    }
                },
                "write" => {
                    if let Some(path) = arguments.get("path").and_then(Value::as_str) {
                        parent.spawn(text(path.to_owned(), SMALL, TEXT));
                    }
                    if let Some(content) = arguments.get("content").and_then(Value::as_str) {
                        let added: String =
                            content.lines().map(|line| format!("+{line}\n")).collect();
                        diff(parent, &added, 120);
                    }
                }
                "shell" => {
                    let command = arguments
                        .get("command")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    body(parent, &format!("$ {command}"), 40, TEXT);
                }
                _ => {
                    let pretty = serde_json::to_string_pretty(arguments).unwrap_or_default();
                    body(parent, &pretty, 60, TEXT);
                }
            }
            if let Some(result) = result {
                section(parent, if *failed { "Error" } else { "Result" });
                if *failed {
                    body(parent, result, 80, RED);
                } else if has_hunk(result) {
                    diff(parent, result, 300);
                } else {
                    body(parent, result, 120, TEXT);
                }
            }
        }
        Detail::Summary {
            messages,
            tokens,
            text: written,
            error,
        } => {
            parent.spawn(label(
                format!(
                    "{messages} messages, about {} tokens",
                    usage::tokens(*tokens)
                ),
                SMALL,
                DIM,
            ));
            if let Some(error) = error {
                section(parent, "Refused");
                body(parent, error, 20, RED);
            }
            if let Some(written) = written {
                section(parent, "Summary");
                body(parent, written, 120, TEXT);
            }
        }
        Detail::Retry { attempt, why } => {
            parent.spawn(label(format!("attempt {attempt}"), SMALL, DIM));
            body(parent, why, 20, ORANGE);
        }
    }
}

/// The shown agent: its state, what waits for the user, its spending and
/// context, the files it changed and its latest calls.
fn overview(parent: &mut ChildSpawnerCommands, info: &AgentInfo, timeline: &Timeline) {
    parent.spawn(text(info.title.clone(), TITLE, TEXT));
    parent.spawn(label(
        format!(
            "{} · {}{}",
            info.model.as_deref().unwrap_or("no model"),
            info.state(),
            info.approvals
                .filter(|mode| *mode != "auto")
                .map(|mode| format!(" · approvals: {mode}"))
                .unwrap_or_default()
        ),
        SMALL,
        info.color(),
    ));
    parent.spawn((
        Node {
            column_gap: px(6),
            margin: UiRect::vertical(px(6)),
            ..default()
        },
        children![
            (
                Action::Stop,
                style::button_node(false),
                children![label("Stop", SMALL, TEXT)],
            ),
            (
                Action::Retry,
                style::button_node(false),
                children![label("Retry", SMALL, TEXT)],
            ),
            (
                Action::Compact,
                style::button_node(false),
                children![label("Compact", SMALL, TEXT)],
            ),
        ],
    ));
    if !info.waiting.is_empty() {
        section(parent, "Waiting for approval");
        for (call, tool, subject) in &info.waiting {
            parent.spawn(text(
                format!("{tool}: {}", cut(subject, 200)),
                SMALL,
                YELLOW,
            ));
            approval_buttons(parent, *call);
        }
    }
    section(parent, "Spending");
    parent.spawn(text(info.spending.summary(), SMALL, TEXT));
    if let Some(turn) = &info.turn
        && turn.calls > 0
    {
        parent.spawn(text(format!("this turn: {}", turn.summary()), SMALL, DIM));
    }
    if let Some(context) = info.context {
        let color = context.percent().map_or(DIM, style::context_color);
        parent.spawn(label(format!("context {}", context.label()), SMALL, color));
        if let Some(share) = context.percent() {
            parent.spawn(style::meter(share as f32 / 100.0, color, percent(100)));
        }
    }
    let mine: Vec<&Span> = timeline
        .spans()
        .rev()
        .filter(|span| span.agent == info.entity && span.kind != Kind::Turn)
        .collect();
    let mut seen = HashSet::new();
    let changed: Vec<(&Span, &str)> = mine
        .iter()
        .filter(|span| span.outcome == Outcome::Done)
        .filter_map(|span| match &span.detail {
            Detail::Tool {
                name, arguments, ..
            } if name == "edit" || name == "write" => arguments
                .get("path")
                .and_then(Value::as_str)
                .filter(|path| seen.insert(*path))
                .map(|path| (*span, path)),
            _ => None,
        })
        .collect();
    if !changed.is_empty() {
        section(parent, "Files changed (newest first; click for the diff)");
        for (span, path) in changed {
            row(parent, span, path.to_owned());
        }
    }
    if !mine.is_empty() {
        section(parent, "Latest calls");
        for span in mine.into_iter().take(20) {
            row(parent, span, span.label.clone());
        }
    }
}

/// A clickable line for `span`: its colour, `title` and how long it took.
fn row(parent: &mut ChildSpawnerCommands, span: &Span, title: String) {
    let took = span
        .end_or(Instant::now())
        .saturating_duration_since(span.start);
    parent.spawn((
        Action::Span(span.id),
        Node {
            column_gap: px(6),
            padding: UiRect::axes(px(4), px(2)),
            align_items: AlignItems::Center,
            border_radius: BorderRadius::all(px(3)),
            overflow: Overflow::clip(),
            ..default()
        },
        BackgroundColor(RAISED),
        children![
            label("■", SMALL, bar_color(span)),
            label(cut(&title, 70), SMALL, TEXT),
            label(style::duration(took), SMALL, DIM),
        ],
    ));
}
