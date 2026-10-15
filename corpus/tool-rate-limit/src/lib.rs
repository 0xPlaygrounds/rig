//! Plugin corpus: 05. Tool rate limits (spec `05-tool-rate-limit.md`).
//!
//! Limits per tool name, from `RIG_HOME/limits.toml`:
//!
//! ```toml
//! [web_search]
//! per_minute = 10
//! shared = true   # one budget for every agent
//!
//! [fetch]
//! per_turn = 3
//! concurrent = 1
//! ```
//!
//! A Gate system sees each call of a limited tool as it is about to run.
//! It refuses one over `per_turn` or `per_session`; it holds one over
//! `concurrent` until a running call of the tool is answered, and one over
//! `per_minute` until its oldest start leaves the minute. A released call
//! is gated again, oldest first. A held call holds back nothing but the
//! calls the kernel orders after it, and an interrupt takes its hold away
//! with the turn.

use std::collections::{BTreeMap, HashMap, VecDeque};

use bevy_log::warn;
use rig_harness::prelude::*;
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// The window `per_minute` counts in.
const WINDOW: Duration = Duration::from_secs(60);

/// The limits of one tool; a limit left out is no limit.
#[derive(Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct Limit {
    /// Calls per turn of an agent; more are refused.
    pub per_turn: Option<u32>,
    /// Calls started in any 60 s; more wait.
    pub per_minute: Option<u32>,
    /// Calls per session; more are refused.
    pub per_session: Option<u32>,
    /// Calls running at once; more wait.
    pub concurrent: Option<u32>,
    /// One budget for every agent, instead of one per agent.
    pub shared: bool,
}

/// Each limited tool's limits, by name. Inserted before the plugin, it
/// replaces `RIG_HOME/limits.toml`.
#[derive(Resource, Deserialize, Clone, Debug, Default)]
#[serde(transparent)]
pub struct Limits(pub BTreeMap<String, Limit>);

impl Limits {
    /// `RIG_HOME/limits.toml`; none when it is missing or wrong, with a
    /// warning for the latter.
    fn load(home: &Home) -> Self {
        let path = home.root().join("limits.toml");
        let Ok(text) = std::fs::read_to_string(&path) else {
            return Self::default();
        };
        toml::from_str(&text).unwrap_or_else(|error| {
            warn!("{} ignored: {error}", path.display());
            Self::default()
        })
    }
}

/// Whose budget a call counts in: its agent's, or `None` for a shared one.
type Budget = Option<Entity>;

/// A tool's calls in one budget.
type Key = (Budget, String);

/// How many calls of each tool an agent made this session.
#[derive(Component, Reflect, Default)]
#[reflect(Component, Saved)]
struct SessionUsage(HashMap<String, u32>);

/// How many calls of each shared-budget tool all agents made this session.
#[derive(Resource, Reflect, Default)]
#[reflect(Resource, Saved)]
struct SharedUsage(HashMap<String, u32>);

/// How many calls of each tool the turn made.
#[derive(Component, Default)]
struct TurnUsage(HashMap<String, u32>);

/// When each budget's calls of each tool were let through, in about the
/// last minute, oldest first.
#[derive(Resource, Default)]
struct Recent(HashMap<Key, VecDeque<Duration>>);

impl Recent {
    /// The starts of `key` in the minute up to `now`, oldest first.
    fn within(&mut self, key: &Key, now: Duration) -> &mut VecDeque<Duration> {
        let starts = self.0.entry(key.clone()).or_default();
        starts.retain(|start| *start + WINDOW > now);
        starts
    }
}

/// On a call this plugin let through and not yet answered: it takes a
/// `concurrent` slot.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Slot(Key);

/// A hold on a call waiting for a slot: until a running call of its tool
/// is answered, or until `until` on Bevy's clock for a full minute.
#[derive(Component)]
struct WaitingFor {
    key: Key,
    until: Option<Duration>,
}

#[derive(Default)]
pub struct ToolRateLimitPlugin;

/// The plugin's stage system, in [`Stage::Gate`]: it refuses or holds a
/// call over its tool's limits.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ToolRateLimitSystems;

impl Plugin for ToolRateLimitPlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<Limits>() {
            let limits = Limits::load(&RigHome::of(app.world()));
            app.insert_resource(limits);
        }
        app.init_resource::<SharedUsage>()
            .init_resource::<Recent>()
            .register_required_components::<Agent, SessionUsage>()
            .register_required_components::<TurnOf, TurnUsage>()
            .configure_sets(AgentPass, ToolRateLimitSystems.in_set(Stage::Gate))
            .add_systems(AgentPass, limit.in_set(ToolRateLimitSystems))
            .add_observer(free_on_answer)
            .add_observer(free)
            .add_command("limits", "Show each tool limit and its use", show_limits);
        #[cfg(feature = "tui")]
        app.add_systems(
            PostUpdate,
            status::show.in_set(rig_tui::StatusSystems::Show),
        );
    }
}

/// Each call of a limited tool about to run, oldest first: refused over
/// `per_turn` or `per_session`, held while `concurrent` or `per_minute`
/// has no slot, else let through and counted. A call a restart runs again
/// was counted before: it only takes a slot.
fn limit(
    mut calls: Query<
        (
            Entity,
            &Effect,
            &ToolRequest,
            &ForAgent,
            Option<&CallOf>,
            Has<Restored>,
            &mut Decision,
        ),
        (ReadyToStart, ToolCalls, Without<Slot>),
    >,
    slots: Query<&Slot>,
    limits: Res<Limits>,
    mut turns: Query<&mut TurnUsage>,
    mut agents: Query<&mut SessionUsage>,
    mut shared: ResMut<SharedUsage>,
    mut recent: ResMut<Recent>,
    time: Res<Time>,
    mut commands: Commands,
) {
    let now = time.elapsed();
    let mut running: HashMap<Key, u32> = HashMap::new();
    for Slot(key) in &slots {
        *running.entry(key.clone()).or_default() += 1;
    }
    let mut queue: Vec<_> = calls.iter_mut().collect();
    queue.sort_by_key(|(_, effect, ..)| effect.id());
    for (call, _, request, &ForAgent(agent), of, restored, mut decision) in queue {
        let tool = request.call.function.name.as_str();
        // Another gate decided it: it never runs, so it takes no slot.
        if decision.is_decided() {
            continue;
        }
        let Some(limit) = limits.0.get(tool) else {
            continue;
        };
        let key = ((!limit.shared).then_some(agent), tool.to_owned());
        let turn = of.map(|&CallOf(turn)| turn);
        let session = if limit.shared {
            Some(&shared.0)
        } else {
            agents.get(agent).ok().map(|own| &own.0)
        };
        let used = |usage: Option<&HashMap<String, u32>>| {
            usage
                .and_then(|usage| usage.get(tool))
                .copied()
                .unwrap_or(0)
        };
        let turn_used = used(
            turn.and_then(|turn| turns.get(turn).ok())
                .map(|turn| &turn.0),
        );
        let over = [
            (limit.per_turn, turn_used, "turn"),
            (limit.per_session, used(session), "session"),
        ]
        .into_iter()
        .find_map(|(most, used, per)| most.filter(|most| used >= *most).map(|most| (most, per)));
        if let Some((most, per)) = over.filter(|_| !restored) {
            let why = format!("limit reached: {most} per {per} for `{tool}`");
            decision.decide(Decider::plugin("tool-rate-limit"), Outcome::refused(why));
            continue;
        }
        let busy = running.entry(key.clone()).or_default();
        if limit.concurrent.is_some_and(|most| *busy >= most) {
            commands.spawn((WaitingFor { key, until: None }, HoldOf(call)));
            continue;
        }
        let starts = recent.within(&key, now);
        let full = limit
            .per_minute
            .is_some_and(|most| starts.len() >= usize::try_from(most).unwrap_or(usize::MAX));
        if let (true, Some(oldest)) = (full, starts.front()) {
            let until = *oldest + WINDOW;
            let wait = WaitingFor {
                key,
                until: Some(until),
            };
            // The status line counts the wait down each second.
            let hold = (wait, HoldOf(call), KeepAwake(Duration::from_secs(1)));
            let hold = commands.spawn(hold).id();
            commands
                .delayed()
                .duration(until.saturating_sub(now))
                .entity(hold)
                .try_despawn();
            continue;
        }
        starts.push_back(now);
        *busy += 1;
        if !restored {
            if let Some(mut turn) = turn.and_then(|turn| turns.get_mut(turn).ok()) {
                *turn.0.entry(tool.to_owned()).or_default() += 1;
            }
            if limit.shared {
                *shared.0.entry(tool.to_owned()).or_default() += 1;
            } else if let Ok(mut own) = agents.get_mut(agent) {
                *own.0.entry(tool.to_owned()).or_default() += 1;
            }
        }
        commands.entity(call).insert(Slot(key));
    }
}

/// An answered call gives its slot back.
fn free_on_answer(answered: On<Insert<Outcome>>, mut commands: Commands) {
    commands.entity(answered.entity).try_remove::<Slot>();
}

/// A slot given back, answered or despawned unanswered as an interrupt
/// does, releases the calls of its tool waiting for one: they are gated
/// again.
fn free(
    freed: On<Remove<Slot>>,
    slots: Query<&Slot>,
    waits: Query<(Entity, &WaitingFor)>,
    mut commands: Commands,
) {
    let Ok(Slot(key)) = slots.get(freed.entity) else {
        return;
    };
    for (hold, wait) in &waits {
        if wait.until.is_none() && wait.key == *key {
            commands.entity(hold).despawn();
        }
    }
}

/// `/limits`: each limit and what the agent used of it.
fn show_limits(
    In(args): In<CommandArgs>,
    limits: Res<Limits>,
    agents: Query<(&SessionUsage, Option<&ActiveTurn>)>,
    turns: Query<&TurnUsage>,
    shared: Res<SharedUsage>,
    mut recent: ResMut<Recent>,
    slots: Query<&Slot>,
    waits: Query<&WaitingFor>,
    time: Res<Time>,
    mut notices: MessageWriter<Notice>,
) {
    if limits.0.is_empty() {
        let none = "No tool limits: RIG_HOME/limits.toml sets none.";
        notices.write(Notice::info(args.agent, none));
        return;
    }
    let (session, turn) = agents
        .get(args.agent)
        .map(|(session, active)| {
            let turn = active.and_then(|active| turns.get(active.turn()).ok());
            (Some(&session.0), turn.map(|turn| &turn.0))
        })
        .unwrap_or_default();
    let count = |n: usize| u32::try_from(n).unwrap_or(u32::MAX);
    let mut lines = vec!["Tool limits:".to_owned()];
    for (tool, limit) in &limits.0 {
        let key = ((!limit.shared).then_some(args.agent), tool.clone());
        let sessions = if limit.shared {
            Some(&shared.0)
        } else {
            session
        };
        let used = |usage: Option<&HashMap<String, u32>>| {
            usage
                .and_then(|usage| usage.get(tool))
                .copied()
                .unwrap_or(0)
        };
        let running = slots.iter().filter(|Slot(of)| *of == key).count();
        let waiting = waits.iter().filter(|wait| wait.key == key).count();
        let minute = recent.within(&key, time.elapsed()).len();
        let parts = [
            (limit.per_turn, used(turn), "this turn"),
            (limit.per_minute, count(minute), "this minute"),
            (limit.per_session, used(sessions), "this session"),
            (limit.concurrent, count(running), "running"),
        ];
        let mut said: Vec<String> = parts
            .into_iter()
            .filter_map(|(most, used, what)| most.map(|most| format!("{used}/{most} {what}")))
            .collect();
        if waiting > 0 {
            said.push(format!("{waiting} waiting"));
        }
        let whose = if limit.shared { "shared" } else { "per agent" };
        lines.push(format!("  {tool} ({whose}): {}", said.join(", ")));
    }
    notices.write(Notice::info(args.agent, lines.join("\n")));
}

#[cfg(feature = "tui")]
mod status {
    use rig_tui::{Side, StatusItem, StatusItems, Tone};

    use super::*;

    /// After the other agents at work; gone before them.
    const WAITING: StatusItem = StatusItem::new("tool-rate-limit", Side::Left, 75, 12);

    /// "fetch: waiting 12 s" on each turn with a call waiting for the
    /// minute; "fetch: waiting for a running call" for one waiting on
    /// `concurrent`.
    pub(super) fn show(
        mut turns: Query<(Entity, &mut StatusItems), With<TurnOf>>,
        waits: Query<(&WaitingFor, &HoldOf)>,
        of: Query<&CallOf>,
        time: Res<Time>,
        mut shown: Local<HashMap<Entity, String>>,
    ) {
        let now = time.elapsed();
        let mut said: HashMap<Entity, Vec<String>> = HashMap::new();
        for (wait, &HoldOf(call)) in &waits {
            let Ok(&CallOf(turn)) = of.get(call) else {
                continue;
            };
            let tool = &wait.key.1;
            let text = match wait.until {
                Some(until) => {
                    let left = until.saturating_sub(now).as_secs().max(1);
                    format!("{tool}: waiting {left} s")
                }
                None => format!("{tool}: waiting for a running call"),
            };
            let texts = said.entry(turn).or_default();
            if !texts.contains(&text) {
                texts.push(text);
            }
        }
        for (turn, mut items) in &mut turns {
            let text = said.remove(&turn).unwrap_or_default().join(", ");
            if shown.get(&turn) != Some(&text) {
                items.show(WAITING.says(text.clone(), Tone::Dim));
                shown.insert(turn, text);
            }
        }
        shown.retain(|turn, _| turns.contains(*turn));
    }
}
