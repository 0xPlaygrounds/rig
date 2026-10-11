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
//! A call over `per_turn` or `per_session` is refused; one over
//! `per_minute` or `concurrent` waits for a slot, without holding back other
//! tools' calls. The kernel starts a tool's call in the same system that
//! spawns it, with no hook before, so only a tool registered through
//! [`AppLimitedToolsExt::add_limited_tool_with`] is limited: it becomes an
//! open tool whose calls this plugin starts on its tool's handler once a
//! slot is free.

use std::collections::{BTreeMap, HashMap, VecDeque};

use bevy_log::warn;
use bevy_tasks::{AsyncComputeTaskPool, TaskPool};
use rig_core::completion::ToolDefinition;
use rig_core::effect::{EffectId, EffectKind, Outcome};
use rig_core::serve::adapters::ToolAdapter;
use rig_core::serve::{Dispatch, ErasedHandler};
use rig_ecs::effects::Handler;
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
    fn load() -> Self {
        let path = Home::from_env().root().join("limits.toml");
        let Ok(text) = std::fs::read_to_string(&path) else {
            return Self::default();
        };
        toml::from_str(&text).unwrap_or_else(|error| {
            warn!("{} ignored: {error}", path.display());
            Self::default()
        })
    }

    fn of(&self, tool: &str) -> Limit {
        self.0.get(tool).copied().unwrap_or_default()
    }
}

/// Whose budget a call counts in: its agent's, or `None` for a shared one.
type Budget = Option<Entity>;

/// How many calls of each tool an agent made this session (its budget).
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

/// When each budget's calls of each tool started, in the last minute.
#[derive(Resource, Default)]
struct Recent(HashMap<(Budget, String), VecDeque<Duration>>);

impl Recent {
    /// The starts of `key` in the window ending `now`, oldest first.
    fn within(&self, key: &(Budget, String), now: Duration) -> impl Iterator<Item = Duration> {
        let starts = self.0.get(key).into_iter().flatten().copied();
        starts.filter(move |start| *start + WINDOW > now)
    }
}

/// The handler of a limited tool, on its entity.
#[derive(Component)]
struct LimitedTool(Handler);

/// A call of a limited tool, with what it needs to start.
#[derive(Component)]
struct Limited {
    tool: String,
    budget: Budget,
    handler: Handler,
    kind: EffectKind,
    effect: EffectId,
    order: u64,
}

impl Limited {
    fn key(&self) -> (Budget, String) {
        (self.budget, self.tool.clone())
    }
}

/// On a limited call waiting for a slot.
#[derive(Component)]
struct Waiting;

/// On a limited call that started.
#[derive(Component)]
struct Started;

/// Starts every waiting call that has a slot.
#[derive(Event)]
struct Release;

/// Registers tools whose calls are held to [`Limits`].
pub trait AppLimitedToolsExt {
    /// [`AppToolsExt::add_tool_with`], its calls held to the tool's limits.
    fn add_limited_tool_with<T: Tool + 'static>(
        &mut self,
        tool: T,
        options: ToolOptions<'_>,
    ) -> &mut Self;
}

impl AppLimitedToolsExt for App {
    fn add_limited_tool_with<T: Tool + 'static>(
        &mut self,
        tool: T,
        options: ToolOptions<'_>,
    ) -> &mut Self {
        let world = self.world_mut();
        let mut tools = world.query::<(Entity, &mut ToolDef)>();
        if tools
            .iter(world)
            .any(|(_, def)| def.0.name.as_str() == T::NAME)
        {
            warn!("limited tool not registered: `{}` exists", T::NAME);
            return self;
        }
        let (description, parameters) = (tool.description(), tool.parameters());
        let handler = Handler(ErasedHandler::new(ToolAdapter::new(tool)));
        self.add_open_tool::<serde_json::Value, _>(T::NAME, &description, options, on_called);
        let world = self.world_mut();
        let mut tools = world.query::<(Entity, &mut ToolDef)>();
        let found = tools
            .iter_mut(world)
            .find(|(_, def)| def.0.name.as_str() == T::NAME);
        if let Some((entity, mut def)) = found {
            // The open tool's schema is any JSON; the model is told the tool's own.
            def.0 = ToolDefinition {
                parameters,
                ..def.0.clone()
            };
            world.entity_mut(entity).insert(LimitedTool(handler));
        }
        self
    }
}

#[derive(Default)]
pub struct ToolRateLimitPlugin;

impl Plugin for ToolRateLimitPlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<Limits>() {
            app.insert_resource(Limits::load());
        }
        app.init_resource::<SharedUsage>()
            .init_resource::<Recent>()
            .register_required_components::<Agent, SessionUsage>()
            .register_required_components::<TurnOf, TurnUsage>()
            .add_observer(release)
            .add_observer(release_on_output)
            .add_observer(release_on_removal)
            .add_command("limits", "Show each tool limit and its use", show_limits);
        #[cfg(feature = "tui")]
        app.add_systems(PostUpdate, status::show.in_set(rig_tui::StatusSystems));
    }
}

/// Counts a call of a limited tool, and refuses it over `per_turn` or
/// `per_session`; otherwise it waits for [`release`] to start it.
fn on_called(
    called: On<ToolCalled<serde_json::Value>>,
    tools: Query<&LimitedTool>,
    limits: Res<Limits>,
    of: Query<&CallOf>,
    mut turns: Query<&mut TurnUsage>,
    mut sessions: Query<&mut SessionUsage>,
    mut shared: ResMut<SharedUsage>,
    mut order: Local<u64>,
    mut commands: Commands,
) {
    let name = called.run.call.function.name.as_str();
    let Ok(LimitedTool(handler)) = tools.get(called.entity) else {
        return;
    };
    let limit = limits.of(name);
    let mut turn = of
        .get(called.call)
        .and_then(|&CallOf(turn)| turns.get_mut(turn));
    let mut session = sessions.get_mut(called.agent);
    let session = match (limit.shared, session.as_mut()) {
        (true, _) => Some(&mut shared.0),
        (false, Ok(session)) => Some(&mut session.0),
        (false, Err(_)) => None,
    };
    let used = |usage: Option<&HashMap<String, u32>>| {
        usage
            .and_then(|usage| usage.get(name))
            .copied()
            .unwrap_or(0)
    };
    let over = [
        (
            limit.per_turn,
            used(turn.as_deref().ok().map(|turn| &turn.0)),
            "turn",
        ),
        (limit.per_session, used(session.as_deref()), "session"),
    ]
    .into_iter()
    .find_map(|(limit, used, per)| {
        limit
            .filter(|&limit| used >= limit)
            .map(|limit| (limit, per))
    });
    if let Some((limit, per)) = over {
        let why = format!("limit reached: {limit} per {per} for `{name}`");
        let refused = ToolResult::failed(ToolExecutionError::refused(why));
        commands.entity(called.call).insert(ToolOutput(refused));
        return;
    }
    for usage in [turn.as_mut().ok().map(|turn| &mut turn.0), session]
        .into_iter()
        .flatten()
    {
        *usage.entry(name.to_owned()).or_default() += 1;
    }
    *order += 1;
    let limited = Limited {
        tool: name.to_owned(),
        budget: (!limit.shared).then_some(called.agent),
        handler: handler.clone(),
        kind: EffectKind::ToolCall {
            name: name.to_owned(),
            args: called.run.call.function.raw_arguments(),
        },
        effect: called.effect,
        order: *order,
    };
    // The status line counts down each second while the call waits.
    let waiting = (limited, Waiting, KeepAwake(Duration::from_secs(1)));
    commands.entity(called.call).insert(waiting);
    commands.trigger(Release);
}

/// A finished limited call frees its slot.
fn release_on_output(
    done: On<Add<ToolOutput>>,
    limited: Query<(), With<Limited>>,
    mut commands: Commands,
) {
    if limited.contains(done.entity) {
        commands.trigger(Release);
    }
}

/// So does one an interrupt cancels.
fn release_on_removal(_: On<Remove<Limited>>, mut commands: Commands) {
    commands.trigger(Release);
}

/// Starts each waiting call, oldest first, that its tool's `concurrent`
/// and `per_minute` allow; for one the minute holds back, comes back when
/// its oldest start leaves the window.
fn release(
    _: On<Release>,
    limits: Res<Limits>,
    waiting: Query<(Entity, &Limited), With<Waiting>>,
    running: Query<&Limited, (With<Started>, Without<ToolOutput>)>,
    mut recent: ResMut<Recent>,
    time: Res<Time>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let now = time.elapsed();
    for starts in recent.0.values_mut() {
        starts.retain(|start| *start + WINDOW > now);
    }
    recent.0.retain(|_, starts| !starts.is_empty());
    let mut busy: HashMap<(Budget, String), u32> = HashMap::new();
    for call in &running {
        *busy.entry(call.key()).or_default() += 1;
    }
    let mut queue: Vec<_> = waiting.iter().collect();
    queue.sort_by_key(|(_, call)| call.order);
    let mut next: Option<Duration> = None;
    for (entity, call) in queue {
        let limit = limits.of(&call.tool);
        let key = call.key();
        let running = busy.entry(key.clone()).or_default();
        if limit.concurrent.is_some_and(|most| *running >= most) {
            continue;
        }
        let starts = recent.0.entry(key).or_default();
        if let Some(most) = limit.per_minute
            && starts.len() >= usize::try_from(most).unwrap_or(usize::MAX)
        {
            let free = starts.front().map(|oldest| *oldest + WINDOW - now);
            next = next.into_iter().chain(free).min();
            continue;
        }
        if limit.per_minute.is_some() {
            starts.push_back(now);
        }
        *running += 1;
        let work = run(call.handler.clone(), call.kind.clone(), call.effect);
        let pool = AsyncComputeTaskPool::get_or_init(TaskPool::default);
        let task = Running::spawn_into::<ToolOutput, _>(pool, &wake, work);
        commands
            .entity(entity)
            .remove::<(Waiting, KeepAwake)>()
            .insert((Started, task));
    }
    if let Some(wait) = next {
        commands.delayed().duration(wait).trigger(Release);
    }
}

/// Runs a call on its tool's handler, as the kernel runs an ordinary tool.
async fn run(handler: Handler, kind: EffectKind, effect: EffectId) -> ToolResult {
    let dispatch = Dispatch::new(effect, kind.streams());
    let reply = handler.0.handle(kind, dispatch).await;
    match reply.into_outcome().await {
        Ok(Outcome::ToolResult { result }) => result,
        Ok(other) => ToolResult::failed(ToolExecutionError::other(format!(
            "the tool answered with a {} outcome",
            other.family()
        ))),
        Err(report) => ToolResult::failed(report.into()),
    }
}

/// `/limits`: each limit and what the agent used of it.
fn show_limits(
    In(args): In<CommandArgs>,
    limits: Res<Limits>,
    agents: Query<(&SessionUsage, Option<&ActiveTurn>)>,
    turns: Query<&TurnUsage>,
    shared: Res<SharedUsage>,
    recent: Res<Recent>,
    calls: Query<(&Limited, Has<Waiting>), Without<ToolOutput>>,
    time: Res<Time>,
    mut notices: MessageWriter<Notice>,
) {
    if limits.0.is_empty() {
        notices.write(Notice::info(
            args.agent,
            "No tool limits: RIG_HOME/limits.toml sets none.",
        ));
        return;
    }
    let (session, turn) = agents
        .get(args.agent)
        .map(|(session, active)| (Some(session), active.and_then(|a| turns.get(a.turn()).ok())))
        .unwrap_or_default();
    let mut lines = vec!["Tool limits:".to_owned()];
    for (tool, limit) in &limits.0 {
        let budget = (!limit.shared).then_some(args.agent);
        let sessions = if limit.shared {
            Some(&shared.0)
        } else {
            session.map(|s| &s.0)
        };
        let used = |usage: Option<&HashMap<String, u32>>| {
            usage
                .and_then(|usage| usage.get(tool))
                .copied()
                .unwrap_or(0)
        };
        let key = (budget, tool.clone());
        let ours = calls.iter().filter(|(call, _)| call.key() == key);
        let (waiting, running): (Vec<_>, Vec<_>) = ours.partition(|(_, waits)| *waits);
        let minute = recent.within(&key, time.elapsed()).count();
        let parts = [
            (limit.per_turn, used(turn.map(|turn| &turn.0)), "this turn"),
            (
                limit.per_minute,
                u32::try_from(minute).unwrap_or(u32::MAX),
                "this minute",
            ),
            (limit.per_session, used(sessions), "this session"),
            (
                limit.concurrent,
                u32::try_from(running.len()).unwrap_or(u32::MAX),
                "running",
            ),
        ];
        let mut said: Vec<String> = parts
            .into_iter()
            .filter_map(|(most, used, what)| most.map(|most| format!("{used}/{most} {what}")))
            .collect();
        if !waiting.is_empty() {
            said.push(format!("{} waiting", waiting.len()));
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
    const WAITING: StatusItem = StatusItem::at(Side::Left, 75, 12);

    /// "fetch: waiting 12 s" on each turn with a waiting call: the seconds
    /// until the minute frees a slot, or until a running call ends.
    pub(super) fn show(
        mut turns: Query<(Entity, &Calls, &mut StatusItems), With<TurnOf>>,
        waiting: Query<&Limited, With<Waiting>>,
        limits: Res<Limits>,
        recent: Res<Recent>,
        time: Res<Time>,
        mut shown: Local<HashMap<Entity, String>>,
    ) {
        let now = time.elapsed();
        for (turn, calls, mut items) in &mut turns {
            let mut said: Vec<String> = Vec::new();
            for call in waiting.iter_many(calls.iter()).flatten() {
                let full = limits
                    .of(&call.tool)
                    .per_minute
                    .and_then(|most| usize::try_from(most).ok())
                    .is_some_and(|most| recent.within(&call.key(), now).count() >= most);
                let oldest = recent.within(&call.key(), now).next().filter(|_| full);
                let text = match oldest {
                    Some(oldest) => {
                        let left = (oldest + WINDOW).saturating_sub(now).as_secs().max(1);
                        format!("{}: waiting {left} s", call.tool)
                    }
                    None => format!("{}: waiting for a running call", call.tool),
                };
                if !said.contains(&text) {
                    said.push(text);
                }
            }
            let text = said.join(", ");
            if shown.get(&turn) != Some(&text) {
                items.show(WAITING.says(text.clone(), Tone::Dim));
                shown.insert(turn, text);
            }
        }
        shown.retain(|turn, _| turns.contains(*turn));
    }
}
