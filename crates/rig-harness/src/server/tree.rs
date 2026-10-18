//! The agent tree: child sessions, their grants, policies and budgets, and
//! stopping a tree leaves first.
//!
//! A child is a session like any other, related to its parent by
//! [`ParentSession`]. The relationship is not `linked_spawn`: a stopped
//! session keeps its entity and history, and stopping is an explicit walk,
//! leaves first ([`stop_tree`], then [`ready_to_stop`] each frame).
//!
//! Limits: a tree is at most [`MAX_DEPTH`] children deep (deeper starts are
//! refused with `-32019`), and at most [`DEFAULT_RUNNING`] of its children
//! run at once (or the root grant's `children`; a parent's own `children`
//! caps its direct children too). A start over a running limit is accepted
//! and [`Queued`]; [`next_queued`] releases it when a slot frees.
//!
//! Budget: every session's [`Usage`] propagates up [`ParentSession`] and is
//! added to each [`TreeBudget`] on the way (the root's, and any a subtree was
//! started with). A budget that is spent gets [`Exhausted`]; its subtree's
//! running turns are then stopped and new commands refused ([`exhausted`]).
//! The overshoot is at most the model calls already in flight when the last
//! usage arrived.

use std::path::Path;
use std::time::Instant;

use bevy_ecs::prelude::*;
use rig_protocol::event::UsageTotals;
use rig_protocol::methods::Start;
use rig_protocol::{Answerers, Grant, Policy, RpcError, SessionId, codes};

use super::{Session, SessionState};

/// How many children deep a tree may grow.
pub const MAX_DEPTH: usize = 4;

/// How many children of one tree run at once when the root's grant says no
/// other number.
pub const DEFAULT_RUNNING: u32 = 8;

/// On a child session: its parent.
#[derive(Component, Debug)]
#[relationship(relationship_target = ChildSessions)]
pub struct ParentSession(pub Entity);

/// On a session: its children. Not linked: stopping keeps their history.
#[derive(Component, Debug)]
#[relationship_target(relationship = ParentSession)]
pub struct ChildSessions(Vec<Entity>);

impl ChildSessions {
    /// The children, in the order they were started.
    pub fn sessions(&self) -> &[Entity] {
        &self.0
    }
}

/// What a session may use. Owner sessions started without a grant have
/// none, and may use everything.
#[derive(Component, Debug, Clone)]
pub struct SessionGrant(pub Grant);

/// Who answers a session's asks.
#[derive(Component, Debug, Clone)]
pub struct SessionPolicy(pub Policy);

/// On a root session (or a subtree started with a budget): the budget and
/// what the sessions below it, itself included, have spent.
#[derive(Component, Debug, Clone, Default)]
pub struct TreeBudget {
    pub tokens: Option<u64>,
    pub cost_usd: Option<f64>,
    pub spent_tokens: u64,
    pub spent_usd: f64,
}

impl TreeBudget {
    /// Whether what is spent reaches the budget.
    pub fn spent(&self) -> bool {
        self.tokens.is_some_and(|max| self.spent_tokens >= max)
            || self.cost_usd.is_some_and(|max| self.spent_usd >= max)
    }

    pub fn totals(&self) -> UsageTotals {
        UsageTotals {
            tokens: self.spent_tokens,
            cost_usd: Some(self.spent_usd),
        }
    }
}

/// On every placed session: what it spent itself.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct Spent {
    pub tokens: u64,
    pub cost_usd: f64,
}

impl Spent {
    pub fn totals(&self) -> UsageTotals {
        UsageTotals {
            tokens: self.tokens,
            cost_usd: Some(self.cost_usd),
        }
    }
}

/// On a [`TreeBudget`]'s session once it is spent. Added once.
#[derive(Component, Debug, Clone, Copy)]
pub struct Exhausted;

/// On a child accepted over a running limit: it waits for a slot, its
/// worker not yet started.
#[derive(Component, Debug, Clone, Copy)]
pub struct Queued(pub Instant);

/// On a session in a tree being stopped. `sent` once its `Stop` went out,
/// which waits until every child of it is stopped.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct StopRequested {
    pub sent: bool,
}

/// A session's usage, rolling up to every [`TreeBudget`] above it.
#[derive(EntityEvent, Debug, Clone, Copy)]
#[entity_event(propagate = &'static ParentSession, auto_propagate)]
pub struct Usage {
    #[event_target]
    pub session: Entity,
    pub tokens: u64,
    pub cost_usd: f64,
}

/// Adds a [`Usage`] to the session's own [`Spent`] (at its target) and to
/// each [`TreeBudget`] it passes, marking a spent one [`Exhausted`].
pub fn add_usage(
    usage: On<Usage>,
    mut sessions: Query<(Option<&mut Spent>, Option<&mut TreeBudget>, Has<Exhausted>)>,
    mut commands: Commands,
) {
    let at = usage.event_target();
    let Ok((spent, budget, exhausted)) = sessions.get_mut(at) else {
        return;
    };
    if at == usage.original_event_target()
        && let Some(mut spent) = spent
    {
        spent.tokens = spent.tokens.saturating_add(usage.tokens);
        spent.cost_usd += usage.cost_usd;
    }
    if let Some(mut budget) = budget {
        budget.spent_tokens = budget.spent_tokens.saturating_add(usage.tokens);
        budget.spent_usd += usage.cost_usd;
        if budget.spent() && !exhausted {
            commands.entity(at).insert(Exhausted);
        }
    }
}

/// The server's tree systems: budget roll-up. `rigd` adds it.
pub struct TreePlugin;

impl bevy_app::Plugin for TreePlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_observer(add_usage)
            .add_observer(super::policy::route_escalation);
    }
}

/// The grant `session` runs under: its own, else everything (an owner
/// session started without one).
pub fn grant_of(world: &World, session: Entity) -> Grant {
    world
        .get::<SessionGrant>(session)
        .map(|g| g.0.clone())
        .unwrap_or_else(unrestricted)
}

/// The grant of an owner session started without one.
pub fn unrestricted() -> Grant {
    Grant {
        tools: None,
        dirs: None,
        net: true,
        children: None,
        approve: Vec::new(),
    }
}

/// The policy `session`'s asks follow: its own, else the default (any owner
/// answers a root's; a child's go to its parent).
pub fn policy_of(world: &World, session: Entity) -> Policy {
    world
        .get::<SessionPolicy>(session)
        .map(|p| p.0.clone())
        .unwrap_or_else(|| default_policy(world.get::<ParentSession>(session).is_some()))
}

/// The default policy of a root (`false`) or a child (`true`).
pub fn default_policy(child: bool) -> Policy {
    Policy {
        answerers: if child {
            Answerers::Parent
        } else {
            Answerers::AnyOwner
        },
        ..Policy::default()
    }
}

/// How many sessions are above `session`.
pub fn depth(world: &World, session: Entity) -> usize {
    let mut depth = 0;
    let mut at = session;
    while let Some(parent) = world.get::<ParentSession>(at) {
        depth += 1;
        at = parent.0;
        if depth > MAX_DEPTH * 4 {
            break; // a cycle, which `ParentSession` never makes
        }
    }
    depth
}

/// The session at the top of `session`'s tree.
pub fn root_of(world: &World, session: Entity) -> Entity {
    let mut at = session;
    for _ in 0..=MAX_DEPTH * 4 {
        match world.get::<ParentSession>(at) {
            Some(parent) => at = parent.0,
            None => break,
        }
    }
    at
}

/// The grant a child of `parent` started with `req` runs under, or the
/// refusal: `-32015` when it asks for more than the parent has (tools,
/// directories, network, children, up-front rules, or a `cwd` outside its
/// directories), `-32019` when the tree is already [`MAX_DEPTH`] deep.
/// A request without a grant inherits the parent's.
pub fn admit_child(world: &World, parent: Entity, req: &Start) -> Result<Grant, RpcError> {
    if world.get::<Session>(parent).is_none() {
        return Err(RpcError::new(
            codes::NOT_FOUND,
            "the parent session is unknown",
        ));
    }
    if depth(world, parent) + 1 > MAX_DEPTH {
        return Err(RpcError::new(
            codes::LIMIT,
            format!("helpers may nest at most {MAX_DEPTH} deep"),
        )
        .with_next("do this part yourself, or ask your parent to start the helper"));
    }
    let has = grant_of(world, parent);
    let grant = match &req.grant {
        None => has,
        Some(asked) => has.narrow(asked)?,
    };
    if let Some(dirs) = &grant.dirs
        && req.host.is_none()
        && !dirs.iter().any(|d| within(&req.cwd, &d.path))
    {
        return Err(RpcError::new(
            codes::GRANT_WIDER,
            format!("{} is outside the granted directories", req.cwd.display()),
        ));
    }
    Ok(grant)
}

/// Whether `path` is `dir` or below it, lexically, with no `..` in `path`.
pub fn within(path: &Path, dir: &Path) -> bool {
    !path
        .components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
        && path.starts_with(dir)
}

/// Gives a just-spawned session its place in the tree: its parent, grant,
/// policy, its own usage and, for a root or a start with a budget, a
/// [`TreeBudget`]. `grant` is [`admit_child`]'s for a child, else the
/// owner's request.
pub fn place(
    world: &mut World,
    session: Entity,
    parent: Option<Entity>,
    req: &Start,
    grant: Option<Grant>,
) {
    let policy = req
        .policy
        .clone()
        .unwrap_or_else(|| default_policy(parent.is_some()));
    let mut entity = world.entity_mut(session);
    entity.insert((SessionPolicy(policy), Spent::default()));
    if let Some(grant) = grant {
        entity.insert(SessionGrant(grant));
    }
    if let Some(parent) = parent {
        entity.insert(ParentSession(parent));
    }
    if parent.is_none() || req.budget.is_some() {
        let budget = req.budget.unwrap_or_default();
        entity.insert(TreeBudget {
            tokens: budget.tokens,
            cost_usd: budget.cost_usd,
            ..TreeBudget::default()
        });
    }
}

/// Whether a session in this state counts against a running limit.
pub fn is_running(state: SessionState) -> bool {
    matches!(
        state,
        SessionState::Running
            | SessionState::NeedsApproval
            | SessionState::NeedsAnswer
            | SessionState::Stopping
    )
}

fn running(world: &World, session: Entity) -> bool {
    world
        .get::<SessionState>(session)
        .is_some_and(|s| is_running(*s))
        && world.get::<Queued>(session).is_none()
}

/// Every session below `root`, depth first, each child after its own
/// children (leaves first).
pub fn descendants_leaves_first(world: &World, root: Entity) -> Vec<Entity> {
    fn walk(world: &World, at: Entity, out: &mut Vec<Entity>, depth: usize) {
        if depth > MAX_DEPTH * 4 {
            return;
        }
        if let Some(children) = world.get::<ChildSessions>(at) {
            for &child in children.sessions() {
                walk(world, child, out, depth + 1);
                out.push(child);
            }
        }
    }
    let mut out = Vec::new();
    walk(world, root, &mut out, 0);
    out
}

/// Whether `session` may run now, under its parent's `children` and its
/// tree's running limit. A root always may.
pub fn may_run(world: &World, session: Entity) -> bool {
    let Some(parent) = world.get::<ParentSession>(session).map(|p| p.0) else {
        return true;
    };
    if let Some(limit) = grant_of(world, parent).children {
        let siblings = world
            .get::<ChildSessions>(parent)
            .map(|c| c.sessions())
            .unwrap_or_default();
        let busy = siblings
            .iter()
            .filter(|&&s| s != session && running(world, s))
            .count();
        if busy >= limit as usize {
            return false;
        }
    }
    let root = root_of(world, session);
    let limit = grant_of(world, root).children.unwrap_or(DEFAULT_RUNNING);
    let busy = descendants_leaves_first(world, root)
        .into_iter()
        .filter(|&s| s != session && running(world, s))
        .count();
    busy < limit as usize
}

/// Starts `session` now, or marks it [`Queued`] when a running limit is
/// full. True when it may run now.
pub fn admit_run(world: &mut World, session: Entity) -> bool {
    if may_run(world, session) {
        return true;
    }
    if world.get::<Queued>(session).is_none() {
        world.entity_mut(session).insert(Queued(Instant::now()));
    }
    false
}

/// The queued sessions that may run now, oldest first: each loses
/// [`Queued`] and becomes `Running`, for `rigd` to start its worker and
/// forward its command.
pub fn next_queued(world: &mut World) -> Vec<Entity> {
    let mut queued: Vec<(Instant, Entity)> = world
        .query::<(Entity, &Queued)>()
        .iter(world)
        .map(|(e, q)| (q.0, e))
        .collect();
    queued.sort();
    let mut released = Vec::new();
    for (_, session) in queued {
        if !may_run_queued(world, session) {
            continue;
        }
        let mut entity = world.entity_mut(session);
        entity.remove::<Queued>();
        entity.insert(SessionState::Running);
        released.push(session);
    }
    released
}

fn may_run_queued(world: &World, session: Entity) -> bool {
    world.get::<StopRequested>(session).is_none() && may_run(world, session)
}

/// Stops `root` and every descendant, leaves first, and returns the
/// sessions told to stop in that order. Each is marked `Stopping` with a
/// [`StopRequested`]; [`ready_to_stop`] says when each one's `Stop` goes
/// out, after its children stopped. A queued child never started, so it is
/// dropped from the queue. Sessions already stopped are left as they are.
pub fn stop_tree(world: &mut World, root: Entity) -> Vec<SessionId> {
    let mut order = descendants_leaves_first(world, root);
    order.push(root);
    let mut ids = Vec::new();
    for session in order {
        let Some(id) = world.get::<Session>(session).map(|s| s.id.clone()) else {
            continue;
        };
        if world
            .get::<SessionState>(session)
            .is_some_and(|s| *s == SessionState::Stopped)
        {
            continue;
        }
        let mut entity = world.entity_mut(session);
        entity.remove::<Queued>();
        if entity.get::<StopRequested>().is_none() {
            entity.insert(StopRequested::default());
        }
        entity.insert(SessionState::Stopping);
        ids.push(id);
    }
    ids
}

/// The sessions whose `Stop` should go out now, leaves first: being
/// stopped, not yet sent, with every child stopped. Each is marked sent.
pub fn ready_to_stop(world: &mut World) -> Vec<Entity> {
    let pending: Vec<Entity> = world
        .query::<(Entity, &StopRequested)>()
        .iter(world)
        .filter(|(_, s)| !s.sent)
        .map(|(e, _)| e)
        .collect();
    let mut ready: Vec<(usize, Entity)> = pending
        .into_iter()
        .filter(|&s| {
            world
                .get::<ChildSessions>(s)
                .map(|c| c.sessions())
                .unwrap_or_default()
                .iter()
                .all(|&c| {
                    world
                        .get::<SessionState>(c)
                        .is_none_or(|st| *st == SessionState::Stopped)
                })
        })
        .map(|s| (usize::MAX - depth(world, s), s))
        .collect();
    ready.sort();
    for &(_, session) in &ready {
        if let Some(mut stop) = world.get_mut::<StopRequested>(session) {
            stop.sent = true;
        }
    }
    ready.into_iter().map(|(_, s)| s).collect()
}

/// The spent budget `session` falls under, if any: its own or an
/// ancestor's.
pub fn exhausted(world: &World, session: Entity) -> Option<Entity> {
    let mut at = session;
    for _ in 0..=MAX_DEPTH * 4 {
        if world.get::<Exhausted>(at).is_some() {
            return Some(at);
        }
        at = world.get::<ParentSession>(at)?.0;
    }
    None
}

/// The running sessions under a spent budget at `top` (itself included),
/// whose turns `rigd` stops.
pub fn running_under(world: &World, top: Entity) -> Vec<Entity> {
    let mut all = descendants_leaves_first(world, top);
    all.push(top);
    all.into_iter().filter(|&s| running(world, s)).collect()
}

#[cfg(test)]
pub(crate) mod tests;
