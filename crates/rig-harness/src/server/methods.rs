//! The `rig/*` methods `rigd` answers itself, each a one-shot system in the
//! [`channel`](crate::channel) table. `rig/hello` and `rig/session/wait` are
//! the connection's and the waits' own ([`dispatch`](super::dispatch),
//! [`wait`](super::wait)).

use bevy_app::App;
use bevy_ecs::prelude::*;
use rig_protocol::event::{
    ChildStarted, CommandAccepted, DriverChanged, SessionStarted, SessionStopped, State,
    UsageTotals,
};
use rig_protocol::methods::{
    Answer, Attach, AttachRole, Attached, Command, Detach, Disposition, Empty, EventsRead, List,
    Read, ReadEvents, ReadResult, Reload, Reloading, Resolved, Send, SendMode, Sent, SessionInfo,
    Start, Started, Stop, StopScope, Stopping, Subscribe, Subscribed, SubscriptionId, Unsubscribe,
    WorkerCommand,
};
use rig_protocol::{BuildId, ClientId, CommandSeq, Key, RpcError, Seq, SessionId, codes};

use super::dispatch::{Sessions, find, may_touch, not_found};
use super::policy::{OpenAsk, may_answer};
use super::record::{Draft, Track, publish};
use super::tree::{ChildSessions, ParentSession};
use super::{
    Client, Driver, Journal, Server, Session, SessionState, SubscribedTo, Subscription, Worker,
    workers,
};
use crate::channel::{Call, ChannelAppExt};

/// Registers every `rig/*` method `rigd` answers.
pub fn add(app: &mut App) {
    app.add_method::<Start, _, _>(start)
        .add_method::<Send, _, _>(send)
        .add_method::<Answer, _, _>(answer)
        .add_method::<Stop, _, _>(stop)
        .add_method::<List, _, _>(list)
        .add_method::<Read, _, _>(read)
        .add_method::<Subscribe, _, _>(subscribe)
        .add_method::<Unsubscribe, _, _>(unsubscribe)
        .add_method::<ReadEvents, _, _>(read_events)
        .add_method::<Attach, _, _>(attach)
        .add_method::<Detach, _, _>(detach)
        .add_method::<Reload, _, _>(reload);
}

fn internal(e: impl std::fmt::Display) -> RpcError {
    RpcError::new(codes::INTERNAL, e.to_string())
}

/// The session a checked call names: dispatch has found it and checked
/// the caller may touch it.
fn session_of(world: &World, id: &SessionId) -> Result<Entity, RpcError> {
    find(world, id).ok_or_else(|| not_found(&id.0))
}

fn client_id(world: &World, conn: Entity) -> ClientId {
    world
        .get::<Client>(conn)
        .map_or_else(|| ClientId("gone".into()), |c| c.id.clone())
}

/// Journals `command` as accepted on `session`, keeps it until the worker
/// applies it, and forwards it to the worker, starting one when the
/// session has none.
pub fn accept(
    world: &mut World,
    session: Entity,
    by: ClientId,
    key: Key,
    method: &str,
    command: Command,
) -> Result<CommandSeq, RpcError> {
    let seq = world
        .get::<Journal>(session)
        .map_or(CommandSeq(1), |j| CommandSeq(j.last_command.0 + 1));
    let text = match &command {
        Command::Send(send) => send.text.clone(),
        _ => None,
    };
    let accepted = CommandAccepted {
        command_seq: seq,
        key,
        method: method.to_owned(),
        by,
        text,
        command: Some(command.clone()),
    };
    publish(world, session, vec![Draft::new("command.accepted", accepted)])?;
    if let Some(mut journal) = world.get_mut::<Journal>(session) {
        journal.last_command = seq;
    }
    let forwarded = WorkerCommand {
        command_seq: seq,
        command,
    };
    if let Some(mut track) = world.get_mut::<Track>(session) {
        track.commands.push_back(forwarded.clone());
    }
    workers::forward(world, session, forwarded)?;
    Ok(seq)
}

fn start(In(call): In<Call<Start>>, world: &mut World) -> Result<Started, RpcError> {
    let params = call.params;
    if !params.cwd.is_absolute() || !params.cwd.is_dir() {
        return Err(RpcError::new(
            codes::INVALID_PARAMS,
            format!("cwd {} is not an absolute directory", params.cwd.display()),
        )
        .with_next("pass --cwd with an existing directory"));
    }
    let parent = match &call.caller.principal {
        super::Principal::Session(parent) => Some(session_of(world, parent)?),
        super::Principal::Owner => None,
    };
    if let Some(parent) = parent {
        refuse_spent(world, parent)?;
    }
    let grant = match parent {
        Some(parent) => Some(super::tree::admit_child(world, parent, &params)?),
        None => params.grant.clone(),
    };
    let build = match (parent, &params.build) {
        (Some(parent), _) => world.get::<Session>(parent).map(|s| s.build.clone()),
        (None, Some(build)) => Some(build.clone()),
        (None, None) => None,
    }
    .unwrap_or_else(workers::own_build);
    let home = world
        .get_resource::<Server>()
        .map(|server| server.home.clone())
        .ok_or_else(|| internal("no server"))?;
    let (id, dir) = home.new_session().map_err(internal)?;
    std::fs::write(dir.directory(), params.cwd.as_os_str().as_encoded_bytes()).ok();
    if let Some(name) = &params.name {
        dir.set_name(Some(name)).ok();
    }
    let id = SessionId(id.as_str().to_owned());
    let file = rig_memory::journal::Journal::open(dir.path(), &id).map_err(internal)?;
    let session = world
        .spawn((
            Session {
                id: id.clone(),
                cwd: params.cwd.clone(),
                host: params.host.clone(),
                build: build.clone(),
                name: params.name.clone(),
            },
            Journal {
                file: Some(file),
                ..Journal::default()
            },
            Track {
                unload: params.unload,
                ..Track::default()
            },
        ))
        .id();
    super::tree::place(world, session, parent, &params, grant.clone());
    world
        .get_resource_or_init::<Sessions>()
        .0
        .insert(id.clone(), session);
    let parent_id = parent.and_then(|p| world.get::<Session>(p).map(|s| s.id.clone()));
    let started = SessionStarted {
        cwd: params.cwd.clone(),
        name: params.name.clone(),
        host: params.host.clone(),
        parent: parent_id,
        build,
        grant,
    };
    let mut seq = publish(world, session, vec![Draft::new("session.started", started)])?;
    if let Some(parent) = parent {
        let child = ChildStarted {
            session_id: id.clone(),
            name: params.name.clone(),
        };
        publish(world, parent, vec![Draft::new("child.started", child)])?;
    }
    let mut command_seq = None;
    if let Some(prompt) = params.prompt {
        let send = Send {
            session_id: id.clone(),
            text: Some(prompt),
            content: None,
            mode: None,
            key: params.key.clone(),
        };
        let by = client_id(world, call.caller.client);
        command_seq = Some(accept(
            world,
            session,
            by,
            params.key,
            <Send as rig_protocol::Method>::NAME,
            Command::Send(send),
        )?);
        seq = Seq(world
            .get::<Journal>(session)
            .map_or(0, |j| j.next_seq.0.saturating_sub(1)));
    }
    Ok(Started {
        session_id: id,
        seq,
        command_seq,
    })
}

/// Refuses new work in a tree whose budget is spent.
fn refuse_spent(world: &World, session: Entity) -> Result<(), RpcError> {
    match super::tree::exhausted(world, session) {
        Some(top) => {
            let id = world.get::<Session>(top).map(|s| s.id.0.clone()).unwrap_or_default();
            Err(RpcError::new(codes::LIMIT, "the tree's token or cost budget is spent")
                .with_next(format!("rig stop {id}, then start again with a larger --budget-tokens")))
        }
        None => Ok(()),
    }
}

fn send(In(call): In<Call<Send>>, world: &mut World) -> Result<Sent, RpcError> {
    let session = session_of(world, &call.params.session_id)?;
    refuse_spent(world, session)?;
    if call.params.text.is_none() && call.params.content.is_none() {
        return Err(RpcError::new(codes::INVALID_PARAMS, "send needs text or content"));
    }
    let state = world
        .get::<SessionState>(session)
        .copied()
        .unwrap_or_default();
    let disposition = match (state, call.params.mode) {
        (SessionState::Idle | SessionState::Stopped | SessionState::Faulted, _) => {
            Disposition::Started
        }
        (_, Some(SendMode::Queue)) => Disposition::Queued,
        _ => Disposition::Steered,
    };
    if matches!(state, SessionState::Stopped | SessionState::Faulted)
        && let Some(mut track) = world.get_mut::<Track>(session)
    {
        // A new message revives a stopped or faulted session.
        track.stopping = false;
        track.crashes.clear();
    }
    let by = client_id(world, call.caller.client);
    let key = call.params.key.clone();
    let command_seq = accept(
        world,
        session,
        by,
        key,
        <Send as rig_protocol::Method>::NAME,
        Command::Send(call.params),
    )?;
    Ok(Sent {
        command_seq,
        disposition,
    })
}

fn answer(In(call): In<Call<Answer>>, world: &mut World) -> Result<Resolved, RpcError> {
    let session = session_of(world, &call.params.session_id)?;
    let mut asks = world.query::<(Entity, &OpenAsk)>();
    let open = asks
        .iter(world)
        .find(|(_, ask)| ask.id == call.params.ask_id && ask.session == session)
        .map(|(entity, _)| entity);
    let open_in_track = world
        .get::<Track>(session)
        .is_some_and(|t| t.asks.iter().any(|a| a.ask.ask_id == call.params.ask_id));
    if !open_in_track {
        return Err(RpcError::new(
            codes::ALREADY_RESOLVED,
            format!("ask {} is not open", call.params.ask_id.0),
        )
        .with_next(format!("rig log {} shows how it was resolved", call.params.session_id.0)));
    }
    if let Some(open) = open
        && !may_answer(world, open, &call.caller)
    {
        return Err(RpcError::new(
            codes::UNAUTHORIZED,
            "this client may not answer that ask",
        ));
    }
    let by = client_id(world, call.caller.client);
    let resolved = rig_protocol::event::AskResolved {
        ask_id: call.params.ask_id.clone(),
        by: format!("client:{}", by.0),
        option_id: call.params.option_id.clone(),
        text: call.params.text.clone(),
    };
    let key = call.params.key.clone();
    accept(
        world,
        session,
        by,
        key,
        <Answer as rig_protocol::Method>::NAME,
        Command::Answer(call.params),
    )?;
    publish(world, session, vec![Draft::new("ask.resolved", resolved)])?;
    if let Some(open) = open {
        world.despawn(open);
    }
    Ok(Resolved { resolved: true })
}

fn stop(In(call): In<Call<Stop>>, world: &mut World) -> Result<Stopping, RpcError> {
    let session = session_of(world, &call.params.session_id)?;
    let by = client_id(world, call.caller.client);
    if call.params.scope == StopScope::Turn {
        let key = call.params.key.clone();
        accept(
            world,
            session,
            by,
            key,
            <Stop as rig_protocol::Method>::NAME,
            Command::Stop(call.params.clone()),
        )?;
        return Ok(Stopping {
            stopping: vec![call.params.session_id],
        });
    }
    let _ = by;
    let stopping = super::tree::stop_tree(world, session);
    tree_frame(world);
    Ok(Stopping { stopping })
}

/// Stops one session: its worker is told to stop and exits, and the
/// session is journaled stopped once it has (at once when it has none).
pub fn stop_session(
    world: &mut World,
    session: Entity,
    by: ClientId,
    key: Key,
) -> Result<(), RpcError> {
    let id = world
        .get::<Session>(session)
        .map(|s| s.id.clone())
        .ok_or_else(|| internal("no session"))?;
    if let Some(mut track) = world.get_mut::<Track>(session) {
        track.stopping = true;
    }
    world.entity_mut(session).insert(SessionState::Stopping);
    if world.get::<Worker>(session).is_some() {
        let stop = Stop {
            session_id: id,
            scope: StopScope::Session,
            key: key.clone(),
        };
        accept(
            world,
            session,
            by,
            key,
            <Stop as rig_protocol::Method>::NAME,
            Command::Stop(stop),
        )?;
        workers::stop_after(world, session, workers::STOP_GRACE);
    } else {
        stopped(world, session)?;
    }
    Ok(())
}

/// Journals that `session` stopped.
pub fn stopped(world: &mut World, session: Entity) -> Result<(), RpcError> {
    if let Some(mut track) = world.get_mut::<Track>(session) {
        track.commands.clear();
    }
    publish(world, session, vec![Draft::new("session.stopped", SessionStopped {})])?;
    let parent = world.get::<ParentSession>(session).map(|p| p.0);
    let id = world.get::<Session>(session).map(|s| s.id.clone());
    if let (Some(parent), Some(session_id)) = (parent, id) {
        let ended = rig_protocol::event::ChildEnded {
            session_id,
            status: State::Stopped,
        };
        publish(world, parent, vec![Draft::new("child.ended", ended)])?;
    }
    Ok(())
}

/// The listing of one session.
pub fn info(world: &World, session: Entity) -> Option<SessionInfo> {
    let s = world.get::<Session>(session)?;
    let state = world.get::<SessionState>(session).copied().unwrap_or_default();
    let track = world.get::<Track>(session)?;
    let journal = world.get::<Journal>(session)?;
    let id_of = |e: Entity| world.get::<Session>(e).map(|s| s.id.clone());
    Some(SessionInfo {
        session_id: s.id.clone(),
        name: s.name.clone(),
        cwd: s.cwd.clone(),
        host: s.host.clone(),
        state: wire_state(state),
        parent: world.get::<ParentSession>(session).and_then(|p| id_of(p.0)),
        children: world
            .get::<ChildSessions>(session)
            .map(|c| c.iter().filter_map(id_of).collect())
            .unwrap_or_default(),
        last_seq: Seq(journal.next_seq.0.saturating_sub(1)),
        usage: UsageTotals {
            tokens: track.tokens,
            cost_usd: track.cost_usd,
        },
        worker: world.get::<Worker>(session).is_some(),
    })
}

pub fn wire_state(state: SessionState) -> State {
    match state {
        SessionState::Idle => State::Idle,
        SessionState::Running => State::Running,
        SessionState::NeedsApproval => State::NeedsApproval,
        SessionState::NeedsAnswer => State::NeedsAnswer,
        SessionState::Stopping => State::Stopping,
        SessionState::Stopped => State::Stopped,
        SessionState::Faulted => State::Faulted,
    }
}

fn list(In(call): In<Call<List>>, world: &mut World) -> Result<Vec<SessionInfo>, RpcError> {
    let mut sessions = world.query_filtered::<Entity, With<Session>>();
    let sessions: Vec<Entity> = sessions.iter(world).collect();
    let mut infos: Vec<SessionInfo> = sessions
        .into_iter()
        .filter(|s| may_touch(world, &call.caller.principal, *s))
        .filter_map(|s| info(world, s))
        .filter(|info| call.params.all || info.state != State::Stopped)
        .collect();
    infos.sort_by(|a, b| a.session_id.cmp(&b.session_id));
    Ok(infos)
}

fn read(In(call): In<Call<Read>>, world: &mut World) -> Result<ReadResult, RpcError> {
    let session = session_of(world, &call.params.session_id)?;
    let info = info(world, session).ok_or_else(|| not_found(&call.params.session_id.0))?;
    let track = world.get::<Track>(session);
    Ok(ReadResult {
        last_seq: info.last_seq,
        info,
        last_answer: track.and_then(|t| t.ended.as_ref()).and_then(|e| e.answer.clone()),
        open_asks: track
            .map(|t| t.asks.clone())
            .unwrap_or_default(),
    })
}

/// A subscription's id and its connection-local number.
#[derive(Component)]
pub struct SubscriptionNumber(pub SubscriptionId);

/// Where a subscription that doesn't follow ends.
#[derive(Component)]
pub struct Until(pub Seq);

fn subscribe(In(call): In<Call<Subscribe>>, world: &mut World) -> Result<Subscribed, RpcError> {
    let session = session_of(world, &call.params.session_id)?;
    let last = world
        .get::<Journal>(session)
        .map_or(0, |j| j.next_seq.0.saturating_sub(1));
    let number = world.get_resource_mut::<Server>().map_or(0, |mut server| {
        server.next_subscription += 1;
        server.next_subscription
    });
    let id = SubscriptionId(number);
    let mut sub = world.spawn((
        Subscription {
            client: call.caller.client,
            after: call.params.after_seq,
            follow: call.params.follow,
        },
        SubscribedTo(session),
        SubscriptionNumber(id),
    ));
    if !call.params.follow {
        sub.insert(Until(Seq(last)));
    }
    Ok(Subscribed {
        subscription: id,
        next_seq: Seq(call.params.after_seq.0.max(last) + 1),
    })
}

fn unsubscribe(In(call): In<Call<Unsubscribe>>, world: &mut World) -> Result<Empty, RpcError> {
    let mut subs = world.query::<(Entity, &Subscription, &SubscriptionNumber)>();
    let found = subs
        .iter(world)
        .find(|(_, sub, n)| sub.client == call.caller.client && n.0 == call.params.subscription)
        .map(|(e, _, _)| e);
    if let Some(entity) = found {
        world.despawn(entity);
    }
    Ok(Empty {})
}

/// The most records one `rig/events/read` returns.
const READ_LIMIT: u32 = 1000;

fn read_events(
    In(call): In<Call<ReadEvents>>,
    world: &mut World,
) -> Result<EventsRead, RpcError> {
    let session = session_of(world, &call.params.session_id)?;
    let journal = world.get::<Journal>(session).ok_or_else(|| internal("no journal"))?;
    let limit = call.params.limit.unwrap_or(READ_LIMIT).min(READ_LIMIT) as usize;
    let events = match &journal.file {
        Some(file) => file.read_after(call.params.after_seq, limit).map_err(internal)?,
        None => journal
            .recent
            .iter()
            .filter(|r| r.seq > call.params.after_seq)
            .take(limit)
            .cloned()
            .collect(),
    };
    let next_seq = Seq(events.last().map_or(call.params.after_seq.0, |r| r.seq.0) + 1);
    Ok(EventsRead { events, next_seq })
}

/// Journals `driver.changed` on `session`.
pub fn driver_changed(world: &mut World, session: Entity, driver: Option<ClientId>) {
    publish(world, session, vec![Draft::new("driver.changed", DriverChanged { driver })]).ok();
}

fn attach(In(call): In<Call<Attach>>, world: &mut World) -> Result<Attached, RpcError> {
    let session = session_of(world, &call.params.session_id)?;
    let current = world.get::<Driver>(session).map(|d| d.0);
    let me = call.caller.client;
    let driver = if call.params.role == AttachRole::Driver
        && (current.is_none() || call.params.takeover)
        && current != Some(me)
    {
        world.entity_mut(session).insert(Driver(me));
        let id = client_id(world, me);
        driver_changed(world, session, Some(id.clone()));
        Some(id)
    } else {
        current.map(|c| client_id(world, c))
    };
    Ok(Attached { driver })
}

fn detach(In(call): In<Call<Detach>>, world: &mut World) -> Result<Empty, RpcError> {
    let session = session_of(world, &call.params.session_id)?;
    if world.get::<Driver>(session).map(|d| d.0) == Some(call.caller.client) {
        world.entity_mut(session).remove::<Driver>();
        driver_changed(world, session, None);
    }
    Ok(Empty {})
}

fn reload(In(call): In<Call<Reload>>, world: &mut World) -> Result<Reloading, RpcError> {
    let session = session_of(world, &call.params.session_id)?;
    let by = client_id(world, call.caller.client);
    let build: BuildId = call.params.build.clone();
    let key = call.params.key.clone();
    let command_seq = accept(
        world,
        session,
        by,
        key,
        <Reload as rig_protocol::Method>::NAME,
        Command::Reload(call.params),
    )?;
    workers::begin_reload(world, session, build);
    Ok(Reloading { command_seq })
}

/// Marks a spent budget whose `budget.exhausted` is journaled.
#[derive(Component)]
pub struct Announced;

fn rigd() -> ClientId {
    ClientId("rigd".into())
}

/// The tree's part of a frame: stops whose children have stopped go out,
/// queued children whose turn came start, and a spent budget stops its
/// tree's turns once.
pub fn tree_frame(world: &mut World) -> Option<std::time::Instant> {
    for session in super::tree::ready_to_stop(world) {
        let Some(id) = world.get::<Session>(session).map(|s| s.id.0.clone()) else {
            continue;
        };
        let seq = world.get::<Journal>(session).map_or(0, |j| j.next_seq.0);
        let key = Key(format!("rigd-stop-{id}-{seq}"));
        if let Err(e) = stop_session(world, session, rigd(), key) {
            bevy_log::warn!("rigd: could not stop {id}: {}", e.message);
        }
    }
    for session in super::tree::next_queued(world) {
        if world.get::<Worker>(session).is_none()
            && let Err(e) = workers::spawn_worker(world, session)
        {
            bevy_log::warn!("rigd: a queued session did not start: {}", e.message);
        }
    }
    let mut spent = world.query_filtered::<Entity, (With<super::tree::Exhausted>, Without<Announced>)>();
    let spent: Vec<Entity> = spent.iter(world).collect();
    for top in spent {
        world.entity_mut(top).insert(Announced);
        let totals = world
            .get::<super::tree::TreeBudget>(top)
            .map(|b| b.totals())
            .unwrap_or_default();
        let exhausted = rig_protocol::event::BudgetExhausted {
            tokens: totals.tokens,
            cost_usd: totals.cost_usd,
        };
        publish(world, top, vec![Draft::new("budget.exhausted", exhausted)]).ok();
        for session in super::tree::running_under(world, top) {
            let Some(id) = world.get::<Session>(session).map(|s| s.id.clone()) else {
                continue;
            };
            let key = Key(format!("rigd-budget-{}", id.0));
            let stop = Stop {
                session_id: id,
                scope: StopScope::Turn,
                key: key.clone(),
            };
            let name = <Stop as rig_protocol::Method>::NAME;
            accept(world, session, rigd(), key, name, Command::Stop(stop)).ok();
        }
    }
    None
}

/// Denies every ask nobody answered by its deadline; returns the next
/// deadline.
pub fn expire_asks(world: &mut World) -> Option<std::time::Instant> {
    let now = std::time::Instant::now();
    let mut asks = world.query::<(Entity, &OpenAsk)>();
    let asks: Vec<(Entity, Entity, rig_protocol::AskId, std::time::Instant)> = asks
        .iter(world)
        .map(|(e, a)| (e, a.session, a.id.clone(), a.deadline))
        .collect();
    let mut next = None::<std::time::Instant>;
    for (entity, session, ask_id, deadline) in asks {
        let info = world
            .get::<Track>(session)
            .and_then(|t| t.asks.iter().find(|a| a.ask.ask_id == ask_id).cloned());
        let Some(info) = info else {
            // Resolved some other way.
            world.despawn(entity);
            continue;
        };
        if deadline > now {
            next = Some(next.map_or(deadline, |n| n.min(deadline)));
            continue;
        }
        let deny = info
            .ask
            .options
            .iter()
            .find(|o| o.kind.contains("reject") || o.kind.contains("deny"))
            .map(|o| o.option_id.clone());
        let resolved = rig_protocol::event::AskResolved {
            ask_id: ask_id.clone(),
            by: "timeout".to_owned(),
            option_id: deny.clone(),
            text: None,
        };
        let key = Key(format!("rigd-timeout-{}", ask_id.0));
        let answer = Answer {
            session_id: info.session_id.clone(),
            ask_id,
            option_id: deny,
            text: None,
            key: key.clone(),
        };
        let name = <Answer as rig_protocol::Method>::NAME;
        accept(world, session, rigd(), key, name, Command::Answer(answer)).ok();
        publish(world, session, vec![Draft::new("ask.resolved", resolved)]).ok();
        world.despawn(entity);
    }
    next
}
