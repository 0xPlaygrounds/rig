//! Children on another host. A child started with `host` runs on that
//! host's own `rigd`, reached through its `[remote.hosts]` transport
//! (`rig connect --stdio` over ssh or koh, authenticated by them), and is
//! proxied by this `rigd`: it is a local [`Session`](crate::server::Session) with a
//! [`RemoteSession`], whose journal mirrors the remote one, and whose
//! commands go to the remote session instead of a local worker. Everything
//! above it (the tree, waits, asks and their policy, budgets, `rig log`)
//! works on the local session as on any other.
//!
//! One proxy thread per remote child forwards its commands in order, so the
//! remote session numbers its messages as this one did; `turn.ended`'s
//! `through` is still mapped back to the local command it answers. A second
//! thread follows the remote journal. The remote session is started with
//! `unload: "turn"`, so a proxy that goes away (this `rigd` exiting) leaves
//! no remote worker running beyond the turn in progress.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use bevy_ecs::prelude::*;
use crossbeam_channel::{Receiver, Sender};
use rig_client::{Client, Target};
use rig_protocol::methods::{
    Answer, Command, Context, Send as SendMethod, Start, Started, Stop, Unload, WorkerCommand,
};
use rig_protocol::{EventRecord, Key, Seq, SessionId};

use crate::prelude::Wake;
use crate::server::Server;
use crate::server::record::{Draft, Track, publish};

/// On a local session that runs on another host: where, and the proxy
/// that carries its commands there.
#[derive(Component)]
pub struct RemoteSession {
    pub host: String,
    /// The session's id on that host, once started.
    pub remote: Option<SessionId>,
    /// The live proxy's queue; `None` once the link was let go of, while
    /// the remote child was idle, until the next command reopens it.
    commands: Option<Sender<WorkerCommand>>,
    /// Remote command numbers of the sends, with the local ones they are.
    sends: Vec<(u64, u64)>,
    /// The last remote record mirrored: a reopened link follows after it.
    seen: Seq,
}

impl RemoteSession {
    /// Whether a proxy (and its connection to the host) is live.
    pub fn linked(&self) -> bool {
        self.commands.is_some()
    }
}

/// What the proxy threads report.
enum Inbound {
    Started(Entity, SessionId),
    Sent {
        session: Entity,
        local: u64,
        remote: u64,
    },
    Event(Entity, EventRecord),
    Failed(Entity, String),
}

/// The proxies' shared state: one client per host, and what they report.
#[derive(Resource)]
pub struct Remotes {
    clients: Arc<Mutex<HashMap<String, Arc<Client>>>>,
    tx: Sender<Inbound>,
    rx: Receiver<Inbound>,
}

impl Default for Remotes {
    fn default() -> Remotes {
        let (tx, rx) = crossbeam_channel::unbounded();
        Remotes {
            clients: Arc::default(),
            tx,
            rx,
        }
    }
}

/// The remote proxy's systems: `rigd` adds it.
pub struct RemotePlugin;

impl bevy_app::Plugin for RemotePlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.init_resource::<Remotes>()
            // With the tree: mirrored records settle waits and reach
            // subscribers in the frame they arrive.
            .add_systems(
                bevy_app::Update,
                mirror.in_set(crate::server::ServerSystems::Tree),
            );
    }
}

/// The kinds the remote journal has that this one makes itself, or that
/// describe the remote worker's restarts only. Its `worker.ready` passes:
/// it confirms the remote start.
const LOCAL_KINDS: &[&str] = &["session.started", "command.accepted", "worker.exited"];

/// Starts the proxy of `session`, a child started with `host`: connects to
/// that host's `rigd`, starts the session there, and follows its journal.
/// Commands forwarded meanwhile wait in order.
pub fn open(
    world: &mut World,
    session: Entity,
    host: &str,
    req: &Start,
    grant: Option<rig_protocol::Grant>,
) {
    let Some(home) = world
        .get_resource::<Server>()
        .map(|s| s.home.root().to_path_buf())
    else {
        return;
    };
    let wake = world.get_resource::<Wake>().cloned().unwrap_or_default();
    let remotes = world.get_resource_or_init::<Remotes>();
    let (clients, tx) = (remotes.clients.clone(), remotes.tx.clone());
    let (commands, queued) = crossbeam_channel::unbounded::<WorkerCommand>();
    let start = Start {
        cwd: req.cwd.clone(),
        prompt: None,
        name: req.name.clone(),
        host: None,
        build: None,
        grant,
        policy: None,
        budget: None,
        context: Some(
            req.context
                .clone()
                .unwrap_or_else(|| Context { facts: Vec::new() }),
        ),
        unload: Some(Unload::Turn),
        view: false,
        key: Key(format!("{}:remote", req.key.0)),
    };
    let host = host.to_owned();
    world.entity_mut(session).insert(RemoteSession {
        host: host.clone(),
        remote: None,
        commands: Some(commands),
        sends: Vec::new(),
        seen: Seq(0),
    });
    spawn_proxy(
        session,
        home,
        host,
        clients,
        tx,
        wake,
        Link::Start(start),
        queued,
    );
}

/// What a proxy does first: start the remote session, or follow one it
/// started before after the last record mirrored.
enum Link {
    Start(Start),
    Resume(SessionId, Seq),
}

#[allow(clippy::too_many_arguments)]
fn spawn_proxy(
    session: Entity,
    home: std::path::PathBuf,
    host: String,
    clients: Arc<Mutex<HashMap<String, Arc<Client>>>>,
    tx: Sender<Inbound>,
    wake: Wake,
    link: Link,
    queued: Receiver<WorkerCommand>,
) {
    let report = move |inbound: Inbound| {
        tx.send(inbound).ok();
        wake.wake();
    };
    let spawned = std::thread::Builder::new()
        .name(format!("rigd remote {host}"))
        .spawn(move || proxy(session, &home, &host, &clients, link, &queued, &report));
    if let Err(error) = spawned {
        bevy_log::error!("rigd: no proxy thread: {error}");
    }
}

/// Opens the link of `session` again, after it was let go of while idle.
fn reopen(world: &mut World, session: Entity) -> Option<Sender<WorkerCommand>> {
    let home = world.get_resource::<Server>()?.home.root().to_path_buf();
    let wake = world.get_resource::<Wake>().cloned().unwrap_or_default();
    let remotes = world.get_resource::<Remotes>()?;
    let (clients, tx) = (remotes.clients.clone(), remotes.tx.clone());
    let mut remote = world.get_mut::<RemoteSession>(session)?;
    let id = remote.remote.clone()?;
    let (commands, queued) = crossbeam_channel::unbounded::<WorkerCommand>();
    remote.commands = Some(commands.clone());
    let link = Link::Resume(id, remote.seen);
    let host = remote.host.clone();
    spawn_proxy(session, home, host, clients, tx, wake, link, queued);
    Some(commands)
}

/// Lets go of the link of every remote child that is idle or done: no turn
/// running, no ask open. Its proxy thread ends, and a host none of the
/// live links use loses its connection, so the remote `rigd` can exit.
pub fn release(world: &mut World) {
    let mut remotes = world.query::<(&mut RemoteSession, &Track, &crate::server::SessionState)>();
    for (mut remote, track, state) in remotes.iter_mut(world) {
        let quiet = matches!(
            state,
            crate::server::SessionState::Idle
                | crate::server::SessionState::Stopped
                | crate::server::SessionState::Faulted
        ) && track.asks.is_empty()
            && track.turn_state() == crate::server::SessionState::Idle;
        if remote.linked() && remote.remote.is_some() && quiet {
            remote.commands = None;
        }
    }
    let mut live = world.query::<&RemoteSession>();
    let hosts: std::collections::HashSet<String> = live
        .iter(world)
        .filter(|r| r.linked())
        .map(|r| r.host.clone())
        .collect();
    if let Some(remotes) = world.get_resource::<Remotes>()
        && let Ok(mut clients) = remotes.clients.lock()
    {
        clients.retain(|host, _| hosts.contains(host));
    }
}

fn client(
    home: &std::path::Path,
    host: &str,
    clients: &Mutex<HashMap<String, Arc<Client>>>,
) -> Result<Arc<Client>, String> {
    if let Some(client) = clients.lock().ok().and_then(|c| c.get(host).cloned()) {
        return Ok(client);
    }
    let target = Target::remote(home, host).map_err(|e| e.to_string())?;
    let client = Arc::new(Client::connect(&target, "rigd").map_err(|e| e.to_string())?);
    if let Ok(mut clients) = clients.lock() {
        clients.insert(host.to_owned(), client.clone());
    }
    Ok(client)
}

/// The proxy thread of one remote child.
fn proxy(
    session: Entity,
    home: &std::path::Path,
    host: &str,
    clients: &Mutex<HashMap<String, Arc<Client>>>,
    link: Link,
    queued: &Receiver<WorkerCommand>,
    report: &(impl Fn(Inbound) + std::marker::Send + Sync + Clone + 'static),
) {
    let fail = |why: String| report(Inbound::Failed(session, format!("on {host}: {why}")));
    let client = match client(home, host, clients) {
        Ok(client) => client,
        Err(why) => return fail(why),
    };
    let (remote, after) = match link {
        Link::Start(start) => match client.call::<Start>(&start) {
            Ok(Started { session_id, .. }) => {
                report(Inbound::Started(session, session_id.clone()));
                (session_id, Seq(0))
            }
            Err(error) => return fail(error.to_string()),
        },
        Link::Resume(id, after) => (id, after),
    };
    let events = match client.subscribe(&remote, after) {
        Ok(events) => events,
        Err(error) => return fail(error.to_string()),
    };
    let follow = report.clone();
    std::thread::Builder::new()
        .name(format!("rigd remote events {}", remote.0))
        .spawn(move || {
            for record in events.flatten() {
                follow(Inbound::Event(session, record));
            }
        })
        .ok();
    for forwarded in queued.iter() {
        let local = forwarded.command_seq.0;
        let sent = match forwarded.command {
            Command::Send(mut send) => {
                send.session_id = remote.clone();
                client.call::<SendMethod>(&send).map(|sent| {
                    report(Inbound::Sent {
                        session,
                        local,
                        remote: sent.command_seq.0,
                    });
                })
            }
            Command::Answer(mut answer) => {
                answer.session_id = remote.clone();
                client.call::<Answer>(&answer).map(|_| ())
            }
            Command::Stop(mut stop) => {
                stop.session_id = remote.clone();
                client.call::<Stop>(&stop).map(|_| ())
            }
            Command::Plugin {
                method, mut params, ..
            } => {
                if let Some(id) = params.get_mut("sessionId") {
                    *id = serde_json::Value::String(remote.0.clone());
                }
                client.call_raw(&method, params).map(|_| ())
            }
            Command::Reload(_) => Ok(()),
        };
        if let Err(error) = sent {
            fail(error.to_string());
        }
    }
}

/// Sends `command` to the remote session of `session`, in order; false
/// when `session` is not remote.
pub fn forward(world: &mut World, session: Entity, command: WorkerCommand) -> bool {
    let commands = match world.get::<RemoteSession>(session) {
        Some(remote) => remote.commands.clone(),
        None => return false,
    };
    // A link let go of while the child idled opens again for its command.
    if let Some(commands) = commands.or_else(|| reopen(world, session)) {
        commands.send(command).ok();
    }
    true
}

/// Journals what the proxies report on their local sessions.
fn mirror(world: &mut World) {
    let inbound: Vec<Inbound> = match world.get_resource::<Remotes>() {
        Some(remotes) => remotes.rx.try_iter().take(4096).collect(),
        None => return,
    };
    for inbound in inbound {
        match inbound {
            Inbound::Started(session, id) => {
                if let Some(mut remote) = world.get_mut::<RemoteSession>(session) {
                    remote.remote = Some(id);
                }
            }
            Inbound::Sent {
                session,
                local,
                remote,
            } => {
                if let Some(mut proxy) = world.get_mut::<RemoteSession>(session) {
                    proxy.sends.push((remote, local));
                }
            }
            Inbound::Failed(session, why) => {
                // Its session ends failed, so a start and a wait see it.
                let fault = rig_protocol::event::SessionFaulted {
                    reason: format!("the remote session failed {why}"),
                    log: None,
                };
                publish(world, session, vec![Draft::new("session.faulted", fault)]).ok();
            }
            Inbound::Event(session, record) => {
                if let Some(mut proxy) = world.get_mut::<RemoteSession>(session) {
                    proxy.seen = proxy.seen.max(record.seq);
                }
                if let Some(draft) = mirrored(world, session, record) {
                    crate::server::workers::ingest(world, session, vec![draft]).ok();
                }
            }
        }
    }
    release(world);
}

/// The local draft of a remote record: none for the kinds this journal
/// makes itself or an ask already resolved here; `through` mapped to the
/// local command.
fn mirrored(world: &World, session: Entity, record: EventRecord) -> Option<Draft> {
    if LOCAL_KINDS.contains(&record.kind.as_str()) {
        return None;
    }
    let mut data = record.data;
    if record.kind == "ask.resolved" {
        let id = data.get("askId").and_then(serde_json::Value::as_str)?;
        let open = world
            .get::<Track>(session)
            .is_some_and(|t| t.asks.iter().any(|a| a.ask.ask_id.0 == id));
        if !open {
            return None;
        }
    }
    if record.kind == "turn.ended"
        && let Some(proxy) = world.get::<RemoteSession>(session)
        && let Some(through) = data.get("through").and_then(serde_json::Value::as_u64)
    {
        let local = proxy
            .sends
            .iter()
            .filter(|(remote, _)| *remote <= through)
            .map(|(_, local)| *local)
            .max()
            .unwrap_or(0);
        if let Some(slot) = data.get_mut("through") {
            *slot = serde_json::Value::from(local);
        }
    }
    Some(Draft {
        kind: record.kind,
        data,
        source: None,
    })
}

#[cfg(test)]
mod tests;
