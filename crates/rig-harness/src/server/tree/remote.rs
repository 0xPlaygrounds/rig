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
    commands: Sender<WorkerCommand>,
    /// Remote command numbers of the sends, with the local ones they are.
    sends: Vec<(u64, u64)>,
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
            .add_systems(bevy_app::Update, mirror);
    }
}

/// The kinds the remote journal has that this one makes itself, or that
/// describe the remote worker only.
const LOCAL_KINDS: &[&str] = &[
    "session.started",
    "command.accepted",
    "worker.ready",
    "worker.exited",
];

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
        commands,
        sends: Vec::new(),
    });
    let report = move |inbound: Inbound| {
        tx.send(inbound).ok();
        wake.wake();
    };
    let spawned = std::thread::Builder::new()
        .name(format!("rigd remote {host}"))
        .spawn(move || proxy(session, &home, &host, &clients, &start, &queued, &report));
    if let Err(error) = spawned {
        bevy_log::error!("rigd: no proxy thread: {error}");
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
    start: &Start,
    queued: &Receiver<WorkerCommand>,
    report: &(impl Fn(Inbound) + std::marker::Send + Sync + Clone + 'static),
) {
    let fail = |why: String| report(Inbound::Failed(session, format!("on {host}: {why}")));
    let client = match client(home, host, clients) {
        Ok(client) => client,
        Err(why) => return fail(why),
    };
    let remote = match client.call::<Start>(start) {
        Ok(Started { session_id, .. }) => session_id,
        Err(error) => return fail(error.to_string()),
    };
    report(Inbound::Started(session, remote.clone()));
    let events = match client.subscribe(&remote, Seq(0)) {
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
    match world.get::<RemoteSession>(session) {
        Some(remote) => {
            remote.commands.send(command).ok();
            true
        }
        None => false,
    }
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
                let notice = rig_protocol::event::Notice {
                    level: "error".into(),
                    text: format!("the remote session failed {why}"),
                };
                publish(world, session, vec![Draft::new("notice", notice)]).ok();
            }
            Inbound::Event(session, record) => {
                if let Some(draft) = mirrored(world, session, record) {
                    ingest(world, session, draft);
                }
            }
        }
    }
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

/// Journals a mirrored event as a worker's would be: its asks routed by
/// policy, its usage rolled up the tree.
fn ingest(world: &mut World, session: Entity, draft: Draft) {
    let ask = (draft.kind == "ask.opened")
        .then(|| crate::server::policy::AskDetails::from_data(&draft.data))
        .flatten();
    let usage = (draft.kind == "usage")
        .then(|| serde_json::from_value::<rig_protocol::event::Usage>(draft.data.clone()).ok())
        .flatten();
    if publish(world, session, vec![draft]).is_err() {
        return;
    }
    if let Some(details) = ask {
        let timeout = std::time::Duration::from_secs(
            crate::server::tree::policy_of(world, session).timeout_s,
        );
        let opened = details.opened.clone();
        let ask = world
            .spawn((
                crate::server::policy::OpenAsk {
                    id: opened.ask_id,
                    session,
                    agent: opened.agent_id,
                    deadline: std::time::Instant::now() + timeout,
                    answerers: opened.answerers,
                },
                details,
            ))
            .id();
        crate::server::policy::on_ask(world, ask);
    }
    if let Some(usage) = usage {
        world.trigger(crate::server::tree::Usage {
            session,
            tokens: usage.tokens,
            cost_usd: usage.cost_usd.unwrap_or_default(),
        });
    }
}

#[cfg(test)]
mod tests;
