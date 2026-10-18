//! Each frame: new connections, then at most [`PER_FRAME`] frames from each
//! connection in turn, each request authenticated, checked against its
//! caller's sessions and its idempotency key, and run through the
//! [`channel`](crate::channel) method table.

use std::collections::HashMap;

use bevy_ecs::prelude::*;
use crossbeam_channel::{Receiver, Sender};
use rig_protocol::methods::{ClientInfo, Hello, HelloResult, Limits, ServerInfo, Start, Wait};
use rig_protocol::{
    ClientId, Frame, Key, MAX_FRAME, Method, Notification, Peer, Request, RequestId, Response,
    RpcError, SessionId, codes,
};

use super::net::{self, Inbound};
use super::tree::ParentSession;
use super::{Client, Journal, Keyed, OutQueue, Principal, Server};
use crate::channel::{self, Caller, ChannelMethods, Reply};

/// The most frames read from one connection in one frame, so a flooding
/// client can't starve the rest.
pub const PER_FRAME: usize = 64;

/// A connection: its inbound frames, and before `rig/hello` its out-queue
/// and peer.
#[derive(Component)]
pub struct Inbox(pub Receiver<Inbound>);

/// A connection's socket, shut down when the connection is despawned.
#[derive(Component)]
pub struct SocketOf(#[allow(dead_code, reason = "held to be dropped")] pub net::Socket);

/// A connection that has not said `rig/hello` yet.
#[derive(Component)]
pub struct Unhello {
    pub out: Sender<Frame>,
    pub peer: Peer,
}

/// Every session by id.
#[derive(Resource, Default)]
pub struct Sessions(pub HashMap<SessionId, Entity>);

/// The live worker tokens, each its session's.
#[derive(Resource, Default)]
pub struct WorkerTokens(pub HashMap<String, Entity>);

/// The keys of `rig/session/start`, which have no session to keep them.
#[derive(Resource, Default)]
pub struct StartKeys(pub HashMap<Key, Keyed>);

/// Takes new connections and serves every connection's frames.
pub fn serve(world: &mut World) {
    let accepted: Vec<net::Accepted> = world
        .get_resource::<Server>()
        .map(|server| server.accepted.try_iter().collect())
        .unwrap_or_default();
    for conn in accepted {
        world.spawn((
            Inbox(conn.inbox),
            SocketOf(conn.socket),
            Unhello {
                out: conn.out,
                peer: conn.peer,
            },
        ));
    }
    let mut conns = world.query::<(Entity, &Inbox)>();
    let conns: Vec<(Entity, Receiver<Inbound>)> = conns
        .iter(world)
        .map(|(entity, inbox)| (entity, inbox.0.clone()))
        .collect();
    for (conn, inbox) in conns {
        for inbound in inbox.try_iter().take(PER_FRAME) {
            match inbound {
                Inbound::Frame(Frame::Request(request)) => self::request(world, conn, request),
                // Clients send no responses or notifications on this wire.
                Inbound::Frame(_) => {}
                Inbound::Closed => {
                    disconnect(world, conn);
                    break;
                }
            }
        }
    }
}

fn request(world: &mut World, conn: Entity, request: Request) {
    if let Some(reply) = handle(world, conn, &request) {
        respond(world, conn, request.id, reply);
    }
}

/// Queues the answer to request `id` on `conn`; a client whose queue is
/// full is disconnected, never waited for.
pub fn respond(world: &mut World, conn: Entity, id: RequestId, reply: Reply) {
    send(world, conn, Frame::Response(Response::new(id, reply)));
}

/// Queues a notification on `conn`.
pub fn notify(world: &mut World, conn: Entity, method: &str, params: serde_json::Value) {
    send(
        world,
        conn,
        Frame::Notification(Notification::new(method, params)),
    );
}

fn send(world: &mut World, conn: Entity, frame: Frame) {
    let Some(out) = out_of(world, conn) else {
        return;
    };
    match net::offer(&out, frame) {
        net::Offered::Queued => {}
        net::Offered::Full => {
            let name = world.get::<Client>(conn).map(|c| c.id.0.clone());
            bevy_log::warn!(
                "rigd: client {} reads too slowly: its queue is full; it is disconnected",
                name.unwrap_or_default()
            );
            disconnect(world, conn);
        }
        // Gone already: its reader sees the end and says so.
        net::Offered::Closed => disconnect(world, conn),
    }
}

/// The out-queue of a connection, before or after `rig/hello`.
pub fn out_of(world: &World, conn: Entity) -> Option<Sender<Frame>> {
    let entity = world.get_entity(conn).ok()?;
    entity
        .get::<Client>()
        .map(|client| client.out.0.clone())
        .or_else(|| entity.get::<Unhello>().map(|u| u.out.clone()))
}

fn handle(world: &mut World, conn: Entity, request: &Request) -> Option<Reply> {
    if request.method == Hello::NAME {
        return Some(hello(world, conn, &request.params));
    }
    let Some(principal) = world.get::<Client>(conn).map(|c| c.principal.clone()) else {
        return Some(Err(RpcError::new(
            codes::UNAUTHORIZED,
            "the first request on a connection must be rig/hello",
        )));
    };
    let caller = Caller {
        client: conn,
        principal,
    };
    if request.method.starts_with("worker/") {
        super::workers::heard(world, conn);
    }
    let params = &request.params;
    let session = match params.get("sessionId").and_then(|id| id.as_str()) {
        Some(id) => match find(world, &SessionId(id.to_owned())) {
            Some(session) if may_touch(world, &caller.principal, session) => Some(session),
            Some(_) => {
                return Some(Err(RpcError::new(
                    codes::UNAUTHORIZED,
                    format!("session {id} is not this caller's"),
                )));
            }
            None => return Some(Err(not_found(id))),
        },
        None => None,
    };
    if request.method.starts_with("worker/") && caller.principal == Principal::Owner {
        return Some(Err(RpcError::new(
            codes::UNAUTHORIZED,
            "worker methods are a worker's own",
        )));
    }
    if request.method == Wait::NAME {
        return super::wait::begin(world, caller, request.id.clone(), params.clone());
    }
    if request.method.starts_with("x/") {
        return super::workers::route_plugin(
            world,
            caller,
            session,
            request.id.clone(),
            &request.method,
            params.clone(),
        );
    }
    let key = params
        .get("key")
        .and_then(|key| key.as_str())
        .map(|key| Key(key.to_owned()));
    if let Some(key) = &key
        && let Some(earlier) = keyed(world, session, key)
    {
        return Some(
            // Null: a key restored from the journal, whose request isn't
            // kept as sent; the key alone answers it.
            if earlier.method == request.method
                && (earlier.params.is_null() || &earlier.params == params)
            {
                Ok(earlier.result)
            } else {
                Err(RpcError::new(
                    codes::KEY_CONFLICT,
                    format!("key {} was used for another command; use a new key", key.0),
                ))
            },
        );
    }
    let reply =
        channel::call(world, &request.method, caller, params.clone()).unwrap_or_else(|| {
            Err(RpcError::new(
                codes::METHOD_NOT_FOUND,
                format!("no method {}", request.method),
            ))
        });
    if let (Some(key), Ok(result)) = (key, &reply) {
        let keyed = Keyed {
            method: request.method.clone(),
            params: params.clone(),
            result: result.clone(),
        };
        let journal = session.and_then(|s| world.get_mut::<Journal>(s));
        match journal {
            Some(mut journal) => {
                journal.keys.insert(key, keyed);
            }
            None if request.method == Start::NAME => {
                world
                    .get_resource_or_init::<StartKeys>()
                    .0
                    .insert(key, keyed);
            }
            None => {}
        }
    }
    Some(reply)
}

/// What `key` answered before, on `session` or among starts.
fn keyed(world: &World, session: Option<Entity>, key: &Key) -> Option<Keyed> {
    let earlier = match session {
        Some(session) => world.get::<Journal>(session)?.keys.get(key),
        None => world.get_resource::<StartKeys>()?.0.get(key),
    };
    earlier.map(|k| Keyed {
        method: k.method.clone(),
        params: k.params.clone(),
        result: k.result.clone(),
    })
}

pub fn not_found(id: &str) -> RpcError {
    RpcError::new(codes::NOT_FOUND, format!("no session {id}"))
        .with_next("rig agents --all lists the sessions")
}

/// The session `id`.
pub fn find(world: &World, id: &SessionId) -> Option<Entity> {
    world.get_resource::<Sessions>()?.0.get(id).copied()
}

/// Whether `principal` may act on `session`: the owner on any, a worker on
/// its own session and that session's descendants.
pub fn may_touch(world: &World, principal: &Principal, session: Entity) -> bool {
    let Principal::Session(own) = principal else {
        return true;
    };
    let Some(own) = find(world, own) else {
        return false;
    };
    let mut at = Some(session);
    // A tree is at most a few levels deep; the bound only guards a cycle.
    for _ in 0..64 {
        match at {
            Some(s) if s == own => return true,
            Some(s) => at = world.get::<ParentSession>(s).map(|p| p.0),
            None => return false,
        }
    }
    false
}

fn hello(world: &mut World, conn: Entity, params: &serde_json::Value) -> Reply {
    let hello: Hello = serde_json::from_value(params.clone())
        .map_err(|e| RpcError::new(codes::INVALID_PARAMS, format!("rig/hello: {e}")))?;
    let Some(unhello) = world.get::<Unhello>(conn) else {
        return Err(RpcError::new(
            codes::INVALID_REQUEST,
            "rig/hello was said already",
        ));
    };
    let (out, peer) = (unhello.out.clone(), unhello.peer);
    // The highest version both speak; a refusal names the side to update.
    let rig = rig_protocol::upgrade::negotiate(hello.rig)?;
    let unauthorized = || {
        RpcError::new(codes::UNAUTHORIZED, "wrong token or user")
            .with_next("rig reads run/token itself; connect as the user running rig")
    };
    if !peer.uid_ok {
        return Err(unauthorized());
    }
    let owner = world
        .get_resource::<Server>()
        .is_some_and(|server| same(&server.token, &hello.token));
    let principal = if owner {
        Principal::Owner
    } else {
        let session = world
            .get_resource::<WorkerTokens>()
            .and_then(|tokens| tokens.0.get(&hello.token).copied())
            .and_then(|s| world.get::<super::Session>(s))
            .map(|s| s.id.clone())
            .ok_or_else(unauthorized)?;
        Principal::Session(session)
    };
    let ClientInfo { name, .. } = hello.client;
    let number = world.get_resource_mut::<Server>().map_or(0, |mut server| {
        server.next_client += 1;
        server.next_client
    });
    let id = ClientId(format!("c-{number}"));
    bevy_log::debug!("rigd: {name} connected as {}", id.0);
    let (mut methods, events) = world
        .get_resource::<ChannelMethods>()
        .map(|table| (table.methods(), table.events()))
        .unwrap_or_default();
    // A worker's own methods are listed to workers only.
    if principal == Principal::Owner {
        methods.retain(|m| !m.name.starts_with("worker/"));
    }
    world.entity_mut(conn).remove::<Unhello>().insert(Client {
        id: id.clone(),
        principal,
        out: OutQueue(out),
    });
    let result = HelloResult {
        rig,
        server: ServerInfo {
            version: crate::harness_protocol::VERSION.to_owned(),
            pid: std::process::id(),
        },
        client_id: id,
        methods,
        events,
        limits: Limits { frame: MAX_FRAME },
    };
    serde_json::to_value(result).map_err(|e| RpcError::new(codes::INTERNAL, e.to_string()))
}

/// Whether two tokens are equal, in time that doesn't depend on where they
/// differ.
fn same(a: &str, b: &str) -> bool {
    a.len() == b.len() && a.bytes().zip(b.bytes()).fold(0u8, |d, (x, y)| d | (x ^ y)) == 0
}

/// Forgets a connection. What it owns goes with it, as relationships say:
/// its subscriptions and waits are despawned, and its driver leases and
/// worker connection are removed from their sessions.
pub fn disconnect(world: &mut World, conn: Entity) {
    let Ok(entity) = world.get_entity_mut(conn) else {
        return;
    };
    if let Some(client) = entity.get::<Client>() {
        bevy_log::debug!("rigd: {} disconnected", client.id.0);
    }
    entity.despawn();
}
