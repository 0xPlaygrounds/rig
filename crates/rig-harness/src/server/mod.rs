//! The agent server, `rigd`: a Bevy app whose world holds every session as
//! an entity, with its clients, subscriptions, workers and agent tree beside
//! them. It journals each session's events, dispatches requests through the
//! [`channel`](crate::channel) method table, and supervises one worker
//! process per runnable session. No user plugin runs in it.

pub mod policy;
pub mod tree;

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::time::Instant;

use bevy_app::{App, AppExit, Plugin};
use bevy_ecs::prelude::*;
use rig_protocol::{BuildId, ClientId, CommandSeq, EventRecord, Frame, Key, Seq, SessionId};

/// The server's plugin: added with [`HeadlessPlugins`](crate::HeadlessPlugins)
/// alone by [`run`].
pub struct ServerPlugin;

impl Plugin for ServerPlugin {
    fn build(&self, _app: &mut App) {}
}

/// Runs `rigd` in this process until it exits.
pub fn run() -> AppExit {
    eprintln!("rig: the agent server is not built yet");
    AppExit::from_code(2)
}

/// A session `rigd` knows, live or unloaded.
#[derive(Component)]
#[require(Journal, SessionState)]
pub struct Session {
    pub id: SessionId,
    pub cwd: PathBuf,
    pub host: Option<String>,
    pub build: BuildId,
    pub name: Option<String>,
}

#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SessionState {
    #[default]
    Idle,
    Running,
    NeedsApproval,
    NeedsAnswer,
    Stopping,
    Stopped,
    Faulted,
}

/// A session's journal: the file, the recent records kept for subscribers,
/// the next positions, and the idempotency keys seen.
#[derive(Component, Default)]
pub struct Journal {
    pub file: Option<rig_memory::journal::Journal>,
    pub recent: VecDeque<EventRecord>,
    pub next_seq: Seq,
    pub next_command: CommandSeq,
    pub keys: HashMap<Key, Keyed>,
}

/// What a key was first used for, and what it answered.
pub struct Keyed {
    pub method: String,
    pub params: serde_json::Value,
    pub result: serde_json::Value,
}

/// The worker process running a session.
#[derive(Component)]
pub struct Worker {
    pub pid: u32,
    /// The worker's own connection.
    pub client: Entity,
    pub applied: CommandSeq,
    pub quiescent: bool,
    pub crashes: VecDeque<Instant>,
}

/// One connection.
#[derive(Component)]
pub struct Client {
    pub id: ClientId,
    pub principal: Principal,
    pub out: OutQueue,
}

/// A connection's bounded queue of frames to write.
pub struct OutQueue(pub crossbeam_channel::Sender<Frame>);

/// Who a connection acts for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Principal {
    /// The user: every session.
    Owner,
    /// A worker: its session and that session's descendants.
    Session(SessionId),
}

/// On a [`Subscription`]: what it follows.
#[derive(Component)]
#[relationship(relationship_target = Subscribers)]
pub struct SubscribedTo(pub Entity);

#[derive(Component)]
#[relationship_target(relationship = SubscribedTo, linked_spawn)]
pub struct Subscribers(Vec<Entity>);

/// A client's subscription to a session's events.
#[derive(Component)]
pub struct Subscription {
    pub client: Entity,
    pub after: Seq,
    pub follow: bool,
}

/// On a session: the client holding the driver lease.
#[derive(Component)]
pub struct Driver(pub Entity);
