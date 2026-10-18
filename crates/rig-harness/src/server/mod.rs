//! The agent server, `rigd`: a Bevy app whose world holds every session as
//! an entity, with its clients, subscriptions, workers and agent tree beside
//! them. It journals each session's events, dispatches requests through the
//! [`channel`](crate::channel) method table, and supervises one worker
//! process per runnable session. No user plugin runs in it.

mod dispatch;
mod fanout;
pub mod methods;
mod net;
pub mod policy;
pub mod record;
mod restore;
pub mod tree;
mod wait;
pub mod workers;

#[cfg(test)]
mod tests;

use std::collections::{HashMap, VecDeque};
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime};

use bevy_app::{App, AppExit, Plugin, PluginGroup, ScheduleRunnerPlugin, TaskPoolPlugin, Update};
use bevy_ecs::prelude::*;
use crossbeam_channel::{Receiver, Sender};
use rig_ecs::tasks::Wake;
use rig_protocol::{BuildId, ClientId, CommandSeq, EventRecord, Frame, Key, Seq, SessionId};

use crate::clock::WakeAt;
use crate::harness_protocol::Home;

pub use dispatch::{Sessions, find, may_touch};
pub use workers::spawn_worker;

/// How long `rigd` lives with no worker and no client (D24).
pub const IDLE_EXIT: Duration = Duration::from_secs(10);
/// How long a quiescent worker of an unattended session stays loaded.
pub const IDLE_UNLOAD: Duration = Duration::from_secs(60);

/// What a server runs with.
#[derive(Clone, Debug)]
pub struct Options {
    pub idle_exit: Duration,
    pub idle_unload: Duration,
    /// The program and arguments every worker runs instead of its build's
    /// binary, for tests.
    pub worker: Option<(PathBuf, Vec<std::ffi::OsString>)>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            idle_exit: IDLE_EXIT,
            idle_unload: IDLE_UNLOAD,
            worker: None,
        }
    }
}

/// The server's own state: its home and token, the channels its threads
/// feed, and its counters.
#[derive(Resource)]
pub struct Server {
    pub home: Home,
    pub token: String,
    pub options: Options,
    accepted: Receiver<net::Accepted>,
    exits: Sender<workers::Exited>,
    exited: Receiver<workers::Exited>,
    next_client: u64,
    next_subscription: u64,
    idle_since: Option<Instant>,
    /// The entity whose [`WakeAt`] is the server's next deadline.
    alarm: Entity,
}

/// The server's plugin: the method table, the tree, and the frame that
/// serves connections, journals, settles waits, fans out and supervises
/// workers. [`run`] adds it to a bare Bevy app; a test may add it with a
/// [`Server`] it made with [`server`].
pub struct ServerPlugin;

impl Plugin for ServerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Sessions>()
            .init_resource::<dispatch::WorkerTokens>()
            .init_resource::<dispatch::StartKeys>()
            .add_plugins(tree::TreePlugin)
            .add_systems(Update, frame);
        methods::add(app);
        workers::add(app);
    }
}

/// One frame of `rigd`.
fn frame(world: &mut World) {
    dispatch::serve(world);
    let mut next = workers::supervise(world);
    let mut soon = |at: Option<Instant>| {
        if let Some(at) = at {
            next = Some(next.map_or(at, |n: Instant| n.min(at)));
        }
    };
    soon(methods::tree_frame(world));
    soon(methods::expire_asks(world));
    soon(wait::settle(world));
    soon(fanout::fan_out(world));
    soon(idle(world));
    let Some(alarm) = world.get_resource::<Server>().map(|s| s.alarm) else {
        return;
    };
    let mut alarm = world.entity_mut(alarm);
    match next {
        Some(at) => {
            let left = at.saturating_duration_since(Instant::now());
            alarm.insert(WakeAt(SystemTime::now() + left));
        }
        None => {
            alarm.remove::<WakeAt>();
        }
    }
}

/// Exits after [`Options::idle_exit`] with no worker and no connection.
fn idle(world: &mut World) -> Option<Instant> {
    let busy = world.query::<&Worker>().iter(world).next().is_some()
        || world
            .query::<&dispatch::Inbox>()
            .iter(world)
            .next()
            .is_some();
    let mut server = world.get_resource_mut::<Server>()?;
    if busy {
        server.idle_since = None;
        return None;
    }
    let since = *server.idle_since.get_or_insert_with(Instant::now);
    let at = since + server.options.idle_exit;
    if at <= Instant::now() {
        world.write_message(AppExit::Success);
        return None;
    }
    Some(at)
}

/// Why a server could not start.
#[derive(Debug)]
pub enum StartError {
    /// Another server holds `run/server.lock`.
    Running,
    Io(std::io::Error),
}

impl From<std::io::Error> for StartError {
    fn from(e: std::io::Error) -> Self {
        StartError::Io(e)
    }
}

/// The lock `rigd` holds for its whole life.
pub struct ServerLock(#[allow(dead_code)] std::fs::File);

/// Takes `run/server.lock`, writes a fresh `run/token` (0600), and binds
/// `run/rig.sock`, in that order: a connectable socket means the token is
/// there.
#[cfg(unix)]
pub fn bind(home: &Home) -> Result<(ServerLock, String, rig_protocol::Listener), StartError> {
    use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
    let run = home.root().join("run");
    match std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(&run)
    {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(e) => return Err(e.into()),
    }
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .mode(0o600)
        .open(run.join("server.lock"))?;
    match lock.try_lock() {
        Ok(()) => {}
        Err(std::fs::TryLockError::WouldBlock) => return Err(StartError::Running),
        Err(std::fs::TryLockError::Error(e)) => return Err(e.into()),
    }
    let mut bytes = [0u8; 32];
    getrandom::fill(&mut bytes).map_err(|e| std::io::Error::other(e.to_string()))?;
    let token: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    let staged = run.join("token.new");
    std::fs::remove_file(&staged).ok();
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(&staged)?;
    file.write_all(token.as_bytes())?;
    file.sync_data()?;
    std::fs::rename(&staged, run.join("token"))?;
    let listener = rig_protocol::Listener::bind(&rig_protocol::Endpoint::for_home(home.root()))?;
    Ok((ServerLock(lock), token, listener))
}

/// The bare app `rigd` runs: Bevy's minimal plugins with the log on
/// stderr, the wake loop, and [`ServerPlugin`]. No user plugin.
pub fn server_app() -> App {
    let mut app = App::new();
    app.set_error_handler(crate::error::warn).add_plugins((
        bevy::MinimalPlugins
            .build()
            .set(TaskPoolPlugin {
                task_pool_options: crate::task_pools(),
            })
            .disable::<ScheduleRunnerPlugin>(),
        bevy_log::LogPlugin::default(),
        crate::host::runner::RunnerPlugin,
        crate::host::signals::ExitOnSignalPlugin,
    ));
    app
}

/// Gives `app` a [`Server`] for `home` serving `listener`, restores every
/// journaled session, and starts accepting.
#[cfg(unix)]
pub fn serve(
    app: &mut App,
    home: Home,
    token: String,
    listener: rig_protocol::Listener,
    options: Options,
) {
    let (accepted_tx, accepted) = crossbeam_channel::unbounded();
    let (exits, exited) = crossbeam_channel::unbounded();
    let alarm = app.world_mut().spawn(Name::new("rigd alarm")).id();
    app.insert_resource(Server {
        home,
        token,
        options,
        accepted,
        exits,
        exited,
        next_client: 0,
        next_subscription: 0,
        idle_since: None,
        alarm,
    })
    .add_plugins(ServerPlugin);
    restore::restore(app.world_mut());
    let wake = app
        .world()
        .get_resource::<Wake>()
        .cloned()
        .unwrap_or_default();
    net::accept(listener, accepted_tx, wake);
}

/// Runs `rigd` in this process until it exits: exits 0 at once when
/// another server runs for this home.
pub fn run() -> AppExit {
    let home = Home::from_env();
    #[cfg(unix)]
    {
        let (lock, token, listener) = match bind(&home) {
            Ok(bound) => bound,
            Err(StartError::Running) => return AppExit::Success,
            Err(StartError::Io(e)) => {
                eprintln!("rig: the agent server could not start: {e}");
                return AppExit::from_code(1);
            }
        };
        let mut app = server_app();
        serve(&mut app, home.clone(), token, listener, Options::default());
        let exit = app.run();
        std::fs::remove_file(rig_protocol::Endpoint::for_home(home.root()).path).ok();
        drop(lock);
        exit
    }
    #[cfg(not(unix))]
    {
        let _ = home;
        eprintln!("rig: the agent server needs the named-pipe transport on this platform");
        AppExit::from_code(1)
    }
}

/// A session `rigd` knows, live or unloaded.
#[derive(Component)]
#[require(Journal, SessionState, record::Track)]
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
/// the next seq, the last accepted command, and the idempotency keys seen.
#[derive(Component, Default)]
pub struct Journal {
    pub file: Option<rig_memory::journal::Journal>,
    pub recent: VecDeque<EventRecord>,
    pub next_seq: Seq,
    /// The last accepted command.
    pub last_command: CommandSeq,
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
