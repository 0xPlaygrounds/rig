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

// They drive processes with Unix signals; Windows runs the conformance
// suite against the same server.
#[cfg(all(test, unix))]
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

/// How long a busy worker may stay silent (§1.5, a hang).
pub const WATCHDOG: Duration = Duration::from_secs(60);

/// What a server runs with.
#[derive(Clone, Debug)]
pub struct Options {
    pub idle_exit: Duration,
    pub idle_unload: Duration,
    /// How long a worker with work to do may go without a word before it
    /// is killed and restarted as hung.
    pub watchdog: Duration,
    /// Crashes within a minute that fault a session.
    pub crash_limit: usize,
    /// The program and arguments every worker runs instead of its build's
    /// binary, for tests.
    pub worker: Option<(PathBuf, Vec<std::ffi::OsString>)>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            idle_exit: IDLE_EXIT,
            idle_unload: IDLE_UNLOAD,
            watchdog: WATCHDOG,
            crash_limit: 3,
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
}

/// In `rigd`: a channel method that panics answers `-32603` and stays
/// registered, where a worker's panic ends the worker.
#[derive(Resource)]
pub struct CatchPanics;

/// The server's plugin: the method table, the tree, and the frame that
/// serves connections, journals, settles waits, fans out and supervises
/// workers. [`run`] adds it to a bare Bevy app; a test may add it with a
/// [`Server`] it made with [`serve`].
pub struct ServerPlugin;

impl Plugin for ServerPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(CatchPanics)
            .init_resource::<Sessions>()
            .init_resource::<dispatch::WorkerTokens>()
            .init_resource::<dispatch::StartKeys>()
            .add_plugins(tree::TreePlugin)
            .add_observer(exit)
            .add_observer(driver_gone)
            .configure_sets(
                Update,
                (
                    ServerSystems::Serve,
                    ServerSystems::Supervise,
                    ServerSystems::Tree,
                    ServerSystems::Asks,
                    ServerSystems::Waits,
                    ServerSystems::FanOut,
                    ServerSystems::Idle,
                )
                    .chain(),
            )
            .add_systems(
                Update,
                (
                    step(serve_step).in_set(ServerSystems::Serve),
                    step(workers::supervise).in_set(ServerSystems::Supervise),
                    step(methods::tree_frame).in_set(ServerSystems::Tree),
                    step(methods::expire_asks).in_set(ServerSystems::Asks),
                    step(wait::settle).in_set(ServerSystems::Waits),
                    step(fanout::fan_out).in_set(ServerSystems::FanOut),
                    step(idle).in_set(ServerSystems::Idle),
                ),
            )
            .add_systems(
                bevy_app::Last,
                unlink.run_if(bevy_ecs::schedule::common_conditions::on_message::<AppExit>),
            );
        methods::add(app);
        workers::add(app);
    }
}

/// A signal's [`Exit`](crate::host::exit::Exit): `rigd` has no effects to
/// settle, so it exits at once; its workers see the socket close.
fn exit(exit: On<crate::host::exit::Exit>, mut exits: MessageWriter<AppExit>) {
    exits.write(exit.0.exit());
}

/// Removes the socket as `rigd` exits, so the next server binds at once
/// and no client connects to one that is going.
fn unlink(server: Option<Res<Server>>) {
    if let Some(server) = server {
        std::fs::remove_file(rig_protocol::Endpoint::for_home(server.home.root()).path).ok();
    }
}

/// A driver's lease ended (detach, or its client went): journaled.
fn driver_gone(gone: On<Remove<Driver>>, mut commands: Commands) {
    let session = gone.entity;
    commands.queue(move |world: &mut World| methods::driver_changed(world, session, None));
}

/// rigd's frame, in order: each step its own system, so one that fails
/// skips none of the others, and each with its own deadline, a
/// [`WakeAt`] on an alarm entity of its own that the loop wakes for.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ServerSystems {
    /// New connections and their requests.
    Serve,
    /// Worker exits, the watchdog, unloads, kills and reloads.
    Supervise,
    /// The tree's stops, queued children and spent budgets.
    Tree,
    /// Asks past their deadline.
    Asks,
    /// Waits that can be answered (after every step that journals).
    Waits,
    /// Records to subscribers.
    FanOut,
    /// The idle exit.
    Idle,
}

/// A step's alarm: the entity whose [`WakeAt`] is its next deadline.
#[derive(Component)]
struct Alarm;

/// Runs a step that returns its next deadline, and arms its alarm.
fn step(run: fn(&mut World) -> Option<Instant>) -> impl FnMut(&mut World, Local<Option<Entity>>) {
    move |world: &mut World, mut alarm: Local<Option<Entity>>| {
        let at = run(world);
        let entity =
            *alarm.get_or_insert_with(|| world.spawn((Name::new("rigd alarm"), Alarm)).id());
        let Ok(mut entity) = world.get_entity_mut(entity) else {
            return;
        };
        match at {
            Some(at) => {
                let left = at.saturating_duration_since(Instant::now());
                entity.insert(WakeAt(SystemTime::now() + left));
            }
            None => {
                entity.remove::<WakeAt>();
            }
        }
    }
}

fn serve_step(world: &mut World) -> Option<Instant> {
    dispatch::serve(world);
    None
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
/// there. On Windows the files take the rig home's own access rights and
/// the pipe has an owner-only DACL.
pub fn bind(home: &Home) -> Result<(ServerLock, String, rig_protocol::Listener), StartError> {
    #[cfg(unix)]
    use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
    let run = home.root().join("run");
    let mut dir = std::fs::DirBuilder::new();
    dir.recursive(true);
    #[cfg(unix)]
    dir.mode(0o700);
    match dir.create(&run) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(e) => return Err(e.into()),
    }
    let owner_only = || {
        let mut open = std::fs::OpenOptions::new();
        open.write(true);
        #[cfg(unix)]
        open.mode(0o600);
        open
    };
    let lock = owner_only()
        .create(true)
        .truncate(false)
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
    let mut file = owner_only().create_new(true).open(&staged)?;
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
    app.set_error_handler(crate::error::warn)
        .add_plugins((
            bevy::MinimalPlugins
                .build()
                .set(TaskPoolPlugin {
                    task_pool_options: crate::task_pools(),
                })
                .disable::<ScheduleRunnerPlugin>(),
            bevy_log::LogPlugin::default(),
            crate::host::runner::RunnerPlugin,
            crate::host::signals::ExitOnSignalPlugin,
        ))
        .insert_resource(crate::host::runner::FrameFloor(Duration::from_millis(1)));
    app
}

/// Gives `app` a [`Server`] for `home` serving `listener`, restores every
/// journaled session, and starts accepting.
pub fn serve(
    app: &mut App,
    home: Home,
    token: String,
    listener: rig_protocol::Listener,
    options: Options,
) {
    let (accepted_tx, accepted) = crossbeam_channel::unbounded();
    let (exits, exited) = crossbeam_channel::unbounded();
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
    let (lock, token, listener) = match bind(&home) {
        Ok(bound) => bound,
        Err(StartError::Running) => return AppExit::Success,
        Err(StartError::Io(e)) => {
            eprintln!("rig: the agent server could not start: {e}");
            return AppExit::from_code(1);
        }
    };
    let mut app = server_app();
    serve(&mut app, home, token, listener, Options::default());
    let exit = app.run();
    drop(lock);
    exit
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
    /// The last accepted message (a send, or a start's prompt): what a
    /// wait waits for unless it names a command. A stop, an answer or a
    /// reload only changes the turn a message started.
    pub last_message: CommandSeq,
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
#[relationship_target(relationship = SubscribedTo)]
pub struct Subscribers(Vec<Entity>);

/// A client's subscription to a session's events: on an entity of its own,
/// [`SubscribedTo`] its session and [`OwnedBy`] its client.
#[derive(Component)]
pub struct Subscription {
    pub after: Seq,
    pub follow: bool,
}

/// On a subscription or a pending wait: the client it belongs to. The
/// client's going despawns it.
#[derive(Component)]
#[relationship(relationship_target = Owns)]
pub struct OwnedBy(pub Entity);

/// On a client: its subscriptions and pending waits.
#[derive(Component)]
#[relationship_target(relationship = OwnedBy, linked_spawn)]
pub struct Owns(Vec<Entity>);

/// On a session: the client holding the driver lease. It goes with the
/// client, and `driver.changed` says so.
#[derive(Component)]
#[relationship(relationship_target = Drives)]
pub struct Driver(pub Entity);

/// On a client: the sessions it drives.
#[derive(Component)]
#[relationship_target(relationship = Driver)]
pub struct Drives(Vec<Entity>);

/// On a session: its worker's own connection, once it said `worker/hello`.
/// It goes with the connection.
#[derive(Component)]
#[relationship(relationship_target = ServesWorker)]
pub struct WorkerConn(pub Entity);

/// On a worker's connection: the session it is the worker of.
#[derive(Component)]
#[relationship_target(relationship = WorkerConn)]
pub struct ServesWorker(Vec<Entity>);
