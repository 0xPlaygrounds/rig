//! Worker supervision: a session's worker is the agent binary of its build
//! in the `worker` role, started on the first command to a session that
//! has none. It says `worker/hello`, gets the commands it has not applied,
//! and reports events, applied commands and quiescence; `rigd` restarts it
//! after a crash (three in a minute fault the session), stops it when the
//! session stops, unloads it when idle, and swaps its build on reload.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::{Child, Command as Process, ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bevy_app::App;
use bevy_ecs::prelude::*;
use crossbeam_channel::Sender;
use rig_ecs::tasks::Wake;
use rig_protocol::event::{
    ReloadDone, ReloadFailed, ReloadStarted, SessionFaulted, WorkerExited, WorkerReady,
};
use rig_protocol::methods::{
    Command, Empty, WORKER_COMMAND, WorkerApplied, WorkerCommand, WorkerEvents, WorkerHello,
    WorkerQuiescent, WorkerReply, WorkerWelcome,
};
use rig_protocol::{
    BuildId, CommandSeq, Outcome, PROTOCOL, RequestId, RpcError, SessionId, Side, codes,
};

use super::dispatch::{WorkerTokens, find, notify, respond};
use super::policy::OpenAsk;
use super::record::{Draft, Track, publish};
use super::{Principal, Server, Session, SessionState, Worker};
use crate::channel::{Call, ChannelAppExt, Reply};
use crate::harness_protocol::env;

/// The JSON of a restricted session's grant, in its worker's environment.
pub const GRANT_ENV: &str = "RIG_GRANT";
/// Set to 1 for a session nobody will answer during its run: started to
/// unload after its turn, with no driver (M-5).
pub const UNATTENDED_ENV: &str = "RIG_UNATTENDED";
/// How long a stopping or unloading worker has to exit before it is killed.
pub const STOP_GRACE: Duration = Duration::from_secs(10);
/// The window [`Options::crash_limit`](super::Options) crashes are counted in.
const CRASH_WINDOW: Duration = Duration::from_secs(60);
/// How long a new build has to say hello before a reload rolls back.
const RELOAD_HELLO: Duration = Duration::from_secs(30);

/// The worker's process, shared with the thread that waits for it.
#[derive(Component)]
pub struct WorkerProcess {
    /// Asked to exit (unload, stop, reload) at this time; killed after
    /// [`STOP_GRACE`].
    pub asked: Option<Instant>,
    pub started: Instant,
    /// Plugin calls waiting for the worker's hello.
    pub held: Vec<WorkerCommand>,
    pub hello: bool,
    /// Since when the worker has been idle with nothing waiting on it.
    pub quiet_since: Option<Instant>,
    /// When the worker last said anything: the watchdog's clock.
    pub heard: Instant,
}

/// A reload in progress: the build being replaced.
#[derive(Component)]
pub struct Reloading {
    pub to: BuildId,
    pub from: BuildId,
    /// Set once the new build's worker is spawned.
    pub spawned: Option<Instant>,
}

/// A worker process that exited.
pub struct Exited {
    pub session: Entity,
    pub pid: u32,
    pub status: Option<ExitStatus>,
}

/// Pending `x/*` calls a worker answers, by reply number.
#[derive(Resource, Default)]
pub struct PluginCalls {
    next: u64,
    pending: HashMap<u64, (Entity, RequestId)>,
}

/// Registers the `worker/*` methods.
pub fn add(app: &mut App) {
    app.init_resource::<PluginCalls>()
        .add_method::<WorkerHello, _, _>(hello)
        .add_method::<WorkerEvents, _, _>(events)
        .add_method::<WorkerApplied, _, _>(applied)
        .add_method::<WorkerQuiescent, _, _>(quiescent)
        .add_method::<WorkerReply, _, _>(reply);
}

fn internal(e: impl std::fmt::Display) -> RpcError {
    RpcError::new(codes::INTERNAL, e.to_string())
}

/// The build this server runs: the `<BuildId>` of
/// `bin/builds/<BuildId>/rig-agent`, or `dev` for a binary elsewhere.
pub fn own_build() -> BuildId {
    let exe = std::env::current_exe().ok();
    let id = exe.as_ref().and_then(|exe| {
        let dir = exe.parent()?;
        let builds = dir.parent()?;
        (builds.file_name()? == "builds")
            .then(|| dir.file_name()?.to_str().map(str::to_owned))
            .flatten()
    });
    BuildId(id.unwrap_or_else(|| "dev".to_owned()))
}

/// The binary of `build`: its immutable artifact, else this server's own.
fn binary(home: &crate::harness_protocol::Home, build: &BuildId) -> Option<PathBuf> {
    let artifact = home.build(&build.0);
    if artifact.is_file() {
        return Some(artifact);
    }
    std::env::current_exe().ok()
}

fn token() -> Result<String, RpcError> {
    let mut bytes = [0u8; 32];
    getrandom::fill(&mut bytes).map_err(internal)?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

/// The session `caller` is the worker of: the connection that said
/// `worker/hello` for it. Another connection with the worker's token is a
/// plain client of the session.
fn own_session(world: &World, caller: &crate::channel::Caller) -> Result<Entity, RpcError> {
    let session = principal_session(world, caller)?;
    match world.get::<Worker>(session) {
        Some(worker) if worker.client == caller.client => Ok(session),
        _ => Err(RpcError::new(
            codes::UNAUTHORIZED,
            "worker methods come from the worker's own connection, after worker/hello",
        )),
    }
}

fn principal_session(world: &World, caller: &crate::channel::Caller) -> Result<Entity, RpcError> {
    let Principal::Session(id) = &caller.principal else {
        return Err(RpcError::new(codes::UNAUTHORIZED, "not a worker"));
    };
    find(world, id).ok_or_else(|| RpcError::new(codes::NOT_FOUND, "no such session"))
}

/// Starts `session`'s worker.
pub fn spawn_worker(world: &mut World, session: Entity) -> Result<(), RpcError> {
    let (home, exits, wake, program) = {
        let server = world
            .get_resource::<Server>()
            .ok_or_else(|| internal("no server"))?;
        (
            server.home.clone(),
            server.exits.clone(),
            world.get_resource::<Wake>().cloned().unwrap_or_default(),
            server.options.worker.clone(),
        )
    };
    let s = world
        .get::<Session>(session)
        .ok_or_else(|| internal("no session"))?;
    let (id, cwd, build) = (s.id.clone(), s.cwd.clone(), s.build.clone());
    let (exe, args) = match program {
        Some(program) => program,
        None => (
            binary(&home, &build).ok_or_else(|| internal("no agent binary"))?,
            Vec::new(),
        ),
    };
    let token = token()?;
    let dir =
        id.0.parse::<crate::harness_protocol::SessionId>()
            .map(|id| home.session(&id))
            .map_err(internal)?;
    let log_path = dir.path().join("worker.log");
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(internal)?;
    let mut cmd = Process::new(&exe);
    cmd.args(args)
        .current_dir(&cwd)
        .env(env::ROLE, "worker")
        .env(env::HOME, home.root())
        .env(env::SESSION, &id.0)
        .env(env::WORKER_TOKEN, &token)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(log);
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(&mut cmd, 0);
    #[cfg(target_os = "linux")]
    die_with_server(&mut cmd);
    let unattended = world
        .get::<Track>(session)
        .is_some_and(|t| t.unload == Some(rig_protocol::methods::Unload::Turn))
        && world.get::<super::Driver>(session).is_none();
    if unattended {
        cmd.env(UNATTENDED_ENV, "1");
    }
    if let Some(grant) = world.get::<super::tree::SessionGrant>(session) {
        cmd.env(
            GRANT_ENV,
            serde_json::to_string(&grant.0).map_err(internal)?,
        );
        crate::sandbox::wrap(&mut cmd, &grant.0).map_err(|e| {
            RpcError::new(codes::SANDBOX_UNAVAILABLE, e.to_string())
                .with_next("start the child without the narrower grant, or on Linux or macOS")
        })?;
    }
    let child = cmd.spawn().map_err(|e| {
        RpcError::new(
            codes::WORKER_FAILED,
            format!("could not start {}: {e}", exe.display()),
        )
        .with_next("rig build, then retry")
    })?;
    let pid = child.id();
    let child = Arc::new(Mutex::new(child));
    watch(child, session, pid, exits, wake);
    world
        .get_resource_or_init::<WorkerTokens>()
        .0
        .insert(token, session);
    let track = world.get::<Track>(session);
    let crashes = track.map(|t| t.crashes.clone()).unwrap_or_default();
    let applied = track.map_or(CommandSeq(0), |t| t.applied);
    world.entity_mut(session).insert((
        Worker {
            pid,
            client: Entity::PLACEHOLDER,
            applied,
            quiescent: false,
            crashes,
        },
        WorkerProcess {
            asked: None,
            started: Instant::now(),
            held: Vec::new(),
            hello: false,
            quiet_since: None,
            heard: Instant::now(),
        },
    ));
    bevy_log::debug!("rigd: worker {pid} for {}", id.0);
    Ok(())
}

/// Has the kernel kill the worker when `rigd` dies, however busy or
/// stopped it is (§1.1): it can't wait to read the end of its connection.
/// `rigd` spawns workers from its main thread, the one the parent-death
/// signal is tied to, which lives as long as `rigd`.
#[cfg(target_os = "linux")]
fn die_with_server(cmd: &mut Process) {
    let server = std::process::id();
    let hook = move || {
        // SAFETY: runs in the child between fork and exec, where only
        // async-signal-safe calls are allowed: prctl(2) and getppid(2) are
        // plain syscalls on integers, touching no memory and taking no lock.
        let (set, parent) = unsafe {
            (
                libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL, 0, 0, 0),
                libc::getppid(),
            )
        };
        // The server died before the signal was armed: never run orphaned.
        if set != 0 || u32::try_from(parent).ok() != Some(server) {
            return Err(std::io::Error::from_raw_os_error(libc::ESRCH));
        }
        Ok(())
    };
    // SAFETY: the hook only makes the async-signal-safe calls above; its
    // error is a raw OS error, which allocates nothing.
    unsafe {
        std::os::unix::process::CommandExt::pre_exec(cmd, hook);
    }
}

/// Waits for the worker on a thread of its own, polling so the world can
/// still kill it.
fn watch(child: Arc<Mutex<Child>>, session: Entity, pid: u32, exits: Sender<Exited>, wake: Wake) {
    std::thread::Builder::new()
        .name("rigd worker".into())
        .spawn(move || {
            loop {
                let status = match child.lock() {
                    Ok(mut child) => child.try_wait(),
                    Err(_) => return,
                };
                match status {
                    Ok(Some(status)) => {
                        exits
                            .send(Exited {
                                session,
                                pid,
                                status: Some(status),
                            })
                            .ok();
                        wake.wake();
                        return;
                    }
                    Ok(None) => std::thread::sleep(Duration::from_millis(20)),
                    Err(_) => {
                        exits
                            .send(Exited {
                                session,
                                pid,
                                status: None,
                            })
                            .ok();
                        wake.wake();
                        return;
                    }
                }
            }
        })
        .ok();
}

/// Sends `pid` a signal: `TERM` asks it to save and exit, `KILL` ends it
/// and its process group.
#[cfg(unix)]
fn signal(pid: u32, kill: bool) {
    let Ok(pid) = libc::pid_t::try_from(pid) else {
        return;
    };
    // SAFETY: kill(2) takes plain integers and touches no memory of ours;
    // a stale pid is at worst ESRCH. The group is the worker's own, made
    // by `process_group(0)` at spawn.
    unsafe {
        if kill {
            libc::kill(-pid, libc::SIGKILL);
        } else {
            libc::kill(pid, libc::SIGTERM);
        }
    }
}

#[cfg(not(unix))]
fn signal(_pid: u32, _kill: bool) {}

/// Notes that the worker on `conn` spoke, for the watchdog.
pub fn heard(world: &mut World, conn: Entity) {
    let mut workers = world.query::<(&Worker, &mut WorkerProcess)>();
    for (worker, mut process) in workers.iter_mut(world) {
        if worker.client == conn {
            process.heard = Instant::now();
        }
    }
}

/// Asks `session`'s worker to exit, and kills it if it hasn't after `grace`.
pub fn stop_after(world: &mut World, session: Entity, grace: Duration) {
    if let Some(mut process) = world.get_mut::<WorkerProcess>(session)
        && process.asked.is_none()
    {
        process.asked = Some(Instant::now() + grace);
    }
}

/// Asks the worker to save and exit now (unload).
pub fn unload(world: &mut World, session: Entity) {
    let Some(pid) = world.get::<Worker>(session).map(|w| w.pid) else {
        return;
    };
    stop_after(world, session, STOP_GRACE);
    signal(pid, false);
}

/// Forwards `command` to `session`'s worker, starting one when it has none;
/// a worker that has not said hello gets it in its replay.
pub fn forward(world: &mut World, session: Entity, command: WorkerCommand) -> Result<(), RpcError> {
    // A remote session's commands go to its host, never to a local worker.
    if super::tree::remote::forward(world, session, command.clone()) {
        return Ok(());
    }
    match world.get::<Worker>(session).map(|w| w.client) {
        Some(client) if client != Entity::PLACEHOLDER => {
            notify(
                world,
                client,
                WORKER_COMMAND,
                serde_json::to_value(command).map_err(internal)?,
            );
            Ok(())
        }
        Some(_) => Ok(()),
        None if super::tree::admit_run(world, session) => spawn_worker(world, session),
        // Queued under a running limit: started by the tree when its turn comes.
        None => Ok(()),
    }
}

/// Sends `session`'s worker a plugin call nobody waits on, such as the
/// budget gate, held until the worker says hello.
pub fn tell(world: &mut World, session: Entity, method: &str, params: serde_json::Value) {
    let reply = {
        let mut calls = world.get_resource_or_init::<PluginCalls>();
        calls.next += 1;
        calls.next
    };
    let command = WorkerCommand {
        command_seq: CommandSeq(0),
        command: Command::Plugin {
            method: method.to_owned(),
            params,
            reply,
            // rigd's own calls carry the owner's authority.
            owner: true,
        },
    };
    if world.get::<WorkerProcess>(session).is_some_and(|p| p.hello) {
        forward(world, session, command).ok();
    } else if let Some(mut process) = world.get_mut::<WorkerProcess>(session) {
        process.held.push(command);
    }
}

/// Routes an `x/<plugin>/<verb>` call to the worker of the session it
/// names, answered when the worker replies.
pub fn route_plugin(
    world: &mut World,
    caller: crate::channel::Caller,
    session: Option<Entity>,
    id: RequestId,
    method: &str,
    params: serde_json::Value,
) -> Option<Reply> {
    let Some(session) = session else {
        return Some(Err(RpcError::new(
            codes::INVALID_PARAMS,
            format!("{method} needs a sessionId"),
        )));
    };
    let reply = {
        let mut calls = world.get_resource_or_init::<PluginCalls>();
        calls.next += 1;
        let reply = calls.next;
        calls.pending.insert(reply, (caller.client, id));
        reply
    };
    let command = WorkerCommand {
        command_seq: CommandSeq(0),
        command: Command::Plugin {
            method: method.to_owned(),
            params,
            reply,
            owner: matches!(caller.principal, Principal::Owner),
        },
    };
    let ready = world.get::<WorkerProcess>(session).is_some_and(|p| p.hello);
    if ready {
        return forward(world, session, command).err().map(Err);
    }
    if world.get::<Worker>(session).is_none()
        && let Err(e) = spawn_worker(world, session)
    {
        return Some(Err(e));
    }
    if let Some(mut process) = world.get_mut::<WorkerProcess>(session) {
        process.held.push(command);
    }
    None
}

fn hello(In(call): In<Call<WorkerHello>>, world: &mut World) -> Result<WorkerWelcome, RpcError> {
    let session = principal_session(world, &call.caller)?;
    match world.get::<Worker>(session) {
        Some(worker) if worker.client == Entity::PLACEHOLDER => {}
        _ => {
            return Err(RpcError::new(
                codes::UNAUTHORIZED,
                "this session's worker said hello already",
            ));
        }
    }
    if call.params.rig != PROTOCOL {
        let mut error = RpcError::version(Some(call.params.rig)).with_next("rig build");
        if let Some(data) = error.data.as_mut() {
            data.update = Some(Side::Worker);
        }
        return Err(error);
    }
    let applied = call.params.applied;
    let mut entity = world.entity_mut(session);
    if let Some(mut worker) = entity.get_mut::<Worker>() {
        worker.client = call.caller.client;
        worker.applied = applied;
    }
    let pid = entity.get::<Worker>().map_or(0, |w| w.pid);
    let held = entity
        .get_mut::<WorkerProcess>()
        .map(|mut p| {
            p.hello = true;
            std::mem::take(&mut p.held)
        })
        .unwrap_or_default();
    let mut track = entity
        .get_mut::<Track>()
        .ok_or_else(|| internal("no session"))?;
    track.applied = track.applied.max(applied);
    let through = track.applied;
    track.commands.retain(|c| c.command_seq > through);
    let mut replay: Vec<WorkerCommand> = track.commands.iter().cloned().collect();
    replay.extend(held);
    let known_through = track.known.clone();
    let mut drafts = vec![Draft::new(
        "worker.ready",
        WorkerReady {
            pid,
            build: call.params.build.clone(),
        },
    )];
    if let Some(reloading) = entity.take::<Reloading>() {
        if let Some(mut s) = entity.get_mut::<Session>() {
            s.build = reloading.to.clone();
        }
        drafts.push(Draft::new(
            "reload.done",
            ReloadDone {
                build: reloading.to,
            },
        ));
    }
    publish(world, session, drafts)?;
    Ok(WorkerWelcome {
        known_through,
        replay,
    })
}

fn events(In(call): In<Call<WorkerEvents>>, world: &mut World) -> Result<Empty, RpcError> {
    let session = own_session(world, &call.caller)?;
    let drafts = call
        .params
        .events
        .into_iter()
        .map(|e| Draft {
            kind: e.kind,
            data: e.data,
            source: e.source,
        })
        .collect();
    ingest(world, session, drafts)?;
    if let Some(mut worker) = world.get_mut::<Worker>(session) {
        worker.quiescent = false;
    }
    Ok(Empty {})
}

/// Takes in events a session produced, from its worker or a remote
/// server's journal: journals them, opens and routes their asks, and rolls
/// their usage up the tree.
pub fn ingest(world: &mut World, session: Entity, drafts: Vec<Draft>) -> Result<(), RpcError> {
    let asks: Vec<super::policy::AskDetails> = drafts
        .iter()
        .filter(|e| e.kind == "ask.opened")
        .filter_map(|e| super::policy::AskDetails::from_data(&e.data))
        .collect();
    let usage: Vec<rig_protocol::event::Usage> = drafts
        .iter()
        .filter(|e| e.kind == "usage")
        .filter_map(|e| serde_json::from_value(e.data.clone()).ok())
        .collect();
    publish(world, session, drafts)?;
    let timeout = Duration::from_secs(super::tree::policy_of(world, session).timeout_s);
    for details in asks {
        let ask = details.opened.clone();
        let now_ms = super::record::now_ms();
        let left = if ask.deadline_ms > now_ms {
            Duration::from_millis(ask.deadline_ms - now_ms)
        } else {
            timeout
        };
        let open = world
            .spawn((
                OpenAsk {
                    id: ask.ask_id,
                    session,
                    agent: ask.agent_id,
                    deadline: Instant::now() + left,
                    answerers: ask.answerers,
                },
                details,
            ))
            .id();
        super::policy::on_ask(world, open);
    }
    for usage in usage {
        world.trigger(super::tree::Usage {
            session,
            tokens: usage.tokens,
            cost_usd: usage.cost_usd.unwrap_or_default(),
        });
    }
    Ok(())
}

fn applied(In(call): In<Call<WorkerApplied>>, world: &mut World) -> Result<Empty, RpcError> {
    let session = own_session(world, &call.caller)?;
    let through = call.params.through;
    if let Some(mut worker) = world.get_mut::<Worker>(session) {
        worker.applied = worker.applied.max(through);
    }
    if let Some(mut track) = world.get_mut::<Track>(session) {
        track.applied = track.applied.max(through);
        track.commands.retain(|c| c.command_seq > through);
    }
    Ok(Empty {})
}

fn quiescent(In(call): In<Call<WorkerQuiescent>>, world: &mut World) -> Result<Empty, RpcError> {
    let session = own_session(world, &call.caller)?;
    if let Some(mut worker) = world.get_mut::<Worker>(session) {
        worker.quiescent = call.params.quiescent;
    }
    Ok(Empty {})
}

fn reply(In(call): In<Call<WorkerReply>>, world: &mut World) -> Result<Empty, RpcError> {
    own_session(world, &call.caller)?;
    let pending = world
        .get_resource_mut::<PluginCalls>()
        .and_then(|mut calls| calls.pending.remove(&call.params.reply));
    if let Some((client, id)) = pending {
        let reply = match call.params.outcome {
            Outcome::Result(value) => Ok(value),
            Outcome::Error(error) => Err(error),
        };
        respond(world, client, id, reply);
    }
    Ok(Empty {})
}

/// Starts swapping `session`'s build at its worker's next quiescent point.
pub fn begin_reload(world: &mut World, session: Entity, to: BuildId) {
    let Some(from) = world.get::<Session>(session).map(|s| s.build.clone()) else {
        return;
    };
    publish(
        world,
        session,
        vec![Draft::new(
            "reload.started",
            ReloadStarted { build: to.clone() },
        )],
    )
    .ok();
    world.entity_mut(session).insert(Reloading {
        to,
        from,
        spawned: None,
    });
}

/// Handles the frame's worker exits, unloads, kills and reloads; returns
/// the earliest deadline among them.
pub fn supervise(world: &mut World) -> Option<Instant> {
    let exits: Vec<Exited> = world
        .get_resource::<Server>()
        .map(|server| server.exited.try_iter().collect())
        .unwrap_or_default();
    for exit in exits {
        exited(world, exit);
    }
    let now = Instant::now();
    let mut next = None::<Instant>;
    let mut soon = |at: Instant| next = Some(next.map_or(at, |n: Instant| n.min(at)));
    let watchdog = world
        .get_resource::<Server>()
        .map_or(super::WATCHDOG, |s| s.options.watchdog);
    let mut workers = world.query::<(
        Entity,
        &Worker,
        &WorkerProcess,
        &Track,
        &SessionState,
        Option<&Reloading>,
    )>();
    let mut kill = Vec::new();
    let mut reload = Vec::new();
    for (session, worker, process, track, state, reloading) in workers.iter(world) {
        // A worker with work to do that says nothing is hung: it has a
        // command to apply or a turn running (an open ask waits on people).
        let busy = !track.commands.is_empty() || *state == SessionState::Running;
        if busy && process.asked.is_none() {
            let by = process.heard + watchdog;
            if by <= now {
                bevy_log::warn!("rigd: worker {} is hung; killing it", worker.pid);
                kill.push(worker.pid);
            } else {
                soon(by);
            }
        }
        if let Some(asked) = process.asked {
            if asked <= now {
                kill.push(worker.pid);
            } else {
                soon(asked);
            }
        }
        if let Some(reloading) = reloading
            && reloading.spawned.is_none()
            && worker.quiescent
            && process.asked.is_none()
        {
            reload.push(session);
        }
        if let Some(reloading) = reloading
            && let Some(spawned) = reloading.spawned
            && !process.hello
        {
            let by = spawned + RELOAD_HELLO;
            if by <= now {
                kill.push(worker.pid);
            } else {
                soon(by);
            }
        }
    }
    for pid in kill {
        signal(pid, true);
    }
    if let Some(at) = unload_idle(world) {
        soon(at);
    }
    for session in reload {
        unload(world, session);
    }
    next
}

/// Unloads every worker that has been quiescent, with no command, ask,
/// driver or reload waiting on it, for its session's unload delay; returns
/// when the next one is due.
fn unload_idle(world: &mut World) -> Option<Instant> {
    let delay = world.get_resource::<Server>()?.options.idle_unload;
    let now = Instant::now();
    let mut workers = world.query::<(
        Entity,
        &Worker,
        &mut WorkerProcess,
        &Track,
        Option<&super::Driver>,
        Option<&Reloading>,
    )>();
    let mut due = Vec::new();
    let mut next = None::<Instant>;
    for (session, worker, mut process, track, driver, reloading) in workers.iter_mut(world) {
        let quiet = worker.quiescent
            && process.hello
            && process.asked.is_none()
            && track.commands.is_empty()
            && track.asks.is_empty()
            && driver.is_none()
            && reloading.is_none();
        if !quiet {
            process.quiet_since = None;
            continue;
        }
        let since = *process.quiet_since.get_or_insert(now);
        let delay = match track.unload {
            Some(rig_protocol::methods::Unload::Turn) => Duration::ZERO,
            _ => delay,
        };
        let at = since + delay;
        if at <= now {
            due.push(session);
        } else {
            next = Some(next.map_or(at, |n: Instant| n.min(at)));
        }
    }
    for session in due {
        unload(world, session);
    }
    next
}

/// One worker exit: journaled, then the worker restarted, the session
/// stopped or faulted, or a reload finished or rolled back.
fn exited(world: &mut World, exit: Exited) {
    let session = exit.session;
    if world.get::<Worker>(session).map(|w| w.pid) != Some(exit.pid) {
        return;
    }
    let process = world.entity_mut(session).take::<WorkerProcess>();
    world.entity_mut(session).remove::<Worker>();
    if let Some(mut tokens) = world.get_resource_mut::<WorkerTokens>() {
        tokens.0.retain(|_, s| *s != session);
    }
    let asked = process.as_ref().is_some_and(|p| p.asked.is_some());
    let hello = process.as_ref().is_some_and(|p| p.hello);
    for held in process.into_iter().flat_map(|p| p.held) {
        if let Command::Plugin { reply, .. } = held.command {
            fail_plugin(world, reply);
        }
    }
    let code = exit.status.and_then(|s| s.code());
    #[cfg(unix)]
    let sig = exit
        .status
        .and_then(|s| std::os::unix::process::ExitStatusExt::signal(&s));
    #[cfg(not(unix))]
    let sig = None;
    let clean = code == Some(0) || asked;
    let (stopping, pending, state) = {
        let track = world.get::<Track>(session);
        (
            track.is_some_and(|t| t.stopping),
            track.is_some_and(|t| !t.commands.is_empty()),
            world
                .get::<SessionState>(session)
                .copied()
                .unwrap_or_default(),
        )
    };
    let reloading = world
        .get::<Reloading>(session)
        .map(|r| (r.from.clone(), r.to.clone(), r.spawned));
    let crash_limit = world
        .get_resource::<Server>()
        .map_or(3, |s| s.options.crash_limit);
    let busy = matches!(
        state,
        SessionState::Running | SessionState::NeedsApproval | SessionState::NeedsAnswer
    );
    let mut faulted = false;
    let restart = if stopping {
        false
    } else if reloading.is_some() {
        true
    } else if !clean {
        let now = Instant::now();
        if let Some(mut track) = world.get_mut::<Track>(session) {
            track
                .crashes
                .retain(|at| now.duration_since(*at) < CRASH_WINDOW);
            track.crashes.push_back(now);
            faulted = track.crashes.len() >= crash_limit;
        }
        !faulted
    } else {
        pending
    };
    let log = world.get::<Session>(session).and_then(|s| {
        let home = world.get_resource::<Server>()?.home.clone();
        let id = s.id.0.parse::<crate::harness_protocol::SessionId>().ok()?;
        Some(home.session(&id).path().join("worker.log"))
    });
    let mut drafts = vec![Draft::new(
        "worker.exited",
        WorkerExited {
            code,
            signal: sig,
            restarting: restart,
            log: log.clone(),
        },
    )];
    if faulted {
        drafts.push(Draft::new(
            "session.faulted",
            SessionFaulted {
                reason: format!("the worker crashed {crash_limit} times in a minute"),
                log: log.clone(),
            },
        ));
    }
    publish(world, session, drafts).ok();
    if stopping {
        super::methods::stopped(world, session).ok();
        return;
    }
    if let Some((from, to, spawned)) = reloading {
        if spawned.is_some() && !hello {
            // The new build died or never said hello: back to the old one.
            world.entity_mut(session).remove::<Reloading>();
            if let Some(mut s) = world.get_mut::<Session>(session) {
                s.build = from.clone();
            }
            let failed = ReloadFailed {
                build: to,
                error: "the new build did not start".to_owned(),
                kept: from,
            };
            publish(world, session, vec![Draft::new("reload.failed", failed)]).ok();
        } else {
            if let Some(mut s) = world.get_mut::<Session>(session) {
                s.build = to;
            }
            if let Some(mut r) = world.get_mut::<Reloading>(session) {
                r.spawned = Some(Instant::now());
            }
        }
    }
    if restart || (busy && !asked && !faulted) {
        if let Err(e) = spawn_worker(world, session) {
            let fault = SessionFaulted {
                reason: e.message,
                log,
            };
            publish(world, session, vec![Draft::new("session.faulted", fault)]).ok();
        }
    } else if faulted {
        world.entity_mut(session).insert(SessionState::Faulted);
    }
}

fn fail_plugin(world: &mut World, reply: u64) {
    let pending = world
        .get_resource_mut::<PluginCalls>()
        .and_then(|mut calls| calls.pending.remove(&reply));
    if let Some((client, id)) = pending {
        let error = RpcError::new(codes::WORKER_FAILED, "the worker exited before answering");
        respond(world, client, id, Err(error));
    }
}

/// The session ids of every live worker.
pub fn live(world: &mut World) -> Vec<SessionId> {
    let mut workers = world.query_filtered::<&Session, With<Worker>>();
    workers.iter(world).map(|s| s.id.clone()).collect()
}
