//! Background jobs (spec `22-background-jobs.md`): the `job_start`,
//! `job_output`, `job_kill` and `job_list` tools, `/jobs`, and a status
//! item with the running count.
//!
//! A job is `sh -c`, started as every child process starts ([`Processes`],
//! so a sandbox plugin's wrapper applies), writing its output straight to a
//! log file in the session's `jobs` directory, so it outlives the turn that
//! started it and, with [`JobSettings::survive_reload`], the process. A
//! small wrapper shell writes the job's exit code next to the log, for a
//! process that re-attaches by pid after `/reload`. While it runs, a
//! [`Running`] task on a job entity waits for it on a thread of its own and
//! ends as a [`Done<JobExit>`]; an observer of that tells the agent that
//! started it with a [`Deliver`]: a `Note` (with its next model call,
//! whenever that is), or with [`JobSettings::notify_idle`] a `Steer`, which
//! also starts a turn of an idle agent.
//!
//! What each agent started is its [`AgentJobs`], saved with the session.
//! The job entity is [`JobOf`] its agent and holds the process group: it is
//! killed when the entity goes (the agent despawned, the app dropped), and
//! on any exit but a `/reload` ([`ExitReason::Reload`]) with
//! `survive_reload`.

use std::fs::{self, File};
use std::io::{Read as _, Seek as _, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Child, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use bevy_app::OnAppExitSystems;
use bevy_tasks::{IoTaskPool, TaskPool};
use rig_harness::prelude::*;
use rig_harness::rig_ecs::session_log::Reset;
use rig_tools::process::Launcher;
use schemars::JsonSchema;
use serde::Deserialize;

#[cfg(feature = "tui")]
mod status;
#[cfg(test)]
mod tests;

/// Lines `job_output` returns unless asked for more.
const DEFAULT_TAIL: usize = 50;
/// Lines the note about a job's exit quotes.
const NOTE_LINES: usize = 5;
/// The end of the log the note's lines are taken from.
const NOTE_BYTES: u64 = 8 * 1024;
/// Runs the job as `$1` and writes its exit code to `$2`, so a process that
/// re-attaches by pid, and so is not its parent, learns it too.
const WRAPPER: &str = r#"sh -c "$1"; s=$?; echo "$s" > "$2"; exit "$s""#;

/// The `[jobs]` table of the user's settings.
#[derive(Resource, Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct JobSettings {
    /// A job that exits while its agent is idle starts a turn; otherwise
    /// the note waits for the agent's next model call.
    pub notify_idle: bool,
    /// `/reload` keeps running jobs, and the new process re-attaches to them.
    pub survive_reload: bool,
    /// How much of the end of a job's output `job_output` reads.
    pub buffer_bytes: u64,
}

impl Default for JobSettings {
    fn default() -> Self {
        Self {
            notify_idle: false,
            survive_reload: false,
            buffer_bytes: 2 * 1024 * 1024,
        }
    }
}

/// Where a job is.
#[derive(Reflect, Clone, Debug, Default, PartialEq, Eq)]
pub enum JobState {
    /// It runs.
    #[default]
    Running,
    /// It exited with this code, or by a signal when `None`.
    Exited(Option<i32>),
    /// `job_kill` killed it.
    Killed,
    /// It ended with the process that started it, or was not found again.
    Lost,
}

/// A job an agent started.
#[derive(Reflect, Clone, Debug, Default)]
pub struct JobRecord {
    /// Its id, unique for the agent.
    pub id: u32,
    /// Its name.
    pub name: String,
    /// Its command line.
    pub command: String,
    /// Its log file.
    pub log: String,
    /// The pid of its process group's leader.
    pub pid: u32,
    /// When it started, in ms since the Unix epoch.
    pub started_ms: u64,
    /// When it ended.
    pub ended_ms: Option<u64>,
    /// Where it is.
    pub state: JobState,
}

impl JobRecord {
    /// The file the wrapper writes the exit code to.
    fn status_file(&self) -> PathBuf {
        Path::new(&self.log).with_extension("status")
    }

    /// One line about it.
    fn line(&self) -> String {
        let took = duration(self.ended_ms.unwrap_or_else(now_ms), self.started_ms);
        let state = match self.state {
            JobState::Running => format!("running for {took}, pid {}", self.pid),
            JobState::Exited(code) => format!("exited with {} after {took}", code_label(code)),
            JobState::Killed => format!("killed after {took}"),
            JobState::Lost => "lost in a restart".to_owned(),
        };
        format!(
            "{} `{}`: {state}; `{}`; log {}",
            self.id, self.name, self.command, self.log
        )
    }
}

/// The jobs an agent started, kept with the session.
#[derive(Component, Reflect, Clone, Debug, Default)]
#[reflect(Component, Saved(Reset), Default)]
pub struct AgentJobs(pub Vec<JobRecord>);

/// A running job's entity: the agent that started it.
#[derive(Component, Debug)]
#[relationship(relationship_target = LiveJobs)]
pub struct JobOf(pub Entity);

/// An agent's running jobs, killed with it.
#[derive(Component, Debug)]
#[relationship_target(relationship = JobOf, linked_spawn)]
pub struct LiveJobs(Vec<Entity>);

/// A running job: its id and process group.
#[derive(Component)]
pub struct LiveJob {
    id: u32,
    group: Group,
}

/// How a job ended, as its waiting task says.
pub enum JobExit {
    /// With this code, or by a signal when `None`.
    Exited(Option<i32>),
    /// Unknown: a re-attached job whose process was gone without a code.
    Lost,
}

/// A process group, killed when this is dropped unless it ended or was kept.
struct Group {
    pid: u32,
    state: Arc<AtomicU8>,
}

const LIVE: u8 = 0;
const KEPT: u8 = 1;
const GONE: u8 = 2;

impl Group {
    fn new(pid: u32) -> Self {
        Self {
            pid,
            state: Arc::new(AtomicU8::new(LIVE)),
        }
    }

    /// Kills the group, once.
    fn kill(&self) {
        end(&self.state, self.pid);
    }

    /// Leaves the group running, for the next process.
    fn keep(&self) {
        self.state
            .compare_exchange(LIVE, KEPT, Ordering::AcqRel, Ordering::Acquire)
            .ok();
    }
}

impl Drop for Group {
    fn drop(&mut self) {
        self.kill();
    }
}

/// Kills the group `pid` leads unless it is gone or kept, and marks it
/// gone: once its leader was reaped, the group's id may name another.
fn end(state: &AtomicU8, pid: u32) {
    if state
        .compare_exchange(LIVE, GONE, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
    {
        #[cfg(unix)]
        rig_tools::process::kill_group_of(pid);
        #[cfg(not(unix))]
        let _ = pid;
    }
}

/// The tools, `/jobs`, the status item, and the jobs' lifecycle.
#[derive(Default)]
pub struct BackgroundJobsPlugin;

impl Plugin for BackgroundJobsPlugin {
    fn build(&self, app: &mut App) {
        let options = |footprint| ToolOptions {
            footprint: Some(footprint),
        };
        app.init_resource::<Processes>()
            .add_setting::<JobSettings>("jobs")
            .register_required_components::<Agent, AgentJobs>()
            .add_open_tool(
                "job_start",
                "Start a shell command in the background and return its job id at once. \
                 It keeps running after this turn; you are told when it exits. Use it for \
                 commands that run long or never end, such as dev servers, watchers or slow \
                 test suites, then keep working; check them with `job_output` and stop them \
                 with `job_kill`.",
                options(Touches::Writes(Scope::Anything)),
                on_start,
            )
            .add_open_tool(
                "job_output",
                "The last lines a background job printed.",
                options(Touches::Reads(Scope::Anything)),
                on_output,
            )
            .add_open_tool(
                "job_kill",
                "Kill a background job and its child processes.",
                options(Touches::Writes(Scope::Anything)),
                on_kill,
            )
            .add_open_tool(
                "job_list",
                "List your background jobs.",
                options(Touches::Reads(Scope::Anything)),
                on_list,
            )
            .add_command("jobs", "List the background jobs", list_command)
            .add_observer(on_exit)
            .add_observer(reattach)
            .add_systems(
                Last,
                kill_on_exit
                    .in_set(OnAppExitSystems)
                    .run_if(resource_exists::<ExitReason>),
            );
        #[cfg(feature = "tui")]
        status::add(app);
    }
}

/// The arguments of `job_start`.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct StartArgs {
    /// The command line, run with `sh -c` in the working directory.
    command: String,
    /// A short name for the job, such as `tests`.
    name: Option<String>,
}

/// The arguments of `job_output`.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct OutputArgs {
    /// The job's id.
    id: u32,
    /// How many of the last lines (default 50).
    tail: Option<usize>,
}

/// The arguments of `job_kill`.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct KillArgs {
    /// The job's id.
    id: u32,
}

/// The arguments of `job_list`.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ListArgs {}

/// Decides `call`: its result's text, or why it failed.
fn decide(
    calls: &mut Query<&mut Decision, ToolCalls>,
    call: Entity,
    result: Result<String, String>,
) {
    let Ok(mut decision) = calls.get_mut(call) else {
        return;
    };
    decision.decide(
        Decider::plugin("background-jobs"),
        match result {
            Ok(text) => ToolResult::success(text.into()),
            Err(why) => ToolResult::failed(ToolExecutionError::other(why)),
        },
    );
}

fn pool() -> &'static TaskPool {
    IoTaskPool::get_or_init(TaskPool::default)
}

fn on_start(
    called: On<ToolCalled<StartArgs>>,
    mut agents: Query<&mut AgentJobs>,
    paths: Option<Res<SessionPaths>>,
    processes: Res<Processes>,
    wake: Res<Wake>,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut commands: Commands,
) {
    let (call, agent) = (called.call, called.agent);
    let Ok(mut jobs) = agents.get_mut(agent) else {
        return;
    };
    let Some(paths) = paths else {
        let why = "No session directory to keep the job's log in.".to_owned();
        decide(&mut calls, call, Err(why));
        return;
    };
    let id = jobs.0.iter().map(|job| job.id).max().unwrap_or(0) + 1;
    let log = paths
        .path()
        .join("jobs")
        .join(&called.agent_id.0)
        .join(format!("{id}.log"));
    let StartArgs { command, name } = &called.args;
    let name = name.clone().unwrap_or_else(|| default_name(command));
    let child = match launch(&processes.launcher(), command, &log) {
        Ok(child) => child,
        Err(error) => {
            let why = format!("Could not start the job: {error}");
            decide(&mut calls, call, Err(why));
            return;
        }
    };
    let pid = child.id();
    jobs.0.push(JobRecord {
        id,
        name: name.clone(),
        command: command.clone(),
        log: log.display().to_string(),
        pid,
        started_ms: now_ms(),
        ended_ms: None,
        state: JobState::Running,
    });
    let group = Group::new(pid);
    let state = group.state.clone();
    let waiting = async move {
        let code = blocking(move || Ok(wait_child(child, &state))).await;
        JobExit::Exited(code.ok().flatten())
    };
    commands.spawn((
        Name::new(format!("job {name}")),
        JobOf(agent),
        LiveJob { id, group },
        Running::spawn(pool(), &wake, waiting),
    ));
    let text = format!(
        "Started job {id} `{name}` (pid {pid}), logging to {}. You will be told when it exits.",
        log.display()
    );
    decide(&mut calls, call, Ok(text));
}

/// Starts `command` as `launcher` starts processes, its output to `log`.
fn launch(launcher: &Launcher, command: &str, log: &Path) -> std::io::Result<Child> {
    if let Some(dir) = log.parent() {
        fs::create_dir_all(dir)?;
    }
    let out = File::create(log)?;
    let err = out.try_clone()?;
    launcher
        .command("sh")
        .args(["-c", WRAPPER, "rig-job", command])
        .arg(log.with_extension("status"))
        .stdin(Stdio::null())
        .stdout(out)
        .stderr(err)
        .spawn()
}

/// Waits for `child`, then kills what is left of its group: its exit code,
/// or `None` for a signal.
fn wait_child(mut child: Child, state: &AtomicU8) -> Option<i32> {
    let status = child.wait().ok();
    end(state, child.id());
    status.and_then(|status| status.code())
}

fn on_output(
    called: On<ToolCalled<OutputArgs>>,
    agents: Query<&AgentJobs>,
    settings: Res<JobSettings>,
    wake: Res<Wake>,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut commands: Commands,
) {
    let call = called.call;
    let Some(job) = find(&agents, called.agent, called.args.id) else {
        let why = format!("No job {}; job_list lists them.", called.args.id);
        decide(&mut calls, call, Err(why));
        return;
    };
    let (log, bytes) = (PathBuf::from(&job.log), settings.buffer_bytes);
    let lines = called.args.tail.unwrap_or(DEFAULT_TAIL);
    let head = job.line();
    let reading = async move {
        match blocking(move || Ok(tail(&log, bytes, lines))).await {
            Ok(text) => ToolResult::success(format!("{head}\n\n{text}").into()),
            Err(error) => ToolResult::failed(error),
        }
    };
    commands
        .entity(call)
        .insert(Running::answering(pool(), &wake, reading));
}

fn on_kill(
    called: On<ToolCalled<KillArgs>>,
    mut agents: Query<&mut AgentJobs>,
    live: Query<(&JobOf, &LiveJob)>,
    mut calls: Query<&mut Decision, ToolCalls>,
) {
    let (call, agent, id) = (called.call, called.agent, called.args.id);
    let job = live.iter().find(|(of, job)| of.0 == agent && job.id == id);
    let record = agents
        .get_mut(agent)
        .ok()
        .and_then(|jobs| jobs.into_inner().0.iter_mut().find(|job| job.id == id));
    let result = match (job, record) {
        (Some((_, job)), Some(record)) => {
            job.group.kill();
            record.state = JobState::Killed;
            Ok(format!("Killed job {id} `{}`.", record.name))
        }
        (None, Some(record)) => Err(format!("Job {id} is not running: {}", record.line())),
        (_, None) => Err(format!("No job {id}; job_list lists them.")),
    };
    decide(&mut calls, call, result);
}

fn on_list(
    called: On<ToolCalled<ListArgs>>,
    agents: Query<&AgentJobs>,
    mut calls: Query<&mut Decision, ToolCalls>,
) {
    let lines = agents
        .get(called.agent)
        .map(|jobs| jobs.0.iter().map(JobRecord::line).collect::<Vec<_>>())
        .unwrap_or_default();
    let text = if lines.is_empty() {
        "No jobs.".to_owned()
    } else {
        lines.join("\n")
    };
    decide(&mut calls, called.call, Ok(text));
}

/// `/jobs`: every agent's jobs.
fn list_command(
    In(args): In<CommandArgs>,
    agents: Query<(&Name, &AgentJobs)>,
    mut notices: MessageWriter<Notice>,
) {
    let mut lines = Vec::new();
    for (name, jobs) in &agents {
        lines.extend(jobs.0.iter().map(|job| format!("{name} {}", job.line())));
    }
    let text = if lines.is_empty() {
        "No background jobs.".to_owned()
    } else {
        lines.join("\n")
    };
    notices.write(Notice::info(args.agent, text));
}

/// Tells the agent a job ended, unless it killed it, and forgets the job's
/// entity.
fn on_exit(
    done: On<Add<Done<JobExit>>>,
    live: Query<(&JobOf, &LiveJob, &Done<JobExit>)>,
    mut agents: Query<&mut AgentJobs>,
    settings: Res<JobSettings>,
    mut commands: Commands,
) {
    let Ok((&JobOf(agent), job, Done(exit))) = live.get(done.entity) else {
        return;
    };
    commands.entity(done.entity).despawn();
    let Ok(mut jobs) = agents.get_mut(agent) else {
        return;
    };
    let Some(record) = jobs.0.iter_mut().find(|record| record.id == job.id) else {
        return;
    };
    let ended = now_ms();
    record.ended_ms = Some(ended);
    if record.state == JobState::Killed {
        return;
    }
    let how = match exit {
        JobExit::Exited(code) => {
            record.state = JobState::Exited(*code);
            format!("exited with {}", code_label(*code))
        }
        JobExit::Lost => {
            record.state = JobState::Lost;
            "ended while the agent restarted".to_owned()
        }
    };
    let last = tail(Path::new(&record.log), NOTE_BYTES, NOTE_LINES);
    let text = format!(
        "job `{}` (id {}) {how} after {}; last lines:\n{last}",
        record.name,
        record.id,
        duration(ended, record.started_ms)
    );
    let mode = if settings.notify_idle {
        DeliveryMode::Steer
    } else {
        DeliveryMode::Inform
    };
    let author = Author::plugin("background-jobs");
    commands.trigger(Deliver::new(agent, text, mode).with_author(author));
}

/// When a restored agent gets its saved jobs, waits again for those still
/// running with `survive_reload` (the previous process kept them only for
/// `/reload`); otherwise kills any that still runs, as after a crash.
fn reattach(
    inserted: On<Insert<AgentJobs>>,
    mut agents: Query<&mut AgentJobs>,
    settings: Res<JobSettings>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let agent = inserted.entity;
    let Ok(mut jobs) = agents.get_mut(agent) else {
        return;
    };
    let running = jobs
        .0
        .iter_mut()
        .filter(|record| record.state == JobState::Running);
    for record in running {
        let (pid, status) = (record.pid, record.status_file());
        let alive = ours(pid, &status);
        if !(alive && settings.survive_reload) {
            if alive {
                Group::new(pid).kill();
            }
            record.state = JobState::Lost;
            continue;
        }
        let group = Group::new(pid);
        let state = group.state.clone();
        let waiting = async move {
            let exit = blocking(move || Ok(wait_foreign(pid, &status, &state))).await;
            exit.unwrap_or(JobExit::Lost)
        };
        commands.spawn((
            Name::new(format!("job {}", record.name)),
            JobOf(agent),
            LiveJob {
                id: record.id,
                group,
            },
            Running::spawn(pool(), &wake, waiting),
        ));
    }
}

/// Waits for the job `pid` leads, which an earlier process started, and
/// reads the code its wrapper wrote.
fn wait_foreign(pid: u32, status: &Path, state: &AtomicU8) -> JobExit {
    if ours(pid, status) {
        wait_pid(pid);
        end(state, pid);
    }
    let code = fs::read_to_string(status).ok();
    match code.and_then(|code| code.trim().parse().ok()) {
        Some(code) => JobExit::Exited(Some(code)),
        None => JobExit::Lost,
    }
}

/// Whether `pid` still leads the job whose code goes to `status`, not
/// another process that got its pid.
fn ours(pid: u32, status: &Path) -> bool {
    let cmdline = fs::read(format!("/proc/{pid}/cmdline")).unwrap_or_default();
    let status = status.display().to_string();
    cmdline
        .split(|byte| *byte == 0)
        .any(|arg| arg == status.as_bytes())
}

/// Blocks until the process `pid`, not a child of this one, exits.
#[cfg(target_os = "linux")]
fn wait_pid(pid: u32) {
    use std::os::fd::{AsRawFd as _, FromRawFd as _, OwnedFd};
    let Ok(pid) = libc::pid_t::try_from(pid) else {
        return;
    };
    // SAFETY: `pidfd_open` takes a pid and flags and returns a new file
    // descriptor or -1.
    let fd = unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0) };
    let Ok(fd) = i32::try_from(fd) else {
        return;
    };
    if fd < 0 {
        return;
    }
    // SAFETY: `fd` is the new descriptor `pidfd_open` returned, owned here.
    let fd = unsafe { OwnedFd::from_raw_fd(fd) };
    let mut poll = libc::pollfd {
        fd: fd.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: one valid `pollfd`; -1 waits until the process exits.
    while unsafe { libc::poll(&mut poll, 1, -1) } < 0
        && std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted
    {}
}

/// Elsewhere a job is not re-attached: `ours` never finds it.
#[cfg(not(target_os = "linux"))]
fn wait_pid(_pid: u32) {}

/// Kills every job on the way out, except on `/reload` with
/// `survive_reload`, which leaves them to the next process.
fn kill_on_exit(settings: Res<JobSettings>, reason: Res<ExitReason>, live: Query<&LiveJob>) {
    let keep = settings.survive_reload && *reason == ExitReason::Reload;
    for job in &live {
        if keep {
            job.group.keep();
        } else {
            job.group.kill();
        }
    }
}

fn find<'a>(agents: &'a Query<&AgentJobs>, agent: Entity, id: u32) -> Option<&'a JobRecord> {
    let jobs = agents.get(agent).ok()?;
    jobs.0.iter().find(|job| job.id == id)
}

/// The last `lines` lines of the last `bytes` of `log`.
fn tail(log: &Path, bytes: u64, lines: usize) -> String {
    let read = || -> std::io::Result<String> {
        let mut file = File::open(log)?;
        let start = file.metadata()?.len().saturating_sub(bytes);
        file.seek(SeekFrom::Start(start))?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        Ok(String::from_utf8_lossy(&buffer).into_owned())
    };
    let text = read().unwrap_or_default();
    let all: Vec<&str> = text.lines().collect();
    let kept = all
        .get(all.len().saturating_sub(lines)..)
        .unwrap_or_default();
    if kept.is_empty() {
        "(no output)".to_owned()
    } else {
        kept.join("\n")
    }
}

/// A job's name from its command: its first 40 characters on one line.
fn default_name(command: &str) -> String {
    let line = command.split_whitespace().collect::<Vec<_>>().join(" ");
    line.chars().take(40).collect()
}

fn code_label(code: Option<i32>) -> String {
    code.map_or_else(|| "a signal".to_owned(), |code| code.to_string())
}

fn now_ms() -> u64 {
    let since = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    u64::try_from(since.as_millis()).unwrap_or(u64::MAX)
}

/// `3m12s` from `start` to `end`, in ms.
fn duration(end: u64, start: u64) -> String {
    let secs = end.saturating_sub(start) / 1000;
    match (secs / 3600, secs / 60 % 60, secs % 60) {
        (0, 0, s) => format!("{s}s"),
        (0, m, s) => format!("{m}m{s}s"),
        (h, m, _) => format!("{h}h{m}m"),
    }
}
