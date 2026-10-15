//! SN-20: tool reliability statistics. Every tool call that settles, a
//! subagent's too, is counted from its effect record as the session's
//! effect log gets it: calls, errors, cancelled and refused calls, how long
//! each ran, its result's size and its error message, masked so similar
//! ones group ([`mask`]). The session's totals are saved with it; the
//! project's are added to `RIG_HOME/tool-stats/<project>.json` by a worker
//! thread, so no call or frame waits on the file.
//!
//! `/toolstats [--by-agent] [--all] [--since 7d]` shows them, and
//! `/toolstats reset` clears the project's after the user confirms.

mod stats;

use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;
use std::time::{SystemTime, UNIX_EPOCH};

use futures::channel::oneshot;
use rig_harness::prelude::*;
use rig_harness::rig_ecs::StopTurnsOnExit;
use serde::{Deserialize, Serialize};

use stats::DAY_MS;
pub use stats::{Key, Table, Totals, mask};

/// How many error messages a row shows.
const TOP_MESSAGES: usize = 3;

/// The question `/toolstats reset` asks.
pub const RESET_TITLE: &str = "Reset the project's tool statistics?";

/// The tool statistics, and `/toolstats`.
#[derive(Default)]
pub struct ToolStatsPlugin;

impl Plugin for ToolStatsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SessionStats>()
            .add_message::<ToolStatsShown>()
            .add_command(
                "toolstats",
                "Tool statistics: [--by-agent] [--all] [--since 7d] | reset",
                toolstats,
            )
            .add_systems(Startup, start_worker)
            .add_systems(Last, count.after(StopTurnsOnExit))
            .add_observer(show)
            .add_observer(confirmed);
    }
}

/// This session's totals, by day, agent name and tool, saved with it.
#[derive(Resource, Reflect, Clone, Debug, Default, Serialize, Deserialize)]
#[reflect(opaque, Resource, Saved, Default, Serialize, Deserialize)]
pub struct SessionStats(pub Table);

/// Which statistics a `/toolstats` showed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatsOf {
    /// This session's.
    Session,
    /// The project's, every session's, from the day `since` (days since
    /// the Unix epoch) on.
    Project {
        /// The first day counted.
        since: u64,
    },
}

/// One row of a `/toolstats` table.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolSummary {
    /// The tool.
    pub tool: String,
    /// The agent, when split by agent.
    pub agent: Option<String>,
    /// The tool's totals.
    pub totals: Totals,
}

impl ToolSummary {
    /// The most frequent error messages, most frequent first, with their
    /// counts.
    pub fn top_errors(&self) -> Vec<(String, u64)> {
        self.totals.top_messages(TOP_MESSAGES)
    }
}

/// A `/toolstats` table as shown, for fronts and tests: the agent it was
/// asked for, what it covers and its rows, by tool.
#[derive(Message, Clone, Debug)]
pub struct ToolStatsShown {
    /// The agent it was typed for.
    pub agent: Entity,
    /// What it covers.
    pub of: StatsOf,
    /// The rows.
    pub rows: Vec<ToolSummary>,
}

/// What the worker thread is asked to do with the project's file.
enum Job {
    /// Add these totals.
    Add(Vec<(Key, Totals)>),
    /// Read the table.
    Read(oneshot::Sender<Result<Table, String>>),
    /// Empty the table.
    Reset(oneshot::Sender<Result<Table, String>>),
}

/// The worker that keeps the project's file, in the order it is asked.
/// Dropping it lets the worker finish what it was asked, and waits.
#[derive(Resource)]
struct Worker {
    jobs: Option<Sender<Job>>,
    thread: Option<JoinHandle<()>>,
}

impl Worker {
    fn send(&self, job: Job) -> bool {
        self.jobs
            .as_ref()
            .is_some_and(|jobs| jobs.send(job).is_ok())
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.jobs = None;
        if let Some(thread) = self.thread.take() {
            thread.join().ok();
        }
    }
}

/// Starts the worker on this project's file.
fn start_worker(home: Res<RigHome>, root: Option<Res<ProjectRoot>>, mut commands: Commands) {
    let root = root.map(|root| root.0.clone()).unwrap_or_default();
    let path = project_file(home.0.root(), &root);
    let (jobs, received) = channel();
    let thread = std::thread::Builder::new()
        .name("tool-stats".to_owned())
        .spawn(move || work(&path, &received))
        .ok();
    commands.insert_resource(Worker {
        jobs: thread.is_some().then_some(jobs),
        thread,
    });
}

/// `RIG_HOME/tool-stats/<project>.json`, the project named by its path.
fn project_file(home: &Path, root: &Path) -> PathBuf {
    let name: String = root
        .to_string_lossy()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect();
    let name = name.trim_matches('-');
    home.join("tool-stats").join(format!("{name}.json"))
}

/// Does the jobs in order until every sender is gone; consecutive
/// additions are written at once. An addition that could not be written
/// is said by the next read.
fn work(path: &Path, jobs: &Receiver<Job>) {
    let mut failed: Option<String> = None;
    let mut next = jobs.recv().ok();
    while let Some(job) = next.take() {
        match job {
            Job::Add(mut rows) => {
                loop {
                    match jobs.try_recv() {
                        Ok(Job::Add(more)) => rows.extend(more),
                        Ok(other) => {
                            next = Some(other);
                            break;
                        }
                        Err(_) => break,
                    }
                }
                let added = update(path, |table| {
                    for (key, totals) in rows {
                        table.add(key, &totals);
                    }
                });
                if let Err(why) = added {
                    failed = Some(why);
                }
            }
            Job::Read(reply) => {
                let read = match failed.take() {
                    Some(why) => Err(format!("some calls were not counted: {why}")),
                    None => read(path),
                };
                reply.send(read).ok();
            }
            Job::Reset(reply) => {
                let reset = write(path, &Table::default());
                reply.send(reset.map(|()| Table::default())).ok();
            }
        }
        if next.is_none() {
            next = jobs.recv().ok();
        }
    }
}

/// The project's table; empty when there is no file yet.
fn read(path: &Path) -> Result<Table, String> {
    match std::fs::read_to_string(path) {
        Ok(text) => {
            serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Table::default()),
        Err(error) => Err(format!("{}: {error}", path.display())),
    }
}

/// Changes the project's table with `change`, holding a lock file so the
/// sessions of one project add to it in turn.
fn update(path: &Path, change: impl FnOnce(&mut Table)) -> Result<(), String> {
    let _lock = lock(path)?;
    let mut table = read(path)?;
    change(&mut table);
    replace(path, &table)
}

/// Writes `table` as the project's, whatever the file held.
fn write(path: &Path, table: &Table) -> Result<(), String> {
    let _lock = lock(path)?;
    replace(path, table)
}

/// The project's lock file, locked until it is dropped.
fn lock(path: &Path) -> Result<std::fs::File, String> {
    let failed = |error: std::io::Error| format!("{}: {error}", path.display());
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(failed)?;
    }
    let lock = std::fs::File::create(path.with_extension("lock")).map_err(failed)?;
    lock.lock().map_err(failed)?;
    Ok(lock)
}

/// Replaces the file with `table`, whole, so no reader sees half of it.
fn replace(path: &Path, table: &Table) -> Result<(), String> {
    let failed = |error: std::io::Error| format!("{}: {error}", path.display());
    let text = serde_json::to_string(table).map_err(|error| error.to_string())?;
    let written = path.with_extension("json.new");
    std::fs::write(&written, text).map_err(failed)?;
    std::fs::rename(&written, path).map_err(failed)
}

/// Today, in days since the Unix epoch.
fn today() -> u64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    (now.as_millis() as u64) / DAY_MS
}

/// Counts each tool call recorded this frame, in the session's table and,
/// through the worker, the project's.
fn count(
    mut recorded: MessageReader<EffectRecorded>,
    agents: Query<(&AgentId, &Name, Has<SpawnedBy>)>,
    mut session: ResMut<SessionStats>,
    worker: Option<Res<Worker>>,
) {
    let mut project = Vec::new();
    for EffectRecorded(record) in recorded.read() {
        let Some((tool, totals)) = Totals::of(record) else {
            continue;
        };
        let day = record.finished_ms.map_or_else(today, |ms| ms / DAY_MS);
        let scope = record.scope.as_deref().unwrap_or_default();
        let agent = agents.iter().find(|(id, ..)| id.0 == scope);
        let (name, role) = match agent {
            Some((_, name, false)) => (name.as_str().to_owned(), "main"),
            Some((_, name, true)) => (name.as_str().to_owned(), "subagent"),
            None => (scope.to_owned(), "subagent"),
        };
        let key = |agent: &str| Key {
            day,
            agent: agent.to_owned(),
            tool: tool.clone(),
        };
        session.0.add(key(&name), &totals);
        project.push((key(role), totals));
    }
    if let (false, Some(worker)) = (project.is_empty(), worker) {
        worker.send(Job::Add(project));
    }
}

/// What `/toolstats` was asked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Asked {
    Show { of: StatsOf, by_agent: bool },
    Reset,
}

fn parse(args: &str) -> Result<Asked, String> {
    let mut words = args.split_whitespace();
    let (mut all, mut since, mut by_agent) = (false, None, false);
    while let Some(word) = words.next() {
        match word {
            "reset" => return Ok(Asked::Reset),
            "--all" => all = true,
            "--by-agent" => by_agent = true,
            "--since" => {
                let span = words.next().unwrap_or_default();
                since = Some(days(span).ok_or_else(|| {
                    format!("--since takes a span such as 7d or 2w, not `{span}`")
                })?);
            }
            other => return Err(format!("/toolstats does not take `{other}`")),
        }
    }
    let of = match (all, since) {
        (false, None) => StatsOf::Session,
        (_, since) => StatsOf::Project {
            since: since.map_or(0, |days: u64| {
                today().saturating_sub(days.saturating_sub(1))
            }),
        },
    };
    Ok(Asked::Show { of, by_agent })
}

/// The days of a span such as `7d` or `2w`.
fn days(span: &str) -> Option<u64> {
    let (number, unit) = span.split_at_checked(span.len().checked_sub(1)?)?;
    let number: u64 = number.parse().ok()?;
    match unit {
        "d" => Some(number),
        "w" => number.checked_mul(7),
        _ => None,
    }
}

/// On an entity waiting for the worker's answer to a `/toolstats`.
#[derive(Component)]
struct Waiting {
    agent: Entity,
    asked: Asked,
}

/// On the question `/toolstats reset` asks.
#[derive(Component)]
struct ResetQuestion;

/// `/toolstats`.
fn toolstats(
    In(args): In<CommandArgs>,
    session: Res<SessionStats>,
    worker: Option<Res<Worker>>,
    answerers: Answerers,
    wake: Res<Wake>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
    mut shown: MessageWriter<ToolStatsShown>,
) {
    let asked = match parse(&args.args) {
        Ok(asked) => asked,
        Err(why) => {
            notices.write(Notice::error(args.agent, why));
            return;
        }
    };
    match asked {
        Asked::Show {
            of: StatsOf::Session,
            by_agent,
        } => {
            let shown_now = shown_of(args.agent, StatsOf::Session, &session.0, by_agent);
            notices.write(Notice::info(args.agent, render(&shown_now)));
            shown.write(shown_now);
        }
        Asked::Show { .. } => {
            let Some(worker) = worker else {
                notices.write(Notice::error(
                    args.agent,
                    "the project's statistics are not kept",
                ));
                return;
            };
            let (reply, answer) = oneshot::channel();
            worker.send(Job::Read(reply));
            wait(&mut commands, &wake, args.agent, asked, answer);
        }
        Asked::Reset => {
            if !answerers.can_answer(Some(args.agent)) {
                let why = "/toolstats reset needs someone to confirm it";
                notices.write(Notice::error(args.agent, why));
                return;
            }
            let ask = Ask::yes_no(RESET_TITLE)
                .body("Every session's tool statistics in this project are cleared.");
            commands.spawn((ask, ForAgent(args.agent), ResetQuestion));
        }
    }
}

/// Waits off the main thread for the worker's `answer`.
fn wait(
    commands: &mut Commands,
    wake: &Wake,
    agent: Entity,
    asked: Asked,
    answer: oneshot::Receiver<Result<Table, String>>,
) {
    let pool = bevy_tasks::IoTaskPool::get_or_init(bevy_tasks::TaskPool::default);
    let answer = async move {
        answer
            .await
            .unwrap_or_else(|_| Err("the statistics worker stopped".to_owned()))
    };
    commands.spawn((Waiting { agent, asked }, Running::spawn(pool, wake, answer)));
}

/// Clears the project's statistics once the user said yes.
fn confirmed(
    answered: On<Add<Outcome>>,
    questions: Query<(&Outcome, &ForAgent), With<ResetQuestion>>,
    worker: Option<Res<Worker>>,
    wake: Res<Wake>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((outcome, &ForAgent(agent))) = questions.get(answered.entity) else {
        return;
    };
    if !Answer::of(outcome).is_some_and(|answer| answer.is("yes")) {
        notices.write(Notice::info(agent, "Tool statistics kept."));
        return;
    }
    let (reply, answer) = oneshot::channel();
    if worker.is_some_and(|worker| worker.send(Job::Reset(reply))) {
        wait(&mut commands, &wake, agent, Asked::Reset, answer);
    }
}

/// Shows the worker's answer.
fn show(
    done: On<Insert<Done<Result<Table, String>>>>,
    waiting: Query<(&Waiting, &Done<Result<Table, String>>)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
    mut shown: MessageWriter<ToolStatsShown>,
) {
    let Ok((Waiting { agent, asked }, Done(answer))) = waiting.get(done.entity) else {
        return;
    };
    match (asked, answer) {
        (_, Err(why)) => {
            notices.write(Notice::error(*agent, format!("tool statistics: {why}")));
        }
        (Asked::Reset, Ok(_)) => {
            notices.write(Notice::info(
                *agent,
                "The project's tool statistics were cleared.",
            ));
        }
        (Asked::Show { of, by_agent }, Ok(table)) => {
            let shown_now = shown_of(*agent, *of, table, *by_agent);
            notices.write(Notice::info(*agent, render(&shown_now)));
            shown.write(shown_now);
        }
    }
    commands.entity(done.entity).despawn();
}

/// The rows of `table` that `of` covers.
fn shown_of(agent: Entity, of: StatsOf, table: &Table, by_agent: bool) -> ToolStatsShown {
    let since = match of {
        StatsOf::Session => 0,
        StatsOf::Project { since } => since,
    };
    let rows = table
        .summed(since, by_agent)
        .into_iter()
        .map(|((tool, agent), totals)| ToolSummary {
            tool,
            agent,
            totals,
        })
        .collect();
    ToolStatsShown { agent, of, rows }
}

/// The table as text.
fn render(shown: &ToolStatsShown) -> String {
    let title = match shown.of {
        StatsOf::Session => "Tool statistics, this session".to_owned(),
        StatsOf::Project { since: 0 } => {
            "Tool statistics, every session of this project".to_owned()
        }
        StatsOf::Project { since } => format!(
            "Tool statistics, this project, the last {} days",
            today().saturating_sub(since) + 1
        ),
    };
    if shown.rows.is_empty() {
        return format!("{title}: no tool calls yet.");
    }
    let mut lines = vec![
        title,
        format!(
            "{:<24} {:>6} {:>6} {:>9} {:>7} {:>7} {:>7} {:>9}",
            "tool", "calls", "errors", "cancelled", "refused", "p50", "p95", "avg size"
        ),
    ];
    for row in &shown.rows {
        let totals = &row.totals;
        let name = match &row.agent {
            Some(agent) => format!("{} ({agent})", row.tool),
            None => row.tool.clone(),
        };
        let ms = |q| {
            totals
                .percentile_ms(q)
                .map_or_else(|| "-".to_owned(), duration)
        };
        let size = totals
            .average_bytes()
            .map_or_else(|| "-".to_owned(), |bytes| format!("{bytes} B"));
        lines.push(format!(
            "{name:<24} {:>6} {:>6} {:>9} {:>7} {:>7} {:>7} {size:>9}",
            totals.calls,
            totals.errors,
            totals.cancelled,
            totals.refused,
            ms(0.5),
            ms(0.95),
        ));
        for (message, count) in row.top_errors() {
            lines.push(format!("    {count}× {message}"));
        }
    }
    lines.join("\n")
}

fn duration(ms: u64) -> String {
    if ms < 1000 {
        format!("{ms}ms")
    } else {
        format!("{:.1}s", ms as f64 / 1000.0)
    }
}

#[cfg(test)]
mod tests;
