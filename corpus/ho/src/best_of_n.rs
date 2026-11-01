//! Plugin corpus (held-out): H16: Best-of-N attempts in git worktrees (spec
//! `H16-best-of-n.md`).
//!
//! `/best-of 3 --verify "cargo test" fix the flaky test in net.rs` makes 3
//! git worktrees of the project as it is now (HEAD plus the uncommitted
//! changes, as a snapshot commit) under `.rig/best-of/<id>/<k>`, and spawns
//! one one-job helper agent per worktree with the task. Each candidate has
//! the project's directory in its worktree as its `WorkDir`, so its file
//! and shell tools work there and refuse paths outside it. When a
//! candidate's turn ends, its tree is snapshotted, its diff measured and
//! the verify command run in it. Once all are done, a picker lists them
//! (passing first, then the smaller diff) with their diffs; `--auto`, or a
//! run nobody answers, takes the first passing one. The winner is merged
//! into the project's tree with a 3-way merge against the user's changes
//! since the start, refused on a conflict, and the agent gets a note naming
//! the winner. `--models a,b,c` gives each candidate its model; N is capped
//! at 5. Stopping the agent cancels every candidate. Worktrees are removed
//! after an apply or a discard; `/best-of clean` removes leftovers.
//!
//! Every git step and verify run is an effect of the agent (`best-of-*`),
//! recorded, gateable and cancelled with its run, which kills its process.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bevy_tasks::{IoTaskPool, TaskPool};
use rig_basics::subagents::{SpawnSubagent, SpawnedAgent, SubagentsPlugin};
use rig_harness::prelude::*;
use rig_harness::rig_core::effect::{Answered, EffectKind};
use rig_harness::rig_core::error::ErrorReport;
use rig_harness::rig_ecs::session_log::Never;
use rig_tools::process::{KillOnDrop, Launcher, OutputLine, Spawn};
use rig_tools::{MAX_BYTES, OutputLimit, Shell, ShellArgs, blocking};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[cfg(test)]
mod tests;

/// The most candidates a run has.
pub const MAX_CANDIDATES: usize = 5;
/// Candidates when neither N nor `--models` says.
const DEFAULT_CANDIDATES: usize = 2;
/// Where the worktrees go, from the project.
const DIR: &str = ".rig/best-of";
/// The tools a candidate gets, those of them its parent has: the ones that
/// work in an agent's `WorkDir`.
const TOOLS: [&str; 5] = ["read", "write", "edit", "search", "shell"];
/// Lines of each diff the picker shows.
const DIFF_LINES: usize = 40;
/// How long the verify command may run.
const VERIFY_SECS: u64 = 600;
/// How long one git command may run.
const GIT_TIMEOUT: Duration = Duration::from_secs(600);

const HELP: &str = "N agents try a task in git worktrees; the verified winner is applied: \
    /best-of [N] [--verify \"cmd\"] [--auto] [--models a,b] <task>, or /best-of clean";

/// The `[best_of]` table of the settings: `verify = "cargo test"`, the
/// command a run verifies its candidates with unless `--verify` says.
#[derive(Resource, Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BestOfConfig {
    pub verify: Option<String>,
}

/// Adds `/best-of`.
#[derive(Default)]
pub struct BestOfPlugin;

impl Plugin for BestOfPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<SubagentsPlugin>() {
            app.add_plugins(SubagentsPlugin);
        }
        app.init_resource::<ProjectRoot>()
            .init_resource::<Processes>()
            .add_setting::<BestOfConfig>("best_of")
            .add_command("best-of", HELP, best_of)
            .add_systems(Update, retire_orphans.before(AgentLoop))
            .add_observer(start_job)
            .add_observer(job_done)
            .add_observer(candidate_spawned)
            .add_observer(spawn_refused)
            .add_observer(candidate_done)
            .add_observer(cancel_on_interrupt);
    }
}

/// A best-of run: its candidates and how it ends.
#[derive(Component, Debug)]
pub struct Run {
    /// Names its directory, `.rig/best-of/<id>`.
    pub id: String,
    /// The agent that asked, which gets the note.
    owner: Entity,
    task: String,
    verify: String,
    auto: bool,
    /// The project's root and the run's directory.
    root: PathBuf,
    dir: PathBuf,
    /// The snapshot every worktree starts from, once made.
    base: Option<String>,
    /// The project's directory inside the repository, such as `crates/x/`.
    prefix: String,
    /// The tools each candidate gets.
    tools: Vec<String>,
    pub candidates: Vec<Candidate>,
}

/// One attempt.
#[derive(Debug)]
pub struct Candidate {
    /// 1-based.
    pub k: usize,
    model: Option<String>,
    agent: Option<Entity>,
    /// How it went, once known: its verdict, or why it has none.
    pub result: Option<Result<Verdict, String>>,
}

/// A candidate's tree once its turn ended.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Verdict {
    pub passed: bool,
    /// Lines added and removed.
    pub size: usize,
    /// The snapshot commit of its tree.
    commit: String,
    diff: String,
}

impl Run {
    fn worktree(&self, k: usize) -> PathBuf {
        self.dir.join(k.to_string())
    }

    /// The project's directory in candidate `k`'s worktree, where it works.
    fn work_dir(&self, k: usize) -> PathBuf {
        self.worktree(k).join(&self.prefix)
    }

    /// The candidates with a verdict, best first: passing, then the smaller
    /// diff.
    fn ranked(&self) -> Vec<(usize, &Verdict)> {
        let mut ranked: Vec<(usize, &Verdict)> = self
            .candidates
            .iter()
            .filter_map(|candidate| Some((candidate.k, candidate.result.as_ref()?.as_ref().ok()?)))
            .collect();
        ranked.sort_by_key(|(k, verdict)| (!verdict.passed, verdict.size, *k));
        ranked
    }

    fn candidate(&mut self, k: usize) -> Option<&mut Candidate> {
        self.candidates
            .iter_mut()
            .find(|candidate| candidate.k == k)
    }
}

/// On a candidate agent: its run.
#[derive(Component, Debug)]
pub struct CandidateOf {
    pub run: Entity,
    pub k: usize,
}

/// Saved on a candidate, so one a restart brings back without its run is
/// retired.
#[derive(Component, Reflect, Default)]
#[reflect(Component, Saved(Never), Default)]
pub struct WasCandidate;

/// On a candidate's `SpawnSubagent` request.
#[derive(Component)]
struct CandidateRequest {
    run: Entity,
    k: usize,
}

/// On a picker: the run, and the candidate of each item.
#[derive(Component)]
struct PickerOf {
    run: Entity,
    items: Vec<usize>,
}

/// On a `best-of-*` effect: the git or verify work it does, and its run.
#[derive(Component, Clone, Debug)]
struct Job {
    run: Option<Entity>,
    work: Work,
}

#[derive(Clone, Debug)]
enum Work {
    /// Snapshots the project and makes `n` worktrees in `dir`.
    Setup {
        root: PathBuf,
        dir: PathBuf,
        n: usize,
    },
    /// Snapshots candidate `k`'s worktree and runs `command` in `cwd`.
    Verify {
        k: usize,
        worktree: PathBuf,
        cwd: PathBuf,
        base: String,
        command: String,
    },
    /// Merges candidate `k`'s `commit` into the project's tree.
    Apply {
        k: usize,
        why: String,
        root: PathBuf,
        base: String,
        commit: String,
    },
    /// Removes the worktrees of the run directories `dirs`.
    Clean {
        root: PathBuf,
        dirs: Vec<PathBuf>,
        asked: bool,
    },
}

impl Work {
    fn name(&self) -> &'static str {
        match self {
            Self::Setup { .. } => "best-of-setup",
            Self::Verify { .. } => "best-of-verify",
            Self::Apply { .. } => "best-of-apply",
            Self::Clean { .. } => "best-of-clean",
        }
    }

    fn payload(&self) -> Value {
        match self {
            Self::Setup { dir, n, .. } => json!({ "dir": dir, "candidates": n }),
            Self::Verify {
                k,
                worktree,
                command,
                ..
            } => json!({ "candidate": k, "worktree": worktree, "command": command }),
            Self::Apply { k, commit, .. } => json!({ "candidate": k, "commit": commit }),
            Self::Clean { dirs, .. } => json!({ "dirs": dirs }),
        }
    }

    /// Does the work, starting processes with `launcher`.
    async fn run(self, launcher: Launcher) -> Result<Value, String> {
        let git = Git(launcher.clone());
        match self {
            Self::Setup { root, dir, n } => setup(&git, &root, &dir, n).await,
            Self::Verify {
                worktree,
                cwd,
                base,
                command,
                ..
            } => verify(&git, launcher.in_dir(cwd), &worktree, &base, command).await,
            Self::Apply {
                root, base, commit, ..
            } => apply(&git, &root, &base, &commit).await,
            Self::Clean { root, dirs, .. } => {
                for dir in &dirs {
                    remove_worktrees(&git, &root, dir).await;
                }
                Ok(json!({ "removed": dirs.len() }))
            }
        }
    }
}

/// Spawns `work` as an effect of `agent`, a child of `run` when it has one
/// so it is cancelled with it.
fn spawn_job(commands: &mut Commands, agent: Entity, run: Option<Entity>, work: Work) {
    let kind = EffectKind::Custom {
        kind: work.name().into(),
        payload: work.payload(),
    };
    let request = EffectRequest {
        kind,
        handler: None,
    };
    let name = Name::new(work.name());
    let mut job = commands.spawn((request, ForAgent(agent), name, Job { run, work }));
    if let Some(run) = run {
        job.insert(ChildOf(run));
    }
}

/// What `/best-of` was asked.
#[derive(Debug, PartialEq, Eq)]
enum Asked {
    Clean,
    Run {
        n: usize,
        verify: Option<String>,
        auto: bool,
        models: Vec<String>,
        task: String,
    },
}

/// The next word of `text`, quoted or not, and the rest.
fn word(text: &str) -> (&str, &str) {
    let text = text.trim_start();
    for quote in ['"', '\''] {
        if let Some(body) = text.strip_prefix(quote) {
            return body.split_once(quote).unwrap_or((body, ""));
        }
    }
    text.split_once(char::is_whitespace).unwrap_or((text, ""))
}

/// Reads `/best-of`'s arguments: options and N first, then the task.
fn parse(args: &str) -> Result<Asked, String> {
    if args.trim() == "clean" {
        return Ok(Asked::Clean);
    }
    let (mut n, mut verify, mut auto, mut models) = (None, None, false, Vec::new());
    let mut rest = args.trim();
    loop {
        let (next, after) = word(rest);
        match next {
            "--auto" => auto = true,
            "--verify" => {
                let (command, after) = word(after);
                verify = Some(command.to_owned());
                rest = after.trim();
                continue;
            }
            "--models" => {
                let (list, after) = word(after);
                models = list
                    .split(',')
                    .map(str::trim)
                    .filter(|model| !model.is_empty())
                    .map(str::to_owned)
                    .collect();
                rest = after.trim();
                continue;
            }
            _ if n.is_none() && next.parse::<usize>().is_ok() => n = next.parse().ok(),
            _ => break,
        }
        rest = after.trim();
    }
    if rest.is_empty() {
        return Err(format!("/best-of needs a task. {HELP}"));
    }
    let n = n.unwrap_or(match models.len() {
        0 => DEFAULT_CANDIDATES,
        given => given,
    });
    if n == 0 {
        return Err("/best-of needs at least one candidate".to_owned());
    }
    Ok(Asked::Run {
        n: n.min(MAX_CANDIDATES),
        verify,
        auto,
        models,
        task: rest.to_owned(),
    })
}

/// A run's id: the time, made unique within the process.
fn run_id() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|since| since.as_millis())
        .unwrap_or_default();
    format!("{millis}-{}", NEXT.fetch_add(1, Ordering::Relaxed))
}

/// `/best-of`: starts a run, or cleans up leftovers.
fn best_of(
    In(args): In<CommandArgs>,
    config: Res<BestOfConfig>,
    root: Res<ProjectRoot>,
    runs: Query<&Run>,
    access: Query<&ToolAccess>,
    defs: Query<&ToolDef>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let asked = match parse(&args.args) {
        Ok(asked) => asked,
        Err(why) => {
            notices.write(Notice::error(agent, why));
            return;
        }
    };
    let root = root.0.clone();
    let Asked::Run {
        n,
        verify,
        auto,
        models,
        task,
    } = asked
    else {
        let running: Vec<&PathBuf> = runs.iter().map(|run| &run.dir).collect();
        let dirs = std::fs::read_dir(root.join(DIR)).into_iter().flatten();
        let dirs = dirs.flatten().map(|entry| entry.path());
        let dirs: Vec<PathBuf> = dirs
            .filter(|dir| dir.is_dir() && !running.contains(&dir))
            .collect();
        let work = Work::Clean {
            root,
            dirs,
            asked: true,
        };
        spawn_job(&mut commands, agent, None, work);
        return;
    };
    let Some(verify) = verify.or_else(|| config.verify.clone()) else {
        let why = "/best-of needs a verify command: --verify \"cmd\", or `verify` in [best_of]";
        notices.write(Notice::error(agent, why));
        return;
    };
    let id = run_id();
    let dir = root.join(DIR).join(&id);
    let candidates = (1..=n)
        .map(|k| Candidate {
            k,
            model: (!models.is_empty())
                .then(|| models.get((k - 1) % models.len()).cloned())
                .flatten(),
            agent: None,
            result: None,
        })
        .collect();
    let work = Work::Setup {
        root: root.clone(),
        dir: dir.clone(),
        n,
    };
    let run = Run {
        id,
        owner: agent,
        task,
        verify,
        auto,
        root,
        dir,
        base: None,
        prefix: String::new(),
        tools: TOOLS
            .iter()
            .filter(|tool| defs.iter().any(|def| def.0.name.as_str() == **tool))
            .filter(|tool| access.get(agent).is_ok_and(|access| access.keeps(tool)))
            .map(|tool| (*tool).to_owned())
            .collect(),
        candidates,
    };
    notices.write(Notice::info(
        agent,
        format!("best-of: {n} candidates, verified by `{}`", run.verify),
    ));
    let run = commands.spawn((Name::new("best-of"), run)).id();
    spawn_job(&mut commands, agent, Some(run), work);
}

/// Runs a job's work off the main thread once its effect started, its
/// processes started as `Processes` say now.
fn start_job(
    started: On<Add<Started>>,
    jobs: Query<&Job>,
    processes: Res<Processes>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let Ok(job) = jobs.get(started.entity) else {
        return;
    };
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    let done = job.work.clone().run(processes.launcher());
    let task = async move {
        match done.await {
            Ok(payload) => Outcome::custom(payload),
            Err(why) => Outcome(Err(ErrorReport::from(ToolExecutionError::other(why)))),
        }
    };
    commands
        .entity(started.entity)
        .insert(Running::answering(pool, &wake, task));
}

/// Git, started as the app's `Processes` say.
struct Git(Launcher);

/// Who git's snapshot commits are by.
const AUTHOR: [(&str, &str); 4] = [
    ("GIT_AUTHOR_NAME", "rig best-of"),
    ("GIT_AUTHOR_EMAIL", "best-of@rig.invalid"),
    ("GIT_COMMITTER_NAME", "rig best-of"),
    ("GIT_COMMITTER_EMAIL", "best-of@rig.invalid"),
];

impl Git {
    /// Runs git `args` in `dir`, with `index` as its index and `stdin` as
    /// its input when given: whether it succeeded, its stdout and its
    /// stderr. Dropping the future kills git.
    async fn output(
        &self,
        dir: &Path,
        args: &[&str],
        index: Option<&Path>,
        stdin: Option<Vec<u8>>,
    ) -> Result<(bool, String, String), String> {
        let failed = |error: String| format!("git {}: {error}", args.join(" "));
        let mut launcher = AUTHOR
            .iter()
            .fold(self.0.clone(), |launcher, (name, value)| {
                launcher.with_env(name, value)
            });
        if let Some(index) = index {
            launcher = launcher.with_env("GIT_INDEX_FILE", index);
        }
        let all = [OsStr::new("-C"), dir.as_os_str()]
            .into_iter()
            .chain(args.iter().map(OsStr::new));
        let how = Spawn {
            stdin,
            merge_stderr: false,
        };
        let mut process = launcher
            .spawn("git", all, how)
            .map_err(|error| failed(error.to_string()))?;
        let _stop = KillOnDrop(process.group());
        let ended = blocking(move || {
            let status = process
                .wait_for(GIT_TIMEOUT)
                .map_err(|error| ToolExecutionError::other(error.to_string()))?
                .ok_or_else(|| ToolExecutionError::timeout("timed out"))?;
            let (mut out, mut err) = (String::new(), String::new());
            for line in process.lines().iter() {
                match line {
                    OutputLine::Out(bytes) => out.push_str(&String::from_utf8_lossy(&bytes)),
                    OutputLine::Err(bytes) => err.push_str(&String::from_utf8_lossy(&bytes)),
                }
            }
            Ok((status.success(), out, err))
        });
        ended.await.map_err(|error| failed(error.to_string()))
    }

    /// Runs git `args` in `dir`, as [`output`](Self::output) does; its
    /// stdout, or why it failed.
    async fn run_with(
        &self,
        dir: &Path,
        args: &[&str],
        index: Option<&Path>,
        stdin: Option<Vec<u8>>,
    ) -> Result<String, String> {
        match self.output(dir, args, index, stdin).await? {
            (true, out, _) => Ok(out),
            (false, _, err) => Err(format!("git {}: {}", args.join(" "), err.trim())),
        }
    }

    /// Runs git `args` in `dir`; its stdout, or why it failed.
    async fn run(&self, dir: &Path, args: &[&str]) -> Result<String, String> {
        self.run_with(dir, args, None, None).await
    }

    /// A commit of `dir`'s whole tree as it is, tracked and untracked files
    /// that are not ignored, on top of `parent`, made with an index of its
    /// own so the user's index is left alone.
    async fn snapshot(&self, dir: &Path, parent: &str) -> Result<String, String> {
        let args = ["rev-parse", "--path-format=absolute", "--git-path", "index"];
        let index = PathBuf::from(self.run(dir, &args).await?.trim());
        let temporary = Removed(index.with_file_name(format!("best-of-index-{}", run_id())));
        // A copy keeps what git knows of the files, so unchanged ones are
        // not read again; a repository with no index starts from none.
        std::fs::copy(&index, &temporary.0).ok();
        let index = Some(temporary.0.as_path());
        self.run_with(dir, &["add", "-A"], index, None).await?;
        let tree = self.run_with(dir, &["write-tree"], index, None).await?;
        let args = [
            "commit-tree",
            tree.trim(),
            "-p",
            parent,
            "-m",
            "best-of snapshot",
        ];
        Ok(self.run(dir, &args).await?.trim().to_owned())
    }
}

/// A file removed when this is dropped, also when its job is cancelled.
struct Removed(PathBuf);

impl Drop for Removed {
    fn drop(&mut self) {
        std::fs::remove_file(&self.0).ok();
    }
}

/// Snapshots the project and adds `n` worktrees of it in `dir`.
async fn setup(git: &Git, root: &Path, dir: &Path, n: usize) -> Result<Value, String> {
    let prefix = git.run(root, &["rev-parse", "--show-prefix"]).await?;
    let parent = dir.parent().unwrap_or(dir);
    std::fs::create_dir_all(parent).map_err(|error| format!("{}: {error}", parent.display()))?;
    // Keeps the worktrees out of the project's own snapshots and status.
    std::fs::write(parent.join(".gitignore"), "*\n").ok();
    let base = git.snapshot(root, "HEAD").await?;
    for k in 1..=n {
        let path = dir.join(k.to_string());
        let path = path.to_string_lossy();
        let added = git
            .run(root, &["worktree", "add", "--detach", &path, &base])
            .await;
        if let Err(why) = added {
            remove_worktrees(git, root, dir).await;
            return Err(why);
        }
    }
    Ok(json!({ "base": base, "prefix": prefix.trim() }))
}

/// Removes the worktrees in the run directory `dir`, and `dir`.
async fn remove_worktrees(git: &Git, root: &Path, dir: &Path) {
    let entries = std::fs::read_dir(dir).into_iter().flatten().flatten();
    for entry in entries {
        let path = entry.path();
        let path = path.to_string_lossy();
        git.run(root, &["worktree", "remove", "--force", &path])
            .await
            .ok();
    }
    std::fs::remove_dir_all(dir).ok();
    git.run(root, &["worktree", "prune"]).await.ok();
}

/// Snapshots a candidate's worktree, measures its diff from `base` and
/// runs the verify command in it, started by `launcher`.
async fn verify(
    git: &Git,
    launcher: Launcher,
    worktree: &Path,
    base: &str,
    command: String,
) -> Result<Value, String> {
    let commit = git.snapshot(worktree, base).await?;
    let numstat = git
        .run(worktree, &["diff", "--numstat", base, &commit])
        .await?;
    let size = numstat
        .lines()
        .flat_map(|line| line.split('\t').take(2))
        .filter_map(|count| count.parse::<usize>().ok())
        .sum();
    let diff = git.run(worktree, &["diff", base, &commit]).await?;
    let shell = Shell {
        launcher,
        cut: Some(OutputLimit {
            lines: 20,
            bytes: MAX_BYTES,
        }),
        ..Shell::default()
    };
    let args = ShellArgs::new(command).timeout_secs(VERIFY_SECS);
    let ran = shell.run(args).await.map_err(|error| error.to_string())?;
    let verdict = Verdict {
        passed: ran.status.success(),
        size,
        commit,
        diff,
    };
    serde_json::to_value(verdict).map_err(|error| error.to_string())
}

/// Merges `commit`, made from `base`, into the project's tree as it is now:
/// a 3-way merge with what the user changed since, applied to the files
/// only when it is clean.
async fn apply(git: &Git, root: &Path, base: &str, commit: &str) -> Result<Value, String> {
    let top = git.run(root, &["rev-parse", "--show-toplevel"]).await?;
    let top = PathBuf::from(top.trim());
    let now = git.snapshot(&top, "HEAD").await?;
    let base_arg = format!("--merge-base={base}");
    let args = [
        "merge-tree",
        "--write-tree",
        "--name-only",
        "--no-messages",
        &base_arg,
        &now,
        commit,
    ];
    let (clean, out, err) = git.output(&top, &args, None, None).await?;
    let mut lines = out.lines();
    let tree = lines.next().unwrap_or_default().trim().to_owned();
    if !clean {
        let files: Vec<&str> = lines.take_while(|line| !line.is_empty()).collect();
        if files.is_empty() {
            return Err(format!("git merge-tree: {}", err.trim()));
        }
        return Err(format!(
            "it conflicts with your newer changes in {}",
            files.join(", ")
        ));
    }
    let patch = git.run(&top, &["diff", "--binary", &now, &tree]).await?;
    if !patch.trim().is_empty() {
        let args = ["apply", "--whitespace=nowarn", "-"];
        git.run_with(&top, &args, None, Some(patch.into_bytes()))
            .await?;
    }
    Ok(json!({ "commit": commit, "tree": tree }))
}

/// A candidate's agent was spawned: it works in the project's directory in
/// its worktree, as its `WorkDir`, before it starts its task.
fn candidate_spawned(
    spawned: On<Add<SpawnedAgent>>,
    requests: Query<(&SpawnedAgent, &CandidateRequest)>,
    mut runs: Query<&mut Run>,
    mut commands: Commands,
) {
    let Ok((&SpawnedAgent(agent), &CandidateRequest { run: run_entity, k })) =
        requests.get(spawned.entity)
    else {
        return;
    };
    let Ok(mut run) = runs.get_mut(run_entity) else {
        return;
    };
    let of = CandidateOf { run: run_entity, k };
    let dir = WorkDir(run.work_dir(k));
    commands.entity(agent).insert((dir, of, WasCandidate));
    if let Some(candidate) = run.candidate(k) {
        candidate.agent = Some(agent);
    }
}

/// The payload of an effect answered as hoped, or why it failed.
fn payload(outcome: &Outcome) -> Result<&Value, String> {
    match &outcome.0 {
        Ok(Answered::Custom { payload }) => Ok(payload),
        Ok(_) => Err("it was answered without running".to_owned()),
        Err(report) => Err(report.message.clone()),
    }
}

/// A job settled: the run goes on.
fn job_done(
    settled: On<Add<Settled>>,
    jobs: Query<(&Job, &Outcome, &ForAgent)>,
    mut runs: Query<&mut Run>,
    answerers: Answerers,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((job, outcome, &ForAgent(agent))) = jobs.get(settled.entity) else {
        return;
    };
    let done = payload(outcome);
    if let Work::Clean { dirs, asked, .. } = &job.work {
        if *asked {
            let text = match done {
                Ok(_) => format!("best-of: removed {} leftover runs", dirs.len()),
                Err(why) => format!("best-of clean: {why}"),
            };
            notices.write(Notice::info(agent, text));
        }
        return;
    }
    let Some((run_entity, mut run)) = job
        .run
        .and_then(|entity| Some((entity, runs.get_mut(entity).ok()?)))
    else {
        return;
    };
    match (&job.work, done) {
        (Work::Setup { .. }, Ok(setup)) => {
            let text = |key: &str| setup.get(key).and_then(Value::as_str).map(str::to_owned);
            run.base = text("base");
            run.prefix = text("prefix").unwrap_or_default();
            for candidate in &run.candidates {
                let mut request = SpawnSubagent::helper(format!("best-of {}", candidate.k))
                    .brief(format!(
                        "{}\n\nWhen you are done, your changes are checked with `{}`.",
                        run.task, run.verify
                    ))
                    .once();
                request.model.clone_from(&candidate.model);
                request.tools = Some(run.tools.clone());
                let k = candidate.k;
                commands.spawn((
                    request,
                    ForAgent(run.owner),
                    ChildOf(run_entity),
                    CandidateRequest { run: run_entity, k },
                ));
            }
        }
        (Work::Setup { .. }, Err(why)) => {
            notices.write(Notice::error(run.owner, format!("best-of: {why}")));
            commands.entity(run_entity).despawn();
        }
        (Work::Verify { k, .. }, done) => {
            let verdict = done.and_then(|payload| {
                serde_json::from_value::<Verdict>(payload.clone())
                    .map_err(|error| error.to_string())
            });
            if let Some(candidate) = run.candidate(*k) {
                candidate.result = Some(verdict);
            }
            decide(run_entity, &run, &answerers, &mut commands, &mut notices);
        }
        (Work::Apply { k, why, .. }, Ok(_)) => {
            let note = note(&run, *k, why);
            let deliver = Deliver::new(run.owner, note, DeliveryMode::Inform);
            commands.trigger(deliver.with_author(Author::plugin("best-of")));
            finish(run_entity, &run, &mut commands);
        }
        (Work::Apply { k, .. }, Err(why)) => {
            let kept = run.worktree(*k);
            let text = format!(
                "best-of: candidate {k} was not applied: {why}. Its worktree stays in {}; \
                 /best-of clean removes it.",
                kept.display()
            );
            notices.write(Notice::error(run.owner, text));
            commands.entity(run_entity).despawn();
        }
        (Work::Clean { .. }, _) => {}
    }
}

/// How `candidate` did, in a few words.
fn summary(candidate: &Candidate) -> String {
    let k = candidate.k;
    match &candidate.result {
        Some(Ok(verdict)) if verdict.passed => {
            format!("candidate {k} passed with {} changed lines", verdict.size)
        }
        Some(Ok(verdict)) => {
            format!(
                "candidate {k} failed verification ({} changed lines)",
                verdict.size
            )
        }
        Some(Err(why)) => format!("candidate {k} failed: {why}"),
        None => format!("candidate {k} did not finish"),
    }
}

/// The note the agent gets about the winner.
fn note(run: &Run, k: usize, why: &str) -> String {
    let others: Vec<String> = run
        .candidates
        .iter()
        .filter(|candidate| candidate.k != k)
        .map(summary)
        .collect();
    let mut text = format!(
        "Best-of-{n} on \"{task}\": candidate {k} won ({why}) and its changes were applied to \
         the working tree. Verified with `{verify}`.",
        n = run.candidates.len(),
        task = run.task,
        verify = run.verify,
    );
    if !others.is_empty() {
        text.push_str(&format!(" Others: {}.", others.join("; ")));
    }
    text
}

/// Removes the run's worktrees and ends it.
fn finish(run_entity: Entity, run: &Run, commands: &mut Commands) {
    let work = Work::Clean {
        root: run.root.clone(),
        dirs: vec![run.dir.clone()],
        asked: false,
    };
    spawn_job(commands, run.owner, None, work);
    commands.entity(run_entity).despawn();
}

/// Once every candidate is done: the winner is applied (`--auto`, or nobody
/// to ask), or the user picks one.
fn decide(
    run_entity: Entity,
    run: &Run,
    answerers: &Answerers,
    commands: &mut Commands,
    notices: &mut MessageWriter<Notice>,
) {
    if run
        .candidates
        .iter()
        .any(|candidate| candidate.result.is_none())
    {
        return;
    }
    let ranked = run.ranked();
    if run.auto || !answerers.can_answer(Some(run.owner)) {
        match ranked.iter().find(|(_, verdict)| verdict.passed) {
            Some((k, verdict)) => {
                let why = format!(
                    "the first to pass verification, {} changed lines",
                    verdict.size
                );
                start_apply(run_entity, run, *k, &verdict.commit, why, commands);
            }
            None => {
                let how: Vec<String> = run.candidates.iter().map(summary).collect();
                let text = format!(
                    "best-of: no candidate passed verification; nothing was applied ({})",
                    how.join("; ")
                );
                notices.write(Notice::error(run.owner, text));
                finish(run_entity, run, commands);
            }
        }
        return;
    }
    let mut body = String::new();
    for candidate in &run.candidates {
        if let Some(Err(why)) = &candidate.result {
            body.push_str(&format!("Candidate {}: failed: {why}\n\n", candidate.k));
        }
    }
    let mut picker =
        Picker::new(format!("Best of {}: apply which?", run.candidates.len())).for_agent(run.owner);
    let mut items = Vec::new();
    for (k, verdict) in &ranked {
        let status = if verdict.passed { "passed" } else { "failed" };
        let label = format!("candidate {k}: {status}, {} changed lines", verdict.size);
        picker = picker.item(label, "");
        items.push(*k);
        let diff: Vec<&str> = verdict.diff.lines().take(DIFF_LINES).collect();
        body.push_str(&format!(
            "Candidate {k} ({status}):\n{}\n\n",
            diff.join("\n")
        ));
    }
    commands
        .spawn((
            picker.body(body.trim_end()),
            PickerOf {
                run: run_entity,
                items,
            },
            ChildOf(run_entity),
        ))
        .observe(picked);
}

/// Spawns the job that applies candidate `k`.
fn start_apply(
    run_entity: Entity,
    run: &Run,
    k: usize,
    commit: &str,
    why: String,
    commands: &mut Commands,
) {
    let Some(base) = run.base.clone() else {
        return;
    };
    let work = Work::Apply {
        k,
        why,
        root: run.root.clone(),
        base,
        commit: commit.to_owned(),
    };
    spawn_job(commands, run.owner, Some(run_entity), work);
}

/// The user picked a candidate, or closed the picker, which discards them
/// all.
fn picked(
    picked: On<Picked>,
    pickers: Query<&PickerOf>,
    runs: Query<&Run>,
    mut commands: Commands,
) {
    let Ok(PickerOf {
        run: run_entity,
        items,
    }) = pickers.get(picked.entity)
    else {
        return;
    };
    let Ok(run) = runs.get(*run_entity) else {
        return;
    };
    let chosen = picked
        .chosen
        .as_ref()
        .and_then(|chosen| chosen.items.first());
    let chosen = chosen.and_then(|index| items.get(*index).copied());
    let commit = chosen.and_then(|k| {
        let candidate = run.candidates.iter().find(|candidate| candidate.k == k)?;
        Some((k, candidate.result.as_ref()?.as_ref().ok()?.commit.clone()))
    });
    match commit {
        Some((k, commit)) => {
            start_apply(
                *run_entity,
                run,
                k,
                &commit,
                "you picked it".to_owned(),
                &mut commands,
            );
        }
        None => finish(*run_entity, run, &mut commands),
    }
}

/// A candidate's request was refused: it fails.
fn spawn_refused(
    settled: On<Add<Settled>>,
    requests: Query<(&CandidateRequest, &Outcome), Without<SpawnedAgent>>,
    mut runs: Query<&mut Run>,
    answerers: Answerers,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((&CandidateRequest { run: run_entity, k }, outcome)) = requests.get(settled.entity)
    else {
        return;
    };
    let Ok(mut run) = runs.get_mut(run_entity) else {
        return;
    };
    let why = payload(outcome)
        .err()
        .unwrap_or_else(|| "not started".to_owned());
    if let Some(candidate) = run.candidate(k) {
        candidate.result = Some(Err(format!("not started: {why}")));
    }
    decide(run_entity, &run, &answerers, &mut commands, &mut notices);
}

/// A candidate's turn ended: an answered one is verified, any other fails.
/// `rig_basics::subagents` retires the candidate, a one-job helper, as it ends.
fn candidate_done(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent)>,
    candidates: Query<&CandidateOf>,
    mut runs: Query<&mut Run>,
    answerers: Answerers,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((outcome, &ForAgent(agent))) = turns.get(settled.entity) else {
        return;
    };
    let Ok(of) = candidates.get(agent) else {
        return;
    };
    let Ok(mut run) = runs.get_mut(of.run) else {
        return;
    };
    let failed = match outcome {
        TurnOutcome::Answered(_) => None,
        TurnOutcome::Failed(why) => Some(why.clone()),
        TurnOutcome::Stopped(why) => Some(why.clone().unwrap_or_else(|| "stopped".to_owned())),
    };
    match (failed, run.base.clone()) {
        (None, Some(base)) => {
            let work = Work::Verify {
                k: of.k,
                worktree: run.worktree(of.k),
                cwd: run.work_dir(of.k),
                base,
                command: run.verify.clone(),
            };
            spawn_job(&mut commands, run.owner, Some(of.run), work);
        }
        (failed, _) => {
            let why = failed.unwrap_or_else(|| "the run has no snapshot".to_owned());
            if let Some(candidate) = run.candidate(of.k) {
                candidate.result = Some(Err(why));
            }
            decide(of.run, &run, &answerers, &mut commands, &mut notices);
        }
    }
}

/// Stopping the agent cancels its runs: the candidates are retired, which
/// stops their turns and commands, the run's effects are cancelled with it,
/// and its worktrees are removed.
fn cancel_on_interrupt(
    interrupt: On<StopTurn>,
    runs: Query<(Entity, &Run)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (run_entity, run) in runs.iter().filter(|(_, run)| run.owner == interrupt.entity) {
        for agent in run
            .candidates
            .iter()
            .filter_map(|candidate| candidate.agent)
        {
            commands.trigger(Retire { entity: agent });
        }
        notices.write(Notice::info(
            run.owner,
            "best-of: cancelled every candidate",
        ));
        finish(run_entity, run, &mut commands);
    }
}

/// Retires each candidate a restart brought back without its run, before
/// the agent loop, so its restored turn never goes on in a worktree nobody
/// waits for.
fn retire_orphans(
    restored: Query<Entity, (Added<WasCandidate>, Without<CandidateOf>)>,
    mut commands: Commands,
) {
    for candidate in &restored {
        commands.trigger(Retire { entity: candidate });
    }
}
