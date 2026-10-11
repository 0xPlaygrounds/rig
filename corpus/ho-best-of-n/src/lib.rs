//! Plugin corpus (held-out): H16: Best-of-N attempts in git worktrees (spec
//! `H16-best-of-n.md`).
//!
//! `/best-of 3 --verify "cargo test" fix the flaky test in net.rs` makes 3
//! git worktrees of the project as it is now (HEAD plus the uncommitted
//! changes, as a snapshot commit) under `.rig/best-of/<id>/<k>`, and spawns
//! one one-job helper agent per worktree with the task. A candidate's file
//! and shell tools work in its worktree: a Prepare system resolves the
//! `path` of `read`, `write`, `edit` and `search`, and `shell`'s `cwd`,
//! against the worktree, and a Gate system refuses a path that leaves it.
//! When a candidate's turn ends, its tree is snapshotted, its diff measured
//! and the verify command run in it. Once all are done, a picker lists them
//! (passing first, then the smaller diff) with their diffs; `--auto`, or a
//! run nobody answers, takes the first passing one. The winner is merged
//! into the project's tree with a 3-way merge against the user's changes
//! since the start, refused on a conflict, and the agent gets a note naming
//! the winner. `--models a,b,c` gives each candidate its model; N is capped
//! at 5. Stopping the agent cancels every candidate. Worktrees are removed
//! after an apply or a discard; `/best-of clean` removes leftovers.
//!
//! Every git step and verify run is an effect of the agent (`best-of-*`),
//! recorded, gateable and cancelled with its run.

use std::path::{Component as Part, Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use bevy_tasks::{IoTaskPool, TaskPool};
use rig_harness::prelude::*;
use rig_harness::rig_core::effect::{EffectKind, Outcome as Answered};
use rig_harness::rig_core::error::ErrorReport;
use rig_subagents::{SpawnSubagent, SpawnedAgent, SubagentsPlugin};
use rig_tools::process::Launcher;
use rig_tools::{Cut, MAX_BYTES, Shell, ShellArgs, blocking};
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
/// The tools a candidate gets, those of them its parent has.
const TOOLS: [&str; 5] = ["read", "write", "edit", "search", "shell"];
/// Lines of each diff the picker shows.
const DIFF_LINES: usize = 40;
/// How long the verify command may run.
const VERIFY_SECS: u64 = 600;

const HELP: &str = "N agents try a task in git worktrees; the verified winner is applied: \
    /best-of [N] [--verify \"cmd\"] [--auto] [--models a,b] <task>, or /best-of clean";

/// The `[best_of]` table of the settings: `verify = "cargo test"`, the
/// command a run verifies its candidates with unless `--verify` says.
#[derive(Resource, Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BestOfConfig {
    pub verify: Option<String>,
}

/// The best-of plugin's Prepare system that resolves candidates' paths in
/// their worktree, for a plugin that edits tool arguments to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorktreeSystems;

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
            .configure_sets(AgentPass, WorktreeSystems.in_set(Stage::Prepare))
            .add_systems(
                AgentPass,
                (
                    in_worktree.in_set(WorktreeSystems),
                    keep_inside.in_set(Stage::Gate),
                    mark_candidates.in_set(Stage::Review),
                ),
            )
            .add_systems(Update, retire_orphans.before(AgentLoop))
            .add_observer(start_job)
            .add_observer(job_done)
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
    /// How git and the verify command start, as `Processes` said when the
    /// run started.
    launcher: Launcher,
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

/// On a candidate agent: its run, its worktree and the directory its paths
/// resolve from.
#[derive(Component, Debug)]
pub struct CandidateOf {
    pub run: Entity,
    pub k: usize,
    worktree: PathBuf,
    cwd: PathBuf,
}

/// Saved on a candidate, so one a restart brings back without its run is
/// retired.
#[derive(Component, Reflect, Default)]
#[reflect(Component, Saved, Default)]
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
        launcher: Launcher,
    },
    /// Snapshots candidate `k`'s worktree and runs `command` in it.
    Verify {
        k: usize,
        worktree: PathBuf,
        cwd: PathBuf,
        base: String,
        command: String,
        launcher: Launcher,
    },
    /// Merges candidate `k`'s `commit` into the project's tree.
    Apply {
        k: usize,
        why: String,
        root: PathBuf,
        base: String,
        commit: String,
        launcher: Launcher,
    },
    /// Removes the worktrees of the run directories `dirs`.
    Clean {
        root: PathBuf,
        dirs: Vec<PathBuf>,
        launcher: Launcher,
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
    processes: Res<Processes>,
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
    let launcher = processes.launcher();
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
            launcher,
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
        launcher: launcher.clone(),
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
        launcher,
        tools: TOOLS
            .iter()
            .filter(|tool| defs.iter().any(|def| def.0.name.as_str() == **tool))
            .filter(|tool| access.get(agent).is_ok_and(|access| access.allows(tool)))
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

/// Runs a job's work off the main thread once its effect started.
fn start_job(
    started: On<Add<Started>>,
    jobs: Query<&Job>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let Ok(job) = jobs.get(started.entity) else {
        return;
    };
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    let work = job.work.clone();
    let task = async move {
        let done = match work {
            Work::Verify {
                worktree,
                cwd,
                base,
                command,
                launcher,
                ..
            } => verify(worktree, cwd, base, command, launcher).await,
            work => blocking(move || Ok(run_blocking(work)))
                .await
                .unwrap_or_else(|error| Err(error.to_string())),
        };
        match done {
            Ok(payload) => Outcome::custom(payload),
            Err(why) => Outcome(Err(ErrorReport::from(ToolExecutionError::other(why)))),
        }
    };
    commands
        .entity(started.entity)
        .insert(Running::answering(pool, &wake, task));
}

/// The work of a setup, apply or clean job.
fn run_blocking(work: Work) -> Result<Value, String> {
    match work {
        Work::Setup {
            root,
            dir,
            n,
            launcher,
        } => setup(&Git(launcher), &root, &dir, n),
        Work::Apply {
            root,
            base,
            commit,
            launcher,
            ..
        } => apply(&Git(launcher), &root, &base, &commit),
        Work::Clean {
            root,
            dirs,
            launcher,
            ..
        } => {
            let git = Git(launcher);
            for dir in &dirs {
                remove_worktrees(&git, &root, dir);
            }
            git.run(&root, &["worktree", "prune"]).ok();
            Ok(json!({ "removed": dirs.len() }))
        }
        Work::Verify { .. } => Err("verify runs on its own".to_owned()),
    }
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
    fn command(&self, dir: &Path, args: &[&str]) -> Command {
        let mut command = self.0.command("git");
        command.arg("-C").arg(dir).args(args).stdin(Stdio::null());
        command.envs(AUTHOR);
        command
    }

    /// Runs git in `dir`; its output, or why it failed.
    fn run(&self, dir: &Path, args: &[&str]) -> Result<String, String> {
        let (ok, out, err) = self.output(self.command(dir, args), args)?;
        if ok {
            Ok(out)
        } else {
            Err(format!("git {}: {}", args.join(" "), err.trim()))
        }
    }

    /// Runs `command` (git `args`): whether it succeeded, its stdout and
    /// its stderr.
    fn output(
        &self,
        mut command: Command,
        args: &[&str],
    ) -> Result<(bool, String, String), String> {
        let output = command
            .output()
            .map_err(|error| format!("git {}: {error}", args.join(" ")))?;
        let text = |bytes: &[u8]| String::from_utf8_lossy(bytes).into_owned();
        Ok((
            output.status.success(),
            text(&output.stdout),
            text(&output.stderr),
        ))
    }

    /// A commit of `dir`'s whole tree as it is, tracked and untracked files
    /// that are not ignored, on top of `parent`, made with an index of its
    /// own so the user's index is left alone.
    fn snapshot(&self, dir: &Path, parent: &str) -> Result<String, String> {
        let index = self.run(
            dir,
            &["rev-parse", "--path-format=absolute", "--git-path", "index"],
        )?;
        let index = PathBuf::from(index.trim());
        let temporary = index.with_file_name(format!("best-of-index-{}", run_id()));
        // A copy keeps what git knows of the files, so unchanged ones are
        // not read again; a repository with no index starts from none.
        std::fs::copy(&index, &temporary).ok();
        let tree = self.write_tree(dir, &temporary);
        std::fs::remove_file(&temporary).ok();
        let tree = tree?;
        let commit = self.run(
            dir,
            &["commit-tree", &tree, "-p", parent, "-m", "best-of snapshot"],
        )?;
        Ok(commit.trim().to_owned())
    }

    /// The tree of every file of `dir` that is not ignored, added to the
    /// index `index`.
    fn write_tree(&self, dir: &Path, index: &Path) -> Result<String, String> {
        let with_index = |args: &[&str]| {
            let mut command = self.command(dir, args);
            command.env("GIT_INDEX_FILE", index);
            match self.output(command, args)? {
                (true, out, _) => Ok(out),
                (false, _, err) => Err(format!("git {}: {}", args.join(" "), err.trim())),
            }
        };
        with_index(&["add", "-A"])?;
        Ok(with_index(&["write-tree"])?.trim().to_owned())
    }
}

/// Snapshots the project and adds `n` worktrees of it in `dir`.
fn setup(git: &Git, root: &Path, dir: &Path, n: usize) -> Result<Value, String> {
    let prefix = git.run(root, &["rev-parse", "--show-prefix"])?;
    let parent = dir.parent().unwrap_or(dir);
    std::fs::create_dir_all(parent).map_err(|error| format!("{}: {error}", parent.display()))?;
    // Keeps the worktrees out of the project's own snapshots and status.
    std::fs::write(parent.join(".gitignore"), "*\n").ok();
    let base = git.snapshot(root, "HEAD")?;
    for k in 1..=n {
        let path = dir.join(k.to_string());
        let path = path.to_string_lossy();
        let added = git.run(root, &["worktree", "add", "--detach", &path, &base]);
        if let Err(why) = added {
            remove_worktrees(git, root, dir);
            return Err(why);
        }
    }
    Ok(json!({ "base": base, "prefix": prefix.trim() }))
}

/// Removes the worktrees in the run directory `dir`, and `dir`.
fn remove_worktrees(git: &Git, root: &Path, dir: &Path) {
    let entries = std::fs::read_dir(dir).into_iter().flatten().flatten();
    for entry in entries {
        let path = entry.path();
        let path = path.to_string_lossy();
        git.run(root, &["worktree", "remove", "--force", &path])
            .ok();
    }
    std::fs::remove_dir_all(dir).ok();
    git.run(root, &["worktree", "prune"]).ok();
}

/// Snapshots a candidate's worktree, measures its diff from `base` and
/// runs the verify command in it.
async fn verify(
    worktree: PathBuf,
    cwd: PathBuf,
    base: String,
    command: String,
    launcher: Launcher,
) -> Result<Value, String> {
    let git = Git(launcher.clone());
    let measured = blocking(move || {
        let commit = git
            .snapshot(&worktree, &base)
            .map_err(ToolExecutionError::other)?;
        let measure = || -> Result<(usize, String), String> {
            let numstat = git.run(&worktree, &["diff", "--numstat", &base, &commit])?;
            let size = numstat
                .lines()
                .flat_map(|line| line.split('\t').take(2))
                .filter_map(|count| count.parse::<usize>().ok())
                .sum();
            let diff = git.run(&worktree, &["diff", &base, &commit])?;
            Ok((size, diff))
        };
        let (size, diff) = measure().map_err(ToolExecutionError::other)?;
        Ok((commit, size, diff))
    });
    let (commit, size, diff) = measured.await.map_err(|error| error.to_string())?;
    let shell = Shell {
        launcher: launcher.in_dir(cwd),
        spill: None,
        cut: Some(Cut {
            lines: 20,
            bytes: MAX_BYTES,
        }),
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
fn apply(git: &Git, root: &Path, base: &str, commit: &str) -> Result<Value, String> {
    let top = git.run(root, &["rev-parse", "--show-toplevel"])?;
    let top = PathBuf::from(top.trim());
    let now = git.snapshot(&top, "HEAD")?;
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
    let (clean, out, err) = git.output(git.command(&top, &args), &args)?;
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
    let patch = git.run(&top, &["diff", "--binary", &now, &tree])?;
    if !patch.trim().is_empty() {
        let args = ["apply", "--whitespace=nowarn", "-"];
        let mut command = git.command(&top, &args);
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = command
            .spawn()
            .map_err(|error| format!("git apply: {error}"))?;
        if let Some(mut stdin) = child.stdin.take() {
            use std::io::Write as _;
            stdin
                .write_all(patch.as_bytes())
                .map_err(|error| format!("git apply: {error}"))?;
        }
        let output = child
            .wait_with_output()
            .map_err(|error| format!("git apply: {error}"))?;
        if !output.status.success() {
            let err = String::from_utf8_lossy(&output.stderr);
            return Err(format!("git apply: {}", err.trim()));
        }
    }
    Ok(json!({ "commit": commit, "tree": tree }))
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
                    .brief(run.task.clone())
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
            let deliver = Deliver::new(run.owner, note, DeliveryMode::Note);
            commands.trigger(deliver.with_origin(Origin::plugin("best-of")));
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
        launcher: run.launcher.clone(),
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
        launcher: run.launcher.clone(),
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
    let chosen = match picked.chosen {
        Some(Chosen::Item(index)) => items.get(index).copied(),
        _ => None,
    };
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

/// Marks each candidate as its request is answered, before it starts its
/// task: where its tools work, what it is told, and its run.
fn mark_candidates(
    spawned: Query<(&SpawnedAgent, &CandidateRequest), Unreviewed>,
    mut runs: Query<&mut Run>,
    mut commands: Commands,
) {
    for (&SpawnedAgent(agent), &CandidateRequest { run: run_entity, k }) in &spawned {
        let Ok(mut run) = runs.get_mut(run_entity) else {
            continue;
        };
        let worktree = run.worktree(k);
        let cwd = worktree.join(&run.prefix);
        let role = format!(
            "You are candidate {k} of {n}: other agents try the same task in their own copies \
             of the project. Your copy is the git worktree {worktree}; your working directory \
             is {cwd}, where relative paths and `shell` commands resolve. Paths outside the \
             worktree are refused. Do the whole task there; when you are done, your changes \
             are checked with `{verify}`. End with a short summary of what you changed.",
            n = run.candidates.len(),
            worktree = worktree.display(),
            cwd = cwd.display(),
            verify = run.verify,
        );
        let section = PromptSection::new(PromptSection::ORDER_ROLE, "best_of_role", role);
        commands.spawn((Name::new("best_of_role"), section, SectionOf(agent)));
        commands.entity(agent).insert((
            CandidateOf {
                run: run_entity,
                k,
                worktree,
                cwd,
            },
            WasCandidate,
        ));
        if let Some(candidate) = run.candidate(k) {
            candidate.agent = Some(agent);
        }
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
/// rig-subagents retires the candidate, a one-job helper, as it ends.
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
                worktree: of.worktree.clone(),
                cwd: of.cwd.clone(),
                base,
                command: run.verify.clone(),
                launcher: run.launcher.clone(),
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
    interrupt: On<Interrupt>,
    runs: Query<(Entity, &Run)>,
    mut commit: Commit,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (run_entity, run) in runs.iter().filter(|(_, run)| run.owner == interrupt.entity) {
        for agent in run
            .candidates
            .iter()
            .filter_map(|candidate| candidate.agent)
        {
            commit.retire(agent);
        }
        notices.write(Notice::info(
            run.owner,
            "best-of: cancelled every candidate",
        ));
        finish(run_entity, run, &mut commands);
    }
}

/// Resolves the `path` of a candidate's file tools, and `shell`'s `cwd`,
/// from its working directory in its worktree, as each call is admitted.
fn in_worktree(
    mut calls: Query<(&mut ToolRequest, &ForAgent), Added<Admitted>>,
    candidates: Query<&CandidateOf>,
) {
    for (mut request, &ForAgent(agent)) in &mut calls {
        let Ok(candidate) = candidates.get(agent) else {
            continue;
        };
        let function = &mut request.call.function;
        let (key, default) = match function.name.as_str() {
            "shell" => ("cwd", Some(".")),
            "search" => ("path", Some(".")),
            "read" | "write" | "edit" => ("path", None),
            _ => continue,
        };
        let named = function
            .arguments
            .get(key)
            .and_then(Value::as_str)
            .or(default);
        // A missing `path` is the tool's to refuse.
        let Some(named) = named else {
            continue;
        };
        let resolved = normalize(&candidate.cwd.join(named));
        let resolved = resolved.to_string_lossy().into_owned();
        function
            .arguments
            .insert(key.to_owned(), Value::String(resolved));
    }
}

/// Refuses a candidate's call whose path leaves its worktree.
fn keep_inside(
    calls: Query<(Entity, &ToolRequest, &ForAgent), Undecided>,
    candidates: Query<&CandidateOf>,
    mut commands: Commands,
) {
    for (call, request, &ForAgent(agent)) in &calls {
        let Ok(candidate) = candidates.get(agent) else {
            continue;
        };
        let function = &request.call.function;
        let key = match function.name.as_str() {
            "shell" => "cwd",
            "read" | "write" | "edit" | "search" => "path",
            _ => continue,
        };
        let Some(named) = function.arguments.get(key).and_then(Value::as_str) else {
            continue;
        };
        if !normalize(Path::new(named)).starts_with(&candidate.worktree) {
            let why = format!(
                "{named} is outside your worktree {}; work only inside it",
                candidate.worktree.display()
            );
            commands.entity(call).insert_if_new(Outcome::refused(why));
        }
    }
}

/// `path` with `.` and `..` taken out, without following links.
fn normalize(path: &Path) -> PathBuf {
    let mut normal = PathBuf::new();
    for part in path.components() {
        match part {
            Part::CurDir => {}
            Part::ParentDir => {
                normal.pop();
            }
            part => normal.push(part),
        }
    }
    normal
}

/// Retires each candidate a restart brought back without its run, before
/// the agent loop, so its restored turn never goes on in a worktree nobody
/// waits for.
fn retire_orphans(
    restored: Query<Entity, (Added<WasCandidate>, Without<CandidateOf>)>,
    mut commit: Commit,
) {
    for candidate in &restored {
        commit.retire(candidate);
    }
}
