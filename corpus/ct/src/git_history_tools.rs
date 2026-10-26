//! CT-05: read-only git history tools.
//!
//! [`GitHistoryToolsPlugin`] adds four tools that read a repository's
//! history without `shell`: `git_log`, `git_show`, `git_blame` and
//! `git_diff`. Each call is built for where its agent works (its
//! `WorkDir`, else the `ProjectRoot`, in the app's workspace) and starts
//! `git` as the app's `Processes` say, so a sandbox plugin's policy
//! applies.
//!
//! They never change the repository, its index or the working tree: every
//! `git` they start runs with `GIT_OPTIONAL_LOCKS=0`, so none refreshes the
//! index, and they run while another git process holds `index.lock`. They
//! say so in their annotations (`read_only_hint`), which makes each call's
//! footprint `Reads(Anything)`: an approval plugin that lets reads run
//! without asking lets these run too, and the harness neither checkpoints
//! nor previews them.
//!
//! Output is compact: `git_log` gives one line per commit (20 by default),
//! `git_blame` groups consecutive lines by commit with each commit's
//! subject once, and `git_show` and `git_diff` stop after 400 lines,
//! saying how many were left out and how to narrow. A path outside the
//! repository or a revision that does not exist is a short error of its
//! own, not git's stderr.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rig_harness::prelude::*;
use rig_harness::rig_core::completion::ToolAnnotations;
use rig_harness::rig_tools::process::Spawn;
use rig_harness::rig_tools::workspace::{Local, Workspace};
use schemars::JsonSchema;
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// The most lines `git_show` and `git_diff` give, and `git_blame` covers.
pub const MAX_LINES: usize = 400;

/// How many commits `git_log` lists unless asked.
pub const DEFAULT_LIMIT: usize = 20;

/// The most commits `git_log` lists in one call.
pub const MAX_LIMIT: usize = 200;

/// Ends each tool's description.
const RATHER: &str = "Use it to read git history rather than `git` in `shell`: it is \
    read-only, needs no approval and keeps output short.";

/// How long one `git` may run.
const GIT_TIMEOUT: Duration = Duration::from_secs(30);

/// Adds the read-only `git_log`, `git_show`, `git_blame` and `git_diff`
/// tools.
#[derive(Default)]
pub struct GitHistoryToolsPlugin;

impl Plugin for GitHistoryToolsPlugin {
    fn build(&self, app: &mut App) {
        let processes = app.world_mut().get_resource_or_init::<Processes>().clone();
        let repo = Repo {
            workspace: Arc::new(Local),
            base: PathBuf::from("."),
            confine: false,
            processes,
        };
        let options = ToolOptions::default;
        app.add_tool_per_call(GitLog(repo.clone()), options())
            .add_tool_per_call(GitShow(repo.clone()), options())
            .add_tool_per_call(GitBlame(repo.clone()), options())
            .add_tool_per_call(GitDiff(repo), options());
    }
}

/// Each git tool reads its repo as seen from the call's place.
macro_rules! git_tools_per_call {
    ($($tool:ident),*) => {$(
        impl PerCallTool for $tool {
            fn at(&self, place: CallPlace) -> Self {
                Self(self.0.at(place))
            }
        }
    )*};
}

git_tools_per_call!(GitLog, GitShow, GitBlame, GitDiff);

/// Where a call reads: the workspace, the directory it starts from, and how
/// it starts `git`.
#[derive(Clone)]
struct Repo {
    workspace: Arc<dyn Workspace>,
    base: PathBuf,
    confine: bool,
    processes: Processes,
}

impl Repo {
    /// This repo as seen from `place`.
    fn at(&self, place: CallPlace) -> Self {
        Self {
            workspace: place.workspace,
            base: place.base,
            confine: place.confine,
            processes: self.processes.clone(),
        }
    }

    /// What `git args` printed in `dir`, or its error, made readable.
    fn git(&self, dir: &Path, args: &[String]) -> Result<String, ToolExecutionError> {
        let launcher = self
            .processes
            .launcher()
            .with_env("GIT_OPTIONAL_LOCKS", "0")
            .with_env("GIT_LITERAL_PATHSPECS", "1")
            .with_env("GIT_PAGER", "cat")
            .with_env("LC_ALL", "C");
        let mut all: Vec<OsString> = [
            "--no-pager",
            "-c",
            "color.ui=false",
            "-c",
            "core.fsmonitor=false",
        ]
        .into_iter()
        .map(OsString::from)
        .collect();
        all.extend(args.iter().map(OsString::from));
        let process = self
            .workspace
            .spawn(
                &launcher,
                Some(dir),
                std::ffi::OsStr::new("git"),
                &all,
                Spawn::default(),
            )
            .map_err(|error| ToolExecutionError::other(format!("could not start git: {error}")))?;
        let output = process
            .output(Some(GIT_TIMEOUT))
            .map_err(|error| ToolExecutionError::timeout(format!("git did not finish: {error}")))?;
        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).into_owned())
        } else {
            Err(ToolExecutionError::other(readable(
                &String::from_utf8_lossy(&output.stderr),
            )))
        }
    }

    /// The top of the repository the call starts in.
    fn root(&self) -> Result<PathBuf, ToolExecutionError> {
        let args = ["rev-parse", "--show-toplevel"].map(str::to_owned);
        let top = self.git(&self.base, &args).map_err(|_| {
            ToolExecutionError::not_found(format!(
                "{} is not inside a git repository",
                self.base.display()
            ))
        })?;
        Ok(self.workspace.canonical(Path::new(top.trim_end())))
    }

    /// `path` as git names it from `root`, refused when it lies outside.
    fn path(&self, root: &Path, path: &str) -> Result<String, ToolExecutionError> {
        let placed = self.workspace.place(Some(&self.base), self.confine, path)?;
        let inside = placed.strip_prefix(root).map_err(|_| {
            ToolExecutionError::invalid_args(format!(
                "`{path}` is outside the repository at {}",
                root.display()
            ))
        })?;
        let inside = inside.to_string_lossy();
        Ok(if inside.is_empty() {
            ".".to_owned()
        } else {
            inside.into_owned()
        })
    }

    /// `rev` checked to name an object of the repository at `root`.
    fn rev(&self, root: &Path, rev: &str) -> Result<String, ToolExecutionError> {
        let unknown = || {
            ToolExecutionError::not_found(format!(
                "no commit or object `{rev}` in this repository; `git_log` lists commits"
            ))
        };
        if rev.is_empty() || rev.starts_with('-') {
            return Err(unknown());
        }
        let args = ["rev-parse", "--verify", "--quiet"].map(str::to_owned);
        let mut args = args.to_vec();
        args.push(format!("{rev}^{{object}}"));
        self.git(root, &args).map_err(|_| unknown())?;
        Ok(rev.to_owned())
    }

    /// The root and, when given, `path` inside it.
    fn root_and(
        &self,
        path: Option<&str>,
    ) -> Result<(PathBuf, Option<String>), ToolExecutionError> {
        let root = self.root()?;
        let path = path.map(|path| self.path(&root, path)).transpose()?;
        Ok((root, path))
    }
}

/// Git's error output as one short line, without its `fatal:` prefixes.
fn readable(stderr: &str) -> String {
    let lines: Vec<&str> = stderr
        .lines()
        .map(|line| {
            line.trim_start_matches("fatal: ")
                .trim_start_matches("error: ")
                .trim()
        })
        .filter(|line| !line.is_empty() && !line.starts_with("hint:"))
        .collect();
    if lines.is_empty() {
        "git failed".to_owned()
    } else {
        format!("git: {}", lines.join("; "))
    }
}

/// `text` cut after [`MAX_LINES`] lines, saying how many were left out.
fn capped(text: &str) -> String {
    let total = text.lines().count();
    if total <= MAX_LINES {
        return text.trim_end().to_owned();
    }
    let kept: Vec<&str> = text.lines().take(MAX_LINES).collect();
    format!(
        "{}\n… {} more lines left out. Narrow with `path` (a file or directory).",
        kept.join("\n"),
        total.saturating_sub(MAX_LINES)
    )
}

/// The annotations of every tool here: read-only, on a known store.
fn read_only() -> ToolAnnotations {
    ToolAnnotations::default()
        .with_read_only(true)
        .with_open_world(false)
}

/// Runs `work` for `repo` off the main thread.
async fn run(
    repo: &Repo,
    work: impl FnOnce(Repo) -> Result<String, ToolExecutionError> + Send + 'static,
) -> Result<String, ToolExecutionError> {
    let repo = repo.clone();
    blocking(move || work(repo)).await
}

/// `git_log`.
#[derive(Clone)]
struct GitLog(Repo);

/// The arguments of `git_log`.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct LogArgs {
    /// Only commits touching this file or directory.
    path: Option<String>,
    /// Only commits after this date, such as `2 weeks ago` or `2026-01-31`.
    since: Option<String>,
    /// Only commits whose author matches this.
    author: Option<String>,
    /// Only commits whose message matches this.
    grep: Option<String>,
    /// The most commits to list: 20 unless given, at most 200.
    limit: Option<usize>,
}

impl Tool for GitLog {
    const NAME: &'static str = "git_log";
    type Args = LogArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        format!(
            "List commits, newest first, one line each: short hash, date, author, subject. {RATHER}"
        )
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<LogArgs>()
    }

    fn annotations(&self) -> ToolAnnotations {
        read_only()
    }

    async fn call(&self, _: &mut ToolContext, args: LogArgs) -> Result<String, ToolExecutionError> {
        run(&self.0, move |repo| {
            let (root, path) = repo.root_and(args.path.as_deref())?;
            let limit = args.limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT);
            let mut git = vec![
                "log".to_owned(),
                "--no-color".to_owned(),
                "--date=short".to_owned(),
                "--format=%h %ad %an: %s".to_owned(),
                format!("--max-count={}", limit.saturating_add(1)),
            ];
            git.extend(args.since.map(|since| format!("--since={since}")));
            git.extend(args.author.map(|author| format!("--author={author}")));
            git.extend(args.grep.map(|grep| format!("--grep={grep}")));
            git.push("--".to_owned());
            git.extend(path);
            let text = repo.git(&root, &git)?;
            let mut lines: Vec<&str> = text.lines().collect();
            if lines.is_empty() {
                return Ok("No commits match.".to_owned());
            }
            let more = lines.len() > limit;
            lines.truncate(limit);
            let mut out = lines.join("\n");
            if more {
                out.push_str(&format!(
                    "\n… more commits; raise `limit` (now {limit}) or narrow with `since`, `author` or `path`."
                ));
            }
            Ok(out)
        })
        .await
    }
}

/// `git_show`.
#[derive(Clone)]
struct GitShow(Repo);

/// The arguments of `git_show`.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ShowArgs {
    /// The commit (or other object), such as `HEAD~1` or a hash.
    rev: String,
    /// Only this file's or directory's change.
    path: Option<String>,
}

impl Tool for GitShow {
    const NAME: &'static str = "git_show";
    type Args = ShowArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        format!("Show a commit: its message and its change, at most {MAX_LINES} lines. {RATHER}")
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<ShowArgs>()
    }

    fn annotations(&self) -> ToolAnnotations {
        read_only()
    }

    async fn call(
        &self,
        _: &mut ToolContext,
        args: ShowArgs,
    ) -> Result<String, ToolExecutionError> {
        run(&self.0, move |repo| {
            let (root, path) = repo.root_and(args.path.as_deref())?;
            let rev = repo.rev(&root, &args.rev)?;
            let mut git = ["show", "--no-color", "--no-ext-diff", "--date=short"]
                .map(str::to_owned)
                .to_vec();
            git.push(rev);
            git.push("--".to_owned());
            git.extend(path);
            Ok(capped(&repo.git(&root, &git)?))
        })
        .await
    }
}

/// `git_diff`.
#[derive(Clone)]
struct GitDiff(Repo);

/// The arguments of `git_diff`.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct DiffArgs {
    /// The revision to compare from; `HEAD` unless given.
    from: Option<String>,
    /// The revision to compare to; the working tree unless given.
    to: Option<String>,
    /// Only this file's or directory's changes.
    path: Option<String>,
}

impl Tool for GitDiff {
    const NAME: &'static str = "git_diff";
    type Args = DiffArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        format!(
            "Show what changed between two revisions, or from one (`HEAD` unless given) to the working tree, at most {MAX_LINES} lines. {RATHER}"
        )
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<DiffArgs>()
    }

    fn annotations(&self) -> ToolAnnotations {
        read_only()
    }

    async fn call(
        &self,
        _: &mut ToolContext,
        args: DiffArgs,
    ) -> Result<String, ToolExecutionError> {
        run(&self.0, move |repo| {
            let (root, path) = repo.root_and(args.path.as_deref())?;
            let from = repo.rev(&root, args.from.as_deref().unwrap_or("HEAD"))?;
            let to = args.to.map(|to| repo.rev(&root, &to)).transpose()?;
            let mut git = ["diff", "--no-color", "--no-ext-diff"]
                .map(str::to_owned)
                .to_vec();
            git.push(from);
            git.extend(to);
            git.push("--".to_owned());
            git.extend(path);
            let text = repo.git(&root, &git)?;
            Ok(if text.trim().is_empty() {
                "No differences.".to_owned()
            } else {
                capped(&text)
            })
        })
        .await
    }
}

/// `git_blame`.
#[derive(Clone)]
struct GitBlame(Repo);

/// The arguments of `git_blame`.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct BlameArgs {
    /// The file.
    path: String,
    /// The first line, from 1.
    start_line: usize,
    /// The last line, at most 400 lines after `start_line`.
    end_line: usize,
}

impl Tool for GitBlame {
    const NAME: &'static str = "git_blame";
    type Args = BlameArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        format!(
            "Show which commit last changed each line of a range of a file, the lines grouped by commit with each commit's subject. {RATHER}"
        )
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<BlameArgs>()
    }

    fn annotations(&self) -> ToolAnnotations {
        read_only()
    }

    async fn call(
        &self,
        _: &mut ToolContext,
        args: BlameArgs,
    ) -> Result<String, ToolExecutionError> {
        run(&self.0, move |repo| {
            let (start, end) = (args.start_line, args.end_line);
            if start == 0 || end < start {
                return Err(ToolExecutionError::invalid_args(
                    "`start_line` counts from 1 and `end_line` is not before it",
                ));
            }
            if end - start >= MAX_LINES {
                return Err(ToolExecutionError::invalid_args(format!(
                    "blame at most {MAX_LINES} lines at once"
                )));
            }
            let root = repo.root()?;
            let path = repo.path(&root, &args.path)?;
            let git = [
                "blame".to_owned(),
                "--porcelain".to_owned(),
                format!("-L{start},{end}"),
                "--".to_owned(),
                path,
            ];
            Ok(blamed(&repo.git(&root, &git)?))
        })
        .await
    }
}

/// What one commit of a blame says about itself.
#[derive(Default)]
struct Commit {
    author: String,
    summary: String,
}

/// `git blame --porcelain` output as runs of lines by commit, each
/// commit's author and subject given once, at its first run.
fn blamed(porcelain: &str) -> String {
    let mut commits: Vec<(String, Commit)> = Vec::new();
    let mut runs: Vec<(String, Vec<String>)> = Vec::new();
    let mut current: Option<(String, String)> = None;
    for line in porcelain.lines() {
        if let Some(text) = line.strip_prefix('\t') {
            let Some((hash, number)) = current.take() else {
                continue;
            };
            let row = format!("{number:>6}  {text}");
            match runs.last_mut() {
                Some((last, rows)) if *last == hash => rows.push(row),
                _ => runs.push((hash, vec![row])),
            }
            continue;
        }
        let mut words = line.split(' ');
        let first = words.next().unwrap_or_default();
        if first.len() == 40 && first.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            let number = words.nth(1).unwrap_or_default().to_owned();
            if !commits.iter().any(|(hash, _)| hash == first) {
                commits.push((first.to_owned(), Commit::default()));
            }
            current = Some((first.to_owned(), number));
            continue;
        }
        let Some((hash, _)) = &current else { continue };
        let Some((_, commit)) = commits.iter_mut().find(|(known, _)| known == hash) else {
            continue;
        };
        if let Some(author) = line.strip_prefix("author ") {
            author.clone_into(&mut commit.author);
        } else if let Some(summary) = line.strip_prefix("summary ") {
            summary.clone_into(&mut commit.summary);
        }
    }
    let mut shown: Vec<&str> = Vec::new();
    let mut out = Vec::new();
    for (hash, rows) in &runs {
        let short = hash.get(..7).unwrap_or(hash);
        if shown.contains(&hash.as_str()) {
            out.push(format!("{short} (see above)"));
        } else {
            shown.push(hash);
            let commit = commits
                .iter()
                .find(|(known, _)| known == hash)
                .map(|(_, commit)| commit);
            let (author, summary) =
                commit.map_or(("", ""), |c| (c.author.as_str(), c.summary.as_str()));
            out.push(format!("{short} {author}: {summary}"));
        }
        out.extend(rows.iter().cloned());
    }
    out.join("\n")
}
