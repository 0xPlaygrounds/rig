//! OC-14: untested-change nudge from coverage.
//!
//! After a turn in which an agent's own file tools (`edit`, `write`, any
//! tool that says which paths it writes) changed lines of the project's
//! sources, the project's coverage command runs in the background, as
//! `coverage.toml` at the project's root says:
//!
//! ```toml
//! command = "cargo llvm-cov --lcov --output-path {out}"
//! format = "lcov"            # or "cobertura"
//! sources = ["src/**/*.rs"]  # every changed file when left out
//! max_rounds = 1             # follow-up turns per change, 1 by default
//! timeout_secs = 300         # 5 minutes by default
//! ```
//!
//! The lines the agent changed are read from the unified diff each of its
//! calls published (`FilesChanged`), carried through its later edits of
//! the same file, so a line the user changed is never one of them. Those
//! the report counts as instrumented but never run (blank lines, comments
//! and braces aside) start one follow-up turn that lists them with their
//! code and asks for tests; a failing test run lists the failures instead.
//! An agent busy when the result comes gets it with the user's next
//! message. After `max_rounds` follow-ups, a notice says what remains
//! uncovered. A new turn or an interrupt kills a running command.
//! `/coverage` shows the last result.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use bevy_tasks::{IoTaskPool, TaskPool};
use glob::Pattern;
use rig_harness::prelude::*;
use rig_harness::rig_core::tool::{FilesChanged, Scope, ToolExecutionError, Touches};
use rig_harness::rig_tools::process::{KillOnDrop, Spawn};
use rig_harness::rig_tools::workspace::Workspace;
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// The plugin's name: its notes' author, and its attachments' label.
const COVERAGE: &str = "coverage";
/// Failure lines a follow-up lists at most.
const FAILURE_LINES: usize = 20;

/// Runs the project's coverage after an agent's changes and asks it to
/// test the changed lines no test runs.
#[derive(Default)]
pub struct CoverageGapNudgePlugin;

/// The plugin's stage system: in `PrepareSystems::Edit`, a waiting note
/// goes with the user's next message.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CoverageSystems;

impl Plugin for CoverageGapNudgePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .init_resource::<Processes>()
            .register_required_components::<Agent, Coverage>()
            .configure_sets(AgentPass, CoverageSystems.in_set(PrepareSystems::Edit))
            .add_systems(AgentPass, attach_waiting.in_set(CoverageSystems))
            .add_command(
                "coverage",
                "The last coverage result of changed lines",
                show_last,
            )
            .add_observer(track_changes)
            .add_observer(mark_follow_up)
            .add_observer(turn_ended)
            .add_observer(cancel_on_new_turn)
            .add_observer(cancel_on_interrupt)
            .add_observer(finished);
        #[cfg(feature = "tui")]
        status::add(app);
    }
}

/// `coverage.toml`, at the project's root.
#[derive(Deserialize, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct CoverageConfig {
    /// The shell command; `{out}` is where it writes its report.
    pub command: String,
    /// What the report is.
    pub format: ReportFormat,
    /// Globs over project-relative paths that count as source; none
    /// counts every file.
    pub sources: Vec<String>,
    /// Follow-up turns for one change before giving up with a summary.
    pub max_rounds: u32,
    /// How long the command may run, in seconds.
    pub timeout_secs: u64,
}

impl Default for CoverageConfig {
    fn default() -> Self {
        Self {
            command: String::new(),
            format: ReportFormat::Lcov,
            sources: Vec::new(),
            max_rounds: 1,
            timeout_secs: 300,
        }
    }
}

/// The format of a coverage report.
#[derive(Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ReportFormat {
    /// `SF:` / `DA:line,hits` records.
    Lcov,
    /// Cobertura XML: `<class filename>` with `<line number hits>`.
    Cobertura,
}

/// What a coverage run found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Checked {
    /// Nothing to check: no `coverage.toml`, or no changed source line.
    Skipped,
    /// Every changed line that can run was run by a test.
    Covered,
    /// Changed lines no test ran, by file.
    Uncovered(Vec<Gap>),
    /// The tests failed: the lines of their output that say which.
    TestsFailed(Vec<String>),
    /// The command could not run, ran too long, or left no report.
    Failed(String),
}

/// Changed lines of one file that no test ran.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gap {
    /// The file as the agent's tools named it.
    pub path: String,
    /// The file relative to the project.
    pub shown: String,
    /// Each line's number and code.
    pub lines: Vec<(usize, String)>,
}

/// An agent's coverage state.
#[derive(Component, Default, Debug)]
pub struct Coverage {
    /// Whether its coverage command runs.
    pub running: bool,
    /// How many runs ended.
    pub runs: u64,
    /// What the last run found.
    pub last: Option<Checked>,
    /// The lines its tools changed that wait for a check, by file.
    tracked: BTreeMap<String, BTreeSet<usize>>,
    /// Follow-ups sent for the change being checked.
    rounds: u32,
    /// A follow-up that goes with the user's next message.
    waiting: Option<String>,
}

/// On the entity whose task runs the coverage command for this agent.
#[derive(Component)]
struct CoverageRun(Entity);

/// On a message that carries a follow-up, and on the turn it went into.
#[derive(Component)]
struct FollowUp;

/// What a run's task hands back.
struct Ran {
    found: Checked,
    max_rounds: u32,
}

/// Each settled tool call of an agent that names what it writes moves the
/// agent's tracked lines through its diffs and adds the lines it wrote. A
/// call that may write anything (`shell`) is left out: what a snapshot
/// around it saw changed may be the user's.
fn track_changes(
    settled: On<Add<Settled>>,
    calls: Query<(&Published, &Outcome, &ForAgent, Option<&Footprint>), With<ToolRequest>>,
    mut agents: Query<&mut Coverage>,
) {
    let Ok((published, outcome, &ForAgent(agent), footprint)) = calls.get(settled.entity) else {
        return;
    };
    let anything = footprint.is_none_or(|it| matches!(it.0, Touches::Writes(Scope::Anything)));
    let Ok(Some(FilesChanged(changes))) = published.0.get::<FilesChanged>() else {
        return;
    };
    let Ok(mut state) = agents.get_mut(agent) else {
        return;
    };
    if anything || !outcome.tool_result().is_success() {
        return;
    }
    for change in changes {
        match &change.after {
            None => {
                state.tracked.remove(&change.path);
            }
            Some(_) if !change.hunks.is_empty() => {
                let lines = state.tracked.remove(&change.path).unwrap_or_default();
                let moved = through_diff(&lines, &change.unified());
                if !moved.is_empty() {
                    state.tracked.insert(change.path.clone(), moved);
                }
            }
            Some(_) => {}
        }
    }
}

/// `lines` of a file before `diff`, a unified diff of it, as numbered
/// after it (a removed line goes), with the lines it added.
fn through_diff(lines: &BTreeSet<usize>, diff: &str) -> BTreeSet<usize> {
    let mut mapped = HashMap::new();
    let mut removed = BTreeSet::new();
    let mut added = BTreeSet::new();
    // Each hunk's last old line, and how far it moves the lines after it.
    let mut shifts: Vec<(usize, i64)> = Vec::new();
    let (mut old, mut new) = (0, 0);
    for line in diff.lines() {
        if let Some(header) = line.strip_prefix("@@ ") {
            let Some((old_start, old_len, new_start, new_len)) = hunk(header) else {
                continue;
            };
            let last = match old_len {
                0 => old_start,
                len => old_start + len - 1,
            };
            shifts.push((last, new_len as i64 - old_len as i64));
            (old, new) = (old_start, new_start);
            continue;
        }
        if shifts.is_empty() {
            continue;
        }
        match line.chars().next() {
            Some(' ') => {
                mapped.insert(old, new);
                (old, new) = (old + 1, new + 1);
            }
            Some('-') => {
                removed.insert(old);
                old += 1;
            }
            Some('+') => {
                added.insert(new);
                new += 1;
            }
            _ => {}
        }
    }
    let moved = lines.iter().filter_map(|&line| {
        if let Some(&to) = mapped.get(&line) {
            return Some(to);
        }
        if removed.contains(&line) {
            return None;
        }
        let shift: i64 = shifts
            .iter()
            .filter(|(last, _)| line > *last)
            .map(|(_, by)| by)
            .sum();
        usize::try_from(line as i64 + shift).ok()
    });
    moved.chain(added).collect()
}

/// `-a,b +c,d @@`: the old start and length, the new start and length.
fn hunk(header: &str) -> Option<(usize, usize, usize, usize)> {
    let mut parts = header.split_whitespace();
    let range = |part: Option<&str>, sign: char| {
        let (start, len) = match part?.strip_prefix(sign)?.split_once(',') {
            Some((start, len)) => (start.parse().ok()?, len.parse().ok()?),
            None => (part?.strip_prefix(sign)?.parse().ok()?, 1),
        };
        Some((start, len))
    };
    let (old_start, old_len) = range(parts.next(), '-')?;
    let (new_start, new_len) = range(parts.next(), '+')?;
    Some((old_start, old_len, new_start, new_len))
}

/// A follow-up's message, or the user's message that carries one, marks
/// the turn it went into.
fn mark_follow_up(
    delivered: On<Add<DeliveredIn>>,
    messages: Query<&DeliveredIn, With<FollowUp>>,
    mut commands: Commands,
) {
    if let Ok(&DeliveredIn(turn)) = messages.get(delivered.entity) {
        commands.entity(turn).try_insert(FollowUp);
    }
}

/// A turn that answered starts a coverage run for its agent's tracked
/// lines. A turn the user started begins a new count of follow-ups.
fn turn_ended(
    settled: On<Add<Settled>>,
    turns: Query<(&ForAgent, &TurnOutcome, Has<FollowUp>), With<Turn>>,
    mut agents: Query<(&mut Coverage, Option<&WorkDir>)>,
    root: Res<ProjectRoot>,
    workspace: Res<ProjectWorkspace>,
    processes: Res<Processes>,
    scratch: Option<Res<Scratch>>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let Ok((&ForAgent(agent), outcome, follow_up)) = turns.get(settled.entity) else {
        return;
    };
    let Ok((mut state, work_dir)) = agents.get_mut(agent) else {
        return;
    };
    if !follow_up {
        state.rounds = 0;
    }
    if !matches!(outcome, TurnOutcome::Answered(_)) || state.tracked.is_empty() {
        return;
    }
    let workspace = workspace.0.clone();
    let root = work_dir.map_or(&root.0, |dir| &dir.0).clone();
    let scratch = scratch
        .and_then(|it| it.0.clone())
        .filter(|_| workspace.host().is_none());
    let job = Job {
        workspace,
        launcher: processes.launcher(),
        root,
        scratch,
        tracked: state.tracked.clone(),
    };
    state.running = true;
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    commands.spawn((
        CoverageRun(agent),
        Running::spawn(pool, &wake, job.run()),
        Name::new("coverage run"),
    ));
}

/// A run's inputs, off the main thread.
struct Job {
    workspace: Arc<dyn Workspace>,
    launcher: rig_harness::rig_tools::process::Launcher,
    root: PathBuf,
    /// The session's local scratch directory, where the report goes.
    scratch: Option<PathBuf>,
    tracked: BTreeMap<String, BTreeSet<usize>>,
}

/// The changed lines of each source file, by tracked path and shown path.
type Sources = Vec<(String, String, BTreeSet<usize>)>;

impl Job {
    /// Reads `coverage.toml`, runs the command (killed with its group if
    /// this future is dropped), and compares its report with the lines.
    async fn run(self) -> Ran {
        let Job {
            workspace,
            launcher,
            root,
            scratch,
            tracked,
        } = self;
        let ws = workspace.clone();
        let started = blocking(move || Ok(start(&*ws, &launcher, &root, scratch, tracked))).await;
        let (config, sources, out, process) = match started {
            Ok(Ok(Some(started))) => started,
            Ok(Ok(None)) => return ran(Checked::Skipped, 1),
            Ok(Err(why)) => return ran(Checked::Failed(why), 1),
            Err(error) => return ran(Checked::Failed(error.to_string()), 1),
        };
        let max_rounds = config.max_rounds;
        // Dropped with this future, as when the run is called off.
        let _kill = KillOnDrop(process.group());
        let timeout = Duration::from_secs(config.timeout_secs);
        let output = blocking(move || {
            process
                .output(Some(timeout))
                .map_err(|error| ToolExecutionError::other(error.to_string()))
        })
        .await;
        let output = match output {
            Ok(output) => output,
            Err(error) => return ran(Checked::Failed(error.to_string()), max_rounds),
        };
        if !output.status.success() {
            let text = String::from_utf8_lossy(&output.stdout).into_owned();
            return ran(Checked::TestsFailed(failures(&text)), max_rounds);
        }
        let found = blocking(move || Ok(compare(&*workspace, &out, config.format, sources))).await;
        ran(
            found.unwrap_or_else(|error| Checked::Failed(error.to_string())),
            max_rounds,
        )
    }
}

fn ran(found: Checked, max_rounds: u32) -> Ran {
    Ran { found, max_rounds }
}

/// A started run: its settings, the lines it checks, its report's path
/// and its process.
type Started = (
    CoverageConfig,
    Sources,
    PathBuf,
    rig_harness::rig_tools::process::Process,
);

/// Reads the settings, keeps the tracked lines of source files, and
/// starts the command; `None` when there is nothing to check.
fn start(
    workspace: &dyn Workspace,
    launcher: &rig_harness::rig_tools::process::Launcher,
    root: &Path,
    scratch: Option<PathBuf>,
    tracked: BTreeMap<String, BTreeSet<usize>>,
) -> Result<Option<Started>, String> {
    let Ok(text) = workspace.read_text(&root.join("coverage.toml")) else {
        return Ok(None);
    };
    let config: CoverageConfig =
        toml::from_str(&text).map_err(|error| format!("coverage.toml: {error}"))?;
    if config.command.trim().is_empty() {
        return Err("coverage.toml names no command".to_owned());
    }
    let globs = config
        .sources
        .iter()
        .map(|glob| Pattern::new(glob).map_err(|error| format!("coverage.toml: {glob}: {error}")))
        .collect::<Result<Vec<_>, _>>()?;
    let canonical = workspace.canonical(root);
    let sources: Sources = tracked
        .into_iter()
        .filter_map(|(path, lines)| {
            let shown = relative(Path::new(&path), &canonical, root);
            let source = globs.is_empty() || globs.iter().any(|glob| glob.matches(&shown));
            source.then_some((path, shown, lines))
        })
        .collect();
    if sources.is_empty() {
        return Ok(None);
    }
    let extension = match config.format {
        ReportFormat::Lcov => "lcov",
        ReportFormat::Cobertura => "xml",
    };
    let out = match scratch {
        Some(dir) => {
            // A directory of this machine's: the workspace is local.
            std::fs::create_dir_all(&dir).ok();
            dir.join(format!("coverage-report.{extension}"))
        }
        None => root.join(format!(".coverage-report.{extension}")),
    };
    workspace.remove(&out).ok();
    let command = config.command.replace("{out}", &out.display().to_string());
    let args = [OsString::from("-c"), OsString::from(command)];
    let how = Spawn {
        stdin: None,
        merge_stderr: true,
    };
    let process = workspace
        .spawn(launcher, Some(root), "sh".as_ref(), &args, how)
        .map_err(|error| format!("could not start the coverage command: {error}"))?;
    Ok(Some((config, sources, out, process)))
}

/// `path` relative to the project, as `/`-separated text.
fn relative(path: &Path, canonical: &Path, root: &Path) -> String {
    let inside = path
        .strip_prefix(canonical)
        .or_else(|_| path.strip_prefix(root))
        .unwrap_or(path);
    inside.to_string_lossy().replace('\\', "/")
}

/// The lines of a failed run's output that say what failed, else its last
/// lines.
fn failures(output: &str) -> Vec<String> {
    let failed: Vec<String> = output
        .lines()
        .filter(|line| {
            let lower = line.to_lowercase();
            lower.contains("fail") || lower.contains("panicked") || lower.contains("error")
        })
        .take(FAILURE_LINES)
        .map(str::to_owned)
        .collect();
    if !failed.is_empty() {
        return failed;
    }
    let lines: Vec<&str> = output.lines().collect();
    let from = lines.len().saturating_sub(FAILURE_LINES);
    lines.into_iter().skip(from).map(str::to_owned).collect()
}

/// Reads the report and finds the checked lines it counts as instrumented
/// but never run, leaving out lines that cannot run.
fn compare(
    workspace: &dyn Workspace,
    out: &Path,
    format: ReportFormat,
    sources: Sources,
) -> Checked {
    let Ok(report) = workspace.read_text(out) else {
        return Checked::Failed(format!("the command wrote no report at {}", out.display()));
    };
    let hits = match format {
        ReportFormat::Lcov => lcov(&report),
        ReportFormat::Cobertura => cobertura(&report),
    };
    let mut gaps = Vec::new();
    for (path, shown, lines) in sources {
        let Some(file) = hits.iter().find(|(name, _)| same_file(name, &path, &shown)) else {
            // Not in the report: nothing of it was instrumented.
            continue;
        };
        let code = workspace.read_text(Path::new(&path)).unwrap_or_default();
        let code: Vec<&str> = code.lines().collect();
        let missed: Vec<(usize, String)> = lines
            .into_iter()
            .filter(|line| file.1.get(line) == Some(&0))
            .filter_map(|line| {
                let text = code.get(line.checked_sub(1)?)?;
                runnable(text).then(|| (line, (*text).to_owned()))
            })
            .collect();
        if !missed.is_empty() {
            gaps.push(Gap {
                path,
                shown,
                lines: missed,
            });
        }
    }
    match gaps.is_empty() {
        true => Checked::Covered,
        false => Checked::Uncovered(gaps),
    }
}

/// Whether the report's `name` is the file `path`, shown as `shown`: the
/// same absolute path, or a path the project-relative one ends with.
fn same_file(name: &str, path: &str, shown: &str) -> bool {
    let name = name.trim_start_matches("./");
    !name.is_empty() && (name == path || Path::new(shown).ends_with(name))
}

/// Whether a line can run: not blank, a comment, or braces alone.
fn runnable(code: &str) -> bool {
    let code = code.trim();
    let comment = ["//", "#", "/*", "*", "--"]
        .iter()
        .any(|it| code.starts_with(it));
    let punctuation = code.chars().all(|c| "{}()[];,".contains(c));
    !comment && !punctuation
}

/// The hits of each line of each file of an lcov report.
fn lcov(report: &str) -> Vec<(String, HashMap<usize, u64>)> {
    let mut files: Vec<(String, HashMap<usize, u64>)> = Vec::new();
    for line in report.lines().map(str::trim) {
        if let Some(name) = line.strip_prefix("SF:") {
            files.push((name.to_owned(), HashMap::new()));
        } else if let Some(record) = line.strip_prefix("DA:")
            && let Some((_, lines)) = files.last_mut()
        {
            let mut fields = record.split(',');
            let number = fields.next().and_then(|it| it.parse::<usize>().ok());
            let hits = fields.next().and_then(|it| it.parse::<u64>().ok());
            if let (Some(number), Some(hits)) = (number, hits) {
                *lines.entry(number).or_default() += hits;
            }
        }
    }
    files
}

/// The hits of each line of each class's file of a Cobertura report.
fn cobertura(report: &str) -> Vec<(String, HashMap<usize, u64>)> {
    use quick_xml::events::{BytesStart, Event};
    let attribute = |element: &BytesStart, name: &str| {
        let value = element.try_get_attribute(name).ok().flatten()?.value;
        std::str::from_utf8(&value).ok().map(str::to_owned)
    };
    let mut files: Vec<(String, HashMap<usize, u64>)> = Vec::new();
    let mut reader = quick_xml::Reader::from_str(report);
    loop {
        let element = match reader.read_event() {
            Ok(Event::Start(element) | Event::Empty(element)) => element,
            Ok(Event::Eof) | Err(_) => break,
            Ok(_) => continue,
        };
        match element.name().as_ref() {
            b"class" => {
                let name = attribute(&element, "filename").unwrap_or_default();
                files.push((name, HashMap::new()));
            }
            b"line" => {
                let number = attribute(&element, "number").and_then(|it| it.parse::<usize>().ok());
                let hits = attribute(&element, "hits").and_then(|it| it.parse::<u64>().ok());
                if let (Some((_, lines)), Some(number), Some(hits)) =
                    (files.last_mut(), number, hits)
                {
                    *lines.entry(number).or_default() += hits;
                }
            }
            _ => {}
        }
    }
    files
}

/// A run ended: its agent gets a follow-up, a summary, or nothing.
fn finished(
    done: On<Insert<Done<Ran>>>,
    runs: Query<(&CoverageRun, &Done<Ran>)>,
    mut agents: Query<(&mut Coverage, Has<ActiveTurn>)>,
    unsent: Query<(&Incoming, &ForAgent), Without<Outcome>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((&CoverageRun(agent), Done(ran))) = runs.get(done.entity) else {
        return;
    };
    commands.entity(done.entity).despawn();
    let Ok((mut state, busy)) = agents.get_mut(agent) else {
        return;
    };
    state.running = false;
    state.runs += 1;
    state.last = Some(ran.found.clone());
    let more = state.rounds < ran.max_rounds;
    let note = match &ran.found {
        Checked::Uncovered(gaps) if more => {
            state.tracked = gaps
                .iter()
                .map(|gap| {
                    (
                        gap.path.clone(),
                        gap.lines.iter().map(|(line, _)| *line).collect(),
                    )
                })
                .collect();
            Some(uncovered_note(gaps))
        }
        Checked::TestsFailed(lines) if more => Some(failed_note(lines)),
        Checked::Uncovered(gaps) => {
            notices.write(Notice::warn(agent, summary(gaps)));
            None
        }
        Checked::TestsFailed(_) => {
            let text = "The tests still fail after the coverage follow-ups; see /coverage.";
            notices.write(Notice::warn(agent, text));
            None
        }
        Checked::Failed(why) => {
            notices.write(Notice::warn(agent, format!("coverage: {why}")));
            None
        }
        Checked::Skipped | Checked::Covered => None,
    };
    let Some(text) = note else {
        state.tracked.clear();
        state.rounds = 0;
        return;
    };
    state.rounds += 1;
    let users_message = unsent.iter().any(|(message, for_agent)| {
        for_agent.0 == agent && message.author.kind == AuthorKind::User
    });
    if busy || users_message {
        state.waiting = Some(text);
        return;
    }
    let message = Incoming::new(text, DeliveryMode::Queue).with_author(Author::plugin(COVERAGE));
    commands.spawn((message, ForAgent(agent), FollowUp));
}

/// The follow-up for changed lines no test ran.
fn uncovered_note(gaps: &[Gap]) -> String {
    let mut text = String::from(
        "Coverage after your last changes: these lines you added or changed are not run by any test.\n",
    );
    for gap in gaps {
        for range in ranges(&gap.lines) {
            let (first, last) = (range.first().map(|it| it.0), range.last().map(|it| it.0));
            let span = match (first, last) {
                (Some(first), Some(last)) if first != last => format!("lines {first}-{last}"),
                (Some(first), _) => format!("line {first}"),
                _ => continue,
            };
            text.push_str(&format!("\n{}, {span}:\n", gap.shown));
            for (line, code) in range {
                text.push_str(&format!("{line:>5} | {code}\n"));
            }
        }
    }
    text.push_str("\nAdd tests that exercise them, or say in a sentence why they need none.");
    text
}

/// `lines` cut where their numbers skip.
fn ranges(lines: &[(usize, String)]) -> Vec<&[(usize, String)]> {
    let mut ranges = Vec::new();
    let mut rest = lines;
    while !rest.is_empty() {
        let mut end = 1;
        while let (Some(a), Some(b)) = (rest.get(end - 1), rest.get(end))
            && b.0 == a.0 + 1
        {
            end += 1;
        }
        let (range, after) = rest.split_at(end);
        ranges.push(range);
        rest = after;
    }
    ranges
}

/// The follow-up for a failing test run.
fn failed_note(lines: &[String]) -> String {
    format!(
        "The tests failed while measuring coverage after your last changes, so coverage \
         says nothing yet. The failures:\n\n{}\n\nFix them, or say why they fail.",
        lines.join("\n")
    )
}

/// `3 changed lines in parser.rs remain uncovered`.
fn summary(gaps: &[Gap]) -> String {
    let parts: Vec<String> = gaps
        .iter()
        .map(|gap| {
            let name = Path::new(&gap.shown)
                .file_name()
                .map_or(gap.shown.clone(), |name| {
                    name.to_string_lossy().into_owned()
                });
            match gap.lines.len() {
                1 => format!("1 changed line in {name}"),
                n => format!("{n} changed lines in {name}"),
            }
        })
        .collect();
    let lines: usize = gaps.iter().map(|gap| gap.lines.len()).sum();
    let verb = if lines == 1 { "remains" } else { "remain" };
    format!("{} {verb} uncovered", parts.join(", "))
}

/// A follow-up that waits goes with the user's next message.
fn attach_waiting(
    mut typed: Query<(Entity, &mut Incoming, &ForAgent), (Added<Admitted>, IncomingMessages)>,
    mut agents: Query<&mut Coverage>,
    mut commands: Commands,
) {
    for (entity, mut message, &ForAgent(agent)) in &mut typed {
        let users = message.author.kind == AuthorKind::User
            && message.mode != DeliveryMode::Note
            && message.command().is_none();
        let Ok(mut state) = agents.get_mut(agent) else {
            continue;
        };
        let Some(text) = state.waiting.take().filter(|_| users) else {
            continue;
        };
        let attachment = Attachment {
            label: COVERAGE.to_owned(),
            content: UserContent::text(text),
        };
        message.parts.push(MessagePart::Attachment(attachment));
        commands.entity(entity).insert(FollowUp);
    }
}

/// A new turn calls off its agent's coverage run.
fn cancel_on_new_turn(
    added: On<Add<TurnOf>>,
    turns: Query<&TurnOf>,
    runs: Query<(Entity, &CoverageRun)>,
    mut agents: Query<&mut Coverage>,
    mut commands: Commands,
) {
    if let Ok(&TurnOf(agent)) = turns.get(added.entity) {
        cancel(agent, &runs, &mut agents, &mut commands);
    }
}

/// An interrupt calls off its agent's coverage run.
fn cancel_on_interrupt(
    interrupt: On<Interrupt>,
    runs: Query<(Entity, &CoverageRun)>,
    mut agents: Query<&mut Coverage>,
    mut commands: Commands,
) {
    cancel(interrupt.entity, &runs, &mut agents, &mut commands);
}

/// Despawns `agent`'s run, which drops its task and kills its command.
fn cancel(
    agent: Entity,
    runs: &Query<(Entity, &CoverageRun)>,
    agents: &mut Query<&mut Coverage>,
    commands: &mut Commands,
) {
    for (run, _) in runs.iter().filter(|(_, run)| run.0 == agent) {
        commands.entity(run).despawn();
        if let Ok(mut state) = agents.get_mut(agent) {
            state.running = false;
        }
    }
}

/// `/coverage`: the last result.
fn show_last(
    In(args): In<CommandArgs>,
    agents: Query<&Coverage>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(state) = agents.get(args.agent) else {
        return;
    };
    let text = match (&state.last, state.running) {
        (_, true) => "Coverage is running…".to_owned(),
        (None, false) => "No coverage result yet.".to_owned(),
        (Some(Checked::Skipped), _) => {
            "Nothing was checked: no coverage.toml, or no changed source lines.".to_owned()
        }
        (Some(Checked::Covered), _) => "Every changed line that can run is covered.".to_owned(),
        (Some(Checked::Uncovered(gaps)), _) => {
            format!("{}.\n\n{}", summary(gaps), uncovered_note(gaps))
        }
        (Some(Checked::TestsFailed(lines)), _) => failed_note(lines),
        (Some(Checked::Failed(why)), _) => format!("The coverage run failed: {why}"),
    };
    notices.write(Notice::info(args.agent, text));
}

/// `coverage running…` in the status line while the command runs.
#[cfg(feature = "tui")]
mod status {
    use rig_harness::prelude::*;
    use rig_tui::{Side, StatusItem, StatusItems, StatusSystems, Tone};

    use super::Coverage;

    /// After the agents at work; gone early when the line is narrow.
    const ITEM: StatusItem = StatusItem::new("coverage", Side::Left, 75, 9);

    pub(super) fn add(app: &mut App) {
        app.add_systems(PostUpdate, show.in_set(StatusSystems::Show));
    }

    fn show(mut agents: Query<(&Coverage, &mut StatusItems), Changed<Coverage>>) {
        for (state, mut items) in &mut agents {
            let item = match state.running {
                true => ITEM.says("coverage running…", Tone::Yellow),
                false => ITEM.says("", Tone::Dim),
            };
            items.show(item);
        }
    }
}
