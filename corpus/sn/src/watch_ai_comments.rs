//! Watch mode: the user writes a comment ending in `AI!` (a change) or
//! `AI?` (a question) in their own editor and saves, and the primary agent
//! gets a turn with every AI comment of the project and the code around
//! each. A comment ending in just `AI` is context and never starts one.
//!
//! `/watch on|off`, or `rig --watch`, watches the project's files, without
//! what `.gitignore` leaves out. A save that settled for [`SETTLE`] and holds
//! a trigger queues its file once (its newest content wins), and the queue
//! runs one turn per file whenever the agent is idle. A file left as a
//! file tool wrote it (the [`FilesChanged`] of a call that names its
//! paths), or as this plugin cleaned it, never triggers. A `shell` call's
//! [`FilesChanged`] is a checkpoint: what changed while it ran, the user's
//! saves included, not who changed it. So it is not taken as rig's; the
//! clean-up at the turn's end, and the queued file read again before its
//! turn starts, keep the agent's own `shell` writes in a watch turn from
//! triggering. A question's turn is question only: its calls that may change
//! anything are refused. When a change's turn ends, or a question's turn
//! answers, its file's trigger comments and the context comments it was
//! given are taken out of the files. The status line says
//! `watching` and how many files wait. The watch keeps a headless run going
//! ([`KeepRunning`]), so `rig --watch --print` runs until interrupted and
//! `rig_basics::print` prints each answer.

use std::collections::BTreeSet;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rig_basics::print::KeepRunning;
use rig_harness::prelude::*;
use rig_harness::rig_core::tool::{ContentHash, FilesChanged, Scope, Touches, content_hash};
use rig_harness::rig_tools::workspace::Workspace;
use rig_tui::{AppStatus, Side, StatusItem, StatusSystems, Tone};

/// The command and the option that turn watching on.
pub const WATCH: &str = "watch";

/// How long a save settles before its file is read.
pub const SETTLE: Duration = Duration::from_millis(500);

/// What a refused call of a question's turn is told.
pub const QUESTION_ONLY: &str =
    "question only: this turn answers an `AI?` comment and changes no files";

/// Lines of code shown before and after each comment.
const AROUND: usize = 10;

/// The largest file read for comments.
const MAX_BYTES: u64 = 1 << 20;

/// After the reload build's item; goes early when the line is narrow.
const ITEM: StatusItem = StatusItem::new("sn-watch-ai-comments", Side::Left, 95, 9);

/// The plugin's name, as it decides calls and sends messages.
const NAME: &str = "watch-ai-comments";

/// The plugin's Gate system, which refuses the calls of a question's turn
/// that may change anything, for others to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WatchAiCommentsSystems;

/// Watches the project for AI comments; see the [crate] docs.
#[derive(Default)]
pub struct WatchAiCommentsPlugin;

impl Plugin for WatchAiCommentsPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<rig_harness::watch::WatchPlugin>() {
            app.add_plugins(rig_harness::watch::WatchPlugin);
        }
        if !app.is_plugin_added::<rig_harness::clock::ClockPlugin>() {
            app.add_plugins(rig_harness::clock::ClockPlugin);
        }
        let flagged = app
            .world()
            .get_resource::<Invoked>()
            .is_some_and(|invoked| invoked.flag(WATCH));
        app.init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .init_resource::<AppStatus>()
            .insert_resource(Watching {
                on: flagged,
                ..Watching::default()
            })
            .add_flag(Flag::new(
                WATCH,
                "Act on AI comments saved in the project's files",
            ))
            .add_command(
                WATCH,
                "Act on `AI!` and `AI?` comments saved in the project: /watch on, /watch off",
                watch,
            )
            .configure_sets(AgentPass, WatchAiCommentsSystems.in_set(Stage::Gate))
            .add_systems(AgentPass, question_only.in_set(WatchAiCommentsSystems))
            .add_systems(
                Update,
                (arm, take_changes, read_saves, start_turn)
                    .chain()
                    .before(AgentLoop),
            )
            .add_systems(
                PostUpdate,
                show.in_set(StatusSystems::Show)
                    .run_if(resource_changed::<Watching>),
            )
            .add_observer(note_writes)
            .add_observer(mark_turn)
            .add_observer(clean_up);
    }
}

/// Where watching stands.
#[derive(Resource, Default, Debug)]
struct Watching {
    /// Whether the project is watched.
    on: bool,
    /// The entity with the watch, once armed.
    watch: Option<Entity>,
    /// Whether the watch listed the files there when armed.
    primed: bool,
    /// The files the watch reported, which may hold AI comments.
    files: BTreeSet<PathBuf>,
    /// The files saved since the last settled save.
    saved: BTreeSet<PathBuf>,
    /// The entity whose [`WakeAt`] ends the settling.
    settling: Option<Entity>,
    /// The files whose trigger waits for a turn, in the order saved.
    queue: Vec<PathBuf>,
    /// What rig's tools and this plugin last left in each file.
    written: HashMap<PathBuf, ContentHash>,
}

/// A turn's work, on the message that asks for it, then on its turn.
#[derive(Component, Clone, Debug)]
struct Job {
    /// The file whose trigger started it.
    file: PathBuf,
    /// Whether it only answers questions.
    question: bool,
    /// The context comments it was given: each file and comment text.
    context: Vec<(PathBuf, String)>,
}

/// On a turn this plugin started.
#[derive(Component, Clone, Debug)]
struct JobTurn(Job);

/// `/watch on`, `/watch off`, or `/watch` to say whether it watches.
fn watch(
    In(args): In<CommandArgs>,
    mut watching: ResMut<Watching>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    match args.args.as_str() {
        "on" => watching.on = true,
        "off" => {
            if let Some(watch) = watching.watch.take() {
                commands.entity(watch).despawn();
            }
            if let Some(settling) = watching.settling.take() {
                commands.entity(settling).despawn();
            }
            *watching = Watching {
                written: std::mem::take(&mut watching.written),
                ..Watching::default()
            };
        }
        "" => {}
        other => {
            let why = format!("Unknown /watch argument `{other}`: use /watch on or /watch off.");
            notices.write(Notice::error(args.agent, why));
            return;
        }
    }
    let state = if watching.on { "on" } else { "off" };
    notices.write(Notice::info(
        args.agent,
        format!("Watching for AI comments: {state}."),
    ));
}

/// Arms the watch of the project once watching is on, on this machine,
/// whose files the plugin reads on the main thread; it keeps a headless
/// run going while it is armed.
fn arm(
    mut watching: ResMut<Watching>,
    workspace: Res<ProjectWorkspace>,
    root: Res<ProjectRoot>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    if !watching.on || watching.watch.is_some() {
        return;
    }
    if let Some(host) = workspace.0.host() {
        watching.on = false;
        let why = format!("/watch watches this machine's files; the project is on {host}.");
        notices.write(Notice::error(None, why));
        return;
    }
    let name = Name::new("watching for AI comments");
    let watch = commands
        .spawn((name, FileWatch::tree(&root.0), KeepRunning))
        .id();
    watching.watch = Some(watch);
    watching.primed = false;
}

/// Notes the files saved under the watch and puts the settling off.
fn take_changes(
    mut changes: MessageReader<FileChanges>,
    mut watching: ResMut<Watching>,
    mut commands: Commands,
) {
    let mut saved = false;
    for change in changes.read() {
        if Some(change.watch) != watching.watch {
            continue;
        }
        watching.files.extend(change.paths.iter().cloned());
        // The first look lists what was there before watching.
        if !watching.primed {
            watching.primed = true;
            continue;
        }
        watching.saved.extend(change.paths.iter().cloned());
        saved = true;
    }
    if !saved {
        return;
    }
    let settle = WakeAt::after(SETTLE);
    match watching.settling {
        Some(settling) => {
            commands.entity(settling).insert(settle);
        }
        None => watching.settling = Some(commands.spawn(settle).id()),
    }
}

/// Once the saves settled and no file tool that may write them runs,
/// queues each saved file that holds a trigger, unless rig left it so.
fn read_saves(
    settled: Query<(), With<Due>>,
    running: Query<&Footprint, (ToolCalls, With<Started>, Without<Outcome>)>,
    mut watching: ResMut<Watching>,
    workspace: Res<ProjectWorkspace>,
    mut commands: Commands,
) {
    let Some(settling) = watching
        .settling
        .filter(|settling| settled.contains(*settling))
    else {
        return;
    };
    let saved = Scope::Paths(watching.saved.iter().cloned().collect());
    // A file tool may be writing them: the settling stays due, so they are
    // read on the frame its outcome, and so its facts, are in.
    let writing = |scope: &Scope| scope.overlaps(&saved);
    if running.iter().filter_map(names_writes).any(writing) {
        return;
    }
    commands.entity(settling).despawn();
    watching.settling = None;
    let workspace = &*workspace.0;
    for path in std::mem::take(&mut watching.saved) {
        let Some(text) = read(workspace, &path) else {
            continue;
        };
        let rigs = watching.written.get(&workspace.canonical(&path));
        if rigs == Some(&content_hash(text.as_bytes())) {
            continue;
        }
        let triggered = comments(&path, &text)
            .iter()
            .any(|comment| comment.kind.triggers());
        if triggered && !watching.queue.contains(&path) {
            watching.queue.push(path);
        }
    }
}

/// Starts a turn of the primary agent for the first queued file that still
/// holds a trigger, once the agent is idle.
fn start_turn(
    mut watching: ResMut<Watching>,
    agents: MainAgentQuery,
    busy: Query<(), With<ActiveTurn>>,
    sending: Query<(), With<Job>>,
    workspace: Res<ProjectWorkspace>,
    root: Res<ProjectRoot>,
    mut commands: Commands,
) {
    let Some(agent) = main_agent(&agents) else {
        return;
    };
    if !watching.on || busy.contains(agent) || !sending.is_empty() {
        return;
    }
    let workspace = &*workspace.0;
    while !watching.queue.is_empty() {
        let file = watching.queue.remove(0);
        if let Some((job, text)) = job(workspace, &root.0, &watching.files, file) {
            let author = Author::user().titled("AI comments");
            let message = Incoming::new(text, DeliveryMode::Queue)
                .with_author(author)
                .literal();
            commands.spawn((message, ForAgent(agent), job));
            return;
        }
    }
}

/// The turn for `file`'s triggers and the message that asks for it, with
/// the AI comments of the watched `files`, unless it holds none any more.
fn job(
    workspace: &dyn Workspace,
    root: &Path,
    files: &BTreeSet<PathBuf>,
    file: PathBuf,
) -> Option<(Job, String)> {
    let found: Vec<(PathBuf, String, Vec<Comment>)> = files
        .iter()
        .filter_map(|path| {
            let text = read(workspace, path)?;
            let found = comments(path, &text);
            (!found.is_empty()).then_some((path.clone(), text, found))
        })
        .collect();
    let triggers: Vec<&Comment> = found
        .iter()
        .filter(|(path, ..)| *path == file)
        .flat_map(|(_, _, found)| found.iter().filter(|comment| comment.kind.triggers()))
        .collect();
    if triggers.is_empty() {
        return None;
    }
    let question = triggers
        .iter()
        .all(|comment| comment.kind == Kind::Question);
    let name = |path: &Path| {
        path.strip_prefix(root)
            .unwrap_or(path)
            .display()
            .to_string()
    };
    let mut listed = Vec::new();
    let mut code = Vec::new();
    let mut context = Vec::new();
    for (path, text, found) in &found {
        let lines: Vec<&str> = text.lines().collect();
        for comment in found {
            let at = comment.line + 1;
            let what = comment.kind.label();
            listed.push(format!("- {}:{at} ({what}): {}", name(path), comment.text));
            let from = comment.line.saturating_sub(AROUND);
            let shown = lines.iter().enumerate().skip(from).take(2 * AROUND + 1);
            let shown: Vec<String> = shown
                .map(|(n, line)| format!("{:>5} | {line}", n + 1))
                .collect();
            code.push(format!(
                "{}:{at}\n```\n{}\n```",
                name(path),
                shown.join("\n")
            ));
            if comment.kind == Kind::Context {
                context.push((path.clone(), comment.text.clone()));
            }
        }
    }
    let asked = if question {
        format!(
            "I left `AI?` comments in {} with questions. Answer them in your reply. This turn \
             is question only: change no files.",
            name(&file)
        )
    } else {
        format!(
            "I left `AI!` comments in {} asking for changes. Make them. Leave the AI comments \
             where they are: they are taken out when your turn ends.",
            name(&file)
        )
    };
    let text = format!(
        "{asked} Comments ending in `AI` are context.\n\nThe AI comments in the project:\n{}\n\n\
         The code around each:\n\n{}",
        listed.join("\n"),
        code.join("\n\n")
    );
    Some((
        Job {
            file,
            question,
            context,
        },
        text,
    ))
}

/// Moves a job from its message to the turn it went into.
fn mark_turn(
    delivered: On<Add<DeliveredIn>>,
    messages: Query<(&Job, &DeliveredIn)>,
    mut commands: Commands,
) {
    if let Ok((job, &DeliveredIn(turn))) = messages.get(delivered.entity) {
        commands.entity(turn).try_insert(JobTurn(job.clone()));
    }
}

/// Refuses each call of a question's turn that may change anything.
fn question_only(
    mut calls: Query<(&Footprint, &CallOf, &mut Decision), (ReadyToStart, ToolCalls)>,
    turns: Query<&JobTurn>,
) {
    for (footprint, &CallOf(turn), mut decision) in &mut calls {
        let question = turns.get(turn).is_ok_and(|JobTurn(job)| job.question);
        if question && !footprint.0.reads_only() {
            decision.decide(Decider::plugin(NAME), Outcome::refused(QUESTION_ONLY));
        }
    }
}

/// The paths a call says it writes, when it names them: only such a
/// call's [`FilesChanged`] is what its tool wrote.
fn names_writes(footprint: &Footprint) -> Option<&Scope> {
    match &footprint.0 {
        Touches::Writes(scope @ Scope::Paths(_)) => Some(scope),
        _ => None,
    }
}

/// Keeps what each file tool call left in the files it changed.
fn note_writes(
    published: On<Add<Published>>,
    calls: Query<(&Published, &Footprint)>,
    mut watching: ResMut<Watching>,
) {
    let Ok((Published(facts), footprint)) = calls.get(published.entity) else {
        return;
    };
    if names_writes(footprint).is_none() {
        return;
    }
    let Ok(Some(changed)) = facts.get::<FilesChanged>() else {
        return;
    };
    for change in changed.0 {
        if let Some(after) = change.after.and_then(|it| it.hash) {
            watching.written.insert(PathBuf::from(change.path), after);
        }
    }
}

/// Once a job's turn ended, however it ended, takes its file's triggers and
/// its context comments out of the files, so they never fire twice; a
/// question's only once it is answered.
fn clean_up(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &JobTurn)>,
    mut watching: ResMut<Watching>,
    workspace: Res<ProjectWorkspace>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((outcome, JobTurn(job))) = turns.get(settled.entity) else {
        return;
    };
    if job.question && !matches!(outcome, TurnOutcome::Answered(_)) {
        return;
    }
    let workspace = &*workspace.0;
    let files: BTreeSet<&PathBuf> = std::iter::once(&job.file)
        .chain(job.context.iter().map(|(path, _)| path))
        .collect();
    for path in files {
        let Some(text) = read(workspace, path) else {
            continue;
        };
        let used = |comment: &Comment| {
            (comment.kind.triggers() && *path == job.file)
                || (comment.kind == Kind::Context
                    && job
                        .context
                        .iter()
                        .any(|(file, text)| file == path && *text == comment.text))
        };
        let Some(cleaned) = without(path, &text, used) else {
            continue;
        };
        match workspace.write(path, cleaned.as_bytes()) {
            Ok(()) => {
                let hash = content_hash(cleaned.as_bytes());
                watching.written.insert(workspace.canonical(path), hash);
            }
            Err(error) => {
                let why = format!(
                    "Could not take the AI comments out of {}: {error}",
                    path.display()
                );
                notices.write(Notice::error(None, why));
            }
        }
    }
}

/// `watching` and how many files wait, on the status line.
fn show(watching: Res<Watching>, mut status: ResMut<AppStatus>) {
    let text = match (watching.on, watching.queue.len()) {
        (false, _) => String::new(),
        (true, 0) => "watching".to_owned(),
        (true, queued) => format!("watching · {queued} queued"),
    };
    status.0.show(ITEM.says(text, Tone::Dim));
}

/// The text of `path`, unless it is too large or no text.
fn read(workspace: &dyn Workspace, path: &Path) -> Option<String> {
    let contents = workspace.read(path, MAX_BYTES).ok()?;
    if contents.size > MAX_BYTES {
        return None;
    }
    String::from_utf8(contents.bytes).ok()
}

/// What an AI comment asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// `AI!`: a change.
    Change,
    /// `AI?`: a question.
    Question,
    /// `AI`: context only.
    Context,
}

impl Kind {
    /// Whether a comment of this kind starts a turn.
    fn triggers(self) -> bool {
        self != Self::Context
    }

    /// How the message names it.
    fn label(self) -> &'static str {
        match self {
            Self::Change => "change request",
            Self::Question => "question",
            Self::Context => "context",
        }
    }

    /// The kind of the comment `text` (after its marker), if it is an AI
    /// comment: it ends in `AI!`, `AI?` or `AI` as a word of its own.
    fn of(text: &str) -> Option<Self> {
        let text = text.trim_end();
        let (rest, kind) = if let Some(rest) = text.strip_suffix("AI!") {
            (rest, Self::Change)
        } else if let Some(rest) = text.strip_suffix("AI?") {
            (rest, Self::Question)
        } else {
            (text.strip_suffix("AI")?, Self::Context)
        };
        let word = rest
            .chars()
            .last()
            .is_none_or(|before| !before.is_alphanumeric());
        word.then_some(kind)
    }
}

/// An AI comment.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Comment {
    /// Its line, from 0.
    line: usize,
    /// Where its marker starts in the line.
    at: usize,
    /// Its text, without the marker.
    text: String,
    /// What it asks for.
    kind: Kind,
}

/// The comment markers of `path`'s type; a block comment's is `/*`.
fn markers(path: &Path) -> &'static [&'static str] {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default();
    match extension {
        "py" | "sh" | "bash" | "zsh" | "fish" | "rb" | "pl" | "r" | "toml" | "yaml" | "yml"
        | "nix" | "cmake" | "mk" | "ex" | "exs" | "jl" | "ps1" | "tf" | "conf" | "cfg" => &["#"],
        "rs" | "c" | "h" | "cc" | "cpp" | "hpp" | "cs" | "java" | "js" | "jsx" | "mjs" | "ts"
        | "tsx" | "go" | "swift" | "kt" | "scala" | "dart" | "php" | "zig" | "scss" | "proto" => {
            &["//", "/*"]
        }
        "css" => &["/*"],
        "sql" | "lua" | "hs" | "elm" | "ada" => &["--"],
        "lisp" | "el" | "clj" | "cljs" | "scm" | "asm" | "s" | "ini" => &[";"],
        _ => &["#", "//", "--", "/*"],
    }
}

/// The AI comments of `text`, the content of `path`.
fn comments(path: &Path, text: &str) -> Vec<Comment> {
    let markers = markers(path);
    text.lines()
        .enumerate()
        .filter_map(|(line, content)| {
            comment(markers, content).map(|found| Comment { line, ..found })
        })
        .collect()
}

/// The first AI comment of `line`, by `markers`.
fn comment(markers: &[&str], line: &str) -> Option<Comment> {
    markers
        .iter()
        .flat_map(|marker| {
            line.match_indices(*marker)
                .map(move |(at, _)| (at, *marker))
        })
        .filter_map(|(at, marker)| {
            let mut text = line.get(at + marker.len()..)?.trim();
            if marker == "/*" {
                text = text.strip_suffix("*/").unwrap_or(text).trim_end();
            }
            let text = text
                .trim_start_matches(['/', '*', '!', '#', '-', ';'])
                .trim();
            let kind = Kind::of(text)?;
            Some(Comment {
                line: 0,
                at,
                text: text.to_owned(),
                kind,
            })
        })
        .min_by_key(|comment| comment.at)
}

/// `text`, the content of `path`, without the AI comments `used` picks: a
/// line that was only the comment goes, another keeps its code. `None` when
/// none is there.
fn without(path: &Path, text: &str, used: impl Fn(&Comment) -> bool) -> Option<String> {
    let markers = markers(path);
    let mut changed = false;
    let mut kept: Vec<&str> = Vec::new();
    for line in text.lines() {
        match comment(markers, line).filter(|found| used(found)) {
            Some(found) => {
                changed = true;
                let code = line.get(..found.at).unwrap_or(line).trim_end();
                if !code.is_empty() {
                    kept.push(code);
                }
            }
            None => kept.push(line),
        }
    }
    if !changed {
        return None;
    }
    let mut cleaned = kept.join("\n");
    if text.ends_with('\n') && !cleaned.is_empty() {
        cleaned.push('\n');
    }
    Some(cleaned)
}

#[cfg(test)]
mod tests;
