//! FH-12: open a pull request from the session.
//!
//! `/pr` (`/pr draft`, `/pr --base <branch>`) turns the session's work into
//! a pull request. The plugin keeps, with the session, the files each turn
//! changed (what `git status` and the files' contents show changed between
//! the turn's start and end, so the agent's shell edits count and the
//! user's earlier edits do not) and every command a tool ran, with how it
//! ended. Both snapshots run off the main thread: the turn's first model
//! call waits for the first, the turn's end for the second. `/pr` checks
//! the repository, its remote and `gh` off the main thread, then asks a
//! writer model (`[open-pr] model`, else the agent's) for the title, commit
//! message and description from the user's requests, the diff against the
//! base and those commands, following the repository's PR template when it
//! has one. A review question shows the
//! branch, files, message, title and description, each edited in place
//! from its current text, with Publish, Commit only and Cancel. Nothing is
//! committed or pushed before that answer. The steps then run off the main
//! thread as an effect of the plugin's own (`open-pr`), each one shown as
//! it starts; a failing step stops there and says how to go on. A pull
//! request already open for the branch gets the new commits and, if the
//! user agrees, the new description. The URL is kept with the session, in
//! `/resume`'s line and shown again when the session is resumed.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use bevy_tasks::{IoTaskPool, TaskPool};
use futures::channel::mpsc;
use rig_basics::sessions::ResumeLabel;
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::{CompletionRequest, Message};
use rig_harness::rig_core::effect::EffectKind;
use rig_harness::rig_core::tool::{ContentHash, ExitStatus, content_hash};
use rig_tools::ShellArgs;
use rig_tools::process::{Launcher, Spawn};
use rig_tools::workspace::Workspace;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests;

/// The plugin's name: its settings' table, its effect's kind and the
/// purpose of its model calls.
const NAME: &str = "open-pr";

/// The most of the diff the writer reads.
const DIFF_CHARS: usize = 20_000;

/// What the writer model is told.
const PREAMBLE: &str = "You write pull requests for changes a coding agent made. Answer \
    with one JSON object with the string fields `title` (an imperative line of at most 72 \
    characters), `commit_message` (a subject line, a blank line, then a short body) and \
    `body` (Markdown). Unless a template is given, the body has the sections `## Summary`, \
    `## Changes` and `## How it was tested`. That last section names only those of the \
    listed commands that ran tests, each with whether it passed; when none did it says \
    \"Not tested.\" Never claim a test that is not listed.";

/// Where a repository keeps its pull request template.
const TEMPLATES: [&str; 6] = [
    ".github/pull_request_template.md",
    ".github/PULL_REQUEST_TEMPLATE.md",
    "pull_request_template.md",
    "PULL_REQUEST_TEMPLATE.md",
    "docs/pull_request_template.md",
    "docs/PULL_REQUEST_TEMPLATE.md",
];

/// The plugin's stage systems, for other plugins to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpenPrSystems {
    /// Holds a turn's first model call until the snapshot of its
    /// repository at the turn's start lands, in [`PrepareSystems::Compose`].
    HoldStart,
    /// Holds a turn's proposed end until the snapshot at its end lands, in
    /// [`ReviewSystems::Act`].
    HoldEnd,
}

/// Adds `/pr`.
#[derive(Default)]
pub struct OpenPrPlugin;

impl Plugin for OpenPrPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<OpenPrConfig>(NAME)
            .init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .init_resource::<Processes>()
            .insert_resource(SessionWork::default())
            .configure_sets(
                AgentPass,
                (
                    OpenPrSystems::HoldStart.in_set(PrepareSystems::Compose),
                    OpenPrSystems::HoldEnd.in_set(ReviewSystems::Act),
                ),
            )
            .add_systems(
                AgentPass,
                (
                    hold_first_call.in_set(OpenPrSystems::HoldStart),
                    hold_turn_end.in_set(OpenPrSystems::HoldEnd),
                ),
            )
            .add_observer(snapshot_start)
            .add_observer(snapshot_interrupted)
            .add_observer(taken)
            .add_observer(note_run)
            .add_observer(prepared)
            .add_observer(written)
            .add_observer(answered)
            .add_observer(start_job)
            .add_observer(progress)
            .add_observer(follow_successor::<Writing>)
            .add_systems(
                Update,
                (
                    label.run_if(resource_changed::<SessionWork>),
                    remind.run_if(resource_added::<SessionRestored>),
                ),
            )
            .add_command(
                "pr",
                "Commit this session's changes and open a pull request: /pr [draft] [--base <branch>]",
                pr,
            );
        app.world_mut().spawn(ResumeLabel::new("pr", ""));
    }
}

/// The `[open-pr]` table of the settings.
#[derive(Resource, Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OpenPrConfig {
    /// The catalog model (`vendor/model`) that writes the pull request;
    /// none: the agent's own.
    pub model: Option<String>,
    /// The GitHub CLI, or a program that takes its `pr view`, `pr create`
    /// and `pr edit` arguments.
    pub gh: String,
}

impl Default for OpenPrConfig {
    fn default() -> Self {
        Self {
            model: None,
            gh: "gh".to_owned(),
        }
    }
}

/// What the session did that a pull request is written from: the files
/// its turns changed (absolute paths), the commands its tools ran, and the
/// pull requests opened. Saved with the session.
#[derive(Resource, Reflect, Clone, Debug, Default, Serialize, Deserialize)]
#[reflect(opaque, Resource, Saved, Default, Serialize, Deserialize)]
pub struct SessionWork {
    /// Every file a turn changed.
    pub changed: BTreeSet<PathBuf>,
    /// Every command a tool ran, in order.
    pub runs: Vec<Run>,
    /// The pull requests opened or updated, oldest first.
    pub prs: Vec<String>,
}

/// A command a tool ran.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Run {
    /// The command line.
    pub command: String,
    /// Whether it exited with 0.
    pub passed: bool,
}

/// A repository's top and its uncommitted files, relative to it, with
/// their contents' hash (`None`: deleted).
#[derive(Clone)]
struct Dirty(PathBuf, BTreeMap<String, Option<ContentHash>>);

/// On a turn while it runs: what was dirty in its repository as it
/// started.
#[derive(Component)]
struct Before(Dirty);

/// On the snapshot of a turn's repository taken off the main thread: at
/// the turn's start (its holds are children) or at its end.
#[derive(Component, Clone, Copy)]
enum Snapshot {
    Start(Entity),
    End(Entity),
}

/// What a [`Snapshot`] found.
#[derive(Component)]
struct Taken(Result<Dirty, String>);

impl From<Result<Dirty, String>> for Taken {
    fn from(taken: Result<Dirty, String>) -> Self {
        Self(taken)
    }
}

/// What `/pr` will do: where, on which branch, with what text.
#[derive(Clone, Debug)]
struct Plan {
    root: PathBuf,
    remote: String,
    base: String,
    branch: String,
    /// Whether `branch` is made from where the repository stands.
    new_branch: bool,
    /// The local branches, which a new one is named apart from.
    taken: BTreeSet<String>,
    /// The session's files to commit, relative to `root`.
    files: Vec<String>,
    /// The other uncommitted files, left alone.
    left_alone: Vec<String>,
    message: String,
    title: String,
    body: String,
    draft: bool,
    gh: String,
}

/// The project's processes: where they run and how they start.
#[derive(Clone)]
struct Runner(Arc<dyn Workspace>, Launcher);

/// On `/pr`'s preparation: its agent, the user's requests and the
/// commands run; once done, the plan without its text and what the writer
/// reads of the repository, or what is missing.
#[derive(Component)]
struct Preparing(Entity, Vec<String>, Vec<Run>);

/// The end of `/pr`'s preparation.
#[derive(Component)]
struct Prepared(Result<(Plan, String), String>);

impl From<Result<(Plan, String), String>> for Prepared {
    fn from(prepared: Result<(Plan, String), String>) -> Self {
        Self(prepared)
    }
}

/// What the writer answers.
#[derive(Deserialize, JsonSchema)]
struct Written {
    /// The pull request's title.
    title: String,
    /// The commit message.
    commit_message: String,
    /// The pull request's description.
    body: String,
}

/// On the writer's model call: the plan it writes the text of, and the
/// requests, for a plain draft if it fails.
#[derive(Component)]
struct Writing(Plan, Vec<String>);

/// On the review question: the plan it shows.
#[derive(Component)]
struct Review(Plan);

/// What an edit question edits.
#[derive(Clone, Copy, Debug)]
enum Field {
    Title,
    Body,
    Message,
    Branch,
    Files,
}

/// On a question that edits a field of the plan.
#[derive(Component)]
struct Editing(Plan, Field);

/// On the question whether to update an open pull request's description.
#[derive(Component)]
struct UpdateBody(Plan);

/// What a job does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Action {
    Publish,
    CommitOnly,
    UpdateBody,
}

/// On the plugin's `open-pr` effect: the plan and what to do with it.
#[derive(Component, Clone)]
struct Job(Plan, Action);

/// On the entity that runs a job's steps: its effect and the job.
#[derive(Component)]
struct Worker(Entity, Job);

/// What a job's thread reports.
enum Step {
    /// A step starts.
    Doing(String),
    /// The job ended.
    Done(Finish),
}

/// How a job ended, the `open-pr` effect's outcome.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum Finish {
    Opened { url: String },
    Existing { url: String, body: String },
    Committed { commit: String },
    Updated { url: String },
    Failed { error: String, hint: String },
}

/// The project's processes, as the resources say.
fn runner(workspace: &ProjectWorkspace, processes: &Processes) -> Runner {
    Runner(workspace.0.clone(), processes.launcher())
}

/// Runs `work` on a thread of its own, ending as the component `C`.
fn off_thread<C, T>(
    wake: &Wake,
    work: impl FnOnce() -> Result<T, String> + Send + 'static,
) -> Running
where
    C: Component + From<Result<T, String>>,
    T: Send + 'static,
{
    let work = blocking(move || Ok(work()));
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    Running::spawn_into::<C, _>(pool, wake, async move {
        work.await.unwrap_or_else(|error| Err(error.to_string()))
    })
}

/// Snapshots what is dirty in a turn's repository, in `dir` or above it,
/// as the turn starts.
fn snapshot_start(
    turn: On<Add<TurnOf>>,
    turns: Query<&TurnOf>,
    dirs: Query<&WorkDir>,
    (project, workspace, processes, wake): (
        Res<ProjectRoot>,
        Res<ProjectWorkspace>,
        Res<Processes>,
        Res<Wake>,
    ),
    mut commands: Commands,
) {
    let Ok(&TurnOf(agent)) = turns.get(turn.entity) else {
        return;
    };
    let dir = dirs
        .get(agent)
        .map_or(&project.0, |WorkDir(dir)| dir)
        .clone();
    let runner = runner(&workspace, &processes);
    commands.spawn((
        Name::new("open-pr: the repository as the turn starts"),
        Snapshot::Start(turn.entity),
        off_thread::<Taken, _>(&wake, move || runner.snapshot(&dir)),
    ));
}

/// Holds a turn's first model call while its start snapshot runs, so no
/// tool of the turn changes a file before it.
fn hold_first_call(
    calls: Query<(Entity, &CallOf), (Added<Admitted>, With<TurnCall>)>,
    snapshots: Query<(Entity, &Snapshot)>,
    mut commands: Commands,
) {
    for (call, &CallOf(turn)) in &calls {
        let starts = snapshots.iter().filter(|&(_, &snapshot)| match snapshot {
            Snapshot::Start(of) => of == turn,
            Snapshot::End(_) => false,
        });
        for (snapshot, _) in starts {
            let hold = (Name::new("hold: open-pr snapshot"), HoldOf(call));
            commands.entity(snapshot).with_child(hold);
        }
    }
}

/// Holds a turn's proposed end while a snapshot of its repository runs.
/// A proposal taken away takes the snapshot with it, and the next one
/// takes another.
fn hold_turn_end(
    turns: Query<(Entity, &Before), (Added<TurnOutcome>, With<TurnOf>)>,
    (workspace, processes, wake): (Res<ProjectWorkspace>, Res<Processes>, Res<Wake>),
    mut commands: Commands,
) {
    for (turn, Before(Dirty(root, _))) in &turns {
        let runner = runner(&workspace, &processes);
        let snapshot = end_snapshot(turn, root.clone(), runner, &wake);
        commands.spawn((snapshot, HoldOf(turn)));
    }
}

/// The snapshot at the end of `turn` of its repository at `root`.
fn end_snapshot(turn: Entity, root: PathBuf, runner: Runner, wake: &Wake) -> impl Bundle {
    (
        Name::new("open-pr: what the turn changed"),
        Snapshot::End(turn),
        off_thread::<Taken, _>(wake, move || runner.snapshot(&root)),
    )
}

/// Snapshots the repository of a turn that ended unheld, such as one the
/// user interrupted.
fn snapshot_interrupted(
    settled: On<Add<Settled>>,
    turns: Query<&Before, With<Turn>>,
    (workspace, processes, wake): (Res<ProjectWorkspace>, Res<Processes>, Res<Wake>),
    mut commands: Commands,
) {
    let Ok(Before(Dirty(root, _))) = turns.get(settled.entity) else {
        return;
    };
    let runner = runner(&workspace, &processes);
    commands.spawn(end_snapshot(settled.entity, root.clone(), runner, &wake));
}

/// Keeps a turn's start snapshot on it, or adds the files whose state
/// changed since then to the session's; releases what waited for it.
fn taken(
    done: On<Insert<Taken>>,
    snapshots: Query<(&Snapshot, &Taken)>,
    befores: Query<&Before>,
    mut work: ResMut<SessionWork>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((&snapshot, Taken(taken))) = snapshots.get(done.entity) else {
        return;
    };
    commands.entity(done.entity).despawn();
    match (snapshot, taken) {
        (Snapshot::Start(turn), Ok(dirty)) => {
            commands.entity(turn).try_insert(Before(dirty.clone()));
        }
        // Not in a repository: nothing to record.
        (Snapshot::Start(_), Err(_)) => {}
        (Snapshot::End(turn), after) => {
            let Ok(Before(Dirty(root, before))) = befores.get(turn) else {
                return;
            };
            commands.entity(turn).try_remove::<Before>();
            let Dirty(_, after) = match after {
                Ok(after) => after,
                Err(why) => {
                    let why = format!("/pr cannot tell which files a turn changed: {why}");
                    notices.write(Notice::warn(None, why));
                    return;
                }
            };
            let changed = before.keys().chain(after.keys());
            let changed = changed.filter(|path| before.get(*path) != after.get(*path));
            work.changed.extend(changed.map(|path| root.join(path)));
        }
    }
}

/// Notes each command a tool ran and how it ended.
fn note_run(
    published: On<Add<Published>>,
    calls: Query<(&Published, &ToolRequest)>,
    mut work: ResMut<SessionWork>,
) {
    let Ok((Published(facts), request)) = calls.get(published.entity) else {
        return;
    };
    let args = serde_json::from_value::<ShellArgs>(serde_json::Value::Object(
        request.call.function.arguments.clone(),
    ));
    if let (Ok(Some(status)), Ok(args)) = (facts.get::<ExitStatus>(), args) {
        work.runs.push(Run {
            command: args.command,
            passed: status.success(),
        });
    }
}

/// `/pr [draft] [--base <branch>]`: checks what it needs off the main
/// thread.
fn pr(
    In(args): In<CommandArgs>,
    agents: Query<(&Conversation, Option<&WorkDir>, Has<ActiveTurn>)>,
    (project, workspace, processes, config, work): (
        Res<ProjectRoot>,
        Res<ProjectWorkspace>,
        Res<Processes>,
        Res<OpenPrConfig>,
        Res<SessionWork>,
    ),
    answerers: Answerers,
    wake: Res<Wake>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let Ok((conversation, dir, running)) = agents.get(agent) else {
        return;
    };
    if running {
        notices.write(Notice::turn_running(agent));
        return;
    }
    if !answerers.can_answer(Some(agent)) {
        notices.write(Notice::error(
            agent,
            "/pr shows a review first: nobody can answer it in this run.",
        ));
        return;
    }
    let (draft, base) = match parse(&args.args) {
        Ok(parsed) => parsed,
        Err(why) => {
            notices.write(Notice::error(agent, why));
            return;
        }
    };
    let requests: Vec<String> = (0..conversation.messages().len())
        .filter_map(|at| typed(conversation, at))
        .collect();
    let dir = dir.map_or(&project.0, |WorkDir(dir)| dir).clone();
    let (runner, gh, changed) = (
        runner(&workspace, &processes),
        config.gh.clone(),
        work.changed.clone(),
    );
    commands.spawn((
        Preparing(agent, requests, work.runs.clone()),
        off_thread::<Prepared, _>(&wake, move || {
            runner.prepare(&dir, gh, &changed, draft, base)
        }),
    ));
}

/// `/pr`'s arguments: whether a draft, and the base asked for.
fn parse(args: &str) -> Result<(bool, Option<String>), String> {
    let (mut draft, mut base) = (false, None);
    let mut words = args.split_whitespace();
    while let Some(word) = words.next() {
        match (word, words.clone().next()) {
            ("draft" | "--draft", _) => draft = true,
            ("--base", Some(branch)) => {
                base = Some(branch.to_owned());
                words.next();
            }
            _ => return Err("Use /pr [draft] [--base <branch>].".to_owned()),
        }
    }
    Ok((draft, base))
}

/// Asks the writer for the text of a prepared plan.
fn prepared(
    done: On<Add<Prepared>>,
    preparations: Query<(&Preparing, &Prepared)>,
    config: Res<OpenPrConfig>,
    models: Res<Models>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((Preparing(agent, requests, runs), Prepared(prepared))) = preparations.get(done.entity)
    else {
        return;
    };
    let agent = *agent;
    commands.entity(done.entity).despawn();
    let (plan, context) = match prepared {
        Ok(prepared) => prepared,
        Err(why) => {
            notices.write(Notice::error(agent, why.clone()));
            return;
        }
    };
    let prompt = writer_prompt(plan, requests, runs, context);
    let request = CompletionRequest::new(Message::user(prompt))
        .preamble(PREAMBLE)
        .output_schema(schemars::schema_for!(Written));
    let mut request = ModelRequest::new(request);
    if let Some(reference) = config.model.as_deref() {
        match models.connect(reference) {
            Ok(connection) => request = request.on(connection),
            Err(why) => {
                notices.write(Notice::error(agent, format!("[open-pr] model: {why}")));
                return;
            }
        }
    }
    notices.write(Notice::info(
        agent,
        "Writing the pull request's title and description…",
    ));
    commands.spawn((
        Name::new("pull request text"),
        request,
        Effect::default().with_purpose(NAME),
        ForAgent(agent),
        Writing(plan.clone(), requests.clone()),
    ));
}

/// What the writer is given.
fn writer_prompt(plan: &Plan, requests: &[String], runs: &[Run], context: &str) -> String {
    let requests: Vec<String> = requests
        .iter()
        .map(|request| format!("- {request}"))
        .collect();
    let runs = if runs.is_empty() {
        "No command ran in this session.".to_owned()
    } else {
        let lines = runs.iter().map(|run| {
            let how = if run.passed { "passed" } else { "failed" };
            format!("- `{}`: {how}", run.command)
        });
        lines.collect::<Vec<_>>().join("\n")
    };
    format!(
        "# What the user asked for\n{}\n\n# Files committed\n{}\n\n# Commands run in the session\n{runs}\n\n{context}",
        requests.join("\n"),
        plan.files.join("\n"),
    )
}

/// Takes the writer's answer, or a plain draft when it failed, and shows
/// the review.
fn written(
    settled: On<Add<Settled>>,
    calls: Query<(&Writing, &Outcome, &ForAgent)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((Writing(plan, requests), outcome, &ForAgent(agent))) = calls.get(settled.entity) else {
        return;
    };
    let mut plan = plan.clone();
    let written = outcome
        .completion()
        .and_then(|response| serde_json::from_str::<Written>(response.text().trim()).ok());
    match written {
        Some(written) => {
            plan.title = written.title.trim().to_owned();
            plan.message = written.commit_message.trim().to_owned();
            plan.body = written.body.trim().to_owned();
        }
        None => {
            notices.write(Notice::warn(
                agent,
                "The writer gave no pull request text; the review shows a plain draft.",
            ));
            let title: String = requests
                .first()
                .and_then(|request| request.lines().next())
                .unwrap_or("Changes")
                .chars()
                .take(72)
                .collect();
            plan.message.clone_from(&title);
            let files: Vec<String> = plan.files.iter().map(|file| format!("- {file}")).collect();
            plan.body = format!(
                "## Summary\n{}\n\n## Changes\n{}\n\n## How it was tested\nNot tested.",
                requests.join("\n"),
                files.join("\n")
            );
            plan.title = title;
        }
    }
    if plan.new_branch {
        plan.branch = free_branch(&plan.taken, &plan.title);
    }
    commands.spawn((review(&plan), ForAgent(agent), Review(plan)));
}

/// The review question.
fn review(plan: &Plan) -> Ask {
    let files = |files: &[String]| {
        if files.is_empty() {
            "  (none)".to_owned()
        } else {
            let lines: Vec<String> = files.iter().map(|file| format!("  {file}")).collect();
            lines.join("\n")
        }
    };
    let branch = if plan.new_branch {
        format!("{} (new, from here)", plan.branch)
    } else {
        plan.branch.clone()
    };
    let draft = if plan.draft { " (draft)" } else { "" };
    let body = format!(
        "Branch: {branch} → {}/{}{draft}\n\nFiles to commit:\n{}\n\nLeft alone:\n{}\n\nCommit message:\n{}\n\nTitle: {}\n\n{}",
        plan.remote,
        plan.base,
        files(&plan.files),
        files(&plan.left_alone),
        plan.message,
        plan.title,
        plan.body
    );
    Ask::new("Open this pull request?")
        .body(body)
        .choice("publish", "Publish")
        .choice("commit", "Commit only")
        .choice("title", "Edit the title")
        .choice("body", "Edit the description")
        .choice("message", "Edit the commit message")
        .choice("branch", "Edit the branch")
        .choice("files", "Edit the files")
        .choice("cancel", "Cancel")
}

/// The question that edits `field` of `plan`, starting from its text.
fn edit_question(plan: &Plan, field: Field) -> Ask {
    match field {
        Field::Title => Ask::new("The pull request's title").initial(&plan.title),
        Field::Body => Ask::new("The pull request's description")
            .initial(&plan.body)
            .multiline(),
        Field::Message => Ask::new("The commit message")
            .initial(&plan.message)
            .multiline(),
        Field::Branch => Ask::new("The branch").initial(&plan.branch),
        Field::Files => Ask::new("The files to commit, one a line")
            .body(format!("Uncommitted too:\n{}", plan.left_alone.join("\n")))
            .initial(plan.files.join("\n"))
            .multiline(),
    }
}

/// Acts on each answer to the plugin's questions.
fn answered(
    settled: On<Add<Settled>>,
    asks: Query<
        (
            &Outcome,
            &ForAgent,
            Option<&Review>,
            Option<&Editing>,
            Option<&UpdateBody>,
        ),
        With<Ask>,
    >,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((outcome, &ForAgent(agent), review_of, editing, update)) = asks.get(settled.entity)
    else {
        return;
    };
    let answer = Answer::of(outcome);
    let mut job = |plan: &Plan, action| {
        let payload = serde_json::json!({
            "action": action, "branch": plan.branch, "base": plan.base,
            "files": plan.files, "title": plan.title,
        });
        let kind = EffectKind::Custom {
            kind: NAME.into(),
            payload,
        };
        commands.spawn((
            Name::new("pull request"),
            EffectRequest {
                kind,
                handler: None,
            },
            ForAgent(agent),
            Job(plan.clone(), action),
        ));
    };
    if let Some(Review(plan)) = review_of {
        let choice = answer.as_ref().and_then(|answer| answer.choices.first());
        let field = match choice.map(String::as_str) {
            Some("publish") => return job(plan, Action::Publish),
            Some("commit") => return job(plan, Action::CommitOnly),
            Some("title") => Field::Title,
            Some("body") => Field::Body,
            Some("message") => Field::Message,
            Some("branch") => Field::Branch,
            Some("files") => Field::Files,
            _ => {
                notices.write(Notice::info(
                    agent,
                    "Cancelled: nothing was committed or pushed.",
                ));
                return;
            }
        };
        let ask = edit_question(plan, field);
        commands.spawn((ask, ForAgent(agent), Editing(plan.clone(), field)));
    } else if let Some(Editing(plan, field)) = editing {
        let mut plan = plan.clone();
        if let Some(text) = answer.and_then(|answer| answer.text) {
            edit(&mut plan, *field, text.trim(), agent, &mut notices);
        }
        commands.spawn((review(&plan), ForAgent(agent), Review(plan)));
    } else if let Some(UpdateBody(plan)) = update {
        if answer.is_some_and(|answer| answer.is("yes")) {
            job(plan, Action::UpdateBody);
        } else {
            notices.write(Notice::info(
                agent,
                "The pull request's description is left as it was.",
            ));
        }
    }
}

/// Sets `field` of the plan to the user's edited text.
fn edit(
    plan: &mut Plan,
    field: Field,
    text: &str,
    agent: Entity,
    notices: &mut MessageWriter<Notice>,
) {
    if text.is_empty() {
        return;
    }
    match field {
        Field::Title => plan.title = text.to_owned(),
        Field::Body => plan.body = text.to_owned(),
        Field::Message => plan.message = text.to_owned(),
        Field::Branch => {
            plan.new_branch |= plan.branch != text;
            plan.branch = text.to_owned();
        }
        Field::Files => {
            let mut dirty: BTreeSet<String> = plan
                .files
                .drain(..)
                .chain(plan.left_alone.drain(..))
                .collect();
            for file in text.lines().map(str::trim).filter(|file| !file.is_empty()) {
                if dirty.remove(file) {
                    plan.files.push(file.to_owned());
                } else {
                    notices.write(Notice::warn(
                        agent,
                        format!("{file} has no uncommitted change; left out."),
                    ));
                }
            }
            plan.left_alone = dirty.into_iter().collect();
        }
    }
}

/// Runs a job's steps on a thread of their own once its effect starts.
fn start_job(
    started: On<Add<Started>>,
    jobs: Query<&Job>,
    (workspace, processes): (Res<ProjectWorkspace>, Res<Processes>),
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let Ok(job) = jobs.get(started.entity) else {
        return;
    };
    let (sender, steps) = mpsc::unbounded();
    let (thread_job, runner, failed) =
        (job.clone(), runner(&workspace, &processes), sender.clone());
    let thread = std::thread::Builder::new()
        .name(NAME.to_owned())
        .spawn(move || {
            let say = |step| {
                sender.unbounded_send(step).ok();
            };
            let finish = runner.steps(&thread_job, &|doing: String| say(Step::Doing(doing)));
            say(Step::Done(finish));
        });
    if let Err(error) = thread {
        let error = format!("cannot start: {error}");
        failed
            .unbounded_send(Step::Done(Finish::Failed {
                error,
                hint: "Nothing was changed.".to_owned(),
            }))
            .ok();
    }
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    commands.spawn((
        Worker(started.entity, job.clone()),
        Running::stream(pool, &wake, steps),
    ));
}

/// Shows each step as it starts, and how the job ended; answers its effect.
fn progress(
    reported: On<Insert<Done<Step>>>,
    workers: Query<(&Worker, &Done<Step>)>,
    mut effects: Query<(&mut Decision, &ForAgent), PluginEffects>,
    mut work: ResMut<SessionWork>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((Worker(effect, Job(plan, _)), Done(step))) = workers.get(reported.entity) else {
        return;
    };
    let Ok((mut decision, &ForAgent(agent))) = effects.get_mut(*effect) else {
        return;
    };
    let finish = match step {
        Step::Doing(doing) => {
            notices.write(Notice::info(agent, format!("/pr: {doing}…")));
            let stopped = serde_json::json!({ "stopped_during": doing });
            commands
                .entity(*effect)
                .insert(IfStopped(Outcome::custom(stopped)));
            return;
        }
        Step::Done(finish) => finish,
    };
    decision.decide(
        Decider::plugin("fh-open-pr"),
        Outcome::custom(serde_json::to_value(finish).unwrap_or_default()),
    );
    commands.entity(reported.entity).despawn();
    let mut keep = |url: &str| {
        if !url.is_empty() && !work.prs.iter().any(|pr| pr == url) {
            work.prs.push(url.to_owned());
        }
    };
    let text = match finish {
        Finish::Opened { url } => {
            keep(url);
            format!("Opened {url}")
        }
        Finish::Updated { url } => {
            keep(url);
            format!("Updated the description of {url}")
        }
        Finish::Committed { commit } => format!(
            "Committed {commit} on {}; nothing was pushed. Run /pr again to publish.",
            plan.branch
        ),
        Finish::Existing { url, body } => {
            keep(url);
            if body.trim() != plan.body.trim() {
                let ask =
                    Ask::yes_no("Update the pull request's description?").body(plan.body.clone());
                commands.spawn((ask, ForAgent(agent), UpdateBody(plan.clone())));
            }
            format!("Pushed the new commits to {url}")
        }
        Finish::Failed { error, hint } => {
            notices.write(Notice::error(
                agent,
                format!("/pr stopped: {error}\n{hint}"),
            ));
            return;
        }
    };
    notices.write(Notice::info(agent, text));
}

/// Keeps the latest pull request in this session's `/resume` line.
fn label(work: Res<SessionWork>, mut labels: Query<&mut ResumeLabel>) {
    let text = work.prs.last().cloned().unwrap_or_default();
    for mut label in &mut labels {
        if label.tag == "pr" {
            label.set_if_neq(ResumeLabel::new("pr", text.clone()));
        }
    }
}

/// Shows the session's pull requests again when it is resumed.
fn remind(work: Res<SessionWork>, mut notices: MessageWriter<Notice>) {
    if !work.prs.is_empty() {
        notices.write(Notice::info(
            None,
            format!("Pull requests of this session: {}", work.prs.join(", ")),
        ));
    }
}

/// `rig/<slug of the title>`, numbered when taken.
fn free_branch(taken: &BTreeSet<String>, title: &str) -> String {
    let lower = title.to_lowercase();
    let mut slug = String::new();
    for word in lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|word| !word.is_empty())
    {
        if slug.len() + word.len() > 40 {
            break;
        }
        if !slug.is_empty() {
            slug.push('-');
        }
        slug.push_str(word);
    }
    if slug.is_empty() {
        slug = "changes".to_owned();
    }
    let first = format!("rig/{slug}");
    std::iter::once(first.clone())
        .chain((2..100).map(|n| format!("rig/{slug}-{n}")))
        .find(|branch| !taken.contains(branch))
        .unwrap_or(first)
}

impl Runner {
    /// Runs `program` in `dir` of the workspace: its output, or its error.
    fn run(&self, program: &str, dir: &Path, args: &[&str]) -> Result<String, String> {
        let Self(workspace, launcher) = self;
        let args: Vec<OsString> = args.iter().map(OsString::from).collect();
        let output = workspace
            .spawn(
                launcher,
                Some(dir),
                program.as_ref(),
                &args,
                Spawn::default(),
            )
            .and_then(|process| process.output(None))
            .map_err(|error| format!("cannot run {program}: {error}"))?;
        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).into_owned())
        } else {
            let first = args.first().map(|arg| arg.to_string_lossy());
            Err(format!(
                "{program} {}: {}",
                first.unwrap_or_default(),
                String::from_utf8_lossy(&output.stderr).trim()
            ))
        }
    }

    /// Runs git in `dir`: its trimmed output, or its error.
    fn git(&self, dir: &Path, args: &[&str]) -> Result<String, String> {
        self.run("git", dir, args).map(|out| out.trim().to_owned())
    }

    /// The files `git status` shows in the repository at `root`, relative
    /// to it, with their two-letter status.
    fn status(&self, root: &Path) -> Result<BTreeMap<String, String>, String> {
        let args = ["status", "--porcelain=v1", "-z", "--untracked-files=all"];
        let out = self.run("git", root, &args)?;
        let mut files = BTreeMap::new();
        let mut fields = out.split('\0').filter(|field| !field.is_empty());
        while let Some(entry) = fields.next() {
            let code = entry.get(..2).unwrap_or_default();
            if let Some(path) = entry.get(3..) {
                files.insert(path.to_owned(), code.to_owned());
            }
            // A rename or copy names the file it came from next.
            if code.contains(['R', 'C']) {
                fields.next();
            }
        }
        Ok(files)
    }

    /// The repository `dir` is in, and its uncommitted files with their
    /// contents' hash.
    fn snapshot(&self, dir: &Path) -> Result<Dirty, String> {
        let root = PathBuf::from(self.git(dir, &["rev-parse", "--show-toplevel"])?);
        let hash = |path: &String| {
            let read = self.0.read(&root.join(path), u64::MAX).ok()?;
            Some(content_hash(&read.bytes))
        };
        let files = self.status(&root)?.into_keys();
        let files = files.map(|path| {
            let hash = hash(&path);
            (path, hash)
        });
        Ok(Dirty(root.clone(), files.collect()))
    }

    /// The plan without its text, and what the writer reads of the
    /// repository (the diff and any template), or what is missing.
    fn prepare(
        &self,
        dir: &Path,
        gh: String,
        changed: &BTreeSet<PathBuf>,
        draft: bool,
        base: Option<String>,
    ) -> Result<(Plan, String), String> {
        let root = self
            .git(dir, &["rev-parse", "--show-toplevel"])
            .map(PathBuf::from)
            .map_err(|_| format!("/pr needs a git repository: {} is in none.", dir.display()))?;
        let git = |args: &[&str]| self.git(&root, args);
        let remotes = git(&["remote"]).unwrap_or_default();
        let remote = remotes
            .lines()
            .find(|remote| *remote == "origin")
            .or_else(|| remotes.lines().next())
            .ok_or("/pr needs a remote to push to: this repository has none (git remote add origin <url>).")?
            .to_owned();
        self.run(&gh, &root, &["--version"]).map_err(|_| {
            format!(
                "/pr needs the GitHub CLI (`{gh}`), or an equivalent set as `[open-pr] gh`: it does not run."
            )
        })?;
        let current = git(&["branch", "--show-current"]).unwrap_or_default();
        let default = self.default_branch(&root, &remote);
        let base = base.unwrap_or(default.clone());
        let new_branch = current.is_empty() || current == default || current == base;
        let status = self.status(&root)?;
        let session: BTreeSet<&Path> = changed
            .iter()
            .filter_map(|path| path.strip_prefix(&root).ok())
            .collect();
        let (files, left_alone): (Vec<String>, Vec<String>) = status
            .keys()
            .cloned()
            .partition(|path| session.contains(Path::new(path)));
        let remote_base = format!("{remote}/{base}");
        let base_ref = if git(&["rev-parse", "--verify", "-q", &remote_base]).is_ok() {
            remote_base
        } else {
            base.clone()
        };
        let ahead: u64 = git(&["rev-list", "--count", &format!("{base_ref}..HEAD")])
            .ok()
            .and_then(|count| count.parse().ok())
            .unwrap_or(0);
        if files.is_empty() && ahead == 0 {
            return Err(format!(
                "Nothing for a pull request: no file this session changed is uncommitted, and nothing is committed ahead of {base}."
            ));
        }
        let mut diff = if ahead > 0 {
            git(&["diff", "--no-color", &format!("{base_ref}...HEAD")]).unwrap_or_default()
        } else {
            String::new()
        };
        for file in &files {
            let part = if status.get(file).is_some_and(|code| code == "??") {
                let text = self.0.read_text(&root.join(file)).unwrap_or_default();
                format!("\nnew file {file}:\n{text}")
            } else {
                git(&["diff", "--no-color", "HEAD", "--", file]).unwrap_or_default()
            };
            diff.push_str(&part);
            diff.push('\n');
        }
        let diff: String = diff.chars().take(DIFF_CHARS).collect();
        let template = TEMPLATES
            .iter()
            .find_map(|path| self.0.read_text(&root.join(path)).ok())
            .map(|template| {
                format!("\n\n# Template\nThe body follows this pull request template:\n{template}")
            })
            .unwrap_or_default();
        let context = format!("# Diff against {base}\n{diff}{template}");
        let taken = git(&["branch", "--format=%(refname:short)"]).unwrap_or_default();
        let plan = Plan {
            root,
            remote,
            base,
            branch: current,
            new_branch,
            taken: taken.lines().map(str::to_owned).collect(),
            files,
            left_alone,
            message: String::new(),
            title: String::new(),
            body: String::new(),
            draft,
            gh,
        };
        Ok((plan, context))
    }

    /// The branch the remote's `HEAD` names, else `main` or `master`.
    fn default_branch(&self, root: &Path, remote: &str) -> String {
        let head = format!("refs/remotes/{remote}/HEAD");
        if let Ok(found) = self.git(root, &["symbolic-ref", "--short", &head])
            && let Some(branch) = found.strip_prefix(&format!("{remote}/"))
        {
            return branch.to_owned();
        }
        ["main", "master"]
            .into_iter()
            .find(|branch| {
                self.git(root, &["rev-parse", "--verify", "-q", branch])
                    .is_ok()
            })
            .unwrap_or("main")
            .to_owned()
    }

    /// A job's steps, run in order until one fails.
    fn steps(&self, Job(plan, action): &Job, doing: &dyn Fn(String)) -> Finish {
        let git = |args: &[&str]| self.git(&plan.root, args);
        let gh = |args: &[&str]| self.run(&plan.gh, &plan.root, args);
        let failed = |error: String, hint: String| Finish::Failed { error, hint };
        let branch = plan.branch.as_str();
        if *action == Action::UpdateBody {
            doing(format!(
                "Updating the description of the pull request for {branch}"
            ));
            return match gh(&["pr", "edit", branch, "--body", &plan.body]) {
                Ok(_) => Finish::Updated {
                    url: open_pr(&gh, branch).map(|(url, _)| url).unwrap_or_default(),
                },
                Err(error) => failed(
                    error,
                    "The description is as it was; edit it on GitHub or run /pr again.".to_owned(),
                ),
            };
        }
        if plan.new_branch {
            doing(format!("Creating branch {branch}"));
            if let Err(error) = git(&["switch", "-c", branch]) {
                return failed(
                    error,
                    "Nothing was changed. Fix it, then run /pr again.".to_owned(),
                );
            }
        }
        if !plan.files.is_empty() {
            let paths: Vec<&str> = plan.files.iter().map(String::as_str).collect();
            doing(format!("Staging {}", plan.files.join(", ")));
            if let Err(error) = git(&[&["add", "-A", "--"], &paths[..]].concat()) {
                return failed(
                    error,
                    format!(
                        "On branch {branch}, nothing is committed. Fix it, then run /pr again."
                    ),
                );
            }
            doing("Committing".to_owned());
            if let Err(error) =
                git(&[&["commit", "-q", "-m", &plan.message, "--"], &paths[..]].concat())
            {
                let hint = format!(
                    "The files are staged on {branch}, not committed. Fix what stopped the commit (such as a hook), then run /pr again."
                );
                return failed(error, hint);
            }
        }
        let commit = git(&["rev-parse", "--short", "HEAD"]).unwrap_or_default();
        if *action == Action::CommitOnly {
            return Finish::Committed { commit };
        }
        doing(format!("Pushing {branch} to {}", plan.remote));
        if let Err(error) = git(&["push", "-q", "-u", &plan.remote, branch]) {
            let hint = format!(
                "Committed {commit} on {branch}, not pushed. Fix the push, then run /pr again."
            );
            return failed(error, hint);
        }
        doing("Looking for an open pull request".to_owned());
        if let Some((url, body)) = open_pr(&gh, branch) {
            return Finish::Existing { url, body };
        }
        doing(format!("Opening the pull request into {}", plan.base));
        let mut create = vec![
            "pr",
            "create",
            "--base",
            &plan.base,
            "--head",
            branch,
            "--title",
            &plan.title,
            "--body",
            &plan.body,
        ];
        if plan.draft {
            create.push("--draft");
        }
        match gh(&create) {
            Ok(out) => Finish::Opened {
                url: out
                    .lines()
                    .rev()
                    .map(str::trim)
                    .find(|line| !line.is_empty())
                    .unwrap_or_default()
                    .to_owned(),
            },
            Err(error) => failed(
                error,
                format!("{branch} is pushed. Run /pr again, or `gh pr create`."),
            ),
        }
    }
}

/// The URL and description of the pull request open for `branch`, if one is.
fn open_pr(
    gh: &dyn Fn(&[&str]) -> Result<String, String>,
    branch: &str,
) -> Option<(String, String)> {
    let view = gh(&["pr", "view", branch, "--json", "url,body,state"]).ok()?;
    let view: serde_json::Value = serde_json::from_str(&view).ok()?;
    (view.get("state")?.as_str()? == "OPEN").then_some(())?;
    let url = view.get("url")?.as_str()?.to_owned();
    let body = view.get("body").and_then(|body| body.as_str());
    Some((url, body.unwrap_or_default().to_owned()))
}
