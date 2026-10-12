//! FH-12: open a pull request from the session.
//!
//! `/pr` (`/pr draft`, `/pr --base <branch>`) turns the session's work into
//! a pull request. The plugin keeps, with the session, the files its tool
//! calls changed and the test commands its `shell` calls ran, with how
//! they ended. `/pr` checks the repository, its remote and `gh`, then asks
//! a writer model (`[open-pr] model`, else the agent's) for the title,
//! commit message and description from the user's requests, the diff
//! against the base and those tests, following the repository's PR
//! template when it has one. A review question shows the branch, files,
//! message, title and description, each editable through a follow-up
//! question, with Publish, Commit only and Cancel. Nothing is committed or
//! pushed before that answer. The steps then run off the main thread as an
//! effect of the plugin's own (`open-pr`), each one shown as it starts; a
//! failing step stops there and says how to go on. A pull request already
//! open for the branch gets the new commits and, if the user agrees, the
//! new description. The URL is kept with the session, in `/resume`'s line
//! and shown again when the session is resumed.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use futures::channel::mpsc;
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::{CompletionRequest, Message};
use rig_harness::rig_core::effect::EffectKind;
use rig_harness::rig_core::tool::{ExitStatus, FilesChanged};
use rig_sessions::ResumeLabel;
use rig_tools::process::Launcher;
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
    `## Changes` and `## How it was tested`. That last section names only the test \
    commands listed as run, each with whether it passed; when none are listed it says \
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

/// Adds `/pr`.
#[derive(Default)]
pub struct OpenPrPlugin;

impl Plugin for OpenPrPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<OpenPrConfig>(NAME)
            .init_resource::<ProjectRoot>()
            .init_resource::<Processes>()
            .insert_resource(SessionWork::default())
            .add_observer(note_work)
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
    /// A `shell` command with one of these words in it ran tests.
    pub test_words: Vec<String>,
}

impl Default for OpenPrConfig {
    fn default() -> Self {
        Self {
            model: None,
            gh: "gh".to_owned(),
            test_words: ["test", "spec", "pytest", "nextest"]
                .map(str::to_owned)
                .to_vec(),
        }
    }
}

/// What the session did that a pull request is written from: the files
/// its tool calls changed (absolute paths), the test commands its `shell`
/// calls ran, and the pull requests opened. Saved with the session.
#[derive(Resource, Reflect, Clone, Debug, Default, Serialize, Deserialize)]
#[reflect(opaque, Resource, Saved, Default, Serialize, Deserialize)]
pub struct SessionWork {
    /// Every file a tool call said it changed.
    pub changed: BTreeSet<String>,
    /// Every test command run, in order.
    pub tests: Vec<TestRun>,
    /// The pull requests opened or updated, oldest first.
    pub prs: Vec<String>,
}

/// A test command a `shell` call ran.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TestRun {
    /// The command line.
    pub command: String,
    /// Whether it exited with 0.
    pub passed: bool,
}

/// What `/pr` will do: where, on which branch, with what text.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Plan {
    root: PathBuf,
    remote: String,
    base: String,
    branch: String,
    /// Whether `branch` is made from where the repository stands.
    new_branch: bool,
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

/// What a follow-up question edits.
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

/// Notes the files each tool call changed and each test command `shell`
/// ran.
fn note_work(
    published: On<Add<Published>>,
    calls: Query<(&Published, &ToolRequest)>,
    config: Res<OpenPrConfig>,
    mut work: ResMut<SessionWork>,
) {
    let Ok((Published(facts), request)) = calls.get(published.entity) else {
        return;
    };
    if let Ok(Some(files)) = facts.get::<FilesChanged>() {
        work.changed.extend(files.paths().map(str::to_owned));
    }
    let call = &request.call.function;
    let command = call
        .arguments
        .get("command")
        .and_then(|command| command.as_str());
    if let (Ok(Some(status)), Some(command), "shell") =
        (facts.get::<ExitStatus>(), command, call.name.as_str())
        && config
            .test_words
            .iter()
            .any(|word| command.contains(word.as_str()))
    {
        let passed = status.success();
        work.tests.push(TestRun {
            command: command.to_owned(),
            passed,
        });
    }
}

/// `/pr [draft] [--base <branch>]`: checks what it needs and asks the
/// writer for the text.
fn pr(
    In(args): In<CommandArgs>,
    agents: Query<(&Conversation, Option<&WorkDir>, Has<ActiveTurn>)>,
    (project, processes, config, work): (
        Res<ProjectRoot>,
        Res<Processes>,
        Res<OpenPrConfig>,
        Res<SessionWork>,
    ),
    models: Res<Models>,
    answerers: Answerers,
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
    let requests: Vec<String> = (0..conversation.messages().len())
        .filter_map(|at| typed(conversation, at))
        .collect();
    let dir = dir.map_or(&project.0, |WorkDir(dir)| dir);
    let launcher = processes.launcher();
    let prepared = parse(&args.args)
        .and_then(|(draft, base)| prepare(&launcher, dir, &config, &work, draft, base));
    let (plan, diff) = match prepared {
        Ok(prepared) => prepared,
        Err(why) => {
            notices.write(Notice::error(agent, why));
            return;
        }
    };
    let prompt = writer_prompt(&plan, &requests, &diff, &work.tests);
    let schema = schemars::schema_for!(Written);
    let request = CompletionRequest::new(Message::user(prompt))
        .preamble(PREAMBLE)
        .output_schema(schema);
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
        Writing(plan, requests),
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

/// The plan without its text, and the diff the writer reads, or what is
/// missing.
fn prepare(
    launcher: &Launcher,
    dir: &Path,
    config: &OpenPrConfig,
    work: &SessionWork,
    draft: bool,
    base: Option<String>,
) -> Result<(Plan, String), String> {
    let root = git(launcher, dir, &["rev-parse", "--show-toplevel"])
        .map(PathBuf::from)
        .map_err(|_| format!("/pr needs a git repository: {} is in none.", dir.display()))?;
    let git = |args: &[&str]| git(launcher, &root, args);
    let remotes = git(&["remote"]).unwrap_or_default();
    let remote = remotes
        .lines()
        .find(|remote| *remote == "origin")
        .or_else(|| remotes.lines().next())
        .ok_or("/pr needs a remote to push to: this repository has none (git remote add origin <url>).")?
        .to_owned();
    run(launcher, &config.gh, &root, &["--version"]).map_err(|_| {
        format!(
            "/pr needs the GitHub CLI (`{}`), or an equivalent set as `[open-pr] gh`: it does not run.",
            config.gh
        )
    })?;
    let current = git(&["branch", "--show-current"]).unwrap_or_default();
    let default = default_branch(&git, &remote);
    let base = base.unwrap_or(default.clone());
    let new_branch = current.is_empty() || current == default || current == base;
    let status = status(launcher, &root)?;
    let session: BTreeSet<String> = work
        .changed
        .iter()
        .filter_map(|path| relative(&root, Path::new(path)))
        .collect();
    let (files, left_alone): (Vec<String>, Vec<String>) = status
        .keys()
        .cloned()
        .partition(|path| session.contains(path));
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
            let text = std::fs::read_to_string(root.join(file)).unwrap_or_default();
            format!("\nnew file {file}:\n{text}")
        } else {
            git(&["diff", "--no-color", "HEAD", "--", file]).unwrap_or_default()
        };
        diff.push_str(&part);
        diff.push('\n');
    }
    let diff = diff.chars().take(DIFF_CHARS).collect();
    let plan = Plan {
        root,
        remote,
        base,
        branch: current,
        new_branch,
        files,
        left_alone,
        message: String::new(),
        title: String::new(),
        body: String::new(),
        draft,
        gh: config.gh.clone(),
    };
    Ok((plan, diff))
}

/// What the writer is given.
fn writer_prompt(plan: &Plan, requests: &[String], diff: &str, tests: &[TestRun]) -> String {
    let requests = requests
        .iter()
        .map(|request| format!("- {request}"))
        .collect::<Vec<_>>();
    let tests = if tests.is_empty() {
        "No test command ran in this session.".to_owned()
    } else {
        let lines = tests.iter().map(|test| {
            let how = if test.passed { "passed" } else { "failed" };
            format!("- `{}`: {how}", test.command)
        });
        lines.collect::<Vec<_>>().join("\n")
    };
    let template = TEMPLATES
        .iter()
        .find_map(|path| std::fs::read_to_string(plan.root.join(path)).ok())
        .map(|template| {
            format!("\n\n# Template\nThe body follows this pull request template:\n{template}")
        })
        .unwrap_or_default();
    format!(
        "# What the user asked for\n{}\n\n# Files committed\n{}\n\n# Tests run in the session\n{tests}\n\n# Diff against {}\n{diff}{template}",
        requests.join("\n"),
        plan.files.join("\n"),
        plan.base
    )
}

/// Takes the writer's answer, or a plain draft when it failed, and shows
/// the review.
fn written(
    settled: On<Add<Settled>>,
    calls: Query<(&Writing, &Outcome, &ForAgent)>,
    processes: Res<Processes>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((Writing(plan, requests), outcome, &ForAgent(agent))) = calls.get(settled.entity) else {
        return;
    };
    let mut plan = plan.clone();
    let text = outcome
        .completion()
        .map(|response| response.text())
        .unwrap_or_default();
    let json = text
        .find('{')
        .zip(text.rfind('}'))
        .and_then(|(from, to)| text.get(from..=to));
    match json.and_then(|json| serde_json::from_str::<Written>(json).ok()) {
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
                .map_or("Changes", |request| {
                    request.lines().next().unwrap_or("Changes")
                })
                .chars()
                .take(72)
                .collect();
            plan.message.clone_from(&title);
            plan.body = format!(
                "## Summary\n{}\n\n## Changes\n{}\n\n## How it was tested\nNot tested.",
                requests.join("\n"),
                plan.files
                    .iter()
                    .map(|file| format!("- {file}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            );
            plan.title = title;
        }
    }
    if plan.new_branch {
        plan.branch = free_branch(&processes.launcher(), &plan.root, &plan.title);
    }
    commands.spawn((review(&plan), ForAgent(agent), Review(plan)));
}

/// The review question.
fn review(plan: &Plan) -> Ask {
    let files = |files: &[String]| {
        if files.is_empty() {
            "  (none)".to_owned()
        } else {
            files
                .iter()
                .map(|file| format!("  {file}"))
                .collect::<Vec<_>>()
                .join("\n")
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
        let request = EffectRequest {
            kind,
            handler: None,
        };
        commands.spawn((
            Name::new("pull request"),
            request,
            ForAgent(agent),
            Job(plan.clone(), action),
        ));
    };
    if let Some(Review(plan)) = review_of {
        let field = match answer {
            Some(Answer::Choice(choice)) => match choice.as_str() {
                "publish" => return job(plan, Action::Publish),
                "commit" => return job(plan, Action::CommitOnly),
                "title" => Field::Title,
                "body" => Field::Body,
                "message" => Field::Message,
                "branch" => Field::Branch,
                "files" => Field::Files,
                _ => return cancelled(agent, &mut notices),
            },
            _ => return cancelled(agent, &mut notices),
        };
        let (title, now) = match field {
            Field::Title => ("The title", plan.title.clone()),
            Field::Body => ("The description", plan.body.clone()),
            Field::Message => ("The commit message", plan.message.clone()),
            Field::Branch => ("The branch", plan.branch.clone()),
            Field::Files => (
                "The files to commit, separated by spaces",
                plan.files.join(" "),
            ),
        };
        let ask = Ask::new(title)
            .body(now)
            .free_text()
            .choice("keep", "Keep it");
        commands.spawn((ask, ForAgent(agent), Editing(plan.clone(), field)));
    } else if let Some(Editing(plan, field)) = editing {
        let mut plan = plan.clone();
        if let Some(Answer::Text(text)) = answer {
            edit(&mut plan, *field, text.trim(), agent, &mut notices);
        }
        commands.spawn((self::review(&plan), ForAgent(agent), Review(plan)));
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

/// Says nothing was done.
fn cancelled(agent: Entity, notices: &mut MessageWriter<Notice>) {
    notices.write(Notice::info(
        agent,
        "Cancelled: nothing was committed or pushed.",
    ));
}

/// Sets `field` of the plan to what the user typed.
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
            for file in text.split([' ', ',']).filter(|file| !file.is_empty()) {
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
    processes: Res<Processes>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let Ok(job) = jobs.get(started.entity) else {
        return;
    };
    let (sender, steps) = mpsc::unbounded();
    let (thread_job, launcher, failed) = (job.clone(), processes.launcher(), sender.clone());
    let thread = std::thread::Builder::new()
        .name(NAME.to_owned())
        .spawn(move || {
            let say = |step| {
                sender.unbounded_send(step).ok();
            };
            let finish = steps_of(&thread_job, &launcher, &|doing: String| {
                say(Step::Doing(doing))
            });
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
    let pool = bevy_tasks::IoTaskPool::get_or_init(bevy_tasks::TaskPool::default);
    commands.spawn((
        Worker(started.entity, job.clone()),
        Running::stream(pool, &wake, steps),
    ));
}

/// The job's steps, run in order until one fails.
fn steps_of(Job(plan, action): &Job, launcher: &Launcher, doing: &dyn Fn(String)) -> Finish {
    let git = |args: &[&str]| git(launcher, &plan.root, args);
    let gh = |args: &[&str]| run(launcher, &plan.gh, &plan.root, args);
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
                format!("On branch {branch}, nothing is committed. Fix it, then run /pr again."),
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
    let open = open_pr(&gh, branch).map(|(url, body)| Finish::Existing { url, body });
    if let Some(open) = open {
        return open;
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

/// The URL and description of the pull request open for `branch`, if one is.
fn open_pr(
    gh: &dyn Fn(&[&str]) -> Result<String, String>,
    branch: &str,
) -> Option<(String, String)> {
    let view = gh(&["pr", "view", branch, "--json", "url,body,state"]).ok()?;
    let view: serde_json::Value = serde_json::from_str(&view).ok()?;
    (view.get("state")?.as_str()? == "OPEN").then_some(())?;
    let url = view.get("url")?.as_str()?.to_owned();
    let body = view
        .get("body")
        .and_then(|body| body.as_str())
        .unwrap_or_default();
    Some((url, body.to_owned()))
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
    decision.decide(Outcome::custom(
        serde_json::to_value(finish).unwrap_or_default(),
    ));
    commands.entity(reported.entity).despawn();
    let mut keep = |url: &str| {
        if !url.is_empty() && !work.prs.iter().any(|pr| pr == url) {
            work.prs.push(url.to_owned());
        }
    };
    match finish {
        Finish::Opened { url } => {
            keep(url);
            notices.write(Notice::info(agent, format!("Opened {url}")));
        }
        Finish::Updated { url } => {
            keep(url);
            notices.write(Notice::info(
                agent,
                format!("Updated the description of {url}"),
            ));
        }
        Finish::Committed { commit } => {
            let text = format!(
                "Committed {commit} on {}; nothing was pushed. Run /pr again to publish.",
                plan.branch
            );
            notices.write(Notice::info(agent, text));
        }
        Finish::Existing { url, body } => {
            keep(url);
            notices.write(Notice::info(
                agent,
                format!("Pushed the new commits to {url}"),
            ));
            if body.trim() != plan.body.trim() {
                let ask =
                    Ask::yes_no("Update the pull request's description?").body(plan.body.clone());
                commands.spawn((ask, ForAgent(agent), UpdateBody(plan.clone())));
            }
        }
        Finish::Failed { error, hint } => {
            notices.write(Notice::error(
                agent,
                format!("/pr stopped: {error}\n{hint}"),
            ));
        }
    }
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

/// The branch the remote's `HEAD` names, else `main` or `master`.
fn default_branch(git: &dyn Fn(&[&str]) -> Result<String, String>, remote: &str) -> String {
    let head = format!("refs/remotes/{remote}/HEAD");
    if let Ok(found) = git(&["symbolic-ref", "--short", &head])
        && let Some(branch) = found.strip_prefix(&format!("{remote}/"))
    {
        return branch.to_owned();
    }
    ["main", "master"]
        .into_iter()
        .find(|branch| git(&["rev-parse", "--verify", "-q", branch]).is_ok())
        .unwrap_or("main")
        .to_owned()
}

/// `rig/<slug of the title>`, numbered when taken.
fn free_branch(launcher: &Launcher, root: &Path, title: &str) -> String {
    let lower = title.to_lowercase();
    let words = lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|word| !word.is_empty());
    let mut slug = String::new();
    for word in words {
        if slug.len() + word.len() > 40 {
            break;
        }
        if !slug.is_empty() {
            slug.push('-');
        }
        slug.push_str(word);
    }
    let slug = if slug.is_empty() {
        "changes".to_owned()
    } else {
        slug
    };
    let taken = |branch: &str| {
        git(
            launcher,
            root,
            &[
                "rev-parse",
                "--verify",
                "-q",
                &format!("refs/heads/{branch}"),
            ],
        )
        .is_ok()
    };
    let first = format!("rig/{slug}");
    std::iter::once(first.clone())
        .chain((2..100).map(|n| format!("rig/{slug}-{n}")))
        .find(|branch| !taken(branch))
        .unwrap_or(first)
}

/// The files `git status` shows, relative to the repository's top, with
/// their two-letter status.
fn status(launcher: &Launcher, root: &Path) -> Result<BTreeMap<String, String>, String> {
    let out = run(
        launcher,
        "git",
        root,
        &["status", "--porcelain=v1", "-z", "--untracked-files=all"],
    )?;
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

/// The absolute `path` relative to the repository's top `root`, if in it.
fn relative(root: &Path, path: &Path) -> Option<String> {
    let parent = path.parent()?.canonicalize().ok()?;
    let root = root.canonicalize().ok()?;
    let full = parent.join(path.file_name()?);
    Some(
        full.strip_prefix(root)
            .ok()?
            .to_string_lossy()
            .replace('\\', "/"),
    )
}

/// Runs git in `dir`: its trimmed output, or its error.
fn git(launcher: &Launcher, dir: &Path, args: &[&str]) -> Result<String, String> {
    run(launcher, "git", dir, args).map(|out| out.trim().to_owned())
}

/// Runs `program` in `dir` as every child process starts: its output, or
/// its error.
fn run(launcher: &Launcher, program: &str, dir: &Path, args: &[&str]) -> Result<String, String> {
    let output = launcher
        .command(program)
        .current_dir(dir)
        .args(args)
        .output()
        .map_err(|error| format!("cannot run {program}: {error}"))?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    } else {
        let error = String::from_utf8_lossy(&output.stderr);
        Err(format!(
            "{program} {}: {}",
            args.first().copied().unwrap_or_default(),
            error.trim()
        ))
    }
}
