//! OC-18: mention the agent on GitHub. `rig github`, run inside a CI job,
//! reads the event that started the job (`GITHUB_EVENT_PATH`), and when a
//! comment by a user with write access mentions the trigger (`@rig`), it
//! replies "Working on it…", gives the agent the thread (and a pull
//! request's diff and the review comment's place), keeps the reply up to
//! date with the agent's latest step, then commits and pushes what the
//! agent changed (a new `rig/issue-<n>` branch for an issue, the pull
//! request's branch for a pull request) and edits the reply into the
//! final answer. The session of each thread is kept on the `rig/sessions`
//! branch, so a later comment continues the conversation.
//!
//! The repository's `.github/rig.toml` configures it, in a `[github]`
//! table: `trigger` (default `@rig`), `allow` (the tools that need
//! approval which may run anyway: `"shell"`, or `"shell:cargo test"` for
//! commands that start so) and `open_pr` (open a draft pull request for an
//! issue's branch instead of linking to one).

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use bevy_ecs::system::SystemParam;
use bevy_log::{error, info, warn};
use rig_harness::prelude::*;
use rig_harness::rig_core::tool::{Scope, Touches};
use rig_harness::rig_memory::{DirStore, LoadError, Store};
use rig_harness::rig_tools::process::Launcher;
use rig_reqwest::reqwest;
use serde::Deserialize;
use serde_json::{Value, json};

#[cfg(test)]
mod tests;

/// The front's name and its command, `rig github`.
pub const FRONT: &str = "github";

/// The branch every thread's session is kept on, under a directory per
/// thread (`thread-<n>`).
pub const SESSIONS_BRANCH: &str = "rig/sessions";

/// Where the CI job says the event and the API are, read from the
/// environment (`GITHUB_EVENT_PATH`, `GITHUB_API_URL`, `GITHUB_TOKEN`,
/// `GITHUB_SERVER_URL`, `GITHUB_RUN_ID`) unless the app inserted one.
#[derive(Resource, Clone, Debug, Default)]
pub struct GithubEnv {
    /// The event payload's file.
    pub event_path: PathBuf,
    /// The REST API's base, such as `https://api.github.com`.
    pub api_url: String,
    /// The job's token.
    pub token: String,
    /// The web base, such as `https://github.com`.
    pub server_url: String,
    /// The run's id, for the link to it.
    pub run_id: Option<String>,
}

impl GithubEnv {
    /// The job's environment.
    pub fn from_env() -> Self {
        let var = |name: &str| std::env::var(name).ok();
        Self {
            event_path: var("GITHUB_EVENT_PATH").unwrap_or_default().into(),
            api_url: var("GITHUB_API_URL").unwrap_or_else(|| "https://api.github.com".to_owned()),
            token: var("GITHUB_TOKEN").unwrap_or_default(),
            server_url: var("GITHUB_SERVER_URL").unwrap_or_else(|| "https://github.com".to_owned()),
            run_id: var("GITHUB_RUN_ID"),
        }
    }
}

/// The `[github]` table of the repository's `.github/rig.toml`.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RepoConfig {
    /// What a comment mentions to start the agent; `@rig` by default.
    pub trigger: Option<String>,
    /// Tools that need approval which run anyway: a tool's name, or
    /// `shell:<start>` for the `shell` commands that start so.
    pub allow: Vec<String>,
    /// Open a draft pull request for an issue's branch.
    pub open_pr: bool,
}

impl RepoConfig {
    /// The project's configuration, or the defaults.
    pub fn of(root: &Path) -> Self {
        Settings::read(&root.join(".github/rig.toml"))
            .get("github")
            .unwrap_or_default()
    }

    /// What a comment mentions to start the agent.
    pub fn trigger(&self) -> &str {
        self.trigger.as_deref().unwrap_or("@rig")
    }

    /// Whether the call of `tool` with `arguments`, which needs approval,
    /// may run anyway.
    pub fn allows(&self, tool: &str, arguments: &serde_json::Map<String, Value>) -> bool {
        let command = arguments.get("command").and_then(Value::as_str);
        self.allow.iter().any(|rule| match rule.split_once(':') {
            Some((name, start)) => {
                name == tool && command.is_some_and(|it| it.trim_start().starts_with(start))
            }
            None => rule == tool,
        })
    }
}

/// The comment that mentioned the agent, and where.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mention {
    /// `owner/name`.
    pub repo: String,
    /// The issue's or pull request's number.
    pub number: u64,
    /// Whether it is on a pull request.
    pub pull_request: bool,
    /// The issue's or pull request's title.
    pub title: String,
    /// Its body.
    pub body: String,
    /// Who commented.
    pub author: String,
    /// What the comment says.
    pub comment: String,
    /// A review comment's file and line.
    pub place: Option<(String, Option<u64>)>,
    /// A review comment's thread: the id of its first comment.
    pub review_thread: Option<u64>,
}

impl Mention {
    /// The comment an event payload carries: a new issue comment, pull
    /// request review comment or pull request review; none for another
    /// event or action.
    pub fn of_event(event: &Value) -> Option<Self> {
        let text = |at: &str| {
            event
                .pointer(at)
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned()
        };
        let action = text("/action");
        let (comment, author, thread, place) = if event.get("review").is_some() {
            (
                text("/review/body"),
                text("/review/user/login"),
                "/pull_request",
                None,
            )
        } else if event.get("pull_request").is_some() {
            let line = event.pointer("/comment/line").and_then(Value::as_u64);
            (
                text("/comment/body"),
                text("/comment/user/login"),
                "/pull_request",
                Some((text("/comment/path"), line)),
            )
        } else {
            (
                text("/comment/body"),
                text("/comment/user/login"),
                "/issue",
                None,
            )
        };
        if !matches!(action.as_str(), "created" | "submitted") || comment.is_empty() {
            return None;
        }
        let thread = event.pointer(thread)?;
        Some(Self {
            repo: text("/repository/full_name"),
            number: thread.get("number")?.as_u64()?,
            pull_request: place.is_some()
                || event.get("review").is_some()
                || thread.get("pull_request").is_some(),
            title: thread
                .get("title")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            body: thread
                .get("body")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            author,
            comment,
            review_thread: place.as_ref().and_then(|_| {
                let id = |at: &str| event.pointer(at).and_then(Value::as_u64);
                id("/comment/in_reply_to_id").or_else(|| id("/comment/id"))
            }),
            place,
        })
    }

    /// The comment of the event payload in the file `path`.
    pub fn read(path: &Path) -> Option<Self> {
        let bytes = fs::read(path).ok()?;
        Self::of_event(&serde_json::from_slice(&bytes).ok()?)
    }

    /// The key of its thread's session on [`SESSIONS_BRANCH`].
    pub fn thread(&self) -> String {
        format!("thread-{}", self.number)
    }
}

/// The run's mention, read from the event payload, and its configuration.
#[derive(Resource, Clone, Debug)]
struct Run {
    mention: Option<Mention>,
    config: RepoConfig,
    root: PathBuf,
}

/// Where the progress comment and the branch stand, once set up.
#[derive(Resource, Clone, Debug)]
struct Work {
    /// The reply being edited.
    comment: u64,
    /// The branch the agent works on.
    branch: String,
    /// The branch an issue's pull request would go to.
    base: Option<String>,
}

/// What the run is waiting for or showing in the reply.
#[derive(Resource, Debug, Default)]
struct Progress {
    /// The latest step, not shown yet.
    pending: Option<String>,
    /// The tool calls refused because they need approval.
    denied: Vec<String>,
    /// Error notices about the agent.
    errors: Vec<String>,
    /// The turn's end, once known: its answer, and whether it failed.
    ended: Option<(Option<String>, bool)>,
    /// Whether the final edit started.
    finishing: bool,
}

/// The gate that refuses calls needing approval, for others to order
/// against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GithubSystems;

/// On the entity whose [`KeepRunning`] holds the run until the reply is
/// final.
#[derive(Component)]
struct Job;

/// On a progress edit in flight.
#[derive(Component)]
struct Edit;

/// What setting up found.
enum Setup {
    /// Nothing to do: no mention, or not by a writer.
    Ignore(String),
    /// The agent works with this prompt.
    Go(Work, String),
    /// The reply says why it stopped; `true` for a failure.
    Stop(bool),
}

/// The `rig github` front.
#[derive(Default)]
pub struct GithubMentionPlugin;

impl Plugin for GithubMentionPlugin {
    fn build(&self, app: &mut App) {
        let env = app
            .world()
            .get_resource::<GithubEnv>()
            .cloned()
            .unwrap_or_else(GithubEnv::from_env);
        let root = app
            .world()
            .get_resource::<ProjectRoot>()
            .map(|root| root.0.clone())
            .unwrap_or_default();
        // A run of `rig github` keeps the thread's session, restored from
        // `rig/sessions` into the run's session directory.
        let processes = app.world().get_resource::<Processes>().cloned();
        let store = Front::new(FRONT)
            .on_command(
                FRONT,
                "",
                "Answer the GitHub comment that started this CI job",
            )
            .session_store(move |dir| {
                let at = dir.path().join("github-thread");
                let thread = Mention::read(&env.event_path).map(|mention| mention.thread());
                if let Some(thread) = thread {
                    let launcher = processes
                        .as_ref()
                        .map_or_else(Launcher::default, Processes::launcher);
                    Git::new(launcher.in_dir(&root))
                        .restore(&thread, &at)
                        .map_err(|why| LoadError::Io(std::io::Error::other(why)))?;
                }
                Ok(Arc::new(DirStore::open(at)?) as Arc<dyn Store>)
            });
        app.add_front(store)
            .configure_sets(AgentPass, GithubSystems.in_set(Stage::Gate))
            .add_systems(
                AgentPass,
                needs_approval
                    .in_set(GithubSystems)
                    .run_if(resource_exists::<Run>),
            );
    }

    fn finish(&self, app: &mut App) {
        if !app.runs_front(FRONT) {
            return;
        }
        let env = app
            .world()
            .get_resource::<GithubEnv>()
            .cloned()
            .unwrap_or_else(GithubEnv::from_env);
        let root = app
            .world()
            .get_resource::<ProjectRoot>()
            .map(|root| root.0.clone())
            .unwrap_or_default();
        let run = Run {
            mention: Mention::read(&env.event_path),
            config: RepoConfig::of(&root),
            root,
        };
        // The model's commands never see the token.
        if let Some(processes) = app.world().get_resource::<Processes>() {
            processes.set(processes.launcher().without_env(["GITHUB_TOKEN"]));
        }
        app.insert_resource(run)
            .insert_resource(env)
            .init_resource::<Progress>()
            .run_headless(None)
            .add_systems(Startup, set_up)
            .add_systems(
                Update,
                (show_progress, finish_reply).chain().after(AgentLoop),
            )
            .add_observer(started)
            .add_observer(denied)
            .add_observer(set_up_done)
            .add_observer(edited)
            .add_observer(finished)
            .add_observer(turn_ended);
    }
}

/// The REST API, used from a blocking thread.
#[derive(Clone)]
struct Api {
    client: reqwest::Client,
    runtime: TokioRuntime,
    env: GithubEnv,
    repo: String,
}

impl Api {
    /// Sends `method path` with `body`, and returns the answer's text.
    fn send(
        &self,
        method: &str,
        path: &str,
        body: Option<Value>,
        accept: &str,
    ) -> Result<String, String> {
        let method =
            reqwest::Method::from_bytes(method.as_bytes()).map_err(|why| why.to_string())?;
        let url = format!(
            "{}/repos/{}{path}",
            self.env.api_url.trim_end_matches('/'),
            self.repo
        );
        let mut request = self
            .client
            .request(method, &url)
            .header("accept", accept)
            .header("user-agent", "rig")
            .header("x-github-api-version", "2022-11-28")
            .bearer_auth(&self.env.token);
        if let Some(body) = body {
            request = request.json(&body);
        }
        self.runtime.0.block_on(async move {
            let response = request
                .send()
                .await
                .map_err(|why| format!("{url}: {why}"))?;
            let status = response.status();
            let text = response.text().await.map_err(|why| why.to_string())?;
            if status.is_success() {
                Ok(text)
            } else {
                Err(format!("{url}: {status}: {text}"))
            }
        })
    }

    /// The JSON answer to `GET path`.
    fn get(&self, path: &str) -> Result<Value, String> {
        let text = self.send("GET", path, None, "application/vnd.github+json")?;
        serde_json::from_str(&text).map_err(|why| why.to_string())
    }

    /// Posts a comment on the thread and returns its id.
    fn comment(&self, number: u64, body: &str) -> Result<u64, String> {
        let text = self.send(
            "POST",
            &format!("/issues/{number}/comments"),
            Some(json!({ "body": body })),
            "application/vnd.github+json",
        )?;
        let answer: Value = serde_json::from_str(&text).map_err(|why| why.to_string())?;
        answer
            .get("id")
            .and_then(Value::as_u64)
            .ok_or_else(|| "the new comment has no id".to_owned())
    }

    /// Edits comment `id` to say `body`.
    fn edit(&self, id: u64, body: &str) -> Result<(), String> {
        let path = format!("/issues/comments/{id}");
        let body = Some(json!({ "body": body }));
        self.send("PATCH", &path, body, "application/vnd.github+json")
            .map(drop)
    }
}

/// Git in the project, its commits made as `rig`.
#[derive(Clone)]
struct Git(Launcher);

impl Git {
    fn new(launcher: Launcher) -> Self {
        let mut launcher = launcher;
        for (name, value) in [
            ("GIT_AUTHOR_NAME", "rig"),
            ("GIT_COMMITTER_NAME", "rig"),
            ("GIT_AUTHOR_EMAIL", "rig@users.noreply.github.com"),
            ("GIT_COMMITTER_EMAIL", "rig@users.noreply.github.com"),
        ] {
            launcher = launcher.with_env(name, value);
        }
        Self(launcher)
    }

    /// Runs `git args` with `env` and returns its output.
    fn run(&self, args: &[&str], env: &[(&str, &Path)]) -> Result<Vec<u8>, String> {
        let mut command = self.0.command("git");
        command.args(args).envs(env.iter().copied());
        let output = command.output().map_err(|why| format!("git: {why}"))?;
        if output.status.success() {
            Ok(output.stdout)
        } else {
            let error = String::from_utf8_lossy(&output.stderr);
            Err(format!("git {}: {}", args.join(" "), error.trim()))
        }
    }

    /// Runs `git args` and returns its output's text, trimmed.
    fn text(&self, args: &[&str]) -> Result<String, String> {
        let out = self.run(args, &[])?;
        Ok(String::from_utf8_lossy(&out).trim().to_owned())
    }

    /// The remote branch's commit, fetched, if it exists.
    fn fetch(&self, branch: &str) -> Option<String> {
        let from = format!("refs/heads/{branch}");
        self.run(&["fetch", "-q", "origin", &from], &[]).ok()?;
        self.text(&["rev-parse", "FETCH_HEAD"]).ok()
    }

    /// Writes the files of `thread` on [`SESSIONS_BRANCH`] into `into`.
    fn restore(&self, thread: &str, into: &Path) -> Result<(), String> {
        fs::create_dir_all(into).map_err(|why| why.to_string())?;
        let Some(commit) = self.fetch(SESSIONS_BRANCH) else {
            return Ok(());
        };
        let names = self.text(&["ls-tree", "-r", "--name-only", &commit, "--", thread])?;
        for name in names.lines() {
            let Some(relative) = name
                .strip_prefix(thread)
                .map(|it| it.trim_start_matches('/'))
            else {
                continue;
            };
            let bytes = self.run(&["show", &format!("{commit}:{name}")], &[])?;
            let path = into.join(relative);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).map_err(|why| why.to_string())?;
            }
            fs::write(&path, bytes).map_err(|why| why.to_string())?;
        }
        Ok(())
    }

    /// Puts the files of `from` as `thread` on [`SESSIONS_BRANCH`] and
    /// pushes it, with an index of its own in `index`.
    fn save(&self, thread: &str, from: &Path, index: &Path) -> Result<(), String> {
        let parent = self.fetch(SESSIONS_BRANCH);
        let env = [("GIT_INDEX_FILE", index)];
        match &parent {
            Some(parent) => self.run(&["read-tree", parent], &env)?,
            None => self.run(&["read-tree", "--empty"], &env)?,
        };
        self.run(
            &[
                "rm",
                "--cached",
                "-r",
                "-q",
                "--ignore-unmatch",
                "--",
                thread,
            ],
            &env,
        )?;
        for (path, relative) in files_under(from) {
            let path = path.to_string_lossy();
            let blob =
                String::from_utf8_lossy(&self.run(&["hash-object", "-w", "--", &path], &[])?)
                    .trim()
                    .to_owned();
            let entry = format!("100644,{blob},{thread}/{relative}");
            self.run(&["update-index", "--add", "--cacheinfo", &entry], &env)?;
        }
        let tree = String::from_utf8_lossy(&self.run(&["write-tree"], &env)?)
            .trim()
            .to_owned();
        let message = format!("rig: the session of {thread}");
        let mut args = vec!["commit-tree", tree.as_str(), "-m", message.as_str()];
        if let Some(parent) = &parent {
            args.extend(["-p", parent.as_str()]);
        }
        let commit = self.text(&args)?;
        let to = format!("{commit}:refs/heads/{SESSIONS_BRANCH}");
        self.run(&["push", "-q", "origin", &to], &[]).map(drop)
    }
}

/// Every file under `dir`, with its path from `dir`.
fn files_under(dir: &Path) -> Vec<(PathBuf, String)> {
    let mut found = Vec::new();
    let mut todo = vec![dir.to_owned()];
    while let Some(at) = todo.pop() {
        for entry in fs::read_dir(&at).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                todo.push(path);
            } else if let Ok(relative) = path.strip_prefix(dir) {
                found.push((path.clone(), relative.to_string_lossy().into_owned()));
            }
        }
    }
    found
}

/// What every blocking step needs.
#[derive(Clone)]
struct Context {
    api: Api,
    git: Git,
    mention: Mention,
    config: RepoConfig,
    header: String,
}

/// What the run's systems read to reach GitHub and git.
#[derive(SystemParam)]
struct Github<'w> {
    run: Option<Res<'w, Run>>,
    env: Option<Res<'w, GithubEnv>>,
    runtime: Option<Res<'w, TokioRuntime>>,
    processes: Option<Res<'w, Processes>>,
    wake: Option<Res<'w, Wake>>,
}

impl Github<'_> {
    /// The run's context, if it has a mention.
    fn context(&self) -> Option<Context> {
        let run = self.run.as_ref()?;
        let mention = run.mention.clone()?;
        let env = self.env.as_deref()?.clone();
        let link = env.run_id.as_ref().map_or_else(String::new, |id| {
            format!(
                " ([CI run]({}/{}/actions/runs/{id}))",
                env.server_url, mention.repo
            )
        });
        Some(Context {
            api: Api {
                client: reqwest::Client::new(),
                runtime: self.runtime.as_deref()?.clone(),
                env,
                repo: mention.repo.clone(),
            },
            git: Git::new(self.processes.as_ref()?.launcher().in_dir(&run.root)),
            header: format!("Working on it…{link}"),
            config: run.config.clone(),
            mention,
        })
    }

    /// Runs `work` off the main thread on an entity with `marker`, waking
    /// the loop when done.
    fn spawn<T: Send + Sync + 'static>(
        &self,
        commands: &mut Commands,
        marker: impl Bundle,
        work: impl FnOnce() -> T + Send + 'static,
    ) {
        let Some(wake) = self.wake.as_deref() else {
            return;
        };
        let pool = bevy_tasks::IoTaskPool::get_or_init(bevy_tasks::TaskPool::default);
        commands.spawn((marker, Running::spawn(pool, wake, run_blocking(work))));
    }
}

/// Starts setting up off the main thread: the permission check, the
/// reply, the context and the branch.
fn set_up(github: Github, mut commands: Commands) {
    let context = github.context();
    github.spawn(&mut commands, (Job, KeepRunning), move || match context {
        Some(context) => prepare(&context),
        None => Setup::Ignore("no comment in the event".to_owned()),
    });
}

/// Runs `work` on a thread of its own and awaits it.
async fn run_blocking<T: Send + 'static>(work: impl FnOnce() -> T + Send + 'static) -> Option<T> {
    blocking(move || Ok(work())).await.ok()
}

/// Checks the mention, replies, and checks out the branch: what the agent
/// is asked.
fn prepare(context: &Context) -> Setup {
    let Context {
        api,
        mention,
        config,
        header,
        ..
    } = context;
    if !mention.comment.contains(config.trigger()) {
        return Setup::Ignore(format!("the comment does not mention {}", config.trigger()));
    }
    let permission = api
        .get(&format!("/collaborators/{}/permission", mention.author))
        .ok()
        .and_then(|answer| answer.get("permission")?.as_str().map(str::to_owned));
    if !matches!(permission.as_deref(), Some("admin" | "maintain" | "write")) {
        return Setup::Ignore(format!("@{} may not write here", mention.author));
    }
    let comment = match api.comment(mention.number, header) {
        Ok(id) => id,
        Err(why) => {
            error!("rig github: could not reply: {why}");
            return Setup::Stop(true);
        }
    };
    match ask(context) {
        Ok(Some((work, prompt))) => Setup::Go(Work { comment, ..work }, prompt),
        Ok(None) => {
            let why = "This pull request comes from a fork, and I can only push to this \
                       repository's branches, so I did not work on it.";
            api.edit(comment, why).ok();
            Setup::Stop(false)
        }
        Err(why) => {
            api.edit(comment, &format!("{header}\n\n**Failed:** {why}"))
                .ok();
            Setup::Stop(true)
        }
    }
}

/// What the agent is asked and the branch it works on, checked out; `None`
/// for a pull request from a fork.
fn ask(context: &Context) -> Result<Option<(Work, String)>, String> {
    let Context {
        api, git, mention, ..
    } = context;
    let n = mention.number;
    let kind = if mention.pull_request {
        "pull request"
    } else {
        "issue"
    };
    let mut prompt = format!(
        "You were mentioned on GitHub, in {kind} #{n} of {}.\n\n# {}\n\n{}\n\n# The thread\n",
        mention.repo, mention.title, mention.body,
    );
    let thread = api.get(&format!("/issues/{n}/comments?per_page=100"))?;
    write_comments(&mut prompt, &thread, |_| true);
    if let Some(root) = mention.review_thread {
        let review = api.get(&format!("/pulls/{n}/comments?per_page=100"))?;
        prompt.push_str("\n# The review thread\n");
        write_comments(&mut prompt, &review, |comment| {
            let id = |at: &str| comment.get(at).and_then(Value::as_u64);
            id("id") == Some(root) || id("in_reply_to_id") == Some(root)
        });
    }
    let _ = write!(
        prompt,
        "\n# The request, by @{}\n\n{}\n",
        mention.author, mention.comment
    );
    let (branch, base) = if mention.pull_request {
        let pull = api.get(&format!("/pulls/{n}"))?;
        let at = |at: &str| {
            pull.pointer(at)
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned()
        };
        if at("/head/repo/full_name") != mention.repo {
            return Ok(None);
        }
        let diff = api.send(
            "GET",
            &format!("/pulls/{n}"),
            None,
            "application/vnd.github.diff",
        )?;
        if let Some((path, line)) = &mention.place {
            let line = line.map_or_else(String::new, |line| format!(" line {line}"));
            let _ = write!(prompt, "\nThe review comment is on `{path}`{line}.\n");
        }
        let _ = write!(
            prompt,
            "\n# The pull request's diff\n\n```diff\n{diff}\n```\n"
        );
        (at("/head/ref"), None)
    } else {
        let base = git.text(&["rev-parse", "--abbrev-ref", "HEAD"]).ok();
        (format!("rig/issue-{n}"), base)
    };
    match git.fetch(&branch) {
        Some(commit) => git.run(&["checkout", "-q", "-B", &branch, &commit], &[])?,
        None => git.run(&["checkout", "-q", "-B", &branch], &[])?,
    };
    let _ = write!(
        prompt,
        "\nYou work on the branch `{branch}`. What you change is committed and pushed for you \
         when you finish; do not commit or push. End with a short answer for the thread."
    );
    Ok(Some((
        Work {
            comment: 0,
            branch,
            base,
        },
        prompt,
    )))
}

/// Writes each of the `comments` that is `wanted` into `prompt`, by whom.
fn write_comments(prompt: &mut String, comments: &Value, wanted: impl Fn(&Value) -> bool) {
    for comment in comments
        .as_array()
        .into_iter()
        .flatten()
        .filter(|it| wanted(it))
    {
        let text = |at: &str| {
            comment
                .pointer(at)
                .and_then(Value::as_str)
                .unwrap_or_default()
        };
        let _ = writeln!(prompt, "\n**@{}**: {}", text("/user/login"), text("/body"));
    }
}

/// Sends the prompt once set up, or ends the run.
fn set_up_done(
    done: On<Insert<Done<Option<Setup>>>>,
    mut jobs: Query<&mut Done<Option<Setup>>, With<Job>>,
    agents: PrimaryQuery,
    mut commands: Commands,
) {
    let Ok(mut setup) = jobs.get_mut(done.entity) else {
        return;
    };
    let failed = match setup.0.take() {
        Some(Setup::Go(work, prompt)) => {
            commands.insert_resource(work);
            if let Some(agent) = primary(&agents) {
                commands.trigger(Deliver::new(agent, prompt, DeliveryMode::Steer));
                return;
            }
            true
        }
        Some(Setup::Ignore(why)) => {
            info!("rig github: nothing to do: {why}");
            false
        }
        Some(Setup::Stop(failed)) => failed,
        None => true,
    };
    if failed {
        commands.trigger(Exit(ExitReason::Failed(1)));
    }
    commands.entity(done.entity).despawn();
}

/// A tool call's main argument: its command or path.
fn summary(arguments: &serde_json::Map<String, Value>) -> &str {
    ["command", "path", "pattern"]
        .iter()
        .find_map(|key| arguments.get(*key)?.as_str())
        .unwrap_or_default()
}

/// Each tool call that starts is the latest step.
fn started(
    start: On<Add<Started>>,
    calls: Query<&ToolRequest>,
    progress: Option<ResMut<Progress>>,
) {
    if let (Ok(request), Some(mut progress)) = (calls.get(start.entity), progress) {
        let call = &request.call.function;
        let step = format!("`{} {}`", call.name.as_str(), summary(&call.arguments));
        progress.pending = Some(step);
    }
}

/// Wants each call that may change anything, such as a `shell` command,
/// approved unless `.github/rig.toml` allows it. Nobody can answer in CI,
/// so the harness refuses it, as it does any call another plugin wants
/// approved.
fn needs_approval(
    calls: Query<(Entity, &ToolRequest, &Footprint), (Undecided, ToolCalls, Without<Approved>)>,
    run: Res<Run>,
    mut commands: Commands,
) {
    for (call, request, footprint) in &calls {
        let function = &request.call.function;
        if matches!(footprint.0, Touches::Writes(Scope::Anything))
            && !run
                .config
                .allows(function.name.as_str(), &function.arguments)
        {
            let why = "It may change anything, and the repository's .github/rig.toml does not \
                       allow it ([github] allow).";
            commands.entity(call).queue(NeedsApproval::add(FRONT, why));
        }
    }
}

/// Each call refused for want of approval is listed in the final reply.
fn denied(
    answered: On<Add<Outcome>>,
    calls: Query<(&ToolRequest, &Outcome), With<NeedsApproval>>,
    progress: Option<ResMut<Progress>>,
) {
    let (Ok((request, Outcome(Err(report)))), Some(mut progress)) =
        (calls.get(answered.entity), progress)
    else {
        return;
    };
    if report.refusal {
        let call = &request.call.function;
        let step = format!("`{} {}`", call.name.as_str(), summary(&call.arguments));
        progress.denied.push(step);
    }
}

/// Keeps the error notices about agents for the reply, and each turn's end.
fn turn_ended(ended: On<HeadlessTurnEnded>, mut progress: ResMut<Progress>) {
    progress.ended = Some((ended.answer.clone(), ended.failed));
}

/// Shows the latest step in the reply, one edit at a time.
fn show_progress(
    (github, work): (Github, Option<Res<Work>>),
    mut progress: ResMut<Progress>,
    mut notices: MessageReader<Notice>,
    edits: Query<(), Or<(With<Edit>, With<Final>)>>,
    mut commands: Commands,
) {
    for notice in notices
        .read()
        .filter(|notice| notice.level == NoticeLevel::Error)
    {
        progress.errors.push(notice.text.clone());
    }
    let (Some(work), Some(context)) = (work, github.context()) else {
        return;
    };
    if progress.finishing || !edits.is_empty() {
        return;
    }
    let Some(step) = progress.pending.take() else {
        return;
    };
    let (id, body) = (
        work.comment,
        format!("{}\n\nLatest step: {step}", context.header),
    );
    github.spawn(&mut commands, Edit, move || {
        context.api.edit(id, &body).err()
    });
}

/// A progress edit is done.
fn edited(
    done: On<Insert<Done<Option<Option<String>>>>>,
    edits: Query<&Done<Option<Option<String>>>, With<Edit>>,
    mut commands: Commands,
) {
    if let Ok(Done(answer)) = edits.get(done.entity) {
        if let Some(Some(why)) = answer {
            warn!("rig github: could not show progress: {why}");
        }
        commands.entity(done.entity).despawn();
    }
}

/// On the final edit's work.
#[derive(Component)]
struct Final;

/// Once the turn ended and no edit is in flight: commits, pushes, keeps
/// the session and edits the reply into the answer, off the main thread.
fn finish_reply(
    (github, work, paths): (Github, Option<Res<Work>>, Option<Res<SessionPaths>>),
    mut progress: ResMut<Progress>,
    edits: Query<(), With<Edit>>,
    mut commands: Commands,
) {
    let (Some(work), Some(context)) = (work, github.context()) else {
        return;
    };
    if progress.finishing || !edits.is_empty() {
        return;
    }
    let Some((answer, failed)) = progress.ended.clone() else {
        return;
    };
    progress.finishing = true;
    let ending = Ending {
        work: work.clone(),
        answer,
        failed,
        denied: progress.denied.clone(),
        errors: progress.errors.clone(),
        session: paths.map(|paths| paths.path().to_owned()),
    };
    github.spawn(&mut commands, Final, move || ending.finish(&context));
}

/// How the agent's work ended, for the final reply.
struct Ending {
    work: Work,
    answer: Option<String>,
    failed: bool,
    denied: Vec<String>,
    errors: Vec<String>,
    session: Option<PathBuf>,
}

impl Ending {
    /// Commits and pushes, keeps the session, and edits the reply; `Err`
    /// says why the run fails.
    fn finish(self, context: &Context) -> Result<(), String> {
        let Context {
            api,
            git,
            mention,
            config,
            ..
        } = context;
        let mut body = self.answer.clone().unwrap_or_default();
        let mut problems = self.errors.clone();
        if self.failed && problems.is_empty() {
            problems.push("the agent stopped without an answer".to_owned());
        }
        let branch = &self.work.branch;
        match commit_and_push(git, mention, branch) {
            Ok(false) => {}
            Ok(true) if mention.pull_request => {
                let _ = write!(body, "\n\nPushed to `{branch}`.");
            }
            Ok(true) => {
                let base = self.work.base.as_deref().unwrap_or("main");
                let link = if config.open_pr {
                    open_draft(api, mention, branch, base)
                } else {
                    Ok(format!(
                        "[create a pull request]({}/{}/compare/{base}...{branch}?expand=1)",
                        api.env.server_url, mention.repo
                    ))
                };
                match link {
                    Ok(link) => {
                        let _ = write!(body, "\n\nPushed to `{branch}`: {link}.");
                    }
                    Err(why) => problems.push(why),
                }
            }
            Err(why) => problems.push(why),
        }
        if let Some(dir) = &self.session {
            let saved = git.save(
                &mention.thread(),
                &dir.join("github-thread"),
                &dir.join("github-index"),
            );
            if let Err(why) = saved {
                problems.push(format!("the session was not kept: {why}"));
            }
        }
        if !self.denied.is_empty() {
            body.push_str("\n\nDenied, as they need approval:");
            for denied in &self.denied {
                let _ = write!(body, "\n- {denied}");
            }
        }
        for problem in &problems {
            let _ = write!(body, "\n\n**Error:** {problem}");
        }
        api.edit(self.work.comment, body.trim_start())?;
        problems.first().map_or(Ok(()), |why| Err(why.clone()))
    }
}

/// Commits what changed and pushes it to `branch`; whether anything was
/// pushed.
fn commit_and_push(git: &Git, mention: &Mention, branch: &str) -> Result<bool, String> {
    git.run(&["add", "-A"], &[])?;
    if !git.text(&["status", "--porcelain"])?.is_empty() {
        let first = mention.comment.lines().next().unwrap_or_default();
        let message = format!(
            "rig: {first}\n\nAsked by @{} in #{}.",
            mention.author, mention.number
        );
        git.run(&["commit", "-q", "-m", &message], &[])?;
    }
    let local = git.text(&["rev-parse", "HEAD"])?;
    if git.fetch(branch).as_deref() == Some(local.as_str()) {
        return Ok(false);
    }
    let to = format!("HEAD:refs/heads/{branch}");
    git.run(&["push", "-q", "origin", &to], &[])?;
    Ok(true)
}

/// Opens a draft pull request of `branch` and returns its link.
fn open_draft(api: &Api, mention: &Mention, branch: &str, base: &str) -> Result<String, String> {
    let body = json!({
        "title": mention.title,
        "head": branch,
        "base": base,
        "draft": true,
        "body": format!("For #{}, asked by @{}.", mention.number, mention.author),
    });
    let text = api.send("POST", "/pulls", Some(body), "application/vnd.github+json")?;
    let pull: Value = serde_json::from_str(&text).map_err(|why| why.to_string())?;
    let url = pull
        .get("html_url")
        .and_then(Value::as_str)
        .unwrap_or_default();
    Ok(format!("[draft pull request]({url})"))
}

/// The final edit is done: the run ends, failed if anything did.
fn finished(
    done: On<Insert<Done<Option<Result<(), String>>>>>,
    finals: Query<&Done<Option<Result<(), String>>>, With<Final>>,
    jobs: Query<Entity, With<Job>>,
    mut commands: Commands,
) {
    let Ok(Done(result)) = finals.get(done.entity) else {
        return;
    };
    if let Some(Err(why)) = result {
        error!("rig github: {why}");
        commands.trigger(Exit(ExitReason::Failed(1)));
    } else if result.is_none() {
        commands.trigger(Exit(ExitReason::Failed(1)));
    }
    for job in &jobs {
        commands.entity(job).despawn();
    }
    commands.entity(done.entity).despawn();
}
