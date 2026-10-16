//! The tracker: which one the `origin` remote names, and loading a
//! reference from its API as the block the model reads.

use std::ffi::{OsStr, OsString};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use reqwest::header::{ACCEPT, CONTENT_TYPE, USER_AGENT};
use reqwest::{Client, RequestBuilder};
use rig_harness::rig_tools::process::{Launcher, Spawn};
use rig_harness::rig_tools::workspace::Workspace;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::issue_references::refs::{RefKind, Reference};

/// How long one reference may take to load.
pub const TIMEOUT: Duration = Duration::from_secs(5);
/// Characters of an issue's body kept.
pub const BODY_CHARS: usize = 8_000;
/// Comments kept, the newest.
pub const COMMENTS: usize = 10;

/// Which API a tracker speaks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrackerKind {
    /// GitHub or GitHub Enterprise (REST v3 and GraphQL).
    Github,
    /// GitLab (REST v4).
    Gitlab,
}

/// The `[tracker]` table of `settings.toml`; each key left out comes from
/// the `origin` remote or the environment (`GITHUB_TOKEN`, `GH_TOKEN`,
/// `GITLAB_TOKEN`, `GITHUB_API_URL`, `CI_API_V4_URL`).
#[derive(bevy_ecs::resource::Resource, Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrackerSettings {
    /// `github` or `gitlab`.
    pub kind: Option<TrackerKind>,
    /// The API's base URL, such as `https://api.github.com`.
    pub api: Option<String>,
    /// The token sent with each request.
    pub token: Option<String>,
    /// The repository, `owner/repo` (GitLab: the project's full path).
    pub repo: Option<String>,
    /// The web host whose URLs count as this tracker's.
    pub host: Option<String>,
}

/// Where the project is, to ask its `origin` remote: the launcher, the
/// project's workspace (which may not be this machine) and its root.
#[derive(Clone)]
pub struct Place {
    /// Starts the `git` that names the remote.
    pub launcher: Launcher,
    /// Where `git` runs.
    pub workspace: Arc<dyn Workspace>,
    /// The project's root in it.
    pub root: PathBuf,
}

/// The tracker references are loaded from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tracker {
    /// Its API.
    pub kind: TrackerKind,
    /// Its API's base URL, without a trailing `/`.
    pub api: String,
    /// The current repository.
    pub repo: String,
    /// Its web host.
    pub host: String,
    /// The token, if any.
    pub token: Option<String>,
}

impl Tracker {
    /// The tracker `settings` name, completed from the `origin` remote of
    /// the project at `place` and the environment.
    pub fn resolve(settings: &TrackerSettings, place: &Place) -> Result<Self, String> {
        let origin = match (&settings.repo, &settings.host) {
            (Some(repo), Some(host)) => (host.clone(), repo.clone()),
            _ => origin(place)?,
        };
        let (host, repo) = (
            settings.host.clone().unwrap_or(origin.0),
            settings.repo.clone().unwrap_or(origin.1),
        );
        let kind = settings
            .kind
            .or_else(|| host.contains("gitlab").then_some(TrackerKind::Gitlab))
            .unwrap_or(TrackerKind::Github);
        let env = |names: &[&str]| names.iter().find_map(|name| std::env::var(name).ok());
        let api = settings.api.clone().or_else(|| match kind {
            TrackerKind::Github => env(&["GITHUB_API_URL"]),
            TrackerKind::Gitlab => env(&["CI_API_V4_URL"]),
        });
        let api = api.unwrap_or_else(|| match (kind, host.as_str()) {
            (TrackerKind::Github, "github.com") => "https://api.github.com".to_owned(),
            (TrackerKind::Github, _) => format!("https://{host}/api/v3"),
            (TrackerKind::Gitlab, _) => format!("https://{host}/api/v4"),
        });
        let token = settings.token.clone().or_else(|| match kind {
            TrackerKind::Github => env(&["GITHUB_TOKEN", "GH_TOKEN"]),
            TrackerKind::Gitlab => env(&["GITLAB_TOKEN"]),
        });
        let api = api.trim_end_matches('/').to_owned();
        Ok(Self {
            kind,
            api,
            repo,
            host,
            token,
        })
    }

    /// The tracker once resolved in `cell`, resolving it the first time.
    pub fn once(
        cell: &OnceLock<Result<Self, String>>,
        settings: &TrackerSettings,
        place: &Place,
    ) -> Result<Self, String> {
        cell.get_or_init(|| Self::resolve(settings, place)).clone()
    }
}

/// The host and repository of the `origin` remote of the project at
/// `place`, asked in its workspace.
fn origin(place: &Place) -> Result<(String, String), String> {
    let args = ["remote", "get-url", "origin"].map(OsString::from);
    let git = OsStr::new("git");
    let output = place
        .workspace
        .spawn(
            &place.launcher,
            Some(&place.root),
            git,
            &args,
            Spawn::default(),
        )
        .and_then(|process| process.output(Some(TIMEOUT)))
        .map_err(|failure| format!("no tracker: git: {failure}"))?;
    let url = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    remote(&url).ok_or_else(|| format!("no tracker: origin is {url:?}"))
}

/// The host and repository of a remote URL: `git@host:owner/repo.git`,
/// `https://host/owner/repo` or `ssh://git@host/group/project.git`.
pub fn remote(url: &str) -> Option<(String, String)> {
    let rest = url.split_once("://").map_or(url, |(_, rest)| rest);
    let rest = rest.rsplit_once('@').map_or(rest, |(_, rest)| rest);
    let (host, path) = rest.split_once([':', '/'])?;
    let host = host.split(':').next()?;
    let path = path.trim_start_matches(|c: char| c.is_ascii_digit() || c == '/');
    let repo = path.trim_end_matches('/').trim_end_matches(".git");
    (repo.contains('/') && !host.is_empty()).then(|| (host.to_owned(), repo.to_owned()))
}

/// What a reference loaded as.
struct Loaded {
    title: String,
    state: String,
    labels: Vec<String>,
    author: String,
    body: String,
    total_comments: usize,
    comments: Vec<(String, String)>,
    files: Option<Vec<String>>,
    review: Option<Vec<ReviewComment>>,
}

/// An unresolved review comment.
struct ReviewComment {
    path: String,
    line: Option<u64>,
    author: String,
    body: String,
}

/// The line the model reads for a reference that could not be loaded.
pub fn failure_note(reference: &Reference, why: &str) -> String {
    format!("could not load {reference}: {why}")
}

/// Loads `reference` from `tracker` as the block the model reads.
pub async fn load(
    client: &Client,
    tracker: &Tracker,
    reference: &Reference,
) -> Result<String, String> {
    if reference
        .host
        .as_ref()
        .is_some_and(|host| *host != tracker.host)
    {
        return Err(format!("not on {}", tracker.host));
    }
    let loaded = match tracker.kind {
        TrackerKind::Github => github(client, tracker, reference).await?,
        TrackerKind::Gitlab => gitlab(client, tracker, reference).await?,
    };
    Ok(block(reference, &loaded))
}

/// The block for `loaded`: title, state, labels, author, body, the newest
/// comments; a pull request's changed files and unresolved review comments.
fn block(reference: &Reference, loaded: &Loaded) -> String {
    let kind = match reference.kind {
        RefKind::Issue => "issue",
        RefKind::Pull => "pull request",
    };
    let mut out = format!(
        "<reference ref=\"{reference}\" kind=\"{kind}\">\n{}\n",
        loaded.title
    );
    let labels = match loaded.labels.is_empty() {
        true => "none".to_owned(),
        false => loaded.labels.join(", "),
    };
    out += &format!(
        "state: {} | labels: {labels} | author: {}\n\n",
        loaded.state, loaded.author
    );
    let body: String = loaded.body.chars().take(BODY_CHARS).collect();
    out += body.trim_end();
    if loaded.body.chars().nth(BODY_CHARS).is_some() {
        out += &format!("\n[body cut at {BODY_CHARS} characters]");
    }
    out += &format!(
        "\n\ncomments (last {} of {}):\n",
        loaded.comments.len(),
        loaded.total_comments
    );
    for (author, text) in &loaded.comments {
        out += &format!("- {author}: {}\n", text.trim());
    }
    if let Some(files) = &loaded.files {
        out += &format!("changed files ({}):\n", files.len());
        for file in files {
            out += &format!("- {file}\n");
        }
    }
    if let Some(review) = &loaded.review {
        out += &format!("unresolved review comments ({}):\n", review.len());
        for comment in review {
            let line = comment
                .line
                .map(|line| format!(":{line}"))
                .unwrap_or_default();
            out += &format!(
                "- {}{line} {}: {}\n",
                comment.path,
                comment.author,
                comment.body.trim()
            );
        }
    }
    out + "</reference>"
}

/// A request's answer as JSON, or why not, such as `404 Not Found`.
async fn send(request: RequestBuilder) -> Result<Value, String> {
    let request = request.header(USER_AGENT, "rig-issue-references");
    let response = request
        .send()
        .await
        .map_err(|failure| failure.without_url().to_string())?;
    let status = response.status();
    if !status.is_success() {
        return Err(status.to_string());
    }
    let text = response
        .text()
        .await
        .map_err(|failure| failure.without_url().to_string())?;
    serde_json::from_str(&text).map_err(|failure| format!("not JSON: {failure}"))
}

/// A string at `pointer` in `value`, empty when there is none.
fn text(value: &Value, pointer: &str) -> String {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

/// The items of an array answer.
fn items(value: Result<Value, String>) -> Result<Vec<Value>, String> {
    Ok(value?.as_array().cloned().unwrap_or_default())
}

/// The newest [`COMMENTS`] of `comments`, which are oldest first.
fn newest(comments: Vec<(String, String)>) -> Vec<(String, String)> {
    let skip = comments.len().saturating_sub(COMMENTS);
    comments.into_iter().skip(skip).collect()
}

async fn github(
    client: &Client,
    tracker: &Tracker,
    reference: &Reference,
) -> Result<Loaded, String> {
    let repo = reference.repo.as_deref().unwrap_or(&tracker.repo);
    let base = format!("{}/repos/{repo}", tracker.api);
    let get = |path: String| {
        let request = client
            .get(format!("{base}{path}"))
            .header(ACCEPT, "application/vnd.github+json");
        match &tracker.token {
            Some(token) => request.bearer_auth(token),
            None => request,
        }
    };
    let number = reference.number;
    let main = match reference.kind {
        RefKind::Issue => format!("/issues/{number}"),
        RefKind::Pull => format!("/pulls/{number}"),
    };
    let item = send(get(main)).await?;
    let total = item
        .get("comments")
        .and_then(Value::as_u64)
        .unwrap_or_default();
    // The newest comments are on the last pages, oldest first.
    let last_page = total.div_ceil(100).max(1);
    let page = |page: u64| {
        send(get(format!(
            "/issues/{number}/comments?per_page=100&page={page}"
        )))
    };
    let pull = reference.kind == RefKind::Pull;
    let (comments, files, review) = futures::join!(
        page(last_page),
        async {
            match pull {
                true => Some(items(
                    send(get(format!("/pulls/{number}/files?per_page=100"))).await,
                )),
                false => None,
            }
        },
        async {
            match pull {
                true => Some(github_review(client, tracker, repo, number).await),
                false => None,
            }
        },
    );
    let mut comments = items(comments)?;
    if comments.len() < COMMENTS && last_page > 1 {
        let mut earlier = items(page(last_page - 1).await)?;
        earlier.append(&mut comments);
        comments = earlier;
    }
    let comments = comments
        .iter()
        .map(|comment| (text(comment, "/user/login"), text(comment, "/body")))
        .collect();
    let files = files
        .transpose()?
        .map(|files| files.iter().map(|file| text(file, "/filename")).collect());
    Ok(Loaded {
        title: text(&item, "/title"),
        state: text(&item, "/state"),
        labels: item
            .get("labels")
            .and_then(Value::as_array)
            .map(|labels| labels.iter().map(|label| text(label, "/name")).collect())
            .unwrap_or_default(),
        author: text(&item, "/user/login"),
        body: text(&item, "/body"),
        total_comments: usize::try_from(total).unwrap_or(usize::MAX),
        comments: newest(comments),
        files,
        review: review.transpose()?,
    })
}

/// The unresolved review threads of pull request `number`, from the
/// GraphQL API: REST says nothing of a thread's resolution.
async fn github_review(
    client: &Client,
    tracker: &Tracker,
    repo: &str,
    number: u64,
) -> Result<Vec<ReviewComment>, String> {
    let graphql = match tracker.api.strip_suffix("/api/v3") {
        Some(host) => format!("{host}/api/graphql"),
        None => format!("{}/graphql", tracker.api),
    };
    let (owner, name) = repo.split_once('/').unwrap_or((repo, ""));
    let query = "query($owner: String!, $name: String!, $number: Int!) { repository(owner: $owner, name: $name) { \
                 pullRequest(number: $number) { reviewThreads(first: 100) { nodes { isResolved path line \
                 comments(first: 50) { nodes { author { login } body } } } } } } }";
    let body =
        json!({ "query": query, "variables": { "owner": owner, "name": name, "number": number } });
    let request = client
        .post(graphql)
        .header(CONTENT_TYPE, "application/json")
        .body(body.to_string());
    let request = match &tracker.token {
        Some(token) => request.bearer_auth(token),
        None => request,
    };
    let answer = send(request).await?;
    let threads = answer
        .pointer("/data/repository/pullRequest/reviewThreads/nodes")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let unresolved = threads
        .iter()
        .filter(|thread| thread.get("isResolved").and_then(Value::as_bool) != Some(true));
    let mut comments = Vec::new();
    for thread in unresolved {
        let nodes = thread.pointer("/comments/nodes").and_then(Value::as_array);
        for comment in nodes.into_iter().flatten() {
            comments.push(ReviewComment {
                path: text(thread, "/path"),
                line: thread.get("line").and_then(Value::as_u64),
                author: text(comment, "/author/login"),
                body: text(comment, "/body"),
            });
        }
    }
    Ok(comments)
}

async fn gitlab(
    client: &Client,
    tracker: &Tracker,
    reference: &Reference,
) -> Result<Loaded, String> {
    let repo = reference.repo.as_deref().unwrap_or(&tracker.repo);
    let base = format!("{}/projects/{}", tracker.api, repo.replace('/', "%2F"));
    let get = |path: String| {
        let request = client.get(format!("{base}{path}"));
        match &tracker.token {
            Some(token) => request.header("PRIVATE-TOKEN", token),
            None => request,
        }
    };
    let pull = reference.kind == RefKind::Pull;
    let path = match pull {
        true => format!("/merge_requests/{}", reference.number),
        false => format!("/issues/{}", reference.number),
    };
    let item = send(get(path.clone())).await?;
    let (notes, files, discussions) = futures::join!(
        send(get(format!(
            "{path}/notes?sort=desc&order_by=created_at&per_page=50"
        ))),
        async {
            match pull {
                true => Some(items(send(get(format!("{path}/diffs?per_page=100"))).await)),
                false => None,
            }
        },
        async {
            match pull {
                true => Some(items(
                    send(get(format!("{path}/discussions?per_page=100"))).await,
                )),
                false => None,
            }
        },
    );
    let mut comments: Vec<(String, String)> = items(notes)?
        .iter()
        .filter(|note| note.get("system").and_then(Value::as_bool) != Some(true))
        .take(COMMENTS)
        .map(|note| (text(note, "/author/username"), text(note, "/body")))
        .collect();
    comments.reverse();
    let files = files
        .transpose()?
        .map(|files| files.iter().map(|file| text(file, "/new_path")).collect());
    let review = discussions.transpose()?.map(|discussions| {
        let notes = discussions.iter().flat_map(|discussion| {
            discussion
                .get("notes")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default()
        });
        let open = |note: &Value| {
            note.get("resolvable").and_then(Value::as_bool) == Some(true)
                && note.get("resolved").and_then(Value::as_bool) != Some(true)
        };
        notes
            .filter(open)
            .map(|note| ReviewComment {
                path: text(&note, "/position/new_path"),
                line: note.pointer("/position/new_line").and_then(Value::as_u64),
                author: text(&note, "/author/username"),
                body: text(&note, "/body"),
            })
            .collect()
    });
    let total = item
        .get("user_notes_count")
        .and_then(Value::as_u64)
        .unwrap_or_default();
    Ok(Loaded {
        title: text(&item, "/title"),
        state: text(&item, "/state"),
        labels: item
            .get("labels")
            .and_then(Value::as_array)
            .map(|labels| {
                labels
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default(),
        author: text(&item, "/author/username"),
        body: text(&item, "/description"),
        total_comments: usize::try_from(total).unwrap_or(usize::MAX),
        comments,
        files,
        review,
    })
}

/// The tracker's open issues, number and title, the newest 100.
pub async fn open_issues(client: &Client, tracker: &Tracker) -> Result<Vec<(u64, String)>, String> {
    let (url, number) = match tracker.kind {
        TrackerKind::Github => (
            format!(
                "{}/repos/{}/issues?state=open&per_page=100",
                tracker.api, tracker.repo
            ),
            "/number",
        ),
        TrackerKind::Gitlab => (
            format!(
                "{}/projects/{}/issues?state=opened&per_page=100",
                tracker.api,
                tracker.repo.replace('/', "%2F")
            ),
            "/iid",
        ),
    };
    let request = client.get(url);
    let request = match (&tracker.token, tracker.kind) {
        (Some(token), TrackerKind::Github) => request.bearer_auth(token),
        (Some(token), TrackerKind::Gitlab) => request.header("PRIVATE-TOKEN", token),
        (None, _) => request,
    };
    let issues = items(send(request).await)?;
    Ok(issues
        .iter()
        // GitHub lists pull requests among the issues.
        .filter(|issue| issue.get("pull_request").is_none())
        .filter_map(|issue| Some((issue.pointer(number)?.as_u64()?, text(issue, "/title"))))
        .collect())
}
