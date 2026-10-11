//! Plugin corpus: 42. Share a session (spec `42-share.md`).
//!
//! `/share` renders the agent's conversation as an HTML page with every
//! secret found redacted, and asks with a picker whether to publish it,
//! listing what goes and the secrets taken out. Confirmed, it uploads in
//! the background to the configured target: a secret GitHub gist (with a
//! token from the config, `GH_TOKEN`/`GITHUB_TOKEN`, or `gh auth token`),
//! or an `http` paste server that answers a POST with `{"url": …}`. The
//! link is shown, copied to the clipboard when a clipboard tool is there,
//! and saved with the session, so `/share` again updates the same page and
//! `/unshare` deletes it. A run nobody sits at shares only with
//! `/share --yes`.

use std::io::Write as _;
use std::process::{Command, Stdio};

use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use bevy_tasks::{IoTaskPool, TaskPool};
use rig_core::http_client::{self, HttpClientExt, Method, Request};
use rig_harness::harness_protocol::Home;
use rig_harness::prelude::*;
use rig_tui::{PickItem, PickRequest};
use serde::Deserialize;

mod render;

#[cfg(test)]
mod tests;

/// The file in `RIG_HOME` the configuration is read from, when no
/// [`ShareConfig`] was inserted.
const CONFIG_FILE: &str = "share.json";

/// The name of the page in a gist.
const GIST_FILE: &str = "session.html";

/// Where GitHub's gist API is.
const GIST_API: &str = "https://api.github.com/gists";

/// Where shared sessions go, and how.
#[derive(Resource, Clone, Debug, Deserialize)]
pub struct ShareConfig {
    /// The target.
    #[serde(default)]
    pub target: ShareTarget,
    /// Whether the link is copied to the clipboard.
    #[serde(default = "copy_by_default")]
    pub clipboard: bool,
}

fn copy_by_default() -> bool {
    true
}

impl Default for ShareConfig {
    fn default() -> Self {
        Self {
            target: ShareTarget::default(),
            clipboard: true,
        }
    }
}

/// Where a session is published.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum ShareTarget {
    /// A secret GitHub gist.
    #[default]
    Gist,
    /// A paste server: `POST url` with the page creates (answering
    /// `{"url": …, "id"?: …}`), `PUT url/id` updates, `DELETE url/id`
    /// deletes.
    Http {
        /// The server's URL.
        url: String,
    },
}

impl ShareTarget {
    /// What names the target in a [`Shared`] and to the user.
    fn label(&self) -> String {
        match self {
            Self::Gist => "gist".to_owned(),
            Self::Http { url } => url.clone(),
        }
    }
}

/// On an agent whose session was shared: where, kept with the session.
#[derive(Component, Reflect, Clone, Debug, Default)]
#[reflect(Component, Saved, Default, Clone, Debug)]
pub struct Shared {
    /// The target's label (`gist`, or the server's URL).
    pub target: String,
    /// The page's id at the target.
    pub id: String,
    /// The link.
    pub url: String,
}

/// On an agent: the page `/share` asked the user to confirm.
#[derive(Component)]
struct PendingShare {
    html: String,
}

/// An upload or deletion running for `agent`, on an entity of its own.
#[derive(Component)]
struct ShareJob {
    agent: Entity,
}

/// How a job ended: the page's place, none once deleted, or the target's
/// error.
struct JobResult(Result<Option<Link>, String>);

/// A published page.
struct Link {
    id: String,
    url: String,
    /// Whether the link was copied to the clipboard.
    copied: bool,
}

/// What a job does.
enum Op {
    Publish {
        html: String,
        update: Option<String>,
    },
    Delete {
        id: String,
    },
}

/// Adds `/share` and `/unshare`.
#[derive(Default)]
pub struct SharePlugin;

impl Plugin for SharePlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<ShareConfig>() {
            let path = Home::from_env().root().join(CONFIG_FILE);
            let config = std::fs::read_to_string(path)
                .ok()
                .and_then(|text| serde_json::from_str(&text).ok())
                .unwrap_or_default();
            app.insert_resource::<ShareConfig>(config);
        }
        app.add_message::<PickRequest>()
            .add_command(
                "share",
                "Publish this session as a link (a secret gist, or the configured server); \
                 again to update it. --yes skips the confirmation",
                share,
            )
            .add_command("unshare", "Delete the page /share published", unshare)
            .add_observer(on_job_done);
    }
}

/// `/share [--yes]`, and the picker's `/share confirm` and `/share cancel`.
fn share(
    In(args): In<CommandArgs>,
    agents: Query<(
        &Conversation,
        Option<&Name>,
        Option<&Shared>,
        Option<&PendingShare>,
    )>,
    jobs: Query<&ShareJob>,
    config: Res<ShareConfig>,
    invoked: Option<Res<Invoked>>,
    wake: Res<Wake>,
    mut commands: Commands,
    mut picks: MessageWriter<PickRequest>,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let Ok((conversation, name, shared, pending)) = agents.get(agent) else {
        return;
    };
    if jobs.iter().any(|job| job.agent == agent) {
        notices.write(Notice::error(agent, "A share of this session is running."));
        return;
    }
    let update = shared
        .filter(|shared| shared.target == config.target.label())
        .map(|shared| shared.id.clone());
    match args.args.as_str() {
        "cancel" => {
            commands.entity(agent).remove::<PendingShare>();
            notices.write(Notice::info(agent, "Not shared."));
        }
        "confirm" => match pending {
            Some(pending) => {
                let html = pending.html.clone();
                commands.entity(agent).remove::<PendingShare>();
                start(
                    &mut commands,
                    &wake,
                    &config,
                    agent,
                    Op::Publish { html, update },
                );
                notices.write(Notice::info(agent, "Sharing…"));
            }
            None => {
                notices.write(Notice::error(agent, "Nothing to share: run /share first."));
            }
        },
        "" | "--yes" => {
            let title = name.map_or("rig session", Name::as_str);
            let rendered = render::render(title, conversation.messages());
            if args.args == "--yes" {
                let op = Op::Publish {
                    html: rendered.html,
                    update,
                };
                start(&mut commands, &wake, &config, agent, op);
                notices.write(Notice::info(agent, "Sharing…"));
                return;
            }
            if invoked.is_some_and(|invoked| !invoked.interactive()) {
                notices.write(Notice::error(
                    agent,
                    "Not shared without a confirmation: use /share --yes.",
                ));
                return;
            }
            let action = if update.is_some() { "Update" } else { "Share" };
            let mut label = format!(
                "{action} {} messages on {}",
                rendered.messages,
                config.target.label()
            );
            if !rendered.secrets.is_empty() {
                label.push_str(&format!(
                    ", {} secrets redacted: {}",
                    rendered.secrets.len(),
                    rendered.secrets.join(", ")
                ));
            }
            commands.entity(agent).insert(PendingShare {
                html: rendered.html,
            });
            picks.write(PickRequest {
                agent,
                title: "Share this session?".to_owned(),
                items: vec![
                    PickItem {
                        label,
                        command: "share confirm".to_owned(),
                    },
                    PickItem {
                        label: "Cancel".to_owned(),
                        command: "share cancel".to_owned(),
                    },
                ],
                selected: 1,
            });
        }
        other => {
            notices.write(Notice::error(
                agent,
                format!("/share takes no `{other}`: /share, or /share --yes."),
            ));
        }
    }
}

/// `/unshare`: deletes the page in the background.
fn unshare(
    In(args): In<CommandArgs>,
    agents: Query<&Shared>,
    jobs: Query<&ShareJob>,
    config: Res<ShareConfig>,
    wake: Res<Wake>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let Ok(shared) = agents.get(agent) else {
        notices.write(Notice::error(agent, "This session is not shared."));
        return;
    };
    if shared.target != config.target.label() {
        notices.write(Notice::error(
            agent,
            format!(
                "Shared on {}, not on the configured {}.",
                shared.target,
                config.target.label()
            ),
        ));
        return;
    }
    if jobs.iter().any(|job| job.agent == agent) {
        notices.write(Notice::error(agent, "A share of this session is running."));
        return;
    }
    let op = Op::Delete {
        id: shared.id.clone(),
    };
    start(&mut commands, &wake, &config, agent, op);
    notices.write(Notice::info(agent, "Unsharing…"));
}

/// Spawns the job doing `op` for `agent` on the IO pool.
fn start(commands: &mut Commands, wake: &Wake, config: &ShareConfig, agent: Entity, op: Op) {
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    let clipboard = config.clipboard;
    let work = run(config.clone(), op);
    let work = async move {
        let mut result = work.await;
        if clipboard && let Ok(Some(link)) = &mut result {
            let url = link.url.clone();
            link.copied = blocking(move || Ok(copy(&url))).await.unwrap_or(false);
        }
        JobResult(result)
    };
    commands.spawn((
        Name::new("share"),
        ShareJob { agent },
        Running::spawn(pool, wake, work),
    ));
}

/// Reports a finished job, and keeps or forgets the link.
fn on_job_done(
    done: On<Add<Done<JobResult>>>,
    jobs: Query<(&ShareJob, &Done<JobResult>)>,
    config: Res<ShareConfig>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((job, Done(JobResult(result)))) = jobs.get(done.entity) else {
        return;
    };
    let agent = job.agent;
    match result {
        Ok(Some(link)) => {
            let copied = if link.copied { " (copied)" } else { "" };
            notices.write(Notice::info(agent, format!("Shared: {}{copied}", link.url)));
            commands.entity(agent).insert(Shared {
                target: config.target.label(),
                id: link.id.clone(),
                url: link.url.clone(),
            });
        }
        Ok(None) => {
            notices.write(Notice::info(agent, "Unshared."));
            commands.entity(agent).remove::<Shared>();
        }
        Err(why) => {
            notices.write(Notice::error(agent, format!("Share failed: {why}")));
        }
    }
    commands.entity(done.entity).despawn();
}

/// Does `op` at the configured target.
async fn run(config: ShareConfig, op: Op) -> Result<Option<Link>, String> {
    match config.target {
        ShareTarget::Http { url } => {
            let base = url.trim_end_matches('/').to_owned();
            let html_type = vec![("content-type", "text/html; charset=utf-8".to_owned())];
            let (method, uri, body) = match op {
                Op::Publish { html, update: None } => (Method::POST, url, html),
                Op::Publish {
                    html,
                    update: Some(id),
                } => (Method::PUT, format!("{base}/{id}"), html),
                Op::Delete { id } => {
                    send(
                        Method::DELETE,
                        format!("{base}/{id}"),
                        Vec::new(),
                        String::new(),
                    )
                    .await?;
                    return Ok(None);
                }
            };
            let answer = send(method, uri, html_type, body).await?;
            parse_link(&answer, "url", "id").map(Some)
        }
        ShareTarget::Gist => {
            let token = gist_token().await?;
            let headers = vec![
                ("authorization", format!("Bearer {token}")),
                ("accept", "application/vnd.github+json".to_owned()),
                ("user-agent", "rig-share".to_owned()),
            ];
            let (method, uri, html) = match op {
                Op::Publish { html, update: None } => (Method::POST, GIST_API.to_owned(), html),
                Op::Publish {
                    html,
                    update: Some(id),
                } => (Method::PATCH, format!("{GIST_API}/{id}"), html),
                Op::Delete { id } => {
                    send(
                        Method::DELETE,
                        format!("{GIST_API}/{id}"),
                        headers,
                        String::new(),
                    )
                    .await?;
                    return Ok(None);
                }
            };
            let body = serde_json::json!({
                "description": "A rig session",
                "public": false,
                "files": { GIST_FILE: { "content": html } },
            });
            let answer = send(method, uri, headers, body.to_string()).await?;
            parse_link(&answer, "html_url", "id").map(Some)
        }
    }
}

/// The link in a target's JSON answer: its `url_key`, and its `id_key`
/// or else the link's last path segment.
fn parse_link(answer: &str, url_key: &str, id_key: &str) -> Result<Link, String> {
    let value: serde_json::Value =
        serde_json::from_str(answer).map_err(|error| format!("not JSON ({error}): {answer}"))?;
    let url = value
        .get(url_key)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| format!("no `{url_key}` in the answer: {answer}"))?
        .to_owned();
    let id = value
        .get(id_key)
        .and_then(|id| {
            id.as_str()
                .map(str::to_owned)
                .or_else(|| id.as_u64().map(|n| n.to_string()))
        })
        .or_else(|| {
            url.trim_end_matches('/')
                .rsplit('/')
                .next()
                .map(str::to_owned)
        })
        .unwrap_or_default();
    Ok(Link {
        id,
        url,
        copied: false,
    })
}

/// Sends a request on the shared client, answering its body; a failed
/// status answers the target's error.
async fn send(
    method: Method,
    uri: String,
    headers: Vec<(&'static str, String)>,
    body: String,
) -> Result<String, String> {
    let mut request = Request::builder().method(method).uri(uri);
    for (name, value) in headers {
        request = request.header(name, value);
    }
    let request = request.body(body).map_err(|error| error.to_string())?;
    let response = rig_reqwest::shared()
        .send::<String, Vec<u8>>(request)
        .await
        .map_err(|error| match error.non_success_body() {
            Some(body) => format!("{error}: {body}"),
            None => error.to_string(),
        })?;
    http_client::text(response)
        .await
        .map_err(|error| error.to_string())
}

/// A GitHub token: `GH_TOKEN`, `GITHUB_TOKEN`, or the `gh` CLI's.
async fn gist_token() -> Result<String, String> {
    for name in ["GH_TOKEN", "GITHUB_TOKEN"] {
        if let Ok(token) = std::env::var(name)
            && !token.is_empty()
        {
            return Ok(token);
        }
    }
    blocking(|| {
        let output = Command::new("gh")
            .args(["auth", "token"])
            .output()
            .map_err(ToolExecutionError::from_error)?;
        let token = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        if output.status.success() && !token.is_empty() {
            Ok(token)
        } else {
            Err(ToolExecutionError::other(
                String::from_utf8_lossy(&output.stderr).trim().to_owned(),
            ))
        }
    })
    .await
    .map_err(|error| format!("no GitHub token (set GH_TOKEN or run `gh auth login`): {error}"))
}

/// Copies `text` with the first clipboard tool found; whether it did.
fn copy(text: &str) -> bool {
    let tools: [(&str, &[&str]); 3] = [
        ("wl-copy", &[]),
        ("xclip", &["-selection", "clipboard"]),
        ("pbcopy", &[]),
    ];
    tools.iter().any(|(tool, args)| {
        let Ok(mut child) = Command::new(tool)
            .args(*args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        else {
            return false;
        };
        let written = child
            .stdin
            .take()
            .is_some_and(|mut stdin| stdin.write_all(text.as_bytes()).is_ok());
        child.wait().is_ok_and(|status| status.success()) && written
    })
}
