//! Plugin corpus: 42. Share a session (spec `42-share.md`).
//!
//! `/share` renders the agent's conversation as an HTML page with every
//! secret found redacted, and spawns the upload as a `share` effect held
//! by a question listing what goes and the secrets taken out. Confirmed,
//! the effect starts and uploads in the background to the `[share]`
//! target: a secret GitHub gist (with a token from `GH_TOKEN`,
//! `GITHUB_TOKEN` or `gh auth token`), or an `http` paste server that
//! answers a POST with `{"url": …}`. The link is shown, copied to the
//! clipboard through the terminal view (`CopyToClipboard`, OSC 52; from a
//! worker, an `x/share/link` event its view half, [`ShareView`], copies),
//! and saved with the session, so `/share` again updates the same page and
//! `/unshare` deletes it. A run nobody answers questions in does not share.

use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use bevy_tasks::{IoTaskPool, TaskPool};
use rig_harness::channel::{ChannelAppExt, ChannelEvent};
use rig_harness::prelude::*;
use rig_harness::rig_core::effect::{Answered, EffectKind};
use rig_harness::rig_core::http_client::{self, HttpClientExt, Method, Request};
use rig_harness::rig_core::{ErrorKind, ErrorReport};
use rig_harness::view::Applied;
use rig_harness::worker::Publish;
use rig_tui::CopyToClipboard;
use serde::{Deserialize, Serialize};

mod render;

#[cfg(test)]
mod tests;

/// The effect kind of an upload or a deletion.
const KIND: &str = "share";

/// The name of the page in a gist.
const GIST_FILE: &str = "session.html";

/// Where GitHub's gist API is.
const GIST_API: &str = "https://api.github.com/gists";

/// The `[share]` settings: `target = "gist"`, or `target = "http"` with
/// `url = "https://paste.example/"`; `clipboard = false` keeps links off
/// the clipboard.
#[derive(Resource, Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ShareSettings {
    /// Where pages go.
    pub target: Target,
    /// The paste server, for `http`: `POST url` creates (answering
    /// `{"url": …, "id"?: …}`), `PUT url/id` updates, `DELETE url/id`
    /// deletes.
    pub url: String,
    /// Whether links are copied to the clipboard.
    pub clipboard: bool,
}

impl Default for ShareSettings {
    fn default() -> Self {
        Self {
            target: Target::Gist,
            url: String::new(),
            clipboard: true,
        }
    }
}

/// Where a session is published.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Target {
    /// A secret GitHub gist.
    #[default]
    Gist,
    /// The paste server at `url`.
    Http,
}

impl ShareSettings {
    /// What names the target in a [`Shared`] and to the user.
    fn label(&self) -> String {
        match self.target {
            Target::Gist => "gist".to_owned(),
            Target::Http => self.url.clone(),
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

/// On a `share` effect: what it does, and where.
#[derive(Component)]
struct Upload {
    settings: ShareSettings,
    op: Op,
}

/// What an upload does.
#[derive(Clone)]
enum Op {
    /// Publishes `html`, as a new page or in place of the page `update`.
    Publish {
        html: String,
        update: Option<String>,
    },
    /// Deletes the page `id`.
    Delete { id: String },
}

/// A published page: a `share` effect's answer.
#[derive(Serialize, Deserialize)]
struct Link {
    id: String,
    url: String,
}

/// Adds `/share`, `/unshare` and `--yes`.
#[derive(Default)]
pub struct SharePlugin;

impl Plugin for SharePlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<ShareSettings>("share")
            .add_channel_event::<SharedLink>()
            .init_resource::<Processes>()
            // Inert without the terminal view, which copies them.
            .add_message::<CopyToClipboard>()
            .register_type::<Shared>()
            .add_command(
                "share",
                "Publish this session as a link (a secret gist, or the [share] server); \
                 again to update it",
                share,
            )
            .add_command("unshare", "Delete the page /share published", unshare)
            .add_observer(upload)
            .add_observer(report);
    }
}

/// The agents' `share` effects that did not settle yet.
type InFlight<'w, 's> = Query<'w, 's, &'static ForAgent, (With<Upload>, Without<Settled>)>;

/// `/share`: asks to publish the session, or publishes it at once under
/// `--yes` when nobody can answer.
fn share(
    In(args): In<CommandArgs>,
    agents: Query<(&Conversation, Option<&Name>, Option<&Shared>)>,
    in_flight: InFlight,
    settings: Res<ShareSettings>,
    answerers: Answerers,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let Ok((conversation, name, shared)) = agents.get(agent) else {
        return;
    };
    if !args.args.is_empty() {
        notices.write(Notice::error(agent, "/share takes no arguments."));
        return;
    }
    if in_flight.iter().any(|of| of.0 == agent) {
        notices.write(Notice::error(
            agent,
            "A share of this session is under way.",
        ));
        return;
    }
    let ask = answerers.can_answer(Some(agent));
    if !ask {
        let why = "Not shared: nobody can confirm in this run.";
        notices.write(Notice::error(agent, why));
        return;
    }
    let target = settings.label();
    let update = shared
        .filter(|shared| shared.target == target)
        .map(|shared| shared.id.clone());
    let title = name.map_or("rig session", Name::as_str);
    let rendered = render::render(title, conversation.messages());
    let action = if update.is_some() { "Update" } else { "Share" };
    let mut body = format!(
        "{action} the {} messages of this session on {target}, as an HTML page.",
        rendered.messages
    );
    if rendered.secrets.is_empty() {
        body.push_str("\nNo secrets found.");
    } else {
        body.push_str("\nSecrets found, redacted in the page:");
        for secret in &rendered.secrets {
            body.push_str("\n  ");
            body.push_str(secret);
        }
    }
    let op = Op::Publish {
        html: rendered.html,
        update,
    };
    let effect = spawn_upload(&mut commands, agent, &settings, op);
    if ask {
        let question = Ask::new("Share this session?")
            .body(body)
            .choice("cancel", "Cancel")
            .choice("share", format!("{action} on {target}"));
        commands
            .spawn((question, ForAgent(agent), HoldOf(effect)))
            .observe(answered);
    }
}

/// `/unshare`: deletes the page in the background.
fn unshare(
    In(args): In<CommandArgs>,
    agents: Query<&Shared>,
    in_flight: InFlight,
    settings: Res<ShareSettings>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let Ok(shared) = agents.get(agent) else {
        notices.write(Notice::error(agent, "This session is not shared."));
        return;
    };
    let target = settings.label();
    if shared.target != target {
        let why = format!(
            "Shared on {}, not on the configured {target}.",
            shared.target
        );
        notices.write(Notice::error(agent, why));
        return;
    }
    if in_flight.iter().any(|of| of.0 == agent) {
        notices.write(Notice::error(
            agent,
            "A share of this session is under way.",
        ));
        return;
    }
    let op = Op::Delete {
        id: shared.id.clone(),
    };
    spawn_upload(&mut commands, agent, &settings, op);
}

/// Spawns the open `share` effect doing `op` for `agent`: it starts when
/// nothing holds it, and is recorded with what it did.
fn spawn_upload(
    commands: &mut Commands,
    agent: Entity,
    settings: &ShareSettings,
    op: Op,
) -> Entity {
    let (action, id) = match &op {
        Op::Publish { update, .. } => ("publish", update.clone()),
        Op::Delete { id } => ("delete", Some(id.clone())),
    };
    let payload = serde_json::json!({ "action": action, "target": settings.label(), "id": id });
    let kind = EffectKind::Custom {
        kind: KIND.into(),
        payload,
    };
    let upload = Upload {
        settings: settings.clone(),
        op,
    };
    let request = EffectRequest {
        kind,
        handler: None,
    };
    commands.spawn((request, upload, ForAgent(agent))).id()
}

/// Lets the upload the question holds start, or refuses it, once the user
/// answered; the question lets go of it either way.
fn answered(
    answer: On<Add<Outcome>>,
    asks: Query<(&Outcome, &HoldOf)>,
    mut uploads: Query<&mut Decision, With<Upload>>,
    mut commands: Commands,
) {
    let Ok((outcome, &HoldOf(effect))) = asks.get(answer.entity) else {
        return;
    };
    if !Answer::of(outcome).is_some_and(|answer| answer.is("share"))
        && let Ok(mut decision) = uploads.get_mut(effect)
    {
        decision.decide(Decider::plugin("share"), Outcome::refused("not shared"));
    }
    commands.entity(answer.entity).remove::<HoldOf>();
}

/// Runs each upload that started on the IO pool; its result is its
/// outcome.
fn upload(
    started: On<Add<Started>>,
    uploads: Query<&Upload>,
    processes: Res<Processes>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let Ok(upload) = uploads.get(started.entity) else {
        return;
    };
    let (settings, op) = (upload.settings.clone(), upload.op.clone());
    let processes = processes.clone();
    let work = async move {
        let outcome = match run(&settings, op, &processes).await {
            Ok(link) => Ok(Answered::Custom {
                payload: serde_json::to_value(link).unwrap_or_default(),
            }),
            Err(why) => Err(ErrorReport::new(ErrorKind::Other, why)),
        };
        Outcome(outcome)
    };
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    commands
        .entity(started.entity)
        .insert(Running::answering(pool, &wake, work));
}

/// Says how each settled upload went, and keeps or forgets the link.
fn report(
    settled: On<Add<Settled>>,
    uploads: Query<(&Upload, &Outcome, &ForAgent)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
    mut clipboard: MessageWriter<CopyToClipboard>,
    mut out: Option<Publish>,
) {
    let Ok((upload, outcome, &ForAgent(agent))) = uploads.get(settled.entity) else {
        return;
    };
    match (&upload.op, &outcome.0) {
        (_, Err(report)) if report.refusal => {
            notices.write(Notice::info(
                agent,
                format!("Not shared: {}.", report.message),
            ));
        }
        (_, Err(report)) => {
            notices.write(Notice::error(
                agent,
                format!("Share failed: {}", report.message),
            ));
        }
        (Op::Delete { .. }, Ok(_)) => {
            notices.write(Notice::info(agent, "Unshared."));
            commands.entity(agent).remove::<Shared>();
        }
        (Op::Publish { .. }, Ok(Answered::Custom { payload })) => {
            if let Ok(link) = Link::deserialize(payload) {
                notices.write(Notice::info(agent, format!("Shared: {}", link.url)));
                if upload.settings.clipboard {
                    clipboard.write(CopyToClipboard(link.url.clone()));
                    if let Some(out) = out.as_mut() {
                        out.event(&SharedLink {
                            url: link.url.clone(),
                        });
                    }
                }
                commands.entity(agent).insert(Shared {
                    target: upload.settings.label(),
                    id: link.id,
                    url: link.url,
                });
            }
        }
        (Op::Publish { .. }, Ok(_)) => {}
    }
}

/// Does `op` at the configured target: the page's link, none once
/// deleted, or the target's error.
async fn run(
    settings: &ShareSettings,
    op: Op,
    processes: &Processes,
) -> Result<Option<Link>, String> {
    let (base, headers, (url_key, id_key)) = match settings.target {
        Target::Http => {
            let base = settings.url.trim_end_matches('/').to_owned();
            let html = vec![("content-type", "text/html; charset=utf-8".to_owned())];
            (base, html, ("url", "id"))
        }
        Target::Gist => {
            let token = gist_token(processes.clone()).await?;
            let headers = vec![
                ("authorization", format!("Bearer {token}")),
                ("accept", "application/vnd.github+json".to_owned()),
                ("user-agent", "rig-share".to_owned()),
            ];
            (GIST_API.to_owned(), headers, ("html_url", "id"))
        }
    };
    let (method, uri, html) = match op {
        Op::Publish { html, update: None } => (Method::POST, base, html),
        Op::Publish {
            html,
            update: Some(id),
        } => {
            let method = match settings.target {
                Target::Http => Method::PUT,
                Target::Gist => Method::PATCH,
            };
            (method, format!("{base}/{id}"), html)
        }
        Op::Delete { id } => {
            send(
                Method::DELETE,
                format!("{base}/{id}"),
                headers,
                String::new(),
            )
            .await?;
            return Ok(None);
        }
    };
    let body = match settings.target {
        Target::Http => html,
        Target::Gist => serde_json::json!({
            "description": "A rig session",
            "public": false,
            "files": { GIST_FILE: { "content": html } },
        })
        .to_string(),
    };
    let answer = send(method, uri, headers, body).await?;
    parse_link(&answer, url_key, id_key).map(Some)
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
    Ok(Link { id, url })
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
async fn gist_token(processes: Processes) -> Result<String, String> {
    for name in ["GH_TOKEN", "GITHUB_TOKEN"] {
        if let Ok(token) = std::env::var(name)
            && !token.is_empty()
        {
            return Ok(token);
        }
    }
    blocking(move || {
        let output = processes
            .launcher()
            .command("gh")
            .args(["auth", "token"])
            .output()
            .map_err(ToolExecutionError::from_error)?;
        let token = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        if output.status.success() && !token.is_empty() {
            Ok(token)
        } else {
            let why = String::from_utf8_lossy(&output.stderr).trim().to_owned();
            Err(ToolExecutionError::other(why))
        }
    })
    .await
    .map_err(|error| format!("no GitHub token (set GH_TOKEN or run `gh auth login`): {error}"))
}

/// A link to copy to the clipboard, for the views.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct SharedLink {
    /// The page's link.
    pub url: String,
}

impl ChannelEvent for SharedLink {
    const KIND: &'static str = "x/share/link";
}

/// The view half: copies a shared link to the clipboard; named in
/// `plugins.toml` as the entry's `view`.
#[derive(Default)]
pub struct ShareView;

impl Plugin for ShareView {
    fn build(&self, app: &mut App) {
        app.add_message::<Applied>()
            .add_message::<CopyToClipboard>()
            .add_systems(Update, copy_links);
    }
}

/// Copies each link the agent shared, in the view.
fn copy_links(mut applied: MessageReader<Applied>, mut clipboard: MessageWriter<CopyToClipboard>) {
    for Applied(record) in applied.read() {
        if record.kind != SharedLink::KIND {
            continue;
        }
        if let Ok(link) = serde_json::from_value::<SharedLink>(record.data.clone()) {
            clipboard.write(CopyToClipboard(link.url));
        }
    }
}
