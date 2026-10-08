//! Sessions beyond the one running: `/new`, `/resume` and `/name`. The end
//! of each turn rewrites the session's [`SessionDir::meta`] (its name,
//! title, directory, model, cost and when it was updated), which `/resume`
//! lists. Running another session is the launcher's job, so the agent stays
//! one session per process: it names the next session in
//! [`SessionDir::switch`] and exits with the reload code, as `/reload`
//! does, and the launcher starts it in its own directory.

use std::cmp::Reverse;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bevy_app::OnAppExitSystems;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::error;
use bevy_reflect::prelude::*;
use rig::harness_protocol::{Home, RELOAD_EXIT_CODE, SessionDir, SessionId};
use rig_core::completion::Message;
use rig_core::message::UserContent;
use serde::{Deserialize, Serialize};

use super::launcher;
use crate::core::agent::{
    Agent, AgentId, Conversation, ModelChoice, Notice, PickKind, PickRequest, TurnFinished, TurnOf,
};
use crate::core::commands::{AppCommandsExt, CommandArgs};
use crate::core::journal::{SessionPaths, now_ms};
use crate::core::subagents::Delegated;
use crate::core::usage::{self, Spending};

/// Most characters of a session's title.
const TITLE_CHARS: usize = 60;
/// Most sessions `/resume` lists.
const LISTED: usize = 200;

/// `/new`, `/resume`, `/name`, and the listing cache written at the end of
/// each turn.
pub struct SessionsPlugin;

impl Plugin for SessionsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SessionName>()
            .add_command("new", "Start a new session; this one is kept", new)
            .add_command(
                "resume",
                "Pick an earlier session to resume, or /resume <id>",
                resume,
            )
            .add_command(
                "name",
                "Name this session for /resume, or show its name",
                name,
            )
            .add_observer(on_switch_session)
            .add_systems(PreStartup, restore_name)
            .add_systems(Startup, record_directory)
            .add_systems(
                Last,
                write_meta
                    .in_set(OnAppExitSystems)
                    .after(crate::core::turn::stop_turns_on_exit)
                    .run_if(
                        on_message::<TurnFinished>
                            .or_eager(on_message::<AppExit>)
                            .or_eager(resource_changed::<SessionName>),
                    ),
            );
    }
}

/// The session's name, set with `/name`; `None` until then, when lists
/// show its first message instead.
#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub struct SessionName(pub Option<String>);

/// Run another session: the one named, or a new one. Refused while a turn
/// runs, and without the launcher.
#[derive(Event, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct SwitchSession {
    /// The [`SessionId`] to resume, or `None` for a new session.
    pub session: Option<String>,
}

/// What a session's `meta.json` holds: a cache for listing it, of facts
/// its agent logs hold too, but for the name.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Meta {
    /// The start of the first message typed.
    #[serde(default)]
    pub title: String,
    /// The name set with `/name`.
    #[serde(default)]
    pub name: Option<String>,
    /// The working directory it ran in.
    #[serde(default)]
    pub cwd: Option<PathBuf>,
    /// The first agent's model.
    #[serde(default)]
    pub model: Option<String>,
    /// What the session's model calls cost, in USD.
    #[serde(default)]
    pub cost: f64,
    /// When it was written, in milliseconds since the Unix epoch.
    #[serde(default)]
    pub updated: u64,
}

/// An earlier session, as `/resume` lists it.
#[derive(Clone, Debug)]
pub struct SessionEntry {
    /// Its id.
    pub id: SessionId,
    /// Its listing cache; empty when no turn of it ended.
    pub meta: Meta,
    /// The directory it runs in, if known.
    pub directory: Option<PathBuf>,
    /// When it last changed.
    pub saved: SystemTime,
}

impl SessionEntry {
    /// One line: age, name or title, cost, directory.
    pub fn label(&self) -> String {
        let age = age(self.saved.elapsed().unwrap_or_default());
        let title = match (&self.meta.name, self.meta.title.as_str()) {
            (Some(name), _) => name.clone(),
            (None, "") => format!("session {}", self.id),
            (None, title) => title.to_owned(),
        };
        let mut label = format!("{age:>8}  {title}");
        // The cost before the directory, which may be long enough to be
        // cut off.
        if self.meta.cost > 0.0 {
            label.push_str(&format!("  · {}", usage::dollars(self.meta.cost)));
        }
        if let Some(directory) = &self.directory {
            label.push_str(&format!("  · {}", tilde(directory)));
        }
        label
    }
}

/// The sessions under `home` with an agent log, other than `current`,
/// newest first.
pub fn list(home: &Home, current: &Path) -> Vec<SessionEntry> {
    let Ok(entries) = fs::read_dir(home.sessions()) else {
        return Vec::new();
    };
    let mut sessions: Vec<SessionEntry> = entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let id: SessionId = entry.file_name().to_str()?.parse().ok()?;
            let dir = home.session(&id);
            if dir.path() == current || !dir.is_saved() {
                return None;
            }
            let meta: Option<Meta> = fs::read(dir.meta())
                .ok()
                .and_then(|bytes| serde_json::from_slice(&bytes).ok());
            let saved = match &meta {
                Some(meta) => UNIX_EPOCH + Duration::from_millis(meta.updated),
                None => fs::metadata(dir.path()).ok()?.modified().ok()?,
            };
            Some(SessionEntry {
                directory: dir.working_directory(),
                id,
                meta: meta.unwrap_or_default(),
                saved,
            })
        })
        .collect();
    sessions.sort_by_key(|session| Reverse(session.saved));
    sessions.truncate(LISTED);
    sessions
}

/// How long ago, roughly: `3m ago`, `5h ago`, `2d ago`.
fn age(elapsed: Duration) -> String {
    let seconds = elapsed.as_secs();
    match seconds {
        0..60 => "just now".to_owned(),
        60..3600 => format!("{}m ago", seconds / 60),
        3600..86_400 => format!("{}h ago", seconds / 3600),
        _ => format!("{}d ago", seconds / 86_400),
    }
}

/// `path` with the home directory shown as `~`.
fn tilde(path: &Path) -> String {
    match std::env::home_dir().and_then(|home| path.strip_prefix(home).ok().map(Path::to_owned)) {
        Some(rest) if rest.as_os_str().is_empty() => "~".to_owned(),
        Some(rest) => format!("~/{}", rest.display()),
        None => path.display().to_string(),
    }
}

/// Reads the session's name from its listing cache.
fn restore_name(paths: Option<Res<SessionPaths>>, mut name: ResMut<SessionName>) {
    let Some(paths) = paths else {
        return;
    };
    if let Some(meta) = fs::read(paths.meta())
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Meta>(&bytes).ok())
    {
        // Not a change: nothing to write back.
        name.bypass_change_detection().0 = meta.name;
    }
}

/// Records the working directory of a session the launcher did not start,
/// so `/resume` can list it.
fn record_directory(paths: Option<Res<SessionPaths>>) {
    let Some(paths) = paths else {
        return;
    };
    if paths.directory().exists() {
        return;
    }
    if let Ok(here) = std::env::current_dir()
        && let Err(failure) = fs::write(paths.directory(), here.to_string_lossy().as_bytes())
    {
        error!("could not record the session's directory: {failure}");
    }
}

/// Rewrites the session's listing cache, at the end of a turn, on a new
/// name and on exit, once some agent logged a message.
fn write_meta(
    paths: Option<Res<SessionPaths>>,
    name: Res<SessionName>,
    agents: Query<
        (
            &AgentId,
            &Conversation,
            &Spending,
            Option<&ModelChoice>,
            Has<Delegated>,
        ),
        With<Agent>,
    >,
) {
    let Some(paths) = paths.filter(|paths| paths.is_saved()) else {
        return;
    };
    // The agents the user started come first: the title and model are
    // theirs, not a subagent's.
    let mut agents: Vec<_> = agents.iter().collect();
    agents.sort_by(|a, b| (a.4, &a.0.0).cmp(&(b.4, &b.0.0)));
    let meta = Meta {
        title: agents
            .iter()
            .find_map(|(_, conversation, ..)| first_typed(conversation.messages()))
            .map(|text| title(&text))
            .unwrap_or_default(),
        name: name.0.clone(),
        cwd: paths
            .working_directory()
            .or_else(|| std::env::current_dir().ok()),
        model: agents
            .first()
            .and_then(|(_, _, _, model, _)| model.map(|model| model.0.clone())),
        cost: agents.iter().map(|(_, _, spent, ..)| spent.cost).sum(),
        updated: now_ms(),
    };
    let written = serde_json::to_vec_pretty(&meta)
        .map_err(|failure| failure.to_string())
        .and_then(|bytes| {
            let temporary = paths.meta().with_extension("json.tmp");
            fs::write(&temporary, bytes)
                .and_then(|()| fs::rename(&temporary, paths.meta()))
                .map_err(|failure| failure.to_string())
        });
    if let Err(failure) = written {
        error!("could not write the session's meta.json: {failure}");
    }
}

/// The first text the user typed in `conversation`.
fn first_typed(conversation: &[Message]) -> Option<String> {
    conversation.iter().find_map(|message| match message {
        Message::User { content } => content.iter().find_map(|item| match item {
            UserContent::Text(text) if !text.text.trim().is_empty() => Some(text.text.clone()),
            _ => None,
        }),
        _ => None,
    })
}

/// `text` on one line, cut to [`TITLE_CHARS`].
fn title(text: &str) -> String {
    let line = text.split_whitespace().collect::<Vec<_>>().join(" ");
    match line.char_indices().nth(TITLE_CHARS) {
        Some((cut, _)) => format!("{}…", line.get(..cut).unwrap_or(&line)),
        None => line,
    }
}

fn new(
    In(args): In<CommandArgs>,
    conversations: Query<&Conversation, With<Agent>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if conversations
        .iter()
        .all(|conversation| conversation.messages().is_empty())
    {
        notices.write(Notice::info(args.agent, "This session is new already."));
        return;
    }
    commands.trigger(SwitchSession { session: None });
}

fn resume(
    In(args): In<CommandArgs>,
    mut commands: Commands,
    mut picks: MessageWriter<PickRequest>,
) {
    if args.args.is_empty() {
        picks.write(PickRequest {
            agent: args.agent,
            kind: PickKind::Session,
        });
    } else {
        commands.trigger(SwitchSession {
            session: Some(args.args),
        });
    }
}

fn name(
    In(args): In<CommandArgs>,
    mut name: ResMut<SessionName>,
    mut notices: MessageWriter<Notice>,
) {
    if args.args.is_empty() {
        let text = match &name.0 {
            Some(current) => format!("This session is named “{current}”."),
            None => "This session has no name; /name <title> gives it one.".to_owned(),
        };
        notices.write(Notice::info(args.agent, text));
        return;
    }
    let title = title(&args.args);
    notices.write(Notice::info(
        args.agent,
        format!("This session is now named “{title}”."),
    ));
    name.0 = Some(title);
}

/// Names the next session for the launcher and exits for it.
fn on_switch_session(
    switch: On<SwitchSession>,
    turns: Query<(), With<TurnOf>>,
    paths: Option<Res<SessionPaths>>,
    mut exit: MessageWriter<AppExit>,
    mut notices: MessageWriter<Notice>,
) {
    let refusal = if !turns.is_empty() {
        Some("A turn is running. Press Esc to stop it first.".to_owned())
    } else if launcher::executable().is_none() {
        Some("Switching sessions needs the rig launcher: start the agent with `rig`.".to_owned())
    } else {
        None
    };
    if let Some(refusal) = refusal {
        notices.write(Notice::info(None, refusal));
        return;
    }
    let Some(paths) = paths else {
        return;
    };
    let target = match &switch.session {
        None => String::new(),
        Some(id) => match checked(id, &paths.0) {
            Ok(id) => id.to_string(),
            Err(why) => {
                notices.write(Notice::error(None, why));
                return;
            }
        },
    };
    match fs::write(paths.switch(), target) {
        Ok(()) => {
            exit.write(AppExit::from_code(RELOAD_EXIT_CODE));
        }
        Err(failure) => {
            notices.write(Notice::error(
                None,
                format!("Could not switch sessions: {failure}"),
            ));
        }
    }
}

/// `id` when it names a session with an agent log other than the running
/// one.
fn checked(id: &str, current: &SessionDir) -> Result<SessionId, String> {
    let id: SessionId = id.trim().parse().map_err(|failure| format!("{failure}."))?;
    let dir = Home::from_env().session(&id);
    if dir.path() == current.path() {
        return Err("That is this session.".to_owned());
    }
    if !dir.is_saved() {
        return Err(format!("No saved session {id}."));
    }
    Ok(id)
}
