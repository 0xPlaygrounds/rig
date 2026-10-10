//! Sessions beyond the one running: `/new`, `/resume` and `/name`. The end
//! of each turn rewrites the session's [`SessionDir::meta`] (its
//! [`SessionTitle`], cost and when it was updated), which `/resume` lists
//! and a resumed session reads its title back from. Running another
//! session is the launcher's job, so the agent stays one session per
//! process: it names the next session in [`SessionDir::switch`] and exits
//! with the reload code, as `/reload` does, and the launcher starts it in
//! its own directory.

use std::cmp::Reverse;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bevy_app::OnAppExitSystems;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::error;
use bevy_reflect::prelude::*;
use rig_core::completion::Message;
use rig_core::message::UserContent;
use rig_harness::harness_protocol::{Home, RELOAD_EXIT_CODE, SessionDir, SessionId};
use rig_tools::fs::{read_json, write_json};
use rig_tools::shorten;
use serde::{Deserialize, Serialize};

use rig_core::completion::{UsageTotals, dollars_label, tokens_label};
use rig_ecs::StopTurns;
use rig_ecs::agent::{
    ActiveTurn, Agent, AgentId, Conversation, Notice, SpawnedBy, TurnOf, primary_order,
};
use rig_ecs::commands::{AppCommandsExt, CommandArgs};
use rig_ecs::journal::now_ms;
use rig_harness::front::{PickItem, PickRequest, attached_file};
use rig_harness::prelude::{SessionPaths, launcher};
use rig_telemetry::Spending;

/// Most characters of a session's title.
const TITLE_CHARS: usize = 60;
/// Most sessions `/resume` lists.
const LISTED: usize = 200;

/// `/new`, `/resume`, `/name`, and the listing cache written at the end of
/// each turn.
#[derive(Default)]
pub struct SessionsPlugin;

impl Plugin for SessionsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SessionTitle>()
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
            .add_systems(PreStartup, restore_title)
            .add_systems(
                Last,
                write_meta.in_set(OnAppExitSystems).after(StopTurns).run_if(
                    any_component_removed::<ActiveTurn>
                        .or_eager(on_message::<AppExit>)
                        .or_eager(resource_changed::<SessionTitle>),
                ),
            );
    }
}

/// What the session is listed as: the name `/name` gave it, else the start
/// of the first message the user typed, kept from the first turn on.
#[derive(Resource, Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Resource, Clone, Debug, Default, PartialEq)]
#[serde(default)]
pub struct SessionTitle {
    /// The name set with `/name`.
    pub name: Option<String>,
    /// The start of the first message typed; empty until a turn ended.
    #[serde(rename = "title")]
    pub typed: String,
}

/// Run another session: the one named, or a new one. Refused while a turn
/// runs, and without the launcher.
#[derive(Event, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct SwitchSession {
    /// The [`SessionId`] to resume, or `None` for a new session.
    pub session: Option<String>,
}

/// What a session's `meta.json` holds, for listing it.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[serde(default)]
pub struct Meta {
    /// What it is listed as.
    #[serde(flatten)]
    pub title: SessionTitle,
    /// What the session's model calls cost, in USD, or `None` when no call
    /// was priced, such as a local or subscription model's.
    pub cost: Option<f64>,
    /// The tokens the session's model calls read and wrote.
    pub tokens: u64,
    /// When it was written, in milliseconds since the Unix epoch.
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
        let title = match (&self.meta.title.name, self.meta.title.typed.as_str()) {
            (Some(name), _) => name.clone(),
            (None, "") => format!("session {}", self.id),
            (None, title) => title.to_owned(),
        };
        let mut label = format!("{age:>8}  {title}");
        // The cost before the directory, which may be long enough to be
        // cut off.
        match self.meta.cost {
            Some(cost) if cost > 0.0 => {
                label.push_str(&format!("  · {}", dollars_label(cost)));
            }
            _ if self.meta.tokens > 0 => {
                label.push_str(&format!("  · {} tokens", tokens_label(self.meta.tokens)));
            }
            _ => {}
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
            let meta: Option<Meta> = read_json(&dir.meta());
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

/// Reads the session's title from its listing cache.
fn restore_title(paths: Option<Res<SessionPaths>>, mut title: ResMut<SessionTitle>) {
    let Some(paths) = paths else {
        return;
    };
    if let Some(meta) = read_json::<Meta>(&paths.meta()) {
        // Not a change: nothing to write back.
        *title.bypass_change_detection() = meta.title;
    }
}

/// Rewrites the session's listing cache, at the end of a turn, on a new
/// name and on exit, once some agent logged a message; the first time, the
/// title takes the start of the first message typed.
fn write_meta(
    paths: Option<Res<SessionPaths>>,
    mut title: ResMut<SessionTitle>,
    agents: Query<(&AgentId, &Conversation, Option<&Spending>, Has<SpawnedBy>), With<Agent>>,
) {
    let Some(paths) = paths.filter(|paths| paths.is_saved()) else {
        return;
    };
    // The agents the user started come first: the title is theirs, not a
    // spawned agent's.
    let mut agents: Vec<_> = agents.iter().collect();
    agents.sort_by(|a, b| primary_order(a.3, a.0).cmp(&primary_order(b.3, b.0)));
    let mut spent = UsageTotals::default();
    agents
        .iter()
        .filter_map(|(_, _, each, ..)| *each)
        .for_each(|Spending(each)| spent.add(each));
    if title.typed.is_empty()
        && let Some(typed) = agents
            .iter()
            .find_map(|(_, conversation, ..)| first_typed(conversation))
    {
        title.bypass_change_detection().typed = one_line(&typed);
    }
    let meta = Meta {
        title: title.clone(),
        cost: (spent.unpriced < spent.calls).then_some(spent.cost),
        tokens: spent.total_tokens(),
        updated: now_ms(),
    };
    if let Err(failure) = write_json(&paths.meta(), &meta) {
        error!("could not write the session's meta.json: {failure}");
    }
}

/// The first text the user typed in `conversation`: never text an agent
/// or a plugin delivered, such as a notice or a build failure, nor a
/// file it attached.
fn first_typed(conversation: &Conversation) -> Option<String> {
    let mut messages = conversation.messages().iter().enumerate();
    messages.find_map(|(at, message)| {
        let Message::User { content } = message else {
            return None;
        };
        content
            .iter()
            .enumerate()
            .find_map(|(index, item)| match item {
                UserContent::Text(text)
                    if !text.text.trim().is_empty()
                        && conversation.origin(at, index).is_none()
                        && attached_file(&text.text).is_none() =>
                {
                    Some(text.text.clone())
                }
                _ => None,
            })
    })
}

/// `text` on one line, cut to [`TITLE_CHARS`].
fn one_line(text: &str) -> String {
    shorten(
        &text.split_whitespace().collect::<Vec<_>>().join(" "),
        TITLE_CHARS,
    )
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
    paths: Option<Res<SessionPaths>>,
    mut commands: Commands,
    mut picks: MessageWriter<PickRequest>,
    mut notices: MessageWriter<Notice>,
) {
    if args.args.is_empty() {
        let Some(paths) = paths else {
            return;
        };
        let items: Vec<PickItem> = list(&Home::from_env(), paths.path())
            .into_iter()
            .map(|session| PickItem {
                label: session.label(),
                command: format!("resume {}", session.id),
            })
            .collect();
        if items.is_empty() {
            notices.write(Notice::info(args.agent, "No earlier session to resume."));
            return;
        }
        picks.write(PickRequest {
            agent: args.agent,
            title: "Resume a session".to_owned(),
            items,
            selected: 0,
        });
    } else {
        commands.trigger(SwitchSession {
            session: Some(args.args),
        });
    }
}

fn name(
    In(args): In<CommandArgs>,
    mut title: ResMut<SessionTitle>,
    mut notices: MessageWriter<Notice>,
) {
    if args.args.is_empty() {
        let text = match &title.name {
            Some(current) => format!("This session is named “{current}”."),
            None => "This session has no name; /name <title> gives it one.".to_owned(),
        };
        notices.write(Notice::info(args.agent, text));
        return;
    }
    let name = one_line(&args.args);
    notices.write(Notice::info(
        args.agent,
        format!("This session is now named “{name}”."),
    ));
    title.name = Some(name);
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
