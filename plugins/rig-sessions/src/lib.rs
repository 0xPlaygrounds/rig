//! Sessions beyond the one running: `/new`, `/resume` and `/name`, and
//! [`NameSession`] for a plugin naming any session. The end
//! of each turn rewrites the session's [`Meta`] in its store (its
//! [`SessionTitle`], the [`ResumeLabel`]s other plugins add, cost and when
//! it was updated), which `/resume` lists and a resumed session reads its
//! title back from. With the default
//! `tui` feature, `/resume` without arguments is a picker and the terminal
//! view's status line shows the session's name. Running another
//! session is the launcher's job, so the agent stays one session per
//! process: it names the next session in [`SessionDir::switch`] and exits
//! with the reload code, as `/reload` does, and the launcher starts it in
//! its own directory.

use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::error;
use bevy_reflect::prelude::*;
use rig_harness::harness_protocol::{Home, SessionDir, SessionId};
use rig_tools::shorten;
use serde::{Deserialize, Serialize};

use rig_core::completion::{dollars_label, tokens_label};
use rig_ecs::WriteJournal;
use rig_ecs::agent::{ActiveTurn, Agent, AgentId, Conversation, Notice, SpawnedBy, TurnOf};
use rig_ecs::journal::SessionStore;
use rig_ecs::prelude::now_ms;
use rig_harness::commands::{AppCommandsExt, CommandArgs};
use rig_harness::prelude::{
    Exit, ExitReason, Files, RigHome, SessionPaths, SessionStores, first_typed,
};
use rig_harness::primary::primary_order;
use rig_telemetry::SessionSpending;

#[cfg(feature = "tui")]
mod tui;

/// Most characters of a session's title.
const TITLE_CHARS: usize = 60;
/// Most sessions `/resume` lists.
const LISTED: usize = 200;
/// The file of a session's store that holds its [`Meta`]
/// (`files/meta.json` in its directory).
const META: &str = "meta.json";

/// `/new`, `/resume`, `/name`, and the listing cache written at the end of
/// each turn.
#[derive(Default)]
pub struct SessionsPlugin;

impl Plugin for SessionsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SessionTitle>()
            .init_resource::<RigHome>()
            .init_resource::<SessionStores>()
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
            .add_observer(name_session)
            .add_systems(PreStartup, restore_title)
            .add_systems(
                Last,
                update_meta.in_set(WriteJournal).run_if(
                    any_component_removed::<ActiveTurn>
                        .or_eager(on_message::<AppExit>)
                        .or_eager(resource_changed::<SessionTitle>)
                        .or_eager(labels_changed),
                ),
            );
        #[cfg(feature = "tui")]
        tui::add(app);
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

/// A part of this session's line in `/resume`, such as its goal: spawn
/// one and keep its `text` current. It is written with the session's
/// listing cache ([`Meta::labels`]), so `/resume` shows it for sessions
/// that are not running, after the title, by `tag`; an empty `text`, or
/// Bevy's `Disabled` on the entity, leaves it out.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Clone, Debug, Default)]
pub struct ResumeLabel {
    /// What it is, and what sorts it, such as `goal`.
    pub tag: String,
    /// What `/resume` shows.
    pub text: String,
}

impl ResumeLabel {
    /// `text` in this session's line, sorted by `tag`.
    pub fn new(tag: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            tag: tag.into(),
            text: text.into(),
        }
    }
}

/// Run another session: the one named, or a new one. Refused while a turn
/// runs, and without the launcher.
#[derive(Event, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct SwitchSession {
    /// The [`SessionId`] to resume, or `None` for a new session.
    pub session: Option<String>,
}

/// Names a session: its title in `/resume`, and nothing else of its
/// listing, such as what it cost. Naming this session is `/name`'s.
#[derive(Event, Clone, Debug)]
pub struct NameSession {
    /// The session.
    pub session: SessionId,
    /// Its name.
    pub name: String,
}

/// What `/resume` lists of a session, kept in its store: what it is
/// listed as, what it cost and when it was updated. The agent logs hold
/// the same facts; a plugin that writes a session, such as an import,
/// writes it with [`write_meta`].
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[serde(default)]
pub struct Meta {
    /// What it is listed as.
    #[serde(flatten)]
    pub title: SessionTitle,
    /// The [`ResumeLabel`]s' texts, by tag.
    pub labels: BTreeMap<String, String>,
    /// What the session's model calls cost, in USD, or `None` when no call
    /// was priced, such as a local or subscription model's.
    pub cost: Option<f64>,
    /// The tokens the session's model calls read and wrote.
    pub tokens: u64,
    /// When it was written, in milliseconds since the Unix epoch.
    pub updated: u64,
}

impl Meta {
    /// The listing of the session kept in `store`, if it has one.
    pub fn of(store: &dyn Files) -> Option<Self> {
        serde_json::from_slice(&store.read_file(META).ok()?).ok()
    }
}

/// Keeps `meta` as the listing of the session in `store`.
///
/// # Errors
///
/// When the store cannot write it.
pub fn write_meta(store: &dyn Files, meta: &Meta) -> io::Result<()> {
    store.write_file(META, &serde_json::to_vec_pretty(meta)?)
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
    /// One line: name or title, the [`ResumeLabel`]s, directory.
    pub fn label(&self) -> String {
        let mut label = match (&self.meta.title.name, self.meta.title.typed.as_str()) {
            (Some(name), _) => name.clone(),
            (None, "") => format!("session {}", self.id),
            (None, title) => title.to_owned(),
        };
        for text in self.meta.labels.values() {
            label.push_str(&format!("  · {text}"));
        }
        if let Some(directory) = &self.directory {
            label.push_str(&format!("  · {}", tilde(directory)));
        }
        label
    }

    /// What it cost, or its tokens, and its age, shown whole beside the
    /// label.
    pub fn detail(&self) -> String {
        let age = age(self.saved.elapsed().unwrap_or_default());
        match self.meta.cost {
            Some(cost) if cost > 0.0 => format!("{} · {age}", dollars_label(cost)),
            _ if self.meta.tokens > 0 => {
                format!("{} tokens · {age}", tokens_label(self.meta.tokens))
            }
            _ => age,
        }
    }
}

/// The saved sessions under `home` other than `current`, newest first,
/// each read from its store as `stores` open it.
pub fn list(home: &Home, stores: &SessionStores, current: &Path) -> Vec<SessionEntry> {
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
            let meta = stores.open(&dir).ok().and_then(|store| Meta::of(&*store));
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
fn restore_title(store: Option<Res<SessionStore>>, mut title: ResMut<SessionTitle>) {
    let Some(store) = store else {
        return;
    };
    if let Some(meta) = Meta::of(&*store.0) {
        // Not a change: nothing to write back.
        *title.bypass_change_detection() = meta.title;
    }
}

/// Whether a [`ResumeLabel`] changed, came or went.
fn labels_changed(
    labels: Query<(), Changed<ResumeLabel>>,
    mut removed: RemovedComponents<ResumeLabel>,
) -> bool {
    removed.read().count() > 0 || !labels.is_empty()
}

/// Rewrites the session's listing cache, at the end of a turn, on a new
/// name or label and on exit, once some agent logged a message; the first
/// time, the title takes the start of the first message typed.
fn update_meta(
    (paths, store): (Option<Res<SessionPaths>>, Option<Res<SessionStore>>),
    mut title: ResMut<SessionTitle>,
    agents: Query<(&AgentId, &Conversation, Has<SpawnedBy>), With<Agent>>,
    labels: Query<&ResumeLabel>,
    spent: Option<Res<SessionSpending>>,
) {
    let (Some(_), Some(store)) = (paths.filter(|paths| paths.is_saved()), store) else {
        return;
    };
    // The agents the user started come first: the title is theirs, not a
    // spawned agent's.
    let mut agents: Vec<_> = agents.iter().collect();
    agents.sort_by(|a, b| primary_order(a.2, a.0).cmp(&primary_order(b.2, b.0)));
    if title.typed.is_empty()
        && let Some(typed) = agents
            .iter()
            .find_map(|(_, conversation, _)| first_typed(conversation))
    {
        title.bypass_change_detection().typed = title_of(&typed);
    }
    let labels = labels
        .iter()
        .filter(|label| !label.text.is_empty())
        .map(|label| (label.tag.clone(), title_of(&label.text)))
        .collect();
    let spent = spent.map(|spent| spent.0.total).unwrap_or_default();
    let meta = Meta {
        title: title.clone(),
        labels,
        cost: (spent.unpriced < spent.calls).then_some(spent.cost),
        tokens: spent.total_tokens(),
        updated: now_ms(),
    };
    if let Err(failure) = write_meta(&*store.0, &meta) {
        error!("could not write the session's meta.json: {failure}");
    }
}

/// A session title or label of `text`, as `/resume` lists it: on one line,
/// its runs of whitespace one space, cut to 60 characters. A plugin that
/// writes a session's [`Meta`] titles it so.
pub fn title_of(text: &str) -> String {
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

/// `/resume <id>` runs that session; without arguments, the terminal view
/// lets the user pick one.
fn resume(
    In(args): In<CommandArgs>,
    #[cfg(feature = "tui")] (home, stores): (Res<RigHome>, Res<SessionStores>),
    #[cfg(feature = "tui")] paths: Option<Res<SessionPaths>>,
    #[cfg(feature = "tui")] mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    if args.args.is_empty() {
        #[cfg(feature = "tui")]
        tui::pick_session(
            args.agent,
            (&home.0, &stores, paths.as_deref()),
            &mut notices,
            &mut commands,
        );
        return;
    }
    commands.trigger(SwitchSession {
        session: Some(args.args),
    });
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
    let name = title_of(&args.args);
    notices.write(Notice::info(
        args.agent,
        format!("This session is now named “{name}”."),
    ));
    title.name = Some(name);
}

/// Names the session asked for: this one's [`SessionTitle`], another's
/// listing cache, which keeps its counts.
fn name_session(
    named: On<NameSession>,
    paths: Option<Res<SessionPaths>>,
    (home, stores): (Res<RigHome>, Res<SessionStores>),
    mut title: ResMut<SessionTitle>,
    mut notices: MessageWriter<Notice>,
) {
    let name = title_of(&named.name);
    if paths.and_then(|paths| paths.id()).as_ref() == Some(&named.session) {
        title.name = Some(name);
        return;
    }
    let dir = home.0.session(&named.session);
    let written = if dir.path().is_dir() {
        stores
            .open(&dir)
            .and_then(|store| Ok(rename(&*store, name)?))
            .map_err(|failure| failure.to_string())
    } else {
        Err("there is no such session".to_owned())
    };
    if let Err(why) = written {
        let why = format!("Could not name session {}: {why}.", named.session);
        notices.write(Notice::error(None, why));
    }
}

/// Names the session in `store` `name`, keeping the rest of its listing.
fn rename(store: &dyn Files, name: String) -> io::Result<()> {
    let meta = Meta::of(store).unwrap_or_else(|| Meta {
        updated: now_ms(),
        ..Meta::default()
    });
    let meta = Meta {
        title: SessionTitle {
            name: Some(name),
            ..meta.title
        },
        ..meta
    };
    write_meta(store, &meta)
}

/// Exits to run the session asked for next.
fn on_switch_session(
    switch: On<SwitchSession>,
    turns: Query<(), With<TurnOf>>,
    paths: Option<Res<SessionPaths>>,
    home: Res<RigHome>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    if !turns.is_empty() {
        notices.write(Notice::info(
            None,
            "A turn is running. Press Esc to stop it first.",
        ));
        return;
    }
    let Some(paths) = paths else {
        return;
    };
    let target = match &switch.session {
        None => None,
        Some(id) => match checked(id, &home.0, &paths.0) {
            Ok(id) => Some(id),
            Err(why) => {
                notices.write(Notice::error(None, why));
                return;
            }
        },
    };
    commands.trigger(Exit(ExitReason::Switch(target)));
}

/// `id` when it names a session with an agent log other than the running
/// one.
fn checked(id: &str, home: &Home, current: &SessionDir) -> Result<SessionId, String> {
    let id: SessionId = id.trim().parse().map_err(|failure| format!("{failure}."))?;
    let dir = home.session(&id);
    if dir.path() == current.path() {
        return Err("That is this session.".to_owned());
    }
    if !dir.is_saved() {
        return Err(format!("No saved session {id}."));
    }
    Ok(id)
}

#[cfg(test)]
mod tests;
