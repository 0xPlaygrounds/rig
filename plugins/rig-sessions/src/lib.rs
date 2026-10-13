//! Sessions beyond the one running: `/new`, `/resume` and `/name`, and
//! [`NameSession`] for a plugin naming any session. Each change of a
//! conversation, title or label rewrites the session's [`Meta`] in its
//! store (its [`SessionTitle`], the [`ResumeLabel`]s plugins add, such as
//! rig-telemetry's cost, and when it was updated), which `/resume` lists
//! and a resumed session reads its title back from, as does a session
//! rewritten through rig-memory (`RewriteSession`). With the default
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
use rig_tools::{ago, shorten};
use serde::{Deserialize, Serialize};

use rig_ecs::WriteJournal;
use rig_ecs::agent::{Agent, AgentId, Conversation, Notice, SpawnedBy, TurnOf};
use rig_ecs::journal::{Committed, SessionRewritten, SessionStore};
use rig_ecs::prelude::now_ms;
use rig_harness::commands::{AppCommandsExt, CommandArgs};
use rig_harness::prelude::{
    Exit, ExitReason, Files, RigHome, SessionPaths, SessionStores, first_typed,
};
use rig_harness::primary::primary_order;

#[cfg(feature = "tui")]
mod tui;

/// Most characters of a session's title.
const TITLE_CHARS: usize = 60;
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
            .init_resource::<Listed>()
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
                (
                    reread_title.run_if(on_message::<SessionRewritten>),
                    update_meta.run_if(
                        on_message::<Committed>
                            .or_eager(resource_changed::<SessionTitle>)
                            .or_eager(labels_changed),
                    ),
                )
                    .chain()
                    .after(WriteJournal),
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
/// listing, such as its labels. Naming this session is `/name`'s.
#[derive(Event, Clone, Debug)]
pub struct NameSession {
    /// The session.
    pub session: SessionId,
    /// Its name.
    pub name: String,
}

/// What `/resume` lists of a session, kept in its store: what it is
/// listed as, its labels and when it was updated. The agent logs hold
/// the same facts; a plugin that writes a session, such as an import,
/// writes it with [`write_meta`].
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[serde(default)]
pub struct Meta {
    /// What it is listed as.
    #[serde(flatten)]
    pub title: SessionTitle,
    /// The [`ResumeLabel`]s' texts, by tag, such as rig-telemetry's
    /// `spent`: the session's cost, or its tokens when no call was priced.
    pub labels: BTreeMap<String, String>,
    /// When it was written, in milliseconds since the Unix epoch.
    pub updated: u64,
}

impl Meta {
    /// The listing of the session kept in `store`, if it has one.
    /// Generic, so it is passed as it is where a store is any `Files`,
    /// such as `store.as_deref().and_then(Meta::of)`.
    pub fn of(store: &(impl Files + ?Sized)) -> Option<Self> {
        serde_json::from_slice(&store.read_file(META).ok()?).ok()
    }
}

/// Keeps `meta` as the listing of the session in `store`.
///
/// # Errors
///
/// When the store cannot write it.
pub fn write_meta(store: &(impl Files + ?Sized), meta: &Meta) -> io::Result<()> {
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

    /// How long ago it changed, shown whole beside the label.
    pub fn detail(&self) -> String {
        ago(self.saved.elapsed().unwrap_or_default())
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
            let meta = stores.open(&dir).ok().as_deref().and_then(Meta::of);
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
    sessions
}

/// `path` with the home directory shown as `~`.
fn tilde(path: &Path) -> String {
    match std::env::home_dir().and_then(|home| path.strip_prefix(home).ok().map(Path::to_owned)) {
        Some(rest) if rest.as_os_str().is_empty() => "~".to_owned(),
        Some(rest) => format!("~/{}", rest.display()),
        None => path.display().to_string(),
    }
}

/// What the session's listing cache holds of the title and labels, as
/// last written or read back: a title or label set to what it already
/// says is no change, so resuming a session leaves its place in `/resume`.
#[derive(Resource, Default, PartialEq)]
struct Listed(SessionTitle, BTreeMap<String, String>);

/// Reads the session's title from its listing cache.
fn restore_title(
    store: Option<Res<SessionStore>>,
    mut title: ResMut<SessionTitle>,
    mut listed: ResMut<Listed>,
) {
    let Some(store) = store else {
        return;
    };
    if let Some(meta) = Meta::of(&*store.0) {
        // Not a change: nothing to write back.
        title.bypass_change_detection().clone_from(&meta.title);
        *listed = Listed(meta.title, meta.labels);
    }
}

/// Reads the title back after the session was rewritten
/// ([`SessionRewritten`]): the rewrite mapped the listing cache too, so
/// the title in memory is not written back over it.
fn reread_title(
    store: Option<Res<SessionStore>>,
    mut title: ResMut<SessionTitle>,
    mut listed: ResMut<Listed>,
) {
    if let Some(meta) = store.and_then(|store| Meta::of(&*store.0)) {
        title.set_if_neq(meta.title.clone());
        *listed = Listed(meta.title, meta.labels);
    }
}

/// Whether a [`ResumeLabel`] changed, came or went.
fn labels_changed(
    labels: Query<(), Changed<ResumeLabel>>,
    mut removed: RemovedComponents<ResumeLabel>,
) -> bool {
    removed.read().count() > 0 || !labels.is_empty()
}

/// Rewrites the session's listing cache when a conversation changed, and
/// on a new name or label, once some agent logged a message; the first
/// time, the title takes the start of the first message typed.
fn update_meta(
    (paths, store): (Option<Res<SessionPaths>>, Option<Res<SessionStore>>),
    (mut title, mut listed): (ResMut<SessionTitle>, ResMut<Listed>),
    mut committed: MessageReader<Committed>,
    agents: Query<(&AgentId, &Conversation, Has<SpawnedBy>), With<Agent>>,
    labels: Query<&ResumeLabel>,
) {
    let changed = committed.read().count() > 0;
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
    let now = Listed(title.clone(), labels);
    if !changed && *listed == now {
        return;
    }
    let meta = Meta {
        title: now.0.clone(),
        labels: now.1.clone(),
        updated: now_ms(),
    };
    match write_meta(&*store.0, &meta) {
        Ok(()) => *listed = now,
        Err(failure) => error!("could not write the session's meta.json: {failure}"),
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
