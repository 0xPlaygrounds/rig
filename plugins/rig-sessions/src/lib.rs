//! Sessions beyond the one running: `/new`, `/resume` and `/name`, and
//! [`NameSession`] for a plugin naming any session. Each change of a
//! conversation, title or label rewrites the session's listing in its
//! store (rig-memory's [`SessionMeta`]: the title of its [`SessionTitle`],
//! the [`ResumeLabel`]s plugins add, such as rig-telemetry's cost, and when
//! it was updated), which `/resume` lists and a resumed session reads its
//! title back from, as does a session rewritten through rig-memory
//! (`RewriteSession`). With the default
//! `tui` feature, `/resume` without arguments is a picker and the terminal
//! view's status line shows the session's name. Running another
//! session is the launcher's job, so the agent stays one session per
//! process: it names the next session in [`SessionDir::switch`] and exits
//! with the reload code, as `/reload` does, and the launcher starts it in
//! its own directory.

use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::error;
use bevy_reflect::prelude::*;
use rig_harness::harness_protocol::{Home, SessionDir, SessionId};
use rig_harness::rig_memory::{SessionMeta, title_of};
use rig_tools::ago;

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
                "Pick an earlier session to resume, or /resume <id or name>",
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
            )
            .add_systems(
                Last,
                write_name
                    .after(update_meta)
                    .run_if(resource_changed::<SessionTitle>),
            );
        #[cfg(feature = "tui")]
        tui::add(app);
    }
}

/// What the session is listed as: the name `/name` gave it, else the start
/// of the first message the user typed, kept from the first turn on.
#[derive(Resource, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Resource, Clone, Debug, Default, PartialEq)]
pub struct SessionTitle {
    /// The name set with `/name`, kept in the session's directory
    /// ([`SessionDir::name`]), where `rig --resume <name>` finds it.
    pub name: Option<String>,
    /// The start of the first message typed ([`SessionMeta::title`]);
    /// empty until a turn ended.
    pub typed: String,
}

impl SessionTitle {
    /// What the saved session in `dir`, kept in `store`, is listed as: its
    /// name ([`SessionDir::name`]) and the title of its listing
    /// ([`SessionMeta`]); `None` when it has neither. Generic, as
    /// [`SessionMeta::read`] is.
    pub fn read(dir: &SessionDir, store: &(impl Files + ?Sized)) -> Option<Self> {
        let name = dir.name();
        let meta = SessionMeta::read(store);
        (name.is_some() || meta.is_some()).then(|| Self {
            name,
            typed: meta.map(|meta| meta.title).unwrap_or_default(),
        })
    }

    /// The name, else the start of the first message typed (empty when
    /// neither is known).
    pub fn shown(&self) -> &str {
        self.name.as_deref().unwrap_or(&self.typed)
    }
}

/// A part of this session's line in `/resume`, such as its goal: spawn
/// one and keep its `text` current. It is written with the session's
/// listing cache ([`SessionMeta::labels`]), so `/resume` shows it for sessions
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
    /// The id or name of the session to resume ([`Home::find`]), or
    /// `None` for a new session.
    pub session: Option<String>,
}

/// Names a session ([`SessionDir::set_name`]): its title in `/resume`,
/// and nothing of its listing, such as its labels. Naming this session is
/// `/name`'s.
#[derive(Event, Clone, Debug)]
pub struct NameSession {
    /// The session.
    pub session: SessionId,
    /// Its name.
    pub name: String,
}

/// An earlier session, as `/resume` lists it.
#[derive(Clone, Debug)]
pub struct SessionEntry {
    /// Its id.
    pub id: SessionId,
    /// The name `/name` gave it.
    pub name: Option<String>,
    /// Its listing cache; empty when no turn of it ended.
    pub meta: SessionMeta,
    /// The directory it runs in, if known.
    pub directory: Option<PathBuf>,
    /// When it last changed.
    pub saved: SystemTime,
}

impl SessionEntry {
    /// One line: name or title, the [`ResumeLabel`]s, directory.
    pub fn label(&self) -> String {
        let mut label = match (&self.name, self.meta.title.as_str()) {
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
            let store = stores.open(&dir).ok();
            let meta = store.as_deref().and_then(SessionMeta::read);
            let saved = match &meta {
                Some(meta) => UNIX_EPOCH + Duration::from_millis(meta.updated),
                None => fs::metadata(dir.path()).ok()?.modified().ok()?,
            };
            Some(SessionEntry {
                directory: dir.working_directory(),
                name: dir.name(),
                meta: meta.unwrap_or_default(),
                id,
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

/// What the session in `paths`, kept in `store`, is listed as, and its
/// listing's labels.
fn listed(paths: &SessionPaths, store: &SessionStore) -> Option<Listed> {
    let title = SessionTitle::read(paths, &*store.0)?;
    let labels = SessionMeta::read(&*store.0).unwrap_or_default().labels;
    Some(Listed(title, labels))
}

/// Reads the session's name from its directory and its title from its
/// listing cache.
fn restore_title(
    (paths, store): (Option<Res<SessionPaths>>, Option<Res<SessionStore>>),
    mut title: ResMut<SessionTitle>,
    mut listed_now: ResMut<Listed>,
) {
    let (Some(paths), Some(store)) = (paths, store) else {
        return;
    };
    if let Some(listed) = listed(&paths, &store) {
        // Not a change: nothing to write back.
        *title.bypass_change_detection() = listed.0.clone();
        *listed_now = listed;
    }
}

/// Reads the name and title back after the session was rewritten
/// ([`SessionRewritten`]): the rewrite mapped the listing cache and the
/// name file too, so the title in memory is not written back over them.
fn reread_title(
    (paths, store): (Option<Res<SessionPaths>>, Option<Res<SessionStore>>),
    mut title: ResMut<SessionTitle>,
    mut listed_now: ResMut<Listed>,
) {
    let (Some(paths), Some(store)) = (paths, store) else {
        return;
    };
    if let Some(listed) = listed(&paths, &store) {
        title.set_if_neq(listed.0.clone());
        *listed_now = listed;
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
    let meta = SessionMeta {
        title: now.0.typed.clone(),
        labels: now.1.clone(),
        updated: now_ms(),
        ..SessionMeta::default()
    };
    match meta.write(&*store.0) {
        Ok(()) => *listed = now,
        Err(failure) => error!("could not write the session's meta.json: {failure}"),
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

/// `/resume <id or name>` runs that session; without arguments, the
/// terminal view lets the user pick one.
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

/// Keeps the session's name in its directory ([`SessionDir::set_name`]),
/// where `rig --resume <name>` and `/resume <name>` find it, as its
/// [`SessionTitle`] has it.
fn write_name(paths: Option<Res<SessionPaths>>, title: Res<SessionTitle>) {
    let Some(paths) = paths else {
        return;
    };
    if paths.name() != title.name
        && let Err(failure) = paths.set_name(title.name.as_deref())
    {
        error!("could not write the session's name: {failure}");
    }
}

/// Names the session asked for: this one's [`SessionTitle`], another's
/// name file ([`SessionDir::set_name`]).
fn name_session(
    named: On<NameSession>,
    paths: Option<Res<SessionPaths>>,
    home: Res<RigHome>,
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
        dir.set_name(Some(&name))
            .map_err(|failure| failure.to_string())
    } else {
        Err("there is no such session".to_owned())
    };
    if let Err(why) = written {
        let why = format!("Could not name session {}: {why}.", named.session);
        notices.write(Notice::error(None, why));
    }
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

/// The saved session other than the running one that `name_or_id` names.
fn checked(name_or_id: &str, home: &Home, current: &SessionDir) -> Result<SessionId, String> {
    let id = home.find(name_or_id).map_err(|why| format!("{why}."))?;
    if home.session(&id).path() == current.path() {
        return Err("That is this session.".to_owned());
    }
    Ok(id)
}

#[cfg(test)]
mod tests;
