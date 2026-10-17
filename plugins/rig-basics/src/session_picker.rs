//! The native session views, for any front: [`SessionPicker`] lists saved
//! sessions, or the entries of one agent's conversation, in a plain
//! [`Picker`] the harness fills in place as items are read off the
//! main thread, newest first, and previews the highlighted item
//! ([`PickPreview`]). A plugin opens one with its
//! own source, filter, labels and command:
//!
//! ```no_run
//! use rig_basics::session_picker::{SessionFilter, SessionPicker};
//! use rig_harness::prelude::*;
//!
//! fn delete_which(agent: Entity, mut commands: Commands) {
//!     commands.spawn((
//!         Picker::new("Delete which session?").for_agent(agent),
//!         SessionPicker {
//!             command: "delete {session}".into(),
//!             filter: SessionFilter { running: true, ..Default::default() },
//!             ..Default::default()
//!         },
//!     ));
//! }
//! ```

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_tasks::{AsyncComputeTaskPool, TaskPool};
use rig_core::conversation::{Conversation, EntryRef, MessageId};
use rig_ecs::prelude::{PollTasks, Running, Wake};
use rig_ecs::session_log::SessionStore;
use rig_harness::harness_protocol::{Home, SessionDir, SessionId};
use rig_memory::{SessionMeta, Store, excerpt, read_conversation, title_of};
use rig_tools::ago;

use rig_harness::host::session::{SessionPaths, StoreOpener};
use rig_harness::pick::{Highlighted, Pick, PickItem, PickPreview, Picker};
use rig_harness::services::RigHome;

/// How many sessions one task reads.
const BATCH: usize = 16;

/// A saved session as a listing shows it.
#[derive(Clone, Debug)]
pub struct SessionEntry {
    /// Its id.
    pub id: SessionId,
    /// The name the user gave it.
    pub name: Option<String>,
    /// What its store says about it; empty when it has no listing.
    pub meta: SessionMeta,
    /// The directory it runs in, if known.
    pub directory: Option<PathBuf>,
    /// When it last changed.
    pub updated: SystemTime,
    /// Whether a rig runs it now, this one or another.
    pub running: bool,
}

impl SessionEntry {
    /// The saved session in `dir`, read from its store as `stores` opens it
    /// and from its directory's files; `None` when it is not a saved
    /// session. Its logs are not read.
    pub fn read(dir: &SessionDir, stores: &StoreOpener) -> Option<Self> {
        let id = dir.id().filter(|_| dir.is_saved())?;
        let store = stores.open(dir).ok();
        Some(Self::of(id, dir, store.as_deref()))
    }

    /// The session `id` in `dir`, its listing read from `store`.
    fn of(id: SessionId, dir: &SessionDir, store: Option<&dyn Store>) -> Self {
        let meta = store.and_then(SessionMeta::read).unwrap_or_default();
        let updated = match meta.updated {
            0 => fs::metadata(dir.path()).and_then(|meta| meta.modified()),
            ms => Ok(UNIX_EPOCH + Duration::from_millis(ms)),
        };
        Self {
            name: dir.name(),
            directory: dir.working_directory(),
            updated: updated.unwrap_or(UNIX_EPOCH),
            running: dir.runs_elsewhere(),
            meta,
            id,
        }
    }

    /// One line: its name or title, its labels, its listed values and its
    /// directory.
    pub fn label(&self) -> String {
        let meta = &self.meta;
        let mut label = match (&self.name, meta.title.as_str()) {
            (Some(name), _) => name.clone(),
            (None, "") => format!("session {}", self.id),
            (None, title) => title.to_owned(),
        };
        let listed = meta.listed.values().map(|saved| match saved {
            serde_json::Value::String(text) => title_of(text),
            value => title_of(&value.to_string()),
        });
        let shown = meta.labels.values().cloned().chain(listed);
        for text in shown.filter(|text| !text.is_empty()) {
            label.push_str(&format!("  · {text}"));
        }
        if let Some(directory) = &self.directory {
            label.push_str(&format!("  · {}", tilde(directory)));
        }
        label
    }

    /// What is shown whole beside the label: how long ago it changed.
    pub fn detail(&self) -> String {
        ago(self.updated.elapsed().unwrap_or_default())
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

/// The native session picker: put it on an entity (it requires a
/// [`Picker`]) and the harness fills the picker in place as items are
/// read, newest first, and previews the highlighted one when `preview` is
/// set. Picking runs `command` with `{session}`, `{agent}` and `{entry}`
/// replaced; with an empty `command`, read `Picked` (the item's
/// position is in `Chosen`) and [`SessionPicked`] on the same entity.
#[derive(Component, Clone, Debug)]
#[require(Picker, SessionPicked, SessionView)]
pub struct SessionPicker {
    /// What it lists.
    pub source: SessionSource,
    /// Which of them.
    pub filter: SessionFilter,
    /// The command picking an item runs, its placeholders replaced.
    pub command: String,
    /// The item's label and detail; `None` for the default labels.
    pub show: Option<fn(&SessionItem<'_>) -> (String, Option<String>)>,
    /// Whether the highlighted item is previewed.
    pub preview: bool,
}

/// The saved sessions, no filter, no command, the default labels, and a
/// preview.
impl Default for SessionPicker {
    fn default() -> Self {
        Self {
            source: SessionSource::Saved,
            filter: SessionFilter::default(),
            command: String::new(),
            show: None,
            preview: true,
        }
    }
}

/// One item of a session picker, whichever its source: what
/// [`SessionPicker::show`] labels.
#[derive(Clone, Copy, Debug)]
pub enum SessionItem<'a> {
    /// A saved session.
    Session(&'a SessionEntry),
    /// An entry of an agent's conversation.
    Entry {
        /// The session.
        session: &'a SessionEntry,
        /// The entry's position in the conversation.
        at: usize,
        /// The entry.
        entry: EntryRef<'a>,
    },
}

/// What a [`SessionPicker`] lists.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SessionSource {
    /// The saved sessions of this home.
    Saved,
    /// These sessions.
    Sessions(Vec<SessionId>),
    /// The entries of an agent's conversation, as one-line excerpts: pick
    /// a turn, a message.
    Entries {
        /// The session; this one with `None`.
        session: Option<SessionId>,
        /// The agent; the main one with `None`.
        agent: Option<String>,
    },
}

/// Which sessions a [`SessionPicker`] lists.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SessionFilter {
    /// Whether sessions a rig runs, this one included, are listed too.
    pub running: bool,
    /// Only the sessions that ran in this directory.
    pub project: Option<PathBuf>,
    /// Only the sessions whose label holds this text.
    pub text: String,
}

/// What was picked from a [`SessionPicker`], as data, put on its entity
/// before `Picked` is triggered: the session, agent and entry it names.
#[derive(Component, Clone, Debug, Default, PartialEq, Eq)]
pub struct SessionPicked {
    /// The session; `None` for this one.
    pub session: Option<SessionId>,
    /// The agent, for an entry.
    pub agent: Option<String>,
    /// The entry.
    pub entry: Option<MessageId>,
}

impl SessionFilter {
    /// Whether `session`, shown as `shown`, passes.
    fn passes(&self, session: &SessionEntry, shown: &(String, Option<String>)) -> bool {
        let text = self.text.to_lowercase();
        let holds = |part: &str| part.to_lowercase().contains(&text);
        (self.running || !session.running)
            && (self.project.is_none() || session.directory == self.project)
            && (text.is_empty() || holds(&shown.0) || shown.1.as_deref().is_some_and(holds))
    }
}

/// The default labels of `item`.
fn default_show(item: &SessionItem<'_>) -> (String, Option<String>) {
    match item {
        SessionItem::Session(session) => (session.label(), Some(session.detail())),
        SessionItem::Entry { entry, .. } => (excerpt(entry), Some(entry.id.to_string())),
    }
}

/// What a [`SessionPicker`] read so far, on its entity, and what each of
/// its picker's items names.
#[derive(Component)]
struct SessionView {
    /// Where the sessions it reads are.
    home: Home,
    /// The source it read, to start again when the picker's changes.
    source: Option<SessionSource>,
    /// The sessions read, newest first (one for entries).
    sessions: Vec<SessionEntry>,
    /// The agent whose entries are listed.
    agent: String,
    /// The conversation whose entries are listed.
    conversation: Option<Conversation>,
    /// Why reading failed.
    failed: Option<String>,
    /// What each item of the picker names, and the session it is in.
    shown: Vec<(SessionPicked, SessionDir)>,
    /// The previews read, by what they preview.
    previews: Vec<(SessionPicked, PickPreview)>,
    /// What is being previewed, and the task reading it.
    previewing: Option<(SessionPicked, Entity)>,
}

impl Default for SessionView {
    fn default() -> Self {
        Self {
            home: Home::at(PathBuf::new()),
            source: None,
            sessions: Vec::new(),
            agent: String::new(),
            conversation: None,
            failed: None,
            shown: Vec::new(),
            previews: Vec::new(),
            previewing: None,
        }
    }
}

/// What one task read for a [`SessionPicker`].
enum Read {
    /// Some saved sessions.
    Sessions(Vec<SessionEntry>),
    /// A session, its agent and the agent's conversation.
    Entries(SessionEntry, String, Conversation),
}

/// A task reading for the session picker it names.
#[derive(Component)]
#[relationship(relationship_target = Reading)]
struct ReadFor(Entity);

/// The tasks reading for a session picker: dropped with it.
#[derive(Component, Default)]
#[relationship_target(relationship = ReadFor, linked_spawn)]
struct Reading(Vec<Entity>);

/// What a task of [`Reading`] read.
#[derive(Component)]
struct ReadDone(Option<Result<Read, String>>);

impl From<Result<Read, String>> for ReadDone {
    fn from(read: Result<Read, String>) -> Self {
        Self(Some(read))
    }
}

/// What a preview task of [`Reading`] read: what it previews, and the
/// preview.
#[derive(Component)]
struct PreviewDone(SessionPicked, PickPreview);

impl From<(SessionPicked, PickPreview)> for PreviewDone {
    fn from((picked, preview): (SessionPicked, PickPreview)) -> Self {
        Self(picked, preview)
    }
}

/// Opens session stores off the main thread: the running session's as it
/// runs, any other as [`StoreOpener`] opens it.
#[derive(Clone)]
struct Opener {
    stores: StoreOpener,
    home: Home,
    running: Option<(SessionDir, Arc<dyn Store>)>,
}

impl Opener {
    /// The store of the session in `dir`.
    fn open(&self, dir: &SessionDir) -> Result<Arc<dyn Store>, String> {
        match &self.running {
            Some((running, store)) if running.path() == dir.path() => Ok(store.clone()),
            _ => self.stores.open(dir).map_err(|failure| failure.to_string()),
        }
    }

    /// The session `session`, this one with `None`, its directory and its
    /// store.
    fn session(
        &self,
        session: Option<&SessionId>,
    ) -> Result<(SessionEntry, Arc<dyn Store>), String> {
        let dir = match session {
            Some(id) => self.home.session(id),
            None => self
                .running
                .as_ref()
                .map(|(dir, _)| dir.clone())
                .ok_or("there is no running session")?,
        };
        let id = dir.id().ok_or("the session has no id")?;
        let store = self.open(&dir)?;
        Ok((SessionEntry::of(id, &dir, Some(&*store)), store))
    }
}

/// What the systems of the session picker read of the app.
#[derive(bevy_ecs::system::SystemParam)]
struct Sessions<'w> {
    home: Option<Res<'w, RigHome>>,
    stores: Option<Res<'w, StoreOpener>>,
    paths: Option<Res<'w, SessionPaths>>,
    store: Option<Res<'w, SessionStore>>,
    wake: Option<Res<'w, Wake>>,
}

impl Sessions<'_> {
    fn opener(&self) -> Opener {
        let running = self.paths.as_ref().map(|paths| {
            let store = self.store.as_ref().map(|store| store.0.clone());
            (paths.0.clone(), store)
        });
        let stores = self.stores.as_deref().cloned().unwrap_or_default();
        let running = running.and_then(|(dir, store)| {
            let store = store.or_else(|| stores.open(&dir).ok())?;
            Some((dir, store))
        });
        Opener {
            home: self
                .home
                .as_ref()
                .map_or_else(Home::from_env, |home| home.0.clone()),
            stores,
            running,
        }
    }

    /// Runs `work` for the picker `picker`, ending as `C` on its entity.
    fn spawn<C: Component + From<T>, T: Send + 'static>(
        &self,
        picker: Entity,
        commands: &mut Commands,
        work: impl FnOnce() -> T + Send + 'static,
    ) -> Entity {
        let pool = AsyncComputeTaskPool::get_or_init(TaskPool::default);
        let wake = self
            .wake
            .as_deref()
            .cloned()
            .unwrap_or_else(|| Wake::new(|| {}));
        let running = Running::spawn_into::<C, T>(pool, &wake, async move { work() });
        commands
            .spawn((Name::new("session picker"), ReadFor(picker), running))
            .id()
    }
}

/// Starts reading for each session picker added, or whose source changed,
/// and fills the picker again when its filter or labels did.
fn start(
    mut pickers: Query<
        (Entity, &SessionPicker, &mut SessionView, &mut Picker),
        Changed<SessionPicker>,
    >,
    sessions: Sessions,
    mut commands: Commands,
) {
    for (entity, picker, mut view, mut items) in &mut pickers {
        if view.source.as_ref() == Some(&picker.source) {
            fill(picker, &mut view, &mut items, sessions.paths.as_deref());
            continue;
        }
        commands.entity(entity).despawn_related::<Reading>();
        let opener = sessions.opener();
        *view = SessionView {
            home: opener.home.clone(),
            source: Some(picker.source.clone()),
            ..SessionView::default()
        };
        match &picker.source {
            SessionSource::Saved | SessionSource::Sessions(_) => {
                let dirs: Vec<SessionDir> = match &picker.source {
                    SessionSource::Sessions(ids) => {
                        let dirs = ids.iter().map(|id| opener.home.session(id));
                        dirs.filter(SessionDir::is_saved).collect()
                    }
                    _ => opener
                        .home
                        .saved_sessions()
                        .into_iter()
                        .map(|(_, dir)| dir)
                        .collect(),
                };
                for batch in dirs.chunks(BATCH) {
                    let (batch, opener) = (batch.to_vec(), opener.clone());
                    sessions.spawn::<ReadDone, _>(entity, &mut commands, move || {
                        let read = batch.iter().filter_map(|dir| {
                            let store = opener.open(dir).ok();
                            Some(SessionEntry::of(dir.id()?, dir, store.as_deref()))
                        });
                        Ok(Read::Sessions(read.collect()))
                    });
                }
            }
            SessionSource::Entries { session, agent } => {
                let (session, agent) = (session.clone(), agent.clone());
                sessions.spawn::<ReadDone, _>(entity, &mut commands, move || {
                    let (entry, store) = opener.session(session.as_ref())?;
                    let (agent, conversation) = read_conversation(&*store, agent.as_deref())
                        .map_err(|why| why.to_string())?;
                    Ok(Read::Entries(entry, agent, conversation))
                });
            }
        }
    }
}

/// Takes what a task read into its picker's view and fills the picker.
fn read_done(
    done: On<Insert<ReadDone>>,
    mut tasks: Query<(&ReadFor, &mut ReadDone)>,
    mut pickers: Query<(&SessionPicker, &mut SessionView, &mut Picker)>,
    paths: Option<Res<SessionPaths>>,
    mut commands: Commands,
) {
    let Ok((ReadFor(picker), mut read)) = tasks.get_mut(done.entity) else {
        return;
    };
    let read = read.0.take();
    commands.entity(done.entity).despawn();
    let Ok((picker, mut view, mut items)) = pickers.get_mut(*picker) else {
        return;
    };
    match read {
        Some(Ok(Read::Sessions(read))) => {
            view.sessions.extend(read);
            view.sessions
                .sort_by_key(|session| std::cmp::Reverse(session.updated));
        }
        Some(Ok(Read::Entries(session, agent, conversation))) => {
            view.sessions = vec![session];
            view.agent = agent;
            view.conversation = Some(conversation);
        }
        Some(Err(why)) => view.failed = Some(why),
        None => return,
    }
    fill(picker, &mut view, &mut items, paths.as_deref());
}

/// Fills `items` in place with what `view` read that `picker` lists,
/// labelled as it says, and keeps in `view` what each item names.
fn fill(
    picker: &SessionPicker,
    view: &mut SessionView,
    items: &mut Mut<Picker>,
    paths: Option<&SessionPaths>,
) {
    let show = picker.show.unwrap_or(default_show);
    let this =
        |session: &SessionEntry| paths.and_then(|paths| paths.id()).as_ref() == Some(&session.id);
    let named = |session: &SessionEntry| (!this(session)).then(|| session.id.clone());
    let dir = |session: &SessionEntry| match paths {
        Some(paths) if this(session) => paths.0.clone(),
        _ => view.home.session(&session.id),
    };
    let mut made: Vec<(PickItem, SessionPicked, SessionDir)> = Vec::new();
    let mut push = |item: &SessionItem<'_>, picked: SessionPicked, at: SessionDir| {
        let (label, detail) = show(item);
        let command = placeholders(&picker.command, &picked, paths);
        made.push((
            PickItem {
                label,
                command,
                detail,
                checked: false,
            },
            picked,
            at,
        ));
    };
    let session = view.sessions.first();
    match (&view.conversation, session) {
        (Some(conversation), Some(session)) => {
            for entry in conversation.entries() {
                let picked = SessionPicked {
                    session: named(session),
                    agent: Some(view.agent.clone()),
                    entry: Some(entry.id),
                };
                let item = SessionItem::Entry {
                    session,
                    at: entry.index,
                    entry,
                };
                push(&item, picked, dir(session));
            }
        }
        _ => {
            for session in &view.sessions {
                let mut session = session.clone();
                session.running |= this(&session);
                let item = SessionItem::Session(&session);
                if !picker.filter.passes(&session, &show(&item)) {
                    continue;
                }
                let picked = SessionPicked {
                    session: named(&session),
                    ..SessionPicked::default()
                };
                push(&item, picked, dir(&session));
            }
        }
    }
    let new: Vec<PickItem> = made.iter().map(|(item, ..)| item.clone()).collect();
    if items.items != new {
        items.items = new;
    }
    view.shown = made
        .into_iter()
        .map(|(_, picked, at)| (picked, at))
        .collect();
    if let Some(why) = &view.failed {
        let body = format!("Could not read it: {why}");
        if items.body != body {
            items.body = body;
        }
    }
}

/// `command` with `{session}`, `{agent}` and `{entry}` replaced by what
/// `picked` names (empty where it names none); empty for an empty
/// command.
fn placeholders(command: &str, picked: &SessionPicked, paths: Option<&SessionPaths>) -> String {
    if command.is_empty() {
        return String::new();
    }
    let session = picked
        .session
        .clone()
        .or_else(|| paths.and_then(|paths| paths.id()));
    command
        .replace(
            "{session}",
            &session.map(|id| id.to_string()).unwrap_or_default(),
        )
        .replace("{agent}", picked.agent.as_deref().unwrap_or_default())
        .replace(
            "{entry}",
            &picked.entry.map(|id| id.to_string()).unwrap_or_default(),
        )
}

/// Previews the highlighted item of each session picker that previews:
/// from what was read already, else from a task reading its agent's
/// conversation.
fn preview(
    mut pickers: Query<
        (
            Entity,
            &SessionPicker,
            &Highlighted,
            &mut SessionView,
            Option<&PickPreview>,
        ),
        Or<(Changed<Highlighted>, Changed<SessionView>)>,
    >,
    sessions: Sessions,
    mut commands: Commands,
) {
    for (entity, picker, highlighted, mut view, shown) in &mut pickers {
        let Some((picked, dir)) = view
            .shown
            .get(highlighted.0)
            .cloned()
            .filter(|_| picker.preview)
        else {
            continue;
        };
        let ready = match &view.conversation {
            Some(conversation) => Some(PickPreview::Conversation {
                conversation: conversation.clone(),
                at: picked.entry,
            }),
            None => view
                .previews
                .iter()
                .find(|(previewed, _)| *previewed == picked)
                .map(|(_, preview)| preview.clone()),
        };
        if let Some(ready) = ready {
            if shown != Some(&ready) {
                commands.entity(entity).insert(ready);
            }
            continue;
        }
        if view
            .previewing
            .as_ref()
            .is_some_and(|(previewing, _)| *previewing == picked)
        {
            continue;
        }
        if let Some((_, task)) = view.previewing.take() {
            commands.entity(task).try_despawn();
        }
        let opener = sessions.opener();
        let asked = picked.clone();
        let task = sessions.spawn::<PreviewDone, _>(entity, &mut commands, move || {
            let read = opener.open(&dir).and_then(|store| {
                read_conversation(&*store, asked.agent.as_deref()).map_err(|why| why.to_string())
            });
            let preview = match read {
                Ok((_, conversation)) => PickPreview::Conversation {
                    conversation,
                    at: asked.entry,
                },
                Err(why) => PickPreview::Text(format!("Cannot preview it: {why}")),
            };
            (asked, preview)
        });
        view.bypass_change_detection().previewing = Some((picked, task));
    }
}

/// Keeps a preview read, and shows it while its item is highlighted.
fn preview_done(
    done: On<Insert<PreviewDone>>,
    tasks: Query<(&ReadFor, &PreviewDone)>,
    mut pickers: Query<(&mut SessionView, &Highlighted)>,
    mut commands: Commands,
) {
    let Ok((ReadFor(picker), PreviewDone(picked, preview))) = tasks.get(done.entity) else {
        return;
    };
    commands.entity(done.entity).despawn();
    let Ok((mut view, highlighted)) = pickers.get_mut(*picker) else {
        return;
    };
    let view = view.bypass_change_detection();
    if view
        .previewing
        .as_ref()
        .is_some_and(|(previewing, _)| previewing == picked)
    {
        view.previewing = None;
    }
    view.previews.push((picked.clone(), preview.clone()));
    if view
        .shown
        .get(highlighted.0)
        .is_some_and(|(shown, _)| shown == picked)
    {
        commands.entity(*picker).insert(preview.clone());
    }
}

/// Puts what the item picked from a session picker names on it, before
/// the harness triggers `Picked`.
fn picked(pick: On<Pick>, mut pickers: Query<(&SessionView, &mut SessionPicked)>) {
    let Ok((view, mut picked)) = pickers.get_mut(pick.entity) else {
        return;
    };
    let first = pick
        .chosen
        .as_ref()
        .and_then(|chosen| chosen.items.iter().min());
    if let Some((named, _)) = first.and_then(|at| view.shown.get(*at)) {
        *picked = named.clone();
    }
}

/// Fills and previews every [`SessionPicker`]. A plugin that shows one
/// adds it, unless `app.is_plugin_added::<SessionPickerPlugin>()`.
pub struct SessionPickerPlugin;

impl Plugin for SessionPickerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (start, preview).chain().after(PollTasks))
            .add_observer(read_done)
            .add_observer(preview_done)
            .add_observer(picked);
    }
}

#[cfg(test)]
mod tests;
