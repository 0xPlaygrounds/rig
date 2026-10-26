//! The native session views, for any front: [`SessionPicker`] lists saved
//! sessions, or the entries of one agent's conversation, in a plain
//! [`Picker`] that [`SessionPickerPlugin`] fills in place as items are
//! read off the main thread, newest first, and previews the highlighted item
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
use std::path::{self, Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_tasks::{AsyncComputeTaskPool, TaskPool};
use rig_core::conversation::{Conversation, MessageId};
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

    /// One line: its name or title, its labels and its listed values.
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
        label
    }

    /// What is shown beside the label: the last two parts of its
    /// directory, as `…/rig-drive/work` (the whole directory when that is
    /// no longer), and how long ago it changed.
    pub fn detail(&self) -> String {
        let ago = ago(self.updated.elapsed().unwrap_or_default());
        match &self.directory {
            Some(directory) => format!("{} · {ago}", compact(directory)),
            None => ago,
        }
    }

    /// What is shown when the user asks for more: its whole
    /// [`label`](Self::label), then its whole directory.
    pub fn more(&self) -> String {
        match &self.directory {
            Some(directory) => format!("{}\n{}", self.label(), tilde(directory)),
            None => self.label(),
        }
    }
}

/// `path` as [`tilde`] shows it, cut to its last two parts after `…/` when
/// that is shorter.
fn compact(path: &Path) -> String {
    let shown = tilde(path);
    let parts = Path::new(&shown)
        .components()
        .filter_map(|part| match part {
            path::Component::Normal(part) => Some(part.to_string_lossy()),
            _ => None,
        });
    let compact = match parts.collect::<Vec<_>>().as_slice() {
        [.., parent, last] => {
            let separator = path::MAIN_SEPARATOR;
            format!("…{separator}{parent}{separator}{last}")
        }
        _ => return shown,
    };
    // `~/rig/work` says more than `…/rig/work`, in as many columns.
    match compact.chars().count() < shown.chars().count() {
        true => compact,
        false => shown,
    }
}

/// `path` with the home directory shown as `~`.
fn tilde(path: &Path) -> String {
    match std::env::home_dir().and_then(|home| path.strip_prefix(home).ok().map(Path::to_owned)) {
        Some(rest) if rest.as_os_str().is_empty() => "~".to_owned(),
        Some(rest) => format!("~{}{}", path::MAIN_SEPARATOR, rest.display()),
        None => path.display().to_string(),
    }
}

/// The native session picker: put it on an entity (it requires a
/// [`Picker`]) and, when [`SessionPickerPlugin`] is loaded, the picker is
/// filled in place as items are read, newest first, and previews the
/// highlighted one when `preview` is set. Picking runs `command` with
/// `{session}`, `{agent}` and `{entry}` replaced; with an empty `command`,
/// read `Picked` (the item's position is in `Chosen`) and [`SessionPicked`]
/// on the same entity. A session's label holds its labels and listed values
/// (rig-harness' `SavedAs::listed`). The items are the `Picker`'s, filled in
/// [`SessionPickerSystems`]: a system after it may change their labels,
/// details and more, but not add, remove or reorder them, as a pick names the
/// session at the item's position. A [`SessionFilter`] matches the text
/// the fill wrote, not a later system's.
#[derive(Component, Clone, Debug)]
#[require(Picker, SessionPicked, SessionView)]
pub struct SessionPicker {
    /// What it lists.
    pub source: SessionSource,
    /// Which of them.
    pub filter: SessionFilter,
    /// The command picking an item runs, its placeholders replaced.
    pub command: String,
    /// Whether the highlighted item is previewed.
    pub preview: bool,
}

/// The saved sessions, no filter, no command, and a preview.
impl Default for SessionPicker {
    fn default() -> Self {
        Self {
            source: SessionSource::Saved,
            filter: SessionFilter::default(),
            command: String::new(),
            preview: true,
        }
    }
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
    /// Only the sessions whose label, detail or more holds this text.
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
    fn passes(&self, session: &SessionEntry, shown: &Shown) -> bool {
        let text = self.text.to_lowercase();
        let holds = |part: &str| part.to_lowercase().contains(&text);
        let (label, detail, more) = shown;
        (self.running || !session.running)
            && (self.project.is_none() || session.directory == self.project)
            && (text.is_empty()
                || holds(label)
                || detail.as_deref().is_some_and(holds)
                || more.as_deref().is_some_and(holds))
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

/// Starts reading for each session picker added, or whose source changed;
/// one whose filter changed is filled again.
fn start(
    mut pickers: Query<(Entity, &SessionPicker, &mut SessionView), Changed<SessionPicker>>,
    sessions: Sessions,
    mut commands: Commands,
) {
    for (entity, picker, mut view) in &mut pickers {
        if view.source.as_ref() == Some(&picker.source) {
            view.set_changed();
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

/// Takes what a task read into its picker's view, to be filled in.
fn read_done(
    done: On<Insert<ReadDone>>,
    mut tasks: Query<(&ReadFor, &mut ReadDone)>,
    mut views: Query<&mut SessionView>,
    mut commands: Commands,
) {
    let Ok((ReadFor(picker), mut read)) = tasks.get_mut(done.entity) else {
        return;
    };
    let read = read.0.take();
    commands.entity(done.entity).despawn();
    let Ok(mut view) = views.get_mut(*picker) else {
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
        None => {}
    }
}

/// An item's label, detail and more.
type Shown = (String, Option<String>, Option<String>);

/// Fills each session picker whose view or filter changed in place with
/// what its view read that it lists, and keeps in the view what each item
/// names.
fn fill(
    mut pickers: Query<
        (&SessionPicker, &mut SessionView, &mut Picker),
        Or<(Changed<SessionView>, Changed<SessionPicker>)>,
    >,
    paths: Option<Res<SessionPaths>>,
) {
    for (picker, mut view, mut items) in &mut pickers {
        fill_one(picker, &mut view, &mut items, paths.as_deref());
    }
}

/// Fills `items` in place with what `view` read that `picker` lists,
/// labelled as it says, and keeps in `view` what each item names.
fn fill_one(
    picker: &SessionPicker,
    view: &mut SessionView,
    items: &mut Mut<Picker>,
    paths: Option<&SessionPaths>,
) {
    let this =
        |session: &SessionEntry| paths.and_then(|paths| paths.id()).as_ref() == Some(&session.id);
    let named = |session: &SessionEntry| (!this(session)).then(|| session.id.clone());
    let dir = |session: &SessionEntry| match paths {
        Some(paths) if this(session) => paths.0.clone(),
        _ => view.home.session(&session.id),
    };
    let mut made: Vec<(PickItem, SessionPicked, SessionDir)> = Vec::new();
    let mut push = |(label, detail, more): Shown, picked: SessionPicked, at: SessionDir| {
        let command = placeholders(&picker.command, &picked, paths);
        made.push((
            PickItem {
                label,
                command,
                detail,
                more,
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
                let shown = (excerpt(&entry), Some(entry.id.to_string()), None);
                push(shown, picked, dir(session));
            }
        }
        _ => {
            for session in &view.sessions {
                let mut session = session.clone();
                session.running |= this(&session);
                let shown = (
                    session.label(),
                    Some(session.detail()),
                    Some(session.more()),
                );
                if !picker.filter.passes(&session, &shown) {
                    continue;
                }
                let picked = SessionPicked {
                    session: named(&session),
                    ..SessionPicked::default()
                };
                push(shown, picked, dir(&session));
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
#[derive(Default)]
pub struct SessionPickerPlugin;

/// Where [`SessionPickerPlugin`] fills session pickers' items, in
/// `Update`: order a system after it to change their labels.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionPickerSystems;

impl Plugin for SessionPickerPlugin {
    fn build(&self, app: &mut App) {
        let fill = fill.in_set(SessionPickerSystems);
        app.add_systems(Update, (start, fill, preview).chain().after(PollTasks))
            .add_observer(read_done)
            .add_observer(preview_done)
            .add_observer(picked);
    }
}

#[cfg(test)]
mod tests;
