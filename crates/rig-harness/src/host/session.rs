//! Where this process keeps its session: the directory the launcher names,
//! the store of the session's files, opened by [`SessionStores`], which a
//! plugin replaces to keep them elsewhere or sealed, the plain files of
//! [`Scratch`] with the text log, panics routed to that log instead of
//! the terminal, and the log's warnings and errors as [`Logged`] data. A
//! plugin copies the session into a new one with [`ForkSession`] and
//! removes another with [`DeleteSession`]; [`SessionForked`],
//! [`SessionDeleted`] and [`SessionDeleteFailed`] tell every plugin, such as one keeping files of its
//! own per session. Deleting is crash safe: a session marked deleting is
//! hidden from listings, and the next start finishes it. A forked session
//! names the session it came from ([`ForkedFrom`]). A rewrite of this
//! session (`RewriteSession`) reaches its scratch files too, and
//! [`rewrite_stopped`] rewrites a session no rig runs, scratch included.
//! The first app a
//! session runs, a fork's first included, has [`FirstStart`]; what the
//! user typed in a message is [`typed`].

use std::fmt::{Debug, Write};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Seek, SeekFrom};
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use bevy_app::OnAppExitSystems;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::tracing::field::{Field, Visit};
use bevy_log::tracing::{Event, Level};
use bevy_log::tracing_subscriber::layer::{Context, Layer};
use bevy_log::tracing_subscriber::{Registry, fmt};
use bevy_log::{BoxedFmtLayer, BoxedLayer, error, warn};
use bevy_reflect::Reflect;
use crossbeam_channel::{Receiver, Sender};
use rig::harness_protocol::{Home, SessionDir, SessionId};

use rig_core::completion::Message;
use rig_core::conversation::{Author, Conversation, MessageId};
use rig_core::message::UserContent;
use rig_ecs::WriteJournal;
use rig_ecs::agent::{self, AgentId, Notice};
use rig_ecs::journal::{SessionRewritten, SessionStore, open_session};
use rig_memory::{DirStore, Files, Keep, LoadError, Rewrite, Rewritten, Store, fork, rewrite};
use rig_tools::Spill;
use rig_tools::attach::attached_file;
use serde::{Deserialize, Serialize};

use super::cli::Cli;
use super::launcher::{LauncherRun, refuse};
use crate::describe::Describing;
use crate::plugin_data::PluginData;
use crate::services::{ProjectRoot, RigHome};
use crate::timings::{self, SystemTimings, TimeSystems};
use crate::workspace::ProjectWorkspace;

/// The session's directory. Its files are kept through the session's
/// store ([`SessionStores`]): a plugin keeps its own files of the session
/// there too, under a directory of its own, such as
/// `store.write_file("notes/today.md", ..)`.
#[derive(Resource, Clone, Debug)]
pub struct SessionPaths(pub SessionDir);

impl Deref for SessionPaths {
    type Target = SessionDir;

    fn deref(&self) -> &SessionDir {
        &self.0
    }
}

/// Opens the store of a session's files: the running session's, a fork's,
/// an import's, one `/resume` lists. A plugin replaces it while the app is
/// built, such as with a store that seals what it keeps:
/// `app.insert_resource(SessionStores::new(|dir| ..))`. Once every plugin
/// is built, the harness opens the running session's [`SessionStore`] from
/// it, unless one is there, and reads the session before anything is
/// written; a run whose store does not open, or whose session cannot be
/// read, is refused, and nothing of the session is changed. Every store it
/// opens marks its session saved ([`SessionDir::mark_saved`]) the first
/// time a log is written to it, so `/resume` lists the session. By default
/// a session is kept in its directory ([`DirStore`]: `logs/`, `blobs/` and
/// `files/`), which refuses a session an older rig saved.
#[derive(Resource, Clone)]
pub struct SessionStores(Arc<OpenStore>);

/// How [`SessionStores`] opens a store.
type OpenStore = dyn Fn(&SessionDir) -> Result<Arc<dyn Store>, LoadError> + Send + Sync;

impl SessionStores {
    /// Stores that `open` opens.
    pub fn new(
        open: impl Fn(&SessionDir) -> Result<Arc<dyn Store>, LoadError> + Send + Sync + 'static,
    ) -> Self {
        Self(Arc::new(open))
    }

    /// The store of the session in `dir`, which marks the session saved
    /// once a log is written to it.
    ///
    /// # Errors
    ///
    /// When the store cannot be opened, such as without its key, or holds
    /// a session an older rig saved.
    pub fn open(&self, dir: &SessionDir) -> Result<Arc<dyn Store>, LoadError> {
        Ok(Arc::new(Marking {
            inner: (self.0)(dir)?,
            dir: dir.clone(),
            marked: AtomicBool::new(false),
        }))
    }
}

impl Default for SessionStores {
    fn default() -> Self {
        Self::new(|dir| Ok(Arc::new(DirStore::open(dir.path())?)))
    }
}

/// A store that marks its session saved on the first write to a log, the
/// session's own log of saved values included.
struct Marking {
    inner: Arc<dyn Store>,
    dir: SessionDir,
    marked: AtomicBool,
}

impl Store for Marking {
    fn logs(&self) -> io::Result<Vec<String>> {
        self.inner.logs()
    }

    fn read_log(&self, name: &str) -> io::Result<Vec<u8>> {
        self.inner.read_log(name)
    }

    fn append_log(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        self.inner.append_log(name, bytes)?;
        if !self.marked.load(Ordering::Relaxed) {
            match self.dir.mark_saved() {
                Ok(()) => self.marked.store(true, Ordering::Relaxed),
                Err(failure) => warn!("could not mark the session saved: {failure}"),
            }
        }
        Ok(())
    }

    fn replace_log(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        self.inner.replace_log(name, bytes)
    }

    fn remove_log(&self, name: &str) -> io::Result<()> {
        self.inner.remove_log(name)
    }

    fn put_blob(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        self.inner.put_blob(name, bytes)
    }

    fn blob(&self, name: &str) -> io::Result<Vec<u8>> {
        self.inner.blob(name)
    }

    fn blobs(&self) -> io::Result<Vec<String>> {
        self.inner.blobs()
    }

    fn remove_blob(&self, name: &str) -> io::Result<()> {
        self.inner.remove_blob(name)
    }

    fn destroy(&self) -> io::Result<()> {
        self.inner.destroy()
    }
}

impl Files for Marking {
    fn write_file(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        self.inner.write_file(name, bytes)
    }

    fn append_file(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        self.inner.append_file(name, bytes)
    }

    fn read_file(&self, name: &str) -> io::Result<Vec<u8>> {
        self.inner.read_file(name)
    }

    fn remove_file(&self, name: &str) -> io::Result<()> {
        self.inner.remove_file(name)
    }

    fn files(&self) -> io::Result<Vec<String>> {
        self.inner.files()
    }
}

/// Where rig keeps plain files of the session, which are named by path
/// and so stay out of its store: tool output cut short ([`Self::spill`]),
/// pasted images ([`Self::images`]) and the text log ([`Self::log`]: tracing,
/// panics, stray output). The session's directory by default; a plugin
/// places it elsewhere, or with `None` nowhere, while the app is built:
/// tools then only cut, pasting an image is refused, and the text log and
/// stray output are dropped.
#[derive(Resource, Clone, Debug, PartialEq, Eq)]
pub struct Scratch(pub Option<PathBuf>);

impl Scratch {
    /// Where tools keep output they cut, and plugins large answers, whole,
    /// under handles the model can `read` or `search`.
    pub fn spill(&self) -> Option<Spill> {
        self.0.as_ref().map(|dir| Spill(dir.join("spill")))
    }

    /// Where images pasted into the input are kept, which messages attach
    /// by path.
    pub fn images(&self) -> Option<PathBuf> {
        self.0.as_ref().map(|dir| dir.join("images"))
    }

    /// The text log.
    pub fn log(&self) -> Option<PathBuf> {
        self.0.as_ref().map(|dir| dir.join("agent.log"))
    }
}

/// The session the launcher names, or a new one when the agent runs
/// without it, under `home`, with its directory created.
pub(crate) fn paths_in(home: &Home) -> SessionPaths {
    let id = SessionId::from_env()
        .ok()
        .flatten()
        .unwrap_or_else(SessionId::generate);
    let dir = home.session(&id);
    // A directory that cannot be created shows up as a failed log write.
    // A build describing itself keeps none.
    if crate::describe::describing().is_none() {
        fs::create_dir_all(dir.path()).ok();
    }
    SessionPaths(dir)
}

/// The session's paths in the app's home, taken from the environment the
/// first time.
fn session_paths(world: &mut World) -> SessionPaths {
    let home = RigHome::of(world);
    world
        .get_resource_or_insert_with(|| paths_in(&home))
        .clone()
}

/// This app is its session's first start: no app of the session ran a
/// whole frame before. A fork's first app has it, a `/reload` of a session
/// started earlier does not; "a fork's first start" is [`ForkedFrom`]
/// with it. Inserted when the session plugin is built.
#[derive(Resource, Clone, Copy, Debug)]
pub struct FirstStart;

/// Inserts the [`SessionPaths`] from the environment, unless the log took
/// them already, the default [`SessionStores`] and [`Scratch`], the
/// [`SystemTimings`] (which stay empty without the log), opens the
/// [`SessionStore`] once every plugin is built and then inserts the
/// [`PluginData`], routes panics to the log,
/// tells a first start ([`FirstStart`]), and forks and deletes sessions,
/// finishing at startup the deletes a crash cut short.
pub struct SessionPlugin;

impl Plugin for SessionPlugin {
    fn build(&self, app: &mut App) {
        std::panic::set_hook(Box::new(|info| {
            error!("{info}\n{}", std::backtrace::Backtrace::capture());
        }));
        let paths = session_paths(app.world_mut());
        if !app.world().contains_resource::<Scratch>() {
            app.insert_resource(Scratch(Some(paths.path().to_owned())));
        }
        if !paths.started().exists() {
            app.insert_resource(FirstStart).add_systems(
                Last,
                mark_started
                    .run_if(run_once)
                    .run_if(resource_exists::<FirstStart>)
                    .before(discard_sessionless),
            );
        }
        app.init_resource::<SessionStores>()
            .init_resource::<SystemTimings>()
            .init_resource::<Forks>()
            .add_message::<SessionForked>()
            .add_message::<SessionDeleted>()
            .add_message::<SessionDeleteFailed>()
            .add_message::<SessionRewritten>()
            .add_message::<Notice>()
            .add_observer(|fork: On<ForkSession>, mut forks: ResMut<Forks>| {
                forks.0.push(fork.event().clone());
            })
            .add_observer(delete_session)
            .add_systems(
                Startup,
                finish_deletes.run_if(not(resource_exists::<Describing>)),
            )
            .add_systems(First, timings::next_frame)
            .configure_sets(Last, ForkSessions.after(WriteJournal))
            .add_systems(Last, fork_sessions.in_set(ForkSessions))
            .add_systems(
                Last,
                rewrite_scratch
                    .after(WriteJournal)
                    .run_if(on_message::<SessionRewritten>),
            )
            .add_systems(
                Last,
                discard_sessionless
                    .after(OnAppExitSystems)
                    .run_if(on_message::<AppExit>),
            );
    }

    /// Every plugin chose the stores and the scratch place by now: opens
    /// the session's store, or the one a test or a nested tool put in, and
    /// reads the session; only then moves the text log where [`Scratch`]
    /// says. A session that cannot be opened or read refuses the run, and
    /// the run writes nothing into its directory (`leave_session`). A
    /// build describing itself opens none.
    fn finish(&self, app: &mut App) {
        if !app.world().contains_resource::<Describing>() {
            match open_store(app.world_mut()) {
                Ok(()) => {
                    let wanted = app.world().get_resource::<Scratch>().and_then(Scratch::log);
                    move_text_log(app.world_mut(), wanted);
                }
                Err(failure) => {
                    refuse(
                        app,
                        format!(
                            "the session cannot be opened, so nothing of it was changed: {failure}"
                        ),
                    );
                }
            }
        }
        insert_plugin_data(app.world_mut());
    }
}

/// Inserts the [`PluginData`] of this run, unless the app has one: in the
/// session's store when it opened, for the [`ProjectRoot`] in the
/// [`ProjectWorkspace`].
fn insert_plugin_data(world: &mut World) {
    if world.contains_resource::<PluginData>() {
        return;
    }
    let home = RigHome::of(world);
    let session = world
        .get_resource::<SessionStore>()
        .map(|store| -> Arc<dyn Files> { store.0.clone() });
    let root = world
        .get_resource::<ProjectRoot>()
        .cloned()
        .unwrap_or_default();
    let workspace = world.get_resource::<ProjectWorkspace>();
    let host = workspace.and_then(|workspace| workspace.0.host());
    let data = PluginData::new(&home, session, &root.0, host);
    world.insert_resource(data);
}

/// Opens the session's store, the one in the world or else the one
/// [`SessionStores`] opens for [`SessionPaths`], and reads the session it
/// holds ([`open_session`]), before anything is written to it.
fn open_store(world: &mut World) -> Result<(), LoadError> {
    let store = match world.remove_resource::<SessionStore>() {
        Some(SessionStore(store)) => store,
        None => {
            let (Some(paths), Some(stores)) = (
                world.get_resource::<SessionPaths>(),
                world.get_resource::<SessionStores>(),
            ) else {
                return Ok(());
            };
            stores.open(paths)?
        }
    };
    open_session(world, store)
}

/// Keeps a refused run out of its session's directory: what it logged
/// goes to the launcher's run directory, where the launcher points, or
/// nowhere; it keeps no scratch files; and the session is not marked
/// started.
pub(crate) fn leave_session(world: &mut World) {
    let elsewhere = world.get_resource::<LauncherRun>().map(|run| run.0.log());
    move_text_log(world, elsewhere);
    world.insert_resource(Scratch(None));
    world.remove_resource::<FirstStart>();
}

/// Marks the session started at the end of its first app's first frame.
fn mark_started(paths: Res<SessionPaths>) {
    if let Err(failure) = fs::write(paths.started(), b"") {
        error!("could not mark the session started: {failure}");
    }
}

/// Removes the session's directory when a sessionless front's run ends,
/// however, once everything was written there. A run that failed first
/// moves its text log into the launcher's run directory, where the
/// launcher's failure message points.
fn discard_sessionless(
    mut exits: MessageReader<AppExit>,
    cli: Option<Res<Cli>>,
    paths: Res<SessionPaths>,
    (run, scratch): (Option<Res<LauncherRun>>, Option<Res<Scratch>>),
) {
    if !cli.is_some_and(|cli| cli.sessionless()) {
        return;
    }
    if !exits.read().all(AppExit::is_success)
        && let Some(run) = run
        && let Some(log) = scratch.and_then(|scratch| scratch.log())
    {
        fs::rename(log, run.0.log()).ok();
    }
    fs::remove_dir_all(paths.path()).ok();
}

/// Copies this session into a new one: every agent's log, saved state and
/// stored file, and the directory it works in. With `before`, an agent's
/// id and the id of a message in its conversation, the copy is the session
/// as it was before that message was made (see [`fork`]). The new
/// session's effect log starts empty.
/// [`SessionForked`] names it; switching to it is
/// [`Exit`](super::exit::Exit) with [`ExitReason::Switch`](super::exit::ExitReason::Switch).
#[derive(Event, Clone, Debug, Default)]
pub struct ForkSession {
    /// Where the copy stops: an agent's id and a message's id.
    pub before: Option<(AgentId, MessageId)>,
    /// Who asked, such as an entity of the plugin's own: [`SessionForked`]
    /// says so, so an asker knows its fork from another's.
    pub asker: Option<Entity>,
}

/// A session forked from this one, in the frame it was written.
#[derive(Message, Clone, Debug)]
pub struct SessionForked {
    /// This session.
    pub from: SessionId,
    /// The new session.
    pub to: SessionId,
    /// Where the copy stopped, as [`ForkSession`] asked.
    pub before: Option<(AgentId, MessageId)>,
    /// Who asked, as [`ForkSession`] said.
    pub asker: Option<Entity>,
}

/// The session a session was forked from, and where the copy stopped,
/// as [`ForkSession`] asked, with the text of the message it stopped
/// before: kept in the forked session's store, so a view of a session's
/// family reads no other session's log.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForkedFrom {
    /// The session it was copied from.
    pub session: SessionId,
    /// Where the copy stopped: an agent's id and a message's id.
    pub before: Option<(AgentId, MessageId)>,
    /// What the user wrote in the message the copy stopped before, its
    /// text without attached files; `None` without `before`, or for a
    /// message with no text of the user's.
    pub text: Option<String>,
}

/// How [`ForkedFrom`] is written.
#[derive(Serialize, Deserialize)]
struct Parent {
    session: String,
    before: Option<(AgentId, MessageId)>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    text: Option<String>,
}

/// The file of a forked session's store that holds its [`ForkedFrom`].
const PARENT: &str = "parent.json";

impl ForkedFrom {
    /// The session the session in `store` was forked from, if it was.
    /// Generic, so it is passed as it is where a store is any `Files`,
    /// such as `stores.as_deref().and_then(ForkedFrom::of)`.
    pub fn of(store: &(impl Files + ?Sized)) -> Option<Self> {
        let parent: Parent = serde_json::from_slice(&store.read_file(PARENT).ok()?).ok()?;
        Some(Self {
            session: parent.session.parse().ok()?,
            before: parent.before,
            text: parent.text,
        })
    }

    /// Writes it into the forked session's `store`.
    fn write(&self, store: &dyn Files) -> Result<(), String> {
        let parent = Parent {
            session: self.session.to_string(),
            before: self.before.clone(),
            text: self.text.clone(),
        };
        let text = serde_json::to_vec(&parent).map_err(|failure| failure.to_string())?;
        store
            .write_file(PARENT, &text)
            .map_err(|failure| failure.to_string())
    }
}

/// Rewrites the saved session in `dir`, which no rig runs, through
/// `with`, as done by `by` (rig-memory's `rewrite`): its store opened by
/// `stores`, then its scratch files, kept in its directory (the text log
/// and the spilled tool output). The running session, `running`, is
/// rewritten with `RewriteSession` instead. For a plugin redacting this
/// session's forks too:
///
/// ```no_run
/// use rig_harness::harness_protocol::SessionId;
/// use rig_harness::prelude::*;
///
/// fn redact_fork(
///     fork: SessionId,
///     (home, stores, paths): (Res<RigHome>, Res<SessionStores>, Res<SessionPaths>),
/// ) -> Result<(), String> {
///     let redact = Strings(|text: &mut String| {
///         let hit = text.contains("sk-123");
///         if hit {
///             *text = text.replace("sk-123", "[REDACTED]");
///         }
///         hit
///     });
///     let dir = home.0.session(&fork);
///     rewrite_stopped(&stores, &paths, &dir, Author::plugin("redact"), &redact).map(drop)
/// }
/// ```
///
/// # Errors
///
/// When `dir` is the running session or another rig runs it, when its
/// store does not open or its session cannot be read (nothing written), or
/// when a write fails.
pub fn rewrite_stopped(
    stores: &SessionStores,
    running: &SessionDir,
    dir: &SessionDir,
    by: Author,
    with: &dyn Rewrite,
) -> Result<Rewritten, String> {
    if dir.path() == running.path() {
        return Err("it is this session, which a RewriteSession rewrites".to_owned());
    }
    if dir.runs_elsewhere() {
        return Err("another rig runs it".to_owned());
    }
    let store = stores.open(dir).map_err(|failure| failure.to_string())?;
    let rewritten = rewrite(&*store, by, with).map_err(|failure| failure.to_string())?;
    let scratch = Scratch(Some(dir.path().to_owned()));
    map_spill(&scratch, with)
        .and_then(|()| {
            scratch
                .log()
                .map_or(Ok(false), |log| map_file(&log, LOG, with))
        })
        .map_err(|failure| format!("rewriting its scratch files failed: {failure}"))?;
    Ok(rewritten)
}

/// The text log's name to a [`Rewrite`].
const LOG: &str = "agent.log";

/// Maps the file at `path`, named `name` to `with`, through it: written
/// beside and renamed over, so it is the old file or the new, whole;
/// whether it changed. A file that is not there is left.
fn map_file(path: &Path, name: &str, with: &dyn Rewrite) -> io::Result<bool> {
    let mut bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(failure) if failure.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(failure) => return Err(failure),
    };
    if !with.file(name, &mut bytes) {
        return Ok(false);
    }
    let mut part = path.as_os_str().to_owned();
    part.push(".part");
    fs::write(&part, &bytes)?;
    fs::rename(&part, path)?;
    Ok(true)
}

/// Maps every file of `scratch`'s spilled tool output through `with`,
/// each named `spill/<file>`.
fn map_spill(scratch: &Scratch, with: &dyn Rewrite) -> io::Result<()> {
    let Some(spill) = scratch.spill() else {
        return Ok(());
    };
    let entries = match fs::read_dir(&spill.0) {
        Ok(entries) => entries,
        Err(failure) if failure.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(failure) => return Err(failure),
    };
    for entry in entries {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            let name = format!("spill/{}", entry.file_name().to_string_lossy());
            map_file(&entry.path(), &name, with)?;
        }
    }
    Ok(())
}

/// Maps this session's scratch files through each rewrite of the session:
/// the spilled tool output, and the text log under its lock, then logged
/// to again where it is. Pasted images are left.
fn rewrite_scratch(
    mut rewritten: MessageReader<SessionRewritten>,
    scratch: Option<Res<Scratch>>,
    log: Option<Res<TextLog>>,
) {
    for rewritten in rewritten
        .read()
        .filter(|rewritten| rewritten.result.is_ok())
    {
        let with = &*rewritten.with;
        let spilled = scratch
            .as_deref()
            .map_or(Ok(()), |scratch| map_spill(scratch, with));
        let logged = log.as_deref().map_or(Ok(()), |log| log.rewrite(with));
        if let Err(failure) = spilled.and(logged) {
            error!("rewriting the session's scratch files failed: {failure}");
        }
    }
}

/// Removes a saved session: not this one, and not one another rig runs.
/// The session is marked deleting first ([`SessionDir::mark_deleting`]),
/// which hides it from listings, then everything its store holds is
/// destroyed, wherever [`SessionStores`] keeps it, then its directory is
/// removed. A delete a crash cut short is finished at the next start.
/// [`SessionDeleted`] follows, or [`SessionDeleteFailed`] and an error
/// notice say why not.
#[derive(Event, Clone, Debug)]
pub struct DeleteSession(pub SessionId);

/// A session removed with [`DeleteSession`], or by the next start after
/// a crash cut its delete short.
#[derive(Message, Clone, Debug)]
pub struct SessionDeleted(pub SessionId);

/// A session [`DeleteSession`], or the next start finishing a delete a
/// crash cut short, could not remove, and why, such as `another rig runs
/// it`.
#[derive(Message, Clone, Debug)]
pub struct SessionDeleteFailed {
    /// The session.
    pub id: SessionId,
    /// Why it is still there.
    pub why: String,
}

/// The system in `Last` that copies the sessions [`ForkSession`] asked
/// for, after [`WriteJournal`]: a fork copies what the session logged up
/// to its frame, and a plugin that writes the session's store in that
/// frame says `.before(ForkSessions)` for the fork to have it.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ForkSessions;

/// The forks asked for this frame.
#[derive(Resource, Default)]
struct Forks(Vec<ForkSession>);

fn fork_sessions(
    mut forks: ResMut<Forks>,
    (paths, store, stores): (
        Res<SessionPaths>,
        Option<Res<SessionStore>>,
        Res<SessionStores>,
    ),
    home: Option<Res<RigHome>>,
    agents: Query<(&AgentId, &agent::Conversation)>,
    mut forked: MessageWriter<SessionForked>,
    mut notices: MessageWriter<Notice>,
) {
    for ForkSession { before, asker } in forks.0.drain(..) {
        let home = home
            .as_ref()
            .map_or_else(Home::from_env, |home| home.0.clone());
        let text = before.as_ref().and_then(|(agent, message)| {
            let (_, conversation) = agents.iter().find(|(id, _)| *id == agent)?;
            typed(conversation, conversation.index_of(*message)?)
        });
        let stores = (store.as_deref(), &*stores);
        match fork_into(&home, &paths, stores, (before.as_ref(), text)) {
            Ok((from, to)) => {
                forked.write(SessionForked {
                    from,
                    to,
                    before,
                    asker,
                });
            }
            Err(why) => {
                notices.write(Notice::error(
                    None,
                    format!("Could not fork the session: {why}"),
                ));
            }
        }
    }
}

/// What the user typed in message `at` of `conversation`: the text items
/// of a message of the user's own ([`EntryRef::is_users_own`](rig_core::conversation::EntryRef::is_users_own)), never of
/// one an agent or a plugin delivered, such as a note, nor a file
/// attached, each trimmed, joined by a blank line; `None` when it has
/// none.
pub fn typed(conversation: &Conversation, at: usize) -> Option<String> {
    let entry = conversation.entries().nth(at)?;
    let Message::User { content } = entry.message else {
        return None;
    };
    if !entry.is_users_own() {
        return None;
    }
    let typed: Vec<&str> = content
        .iter()
        .filter_map(|content| match content {
            UserContent::Text(text) if attached_file(&text.text).is_none() => {
                Some(text.text.trim())
            }
            _ => None,
        })
        .filter(|text| !text.is_empty())
        .collect();
    (!typed.is_empty()).then(|| typed.join("\n\n"))
}

/// What the user typed first in `conversation` ([`typed`]).
pub fn first_typed(conversation: &Conversation) -> Option<String> {
    (0..conversation.messages().len()).find_map(|at| typed(conversation, at))
}

/// Copies the session at `paths`, kept in `store`, into a new session of
/// `home` opened by `stores`, as of `before`, whose message's `text` the
/// copy keeps with its parent.
fn fork_into(
    home: &Home,
    paths: &SessionPaths,
    (store, stores): (Option<&SessionStore>, &SessionStores),
    (before, text): (Option<&(AgentId, MessageId)>, Option<String>),
) -> Result<(SessionId, SessionId), String> {
    let from = paths.id().ok_or("this session has no id")?;
    let store = store.ok_or("this session is not stored")?;
    let (to, target) = home.new_session().map_err(|failure| failure.to_string())?;
    let parent = ForkedFrom {
        session: from.clone(),
        before: before.cloned(),
        text,
    };
    let before = before.map(|(agent, message)| (agent.0.as_str(), *message));
    let copied = stores
        .open(&target)
        .and_then(|copy| fork(&*store.0, &*copy, before, &Keep).map(|_| copy))
        .map_err(|failure| failure.to_string())
        .and_then(|copy| parent.write(&*copy))
        .and_then(|()| match paths.working_directory() {
            Some(directory) => {
                fs::write(target.directory(), directory.to_string_lossy().as_bytes())
                    .map_err(|failure| failure.to_string())
            }
            None => Ok(()),
        });
    if let Err(why) = copied {
        fs::remove_dir_all(target.path()).ok();
        return Err(why);
    }
    Ok((from, to))
}

fn delete_session(
    delete: On<DeleteSession>,
    (paths, stores): (Res<SessionPaths>, Res<SessionStores>),
    home: Option<Res<RigHome>>,
    mut deleted: MessageWriter<SessionDeleted>,
    mut failed: MessageWriter<SessionDeleteFailed>,
    mut notices: MessageWriter<Notice>,
) {
    let home = home.map_or_else(Home::from_env, |home| home.0.clone());
    let session = home.session(&delete.0);
    let refusal = if session.path() == paths.path() {
        Some("it is this session".to_owned())
    } else if !session.path().is_dir() {
        Some("there is no such session".to_owned())
    } else if session.runs_elsewhere() {
        Some("another rig runs it".to_owned())
    } else {
        session
            .mark_deleting()
            .map_err(LoadError::from)
            .and_then(|()| finish_delete(&session, &stores))
            .err()
            .map(|failure| failure.to_string())
    };
    match refusal {
        None => {
            deleted.write(SessionDeleted(delete.0.clone()));
        }
        Some(why) => {
            notices.write(Notice::error(
                None,
                format!("Could not delete session {}: {why}.", delete.0),
            ));
            let id = delete.0.clone();
            failed.write(SessionDeleteFailed { id, why });
        }
    }
}

/// Removes the session in `session`, marked deleting: everything its store
/// holds, then its directory. Running it again after a crash finishes it.
/// A session an older rig saved has nothing outside its directory.
fn finish_delete(session: &SessionDir, stores: &SessionStores) -> Result<(), LoadError> {
    match (stores.0)(session) {
        Ok(store) => store.destroy()?,
        Err(LoadError::OldLayout) => {}
        Err(failure) => return Err(failure),
    }
    Ok(fs::remove_dir_all(session.path())?)
}

/// Finishes the deletes a crash cut short: every session of the app's
/// home marked deleting that no other rig runs.
fn finish_deletes(
    home: Option<Res<RigHome>>,
    (paths, stores): (Res<SessionPaths>, Res<SessionStores>),
    mut deleted: MessageWriter<SessionDeleted>,
    mut failed: MessageWriter<SessionDeleteFailed>,
    mut notices: MessageWriter<Notice>,
) {
    let Some(home) = home else {
        return;
    };
    let Ok(entries) = fs::read_dir(home.0.sessions()) else {
        return;
    };
    for entry in entries.flatten() {
        let session = SessionDir::at(entry.path());
        if !session.is_deleting() || session.path() == paths.path() || session.runs_elsewhere() {
            continue;
        }
        match (finish_delete(&session, &stores), session.id()) {
            (Ok(()), Some(id)) => {
                deleted.write(SessionDeleted(id));
            }
            (Ok(()), None) => {}
            (Err(failure), id) => {
                notices.write(Notice::warn(
                    None,
                    format!(
                        "Could not finish deleting the session in {}: {failure}.",
                        session.path().display()
                    ),
                ));
                if let Some(id) = id {
                    let why = failure.to_string();
                    failed.write(SessionDeleteFailed { id, why });
                }
            }
        }
    }
}

/// The `LogPlugin` formatter: plain text appended to the session's text
/// log, so nothing is written to stderr. The log comes before the
/// session's other plugins, so it takes the session from the environment
/// first, and logs into the session's directory until every plugin is
/// built, when the log moves where [`Scratch`] says.
pub(crate) fn log_layer(app: &mut App) -> Option<BoxedFmtLayer> {
    let at = session_paths(app.world_mut()).log();
    let earlier = fs::metadata(&at).ok().map(|earlier| earlier.len());
    let file = OpenOptions::new().create(true).append(true).open(&at).ok();
    let log = TextLog {
        at: file.is_some().then_some(at),
        earlier,
        file: Arc::new(Mutex::new(file)),
    };
    let writer = log.file.clone();
    app.insert_resource(log);
    let layer = fmt::Layer::default().with_ansi(false);
    Some(Box::new(
        layer.with_writer(move || LogWriter(writer.clone())),
    ))
}

/// The text log's file, and where it is.
#[derive(Resource)]
struct TextLog {
    at: Option<PathBuf>,
    /// The length of the file earlier runs left at `at`, which this run
    /// appends to; `None` when this run made it.
    earlier: Option<u64>,
    file: Arc<Mutex<Option<File>>>,
}

impl TextLog {
    /// Maps the text log through `with` under its lock, so no line is
    /// written meanwhile, and logs to the new file.
    fn rewrite(&self, with: &dyn Rewrite) -> io::Result<()> {
        let Some(at) = &self.at else {
            return Ok(());
        };
        let mut file = self.file.lock().unwrap_or_else(PoisonError::into_inner);
        if map_file(at, LOG, with)? {
            *file = OpenOptions::new().append(true).open(at).ok();
        }
        Ok(())
    }
}

/// Writes a line of the text log, or drops it without a file.
struct LogWriter(Arc<Mutex<Option<File>>>);

impl io::Write for LogWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let mut file = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        match file.as_mut() {
            Some(file) => file.write(bytes),
            None => Ok(bytes.len()),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        let mut file = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        file.as_mut().map_or(Ok(()), io::Write::flush)
    }
}

/// Moves what this run wrote to the text log so far to `wanted`, or
/// removes it for no place, and logs there from now on. What earlier runs
/// logged stays where it was.
fn move_text_log(world: &mut World, wanted: Option<PathBuf>) {
    let Some(mut log) = world.get_resource_mut::<TextLog>() else {
        return;
    };
    if log.at == wanted {
        return;
    }
    let handle = log.file.clone();
    let mut file = handle.lock().unwrap_or_else(PoisonError::into_inner);
    let moved = match (&log.at, log.earlier, &wanted) {
        (Some(at), None, Some(to)) => move_file(at, to),
        (Some(at), None, None) => fs::remove_file(at),
        (Some(at), Some(earlier), to) => take_tail(at, earlier, to.as_deref()),
        (None, _, _) => Ok(()),
    };
    if let Err(failure) = moved {
        // Said in the old log, which stays where it was.
        drop(file);
        error!("could not move the text log: {failure}");
        return;
    }
    *file = wanted
        .as_deref()
        .and_then(|to| OpenOptions::new().create(true).append(true).open(to).ok());
    drop(file);
    log.at = wanted;
    log.earlier = None;
}

/// Cuts the file `from` back to its first `length` bytes, appending what
/// followed to `to` when there is one.
fn take_tail(from: &Path, length: u64, to: Option<&Path>) -> io::Result<()> {
    if let Some(to) = to {
        let mut tail = File::open(from)?;
        tail.seek(SeekFrom::Start(length))?;
        if let Some(directory) = to.parent() {
            fs::create_dir_all(directory)?;
        }
        let mut moved = OpenOptions::new().create(true).append(true).open(to)?;
        io::copy(&mut tail, &mut moved)?;
    }
    OpenOptions::new().write(true).open(from)?.set_len(length)
}

/// Moves the file `from` to `to`, making its directory, across file
/// systems too.
fn move_file(from: &Path, to: &Path) -> io::Result<()> {
    if let Some(directory) = to.parent() {
        fs::create_dir_all(directory)?;
    }
    fs::rename(from, to).or_else(|_| {
        fs::copy(from, to)?;
        fs::remove_file(from)
    })
}

/// A warning or error this process logged, Bevy's own included.
#[derive(Reflect, Clone, Debug, Default)]
pub struct Logged {
    /// An error, else a warning.
    pub error: bool,
    /// The module that logged it, such as `bevy_app::hierarchy`.
    pub target: String,
    /// Its message, then its other fields.
    pub message: String,
    /// The `AgentId` its `agent` field names.
    pub agent: Option<String>,
    /// The plugin (its `Name`) whose module logged it, where that can be told.
    pub plugin: Option<String>,
}

/// The warnings and errors logged that nothing took yet. At most 256
/// wait; the rest are only in the text log.
#[derive(Resource)]
pub struct LogEvents(pub Receiver<Logged>);

/// The `LogPlugin` layer passing warnings and errors on to [`LogEvents`],
/// which exists before any plugin `plugins.toml` lists is added, and
/// timing systems into [`SystemTimings`].
pub fn log_events(app: &mut App) -> Option<BoxedLayer> {
    let (sender, receiver) = crossbeam_channel::bounded(256);
    let timings = SystemTimings::default();
    app.insert_resource(LogEvents(receiver))
        .insert_resource(timings.clone());
    Some(Box::new(PassOn(sender).and_then(TimeSystems(timings))))
}

struct PassOn(Sender<Logged>);

impl Layer<Registry> for PassOn {
    fn on_event(&self, event: &Event<'_>, _: Context<'_, Registry>) {
        let level = *event.metadata().level();
        if level <= Level::WARN {
            let target = event.metadata().target().to_owned();
            let error = level == Level::ERROR;
            let mut logged = Logged {
                error,
                target,
                ..Logged::default()
            };
            event.record(&mut logged);
            self.0.try_send(logged).ok();
        }
    }
}

/// A `log` record, such as Bevy's own warnings, names its module in
/// `log.target`.
impl Visit for Logged {
    fn record_str(&mut self, field: &Field, value: &str) {
        match field.name() {
            "log.target" => self.target = value.to_owned(),
            _ => self.record_debug(field, &format_args!("{value}")),
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn Debug) {
        match field.name() {
            "agent" => self.agent = Some(format!("{value:?}")),
            "message" => self.message.insert_str(0, &format!("{value:?}")),
            name if name.starts_with("log.") => {}
            name => write!(self.message, " {name}={value:?}").unwrap_or_default(),
        }
    }
}

#[cfg(test)]
mod tests;
