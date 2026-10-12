//! Where this process keeps its session: the directory the launcher names,
//! the text log, panics routed to that log instead of the terminal, and
//! the log's warnings and errors as [`Logged`] data. A plugin copies the
//! session into a new one with [`ForkSession`] and removes another with
//! [`DeleteSession`]; [`SessionForked`] and [`SessionDeleted`] tell every
//! plugin, such as one keeping files of its own per session. A forked
//! session names the session it came from ([`ForkedFrom`]). The first app
//! a session runs, a fork's first included, has [`FirstStart`]; what the
//! user typed in a message is [`typed`].

use std::fmt::{Debug, Write};
use std::fs::{self, File, OpenOptions, TryLockError};
use std::ops::Deref;
use std::sync::Mutex;

use bevy_app::OnAppExitSystems;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::tracing::field::{Field, Visit};
use bevy_log::tracing::{Event, Level};
use bevy_log::tracing_subscriber::layer::{Context, Layer};
use bevy_log::tracing_subscriber::{Registry, fmt};
use bevy_log::{BoxedFmtLayer, BoxedLayer, error};
use bevy_reflect::Reflect;
use crossbeam_channel::{Receiver, Sender};
use rig::harness_protocol::{Home, SessionDir, SessionId};

use rig_cassette::journal::{JsonlDirStore, MessageId, OriginKind, fork_session};
use rig_core::completion::Message;
use rig_core::message::UserContent;
use rig_ecs::WriteJournal;
use rig_ecs::agent::{AgentId, Conversation, Notice};
use rig_ecs::journal::SessionStore;
use rig_tools::Spill;
use rig_tools::attach::attached_file;
use serde::{Deserialize, Serialize};

use super::cli::Cli;
use super::launcher::LauncherRun;
use crate::services::RigHome;

/// The session's directory: the agent logs the core keeps there through
/// its [`SessionStore`], the effect log, and the launcher's files. Every
/// top-level `*.jsonl` file there but the effect log is an agent's log, so
/// a plugin keeps its files of the session in a directory of its own there.
#[derive(Resource, Clone, Debug)]
pub struct SessionPaths(pub SessionDir);

impl SessionPaths {
    /// Where tools keep output they cut, and plugins large answers, whole,
    /// under handles the model can `read` or `search`.
    pub fn spill(&self) -> Spill {
        Spill(self.path().join("spill"))
    }
}

impl Deref for SessionPaths {
    type Target = SessionDir;

    fn deref(&self) -> &SessionDir {
        &self.0
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

/// The marker file of a session an app started in, beside its logs. A
/// fork does not copy it.
const STARTED: &str = "started";

/// Inserts the [`SessionPaths`] from the environment, unless the log took
/// them already, and the [`SessionStore`] keeping the session there,
/// routes panics to the log, tells a first start ([`FirstStart`]), and
/// forks and deletes sessions.
pub struct SessionPlugin;

impl Plugin for SessionPlugin {
    fn build(&self, app: &mut App) {
        std::panic::set_hook(Box::new(|info| {
            error!("{info}\n{}", std::backtrace::Backtrace::capture());
        }));
        let paths = session_paths(app.world_mut());
        if !app.world().contains_resource::<SessionStore>() {
            app.insert_resource(SessionStore::new(JsonlDirStore::new(paths.path())));
        }
        if !paths.path().join(STARTED).exists() {
            app.insert_resource(FirstStart).add_systems(
                Last,
                mark_started.run_if(run_once).before(discard_sessionless),
            );
        }
        app.init_resource::<Forks>()
            .add_message::<SessionForked>()
            .add_message::<SessionDeleted>()
            .add_observer(|fork: On<ForkSession>, mut forks: ResMut<Forks>| {
                forks.0.push(fork.event().clone());
            })
            .add_observer(delete_session)
            // A fork copies what the session logged up to this frame.
            .add_systems(Last, fork_sessions.after(WriteJournal))
            .add_systems(
                Last,
                discard_sessionless
                    .after(OnAppExitSystems)
                    .run_if(on_message::<AppExit>),
            );
    }
}

/// Marks the session started at the end of its first app's first frame.
fn mark_started(paths: Res<SessionPaths>) {
    if let Err(failure) = fs::write(paths.path().join(STARTED), b"") {
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
    run: Option<Res<LauncherRun>>,
) {
    if !cli.is_some_and(|cli| cli.sessionless()) {
        return;
    }
    if !exits.read().all(AppExit::is_success)
        && let Some(run) = run
    {
        fs::rename(paths.log(), run.0.log()).ok();
    }
    fs::remove_dir_all(paths.path()).ok();
}

/// Copies this session into a new one: every agent's log, saved state and
/// stored file, and the directory it works in. With `before`, an agent's
/// id and the id of a message in its conversation, the copy is the session
/// as it was before that message was made (see [`fork_session`]). The new session's effect log starts empty.
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
/// before: kept in the forked session's [`SessionDir::parent`], so a view
/// of a session's family reads no other session's log.
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

impl ForkedFrom {
    /// The session `session` was forked from, if it was.
    pub fn of(session: &SessionDir) -> Option<Self> {
        let parent: Parent = serde_json::from_slice(&fs::read(session.parent()).ok()?).ok()?;
        Some(Self {
            session: parent.session.parse().ok()?,
            before: parent.before,
            text: parent.text,
        })
    }

    /// Writes it into `session`.
    fn write(&self, session: &SessionDir) -> Result<(), String> {
        let parent = Parent {
            session: self.session.to_string(),
            before: self.before.clone(),
            text: self.text.clone(),
        };
        let text = serde_json::to_vec(&parent).map_err(|failure| failure.to_string())?;
        fs::write(session.parent(), text).map_err(|failure| failure.to_string())
    }
}

/// Removes a saved session's directory: not this one, and not one another
/// rig runs. [`SessionDeleted`] follows, or an error notice says why not.
#[derive(Event, Clone, Debug)]
pub struct DeleteSession(pub SessionId);

/// A session removed with [`DeleteSession`].
#[derive(Message, Clone, Debug)]
pub struct SessionDeleted(pub SessionId);

/// The forks asked for this frame.
#[derive(Resource, Default)]
struct Forks(Vec<ForkSession>);

fn fork_sessions(
    mut forks: ResMut<Forks>,
    paths: Res<SessionPaths>,
    store: Option<Res<SessionStore>>,
    home: Option<Res<RigHome>>,
    agents: Query<(&AgentId, &Conversation)>,
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
        match fork_into(&home, &paths, store.as_deref(), (before.as_ref(), text)) {
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

/// What the user typed in message `at` of `conversation`: its text items
/// of the user's own, never text an agent or a plugin delivered, such as a
/// note, nor a file attached, each trimmed, joined by a blank line; `None`
/// when it has none.
pub fn typed(conversation: &Conversation, at: usize) -> Option<String> {
    let Some(Message::User { content }) = conversation.messages().get(at) else {
        return None;
    };
    let typed: Vec<&str> = content
        .iter()
        .enumerate()
        .filter_map(|(item, content)| match content {
            UserContent::Text(text)
                if conversation
                    .origin(at, item)
                    .is_none_or(|origin| origin.kind == OriginKind::User)
                    && attached_file(&text.text).is_none() =>
            {
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

/// Copies the session at `paths` into a new session of `home`, as of
/// `before`, whose message's `text` the copy keeps with its parent.
fn fork_into(
    home: &Home,
    paths: &SessionPaths,
    store: Option<&SessionStore>,
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
    let copied = fork_session(&*store.0, &JsonlDirStore::new(target.path()), before)
        .map_err(|failure| failure.to_string())
        .and_then(|()| parent.write(&target))
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
    paths: Res<SessionPaths>,
    home: Option<Res<RigHome>>,
    mut deleted: MessageWriter<SessionDeleted>,
    mut notices: MessageWriter<Notice>,
) {
    let home = home.map_or_else(Home::from_env, |home| home.0.clone());
    let session = home.session(&delete.0);
    let refusal = if session.path() == paths.path() {
        Some("it is this session".to_owned())
    } else if !session.path().is_dir() {
        Some("there is no such session".to_owned())
    } else if runs_elsewhere(&session) {
        Some("another rig runs it".to_owned())
    } else {
        fs::remove_dir_all(session.path())
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
        }
    }
}

/// Whether a launcher holds `session`'s lock.
fn runs_elsewhere(session: &SessionDir) -> bool {
    File::options()
        .write(true)
        .open(session.launcher_lock())
        .is_ok_and(|lock| matches!(lock.try_lock(), Err(TryLockError::WouldBlock)))
}

/// The `LogPlugin` formatter: plain text appended to the session's log, so
/// nothing is written to stderr. The log comes before the session's other
/// plugins, so it takes the session from the environment first.
pub(crate) fn log_layer(app: &mut App) -> Option<BoxedFmtLayer> {
    let paths = session_paths(app.world_mut()).log();
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(paths)
        .ok();
    let layer = fmt::Layer::default().with_ansi(false);
    Some(match file {
        Some(file) => Box::new(layer.with_writer(Mutex::new(file))),
        None => Box::new(layer.with_writer(std::io::sink)),
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
/// which exists before any plugin `plugins.toml` lists is added.
pub fn log_events(app: &mut App) -> Option<BoxedLayer> {
    let (sender, receiver) = crossbeam_channel::bounded(256);
    app.insert_resource(LogEvents(receiver));
    Some(Box::new(PassOn(sender)))
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
