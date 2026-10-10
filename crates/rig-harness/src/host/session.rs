//! Where this process keeps its session: the directory the launcher names,
//! the text log, panics routed to that log instead of the terminal, and
//! the log's warnings and errors as [`Logged`] data.

use std::fmt::{Debug, Write};
use std::fs::{self, OpenOptions};
use std::ops::Deref;
use std::sync::Mutex;

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

use rig_cassette::journal::JsonlDirStore;
use rig_ecs::journal::SessionStore;
use rig_tools::Spill;

/// The session's directory: the agent logs the core keeps there through
/// its [`SessionStore`], the effect log, and the launcher's files.
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
/// without it, under the launcher's `RIG_HOME`, with its directory
/// created.
pub(crate) fn paths_from_env() -> SessionPaths {
    let id = SessionId::from_env()
        .ok()
        .flatten()
        .unwrap_or_else(SessionId::generate);
    let dir = Home::from_env().session(&id);
    // A directory that cannot be created shows up as a failed log write.
    fs::create_dir_all(dir.path()).ok();
    SessionPaths(dir)
}

/// Inserts the [`SessionPaths`] from the environment, unless the log took
/// them already, and the [`SessionStore`] keeping the session there, and
/// routes panics to the log.
pub struct SessionPlugin;

impl Plugin for SessionPlugin {
    fn build(&self, app: &mut App) {
        std::panic::set_hook(Box::new(|info| {
            error!("{info}\n{}", std::backtrace::Backtrace::capture());
        }));
        let paths = app
            .world_mut()
            .get_resource_or_insert_with(paths_from_env)
            .clone();
        app.insert_resource(SessionStore::new(JsonlDirStore::new(paths.path())));
    }
}

/// The `LogPlugin` formatter: plain text appended to the session's log, so
/// nothing is written to stderr. The log comes before the session's other
/// plugins, so it takes the session from the environment first.
pub(crate) fn log_layer(app: &mut App) -> Option<BoxedFmtLayer> {
    let paths = app
        .world_mut()
        .get_resource_or_insert_with(paths_from_env)
        .log();
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
pub(crate) fn log_events(app: &mut App) -> Option<BoxedLayer> {
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
