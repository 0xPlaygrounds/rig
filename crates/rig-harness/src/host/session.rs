//! Where this process keeps its session: the directory the launcher names,
//! the text log, and panics routed to that log instead of the terminal.

use std::fs::{self, OpenOptions};
use std::ops::Deref;
use std::sync::Mutex;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::tracing_subscriber::fmt;
use bevy_log::{BoxedFmtLayer, error};
use rig::harness_protocol::{Home, SessionDir, SessionId};

use crate::core::fs_journal::JsonlDirStore;
use crate::core::store::SessionStore;

/// The session's directory: the agent logs and the effect log the core
/// keeps there through its [`SessionStore`], and the launcher's files.
#[derive(Resource, Clone, Debug)]
pub struct SessionPaths(pub SessionDir);

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

/// Inserts the [`SessionPaths`] from the environment, and the
/// [`SessionStore`] keeping the session there, and routes panics to the
/// log.
pub struct SessionPlugin;

impl Plugin for SessionPlugin {
    fn build(&self, app: &mut App) {
        std::panic::set_hook(Box::new(|info| {
            error!("{info}\n{}", std::backtrace::Backtrace::capture());
        }));
        let paths = paths_from_env();
        app.insert_resource(SessionStore::new(JsonlDirStore::new(paths.path())))
            .insert_resource(paths);
    }
}

/// The `LogPlugin` formatter: plain text appended to the session's log, so
/// nothing is written to stderr.
pub(crate) fn log_layer(app: &mut App) -> Option<BoxedFmtLayer> {
    let file = app
        .world()
        .get_resource::<SessionPaths>()
        .and_then(|paths| {
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(paths.log())
                .ok()
        });
    let layer = fmt::Layer::default().with_ansi(false);
    Some(match file {
        Some(file) => Box::new(layer.with_writer(Mutex::new(file))),
        None => Box::new(layer.with_writer(std::io::sink)),
    })
}
