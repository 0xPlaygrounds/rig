//! What every plugin shares about the process it runs in, as resources:
//! the `RIG_HOME` layout ([`RigHome`]), the project the session works on
//! ([`ProjectRoot`]), how child processes start ([`Processes`]), and the
//! tokio runtime rig's HTTP requests run on ([`TokioRuntime`]). Each is
//! taken from the environment unless the app inserted one first, which is
//! how a test points a plugin at a directory of its own; a plugin that
//! should also work in an app without the harness's plugins inits the
//! ones it reads (`app.init_resource::<ProjectRoot>()`).
//!
//! ```
//! use rig_harness::prelude::*;
//!
//! let mut app = App::new();
//! app.insert_resource(ProjectRoot("/srv/project".into()))
//!     .add_plugins(rig_harness::services::ServicesPlugin);
//! let root = &app.world().resource::<ProjectRoot>().0;
//! assert_eq!(root, std::path::Path::new("/srv/project"));
//! ```

use std::path::PathBuf;
use std::sync::{Arc, PoisonError, RwLock};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::warn;
use rig::harness_protocol::{Home, env};
use rig_tools::process::Launcher;

/// The `RIG_HOME` this agent runs under: where its sessions, settings,
/// sign-ins and builds are.
#[derive(Resource, Clone, Debug)]
pub struct RigHome(pub Home);

impl RigHome {
    /// The app's home, else the one the environment names.
    pub fn of(world: &World) -> Home {
        world
            .get_resource::<Self>()
            .map_or_else(Home::from_env, |home| home.0.clone())
    }
}

/// The one the environment names.
impl Default for RigHome {
    fn default() -> Self {
        Self(Home::from_env())
    }
}

/// The directory the session works in: where it was first started, which
/// the launcher starts a resumed session in again. Paths the user and the
/// model name are relative to it.
#[derive(Resource, Clone, Debug, PartialEq, Eq)]
pub struct ProjectRoot(pub PathBuf);

/// This process's working directory, which the launcher sets to the
/// session's.
impl Default for ProjectRoot {
    fn default() -> Self {
        Self(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
    }
}

/// How every child process starts: under which wrapper, in which
/// directory, with which environment. The shell tool and any plugin that
/// starts a process build each command from it when the process starts
/// (`processes.launcher().command(program)`), so a sandbox plugin that
/// changes it ([`Processes::set`]), in `build` or while the app runs,
/// changes how the next process starts. By default no command inherits
/// what makes a program act as this agent ([`env::AGENT_ONLY`]).
#[derive(Resource, Clone, Debug)]
pub struct Processes(Arc<RwLock<Launcher>>);

impl Processes {
    /// How the next process starts.
    pub fn launcher(&self) -> Launcher {
        self.0
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Starts every process from now on as `launcher` says.
    pub fn set(&self, launcher: Launcher) {
        *self.0.write().unwrap_or_else(PoisonError::into_inner) = launcher;
    }
}

impl Default for Processes {
    fn default() -> Self {
        let launcher = Launcher::default().without_env(env::AGENT_ONLY);
        Self(Arc::new(RwLock::new(launcher)))
    }
}

/// The tokio runtime rig's HTTP requests run on, for a plugin whose
/// library needs tokio (an MCP client, a web server): one runtime per
/// process instead of one per plugin.
#[derive(Resource, Clone, Debug)]
pub struct TokioRuntime(pub tokio::runtime::Handle);

/// Inserts [`RigHome`], [`ProjectRoot`], [`Processes`] and
/// [`TokioRuntime`], each unless the app has it.
pub struct ServicesPlugin;

impl Plugin for ServicesPlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<TokioRuntime>() {
            match rig_reqwest::runtime() {
                Ok(handle) => {
                    app.insert_resource(TokioRuntime(handle));
                }
                Err(failure) => warn!("no tokio runtime for plugins: {failure}"),
            }
        }
        app.init_resource::<RigHome>()
            .init_resource::<ProjectRoot>()
            .init_resource::<Processes>();
    }
}

#[cfg(test)]
mod tests;
