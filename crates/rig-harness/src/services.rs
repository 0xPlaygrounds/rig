//! What every plugin shares about the process it runs in, as resources:
//! the `RIG_HOME` layout ([`RigHome`]), the project the session works on
//! ([`ProjectRoot`]) and the directory an agent works in when not that one
//! ([`WorkDir`]), how child processes start ([`Processes`]), and the
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

use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::{Arc, PoisonError, RwLock};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::warn;
use bevy_reflect::prelude::*;
use rig_ecs::agent::SpawnedBy;
use rig_ecs::session_log::ReflectSaved;
use rig_harness_protocol::{Home, env, secret_variables};
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
/// the launcher starts a resumed session in again, and sets the agent's
/// working directory to. Paths the user and the model name are relative to
/// it: a file tool of an agent without a [`WorkDir`] resolves them from
/// here, and a command starts here, in the app's
/// [`ProjectWorkspace`](crate::workspace::ProjectWorkspace)
/// (`rig_basics::coding_tools`).
#[derive(Resource, Clone, Debug, PartialEq, Eq)]
pub struct ProjectRoot(pub PathBuf);

/// This process's working directory, which the launcher sets to the
/// session's.
impl Default for ProjectRoot {
    fn default() -> Self {
        Self(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
    }
}

/// On an agent: the directory it works in, such as a worktree of its own,
/// in place of the [`ProjectRoot`]. Its file tools resolve relative paths
/// from it and refuse a path outside it, and its commands start in it
/// (`rig_basics::coding_tools`); what a command does once started is not confined.
/// Saved with the agent. An agent it spawns works there too, unless given
/// one of its own.
#[derive(Component, Reflect, Clone, Debug, PartialEq, Eq)]
#[reflect(Component, Saved, Clone, Debug, PartialEq)]
pub struct WorkDir(pub PathBuf);

/// How every child process starts: under which wrapper, in which
/// directory, with which environment. The shell tool and any plugin that
/// starts a process build each command from it when the process starts
/// (`processes.launcher().command(program)`), so a sandbox plugin that
/// changes it ([`Processes::set`]), in `build` or while the app runs,
/// changes how the next process starts. By default no command inherits
/// what makes a program act as this agent ([`env::AGENT_ONLY`]), and no
/// command ever inherits a secret
/// ([`secrets`](crate::secrets)), whatever launcher is set.
#[derive(Resource, Clone, Debug)]
pub struct Processes {
    launcher: Arc<RwLock<Launcher>>,
    /// The secret variables of this process, which no command inherits.
    secrets: Arc<[OsString]>,
}

impl Processes {
    /// How the next process starts: the launcher set, keeping every secret
    /// from the command.
    pub fn launcher(&self) -> Launcher {
        let launcher = self
            .launcher
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        let kept: Vec<OsString> = self
            .secrets
            .iter()
            .filter(|secret| !launcher.unset_env.contains(secret))
            .cloned()
            .collect();
        launcher.without_env(kept)
    }

    /// Starts every process from now on as `launcher` says, without the
    /// secrets.
    pub fn set(&self, launcher: Launcher) {
        *self
            .launcher
            .write()
            .unwrap_or_else(PoisonError::into_inner) = launcher;
    }
}

impl Default for Processes {
    fn default() -> Self {
        // The worker's token too: with it a command could call `rigd` as
        // its session.
        let launcher = Launcher::default()
            .without_env(env::AGENT_ONLY)
            .without_env([env::WORKER_TOKEN]);
        Self {
            launcher: Arc::new(RwLock::new(launcher)),
            secrets: secret_variables().into(),
        }
    }
}

/// The tokio runtime rig's HTTP requests run on, for a plugin whose
/// library needs tokio (an MCP client, a web server): one runtime per
/// process instead of one per plugin.
#[derive(Resource, Clone, Debug)]
pub struct TokioRuntime(pub tokio::runtime::Handle);

/// Inserts [`RigHome`], [`ProjectRoot`], [`Processes`] and
/// [`TokioRuntime`], each unless the app has it, and gives an agent
/// spawned by one with a [`WorkDir`] that directory.
pub struct ServicesPlugin;

impl Plugin for ServicesPlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<TokioRuntime>() {
            match rig_http::runtime::handle() {
                Ok(handle) => {
                    app.insert_resource(TokioRuntime(handle));
                }
                Err(failure) => warn!("no tokio runtime for plugins: {failure}"),
            }
        }
        app.init_resource::<RigHome>()
            .init_resource::<ProjectRoot>()
            .init_resource::<Processes>()
            .add_observer(inherit_work_dir);
    }
}

/// Gives a spawned agent its parent's [`WorkDir`], unless it has one.
fn inherit_work_dir(
    spawned: On<Add<SpawnedBy>>,
    parents: Query<&SpawnedBy>,
    dirs: Query<&WorkDir>,
    mut commands: Commands,
) {
    let parent = parents.get(spawned.entity).map(|&SpawnedBy(parent)| parent);
    if let Some(dir) = parent.ok().and_then(|parent| dirs.get(parent).ok()) {
        commands
            .entity(spawned.entity)
            .try_insert_if_new(dir.clone());
    }
}

#[cfg(test)]
mod tests;
