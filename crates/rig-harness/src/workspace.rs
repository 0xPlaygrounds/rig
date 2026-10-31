//! The workspace the agent works in: [`ProjectWorkspace`].

use std::sync::Arc;

use bevy_ecs::prelude::*;
use rig_tools::workspace::{Local, Workspace};

/// The workspace the tools act on: this machine's ([`Local`]) unless a
/// plugin puts another in while the app is built, such as one reached over
/// SSH. Paths in [`ProjectRoot`](crate::services::ProjectRoot),
/// [`WorkDir`](crate::services::WorkDir) and tools' facts
/// ([`FilesRead`](rig_core::tool::FilesRead),
/// [`FilesChanged`](rig_core::tool::FilesChanged)) are in it. Run project
/// commands, such as `git`, with its [`spawn`](Workspace::spawn), so they
/// follow it; other plugins learn it is not this machine from its
/// [`host`](Workspace::host).
///
/// **The main-thread rule.** Tools call it inside
/// [`blocking`](rig_tools::blocking). A system calls it inline only when
/// `host()` is `None` and the path is its own, not one a user or a model
/// wrote: a workspace such as [`Hosts`](rig_tools::remote::Hosts) is this
/// machine but routes `HOST:` paths to another. Otherwise it reads as
/// [`Running::spawn`](rig_ecs::tasks::Running::spawn) work, under a hold
/// ([`HoldOf`](rig_ecs::stage::HoldOf)) on what waits for it, so a remote
/// path never blocks a frame.
///
/// ```
/// use std::sync::Arc;
/// use rig_harness::prelude::*;
/// use rig_harness::rig_tools::workspace::Local;
///
/// let mut app = App::new();
/// // A plugin of another workspace inserts its own in `build`, with the
/// // project's path there.
/// app.insert_resource(ProjectWorkspace(Arc::new(Local)))
///     .insert_resource(ProjectRoot("/srv/app".into()));
/// assert!(app.world().resource::<ProjectWorkspace>().0.host().is_none());
/// ```
#[derive(Resource, Clone, Debug)]
pub struct ProjectWorkspace(pub Arc<dyn Workspace>);

/// This machine.
impl Default for ProjectWorkspace {
    fn default() -> Self {
        Self(Arc::new(Local))
    }
}
