//! What a tool call would leave on disk, learned before it runs: put
//! [`Preview`] on the call in [`Stage::Gate`], and the harness holds it,
//! runs its tool on an [`Overlay`] of the workspace, which keeps every
//! write from the disk, and puts the changes on it as [`Previewed`]. The
//! call is then [`Undecided`] again in a later pass, for the gate to decide
//! with the changes in hand.
//!
//! ```
//! use rig_harness::prelude::*;
//!
//! /// Asks before a call that would change more than 200 lines.
//! fn guard(
//!     calls: Query<(Entity, Option<&Previewed>), (ToolCalls, Undecided)>,
//!     mut commands: Commands,
//! ) {
//!     for (call, previewed) in &calls {
//!         match previewed {
//!             None => {
//!                 commands.entity(call).insert(Preview);
//!             }
//!             Some(Previewed(Ok(changes))) => {
//!                 let lines: usize = changes.iter().map(|it| it.added + it.removed).sum();
//!                 if lines > 200 { /* hold it and ask, with the changes */ }
//!             }
//!             // Not previewable (not registered per call, or `shell`), or
//!             // it would fail.
//!             Some(Previewed(Err(_))) => {}
//!         }
//!     }
//! }
//!
//! App::new().add_systems(AgentPass, guard.in_set(Stage::Gate));
//! ```
//!
//! Only a tool registered per call
//! ([`add_tool_per_call`](crate::per_call::AppPerCallExt::add_tool_per_call))
//! can be previewed; another gets `Previewed(Err(..))`. The tool runs for
//! real but on the overlay: it can read every file, its writes, removals
//! and renames stay in the overlay, and it cannot start a process (so a
//! `shell` call's preview is the overlay's refusal). Anything the tool does
//! other than through its [`Place`]'s workspace, such as a request over
//! the network, it does in the preview too.

use std::sync::Arc;

use bevy_app::{App, Plugin};
use bevy_ecs::prelude::*;
use bevy_tasks::{AsyncComputeTaskPool, TaskPool};
use rig_core::effect::{EffectKind, Outcome};
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::serve::serve_inline;
use rig_core::tool::{FileChange, ToolErrorKind};
use rig_ecs::effects::Handler;
use rig_ecs::prelude::{
    AgentPass, ForAgent, GateSystems, HoldOf, Running, Stage, Started, ToolCalls, ToolRequest, Wake,
};
use rig_tools::overlay::Overlay;

use crate::per_call::{PerCall, Place, base};
use crate::services::{ProjectRoot, WorkDir};
use crate::workspace::ProjectWorkspace;

#[cfg(doc)]
use rig_ecs::prelude::Undecided;

/// Put on a tool call in [`Stage::Gate`] (or [`GateSystems::After`]) to
/// learn what it would leave on disk: the harness holds the call, runs its
/// tool on an overlay of the workspace off the main thread, then puts
/// [`Previewed`] on it and releases it.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct Preview;

/// What a call with [`Preview`] would leave on disk: the files it would
/// create, modify and delete, each with its diff and the hashes of its
/// bytes before and after; or why it cannot say, such as the tool
/// failing, a tool not registered per call, or a command it would start.
#[derive(Component, Clone, Debug)]
pub struct Previewed(pub Result<Vec<FileChange>, ErrorReport>);

/// Previews the tool calls that ask for it ([`Preview`]), after every gate,
/// before Start.
pub(crate) struct PreviewPlugin;

/// Where [`PreviewPlugin`] holds the calls it previews, before Start.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PreviewSystems;

impl Plugin for PreviewPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .configure_sets(
                AgentPass,
                PreviewSystems
                    .after(GateSystems::After)
                    .before(Stage::Start),
            )
            .add_systems(AgentPass, preview.in_set(PreviewSystems))
            .add_observer(previewed);
    }
}

/// What a preview's run found, on its hold.
#[derive(Component)]
struct Ran(Result<Vec<FileChange>, ErrorReport>);

impl From<Result<Vec<FileChange>, ErrorReport>> for Ran {
    fn from(ran: Result<Vec<FileChange>, ErrorReport>) -> Self {
        Self(ran)
    }
}

/// Holds each call that asks for a preview and runs its tool on an overlay
/// of the workspace; a call no registered tool can build is answered at
/// once.
fn preview(
    calls: Query<
        (Entity, &ToolRequest, Option<&ForAgent>),
        (ToolCalls, Added<Preview>, Without<Started>),
    >,
    dirs: Query<&WorkDir>,
    tools: Option<Res<PerCall>>,
    (workspace, root): (Res<ProjectWorkspace>, Res<ProjectRoot>),
    wake: Res<Wake>,
    mut commands: Commands,
) {
    for (call, request, agent) in &calls {
        let overlay = Arc::new(Overlay::new(workspace.0.clone()));
        let (base, confine) = base(agent.map(|agent| agent.0), &dirs, &root);
        let place = Place {
            workspace: overlay.clone(),
            base,
            confine,
            live: None,
            spill: None,
        };
        let name = request.call.function.name.as_str();
        let Some(Handler(handler)) = tools.as_ref().and_then(|tools| tools.build(name, place))
        else {
            let report = ErrorReport::new(
                ErrorKind::Tool(ToolErrorKind::Other),
                format!("`{name}` cannot be previewed: it is not registered per call"),
            );
            commands.entity(call).insert(Previewed(Err(report)));
            continue;
        };
        let kind = EffectKind::from(&request.call);
        let work = async move {
            if let Outcome::ToolResult { result } = serve_inline(&handler, kind).await? {
                result.into_result().map_err(ErrorReport::from)?;
            }
            Ok(overlay.changes())
        };
        let pool = AsyncComputeTaskPool::get_or_init(TaskPool::default);
        commands.spawn((
            Name::new("preview"),
            HoldOf(call),
            Running::spawn_into::<Ran, _>(pool, &wake, work),
        ));
    }
}

/// Puts what a preview found on its call, and releases the call.
fn previewed(ran: On<Insert<Ran>>, holds: Query<(&HoldOf, &Ran)>, mut commands: Commands) {
    let Ok((&HoldOf(call), Ran(found))) = holds.get(ran.entity) else {
        return;
    };
    commands.entity(call).try_insert(Previewed(found.clone()));
    commands.entity(ran.entity).despawn();
}

#[cfg(test)]
mod tests;
