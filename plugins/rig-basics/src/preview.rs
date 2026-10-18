//! What a tool call would leave on disk, learned before it runs: put
//! [`Preview`] on the call in [`Stage::Gate`], and the harness holds it,
//! runs its tool on an [`Overlay`] of the workspace that keeps every write
//! from the disk, and puts the changes on it as [`Previewed`]; the call is
//! then ready to start again for the gate to decide with the changes in
//! hand.
//!
//! [`Stage::Gate`]: rig_ecs::prelude::Stage::Gate
//!
//! Only a tool registered per call
//! ([`add_tool_per_call`](rig_harness::per_call::AppPerCallExt::add_tool_per_call))
//! can be previewed, as [`PerCallTools`] builds its call on the overlay;
//! another gets `Previewed(Err(..))`. The tool runs for
//! real on the overlay: it reads every file, its writes stay in the overlay,
//! and it cannot start a process (so `shell` cannot be previewed). What it
//! does other than through its [`CallPlace`]'s workspace, such as a network
//! request, it does in the preview too. The plugin guide's
//! [gating section](rig_harness::plugin_guide#gating-and-reviewing-tool-calls) has
//! an example.

use std::sync::Arc;

use bevy_app::{App, Plugin};
use bevy_ecs::prelude::*;
use bevy_tasks::{AsyncComputeTaskPool, TaskPool};
use rig_core::effect::{Answered, EffectKind};
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::serve::serve_inline;
use rig_core::tool::{FileChange, ToolErrorKind};
use rig_ecs::effects::Handler;
use rig_ecs::prelude::{
    AgentPass, CallHandler, ForAgent, GateSystems, HoldOf, Running, Started, ToolCalls,
    ToolRequest, Wake,
};
use rig_tools::overlay::Overlay;

use rig_harness::per_call::PerCallTools;
use rig_harness::services::ProjectRoot;
use rig_harness::workspace::ProjectWorkspace;

#[cfg(doc)]
use rig_ecs::prelude::ReadyToStart;

/// Put on a tool call in [`Stage::Gate`] (or [`GateSystems::After`]) to
/// learn what it would leave on disk: the harness holds the call, runs its
/// tool on an overlay of the workspace off the main thread, then puts
/// [`Previewed`] on it and releases it.
///
/// [`Stage::Gate`]: rig_ecs::prelude::Stage::Gate
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct Preview;

/// What a call with [`Preview`] would leave on disk: the files it would
/// create, modify and delete, each with its diff (a deletion's too) and the
/// [`FileState`](rig_core::tool::FileState) of each side, its size and
/// lines; or why it cannot say, such as the tool failing, a tool not
/// registered per call, or a command it would start, whose writes cannot be
/// kept from the disk.
#[derive(Component, Clone, Debug)]
pub struct Previewed(pub Result<Vec<FileChange>, ErrorReport>);

/// Previews the tool calls that ask for it ([`Preview`]), after every gate,
/// before Start. A plugin that puts [`Preview`] on calls adds it, unless
/// `app.is_plugin_added::<PreviewPlugin>()`.
#[derive(Default)]
pub struct PreviewPlugin;

/// Where [`PreviewPlugin`] holds the calls it previews, after
/// [`GateSystems::After`] and before [`GateSystems::Fallback`].
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PreviewSystems;

impl Plugin for PreviewPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .configure_sets(
                AgentPass,
                PreviewSystems
                    .after(GateSystems::After)
                    .before(GateSystems::Fallback),
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
    tools: PerCallTools,
    workspace: Res<ProjectWorkspace>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    for (call, request, agent) in &calls {
        let overlay = Arc::new(Overlay::new(workspace.0.clone()));
        let place = tools.place(agent.map(|agent| agent.0), overlay.clone());
        let name = request.call.function.name.as_str();
        let Some(CallHandler(Handler(handler))) = tools.build(name, place) else {
            let report = ErrorReport::new(
                ErrorKind::Tool(ToolErrorKind::Other),
                format!("`{name}` cannot be previewed: it is not registered per call"),
            );
            commands.entity(call).insert(Previewed(Err(report)));
            continue;
        };
        let kind = EffectKind::from(&request.call);
        let work = async move {
            if let Answered::ToolResult { result } = serve_inline(&handler, kind).await? {
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
