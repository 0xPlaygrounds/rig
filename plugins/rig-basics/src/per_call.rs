//! Tools whose every call is built for where it works
//! ([`AppPerCallExt::add_tool_per_call`]): the app's workspace, the calling
//! agent's [`WorkDir`] or else the [`ProjectRoot`], and, for a tool that
//! reports its output as it runs, where it reports it ([`CallPlace`]).
//!
//! ```
//! use rig_basics::per_call::AppPerCallExt;
//! use rig_harness::prelude::*;
//! use rig_harness::rig_tools::Write;
//!
//! let mut app = App::new();
//! app.add_tool_per_call(Write::default(), ToolOptions::default(), |place| Write {
//!     workspace: place.workspace,
//!     base: Some(place.base),
//!     confine: place.confine,
//! });
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use bevy_app::{App, Plugin};
use bevy_ecs::prelude::*;
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ToolAdapter;
use rig_core::tool::{Touches, tool_definition};
use rig_ecs::effects::Handler;
use rig_ecs::prelude::{
    Admitted, AgentPass, CallHandler, Footprint, ForAgent, PrepareSystems, Tool, ToolCalls,
    ToolDef, ToolOptions, ToolRequest, Wake, WorldToolsExt,
};
use rig_tools::workspace::Workspace;
use rig_tools::{Live, Spill};

use rig_harness::host::session::Scratch;
use rig_harness::progress::CallOutput;
use rig_harness::services::{ProjectRoot, WorkDir};
use rig_harness::workspace::ProjectWorkspace;

/// Where one call of a tool registered with
/// [`add_tool_per_call`](AppPerCallExt::add_tool_per_call) works.
pub struct CallPlace {
    /// The app's workspace ([`ProjectWorkspace`]), or an overlay of it for
    /// a call that is previewed ([`Preview`](crate::preview::Preview)).
    pub workspace: Arc<dyn Workspace>,
    /// The calling agent's [`WorkDir`], else the [`ProjectRoot`]: relative
    /// paths resolve and commands start here.
    pub base: PathBuf,
    /// Whether `base` is a `WorkDir`, outside which the call refuses paths.
    pub confine: bool,
    /// Where the call reports its output as it comes, for a tool
    /// registered with [`add_live_tool_per_call`](AppPerCallExt::add_live_tool_per_call);
    /// read by the front as the call's [`CallOutput`].
    pub live: Option<Live>,
    /// Where output cut short is kept whole, if anywhere
    /// ([`Scratch::spill`]).
    pub spill: Option<Spill>,
}

/// Registering a tool whose every call is built for its [`CallPlace`].
pub trait AppPerCallExt {
    /// Registers `tool` with `options`, as
    /// [`add_tool_with`](rig_ecs::tools::AppToolsExt::add_tool_with) does, and
    /// gives each of its calls the handler of `make(place)` in
    /// [`PrepareSystems::Compose`], so a plugin that wraps or replaces a call's
    /// [`CallHandler`] in [`PrepareSystems::Edit`] gets that one. A call of the
    /// tool can be previewed ([`Preview`](crate::preview::Preview)), as `make`
    /// builds it on an overlay of the workspace.
    fn add_tool_per_call<T: Tool + 'static>(
        &mut self,
        tool: T,
        options: ToolOptions,
        make: impl Fn(CallPlace) -> T + Send + Sync + 'static,
    ) -> &mut Self;

    /// As [`add_tool_per_call`](Self::add_tool_per_call), for a tool that
    /// reports its output as it runs: each call gets a [`CallOutput`], and
    /// its place the [`Live`] that fills it.
    fn add_live_tool_per_call<T: Tool + 'static>(
        &mut self,
        tool: T,
        options: ToolOptions,
        make: impl Fn(CallPlace) -> T + Send + Sync + 'static,
    ) -> &mut Self;
}

impl AppPerCallExt for App {
    fn add_tool_per_call<T: Tool + 'static>(
        &mut self,
        tool: T,
        options: ToolOptions,
        make: impl Fn(CallPlace) -> T + Send + Sync + 'static,
    ) -> &mut Self {
        add_per_call(self, tool, options, false, make)
    }

    fn add_live_tool_per_call<T: Tool + 'static>(
        &mut self,
        tool: T,
        options: ToolOptions,
        make: impl Fn(CallPlace) -> T + Send + Sync + 'static,
    ) -> &mut Self {
        add_per_call(self, tool, options, true, make)
    }
}

fn add_per_call<T: Tool + 'static>(
    app: &mut App,
    tool: T,
    options: ToolOptions,
    live: bool,
    make: impl Fn(CallPlace) -> T + Send + Sync + 'static,
) -> &mut App {
    if !app.is_plugin_added::<PerCallPlugin>() {
        app.add_plugins(PerCallPlugin);
    }
    let definition = tool_definition(&tool);
    // The annotations are the tool's own code's, so they give its default
    // footprint, as `add_tool_with` takes them, not as hints.
    let own = Touches::from_annotations(&definition.annotations);
    let own = options.footprint.is_none().then_some(Footprint(own));
    let handler = ErasedHandler::new(ToolAdapter::new(tool));
    let make = move |place: CallPlace| Handler(ErasedHandler::new(ToolAdapter::new(make(place))));
    let world = app.world_mut();
    if let Some(tool) = world.spawn_tool(definition, handler, options) {
        let make = Box::new(make);
        let mut tool = world.entity_mut(tool);
        tool.insert(PerCall { live, make });
        if let Some(own) = own {
            tool.insert(own);
        }
    }
    app
}

/// Builds the calls of the tools registered per call.
struct PerCallPlugin;

impl Plugin for PerCallPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .add_systems(
                AgentPass,
                build_per_call_handlers.in_set(PrepareSystems::Compose),
            );
    }
}

/// On a tool's entity: how each of its calls is built for its [`CallPlace`].
/// A replacement disables the tool's entity, and this with it.
#[derive(Component)]
pub(crate) struct PerCall {
    /// Whether its calls report their output as they run.
    live: bool,
    make: Box<dyn Fn(CallPlace) -> Handler + Send + Sync>,
}

impl PerCall {
    /// The handler of a call built for `place`.
    pub(crate) fn build(&self, place: CallPlace) -> Handler {
        (self.make)(place)
    }
}

/// The per-call tool named `name`, of `tools`.
pub(crate) fn per_call<'a>(
    tools: &'a Query<(&ToolDef, &PerCall)>,
    name: &str,
) -> Option<&'a PerCall> {
    tools
        .iter()
        .find(|(tool, _)| tool.0.name == name)
        .map(|(_, per_call)| per_call)
}

/// The base of a call of `agent`'s, and whether it confines the call: its
/// [`WorkDir`], else the [`ProjectRoot`].
pub(crate) fn base(
    agent: Option<Entity>,
    dirs: &Query<&WorkDir>,
    root: &ProjectRoot,
) -> (PathBuf, bool) {
    match agent.and_then(|agent| dirs.get(agent).ok()) {
        Some(dir) => (dir.0.clone(), true),
        None => (root.0.clone(), false),
    }
}

/// Gives each call of these tools admitted this pass the handler of its
/// tool built for its [`CallPlace`], and a live call its [`CallOutput`].
fn build_per_call_handlers(
    mut calls: Query<
        (Entity, &ToolRequest, Option<&ForAgent>, &mut CallHandler),
        (ToolCalls, Added<Admitted>),
    >,
    dirs: Query<&WorkDir>,
    tools: Query<(&ToolDef, &PerCall)>,
    (workspace, root, scratch): (
        Res<ProjectWorkspace>,
        Res<ProjectRoot>,
        Option<Res<Scratch>>,
    ),
    wake: Res<Wake>,
    mut commands: Commands,
) {
    for (call, request, agent, mut handler) in &mut calls {
        let Some(per_call) = per_call(&tools, request.call.function.name.as_str()) else {
            continue;
        };
        let live = per_call.live.then(|| {
            let (output, live) = CallOutput::new(&wake);
            commands.entity(call).insert(output);
            live
        });
        let (base, confine) = base(agent.map(|agent| agent.0), &dirs, &root);
        let place = CallPlace {
            workspace: workspace.0.clone(),
            base,
            confine,
            live,
            spill: scratch.as_ref().and_then(|scratch| scratch.spill()),
        };
        *handler = CallHandler(per_call.build(place));
    }
}

#[cfg(test)]
mod tests;
