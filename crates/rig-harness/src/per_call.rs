//! Tools whose every call is built for where it works
//! ([`AppPerCallExt::add_tool_per_call`]): the app's workspace, the calling
//! agent's [`WorkDir`] or else the [`ProjectRoot`], and, for a tool that
//! reports its output as it runs, where it reports it ([`Place`]).
//!
//! ```
//! use rig_harness::prelude::*;
//! use rig_harness::rig_tools::Write;
//!
//! let mut app = App::new();
//! app.add_tool_per_call(Write::default(), ToolOptions::default(), false, |place| Write {
//!     workspace: place.workspace,
//!     base: Some(place.base),
//!     confine: place.confine,
//! });
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use bevy_app::App;
use bevy_ecs::prelude::*;
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ToolAdapter;
use rig_ecs::effects::Handler;
use rig_ecs::prelude::{
    Admitted, AgentPass, AppToolsExt, CallHandler, ForAgent, PrepareSystems, Tool, ToolCalls,
    ToolOptions, ToolRequest, Wake,
};
use rig_tools::workspace::Workspace;
use rig_tools::{Live, Spill};

use crate::host::session::Scratch;
use crate::progress::CallOutput;
use crate::services::{ProjectRoot, WorkDir};
use crate::workspace::ProjectWorkspace;

/// Where one call of a tool registered with
/// [`add_tool_per_call`](AppPerCallExt::add_tool_per_call) works.
pub struct Place {
    /// The app's workspace ([`ProjectWorkspace`]), or an overlay of it for
    /// a call that is previewed ([`Preview`](crate::preview::Preview)).
    pub workspace: Arc<dyn Workspace>,
    /// The calling agent's [`WorkDir`], else the [`ProjectRoot`]: relative
    /// paths resolve and commands start here.
    pub base: PathBuf,
    /// Whether `base` is a `WorkDir`, outside which the call refuses paths.
    pub confine: bool,
    /// Where the call reports its output as it comes, for a tool
    /// registered `live`; read by the front as the call's [`CallOutput`].
    pub live: Option<Live>,
    /// Where output cut short is kept whole, if anywhere
    /// ([`Scratch::spill`]).
    pub spill: Option<Spill>,
}

/// Registering a tool whose every call is built for its [`Place`].
pub trait AppPerCallExt {
    /// Registers `tool` with `options`, as
    /// [`add_tool_with`](AppToolsExt::add_tool_with) does, and gives each of
    /// its calls the handler of `make(place)` in
    /// [`PrepareSystems::Compose`], so a plugin that wraps or replaces a
    /// call's [`CallHandler`] in [`PrepareSystems::Edit`] gets that one. With
    /// `live`, each call gets a [`CallOutput`] and its place the [`Live`]
    /// that fills it. A call of the tool can be previewed
    /// ([`Preview`](crate::preview::Preview)), as `make` builds it on an
    /// overlay of the workspace.
    fn add_tool_per_call<T: Tool + 'static>(
        &mut self,
        tool: T,
        options: ToolOptions<'_>,
        live: bool,
        make: impl Fn(Place) -> T + Send + Sync + 'static,
    ) -> &mut Self;
}

impl AppPerCallExt for App {
    fn add_tool_per_call<T: Tool + 'static>(
        &mut self,
        tool: T,
        options: ToolOptions<'_>,
        live: bool,
        make: impl Fn(Place) -> T + Send + Sync + 'static,
    ) -> &mut Self {
        if !self.world().contains_resource::<PerCall>() {
            self.init_resource::<PerCall>()
                .init_resource::<ProjectRoot>()
                .init_resource::<ProjectWorkspace>()
                .add_systems(AgentPass, give_place.in_set(PrepareSystems::Compose));
        }
        let make = move |place: Place| Handler(ErasedHandler::new(ToolAdapter::new(make(place))));
        if let Some(mut tools) = self.world_mut().get_resource_mut::<PerCall>() {
            tools.0.push(Made {
                name: T::NAME,
                live,
                make: Box::new(make),
            });
        }
        self.add_tool_with(tool, options)
    }
}

/// How a call of one tool is built for its [`Place`].
struct Made {
    name: &'static str,
    /// Whether its calls report their output as they run.
    live: bool,
    make: Box<dyn Fn(Place) -> Handler + Send + Sync>,
}

/// The tools registered per call.
#[derive(Resource, Default)]
pub(crate) struct PerCall(Vec<Made>);

impl PerCall {
    /// The handler of a call of `tool` built for `place`, when `tool` is
    /// registered per call.
    pub(crate) fn build(&self, tool: &str, place: Place) -> Option<Handler> {
        self.made(tool).map(|made| (made.make)(place))
    }

    fn made(&self, tool: &str) -> Option<&Made> {
        self.0.iter().find(|made| made.name == tool)
    }
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
/// tool built for its [`Place`], and a live call its [`CallOutput`].
fn give_place(
    mut calls: Query<
        (Entity, &ToolRequest, Option<&ForAgent>, &mut CallHandler),
        (ToolCalls, Added<Admitted>),
    >,
    dirs: Query<&WorkDir>,
    tools: Res<PerCall>,
    (workspace, root, scratch): (
        Res<ProjectWorkspace>,
        Res<ProjectRoot>,
        Option<Res<Scratch>>,
    ),
    wake: Res<Wake>,
    mut commands: Commands,
) {
    for (call, request, agent, mut handler) in &mut calls {
        let Some(made) = tools.made(request.call.function.name.as_str()) else {
            continue;
        };
        let live = made.live.then(|| {
            let (output, live) = CallOutput::new(&wake);
            commands.entity(call).insert(output);
            live
        });
        let (base, confine) = base(agent.map(|agent| agent.0), &dirs, &root);
        let place = Place {
            workspace: workspace.0.clone(),
            base,
            confine,
            live,
            spill: scratch.as_ref().and_then(|scratch| scratch.spill()),
        };
        *handler = CallHandler((made.make)(place));
    }
}
