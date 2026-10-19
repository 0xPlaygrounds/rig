//! Tools whose every call is built for where it works
//! ([`AppPerCallExt::add_tool_per_call`]): the app's workspace, the calling
//! agent's [`WorkDir`] or else the [`ProjectRoot`], and, for a tool that
//! reports its output as it runs, where it reports it ([`CallPlace`]). A
//! plugin builds such a call somewhere else, such as on an overlay of the
//! workspace, with [`PerCallTools`].
//!
//! ```
//! use rig_harness::prelude::*;
//! use rig_harness::rig_tools::Write;
//!
//! let mut app = App::new();
//! app.add_tool_per_call(Write::default(), ToolOptions::default(), |place| Write {
//!     workspace: place.workspace,
//!     base: Some(place.base),
//!     confine: place.confine,
//!     ..Write::default()
//! });
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use bevy_app::{App, Plugin};
use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParam;
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

use crate::host::session::Scratch;
use crate::progress::CallOutput;
use crate::services::{ProjectRoot, WorkDir};
use crate::workspace::ProjectWorkspace;

/// Where one call of a tool registered with
/// [`add_tool_per_call`](AppPerCallExt::add_tool_per_call) works.
pub struct CallPlace {
    /// The app's workspace ([`ProjectWorkspace`]), or the one a plugin
    /// builds the call on with [`PerCallTools`], such as an overlay of it
    /// for a call that is previewed (rig-basics' `Preview`).
    pub workspace: Arc<dyn Workspace>,
    /// The calling agent's [`WorkDir`], else the [`ProjectRoot`]: relative
    /// paths resolve and commands start here.
    pub base: PathBuf,
    /// Whether `base` is a `WorkDir`, outside which the call refuses paths.
    pub confine: bool,
    /// Where the call reports its output as it comes, for a tool
    /// registered with [`add_live_tool_per_call`](AppPerCallExt::add_live_tool_per_call);
    /// read by the front as the call's [`CallOutput`]. `None` from
    /// [`PerCallTools::place`]; the harness sets it for a call it runs.
    pub live: Option<Live>,
    /// Where output cut short is kept whole, if anywhere
    /// ([`Scratch::spill`]). `None` from [`PerCallTools::place`]; the
    /// harness sets it for a call it runs.
    pub spill: Option<Spill>,
}

/// Registering a tool whose every call is built for its [`CallPlace`].
pub trait AppPerCallExt {
    /// Registers `tool` with `options`, as
    /// [`add_tool_with`](rig_ecs::tools::AppToolsExt::add_tool_with) does, and
    /// gives each of its calls the handler of `make(place)` in
    /// [`PrepareSystems::Compose`], so a plugin that wraps or replaces a call's
    /// [`CallHandler`] in [`PrepareSystems::Edit`] gets that one. A call of the
    /// tool can be previewed (rig-basics' `Preview`), as `make`
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
    let make = move |place: CallPlace| {
        CallHandler(Handler(ErasedHandler::new(ToolAdapter::new(make(place)))))
    };
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
struct PerCall {
    /// Whether its calls report their output as they run.
    live: bool,
    make: Box<dyn Fn(CallPlace) -> CallHandler + Send + Sync>,
}

/// The tools registered per call, and where a call of an agent's works:
/// for a plugin that builds a call somewhere else, as rig-basics' `Preview`
/// builds it on an overlay of the workspace, or wraps a call where it
/// works, as its `Checkpoint` does. The plugin inits [`ProjectRoot`], which
/// this reads: without it the system using this fails. A tool replaced
/// (its entity `Disabled`) is not seen.
///
/// ```
/// use std::sync::Arc;
///
/// use bevy_ecs::system::RunSystemOnce;
/// use rig_harness::prelude::*;
/// use rig_harness::rig_core::effect::EffectKind;
/// use rig_harness::rig_core::message::{ToolFunction, ToolName};
/// use rig_harness::rig_core::serve::serve_inline;
/// use rig_harness::rig_tools::{Write, overlay::Overlay};
///
/// let mut app = App::new();
/// app.add_tool_per_call(Write::default(), ToolOptions::default(), |place| Write {
///     workspace: place.workspace,
///     base: Some(place.base),
///     confine: place.confine,
///     ..Write::default()
/// });
/// // It exists; nothing is written to it.
/// let dir = std::env::current_dir()?;
/// let agent = app.world_mut().spawn(WorkDir(dir.clone())).id();
/// let project = app.world().resource::<ProjectWorkspace>().0.clone();
/// let overlay = Arc::new(Overlay::new(project));
/// let on = overlay.clone();
/// let ran = app.world_mut().run_system_once(move |tools: PerCallTools| {
///     let place = tools.place(Some(agent), on.clone());
///     let (base, confine) = (place.base.clone(), place.confine);
///     let quiet = place.live.is_none() && place.spill.is_none();
///     let missing = tools.build("read", tools.place(None, on.clone()));
///     (tools.build("write", place), missing, base, confine, quiet)
/// });
/// let (write, missing, base, confine, quiet) = ran.map_err(|failure| failure.to_string())?;
/// assert_eq!(base, dir);
/// assert!(confine && quiet && missing.is_none());
/// let CallHandler(handler) = write.ok_or("write is registered per call")?;
/// let args = serde_json::json!({ "path": "per-call-doc.txt", "content": "one\n" });
/// let call = ToolCall::from_wire("w1", ToolFunction::new(ToolName::new("write")?, args));
/// bevy_tasks::block_on(serve_inline(&handler.0, EffectKind::from(&call)))?;
/// // The write went to the overlay, and not to the disk.
/// assert_eq!(overlay.changes().len(), 1);
/// assert!(!dir.join("per-call-doc.txt").exists());
/// # Ok::<_, Box<dyn std::error::Error>>(())
/// ```
#[derive(SystemParam)]
pub struct PerCallTools<'w, 's> {
    tools: Query<'w, 's, (&'static ToolDef, &'static PerCall)>,
    dirs: Query<'w, 's, &'static WorkDir>,
    root: Res<'w, ProjectRoot>,
}

impl PerCallTools<'_, '_> {
    /// A call of `agent`'s working in `workspace`: its base is the agent's
    /// [`WorkDir`], which confines it, else the [`ProjectRoot`]. It has no
    /// `live` and no `spill`, so a call built on an overlay writes nowhere
    /// else; a call that runs for real elsewhere sets `spill` itself
    /// ([`Scratch::spill`]). `confine` limits the file tools' paths; it is
    /// no process sandbox.
    pub fn place(&self, agent: Option<Entity>, workspace: Arc<dyn Workspace>) -> CallPlace {
        let dir = agent.and_then(|agent| self.dirs.get(agent).ok());
        CallPlace {
            workspace,
            base: dir.map_or(&self.root.0, |dir| &dir.0).clone(),
            confine: dir.is_some(),
            live: None,
            spill: None,
        }
    }

    /// The handler of a call of the per-call tool `name` built for
    /// `place`; `None` when no enabled tool of that name is registered per
    /// call. It runs nothing.
    pub fn build(&self, name: &str, place: CallPlace) -> Option<CallHandler> {
        self.find(name).map(|tool| (tool.make)(place))
    }

    /// The per-call tool named `name`.
    fn find(&self, name: &str) -> Option<&PerCall> {
        self.tools
            .iter()
            .find(|(tool, _)| tool.0.name == name)
            .map(|(_, per_call)| per_call)
    }
}

/// Gives each call of these tools admitted this pass the handler of its
/// tool built for its [`CallPlace`], and a live call its [`CallOutput`].
fn build_per_call_handlers(
    mut calls: Query<
        (Entity, &ToolRequest, Option<&ForAgent>, &mut CallHandler),
        (ToolCalls, Added<Admitted>),
    >,
    tools: PerCallTools,
    (workspace, scratch): (Res<ProjectWorkspace>, Option<Res<Scratch>>),
    wake: Res<Wake>,
    mut commands: Commands,
) {
    for (call, request, agent, mut handler) in &mut calls {
        let Some(tool) = tools.find(request.call.function.name.as_str()) else {
            continue;
        };
        let live = tool.live.then(|| {
            let (output, live) = CallOutput::new(&wake);
            commands.entity(call).insert(output);
            live
        });
        let place = CallPlace {
            live,
            spill: scratch.as_ref().and_then(|scratch| scratch.spill()),
            ..tools.place(agent.map(|agent| agent.0), workspace.0.clone())
        };
        *handler = (tool.make)(place);
    }
}

#[cfg(test)]
mod tests;
