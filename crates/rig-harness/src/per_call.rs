//! Tools whose every call is built for where it works
//! ([`AppPerCallExt::add_tool_per_call`]): the app's workspace, the calling
//! agent's [`WorkDir`] or else the [`ProjectRoot`], and, for a tool that
//! reports its output as it runs, where it reports it ([`CallPlace`]). Each
//! such call gets its [`CallPlace`] in [`PerCallSystems::Place`] and the
//! handler its tool builds there ([`PerCallTool::at`]) in
//! [`PerCallSystems::Build`], so a system between them changes where a call
//! works and one after replaces or wraps its [`CallHandler`]. A plugin
//! builds such a call somewhere else, such as on an overlay of the
//! workspace, with [`PerCallTools`].
//!
//! ```
//! use rig_harness::prelude::*;
//! use rig_harness::rig_tools::Write;
//!
//! let mut app = App::new();
//! app.add_tool_per_call(Write::default(), ToolOptions::default());
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
use rig_tools::{Delete, Edit, LineFilter, List, Live, Move, Read, Search, Shell, Spill, Write};

use crate::host::session::Scratch;
use crate::progress::CallOutput;
use crate::services::{ProjectRoot, WorkDir};
use crate::workspace::ProjectWorkspace;

/// Where one call of a tool registered with
/// [`add_tool_per_call`](AppPerCallExt::add_tool_per_call) works: on the
/// call from [`PerCallSystems::Place`], where a system may change it before
/// [`PerCallSystems::Build`] builds the call's handler for it.
#[derive(Component, Clone)]
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
    /// What each line of the call's output goes through first, such as a
    /// redaction plugin's, set before [`PerCallSystems::Build`]; `None`
    /// until a plugin sets it.
    pub filter: Option<LineFilter>,
    /// Where output cut short is kept whole, if anywhere
    /// ([`Scratch::spill`]). `None` from [`PerCallTools::place`]; the
    /// harness sets it for a call it runs.
    pub spill: Option<Spill>,
}

/// A tool built anew for each call, for where the call works.
pub trait PerCallTool: Tool + Clone + 'static {
    /// This tool, as a call at `place` runs it: `self` is the tool as
    /// registered, so what is not about the place comes from it.
    fn at(&self, place: CallPlace) -> Self;
}

/// The file tools work in the place's workspace, from its base, keeping
/// the fields named from the tool as registered.
macro_rules! file_tools_per_call {
    ($($tool:ty { $($keep:ident),* }),*) => {$(
        impl PerCallTool for $tool {
            fn at(&self, place: CallPlace) -> Self {
                Self {
                    workspace: place.workspace,
                    base: Some(place.base),
                    confine: place.confine,
                    $($keep: self.$keep.clone(),)*
                }
            }
        }
    )*};
}

file_tools_per_call!(
    Read { seen },
    Edit { seen },
    Write { seen },
    Search {},
    List {},
    Delete {},
    Move {}
);

impl PerCallTool for Shell {
    fn at(&self, place: CallPlace) -> Self {
        Self {
            launcher: self.launcher.clone().in_dir(&place.base),
            workspace: place.workspace,
            live: place.live,
            filter: place.filter,
            spill: place.spill,
            confine: place.confine,
            ..self.clone()
        }
    }
}

/// Registering a tool whose every call is built for its [`CallPlace`].
pub trait AppPerCallExt {
    /// Registers `tool` with `options`, as
    /// [`add_tool_with`](rig_ecs::tools::AppToolsExt::add_tool_with) does, and
    /// gives each of its calls the handler of `tool.at(place)` in
    /// [`PerCallSystems::Build`], so a plugin that wraps or replaces a call's
    /// [`CallHandler`] after it gets that one. A call of the tool can be
    /// previewed (rig-basics' `Preview`), as it is built on an overlay of
    /// the workspace.
    fn add_tool_per_call<T: PerCallTool>(&mut self, tool: T, options: ToolOptions) -> &mut Self;

    /// As [`add_tool_per_call`](Self::add_tool_per_call), for a tool that
    /// reports its output as it runs: each call gets a [`CallOutput`], and
    /// its place the [`Live`] that fills it.
    fn add_live_tool_per_call<T: PerCallTool>(
        &mut self,
        tool: T,
        options: ToolOptions,
    ) -> &mut Self;
}

impl AppPerCallExt for App {
    fn add_tool_per_call<T: PerCallTool>(&mut self, tool: T, options: ToolOptions) -> &mut Self {
        add_per_call(self, tool, options, false)
    }

    fn add_live_tool_per_call<T: PerCallTool>(
        &mut self,
        tool: T,
        options: ToolOptions,
    ) -> &mut Self {
        add_per_call(self, tool, options, true)
    }
}

fn add_per_call<T: PerCallTool>(
    app: &mut App,
    tool: T,
    options: ToolOptions,
    live: bool,
) -> &mut App {
    if !app.is_plugin_added::<PerCallPlugin>() {
        app.add_plugins(PerCallPlugin);
    }
    let definition = tool_definition(&tool);
    // The annotations are the tool's own code's, so they give its default
    // footprint, as `add_tool_with` takes them, not as hints.
    let own = Touches::from_annotations(&definition.annotations);
    let own = options.footprint.is_none().then_some(Footprint(own));
    let handler = ErasedHandler::new(ToolAdapter::new(tool.clone()));
    let tool = Arc::new(tool);
    let world = app.world_mut();
    if let Some(entity) = world.spawn_tool(definition, handler, options) {
        let mut entity = world.entity_mut(entity);
        entity.insert(PerCall { live, tool });
        if let Some(own) = own {
            entity.insert(own);
        }
    }
    app
}

/// The phases, in [`PrepareSystems::Compose`], in which each call of a tool
/// registered per call gets where it works, then its handler.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PerCallSystems {
    /// Each such call admitted this pass gets its [`CallPlace`], and a live
    /// one its [`CallOutput`].
    Place,
    /// Each such call admitted this pass gets the handler its tool builds
    /// for its [`CallPlace`].
    Build,
}

/// Builds the calls of the tools registered per call.
struct PerCallPlugin;

impl Plugin for PerCallPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .configure_sets(
                AgentPass,
                (PerCallSystems::Place, PerCallSystems::Build)
                    .chain()
                    .in_set(PrepareSystems::Compose),
            )
            .add_systems(
                AgentPass,
                (
                    place_per_call_calls.in_set(PerCallSystems::Place),
                    build_per_call_handlers.in_set(PerCallSystems::Build),
                ),
            );
    }
}

/// On a tool's entity registered per call: whether its calls report their
/// output as they run, and the tool each call is built from. A
/// replacement disables the tool's entity, and this with it.
#[derive(Component, Clone)]
pub struct PerCall {
    /// Whether its calls report their output as they run.
    pub live: bool,
    tool: Arc<dyn BuildAt>,
}

impl PerCall {
    /// The handler of a call of this tool working at `place`. It runs
    /// nothing.
    pub fn build(&self, place: CallPlace) -> CallHandler {
        self.tool.build(place)
    }
}

/// A [`PerCallTool`] whose type is erased.
trait BuildAt: Send + Sync {
    fn build(&self, place: CallPlace) -> CallHandler;
}

impl<T: PerCallTool> BuildAt for T {
    fn build(&self, place: CallPlace) -> CallHandler {
        CallHandler(Handler(ErasedHandler::new(ToolAdapter::new(
            self.at(place),
        ))))
    }
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
/// app.add_tool_per_call(Write::default(), ToolOptions::default());
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
            filter: None,
            spill: None,
        }
    }

    /// The handler of a call of the per-call tool `name` built for
    /// `place`; `None` when no enabled tool of that name is registered per
    /// call. It runs nothing.
    pub fn build(&self, name: &str, place: CallPlace) -> Option<CallHandler> {
        self.find(name).map(|tool| tool.build(place))
    }

    /// The per-call tool named `name`.
    fn find(&self, name: &str) -> Option<&PerCall> {
        self.tools
            .iter()
            .find(|(tool, _)| tool.0.name == name)
            .map(|(_, per_call)| per_call)
    }
}

/// Gives each call of these tools admitted this pass its [`CallPlace`],
/// and a live call its [`CallOutput`].
fn place_per_call_calls(
    calls: Query<(Entity, &ToolRequest, Option<&ForAgent>), (ToolCalls, Added<Admitted>)>,
    tools: PerCallTools,
    (workspace, scratch): (Res<ProjectWorkspace>, Option<Res<Scratch>>),
    wake: Res<Wake>,
    mut commands: Commands,
) {
    for (call, request, agent) in &calls {
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
        commands.entity(call).insert(place);
    }
}

/// Gives each call of these tools admitted this pass the handler of its
/// tool built for its [`CallPlace`].
fn build_per_call_handlers(
    mut calls: Query<(&ToolRequest, &CallPlace, &mut CallHandler), (ToolCalls, Added<Admitted>)>,
    tools: PerCallTools,
) {
    for (request, place, mut handler) in &mut calls {
        if let Some(built) = tools.build(request.call.function.name.as_str(), place.clone()) {
            *handler = built;
        }
    }
}

#[cfg(test)]
mod tests;
