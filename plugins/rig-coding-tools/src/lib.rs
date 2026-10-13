//! The built-in tools of [`rig_tools`], one plugin each, with their rules
//! on when to pick them: `read` and `search` run beside each other, `edit`,
//! `write` and `shell` alone; `delete` and `move` ([`DeleteTool`],
//! [`MoveTool`]) are there to add, gated as writes. Every call works in the
//! app's [`ProjectWorkspace`], from its agent's [`WorkDir`], else from the
//! [`ProjectRoot`]: its paths resolve from there, a command starts there,
//! and with a `WorkDir` a path outside it is refused
//! ([`PrepareSystems::Compose`]). A `shell` call reports its output as it
//! comes ([`CallOutput`]). With the `tui` feature (on by default), each
//! plugin also adds how the terminal view draws its tool's calls.
//! [`AttachPlugin`] sends the files the user names as `@path` with the
//! message.

use std::path::PathBuf;
use std::sync::{Arc, PoisonError, RwLock};

use bevy_tasks::{IoTaskPool, TaskPool};
use rig_harness::prelude::{
    Added, Admitted, AgentPass, App, AppToolsExt, Attachment, AuthorKind, CallHandler, CallOutput,
    Commands, Component, Entity, ForAgent, HoldOf, Incoming, Insert, IntoScheduleConfigs,
    MessagePart, MessageWriter, Name, Notice, On, Plugin, PrepareSystems, Processes, ProjectRoot,
    ProjectWorkspace, Query, Res, Resource, Running, Scratch, Spill, SystemSet, Tool, ToolCalls,
    ToolOptions, ToolRequest, Wake, WorkDir, blocking,
};
use rig_harness::rig_core::completion::ToolAnnotations;
use rig_harness::rig_core::serve::ErasedHandler;
use rig_harness::rig_core::serve::adapters::ToolAdapter;
use rig_harness::rig_core::tool::{ToolContext, ToolExecutionError};
use rig_harness::rig_ecs::effects::Handler;
use rig_tools::attach::{Attached, attachments};
use rig_tools::workspace::Workspace;
use rig_tools::{Cut, Delete, Edit, Live, Move, Read, Search, Shell, Write};
#[cfg(feature = "tui")]
use rig_tui::AppToolRenderersExt;

#[cfg(feature = "tui")]
mod looks;

/// Options with `rules`; what each call touches, so calls on different
/// files run side by side, the tool says: `read` and `search` read the
/// path they name, `edit`, `write`, `delete` and `move` write theirs,
/// `shell` may change anything.
fn options(rules: &'static [&'static str]) -> ToolOptions<'static> {
    ToolOptions {
        rules,
        ..ToolOptions::default()
    }
}

/// The `read` tool.
#[derive(Default)]
pub struct ReadTool;

impl Plugin for ReadTool {
    fn build(&self, app: &mut App) {
        app.add_tool_with(Read::default(), options(Read::RULES));
        per_call(app, false, |place| Read {
            workspace: place.workspace,
            base: Some(place.base),
            confine: place.confine,
        });
        #[cfg(feature = "tui")]
        app.add_tool_renderer(Read::NAME, looks::read);
    }
}

/// The `search` tool.
#[derive(Default)]
pub struct SearchTool;

impl Plugin for SearchTool {
    fn build(&self, app: &mut App) {
        app.add_tool_with(Search::default(), options(Search::RULES));
        per_call(app, false, |place| Search {
            workspace: place.workspace,
            base: Some(place.base),
            confine: place.confine,
        });
        #[cfg(feature = "tui")]
        app.add_tool_renderer(Search::NAME, looks::search);
    }
}

/// The `edit` tool.
#[derive(Default)]
pub struct EditTool;

impl Plugin for EditTool {
    fn build(&self, app: &mut App) {
        app.add_tool_with(Edit::default(), options(Edit::RULES));
        per_call(app, false, |place| Edit {
            workspace: place.workspace,
            base: Some(place.base),
            confine: place.confine,
        });
        #[cfg(feature = "tui")]
        app.add_tool_renderer(Edit::NAME, looks::edit);
    }
}

/// The `write` tool.
#[derive(Default)]
pub struct WriteTool;

impl Plugin for WriteTool {
    fn build(&self, app: &mut App) {
        app.add_tool_with(Write::default(), options(Write::RULES));
        per_call(app, false, |place| Write {
            workspace: place.workspace,
            base: Some(place.base),
            confine: place.confine,
        });
        #[cfg(feature = "tui")]
        app.add_tool_renderer(Write::NAME, looks::write);
    }
}

/// The `delete` tool: removes a file, gated as a write of it. Not in the
/// default plugins; add it where the model may delete files.
#[derive(Default)]
pub struct DeleteTool;

impl Plugin for DeleteTool {
    fn build(&self, app: &mut App) {
        app.add_tool_with(Delete::default(), options(Delete::RULES));
        per_call(app, false, |place| Delete {
            workspace: place.workspace,
            base: Some(place.base),
            confine: place.confine,
        });
        #[cfg(feature = "tui")]
        app.add_tool_renderer(Delete::NAME, looks::delete);
    }
}

/// The `move` tool: moves or renames a file, gated as a write of both
/// paths. Not in the default plugins; add it where the model may move
/// files.
#[derive(Default)]
pub struct MoveTool;

impl Plugin for MoveTool {
    fn build(&self, app: &mut App) {
        app.add_tool_with(Move::default(), options(Move::RULES));
        per_call(app, false, |place| Move {
            workspace: place.workspace,
            base: Some(place.base),
            confine: place.confine,
        });
        #[cfg(feature = "tui")]
        app.add_tool_renderer(Move::NAME, looks::moved);
    }
}

/// The `shell` tool. Each call starts its process as the app's
/// [`Processes`] say when it runs, so a sandbox plugin's policy applies,
/// in its agent's [`WorkDir`], else in the [`ProjectRoot`] unless the
/// `Processes` name a directory; it cuts its output as the app's
/// [`ShellOutput`] says then, and output it cuts is kept whole in the
/// [`spill`](Scratch::spill) directory of the session's [`Scratch`] then. What it prints
/// becomes the call's [`ToolProgress`](rig_harness::prelude::ToolProgress)
/// as it comes, and its answer if it is stopped ([`CallOutput`]).
#[derive(Default)]
pub struct ShellTool;

impl Plugin for ShellTool {
    fn build(&self, app: &mut App) {
        let output = app
            .world_mut()
            .get_resource_or_init::<ShellOutput>()
            .clone();
        let shell = SharedShell {
            shell: Shell::default(),
            processes: app.world_mut().get_resource_or_init::<Processes>().clone(),
            output,
            dir: None,
        };
        let per = shell.clone();
        per_call(app, true, move |place| SharedShell {
            shell: Shell {
                workspace: place.workspace,
                live: place.live,
                confine: place.confine,
                spill: place.spill,
                ..per.shell.clone()
            },
            dir: Some(place.base),
            ..per.clone()
        });
        app.add_tool_with(shell, options(Shell::RULES));
        #[cfg(feature = "tui")]
        app.add_tool_renderer(Shell::NAME, looks::shell);
    }
}

/// How much of a command's output the `shell` tool gives the model, read
/// as each call runs: [`Cut::default`] unless set. A plugin that cuts or
/// summarizes output itself turns the cut off with `set(None)`, such as
/// from its `finish`; the command is still killed past 10 MB of output.
/// Shared by its clones: change it with [`set`](Self::set), not by
/// inserting a new one.
#[derive(Resource, Clone, Debug)]
pub struct ShellOutput(Arc<RwLock<Option<Cut>>>);

impl ShellOutput {
    /// How the next call cuts its output; `None` keeps all of it.
    pub fn cut(&self) -> Option<Cut> {
        *self.0.read().unwrap_or_else(PoisonError::into_inner)
    }

    /// Cuts every call's output from now on as `cut` says.
    pub fn set(&self, cut: Option<Cut>) {
        *self.0.write().unwrap_or_else(PoisonError::into_inner) = cut;
    }
}

impl Default for ShellOutput {
    fn default() -> Self {
        Self(Arc::new(RwLock::new(Some(Cut::default()))))
    }
}

/// [`Shell`], starting each command as [`Processes`] say at the time: in
/// `dir` when its shell is confined there or the `Processes` name no
/// directory, and cutting its output as [`ShellOutput`] says.
#[derive(Clone)]
struct SharedShell {
    shell: Shell,
    processes: Processes,
    output: ShellOutput,
    dir: Option<PathBuf>,
}

impl Tool for SharedShell {
    const NAME: &'static str = Shell::NAME;
    type Args = <Shell as Tool>::Args;
    type Output = <Shell as Tool>::Output;
    type Error = <Shell as Tool>::Error;

    fn description(&self) -> String {
        self.shell.description()
    }

    fn parameters(&self) -> rig_harness::rig_core::serde_json::Value {
        self.shell.parameters()
    }

    fn annotations(&self) -> ToolAnnotations {
        self.shell.annotations()
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let mut launcher = self.processes.launcher();
        if let Some(dir) = &self.dir
            && (self.shell.confine || launcher.cwd.is_none())
        {
            launcher = launcher.in_dir(dir);
        }
        let shell = Shell {
            launcher,
            cut: self.output.cut(),
            ..self.shell.clone()
        };
        shell.call(context, args).await
    }
}

/// Where a call of these tools works: the app's workspace, its agent's
/// [`WorkDir`] or else the [`ProjectRoot`] as its base, confined to the
/// base when it is a `WorkDir`, and, for a tool that reports its output as
/// it runs, where it reports it.
struct Place {
    workspace: Arc<dyn Workspace>,
    base: PathBuf,
    confine: bool,
    live: Option<Live>,
    /// Where output cut short is kept whole, if anywhere.
    spill: Option<Spill>,
}

/// How a call of one of these tools is built for its [`Place`].
struct Made {
    name: &'static str,
    /// Whether its calls report their output as they run.
    live: bool,
    make: Box<dyn Fn(Place) -> Handler + Send + Sync>,
}

/// These tools, by name.
#[derive(Resource, Default)]
struct PerCall(Vec<Made>);

/// Builds each call of `T` for its [`Place`], as `make` says, with a
/// [`Live`] for its output when `live`.
fn per_call<T: Tool + 'static>(
    app: &mut App,
    live: bool,
    make: impl Fn(Place) -> T + Send + Sync + 'static,
) {
    if !app.world().contains_resource::<PerCall>() {
        app.init_resource::<PerCall>()
            .init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .add_systems(AgentPass, give_place.in_set(PrepareSystems::Compose));
    }
    let make = move |place: Place| Handler(ErasedHandler::new(ToolAdapter::new(make(place))));
    if let Some(mut tools) = app.world_mut().get_resource_mut::<PerCall>() {
        tools.0.push(Made {
            name: T::NAME,
            live,
            make: Box::new(make),
        });
    }
}

/// Gives each call of these tools admitted this pass the handler of its
/// tool built for its [`Place`], and a shell call its [`CallOutput`], in
/// [`PrepareSystems::Compose`], so a plugin that wraps or replaces a call's
/// [`CallHandler`] in [`PrepareSystems::Edit`] gets that one.
fn give_place(
    mut calls: Query<
        (Entity, &ToolRequest, &ForAgent, &mut CallHandler),
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
    for (call, request, &ForAgent(agent), mut handler) in &mut calls {
        let name = request.call.function.name.as_str();
        let Some(made) = tools.0.iter().find(|made| made.name == name) else {
            continue;
        };
        let dir = dirs.get(agent).ok();
        let live = made.live.then(|| {
            let (output, live) = CallOutput::new(&wake);
            commands.entity(call).insert(output);
            live
        });
        let place = Place {
            workspace: workspace.0.clone(),
            base: dir.map_or_else(|| root.0.clone(), |dir| dir.0.clone()),
            confine: dir.is_some(),
            live,
            spill: scratch.as_ref().and_then(|scratch| scratch.spill()),
        };
        *handler = CallHandler((made.make)(place));
    }
}

/// Files the user names as `@path` go with the message, before its text,
/// and the `@path` stays in the text so the model knows which file it is
/// ([`rig_tools::attach`]): an image as an image, which the core sends
/// only to a model that reads images, telling the user otherwise; any
/// other file as its text, numbered and capped the way `read` shows it.
/// Paths resolve from the agent's [`WorkDir`], else the [`ProjectRoot`],
/// in the app's [`ProjectWorkspace`]; a path with no file there is warned
/// about. It runs in [`AttachSystems`].
#[derive(Default)]
pub struct AttachPlugin;

/// Where [`AttachPlugin`] reads the files a user's message names into its
/// attachments, in [`PrepareSystems::Compose`]: a plugin that rewrites the
/// text goes before, in `Input`; one that reads or edits the attachments
/// goes `.after(AttachSystems)`. In a workspace that is not this machine,
/// the files are read off the main thread under a hold on the message, so
/// they come later than this set.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AttachSystems;

impl Plugin for AttachPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .configure_sets(AgentPass, AttachSystems.in_set(PrepareSystems::Compose))
            .add_systems(AgentPass, attach.in_set(AttachSystems))
            .add_observer(attach_read);
    }
}

/// The files a message names, read off the main thread, on the hold that
/// keeps the message ([`HoldOf`]) until they are attached.
#[derive(Component)]
struct ReadFiles(Result<Attached, ToolExecutionError>);

impl From<Result<Attached, ToolExecutionError>> for ReadFiles {
    fn from(read: Result<Attached, ToolExecutionError>) -> Self {
        Self(read)
    }
}

/// Reads the files each user's message names into its attachments as it
/// comes in, with a notice about each that could not be, and a warning
/// about each that is not there: inline in this machine's workspace, else
/// as work off the main thread that holds the message.
fn attach(
    mut typed: Query<(Entity, &mut Incoming, &ForAgent), Added<Admitted>>,
    dirs: Query<&WorkDir>,
    workspace: Res<ProjectWorkspace>,
    root: Res<ProjectRoot>,
    wake: Res<Wake>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    for (message, mut typed, &ForAgent(agent)) in &mut typed {
        if typed.author.kind != AuthorKind::User || typed.command().is_some() {
            continue;
        }
        let base = dirs.get(agent).map_or(&root.0, |dir| &dir.0).clone();
        if workspace.0.host().is_none() {
            let attached = attachments(typed.text(), &base, &*workspace.0);
            add_attached(&mut typed, agent, attached, &mut notices);
            continue;
        }
        let (text, workspace) = (typed.text().to_owned(), workspace.0.clone());
        let read = blocking(move || Ok(attachments(&text, &base, &*workspace)));
        let pool = IoTaskPool::get_or_init(TaskPool::default);
        commands.spawn((
            Name::new("attachments"),
            HoldOf(message),
            Running::spawn_into::<ReadFiles, _>(pool, &wake, read),
        ));
    }
}

/// Attaches the files read off the main thread to the message they were
/// read for, and releases it.
fn attach_read(
    read: On<Insert<ReadFiles>>,
    holds: Query<(&HoldOf, &ReadFiles)>,
    mut messages: Query<(&mut Incoming, &ForAgent)>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Ok((&HoldOf(message), ReadFiles(attached))) = holds.get(read.entity) else {
        return;
    };
    if let Ok((mut typed, &ForAgent(agent))) = messages.get_mut(message) {
        match attached {
            Ok(attached) => add_attached(&mut typed, agent, attached.clone(), &mut notices),
            Err(failure) => {
                notices.write(Notice::warn(agent, format!("No files attached: {failure}")));
            }
        }
    }
    commands.entity(read.entity).despawn();
}

/// Puts `attached` before the message's text, with a notice about each
/// file that could not be attached and a warning about each missing one.
fn add_attached(
    typed: &mut Incoming,
    agent: Entity,
    attached: Attached,
    notices: &mut MessageWriter<Notice>,
) {
    for note in attached.notes {
        notices.write(Notice::info(agent, note));
    }
    for missing in attached.missing {
        notices.write(Notice::warn(agent, format!("@{missing}: no such file")));
    }
    // Before the text, which names them.
    let attachments = attached
        .files
        .into_iter()
        .map(|(label, content)| MessagePart::Attachment(Attachment { label, content }));
    typed.parts.splice(0..0, attachments);
}

#[cfg(test)]
mod tests;
