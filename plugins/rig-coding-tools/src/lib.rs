//! The built-in tools of [`rig_tools`], one plugin each, with their rules
//! on when to pick them: `read` and `search` run beside each other, `edit`,
//! `write` and `shell` alone. A call of an agent with a [`WorkDir`] works
//! there: its paths resolve from it, a path outside it is refused, and a
//! command starts in it ([`PrepareSystems::Compose`]). With the `tui` feature
//! (on by default), each plugin also adds how the terminal view draws its
//! tool's calls. [`AttachPlugin`] sends the files the user names as
//! `@path` with the message.

use std::path::{Path, PathBuf};
use std::sync::{Arc, PoisonError, RwLock};

use rig_harness::prelude::{
    Added, Admitted, AgentPass, App, AppToolsExt, Attachment, CallHandler, ForAgent, Incoming,
    IntoScheduleConfigs, MessageWriter, Notice, OriginKind, Plugin, PrepareSystems, Processes,
    Query, Res, Resource, SessionPaths, Tool, ToolCalls, ToolOptions, ToolRequest, WorkDir,
};
use rig_harness::rig_core::completion::ToolAnnotations;
use rig_harness::rig_core::serve::ErasedHandler;
use rig_harness::rig_core::serve::adapters::ToolAdapter;
use rig_harness::rig_core::tool::ToolContext;
use rig_harness::rig_ecs::effects::Handler;
use rig_tools::{Cut, Edit, Read, Search, Shell, Write};
#[cfg(feature = "tui")]
use rig_tui::AppToolRenderersExt;

#[cfg(feature = "tui")]
mod looks;

/// Options with `rules`; what each call touches, so calls on different
/// files run side by side, the tool says: `read` and `search` read the
/// path they name, `edit` and `write` write it, `shell` may change
/// anything.
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
        work_in_dir(app, |base| Read {
            base: Some(base),
            confine: true,
            ..Read::default()
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
        work_in_dir(app, |base| Search {
            base: Some(base),
            confine: true,
            ..Search::default()
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
        work_in_dir(app, |base| Edit {
            base: Some(base),
            confine: true,
            ..Edit::default()
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
        work_in_dir(app, |base| Write {
            base: Some(base),
            confine: true,
            ..Write::default()
        });
        #[cfg(feature = "tui")]
        app.add_tool_renderer(Write::NAME, looks::write);
    }
}

/// The `shell` tool. Each call starts its process as the app's
/// [`Processes`] say when it runs, so a sandbox plugin's policy applies,
/// and cuts its output as the app's [`ShellOutput`] says then; output it
/// cuts is kept whole in the session's [`spill`](SessionPaths::spill)
/// directory.
#[derive(Default)]
pub struct ShellTool;

impl Plugin for ShellTool {
    fn build(&self, app: &mut App) {
        let output = app
            .world_mut()
            .get_resource_or_init::<ShellOutput>()
            .clone();
        let shell = SharedShell {
            shell: Shell {
                spill: app
                    .world()
                    .get_resource::<SessionPaths>()
                    .map(SessionPaths::spill),
                ..Shell::default()
            },
            processes: app.world_mut().get_resource_or_init::<Processes>().clone(),
            output,
            dir: None,
        };
        let in_dir = shell.clone();
        work_in_dir(app, move |dir| SharedShell {
            dir: Some(dir),
            ..in_dir.clone()
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

/// [`Shell`], starting each command as [`Processes`] say at the time, in
/// `dir` when set, where a `cwd` outside it is refused, and cutting its
/// output as [`ShellOutput`] says.
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
        let launcher = self.processes.launcher();
        let shell = Shell {
            launcher: match &self.dir {
                Some(dir) => launcher.in_dir(dir),
                None => launcher,
            },
            cut: self.output.cut(),
            confine: self.dir.is_some(),
            ..self.shell.clone()
        };
        shell.call(context, args).await
    }
}

/// How each of these tools works in a directory, by tool name.
#[derive(Resource, Default)]
struct InDir(Vec<(&'static str, Box<dyn Fn(&Path) -> Handler + Send + Sync>)>);

/// Lets `T`'s calls work in their agent's [`WorkDir`], as `make` builds
/// the tool for a directory.
fn work_in_dir<T: Tool + 'static>(
    app: &mut App,
    make: impl Fn(PathBuf) -> T + Send + Sync + 'static,
) {
    if !app.world().contains_resource::<InDir>() {
        app.init_resource::<InDir>()
            .add_systems(AgentPass, give_work_dirs.in_set(PrepareSystems::Compose));
    }
    let handler = move |dir: &Path| {
        Handler(ErasedHandler::new(ToolAdapter::new(make(
            dir.to_path_buf(),
        ))))
    };
    if let Some(mut tools) = app.world_mut().get_resource_mut::<InDir>() {
        tools.0.push((T::NAME, Box::new(handler)));
    }
}

/// Gives each call of these tools admitted this pass whose agent has a
/// [`WorkDir`] the handler of its tool working there, in
/// [`PrepareSystems::Compose`], so a plugin that wraps or replaces a call's
/// [`CallHandler`] in [`PrepareSystems::Edit`] gets that one.
fn give_work_dirs(
    mut calls: Query<(&ToolRequest, &ForAgent, &mut CallHandler), (ToolCalls, Added<Admitted>)>,
    dirs: Query<&WorkDir>,
    tools: Res<InDir>,
) {
    for (request, &ForAgent(agent), mut handler) in &mut calls {
        let Ok(WorkDir(dir)) = dirs.get(agent) else {
            continue;
        };
        let name = request.call.function.name.as_str();
        if let Some((_, make)) = tools.0.iter().find(|(tool, _)| *tool == name) {
            *handler = CallHandler(make(dir));
        }
    }
}

/// Files the user names as `@path` go with the message, before its text,
/// and the `@path` stays in the text so the model knows which file it is
/// ([`rig_tools::attach`]): an image as an image, which the core sends
/// only to a model that reads images, telling the user otherwise; any
/// other file as its text, numbered and capped the way `read` shows it.
#[derive(Default)]
pub struct AttachPlugin;

impl Plugin for AttachPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(AgentPass, attach.in_set(PrepareSystems::Compose));
    }
}

/// Reads the files each user's message names into its attachments as it
/// comes in, in [`PrepareSystems::Compose`], with a notice about each that
/// could not be: a plugin that rewrites the text goes before, in `Input`,
/// one that edits the text or attachments after, in `Edit`.
fn attach(
    mut typed: Query<(&mut Incoming, &ForAgent), Added<Admitted>>,
    mut notices: MessageWriter<Notice>,
) {
    for (mut typed, &ForAgent(agent)) in &mut typed {
        if typed.origin.kind != OriginKind::User || typed.command().is_some() {
            continue;
        }
        let here = std::env::current_dir().unwrap_or_default();
        let attached =
            rig_tools::attach::attachments(&typed.text, &here, &rig_tools::workspace::Local);
        for note in attached.notes {
            notices.write(Notice::info(agent, note));
        }
        let attachments = attached
            .files
            .into_iter()
            .map(|(label, content)| Attachment { label, content });
        typed.attachments.extend(attachments);
    }
}

#[cfg(test)]
mod tests;
