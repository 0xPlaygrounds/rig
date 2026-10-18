//! The coding tools of [`rig_tools`] the agent ships, one plugin each,
//! each saying in its description when to pick it: `read`, `edit`, `write`
//! and `shell`, as pi has. `read`s run beside each other; `edit`, `write` and `shell`
//! run alone. The library's other tools (`search`, `list`, `delete`,
//! `move`) are the shell's job here; a plugin that wants one registers it
//! with [`AppPerCallExt::add_tool_per_call`].
//!
//! Each tool is made per call ([`AppPerCallExt`]), so every call works in
//! the app's [`ProjectWorkspace`] from its agent's [`WorkDir`], else the
//! [`ProjectRoot`]. With a `WorkDir`, a path outside it is refused. `shell` streams its output as it comes
//! ([`CallOutput`](rig_harness::prelude::CallOutput)). With the `tui`
//! feature (default), each plugin adds its tool's look. [`AttachPlugin`]
//! sends the files a message names as `@path` with it. [`remote::RemoteHostsPlugin`]
//! makes a path written `HOST:PATH` another machine's, for every tool, by
//! the transports `[remote.hosts]` names ([`remote::RemoteSettings`]).

use std::path::PathBuf;
use std::sync::{Arc, PoisonError, RwLock};

use bevy_tasks::{IoTaskPool, TaskPool};
use rig_harness::prelude::{
    Added, Admitted, AgentPass, App, AppPerCallExt, AppSettingsExt, Attachment, AuthorKind,
    Commands, Component, Entity, ForAgent, HoldOf, Incoming, Insert, IntoScheduleConfigs,
    MessagePart, MessageWriter, Name, Notice, On, Plugin, PrepareSystems, Processes, ProjectRoot,
    ProjectWorkspace, Query, Res, Resource, Running, Started, SystemSet, Tool, ToolOptions, Wake,
    Without, WorkDir, blocking,
};
use rig_harness::rig_core::completion::ToolAnnotations;
use rig_harness::rig_core::serde::Deserialize;
use rig_harness::rig_core::tool::{ToolContext, ToolExecutionError};
use rig_tools::attach::{Attached, attachments};
use rig_tools::{Edit, OutputLimit, Read, Shell, Write};
#[cfg(feature = "tui")]
use rig_tui::AppToolRenderersExt;

#[cfg(feature = "tui")]
mod looks;
pub mod remote;

/// The `read` tool.
#[derive(Default)]
pub struct ReadTool;

impl Plugin for ReadTool {
    fn build(&self, app: &mut App) {
        app.add_tool_per_call(Read::default(), ToolOptions::default(), |place| Read {
            workspace: place.workspace,
            base: Some(place.base),
            confine: place.confine,
        });
        #[cfg(feature = "tui")]
        app.add_tool_renderer(Read::NAME, looks::read);
    }
}

/// The `edit` tool.
#[derive(Default)]
pub struct EditTool;

impl Plugin for EditTool {
    fn build(&self, app: &mut App) {
        app.add_tool_per_call(Edit::default(), ToolOptions::default(), |place| Edit {
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
        app.add_tool_per_call(Write::default(), ToolOptions::default(), |place| Write {
            workspace: place.workspace,
            base: Some(place.base),
            confine: place.confine,
        });
        #[cfg(feature = "tui")]
        app.add_tool_renderer(Write::NAME, looks::write);
    }
}

/// The `shell` tool. Each call starts its process through the app's
/// [`Processes`], so a sandbox plugin's policy applies. It runs in its
/// agent's [`WorkDir`], else the directory the `Processes` name, else the
/// [`ProjectRoot`]. Output is cut as [`ShellOutput`] says, with the whole
/// of it kept in the session's
/// [`spill`](rig_harness::prelude::Scratch::spill) directory. While the
/// call runs, its output streams as its
/// [`ToolProgress`](rig_harness::prelude::ToolProgress), and it is the
/// answer if the call is stopped
/// ([`CallOutput`](rig_harness::prelude::CallOutput)).
/// `[shell] max_timeout` ([`ShellSettings`]) caps the timeout a call may
/// ask for.
#[derive(Default)]
pub struct ShellTool;

/// `[shell]` in `settings.toml`, read by [`ShellTool`].
#[derive(Resource, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(crate = "rig_harness::rig_core::serde", default, deny_unknown_fields)]
pub struct ShellSettings {
    /// The most seconds a call may ask for in `timeout_secs`; a longer
    /// timeout is cut to it. Ten minutes unless set.
    pub max_timeout: u64,
}

impl Default for ShellSettings {
    fn default() -> Self {
        Self {
            max_timeout: Shell::default().max_timeout,
        }
    }
}

impl Plugin for ShellTool {
    fn build(&self, app: &mut App) {
        let output = app
            .world_mut()
            .get_resource_or_init::<ShellOutput>()
            .clone();
        let max_timeout = app
            .add_setting::<ShellSettings>("shell")
            .world()
            .resource::<ShellSettings>()
            .max_timeout;
        let shell = SharedShell {
            shell: Shell {
                max_timeout,
                ..Shell::default()
            },
            processes: app.world_mut().get_resource_or_init::<Processes>().clone(),
            output,
            dir: None,
        };
        let per = shell.clone();
        app.add_live_tool_per_call(shell, ToolOptions::default(), move |place| SharedShell {
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
        #[cfg(feature = "tui")]
        app.add_tool_renderer(Shell::NAME, looks::shell);
    }
}

/// How much of a command's output the `shell` tool gives the model, read
/// as each call runs: [`OutputLimit::default`] unless set. A plugin that cuts or
/// summarizes output itself turns the cut off with `set(None)`, such as
/// from its `finish`; the command is still killed past 10 MB of output.
/// Shared by its clones: change it with [`set`](Self::set), not by
/// inserting a new one.
#[derive(Resource, Clone, Debug)]
pub struct ShellOutput(Arc<RwLock<Option<OutputLimit>>>);

impl ShellOutput {
    /// How the next call cuts its output; `None` keeps all of it.
    pub fn cut(&self) -> Option<OutputLimit> {
        *self.0.read().unwrap_or_else(PoisonError::into_inner)
    }

    /// Cuts every call's output from now on as `cut` says.
    pub fn set(&self, cut: Option<OutputLimit>) {
        *self.0.write().unwrap_or_else(PoisonError::into_inner) = cut;
    }
}

impl Default for ShellOutput {
    fn default() -> Self {
        Self(Arc::new(RwLock::new(Some(OutputLimit::default()))))
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

/// Files the user names as `@path` go with the message, before its text,
/// and the `@path` stays in the text so the model knows which file it is
/// ([`rig_tools::attach`]): an image as an image, which the kernel sends
/// only to a model that reads images, telling the user otherwise; any
/// other file as its text, capped the way `read` shows it.
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
/// as work off the main thread that holds the message. A message a restart
/// put back in the inbox ([`Started`]) has them already.
fn attach(
    mut typed: Query<(Entity, &mut Incoming, &ForAgent), (Added<Admitted>, Without<Started>)>,
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
