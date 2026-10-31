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

use std::path::{Path, PathBuf};
use std::sync::{Arc, PoisonError, RwLock};

use bevy_tasks::{IoTaskPool, TaskPool};
use rig_harness::prelude::{
    Added, Admitted, AgentPass, App, AppPerCallExt, Attachment, Author, AuthorKind, CallPlace,
    Commands, Component, Decider, Decision, Effect, Entity, ForAgent, HoldOf, Incoming, Insert,
    IntoScheduleConfigs, MessagePart, MessageWriter, Name, Notice, On, Outcome, PerCallTool,
    Plugin, PrepareSystems, Processes, ProjectRoot, ProjectWorkspace, Query, Recalled, Res,
    Resource, Running, Spawned, Started, StopTurn, SystemSet, Tool, ToolOptions, Wake, Without,
    WorkDir, blocking,
};
use rig_harness::rig_core::completion::ToolAnnotations;
use rig_harness::rig_core::tool::{ToolContext, ToolExecutionError};
use rig_tools::attach::{Attached, Reference, attachments};
use rig_tools::{Edit, OutputLimit, Read, Seen, Shell, Write};
#[cfg(feature = "tui")]
use rig_tui::{RenderSystems, TranscriptRender};

/// The systems that draw the coding tools' calls in the terminal view, in
/// `rig_tui::RenderSystems::Render`. A plugin that draws them its own way
/// runs after it.
#[cfg(feature = "tui")]
#[derive(bevy_ecs::schedule::SystemSet, Clone, Debug, PartialEq, Eq, Hash)]
pub struct CodingToolLooks;

#[cfg(feature = "tui")]
mod looks;
pub mod remote;

/// The files `read` showed and `edit` left, which [`ReadTool`] and
/// [`EditTool`] share, so an edit whose text is not found asks for a new
/// read only of a file that changed since one.
#[derive(Resource, Clone, Debug, Default)]
pub struct SeenFiles(pub Seen);

/// The `read` tool.
#[derive(Default)]
pub struct ReadTool;

impl Plugin for ReadTool {
    fn build(&self, app: &mut App) {
        let seen = app
            .world_mut()
            .get_resource_or_init::<SeenFiles>()
            .0
            .clone();
        app.add_tool_per_call(
            Read {
                seen,
                ..Read::default()
            },
            ToolOptions::default(),
        );
        #[cfg(feature = "tui")]
        app.add_systems(
            TranscriptRender,
            looks::draw_read
                .in_set(CodingToolLooks)
                .in_set(RenderSystems::Render),
        );
    }
}

/// The `edit` tool.
#[derive(Default)]
pub struct EditTool;

impl Plugin for EditTool {
    fn build(&self, app: &mut App) {
        let seen = app
            .world_mut()
            .get_resource_or_init::<SeenFiles>()
            .0
            .clone();
        app.add_tool_per_call(
            Edit {
                seen,
                ..Edit::default()
            },
            ToolOptions::default(),
        );
        #[cfg(feature = "tui")]
        app.add_systems(
            TranscriptRender,
            looks::draw_edit
                .in_set(CodingToolLooks)
                .in_set(RenderSystems::Render),
        );
    }
}

/// The `write` tool.
#[derive(Default)]
pub struct WriteTool;

impl Plugin for WriteTool {
    fn build(&self, app: &mut App) {
        let seen = app
            .world_mut()
            .get_resource_or_init::<SeenFiles>()
            .0
            .clone();
        app.add_tool_per_call(
            Write {
                seen,
                ..Write::default()
            },
            ToolOptions::default(),
        );
        #[cfg(feature = "tui")]
        app.add_systems(
            TranscriptRender,
            looks::draw_write
                .in_set(CodingToolLooks)
                .in_set(RenderSystems::Render),
        );
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
        app.add_live_tool_per_call(shell, ToolOptions::default());
        #[cfg(feature = "tui")]
        app.add_systems(
            TranscriptRender,
            looks::draw_shell
                .in_set(CodingToolLooks)
                .in_set(RenderSystems::Render),
        );
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

impl PerCallTool for SharedShell {
    fn at(&self, place: CallPlace) -> Self {
        Self {
            shell: Shell {
                workspace: place.workspace,
                live: place.live,
                confine: place.confine,
                spill: place.spill,
                ..self.shell.clone()
            },
            dir: Some(place.base),
            ..self.clone()
        }
    }
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
/// in the app's [`ProjectWorkspace`], as the file tools resolve them; a
/// path with no file there, or one that could not be looked at, is warned
/// about. Files on another machine are read off the main thread while the
/// message waits ([`HoldOf`]); stopping the agent's turn meanwhile puts the
/// message back in the input. It runs in [`AttachSystems`].
#[derive(Default)]
pub struct AttachPlugin;

/// Where [`AttachPlugin`] reads the files a user's message names into its
/// attachments, in [`PrepareSystems::Compose`]: a plugin that rewrites the
/// text goes before, in `Input`; one that reads or edits the attachments
/// goes `.after(AttachSystems)`. Files on another machine are read off the
/// main thread under a hold on the message, so they come later than this
/// set, as do those of a message held behind one that waits for them.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AttachSystems;

impl Plugin for AttachPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .configure_sets(AgentPass, AttachSystems.in_set(PrepareSystems::Compose))
            .add_systems(AgentPass, attach.in_set(AttachSystems))
            .add_observer(release_read)
            .add_observer(stop_reading);
    }
}

/// The hold ([`HoldOf`]) on a user's message to this agent while the
/// files it names, or those of a message sent before it, are read.
#[derive(Component)]
struct Reading(Entity);

/// The files a message names, read off the main thread, on the hold that
/// keeps the message ([`HoldOf`]) until they are attached; taken then.
#[derive(Component)]
struct ReadFiles(Option<Result<Attached, ToolExecutionError>>);

impl From<Result<Attached, ToolExecutionError>> for ReadFiles {
    fn from(read: Result<Attached, ToolExecutionError>) -> Self {
        Self(Some(read))
    }
}

/// Reads the files each user's message names into its attachments as it
/// comes in: inline when every one is on this machine, else off the main
/// thread under a hold on the message ([`Workspace::host_for`]). While a
/// message to an agent is held, a later one of the user's own, or one that
/// names files, is held behind it, so it does not pass it. A message a
/// restart put back in the inbox ([`Started`]) has them already.
///
/// [`Workspace::host_for`]: rig_tools::workspace::Workspace::host_for
fn attach(
    mut typed: Query<
        (Entity, &Effect, &mut Incoming, &ForAgent),
        (Added<Admitted>, Without<Started>),
    >,
    reading: Query<&Reading>,
    dirs: Query<&WorkDir>,
    (workspace, root): (Res<ProjectWorkspace>, Res<ProjectRoot>),
    wake: Res<Wake>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let mut waiting: Vec<Entity> = reading.iter().map(|&Reading(agent)| agent).collect();
    let mut typed: Vec<_> = (typed.iter_mut())
        .filter(|(_, _, typed, _)| typed.author.kind == AuthorKind::User)
        .filter(|(_, _, typed, _)| typed.command().is_none())
        .collect();
    typed.sort_unstable_by_key(|(_, effect, ..)| effect.id());
    for (message, _, mut typed, &ForAgent(agent)) in typed {
        let hold = (Name::new("attachments"), Reading(agent), HoldOf(message));
        let base = dirs.get(agent).map_or(&root.0, |dir| &dir.0).clone();
        let workspace = workspace.0.clone();
        let (names, remote) = {
            let mut named = Reference::parse(typed.text()).peekable();
            let names = named.peek().is_some();
            let remote = named.any(|reference| {
                let path = workspace.join(&base, Path::new(&*reference.path));
                workspace.host_for(&path).is_some()
            });
            (names, remote)
        };
        if !names {
            // Only the user's own, which a stop puts back in the input.
            if waiting.contains(&agent) && typed.author == Author::user() {
                commands.spawn((hold, ReadFiles::from(Ok(Attached::default()))));
            }
            continue;
        }
        if remote {
            let text = typed.text().to_owned();
            let read = blocking(move || Ok(attachments(&text, &base, &*workspace)));
            let pool = IoTaskPool::get_or_init(TaskPool::default);
            let running = Running::spawn_into::<ReadFiles, _>(pool, &wake, read);
            commands.spawn((hold, running));
            waiting.push(agent);
            continue;
        }
        let attached = attachments(typed.text(), &base, &*workspace);
        match waiting.contains(&agent) {
            true => {
                commands.spawn((hold, ReadFiles::from(Ok(attached))));
            }
            false => add_attached(&mut typed, agent, attached, &mut notices),
        }
    }
}

/// The holds on messages while their files are read, with what was read.
type ReadHolds<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static Reading,
        &'static HoldOf,
        Option<&'static mut ReadFiles>,
    ),
>;

/// The messages held, as [`release`] and [`stop_reading`] edit them.
type HeldMessages<'w, 's> = Query<
    'w,
    's,
    (
        &'static Effect,
        &'static mut Incoming,
        &'static mut Decision,
    ),
>;

/// Attaches the files read off the main thread, as [`release`] does, each
/// time a read ends.
fn release_read(
    _: On<Insert<ReadFiles>>,
    mut holds: ReadHolds,
    mut messages: HeldMessages,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    release(&mut holds, &mut messages, &[], &mut notices, &mut commands);
}

/// Attaches what was read to each held message, with a notice about each
/// file that could not be and a warning about each that is not there, and
/// releases it, in the order the messages to an agent were sent: one waits
/// until those before it are read. The holds in `gone` are being removed.
fn release(
    holds: &mut ReadHolds,
    messages: &mut HeldMessages,
    gone: &[Entity],
    notices: &mut MessageWriter<Notice>,
    commands: &mut Commands,
) {
    let mut holds: Vec<_> = (holds.iter_mut())
        .filter(|(hold, ..)| !gone.contains(hold))
        .map(|(hold, &Reading(agent), &HoldOf(message), read)| {
            let sent = messages.get(message).ok().map(|(effect, ..)| effect.id());
            (agent, sent, hold, message, read)
        })
        .collect();
    holds.sort_unstable_by_key(|&(agent, sent, ..)| (agent, sent));
    let mut blocked = None;
    for (agent, _, hold, message, read) in holds {
        if blocked == Some(agent) {
            continue;
        }
        let Some(mut read) = read else {
            blocked = Some(agent);
            continue;
        };
        // Taken: this hold is released already.
        let Some(read) = read.0.take() else {
            continue;
        };
        if let Ok((_, mut typed, _)) = messages.get_mut(message) {
            match read {
                Ok(attached) => add_attached(&mut typed, agent, attached, notices),
                Err(failure) => {
                    notices.write(Notice::warn(agent, format!("No files attached: {failure}")));
                }
            }
        }
        commands.entity(hold).despawn();
    }
}

/// Stops reading the files of the user's messages to the stopped agent and
/// the agents below it, which go back to the input instead of to the
/// model, in the order they were sent, as a stopped turn's do; a message
/// that is not the user's own goes on once the messages before it are
/// read ([`release`]).
fn stop_reading(
    stop: On<StopTurn>,
    spawned: Query<&Spawned>,
    mut holds: ReadHolds,
    mut messages: HeldMessages,
    (mut recalled, mut notices): (MessageWriter<Recalled>, MessageWriter<Notice>),
    mut commands: Commands,
) {
    let below = spawned.iter_descendants::<Spawned>(stop.entity);
    let stopped: Vec<Entity> = std::iter::once(stop.entity).chain(below).collect();
    let (mut back, mut gone) = (Vec::new(), Vec::new());
    for (hold, &Reading(agent), &HoldOf(message), _) in &holds {
        let Ok((effect, typed, mut decision)) = messages.get_mut(message) else {
            continue;
        };
        if !stopped.contains(&agent) || typed.author != Author::user() {
            continue;
        }
        if decision.decide(Decider::plugin("attach"), Outcome::custom("recalled")) {
            back.push((effect.id(), agent, typed.text().trim().to_owned()));
        }
        commands.entity(hold).despawn();
        gone.push(hold);
    }
    release(
        &mut holds,
        &mut messages,
        &gone,
        &mut notices,
        &mut commands,
    );
    back.sort_unstable();
    for agent in stopped {
        let texts: Vec<&str> = (back.iter())
            .filter(|(_, to, _)| *to == agent)
            .map(|(_, _, text)| text.as_str())
            .collect();
        if !texts.is_empty() {
            recalled.write(Recalled {
                agent,
                text: texts.join("\n\n"),
                why: None,
            });
        }
    }
}

/// Puts `attached` before the message's text, with a notice about each
/// file that could not be attached and a warning for each other line.
fn add_attached(
    typed: &mut Incoming,
    agent: Entity,
    attached: Attached,
    notices: &mut MessageWriter<Notice>,
) {
    for note in attached.notes {
        notices.write(Notice::info(agent, note));
    }
    for warning in attached.warnings {
        notices.write(Notice::warn(agent, warning));
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
