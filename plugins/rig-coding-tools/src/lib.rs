//! The built-in tools of [`rig_tools`], one plugin each, with their rules
//! on when to pick them: `read` and `search` run beside each other, `edit`,
//! `write` and `shell` alone. With the `tui` feature (on by default), each
//! plugin also adds how the terminal view draws its tool's calls.
//! [`AttachPlugin`] sends the files the user names as `@path` with the
//! message.

use std::sync::{Arc, PoisonError, RwLock};

use rig_harness::prelude::{
    Added, Admitted, AgentPass, App, AppToolsExt, Attachment, ForAgent, Incoming,
    IntoScheduleConfigs, MessageWriter, Notice, OriginKind, Plugin, Processes, Query, Resource,
    SessionPaths, Stage, SystemSet, Tool, ToolOptions,
};
use rig_harness::rig_core::completion::ToolAnnotations;
use rig_harness::rig_core::tool::ToolContext;
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
        };
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

/// [`Shell`], starting each command as [`Processes`] say at the time, and
/// cutting its output as [`ShellOutput`] says.
struct SharedShell {
    shell: Shell,
    processes: Processes,
    output: ShellOutput,
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
        let shell = Shell {
            launcher: self.processes.launcher(),
            cut: self.output.cut(),
            ..self.shell.clone()
        };
        shell.call(context, args).await
    }
}

/// The system in [`Stage::Prepare`] that reads the files a user's message
/// names into its attachments ([`AttachPlugin`]). A plugin that edits the
/// message's text or attachments orders against it.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AttachSystems;

/// Files the user names as `@path` go with the message, before its text,
/// and the `@path` stays in the text so the model knows which file it is
/// ([`rig_tools::attach`]): an image as an image, which the core sends
/// only to a model that reads images, telling the user otherwise; any
/// other file as its text, numbered and capped the way `read` shows it.
#[derive(Default)]
pub struct AttachPlugin;

impl Plugin for AttachPlugin {
    fn build(&self, app: &mut App) {
        app.configure_sets(AgentPass, AttachSystems.in_set(Stage::Prepare))
            .add_systems(AgentPass, attach.in_set(AttachSystems));
    }
}

/// Reads the files each user's message names into its attachments as it
/// comes in, in Prepare, with a notice about each that could not be.
fn attach(
    mut typed: Query<(&mut Incoming, &ForAgent), Added<Admitted>>,
    mut notices: MessageWriter<Notice>,
) {
    for (mut typed, &ForAgent(agent)) in &mut typed {
        if typed.origin.kind != OriginKind::User || typed.command().is_some() {
            continue;
        }
        let (attachments, notes) = rig_tools::attach::attachments(&typed.text);
        for note in notes {
            notices.write(Notice::info(agent, note));
        }
        let attachments = attachments
            .into_iter()
            .map(|(label, content)| Attachment { label, content });
        typed.attachments.extend(attachments);
    }
}

#[cfg(test)]
mod tests;
