//! The built-in tools of [`rig_tools`], one plugin each, with their rules
//! on when to pick them: `read` and `search` run beside each other, `edit`,
//! `write` and `shell` alone. With the `tui` feature (on by default), each
//! plugin also adds how the terminal view draws its tool's calls.
//! [`AttachPlugin`] sends the files the user names as `@path` with the
//! message.

use rig_harness::prelude::{
    Added, Admitted, AgentPass, App, AppToolsExt, Attachment, ForAgent, Incoming,
    IntoScheduleConfigs, MessageWriter, Notice, OriginKind, Plugin, Processes, Query, SessionPaths,
    Stage, Tool, ToolOptions,
};
use rig_harness::rig_core::completion::ToolAnnotations;
use rig_harness::rig_core::tool::ToolContext;
use rig_tools::{Edit, Read, Search, Shell, Write};
#[cfg(feature = "tui")]
use rig_tui::AppToolRenderersExt;

#[cfg(feature = "tui")]
mod looks;

/// Options with `rules`; whether a tool's calls run beside others comes
/// from its annotations: `read` and `search` only read.
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
        app.add_tool_with(Read, options(Read::RULES));
        #[cfg(feature = "tui")]
        app.add_tool_renderer(Read::NAME, looks::read);
    }
}

/// The `search` tool.
#[derive(Default)]
pub struct SearchTool;

impl Plugin for SearchTool {
    fn build(&self, app: &mut App) {
        app.add_tool_with(Search, options(Search::RULES));
        #[cfg(feature = "tui")]
        app.add_tool_renderer(Search::NAME, looks::search);
    }
}

/// The `edit` tool.
#[derive(Default)]
pub struct EditTool;

impl Plugin for EditTool {
    fn build(&self, app: &mut App) {
        app.add_tool_with(Edit, options(Edit::RULES));
        #[cfg(feature = "tui")]
        app.add_tool_renderer(Edit::NAME, looks::edit);
    }
}

/// The `write` tool.
#[derive(Default)]
pub struct WriteTool;

impl Plugin for WriteTool {
    fn build(&self, app: &mut App) {
        app.add_tool_with(Write, options(Write::RULES));
        #[cfg(feature = "tui")]
        app.add_tool_renderer(Write::NAME, looks::write);
    }
}

/// The `shell` tool. Each call starts its process as the app's
/// [`Processes`] say when it runs, so a sandbox plugin's policy applies;
/// output it cuts is kept whole in the session's
/// [`spill`](SessionPaths::spill) directory.
#[derive(Default)]
pub struct ShellTool;

impl Plugin for ShellTool {
    fn build(&self, app: &mut App) {
        let shell = SharedShell {
            shell: Shell {
                spill: app
                    .world()
                    .get_resource::<SessionPaths>()
                    .map(SessionPaths::spill),
                ..Shell::default()
            },
            processes: app.world_mut().get_resource_or_init::<Processes>().clone(),
        };
        app.add_tool_with(shell, options(Shell::RULES));
        #[cfg(feature = "tui")]
        app.add_tool_renderer(Shell::NAME, looks::shell);
    }
}

/// [`Shell`], starting each command as [`Processes`] say at the time.
struct SharedShell {
    shell: Shell,
    processes: Processes,
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
            ..self.shell.clone()
        };
        shell.call(context, args).await
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
        app.add_systems(AgentPass, attach.in_set(Stage::Prepare));
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
