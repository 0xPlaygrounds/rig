//! The built-in tools of [`rig_tools`], one plugin each, with their rules
//! on when to pick them: `read` and `search` run beside each other, `edit`,
//! `write` and `shell` alone. With the `tui` feature (on by default), each
//! plugin also adds how the terminal view draws its tool's calls.
//! [`AttachPlugin`] sends the files the user names as `@path` with the
//! message.

#[cfg(feature = "tui")]
use rig_harness::prelude::PortableTool;
use rig_harness::prelude::{
    App, AppToolsExt, Attachment, Deliver, Footprint, MessageWriter, Notice, On, OriginKind,
    Plugin, SessionPaths, ToolOptions,
};
use rig_tools::{Edit, Read, Search, Shell, Write};
#[cfg(feature = "tui")]
use rig_tui::AppToolRenderersExt;

#[cfg(feature = "tui")]
mod looks;

/// Options for a tool that runs alone.
fn alone(rules: &'static [&'static str]) -> ToolOptions<'static> {
    ToolOptions {
        rules,
        ..ToolOptions::default()
    }
}

/// Options for a tool that only reads, so it runs beside the others that do.
fn read_only(rules: &'static [&'static str]) -> ToolOptions<'static> {
    ToolOptions {
        rules,
        footprint: Footprint::ReadOnly,
    }
}

/// The `read` tool.
#[derive(Default)]
pub struct ReadTool;

impl Plugin for ReadTool {
    fn build(&self, app: &mut App) {
        app.add_tool_with(Read, read_only(Read::RULES));
        #[cfg(feature = "tui")]
        app.add_tool_renderer(Read::NAME, looks::read);
    }
}

/// The `search` tool.
#[derive(Default)]
pub struct SearchTool;

impl Plugin for SearchTool {
    fn build(&self, app: &mut App) {
        app.add_tool_with(Search, read_only(Search::RULES));
        #[cfg(feature = "tui")]
        app.add_tool_renderer(Search::NAME, looks::search);
    }
}

/// The `edit` tool.
#[derive(Default)]
pub struct EditTool;

impl Plugin for EditTool {
    fn build(&self, app: &mut App) {
        app.add_tool_with(Edit, alone(Edit::RULES));
        #[cfg(feature = "tui")]
        app.add_tool_renderer(Edit::NAME, looks::edit);
    }
}

/// The `write` tool.
#[derive(Default)]
pub struct WriteTool;

impl Plugin for WriteTool {
    fn build(&self, app: &mut App) {
        app.add_tool_with(Write, alone(Write::RULES));
        #[cfg(feature = "tui")]
        app.add_tool_renderer(Write::NAME, looks::write);
    }
}

/// The `shell` tool. Output it cuts is kept whole in the session's
/// [`spill`](SessionPaths::spill) directory.
#[derive(Default)]
pub struct ShellTool;

impl Plugin for ShellTool {
    fn build(&self, app: &mut App) {
        let shell = Shell {
            // A command, such as a nested agent run while working on
            // rig-harness itself, must not act as this agent.
            unset_env: &rig_harness::harness_protocol::env::AGENT_ONLY,
            spill: app
                .world()
                .get_resource::<SessionPaths>()
                .map(SessionPaths::spill),
        };
        app.add_tool_with(shell, alone(Shell::RULES));
        #[cfg(feature = "tui")]
        app.add_tool_renderer(Shell::NAME, looks::shell);
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
        app.add_observer(attach);
    }
}

/// Reads the files a user's message names into its attachments, with a
/// notice about each that could not be.
fn attach(mut typed: On<Deliver>, mut notices: MessageWriter<Notice>) {
    if typed.origin.kind != OriginKind::User || typed.command().is_some() {
        return;
    }
    let (attachments, notes) = rig_tools::attach::attachments(&typed.text);
    for note in notes {
        notices.write(Notice::info(typed.entity, note));
    }
    let attachments = attachments
        .into_iter()
        .map(|(label, content)| Attachment { label, content });
    typed.attachments.extend(attachments);
}
