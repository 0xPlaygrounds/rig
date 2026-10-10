//! The built-in tools of [`rig_tools`], one plugin each, with their rules
//! on when to pick them: `read` and `search` run beside each other, `edit`,
//! `write` and `shell` alone. [`ReloadTool`] adds the `reload` tool, with
//! which the agent rebuilds itself, and the system prompt's section on what
//! the agent is. With the `tui` feature (on by default), each plugin also
//! adds how the terminal view draws its tool's calls.

use bevy_app::prelude::*;
#[cfg(feature = "tui")]
use rig_harness::prelude::PortableTool;
use rig_tools::{Edit, Read, Search, Shell, Write};
#[cfg(feature = "tui")]
use rig_tui::AppToolRenderersExt;

use rig_ecs::tools::{AppToolsExt, Footprint, ToolOptions};
use rig_harness::prelude::SessionPaths;

#[cfg(feature = "tui")]
mod looks;
pub mod reload;

pub use reload::ReloadTool;

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
