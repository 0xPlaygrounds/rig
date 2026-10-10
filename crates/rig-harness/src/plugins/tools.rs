//! The built-in tools of [`rig_tools`], one plugin each, with their rules
//! on when to pick them: `read` and `search` run beside each other, `edit`,
//! `write` and `shell` alone.

use bevy_app::prelude::*;
use rig_tools::{Edit, Read, Search, Shell, Write};

use crate::prelude::SessionPaths;
use rig_ecs::tools::{AppToolsExt, Footprint, ToolOptions};

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
    }
}

/// The `search` tool.
#[derive(Default)]
pub struct SearchTool;

impl Plugin for SearchTool {
    fn build(&self, app: &mut App) {
        app.add_tool_with(Search, read_only(Search::RULES));
    }
}

/// The `edit` tool.
#[derive(Default)]
pub struct EditTool;

impl Plugin for EditTool {
    fn build(&self, app: &mut App) {
        app.add_tool_with(Edit, alone(Edit::RULES));
    }
}

/// The `write` tool.
#[derive(Default)]
pub struct WriteTool;

impl Plugin for WriteTool {
    fn build(&self, app: &mut App) {
        app.add_tool_with(Write, alone(Write::RULES));
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
            unset_env: &rig::harness_protocol::env::AGENT_ONLY,
            spill: app
                .world()
                .get_resource::<SessionPaths>()
                .map(SessionPaths::spill),
        };
        app.add_tool_with(shell, alone(Shell::RULES));
    }
}
