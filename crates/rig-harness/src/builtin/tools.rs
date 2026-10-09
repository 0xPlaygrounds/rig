//! The built-in tools of [`rig_tools`]: read, edit, write, shell and search.

use bevy_app::prelude::*;
use rig_tools::{Edit, Read, Search, Shell, Write};

use crate::core::tools::{AppToolsExt, Footprint, ToolOptions};

/// Registers the built-in tools, each with its rules on when to pick it,
/// with [`AppToolsExt::add_tool_with`]: `read` and `search` run beside each
/// other, `edit`, `write` and `shell` alone.
#[derive(Default)]
pub struct BuiltinToolsPlugin;

impl Plugin for BuiltinToolsPlugin {
    fn build(&self, app: &mut App) {
        let alone = |rules| ToolOptions {
            rules,
            ..ToolOptions::default()
        };
        let read_only = |rules| ToolOptions {
            rules,
            footprint: Footprint::ReadOnly,
        };
        // A command, such as a nested agent run while working on rig-harness
        // itself, must not act as this agent.
        let shell = Shell {
            unset_env: &rig::harness_protocol::env::AGENT_ONLY,
        };
        app.add_tool_with(Read, read_only(Read::RULES))
            .add_tool_with(Edit, alone(Edit::RULES))
            .add_tool_with(Write, alone(Write::RULES))
            .add_tool_with(shell, alone(Shell::RULES))
            .add_tool_with(Search, read_only(Search::RULES));
    }
}
