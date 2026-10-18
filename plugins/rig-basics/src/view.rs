//! The view halves of rig's basic plugins: how the terminal view draws
//! their tool calls and entries ([`BasicsViewPlugin`]). They run in the
//! view (the `view` role), over the mirror of a session whose agent runs
//! the plugins' agent halves; their status line items are the agent's
//! facts, which reach the view as the session's `status` events.

use bevy_app::prelude::*;
use rig_harness::rig_core::tool::Tool;
use rig_tools::{Edit, Read, Shell, Write};
use rig_tui::AppToolRenderersExt;

use crate::coding_tools::looks;

/// How the coding tools' and subagents' calls and compaction's summaries
/// are drawn.
#[derive(Default)]
pub struct BasicsViewPlugin;

impl Plugin for BasicsViewPlugin {
    fn build(&self, app: &mut App) {
        app.add_tool_renderer(Read::NAME, looks::read)
            .add_tool_renderer(Edit::NAME, looks::edit)
            .add_tool_renderer(Write::NAME, looks::write)
            .add_tool_renderer(Shell::NAME, looks::shell);
        crate::compaction::tui::add(app);
        crate::subagents::looks::add(app);
    }
}

#[cfg(test)]
mod tests;
