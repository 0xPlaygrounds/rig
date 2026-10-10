//! The `reload` tool: the agent rebuilds and restarts its own harness, to
//! apply its own plugin changes.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig_core::tool::{ToolExecutionError, ToolResult};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::prelude::{ReloadStatus, launcher};
use rig_ecs::agent::Notice;
use rig_ecs::tools::{AppToolsExt, Footprint, ToolCalled, ToolOptions, ToolOutput};

/// The tool's name.
pub const RELOAD_TOOL: &str = "reload";

const RELOAD_DESCRIPTION: &str = "Rebuild the agent with the plugins in plugins.toml and \
    restart on the new build, to apply your own plugin changes. Nothing happens mid-turn: the \
    build starts once this turn and every other running turn have ended, so finish your \
    edits first and end your turn after this call. The user sees a notice and can cancel it. \
    After the restart the session, with this conversation, carries on. If the build fails, \
    this build keeps running and its first errors arrive in your conversation as a note.";

/// Offers the model the `reload` tool when the launcher started the
/// agent, since only then can it rebuild itself.
#[derive(Default)]
pub struct ReloadTool;

impl Plugin for ReloadTool {
    fn build(&self, app: &mut App) {
        if launcher::executable().is_some() {
            app.add_open_tool(
                RELOAD_TOOL,
                RELOAD_DESCRIPTION,
                ToolOptions {
                    rules: &[],
                    // Never run again after the restart it causes.
                    footprint: Footprint::Independent,
                },
                on_reload_tool,
            );
        }
    }
}

/// The arguments of a `reload` call: none.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ReloadArgs {}

/// The `reload` tool: asks for a reload once no turn runs, with a notice
/// naming the agent, and answers whether it is queued.
fn on_reload_tool(
    called: On<ToolCalled<ReloadArgs>>,
    mut status: ResMut<ReloadStatus>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let output = match status.ask(called.agent) {
        Ok(()) => {
            notices.write(Notice::info(
                None,
                format!(
                    "The model of agent {} asked to reload: the agent rebuilds and \
                     restarts once no turn runs. /reload cancel cancels it.",
                    called.caller.short()
                ),
            ));
            ToolResult::success(
                "Reload queued: the build starts once your turn and every other running \
                 turn have ended. End your turn now with a short summary for the user; the \
                 conversation carries on after the restart. A failed build keeps this build \
                 running and its errors arrive as a note."
                    .into(),
            )
        }
        Err(why) => ToolResult::failed(ToolExecutionError::other(format!(
            "{why} Nothing was queued."
        ))),
    };
    commands
        .entity(called.call)
        .insert_if_new(ToolOutput(output));
}
