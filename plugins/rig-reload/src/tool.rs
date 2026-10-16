//! The `reload` tool, with which the model rebuilds and restarts its own
//! agent to apply its own plugin changes, and the system prompt's section
//! on how the agent extends itself: fixed text, the same before and after
//! a reload, so the provider's prompt cache keeps it.

use std::path::Path;

use rig_harness::harness_protocol;
use rig_harness::prelude::*;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::ReloadStatus;

/// The tool's name.
pub const RELOAD_TOOL: &str = "reload";

const RELOAD_DESCRIPTION: &str = "Rebuild the agent with the plugins in plugins.toml and \
    restart on the new build, to apply your own plugin changes. Nothing happens mid-turn: the \
    build starts once this turn and every other running turn have ended, so finish your \
    edits first and end your turn after this call. The user sees a notice and can cancel it. \
    After the restart the session, with this conversation, carries on. If the build fails, \
    this build keeps running and its first errors arrive in your conversation as a note.";

/// Offers the model the `reload` tool and spawns the system prompt's
/// section on how the agent extends itself, when the launcher started the
/// agent, since only then can it rebuild itself.
pub(crate) fn add(app: &mut App) {
    let Some(launcher) = harness_protocol::launcher() else {
        return;
    };
    app.world_mut().spawn((
        Name::new("prompt:rig_harness"),
        PromptSection::new(
            PromptSection::ORDER_PROJECT - 100,
            "rig_harness",
            extending(Path::new(&launcher)),
        ),
    ));
    app.add_open_tool(
        RELOAD_TOOL,
        RELOAD_DESCRIPTION,
        ToolOptions {
            rules: &[],
            // Never run again after the restart it causes.
            footprint: Some(Touches::Independent),
        },
        on_reload_tool,
    );
}

/// The section on how the agent extends itself. It names files, never what
/// they hold: the model reads `plugins.toml` for the plugins, and its tools
/// come with the request.
fn extending(launcher: &Path) -> String {
    let home = Home::from_env();
    format!(
        "You are built from plugins, Bevy plugins in crates of their own, listed in \
         {home}/plugins.toml. You extend yourself with new ones: `$RIG_LAUNCHER plugin new \
         <name>` (the launcher is {launcher}) makes a working one in {home}/plugins/<name> and \
         lists it in plugins.toml; change that list only with `$RIG_LAUNCHER plugin add` or \
         `remove`, and never edit rig's own crates. The plugin guide {guide} shows each \
         extension point. Check a change with `$RIG_LAUNCHER plugin check --build`, not cargo, \
         then call the `reload` tool and end your turn: the agent rebuilds and restarts in this \
         session. A failed build keeps this one running and its errors come to you as a note.",
        launcher = launcher.display(),
        home = home.root().display(),
        guide = rig_harness::PLUGIN_GUIDE,
    )
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
    mut calls: Query<&mut Decision, ToolCalls>,
    mut notices: MessageWriter<Notice>,
) {
    let output = match status.ask(called.agent) {
        Ok(()) => {
            notices.write(Notice::info(
                None,
                format!(
                    "The model of agent {} asked to reload: the agent rebuilds and \
                     restarts once no turn runs. /reload cancel cancels it.",
                    called.agent_id.short()
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
    if let Ok(mut call) = calls.get_mut(called.call) {
        call.decide(Decider::plugin("rig-reload"), output);
    }
}
