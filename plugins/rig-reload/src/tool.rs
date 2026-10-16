//! The `reload` tool, with which the model rebuilds and restarts its own
//! agent to apply its own plugin changes, and the system prompt's section
//! on how the agent extends itself ([`SELF_EXTENSION`]): fixed text, the
//! same before and after a reload, so the provider's prompt cache keeps it.

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

/// Where this build's plugin guide is: `PLUGINS.md` in the rig source it
/// was built from, as a literal for `concat!`.
macro_rules! plugin_guide {
    () => {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../crates/rig-harness/PLUGINS.md")
    };
}
#[cfg(test)]
pub(crate) use plugin_guide;

/// The tag of the system prompt's section on how the agent extends itself.
pub const SELF_EXTENSION_TAG: &str = "self_extension";

/// The system prompt's section on how the agent extends itself, when the
/// launcher started it. It names the launcher and `RIG_HOME` by the
/// variables the launcher exports to the agent and its shell, and lists
/// nothing, so it is the same before and after a reload; the model runs
/// `$RIG_LAUNCHER plugin list` for the plugins, and its tools come with
/// the request. The plugin guide is this build's own, in rig's source.
pub const SELF_EXTENSION: &str = concat!(
    "You are built from Bevy plugins listed in $RIG_HOME/plugins.toml. To add one: \
     `$RIG_LAUNCHER plugin new <name>` makes a working plugin in $RIG_HOME/plugins/<name> and \
     lists it. `$RIG_LAUNCHER plugin list` shows which are enabled and disabled; change the \
     list only with `$RIG_LAUNCHER plugin add`, `remove`, `enable` or `disable`, and never edit \
     rig's own crates. The plugin guide, ",
    plugin_guide!(),
    ", shows every extension point. Check a change with \
     `$RIG_LAUNCHER plugin check --build`, then call `reload` and end your turn: you rebuild \
     and restart in this session. A failed build keeps this one running and its errors come \
     to you as a note."
);

/// Offers the model the `reload` tool and spawns the system prompt's
/// section on how the agent extends itself ([`SELF_EXTENSION`]), when the
/// launcher started the agent, since only then can it rebuild itself.
pub(crate) fn add(app: &mut App) {
    if harness_protocol::launcher().is_none() {
        return;
    }
    let section = PromptSection::new(
        PromptSection::ORDER_PROJECT - 100,
        SELF_EXTENSION_TAG,
        SELF_EXTENSION,
    );
    app.add_open_tool(
        RELOAD_TOOL,
        RELOAD_DESCRIPTION,
        ToolOptions {
            // Never run again after the restart it causes.
            footprint: Some(Touches::Independent),
        },
        on_reload_tool,
    )
    .world_mut()
    .spawn((Name::new("prompt:self_extension"), section));
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
