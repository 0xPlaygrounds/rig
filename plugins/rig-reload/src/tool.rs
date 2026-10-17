//! The `reload` tool, with which the model rebuilds and restarts its own
//! agent to apply its own plugin changes, and the system prompt's section
//! on how the agent extends itself ([`self_extension`]): fixed text, the
//! same before and after a reload, so the provider's prompt cache keeps it.

use rig_harness::harness_protocol;
use rig_harness::prelude::*;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::{MAX_RELOADS, ReloadAsked, ReloadStatus, report};

/// The tool's name.
pub const RELOAD_TOOL: &str = "reload";

const RELOAD_DESCRIPTION: &str = "Rebuild the agent with the plugins in plugins.toml and \
    restart on the new build, to apply your own plugin changes. Nothing happens mid-turn: the \
    build starts once this turn and every other running turn have ended, so finish your \
    edits first and end your turn after this call. The user sees a notice and can cancel it. \
    After the restart the session, with this conversation, carries on, and a message tells \
    you the new build started. If the build fails, this build keeps running and a message \
    brings you its first errors; if the new build stops during startup, a message says it \
    was rolled back. At most 3 reloads per message of the user.";

/// The tag of the system prompt's section on how the agent extends itself.
pub const SELF_EXTENSION_TAG: &str = "self_extension";

/// The system prompt's section on how the agent extends itself, when the
/// launcher started it. It names the launcher and `RIG_HOME` by the
/// variables the launcher exports to the agent and its shell, and lists
/// nothing, so it is the same before and after a reload; the model runs
/// `$RIG_LAUNCHER plugin list` for the plugins, and its tools come with
/// the request. The plugin guide is this build's own
/// ([`rig_harness::PLUGIN_GUIDE`]), so the text is fixed for a build.
pub fn self_extension() -> String {
    format!(
        "You are built from Bevy plugins listed in $RIG_HOME/plugins.toml. To add one: \
         `$RIG_LAUNCHER plugin new <name>` makes a working plugin in $RIG_HOME/plugins/<name> \
         and lists it. `$RIG_LAUNCHER plugin list` shows which are enabled and disabled; change \
         the list only with `$RIG_LAUNCHER plugin add`, `remove`, `enable` or `disable`, and \
         never edit rig's own crates. The plugin guide, {guide}, shows every extension point. \
         Check a change with `$RIG_LAUNCHER plugin check --build`, then call `reload` and end \
         your turn: you rebuild and restart in this session, and a message tells you how it \
         went: the new build started, the build's errors, or a rollback.",
        guide = rig_harness::PLUGIN_GUIDE,
    )
}

/// Offers the model the `reload` tool and spawns the system prompt's
/// section on how the agent extends itself ([`self_extension`]), when the
/// launcher started the agent, since only then can it rebuild itself.
pub(crate) fn add(app: &mut App) {
    if harness_protocol::launcher().is_none() {
        return;
    }
    let section = PromptSection::new(
        PromptSection::ORDER_PROJECT - 100,
        SELF_EXTENSION_TAG,
        self_extension(),
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

/// The arguments of a `reload` call.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ReloadArgs {
    /// What you will do once the new build runs, such as calling the tool
    /// you added; the message after the restart reminds you.
    #[serde(default)]
    then: Option<String>,
}

/// The `reload` tool: asks for a reload once no turn runs, with a notice
/// naming the agent, marks the agent owed its result ([`ReloadAsked`]),
/// and answers whether it is queued. Refused past [`MAX_RELOADS`] since the
/// user's last message.
fn on_reload_tool(
    called: On<ToolCalled<ReloadArgs>>,
    mut status: ResMut<ReloadStatus>,
    asked: Query<&ReloadAsked>,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let attempts = asked.get(called.agent).map_or(0, |asked| asked.attempts);
    let asked = match attempts < MAX_RELOADS {
        true => status.ask(called.agent),
        false => Err(
            "You asked for the most reloads there may be since the user's last \
                      message; tell the user what keeps failing instead.",
        ),
    };
    let output = match asked {
        Ok(()) => {
            let then = called.args.then.clone().unwrap_or_default();
            let marker = ReloadAsked {
                then,
                attempts: attempts.saturating_add(1),
                owed: true,
            };
            commands.entity(called.agent).insert(marker);
            report::keep_run(&mut commands);
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
                 conversation carries on after the restart, and a message then tells you how \
                 the reload went."
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
