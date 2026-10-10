//! The `reload` tool: the agent rebuilds and restarts its own harness, to
//! apply its own plugin changes. Also the system prompt's section on what
//! the agent is, made from its world: the build's plugins, commands and
//! tools, and how the agent extends itself.

use std::path::Path;

use rig_ecs::commands::SlashCommand;
use rig_ecs::tools::ToolDef;
use rig_harness::harness_protocol::Home;
use schemars::JsonSchema;
use serde::Deserialize;

use rig_harness::prelude::*;

/// The tool's name.
pub const RELOAD_TOOL: &str = "reload";

const RELOAD_DESCRIPTION: &str = "Rebuild the agent with the plugins in plugins.toml and \
    restart on the new build, to apply your own plugin changes. Nothing happens mid-turn: the \
    build starts once this turn and every other running turn have ended, so finish your \
    edits first and end your turn after this call. The user sees a notice and can cancel it. \
    After the restart the session, with this conversation, carries on. If the build fails, \
    this build keeps running and its first errors arrive in your conversation as a note.";

/// Offers the model the `reload` tool when the launcher started the
/// agent, since only then can it rebuild itself, and spawns the system
/// prompt's section on what the agent is.
#[derive(Default)]
pub struct ReloadTool;

impl Plugin for ReloadTool {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, describe);
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

/// Spawns the system prompt's section on what this agent is: its plugins,
/// commands and tools, and, when the launcher started it, how it extends
/// itself. Made once, so the prompt stays cached.
fn describe(
    plugins: Query<(&Name, &PluginSource)>,
    slash_commands: Query<&Name, With<SlashCommand>>,
    tools: Query<&Name, With<ToolDef>>,
    mut commands: Commands,
) {
    let list = |mut names: Vec<String>| {
        names.sort();
        names.join(", ")
    };
    let plugins = list(
        plugins
            .iter()
            .map(|(name, source)| {
                let short = name.rsplit("::").next().unwrap_or_default();
                format!("{short} ({})", source.krate)
            })
            .collect(),
    );
    let slash_commands = list(slash_commands.iter().map(Name::to_string).collect());
    let tools = list(tools.iter().map(|name| name.replace("tool:", "")).collect());
    let mut text = format!(
        "You are rig, a coding agent that is a Bevy app made of plugins. Plugins: {plugins}. \
         Slash commands, which the user types: {slash_commands}. Tools: {tools}."
    );
    if let Some(launcher) = launcher::executable() {
        let home = Home::from_env();
        text.push_str(&format!(
            "\nYou extend yourself with plugins, Bevy plugins in crates of their own. \
             `$RIG_LAUNCHER plugin new <name>` (the launcher is {launcher}) makes a working one \
             in {home}/plugins/<name> and lists it in {home}/plugins.toml; change that list only \
             with `$RIG_LAUNCHER plugin add` or `remove`, and never edit rig's own crates. The \
             plugin cookbook {guide} shows each extension point. Check a change with \
             `$RIG_LAUNCHER plugin check --build`, not cargo, then call the `reload` tool and end \
             your turn: the agent rebuilds and restarts in this session. A failed build keeps \
             this one running and its errors come to you as a note.",
            launcher = Path::new(&launcher).display(),
            home = home.root().display(),
            guide = rig_harness::PLUGIN_GUIDE,
        ));
    }
    commands.spawn((
        Name::new("prompt:rig_harness"),
        PromptSection::new(PromptSection::ORDER_PROJECT - 100, "rig_harness", text),
    ));
}
