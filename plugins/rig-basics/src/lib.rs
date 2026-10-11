//! The basic slash commands: `/help`, `/retry` and `/quit`, with the
//! default `tui` feature `/agents` and the agent tree in each agent's
//! status line ([`BasicCommandsPlugin`]), and the project context in the
//! system prompt ([`ProjectContextPlugin`]): `AGENTS.md`, the environment
//! and `/context`.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

use rig_ecs::agent::{Notice, Retry};
use rig_ecs::commands::{AppCommandsExt, CommandArgs, SlashCommand};

pub mod project_context;
#[cfg(feature = "tui")]
mod tui;

pub use project_context::ProjectContextPlugin;

/// Registers the basic commands; with the `tui` feature, `/agents` and the
/// agent tree it picks from in each agent's status line.
#[derive(Default)]
pub struct BasicCommandsPlugin;

impl Plugin for BasicCommandsPlugin {
    fn build(&self, app: &mut App) {
        app.add_command(
            "retry",
            "Send the conversation again after a failed model call",
            retry,
        )
        .add_command("help", "List the commands", help)
        .add_command("quit", "Quit; the session stays for /resume", quit);
        #[cfg(feature = "tui")]
        tui::add(app);
    }
}

/// `/retry`: sends the conversation again; it takes no arguments.
fn retry(In(args): In<CommandArgs>, mut commands: Commands, mut notices: MessageWriter<Notice>) {
    if args.args.is_empty() {
        commands.trigger(Retry { entity: args.agent });
    } else {
        let why = format!("/retry takes no arguments, not `{}`.", args.args);
        notices.write(Notice::error(args.agent, why));
    }
}

/// `/help`: every command with its help.
fn help(
    In(args): In<CommandArgs>,
    commands: Query<(&Name, &SlashCommand)>,
    mut notices: MessageWriter<Notice>,
) {
    let mut lines: Vec<String> = commands
        .iter()
        .map(|(name, command)| format!("{:<11} {}", name.as_str(), command.help))
        .collect();
    lines.sort();
    notices.write(Notice::info(args.agent, lines.join("\n")));
}

fn quit(In(_): In<CommandArgs>, mut exit: MessageWriter<AppExit>) {
    exit.write(AppExit::Success);
}
