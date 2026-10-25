//! The basic slash commands ([`BasicCommandsPlugin`]): `/help` and `/quit`.
//!
//! Every module holds one group of rig's plugins, each listed in
//! `plugins.toml` by the path of the module that defines it, its only
//! path: [`prompt`], [`agents_md`], [`skills`], [`models`], [`login`],
//! [`sessions`], [`compaction`], [`telemetry`], [`activity`],
//! [`coding_tools`], [`reload`], [`print`](mod@print), [`helpers`], [`subagents`] and
//! [`welcome`].
//! [`approval`], [`preview`], [`checkpoint`] and [`session_picker`] are the
//! services only some plugins add.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

use rig_ecs::agent::Notice;
use rig_harness::commands::{AppCommandsExt, CommandArgs, SlashCommand};
use rig_harness::host::exit::{Exit, ExitReason};

pub mod activity;
pub mod agents_md;
pub mod approval;
pub mod checkpoint;
pub mod coding_tools;
pub mod compaction;
pub mod helpers;
pub mod login;
pub mod models;
pub mod preview;
pub mod print;
pub mod prompt;
pub mod reload;
pub mod session_picker;
pub mod sessions;
pub mod skills;
pub mod subagents;
pub mod telemetry;
pub mod welcome;

/// Registers `/help` and `/quit`.
#[derive(Default)]
pub struct BasicCommandsPlugin;

impl Plugin for BasicCommandsPlugin {
    fn build(&self, app: &mut App) {
        app.add_command("help", "List the commands", help)
            .add_command("quit", "Quit; the session stays for /resume", quit);
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

fn quit(In(_): In<CommandArgs>, mut commands: Commands) {
    commands.trigger(Exit(ExitReason::Quit));
}

#[cfg(test)]
mod tests;
