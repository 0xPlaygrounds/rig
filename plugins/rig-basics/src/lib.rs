//! The basic slash commands ([`BasicCommandsPlugin`]): `/help`, `/retry`
//! and `/quit`.
//!
//! Every module holds one group of rig's plugins, each listed in
//! `plugins.toml` by the path of the module that defines it, its only
//! path: [`prompt`], [`models`], [`login_chatgpt`],
//! [`sessions`], [`compaction`], [`telemetry`], [`activity`],
//! [`coding_tools`], [`reload`], [`print`](mod@print) and [`subagents`].
//! [`approval`], [`preview`], [`checkpoint`] and [`session_picker`] are the
//! services only some plugins add, and [`headless`] drives a run nobody
//! sits at.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

use rig_ecs::agent::{Notice, Retry};
use rig_harness::commands::{AppCommandsExt, CommandArgs, SlashCommand};
use rig_harness::host::exit::{Exit, ExitReason};

pub mod activity;
pub mod approval;
pub mod checkpoint;
pub mod coding_tools;
pub mod compaction;
pub mod headless;
pub mod login_chatgpt;
pub mod models;
pub mod preview;
pub mod print;
pub mod prompt;
pub mod reload;
pub mod session_picker;
pub mod sessions;
pub mod subagents;
pub mod telemetry;

/// Registers `/help`, `/retry` and `/quit`.
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

fn quit(In(_): In<CommandArgs>, mut commands: Commands) {
    commands.trigger(Exit(ExitReason::Quit));
}
