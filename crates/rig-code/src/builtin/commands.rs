//! The built-in slash commands: `/model`, `/effort`, `/help` and `/quit`.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

use crate::core::agent::{ModelChoice, Notice, PickKind, PickRequest, SetEffort, SetModel};
use crate::core::commands::{AppCommandsExt, CommandArgs, SlashCommand};
use crate::core::models;

/// Registers the built-in commands with [`AppCommandsExt::add_command`].
pub struct BuiltinCommandsPlugin;

impl Plugin for BuiltinCommandsPlugin {
    fn build(&self, app: &mut App) {
        app.add_command(
            "model",
            "Pick the model, or set it with /model vendor/model",
            model,
        )
        .add_command(
            "effort",
            "Pick the reasoning setting, or set it with /effort <level>",
            effort,
        )
        .add_command("help", "List the commands", help)
        .add_command("quit", "Save and quit", quit);
    }
}

fn model(In(args): In<CommandArgs>, mut commands: Commands, mut picks: MessageWriter<PickRequest>) {
    if args.args.is_empty() {
        picks.write(PickRequest {
            agent: args.agent,
            kind: PickKind::Model,
        });
    } else {
        commands.trigger(SetModel {
            entity: args.agent,
            model: args.args,
        });
    }
}

fn effort(
    In(args): In<CommandArgs>,
    agents: Query<&ModelChoice>,
    mut commands: Commands,
    mut picks: MessageWriter<PickRequest>,
    mut notices: MessageWriter<Notice>,
) {
    let Some(spec) = agents
        .get(args.agent)
        .ok()
        .and_then(|choice| choice.0.as_deref())
        .and_then(models::resolve)
    else {
        notices.write(Notice("Pick a model with /model first.".to_owned()));
        return;
    };
    if args.args.is_empty() {
        picks.write(PickRequest {
            agent: args.agent,
            kind: PickKind::Effort,
        });
        return;
    }
    let options = models::effort_options(spec);
    match options
        .iter()
        .find(|(label, _)| label.split_whitespace().next() == Some(args.args.as_str()))
    {
        Some((_, effort)) => {
            commands.trigger(SetEffort {
                entity: args.agent,
                effort: *effort,
            });
        }
        None => {
            let labels: Vec<&str> = options
                .iter()
                .filter_map(|(label, _)| label.split_whitespace().next())
                .collect();
            notices.write(Notice(format!(
                "{} takes: {}.",
                spec.display_name,
                labels.join(", ")
            )));
        }
    }
}

fn help(
    In(_): In<CommandArgs>,
    commands: Query<&SlashCommand>,
    mut notices: MessageWriter<Notice>,
) {
    let mut lines: Vec<String> = commands
        .iter()
        .map(|command| format!("/{:<10} {}", command.name, command.help))
        .collect();
    lines.sort();
    lines.push("Esc stops a running turn. Ctrl+C clears the input or quits.".to_owned());
    notices.write(Notice(lines.join("\n")));
}

fn quit(In(_): In<CommandArgs>, mut exit: MessageWriter<AppExit>) {
    exit.write(AppExit::Success);
}
