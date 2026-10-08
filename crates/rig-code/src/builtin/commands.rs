//! The built-in slash commands: `/model`, `/effort`, `/help` and `/quit`.
//! Each is registered with [`AppExt::add_command`], as any plugin's command
//! would be.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig_core::catalog::Catalog;
use rig_core::completion::options::Reasoning;

use crate::core::agent::{Connection, Effort, Model};
use crate::core::models::{available_models, effort_options, model_id};
use crate::core::registry::{AppExt, CommandInput, Notice, OpenPicker, PickerOption, SlashCommand};

/// Registers the built-in commands.
#[derive(Default)]
pub struct BuiltinCommandsPlugin;

impl Plugin for BuiltinCommandsPlugin {
    fn build(&self, app: &mut App) {
        app.add_command(
            "model",
            "Pick the model, from those with a credential set",
            model,
        )
        .add_command("effort", "Pick how much the model reasons", effort)
        .add_command("help", "List the commands", help)
        .add_command("quit", "Quit", quit);
    }
}

/// `/model` opens a picker of the models with a credential; `/model
/// vendor/model` switches to that model.
fn model(input: In<CommandInput>, mut commands: Commands) {
    if input.args.is_empty() {
        let options: Vec<PickerOption> = available_models()
            .into_iter()
            .map(|spec| PickerOption {
                label: model_id(spec),
                // The catalog does not say what a model outputs, so image
                // models are listed too; without tool calls a model can only
                // talk.
                detail: if spec.tools {
                    spec.display_name.clone()
                } else {
                    format!("{} (no tool calls)", spec.display_name)
                },
                value: model_id(spec),
            })
            .collect();
        if options.is_empty() {
            commands.trigger(Notice::error(
                input.agent,
                "No provider has a credential. Set its API key variable, such as \
                 OPENAI_API_KEY, and restart.",
            ));
            return;
        }
        commands.trigger(OpenPicker {
            entity: input.agent,
            title: "Model".to_owned(),
            command: "model".to_owned(),
            options,
        });
        return;
    }
    match Catalog::builtin().resolve(&input.args) {
        Some(spec) => {
            commands.entity(input.agent).insert(Model(model_id(spec)));
        }
        None => commands.trigger(Notice::error(
            input.agent,
            format!("`{}` is not a catalog model.", input.args),
        )),
    }
}

/// `/effort` opens a picker of the reasoning settings the agent's model
/// takes; `/effort <setting>` picks one.
fn effort(input: In<CommandInput>, agents: Query<Option<&Connection>>, mut commands: Commands) {
    let agent = input.agent;
    let Ok(Some(connection)) = agents.get(agent) else {
        commands.trigger(Notice::error(agent, "Pick a model with /model first."));
        return;
    };
    let levels = effort_options(&connection.spec);
    if levels.is_empty() {
        commands.trigger(Notice::error(
            agent,
            format!(
                "{} has no reasoning settings.",
                connection.spec.display_name
            ),
        ));
        return;
    }
    // `default` goes back to what the model does when no effort is sent.
    let options: Vec<(String, Option<Reasoning>)> = std::iter::once(("default".to_owned(), None))
        .chain(
            levels
                .into_iter()
                .map(|(label, reasoning)| (label, Some(reasoning))),
        )
        .collect();
    if input.args.is_empty() {
        commands.trigger(OpenPicker {
            entity: agent,
            title: format!("Effort for {}", connection.spec.display_name),
            command: "effort".to_owned(),
            options: options
                .into_iter()
                .map(|(label, reasoning)| PickerOption {
                    detail: match reasoning {
                        None => "the model's default".to_owned(),
                        Some(Reasoning::Off) => "no reasoning".to_owned(),
                        Some(Reasoning::Budget { tokens }) => {
                            format!("{tokens} reasoning tokens")
                        }
                        Some(_) => "effort level".to_owned(),
                    },
                    value: label.clone(),
                    label,
                })
                .collect(),
        });
        return;
    }
    match options.iter().find(|(label, _)| *label == input.args) {
        Some((label, reasoning)) => {
            commands.entity(agent).insert(Effort(*reasoning));
            commands.trigger(Notice::info(agent, format!("Effort: {label}.")));
        }
        None => {
            let labels: Vec<&str> = options.iter().map(|(label, _)| label.as_str()).collect();
            commands.trigger(Notice::error(
                agent,
                format!(
                    "{} takes: {}.",
                    connection.spec.display_name,
                    labels.join(", ")
                ),
            ));
        }
    }
}

/// `/help` lists every registered command.
fn help(input: In<CommandInput>, registered: Query<&SlashCommand>, mut commands: Commands) {
    let mut lines: Vec<String> = registered
        .iter()
        .map(|command| format!("/{:<10} {}", command.name, command.help))
        .collect();
    lines.sort();
    lines.push("Esc stops a running turn. Ctrl+C quits.".to_owned());
    commands.trigger(Notice::info(input.agent, lines.join("\n")));
}

/// `/quit` exits the app.
fn quit(_input: In<CommandInput>, mut exit: MessageWriter<AppExit>) {
    exit.write(AppExit::Success);
}
