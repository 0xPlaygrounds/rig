//! The built-in slash commands: `/model`, `/effort`, `/help` and `/quit`.
//! Each is registered through [`AgentAppExt::add_command`], the same call a
//! third-party plugin makes.

use bevy::prelude::*;
use rig_core::catalog::Catalog;

use crate::core::{
    AgentAppExt, AgentDefaults, Choice, EffortChoice, ModelChoice, ModelEndpoint, Notice,
    OfferChoices, RunCommand, SlashCommand, available_models, effort_label, effort_options,
    model_reference,
};

/// Adds the built-in slash commands.
#[derive(Default)]
pub struct BuiltinCommands;

impl Plugin for BuiltinCommands {
    fn build(&self, app: &mut App) {
        app.add_command(
            SlashCommand::new("model", "pick the model: /model [vendor/model]"),
            model,
        )
        .add_command(
            SlashCommand::new("effort", "pick the reasoning effort: /effort [level]"),
            effort,
        )
        .add_command(SlashCommand::new("help", "list the commands"), help)
        .add_command(SlashCommand::new("quit", "exit"), quit);
    }
}

/// Without an argument, offers the models that have a credential; with
/// one, switches the agent to it.
fn model(
    run: On<RunCommand>,
    mut commands: Commands,
    mut defaults: ResMut<AgentDefaults>,
    mut offers: MessageWriter<OfferChoices>,
    mut notices: MessageWriter<Notice>,
) {
    let agent = run.agent;
    if run.args.is_empty() {
        let choices: Vec<Choice> = available_models()
            .map(|spec| {
                let reference = model_reference(spec);
                Choice {
                    label: format!("{reference}  {}", spec.display_name),
                    value: reference,
                }
            })
            .collect();
        if choices.is_empty() {
            notices.write(Notice::error(
                agent,
                "no model has a credential: set a provider's API key variable, such as \
                 OPENAI_API_KEY, or type /model vendor/model",
            ));
            return;
        }
        offers.write(OfferChoices {
            agent,
            title: "model".to_owned(),
            command: "model".to_owned(),
            choices,
        });
        return;
    }
    let Some(spec) = Catalog::builtin().resolve(&run.args) else {
        notices.write(Notice::error(
            agent,
            format!("{} is not in the model catalog", run.args),
        ));
        return;
    };
    let reference = model_reference(spec);
    commands
        .entity(agent)
        .insert(ModelChoice(Some(reference.clone())));
    defaults.model = Some(reference.clone());
    notices.write(Notice::info(
        agent,
        format!("model: {reference} ({})", spec.display_name),
    ));
}

/// Without an argument, offers the reasoning settings the agent's model
/// takes; with one, picks it.
fn effort(
    run: On<RunCommand>,
    mut commands: Commands,
    agents: Query<Option<&ModelEndpoint>>,
    mut defaults: ResMut<AgentDefaults>,
    mut offers: MessageWriter<OfferChoices>,
    mut notices: MessageWriter<Notice>,
) {
    let agent = run.agent;
    let Ok(Some(endpoint)) = agents.get(agent) else {
        notices.write(Notice::error(agent, "pick a model with /model first"));
        return;
    };
    let options = effort_options(endpoint.spec);
    if options.is_empty() {
        notices.write(Notice::info(
            agent,
            format!("{} has no effort setting", endpoint.spec.display_name),
        ));
        return;
    }
    if run.args.is_empty() {
        offers.write(OfferChoices {
            agent,
            title: format!("effort for {}", endpoint.spec.display_name),
            command: "effort".to_owned(),
            choices: options
                .iter()
                .map(|(label, _)| Choice {
                    value: label.clone(),
                    label: label.clone(),
                })
                .collect(),
        });
        return;
    }
    let Some((_, reasoning)) = options.iter().find(|(label, _)| *label == run.args) else {
        let labels: Vec<&str> = options.iter().map(|(label, _)| label.as_str()).collect();
        notices.write(Notice::error(
            agent,
            format!(
                "{} takes: {}",
                endpoint.spec.display_name,
                labels.join(", ")
            ),
        ));
        return;
    };
    commands
        .entity(agent)
        .insert(EffortChoice(Some(*reasoning)));
    defaults.effort = Some(*reasoning);
    notices.write(Notice::info(
        agent,
        format!("effort: {}", effort_label(endpoint.spec, reasoning)),
    ));
}

/// Lists every registered command.
fn help(run: On<RunCommand>, commands: Query<&SlashCommand>, mut notices: MessageWriter<Notice>) {
    let mut lines: Vec<String> = commands
        .iter()
        .map(|command| format!("/{}  {}", command.name, command.description))
        .collect();
    lines.sort();
    lines.push("Esc stops a turn; Ctrl+C clears the input, then quits.".to_owned());
    notices.write(Notice::info(run.agent, lines.join("\n")));
}

fn quit(_run: On<RunCommand>, mut exit: MessageWriter<AppExit>) {
    exit.write(AppExit::Success);
}
