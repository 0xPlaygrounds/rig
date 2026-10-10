//! Choosing a model and its reasoning setting: `/model`, `/effort` and the picker.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

use crate::front::{PickItem, PickRequest};
use rig_ecs::agent::Notice;
use rig_ecs::commands::{AppCommandsExt, CommandArgs};
use rig_ecs::model::{Connection, Effort, Models, SetEffort, SetModel};

/// Adds `/model` and `/effort`, each a picker without arguments.
#[derive(Default)]
pub struct ModelsPlugin;

impl Plugin for ModelsPlugin {
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
        );
    }
}

fn model(
    In(args): In<CommandArgs>,
    models: Res<Models>,
    mut commands: Commands,
    mut picks: MessageWriter<PickRequest>,
    mut notices: MessageWriter<Notice>,
) {
    if args.args.is_empty() {
        let items: Vec<PickItem> = models
            .0
            .reachable()
            .into_iter()
            .map(|spec| {
                let reference = spec.reference();
                let note = match models.0.plan(spec) {
                    Some(plan) => format!("  ({plan} plan)"),
                    None if spec.provider.requires_credential() => String::new(),
                    None => "  (no key needed)".to_owned(),
                };
                PickItem {
                    label: format!("{reference}  {}{note}", spec.display_name),
                    command: format!("model {reference}"),
                }
            })
            .collect();
        if items.is_empty() {
            notices.write(Notice::error(
                args.agent,
                "No provider with tool-calling models can be reached: set a key such as \
                 OPENAI_API_KEY, or sign in with /login chatgpt.",
            ));
            return;
        }
        picks.write(PickRequest {
            agent: args.agent,
            title: "Model".to_owned(),
            items,
            selected: 0,
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
    agents: Query<&Connection>,
    mut commands: Commands,
    mut picks: MessageWriter<PickRequest>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(Connection { spec, .. }) = agents.get(args.agent) else {
        notices.write(Notice::info(
            args.agent,
            "Pick a model with /model first.".to_owned(),
        ));
        return;
    };
    if args.args.is_empty() {
        picks.write(PickRequest {
            agent: args.agent,
            title: format!("Reasoning for {}", spec.display_name),
            items: spec
                .reasoning
                .choices()
                .into_iter()
                .map(|choice| PickItem {
                    label: choice.label(),
                    command: format!("effort {}", choice.name),
                })
                .collect(),
            selected: 0,
        });
        return;
    }
    match spec.reasoning.named(&args.args) {
        Ok(choice) => {
            commands.trigger(SetEffort {
                entity: args.agent,
                effort: Effort(choice.reasoning),
            });
        }
        Err(why) => {
            let why = format!("{}: {why}.", spec.display_name);
            notices.write(Notice::error(args.agent, why));
        }
    }
}
