//! The terminal view's part: the `/model` and `/effort` pickers, and each
//! agent's model and reasoning setting in its status line.

use rig_harness::prelude::*;
use rig_harness::rig_core::catalog::ModelSpec;
use rig_tui::{PickItem, Picker, Side, StatusItem, StatusItems, StatusSystems, Tone};

use super::{ModelItem, ModelLabel};

/// Where the model is: always shown.
const MODEL: StatusItem = StatusItem::at(Side::Left, 30, u8::MAX);
/// Where the reasoning setting is.
const REASONING: StatusItem = StatusItem::at(Side::Left, 40, 12);

pub(super) fn add(app: &mut App) {
    app.add_systems(PostUpdate, show.in_set(StatusSystems));
}

/// Lets the user pick one of the [`ModelItem`]s, then of the models that
/// can be reached.
pub(super) fn pick_model(
    agent: Entity,
    (models, contributed): (&Models, &Query<&ModelItem>),
    notices: &mut MessageWriter<Notice>,
    commands: &mut Commands,
) {
    let mut contributed: Vec<&ModelItem> = contributed.iter().collect();
    contributed.sort_by(|a, b| a.name.cmp(&b.name));
    let contributed = contributed.into_iter().map(|item| PickItem {
        label: item.label.clone(),
        command: item.command.clone(),
    });
    let reachable = models.0.reachable().into_iter().map(|spec| {
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
    });
    let items: Vec<PickItem> = contributed.chain(reachable).collect();
    if items.is_empty() {
        notices.write(Notice::error(
            agent,
            "No provider with tool-calling models can be reached: set a key such as \
             OPENAI_API_KEY, or sign in with /login chatgpt.",
        ));
        return;
    }
    commands.spawn(Picker {
        items,
        ..Picker::new("Model").for_agent(agent)
    });
}

/// Lets the user pick one of the reasoning settings of `spec`.
pub(super) fn pick_effort(agent: Entity, spec: &ModelSpec, commands: &mut Commands) {
    let title = format!("Reasoning for {}", spec.display_name);
    commands.spawn(Picker {
        items: spec
            .reasoning
            .choices()
            .into_iter()
            .map(|choice| PickItem {
                label: choice.label(),
                command: format!("effort {}", choice.name),
            })
            .collect(),
        ..Picker::new(title).for_agent(agent)
    });
}

/// Each agent's model, its [`ModelLabel`] or else its choice, and its
/// reasoning setting, when one of them changed.
fn show(
    mut agents: Query<(
        Entity,
        Option<&ModelChoice>,
        Option<&ModelLabel>,
        &Effort,
        &mut StatusItems,
    )>,
    changed: Query<(), Or<(Changed<ModelChoice>, Changed<Effort>, Changed<ModelLabel>)>>,
    mut unlabeled: RemovedComponents<ModelLabel>,
) {
    let unlabeled: Vec<Entity> = unlabeled.read().collect();
    for (agent, model, label, effort, mut items) in &mut agents {
        if !changed.contains(agent) && !unlabeled.contains(&agent) {
            continue;
        }
        let model = match (label, model) {
            (Some(label), _) => &label.0,
            (None, Some(model)) => &model.0,
            (None, None) => "no model: /model picks one",
        };
        items.show(MODEL.says(model, Tone::Bold));
        items.show(REASONING.says(format!("reasoning {effort}"), Tone::Dim));
    }
}
