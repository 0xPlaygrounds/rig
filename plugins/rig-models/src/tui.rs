//! The terminal view's part: the `/model` and `/effort` pickers, and each
//! agent's model and reasoning setting in its status line.

use rig_harness::prelude::*;
use rig_harness::rig_core::catalog::ModelSpec;
use rig_tui::{PickItem, PickRequest, Side, StatusItem, StatusItems, StatusSystems, Tone};

/// Where the model is: always shown.
const MODEL: StatusItem = StatusItem::at(Side::Left, 30, u8::MAX);
/// Where the reasoning setting is.
const REASONING: StatusItem = StatusItem::at(Side::Left, 40, 12);

pub(super) fn add(app: &mut App) {
    app.add_message::<PickRequest>()
        .add_systems(PostUpdate, show.in_set(StatusSystems));
}

/// Lets the user pick one of the models that can be reached.
pub(super) fn pick_model(
    agent: Entity,
    models: &Models,
    picks: &mut MessageWriter<PickRequest>,
    notices: &mut MessageWriter<Notice>,
) {
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
            agent,
            "No provider with tool-calling models can be reached: set a key such as \
             OPENAI_API_KEY, or sign in with /login chatgpt.",
        ));
        return;
    }
    picks.write(PickRequest {
        agent,
        title: "Model".to_owned(),
        items,
        selected: 0,
    });
}

/// Lets the user pick one of the reasoning settings of `spec`.
pub(super) fn pick_effort(agent: Entity, spec: &ModelSpec, picks: &mut MessageWriter<PickRequest>) {
    picks.write(PickRequest {
        agent,
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
}

fn show(
    mut agents: Query<
        (Option<&ModelChoice>, &Effort, &mut StatusItems),
        Or<(Changed<ModelChoice>, Changed<Effort>)>,
    >,
) {
    for (model, effort, mut items) in &mut agents {
        let model = model.map_or("no model: /model picks one", |model| &model.0);
        items.show(MODEL.says(model, Tone::Bold));
        items.show(REASONING.says(format!("reasoning {effort}"), Tone::Dim));
    }
}
