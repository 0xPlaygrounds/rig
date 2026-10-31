//! The `/model` and `/effort` pickers, which any front that shows pickers
//! shows.

use rig_harness::prelude::*;
use rig_harness::rig_core::catalog::{Connector, ModelSpec, ModelStatus};

use super::ModelItem;

/// What a catalog model is to `/model`: its reference, and its line with
/// why it cannot be used, if it cannot.
pub(super) struct Listed {
    pub(super) reference: String,
    pub(super) line: String,
}

/// Every catalog model: first those that can be reached, as rig-core
/// orders them, then the others with why not.
pub(super) fn catalog(models: &Connector) -> Vec<Listed> {
    let statuses = models.statuses().into_iter();
    let listed = statuses.map(|(spec, status)| {
        let note = match status {
            ModelStatus::Usable => match models.plan(spec) {
                Some(plan) => format!("  ({plan} plan)"),
                None if spec.provider.requires_credential() => String::new(),
                None => "  (no key needed)".to_owned(),
            },
            ModelStatus::NoKey(tried) => format!("  (no key: set {})", tried.join(" or ")),
            ModelStatus::NoTools => "  (calls no tools)".to_owned(),
            ModelStatus::Unavailable(why) => format!("  (unavailable: {why})"),
        };
        listing(spec, &note)
    });
    listed.collect()
}

/// `spec` listed with `note`.
fn listing(spec: &ModelSpec, note: &str) -> Listed {
    let reference = spec.reference();
    Listed {
        line: format!("{reference}  {}{note}", spec.display_name),
        reference,
    }
}

/// Lets the user pick one of the [`ModelItem`]s, then of the catalog's
/// models, each with why it cannot be used, if it cannot.
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
        detail: None,
        more: None,
        checked: false,
    });
    let listed = catalog(&models.0).into_iter().map(|listed| PickItem {
        label: listed.line,
        command: format!("model {}", listed.reference),
        detail: None,
        more: None,
        checked: false,
    });
    let items: Vec<PickItem> = contributed.chain(listed).collect();
    if items.is_empty() {
        notices.write(Notice::error(agent, "The catalog lists no model."));
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
                detail: None,
                more: None,
                checked: false,
            })
            .collect(),
        ..Picker::new(title).for_agent(agent)
    });
}
