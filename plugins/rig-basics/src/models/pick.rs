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

/// Catalog models except those a signed-in account does not offer: first
/// those that can be reached, as rig-core orders them, then why the rest cannot.
pub(super) fn catalog(models: &Connector) -> Vec<Listed> {
    let statuses = models.statuses().into_iter();
    let listed = statuses.filter_map(|(spec, status)| {
        let note = match status {
            ModelStatus::Usable => match models.plan(spec) {
                Some(plan) => format!("  ({plan} plan)"),
                None if spec.provider.requires_credential() => String::new(),
                None => "  (no key needed)".to_owned(),
            },
            ModelStatus::NoKey(tried) => format!("  (no key: set {})", tried.join(" or ")),
            ModelStatus::NoTools => "  (calls no tools)".to_owned(),
            ModelStatus::Unavailable(why) => format!("  (unavailable: {why})"),
            ModelStatus::Checking => "  (checking your plan)".to_owned(),
            ModelStatus::NotOffered(_) => return None,
        };
        Some(listing(spec, &note))
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

/// The `/model` picker, whose lines follow the catalog's models.
#[derive(Component)]
pub(super) struct ModelList;

/// The [`ModelItem`]s, then the catalog's models, each with why it cannot
/// be used, if it cannot; the agent's `current` model says so. Returns the
/// lines and the one to select at first.
fn model_items(
    models: &Models,
    contributed: &Query<&ModelItem>,
    current: Option<&ModelChoice>,
) -> (Vec<PickItem>, usize) {
    let mut contributed: Vec<&ModelItem> = contributed.iter().collect();
    contributed.sort_by(|a, b| a.name.cmp(&b.name));
    let contributed = contributed.into_iter().map(|item| PickItem {
        label: item.label.clone(),
        command: item.command.clone(),
        detail: None,
        more: None,
        checked: false,
    });
    let current = current.map(|current| format!("model {}", current.0));
    let listed = catalog(&models.0).into_iter().map(|listed| PickItem {
        label: listed.line,
        command: format!("model {}", listed.reference),
        detail: None,
        more: None,
        checked: false,
    });
    let mut items: Vec<PickItem> = contributed.chain(listed).collect();
    let at = items
        .iter()
        .position(|item| Some(&item.command) == current.as_ref());
    if let Some(item) = at.and_then(|at| items.get_mut(at)) {
        item.detail = Some("current".to_owned());
    }
    (items, at.unwrap_or(0))
}

/// Lets the user pick one of the [`model_items`], selecting the agent's
/// `current` model at first.
pub(super) fn pick_model(
    agent: Entity,
    (models, contributed, current): (&Models, &Query<&ModelItem>, Option<&ModelChoice>),
    notices: &mut MessageWriter<Notice>,
    commands: &mut Commands,
) {
    let (items, selected) = model_items(models, contributed, current);
    if items.is_empty() {
        notices.write(Notice::error(agent, "The catalog lists no model."));
        return;
    }
    commands.spawn((
        Picker {
            items,
            selected,
            ..Picker::new("Model").for_agent(agent)
        },
        ModelList,
    ));
}

/// Keeps the lines of an open `/model` picker current once the [`Models`]
/// changed, such as when a sign-in learned which models its account
/// offers, before the terminal view finds its selection again.
pub(super) fn relist(
    models: Res<Models>,
    contributed: Query<&ModelItem>,
    chosen: Query<&ModelChoice>,
    mut lists: Query<&mut Picker, With<ModelList>>,
) {
    if !models.is_changed() {
        return;
    }
    for mut picker in &mut lists {
        let current = picker.agent.and_then(|agent| chosen.get(agent).ok());
        let (items, selected) = model_items(&models, &contributed, current);
        if picker.items != items {
            // Read only by a picker not shown yet, such as one queued behind
            // another; a shown one keeps its own selection.
            picker.items = items;
            picker.selected = selected;
        }
    }
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
