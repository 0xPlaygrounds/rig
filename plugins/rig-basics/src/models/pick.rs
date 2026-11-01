//! The `/model` and `/effort` pickers, which any front that shows pickers
//! shows.

use rig_harness::prelude::*;
use rig_harness::rig_core::catalog::{Connector, ModelSpec, ModelStatus};

use super::ModelItem;

/// What a catalog model is to `/model`: its reference, and its line.
pub(super) struct Listed {
    pub(super) reference: String,
    pub(super) line: String,
}

/// The catalog models that can be used now, as rig-core orders them, and
/// those of a signed-in plan still being checked.
pub(super) fn catalog(models: &Connector) -> Vec<Listed> {
    let statuses = models.statuses().into_iter();
    let listed = statuses.filter_map(|(spec, status)| {
        let note = match status {
            ModelStatus::Usable => match models.plan(spec) {
                Some(plan) => format!("  ({plan} plan)"),
                None if spec.provider.requires_credential() => String::new(),
                None => "  (no key needed)".to_owned(),
            },
            ModelStatus::Checking => "  (checking your plan)".to_owned(),
            ModelStatus::NotOffered(_)
            | ModelStatus::NoKey(_)
            | ModelStatus::Unconfigured
            | ModelStatus::NoTools
            | ModelStatus::Unavailable(_) => return None,
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

/// The [`ModelItem`]s, then the catalog's models that can be used, after
/// the agent's `current` model when it is not one of them, such as one
/// whose key is gone; the current one says so. Returns the lines and the
/// one to select at first.
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
    let mut listed = catalog(&models.0);
    let chosen = current.and_then(|current| models.0.catalog().resolve(&current.0).ok());
    if let Some(chosen) = chosen
        && !listed
            .iter()
            .any(|it| it.reference == chosen.spec.reference())
    {
        // Why it cannot be used, as it is listed anyway.
        let note = match models.0.status(chosen.spec) {
            ModelStatus::NoKey(tried) => format!("  (no key: set {})", tried.join(" or ")),
            ModelStatus::Unconfigured => "  (at its default address)".to_owned(),
            ModelStatus::NoTools => "  (calls no tools)".to_owned(),
            ModelStatus::Unavailable(why) | ModelStatus::NotOffered(why) => {
                format!("  (unavailable: {why})")
            }
            ModelStatus::Usable | ModelStatus::Checking => String::new(),
        };
        listed.insert(0, listing(chosen.spec, &note));
    }
    let current = current.map(|current| format!("model {}", current.0));
    let listed = listed.into_iter().map(|listed| PickItem {
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
/// `current` model at first; with none, says how to make one usable.
pub(super) fn pick_model(
    agent: Entity,
    (models, contributed, current): (&Models, &Query<&ModelItem>, Option<&ModelChoice>),
    notices: &mut MessageWriter<Notice>,
    commands: &mut Commands,
) {
    let (items, selected) = model_items(models, contributed, current);
    if items.is_empty() {
        notices.write(Notice::error(agent, nothing_usable(models)));
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

/// Why no model can be used, and how to make one usable: a sign-in's own
/// reason, when its models are why.
fn nothing_usable(models: &Models) -> String {
    let statuses = models.0.statuses().into_iter();
    let signed_in = statuses.filter(|(spec, _)| models.0.plan(spec).is_some());
    let why = signed_in
        .filter_map(|(_, status)| match status {
            ModelStatus::Unavailable(why) | ModelStatus::NotOffered(why) => Some(why),
            _ => None,
        })
        .next();
    let why = why.unwrap_or_else(|| {
        "export a provider's API key, such as OPENAI_API_KEY, or sign in with /login".to_owned()
    });
    format!("No model can be used now: {why}.")
}

/// Keeps the lines of an open `/model` picker current once the [`Models`]
/// changed, such as when a sign-in learned which models its account
/// offers, before the terminal view finds its selection again; closes one
/// left with no line, saying why, as opening it would have.
pub(super) fn relist(
    models: Res<Models>,
    contributed: Query<&ModelItem>,
    chosen: Query<&ModelChoice>,
    mut lists: Query<(Entity, &mut Picker), With<ModelList>>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    if !models.is_changed() {
        return;
    }
    for (list, mut picker) in &mut lists {
        let current = picker.agent.and_then(|agent| chosen.get(agent).ok());
        let (items, selected) = model_items(&models, &contributed, current);
        if items.is_empty() {
            commands.entity(list).despawn();
            if let Some(agent) = picker.agent {
                notices.write(Notice::error(agent, nothing_usable(&models)));
            }
        } else if picker.items != items {
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
