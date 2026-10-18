//! The terminal view's part: the `/model` and `/effort` pickers, `/model`'s
//! completions, and each agent's model and reasoning setting in its status
//! line.

use rig_harness::prelude::*;
use rig_harness::rig_core::catalog::{Connector, ModelSpec, ModelStatus};
use rig_harness::status::{Side, StatusItem, StatusItems, StatusSystems, Tone};
use rig_tui::{CompletionSource, Editor, MainEditor};

use super::{ModelItem, ModelLabel};

/// Where the model is: always shown.
const MODEL: StatusItem = StatusItem::new("model", Side::Left, 30, u8::MAX);
/// Where the reasoning setting is.
const REASONING: StatusItem = StatusItem::new("reasoning", Side::Left, 40, 12);

pub(super) fn add(app: &mut App) {
    app.add_systems(PostUpdate, show.in_set(StatusSystems::Show))
        .add_systems(Update, complete_models);
}

/// What a catalog model is to `/model`: its reference, and its line with
/// why it cannot be used, if it cannot.
struct Listed {
    reference: String,
    line: String,
}

/// Every catalog model: first those that can be reached, as rig-core
/// orders them, then the others with why not.
fn catalog(models: &Connector) -> Vec<Listed> {
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
        checked: false,
    });
    let listed = catalog(&models.0).into_iter().map(|listed| PickItem {
        label: listed.line,
        command: format!("model {}", listed.reference),
        detail: None,
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

/// On the entity of `/model`'s completions.
#[derive(Component)]
struct ModelCompletions;

/// Offers the [`ModelItem`]s' names and the catalog's models as `/model`'s
/// argument, each with its line: built once the input is a `/model`
/// command, again only after the catalog or the choices plugins add
/// changed. Listing the catalog asks each provider whether it connects,
/// too slow for every start.
fn complete_models(
    models: Res<Models>,
    items: Query<&ModelItem>,
    changed: Query<(), Changed<ModelItem>>,
    mut gone: RemovedComponents<ModelItem>,
    input: Option<Single<&Editor, With<MainEditor>>>,
    completions: Query<Entity, With<ModelCompletions>>,
    mut current: Local<bool>,
    mut commands: Commands,
) {
    if models.is_changed() || !changed.is_empty() || gone.read().count() > 0 {
        *current = false;
    }
    let typed = input.is_some_and(|input| {
        let command = input.text().strip_prefix('/');
        command.and_then(|line| line.split_whitespace().next()) == Some("model")
    });
    if *current || !typed {
        return;
    }
    *current = true;
    let mut contributed: Vec<&ModelItem> = items.iter().collect();
    contributed.sort_by(|a, b| a.name.cmp(&b.name));
    let mut completion = CompletionSource::arg("model");
    for item in contributed {
        completion = completion.candidate(item.name.clone(), item.label.clone());
    }
    for listed in catalog(&models.0) {
        let detail = listed
            .line
            .strip_prefix(&listed.reference)
            .unwrap_or_default();
        completion = completion.candidate(listed.reference.clone(), detail.trim_start());
    }
    match completions.iter().next() {
        Some(entity) => {
            commands.entity(entity).insert(completion);
        }
        None => {
            commands.spawn((Name::new("completions:model"), ModelCompletions, completion));
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
                checked: false,
            })
            .collect(),
        ..Picker::new(title).for_agent(agent)
    });
}

/// Each agent's model, its [`ModelLabel`] or else its choice, and its
/// reasoning setting, when one of them changed.
fn show(
    mut agents: Query<(
        Option<&ModelChoice>,
        Option<&ModelLabel>,
        &ReasoningSetting,
        &mut StatusItems,
    )>,
    changed: Query<
        Entity,
        Or<(
            Changed<ModelChoice>,
            Changed<ReasoningSetting>,
            Changed<ModelLabel>,
        )>,
    >,
    mut unlabeled: RemovedComponents<ModelLabel>,
) {
    for agent in changed.iter().chain(unlabeled.read()) {
        let Ok((model, label, effort, mut items)) = agents.get_mut(agent) else {
            continue;
        };
        let model = match (label, model) {
            (Some(label), _) => &label.0,
            (None, Some(model)) => &model.0,
            (None, None) => "no model: /model picks one",
        };
        items.show(MODEL.says(model, Tone::Bold));
        items.show(REASONING.says(format!("reasoning {effort}"), Tone::Dim));
    }
}
