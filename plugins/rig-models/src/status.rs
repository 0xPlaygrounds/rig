//! Each agent's model and reasoning setting, in its status line.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

use rig_ecs::model::{Effort, ModelChoice};
use rig_harness::front::{Side, StatusItem, StatusItems, StatusSystems, Tone};

/// Where the model is: always shown.
const MODEL: StatusItem = StatusItem::at(Side::Left, 30, u8::MAX);
/// Where the reasoning setting is.
const REASONING: StatusItem = StatusItem::at(Side::Left, 40, 12);

pub(super) fn add(app: &mut App) {
    app.add_systems(PostUpdate, show.in_set(StatusSystems));
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
