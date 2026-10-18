//! The session's name in the status line.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig_harness::status::{AppStatus, Side, StatusItem, StatusSystems, Tone};

use super::SessionTitle;

/// Where the session's name is in the status line: first, and gone before
/// what the session spends when the line is too narrow.
const NAME: StatusItem = StatusItem::new("session", Side::Left, 10, 2);

pub(super) fn add(app: &mut App) {
    app.init_resource::<AppStatus>().add_systems(
        PostUpdate,
        show_name
            .in_set(StatusSystems::Show)
            .run_if(resource_changed::<SessionTitle>),
    );
}

fn show_name(title: Res<SessionTitle>, mut status: ResMut<AppStatus>) {
    let name = title.name.clone().unwrap_or_default();
    status.0.show(NAME.says(name, Tone::Cyan));
}
