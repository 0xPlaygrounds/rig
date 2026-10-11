//! The terminal view's part: the `/resume` picker, and the session's name
//! in the status line.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig_ecs::agent::Notice;
use rig_harness::harness_protocol::Home;
use rig_harness::prelude::SessionPaths;
use rig_tui::{AppStatus, PickItem, PickRequest, Side, StatusItem, StatusSystems, Tone};

use super::{SessionTitle, list};

/// Where the session's name is in the status line: first, and gone before
/// what the session spends when the line is too narrow.
const NAME: StatusItem = StatusItem::at(Side::Left, 10, 2);

pub(super) fn add(app: &mut App) {
    app.init_resource::<AppStatus>()
        .add_message::<PickRequest>()
        .add_systems(
            PostUpdate,
            show_name
                .in_set(StatusSystems)
                .run_if(resource_changed::<SessionTitle>),
        );
}

fn show_name(title: Res<SessionTitle>, mut status: ResMut<AppStatus>) {
    let name = title.name.clone().unwrap_or_default();
    status.0.show(NAME.says(name, Tone::Cyan));
}

/// Lets the user pick an earlier session to resume.
pub(super) fn pick_session(
    agent: Entity,
    paths: Option<&SessionPaths>,
    picks: &mut MessageWriter<PickRequest>,
    notices: &mut MessageWriter<Notice>,
) {
    let Some(paths) = paths else {
        return;
    };
    let items: Vec<PickItem> = list(&Home::from_env(), paths.path())
        .into_iter()
        .map(|session| PickItem {
            label: session.label(),
            command: format!("resume {}", session.id),
        })
        .collect();
    if items.is_empty() {
        notices.write(Notice::info(agent, "No earlier session to resume."));
        return;
    }
    picks.write(PickRequest {
        agent,
        title: "Resume a session".to_owned(),
        items,
        selected: 0,
    });
}
