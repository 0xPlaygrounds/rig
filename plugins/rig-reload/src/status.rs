//! How a rebuild goes, in the terminal view's status line.

use rig_harness::prelude::*;
use rig_tui::{AppStatus, Side, StatusItem, StatusSystems, Tone};

use crate::ReloadStatus;

/// Where the rebuild is: last on the left.
const RELOAD: StatusItem = StatusItem::at(Side::Left, 90, 16);

pub(crate) fn add(app: &mut App) {
    app.init_resource::<AppStatus>().add_systems(
        PostUpdate,
        show_reload
            .in_set(StatusSystems)
            .run_if(resource_changed::<ReloadStatus>),
    );
}

fn show_reload(reload: Res<ReloadStatus>, mut status: ResMut<AppStatus>) {
    let text = match &*reload {
        ReloadStatus::Idle | ReloadStatus::Failed => String::new(),
        ReloadStatus::Queued { .. } => {
            "Reload queued: once no turn runs (/reload cancel)".to_owned()
        }
        ReloadStatus::Ready => "Reloading: restarting…".to_owned(),
        // The launcher's phase, then cargo's latest line.
        ReloadStatus::Building { latest } => format!(
            "Reloading: {} (Esc cancels)",
            latest.as_deref().unwrap_or("Resolving dependencies…")
        ),
    };
    status.0.show(RELOAD.says(text, Tone::Cyan));
}
