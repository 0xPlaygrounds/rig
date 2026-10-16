//! The running count in the terminal view's status line.

use rig_harness::prelude::*;
use rig_tui::{AppStatus, Side, StatusItem, StatusSystems, Tone};

use crate::background_jobs::LiveJob;

/// After the other agents at work.
const JOBS: StatusItem = StatusItem::new("background-jobs", Side::Left, 75, 9);

pub(crate) fn add(app: &mut App) {
    app.init_resource::<AppStatus>().add_systems(
        PostUpdate,
        show_jobs.in_set(StatusSystems::Show).run_if(jobs_changed),
    );
}

fn jobs_changed(mut gone: RemovedComponents<LiveJob>, added: Query<(), Added<LiveJob>>) -> bool {
    gone.read().count() > 0 || !added.is_empty()
}

fn show_jobs(live: Query<(), With<LiveJob>>, mut status: ResMut<AppStatus>) {
    let text = match live.iter().count() {
        0 => String::new(),
        1 => "1 job".to_owned(),
        count => format!("{count} jobs"),
    };
    status.0.show(JOBS.says(text, Tone::Dim));
}
