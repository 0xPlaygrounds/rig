//! A run nobody sits at (`--print`) waits for a reload instead of ending
//! ([`KeepRunning`]), and restarts on the new build.

use rig_harness::prelude::*;

use crate::print::KeepRunning;
use crate::reload::ReloadStatus;

pub(crate) fn add(app: &mut App) {
    app.add_systems(Update, hold_headless_run);
}

/// Keeps a run nobody sits at going until the reload is done: it restarts
/// on the new build, or a failed build leaves this one running.
pub(crate) fn keep_run(commands: &mut Commands) {
    commands.spawn((Reloading, KeepRunning, Name::new("reload")));
}

/// The [`KeepRunning`] a reload holds a run nobody sits at with.
#[derive(Component)]
struct Reloading;

/// Holds a run nobody sits at while a reload is queued, builds or waits to
/// restart, and lets it go once none does.
fn hold_headless_run(
    status: Res<ReloadStatus>,
    held: Query<Entity, With<Reloading>>,
    mut commands: Commands,
) {
    match *status {
        ReloadStatus::Queued { .. } | ReloadStatus::Building { .. } | ReloadStatus::Ready => {
            if held.is_empty() {
                keep_run(&mut commands);
            }
        }
        ReloadStatus::Idle | ReloadStatus::Failed => {
            for entity in &held {
                commands.entity(entity).despawn();
            }
        }
    }
}
