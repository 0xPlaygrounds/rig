//! ECS effect recording, recorded delivery collection and program identity.
//!
//! Add [`ReplayPlugin`] after [`rig_ecs::RigPlugin`] or
//! [`rig_ecs::bus::BusPlugin`] before registering replay handlers. Live recording
//! needs only [`EffectLogResource::install`]; it does not need replay systems.
//!
//! ```
//! use bevy_app::App;
//! use rig_cassette::ecs::ReplayPlugin;
//! use rig_ecs::bus::BusPlugin;
//! let mut app = App::new();
//! app.add_plugins((BusPlugin::default(), ReplayPlugin));
//! ```

use bevy_app::{App, Plugin};
use bevy_ecs::prelude::*;
use rig_ecs::bus::{BusSet, RigEnd, RigSchedule, collect_tasks};

mod delivery;
pub mod identity;
mod replay;

pub use delivery::{ReplayDelivery, ReplayFailure};
pub use replay::{EffectLogResource, Replay};

#[derive(Resource)]
struct ReplayInstalled;

/// Recorded delivery collection and idle-refusal diagnosis for an ECS bus.
///
/// Install after the runtime plugin. Without a delivery plan the systems do
/// nothing; ordinary live execution continues through the runtime's collectors.
///
/// # Panics
///
/// Panics before runtime installation, which would overwrite the replay systems.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReplayPlugin;

impl Plugin for ReplayPlugin {
    fn build(&self, app: &mut App) {
        assert!(
            app.world().contains_resource::<rig_ecs::bus::Policy>(),
            "install ReplayPlugin after RigPlugin or BusPlugin"
        );
        app.insert_resource(ReplayInstalled)
            .add_systems(
                RigSchedule,
                delivery::collect_replayed
                    .run_if(resource_exists::<ReplayDelivery>)
                    .in_set(BusSet::Collect)
                    .before(collect_tasks),
            )
            .add_systems(
                RigEnd,
                delivery::diagnose_idle_replay.run_if(resource_exists::<ReplayDelivery>),
            );
    }
}

#[cfg(test)]
mod tests;
