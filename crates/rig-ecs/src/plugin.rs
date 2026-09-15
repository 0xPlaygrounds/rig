//! Optional Bevy app integration; app-free hosts install and drive the runtime directly.

use bevy_app::{App, Plugin, Update};

use crate::{
    bus::{Bus, Policy, run_to_quiescence},
    systems::install_agent,
};

/// Installs the agent runtime and drives it to quiescence once per `Update`.
///
/// An existing bus keeps its policy, schedule and resources. Otherwise [`Self::bus`]
/// configures the newly installed bus. Calling [`install_agent`] before or after
/// adding this plugin is supported. Like other unique Bevy plugins, add this plugin
/// only once; do not also register `run_to_quiescence` or run `RigSchedule` from
/// `MainScheduleOrder`.
///
/// Assets, replay recording, logging and error policy remain host-owned.
#[derive(Debug, Default, Clone)]
pub struct RigPlugin {
    /// Configuration used only when the world has no installed bus.
    pub bus: Bus,
}

impl Plugin for RigPlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<Policy>() {
            self.bus.install(app.world_mut());
        }
        install_agent(app.world_mut());
        app.add_systems(Update, run_to_quiescence);
    }
}
