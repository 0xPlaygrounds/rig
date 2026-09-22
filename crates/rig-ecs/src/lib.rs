//! Rig's effect bus and agent runtime as entities and systems in a Bevy world.
//!
//! [`RigPlugin`] installs request assembly, dispatch, streaming, tool execution,
//! memory, and retrieval. Hosts steer runs through components and save reflected
//! state with [`checkpoint`]. Concrete recording and replay adapters live in
//! `rig_cassette::ecs`; the optional `assets` feature loads prompts and tools.
//!
//! ```
//! let mut app = bevy_app::App::new();
//! app.add_plugins(rig_ecs::RigPlugin::default());
//! app.update();
//! ```

pub mod agent;
#[cfg(feature = "assets")]
pub mod assets;
pub mod bus;
pub mod checkpoint;
pub mod policy;
pub mod prelude;
pub mod reflect;
pub mod systems;

/// Installs bus and agent systems in [`bus::RigSchedule`] after `Update`, with
/// task-driven wakeups and checkpoint type registration. Adds time and
/// diagnostics plugins if absent.
#[derive(Debug, Clone, Default)]
pub struct RigPlugin {
    /// The bus's configuration: the serving policy, the ambiguity level.
    pub bus: bus::BusPlugin,
}

impl RigPlugin {
    /// The runtime under `policy`.
    pub fn with_policy(policy: rig_core::serve::ServingPolicy) -> Self {
        Self {
            bus: bus::BusPlugin::with_policy(policy),
        }
    }

    /// Build the schedule with ambiguity detection at `level`.
    #[must_use = "the setting applies to the returned value"]
    pub fn ambiguity_detection(mut self, level: bevy_ecs::schedule::LogLevel) -> Self {
        self.bus = self.bus.ambiguity_detection(level);
        self
    }
}

impl bevy_app::Plugin for RigPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        if !app.is_plugin_added::<bevy_time::TimePlugin>() {
            app.add_plugins(bevy_time::TimePlugin);
        }
        if !app.is_plugin_added::<bevy_diagnostic::DiagnosticsPlugin>() {
            app.add_plugins(bevy_diagnostic::DiagnosticsPlugin);
        }
        app.add_plugins((self.bus.clone(), systems::AgentPlugin));
        checkpoint::register_types(app.world_mut());
    }
}
