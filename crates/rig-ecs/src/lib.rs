//! rig inside a Bevy `World`.
//!
//! Two layers. [`bus`]: the effect bus as a plugin — effects are entities,
//! handlers are entities, the driver is a system, an outcome is a
//! component, causality is `ChildOf`. And the agent runtime over it —
//! [`agent`] (the run as a graph: agents, documents, utterances, runs,
//! turns, as entities and relationships), [`policy`] (the verbatim strings
//! and the one fold from the graph to the wire `CompletionRequest`),
//! [`systems`] (one system per named set, in the bus's schedule),
//! and [`checkpoint`] (the world as reflected data: save, load).
//! Concrete log recording, replay delivery and log-derived identity live in
//! `rig_cassette::ecs` with cassette's `ecs` feature. Install its `ReplayPlugin`
//! after this runtime before registering a replay. Neither cassette nor
//! `rig-agent` is a normal dependency of this runtime.
//!
//! The request is a graph in the world and a struct on the wire, with
//! [`policy::fold_request`] as the one function between them. What the
//! agent runtime does today: the run with tools — request assembly, the
//! stream fold, the three output modes and their reprompts, invalid calls
//! as entities with a resolution (fail, ignore, retry, repair, skip), tool
//! calls as effect entities `ChildOf` the turn with the batch as the
//! turn's children, endings, the header; and steering as components a
//! user system writes — [`agent::Cancelled`], [`agent::Retry`],
//! [`agent::RequestPatch`], [`agent::Resolution`], `UsesModel` — read by
//! the library at the next set; memory as the graph (an agent that
//! [`agent::Remembers`] loads its [`agent::Conversation`] before the first
//! turn and appends what the run said at the settle); retrieval as
//! [`agent::Retrieves`] links whose effects run before every fold and
//! attach documents and tools to the turn; resume as a checkpoint load
//! ([`checkpoint::save_world`] with the run's effects, in flight or
//! answered, beside the graph, and [`checkpoint::load_world`] in a fresh
//! world over the log's tail).
//!
//! [`prelude`] names the sets and the components a user's systems write
//! and read, and nothing else. Every component derives `Reflect` and
//! [`checkpoint::register_types`] registers them; the one feature,
//! `assets`, adds prompts and tool definitions as `bevy_asset` assets.
//!
//! A host is `App::new().add_plugins(RigPlugin::default())`; `app.update()`
//! runs [`bus::RigSchedule`] once, after `Update`, and the default runner
//! updates when a task raises [`bus::Wake`]. A test that drives the
//! schedule itself installs [`bus::BusPlugin::install`] and
//! [`systems::AgentPlugin::install`] on a bare `World` and calls
//! `world.run_schedule(RigSchedule)`.
//!
//! The `bus` module is written as if it were already its own crate (every
//! item `pub` or private to its file, no import from a sibling module, no
//! agent-shaped item, its tests in `tests/bus_*.rs`): the agent modules
//! consume it through its public items only. Additional consumers supply
//! evidence for a later crate-boundary decision; they do not require an
//! automatic extraction into a separate `rig-bevy` crate.

pub mod agent;
#[cfg(feature = "assets")]
pub mod assets;
pub mod bus;
pub mod checkpoint;
pub mod policy;
pub mod prelude;
pub mod reflect;
pub mod systems;

/// The whole runtime as one plugin: the bus ([`bus::BusPlugin`]) and the
/// agent systems ([`systems::AgentPlugin`]) in one `RigSchedule` after
/// `Update`, woken by their tasks. `App::new().add_plugins(RigPlugin::default())`
/// is a host; `app.update()` is a tick.
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
