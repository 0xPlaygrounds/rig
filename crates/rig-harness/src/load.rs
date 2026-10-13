//! What each plugin added, as data in the world: one entity per plugin
//! `plugins.toml` lists (its `Name` and [`PluginSource`]), and
//! [`ProvidedBy`] on what that plugin spawned, such as its commands,
//! tools, prompt sections and observers. "What did plugin X add" is a
//! query for `ProvidedBy(x)`; which plugin added a schedule's system is in
//! [`PluginSystems`].
//!
//! Every named entity, observer and schedule system a plugin adds in its
//! `build`, `finish` and `cleanup` is attributed. What it spawns later, such as a
//! tool an MCP server lists, it attributes itself, with its entity from
//! [`PluginEntity<P>`]:
//!
//! ```no_run
//! use rig_harness::prelude::*;
//!
//! #[derive(Default)]
//! struct McpPlugin;
//!
//! impl Plugin for McpPlugin {
//!     fn build(&self, _app: &mut App) {}
//! }
//!
//! fn spawn_listed_tool(me: Option<Res<PluginEntity<McpPlugin>>>, mut commands: Commands) {
//!     let tool = commands.spawn(Name::new("listed tool")).id();
//!     if let Some(me) = me {
//!         commands.entity(tool).insert(ProvidedBy(me.entity()));
//!     }
//! }
//! ```

use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

use bevy_app::prelude::*;
use bevy_ecs::observer::Observer;
use bevy_ecs::prelude::*;
use bevy_ecs::schedule::{Schedules, SystemKey};
use bevy_reflect::prelude::*;

/// Where a plugin comes from, on the entity [`load`] spawns for it.
#[derive(Component, Reflect, Clone, Debug)]
#[reflect(Component, Clone, Debug)]
pub struct PluginSource {
    /// Its package, such as `rig-harness`.
    pub krate: String,
    /// Where the package is built from, as `plugins.toml` names it, such
    /// as `path /home/me/.rig/plugins/viz` or `version 0.3`.
    pub source: String,
}

/// The plugin that spawned this entity.
#[derive(Component, Reflect, Debug)]
#[reflect(Component)]
#[relationship(relationship_target = Provides)]
pub struct ProvidedBy(pub Entity);

/// What a plugin spawned.
#[derive(Component, Reflect, Debug)]
#[reflect(Component)]
#[relationship_target(relationship = ProvidedBy)]
pub struct Provides(Vec<Entity>);

/// The entity [`load`] spawned for the plugin `P`, for what `P` spawns
/// while the app runs.
#[derive(Resource)]
pub struct PluginEntity<P: Plugin> {
    entity: Entity,
    plugin: PhantomData<fn() -> P>,
}

impl<P: Plugin> PluginEntity<P> {
    /// The plugin's entity.
    pub fn entity(&self) -> Entity {
        self.entity
    }
}

/// Which plugin added each schedule system, by the system's name and its
/// schedule's label (such as `AgentPass`), as [`load`] saw it. Two plugins
/// that add a system of the same name to one schedule are told apart by
/// neither; a system a plugin adds later, while the app runs, is no
/// plugin's.
#[derive(Resource, Default, Debug)]
pub struct PluginSystems(HashMap<(String, String), Option<Entity>>);

impl PluginSystems {
    /// The plugin that added the system `system` to the schedule labelled
    /// `schedule`, if one plugin did.
    pub fn plugin(&self, system: &str, schedule: &str) -> Option<Entity> {
        let key = (system.to_owned(), schedule.to_owned());
        self.0.get(&key).copied().flatten()
    }

    /// Records each system added since `before` as `plugin`'s.
    fn record(world: &mut World, before: &HashSet<(String, SystemKey)>, plugin: Entity) {
        let mut added = Vec::new();
        if let Some(schedules) = world.get_resource::<Schedules>() {
            for (label, schedule) in schedules.iter() {
                let label = format!("{label:?}");
                for (key, system, _) in schedule.graph().systems.iter() {
                    if !before.contains(&(label.clone(), key)) {
                        added.push((system.name().to_string(), label.clone()));
                    }
                }
            }
        }
        let mut systems = world.get_resource_or_init::<PluginSystems>();
        for key in added {
            let by = systems.0.entry(key).or_insert(Some(plugin));
            if *by != Some(plugin) {
                *by = None;
            }
        }
    }
}

/// Every system of every schedule, by its schedule's label.
fn systems(world: &World) -> HashSet<(String, SystemKey)> {
    let Some(schedules) = world.get_resource::<Schedules>() else {
        return HashSet::new();
    };
    let systems = schedules.iter().flat_map(|(label, schedule)| {
        let label = format!("{label:?}");
        let keys = schedule.graph().systems.iter().map(|(key, ..)| key);
        keys.map(move |key| (label.clone(), key))
    });
    systems.collect()
}

/// Adds `P`, unless the app has it already (another plugin added it), as
/// the generated `main.rs` adds each plugin `plugins.toml` lists: with an
/// entity of its own, named after it, from `krate` at `source`.
pub fn load<P: Plugin + Default>(app: &mut App, krate: &str, source: &str) {
    if app.is_plugin_added::<P>() {
        return;
    }
    let plugin = P::default();
    let source = PluginSource {
        krate: krate.to_owned(),
        source: source.to_owned(),
    };
    let entity = app
        .world_mut()
        .spawn((Name::new(plugin.name().to_owned()), source))
        .id();
    app.insert_resource(PluginEntity::<P> {
        entity,
        plugin: PhantomData,
    });
    let attribute = attribute(app.world_mut(), entity);
    let before = systems(app.world());
    // `Attribute` before and after it brackets its `finish` and `cleanup`
    // (and the plugins it adds), which run in the order plugins were
    // added.
    app.add_plugins(Attribute::Open(entity));
    app.add_plugins(plugin);
    app.add_plugins(Attribute::Close);
    PluginSystems::record(app.world_mut(), &before, entity);
    stop_attributing(app.world_mut(), attribute);
}

/// Marks each named entity and observer spawned from now on as
/// `plugin`'s, until [`stop_attributing`].
fn attribute(world: &mut World, plugin: Entity) -> Entity {
    world
        .add_observer(
            move |added: On<Add<(Name, Observer)>>, mut commands: Commands| {
                commands
                    .entity(added.entity)
                    .try_insert_if_new(ProvidedBy(plugin));
            },
        )
        .id()
}

/// Spawns what was queued while still attributed, then stops.
fn stop_attributing(world: &mut World, attribute: Entity) {
    world.flush();
    world.despawn(attribute);
}

/// Attributes what a plugin spawns in `finish` and `cleanup`, from its
/// [`Attribute::Open`] to the [`Attribute::Close`] after it.
enum Attribute {
    Open(Entity),
    Close,
}

/// The plugin being finished or cleaned up, the observer attributing to
/// it, and the systems before it.
#[derive(Resource)]
struct Attributing {
    plugin: Entity,
    observer: Entity,
    before: HashSet<(String, SystemKey)>,
}

impl Attribute {
    fn bracket(&self, app: &mut App) {
        let world = app.world_mut();
        match self {
            Self::Open(plugin) => {
                let observer = attribute(world, *plugin);
                let before = systems(world);
                world.insert_resource(Attributing {
                    plugin: *plugin,
                    observer,
                    before,
                });
            }
            Self::Close => {
                if let Some(attributing) = world.remove_resource::<Attributing>() {
                    PluginSystems::record(world, &attributing.before, attributing.plugin);
                    stop_attributing(world, attributing.observer);
                }
            }
        }
    }
}

impl Plugin for Attribute {
    fn build(&self, _app: &mut App) {}

    fn finish(&self, app: &mut App) {
        self.bracket(app);
    }

    fn cleanup(&self, app: &mut App) {
        self.bracket(app);
    }

    fn is_unique(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests;
