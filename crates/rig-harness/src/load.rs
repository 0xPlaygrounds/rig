//! What each plugin added, as data in the world: one entity per plugin
//! `plugins.toml` lists (its `Name` and [`PluginSource`]), and
//! [`ProvidedBy`] on what that plugin's `build` spawned, such as its
//! commands, tools, prompt sections and observers. "What did plugin X add" is a query for `ProvidedBy(x)`.
//! Only what a plugin spawns while it is added is attributed; an entity it
//! spawns later carries `ProvidedBy` only if the plugin inserts it.

use bevy_app::prelude::*;
use bevy_ecs::observer::Observer;
use bevy_ecs::prelude::*;
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

/// The plugin whose `build` spawned this entity.
#[derive(Component, Reflect, Debug)]
#[reflect(Component)]
#[relationship(relationship_target = Provides)]
pub struct ProvidedBy(pub Entity);

/// What a plugin's `build` spawned.
#[derive(Component, Reflect, Debug)]
#[reflect(Component)]
#[relationship_target(relationship = ProvidedBy)]
pub struct Provides(Vec<Entity>);

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
    let world = app.world_mut();
    let entity = world
        .spawn((Name::new(plugin.name().to_owned()), source))
        .id();
    // Marks each named entity and observer the plugin spawns as its own.
    let attribute = world
        .add_observer(
            move |added: On<Add<(Name, Observer)>>, mut commands: Commands| {
                commands
                    .entity(added.entity)
                    .try_insert_if_new(ProvidedBy(entity));
            },
        )
        .id();
    app.add_plugins(plugin);
    // What the plugin queued is spawned while it is still attributed.
    app.world_mut().flush();
    app.world_mut().despawn(attribute);
}
