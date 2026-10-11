//! What each plugin added, as data in the world: one entity per plugin
//! `plugins.toml` lists (its `Name` and [`PluginSource`]), and
//! [`ProvidedBy`] on what that plugin spawned, such as its commands,
//! tools, prompt sections and observers. "What did plugin X add" is a
//! query for `ProvidedBy(x)`.
//!
//! Every named entity and observer a plugin spawns in its `build`,
//! `finish` and `cleanup` is attributed. What it spawns later, such as a
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

use std::marker::PhantomData;

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
    // `Attribute` before and after it brackets its `finish` and `cleanup`
    // (and the plugins it adds), which run in the order plugins were
    // added.
    app.add_plugins(Attribute::Open(entity));
    app.add_plugins(plugin);
    app.add_plugins(Attribute::Close);
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

/// The observer attributing to the plugin being finished or cleaned up.
#[derive(Resource)]
struct Attributing(Entity);

impl Attribute {
    fn bracket(&self, app: &mut App) {
        let world = app.world_mut();
        match self {
            Self::Open(plugin) => {
                let observer = attribute(world, *plugin);
                world.insert_resource(Attributing(observer));
            }
            Self::Close => {
                if let Some(Attributing(observer)) = world.remove_resource::<Attributing>() {
                    stop_attributing(world, observer);
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
