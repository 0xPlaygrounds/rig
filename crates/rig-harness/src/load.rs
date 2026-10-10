//! What this build is and what each plugin added, as data in the world:
//! the [`Build`] resource, one entity per plugin `plugins.toml` lists
//! (its `Name` and [`PluginSource`]), and [`ProvidedBy`] on what that
//! plugin's `build` spawned, such as its commands, tools, prompt sections
//! and observers. "What did plugin X add" is a query for `ProvidedBy(x)`.
//! Only what a plugin spawns while it is added is attributed; an entity it
//! spawns later carries `ProvidedBy` only if the plugin inserts it.

use std::fs;
use std::time::UNIX_EPOCH;

use bevy_app::prelude::*;
use bevy_ecs::observer::Observer;
use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use rig::harness_protocol::Home;

use crate::host::launcher;

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

/// The build this process runs, which the generated `main.rs` inserts.
#[derive(Resource, Reflect, Clone, Debug)]
#[reflect(Resource, Clone, Debug)]
pub struct Build {
    /// The rig source it is built from: a checkout's commit, else the
    /// crates.io version.
    pub revision: String,
    /// When its binary was written, in milliseconds since the Unix epoch.
    pub built_at: Option<u64>,
    /// How the launcher runs it.
    pub kind: BuildKind,
}

/// How the launcher runs a build.
#[derive(Reflect, Clone, Copy, Debug, PartialEq, Eq)]
#[reflect(Clone, Debug, PartialEq)]
pub enum BuildKind {
    /// A new build on trial: rolled back unless it starts.
    Trial,
    /// The last build that started.
    KnownGood,
    /// Not run by the launcher.
    Unmanaged,
}

impl Build {
    /// The running binary, built from `revision`.
    pub fn running(revision: &str) -> Self {
        let binary = std::env::current_exe().ok();
        let built_at = binary
            .as_ref()
            .and_then(|binary| fs::metadata(binary).ok()?.modified().ok())
            .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
            .map(|elapsed| u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX));
        let good = fs::canonicalize(Home::from_env().good()).ok();
        let kind = if launcher::executable().is_none() {
            BuildKind::Unmanaged
        } else if binary.is_some() && binary == good {
            BuildKind::KnownGood
        } else {
            BuildKind::Trial
        };
        Self {
            revision: revision.to_owned(),
            built_at,
            kind,
        }
    }
}
