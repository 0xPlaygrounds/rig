use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

use super::{PluginEntity, PluginSource, ProvidedBy, load};

/// Spawns a named entity in `build`, one in `finish`, and one at startup
/// that it attributes itself.
#[derive(Default)]
struct Spawner;

impl Plugin for Spawner {
    fn build(&self, app: &mut App) {
        app.world_mut().spawn(Name::new("built"));
        app.add_systems(
            Startup,
            |me: Res<PluginEntity<Spawner>>, mut commands: Commands| {
                commands.spawn((Name::new("later"), ProvidedBy(me.entity())));
                commands.spawn(Name::new("unattributed"));
            },
        );
    }

    fn finish(&self, app: &mut App) {
        app.world_mut().spawn(Name::new("finished"));
    }
}

/// Spawns a named entity in `finish`, after `Spawner`.
#[derive(Default)]
struct Other;

impl Plugin for Other {
    fn build(&self, _app: &mut App) {}

    fn finish(&self, app: &mut App) {
        app.world_mut().spawn(Name::new("other's"));
    }
}

#[test]
fn what_a_plugin_spawns_in_build_and_finish_is_its_own() {
    let mut app = App::new();
    load::<Spawner>(&mut app, "spawner", "path plugins/spawner");
    load::<Spawner>(&mut app, "spawner", "path plugins/spawner");
    load::<Other>(&mut app, "other", "version 1");
    app.finish();
    app.cleanup();
    app.update();
    let world = app.world_mut();
    let spawner = world.resource::<PluginEntity<Spawner>>().entity();
    let other = world.resource::<PluginEntity<Other>>().entity();
    assert_eq!(
        world
            .get::<PluginSource>(spawner)
            .map(|source| source.krate.as_str()),
        Some("spawner")
    );
    let mut named = world.query::<(&Name, Option<&ProvidedBy>)>();
    let mut by = |name: &str| {
        named
            .iter(world)
            .find(|(named, _)| named.as_str() == name)
            .and_then(|(_, by)| by.map(|by| by.0))
    };
    assert_eq!(by("built"), Some(spawner));
    assert_eq!(by("finished"), Some(spawner));
    assert_eq!(by("later"), Some(spawner));
    assert_eq!(by("other's"), Some(other));
    assert_eq!(by("unattributed"), None);
}
