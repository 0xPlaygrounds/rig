use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

use bevy_reflect::prelude::*;
use rig_ecs::session_log::ReflectSaved;

use super::{AppKeepsExt, Keeps, PluginEntity, PluginSource, ProvidedBy, load};

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
            .map(|source| source.package.as_str()),
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

/// A saved agent component under a stable key.
#[derive(Component, Reflect, Default)]
#[reflect(Component, Saved)]
#[type_path = "counter"]
#[type_name = "count"]
struct Count(u32);

/// A saved resource.
#[derive(Resource, Reflect, Default)]
#[reflect(Resource, Saved)]
struct Totals(u32);

/// Keeps a count and session files in `build`, says the folder twice, and
/// loads `Inner` in the middle of its build.
#[derive(Default)]
struct Outer;

impl Plugin for Outer {
    fn build(&self, app: &mut App) {
        app.keeps_saved::<Count>();
        load::<Inner>(app, "inner", "version 1");
        app.keeps_files("counter/").keeps_files("counter");
    }
}

/// Keeps a resource, said in `finish`.
#[derive(Default)]
struct Inner;

impl Plugin for Inner {
    fn build(&self, _app: &mut App) {}

    fn finish(&self, app: &mut App) {
        app.keeps_saved::<Totals>();
    }
}

#[test]
fn what_a_plugin_keeps_is_on_its_entity_under_the_saved_key() {
    let mut app = App::new();
    load::<Outer>(&mut app, "outer", "version 1");
    // Added without `load`: no entity to say it on.
    app.add_plugins(|app: &mut App| {
        app.keeps_saved::<Count>();
    });
    app.finish();
    app.cleanup();
    let world = app.world_mut();
    let outer = world.resource::<PluginEntity<Outer>>().entity();
    let inner = world.resource::<PluginEntity<Inner>>().entity();
    let keeps = |plugin| world.get::<Keeps>(plugin).cloned().unwrap_or_default();
    let (outer, inner) = (keeps(outer), keeps(inner));
    assert_eq!(outer.saved, ["counter::count"]);
    assert_eq!(outer.files, ["counter"]);
    assert_eq!(inner.saved, [Totals::type_path()]);
    assert!(inner.files.is_empty());
    assert!(outer.saved("counter::count") && !inner.saved("counter::count"));
    assert!(outer.file("counter/3.json") && outer.file("counter/a/b"));
    assert!(!outer.file("counters/3.json") && !outer.file("counter"));
    let mut kept = world.query::<&Keeps>();
    assert_eq!(kept.iter(world).count(), 2);
}

/// A saved value's stable key is recorded and listed; a key another type
/// has is refused, so the first type keeps it and nothing is listed.
#[test]
fn a_saved_key_is_recorded_once_and_a_taken_one_refused() {
    let mut app = App::new();
    app.saved_as::<Totals>("counter.totals").listed();
    let _ = app.saved_as::<Count>("counter.totals");
    let keys = app.world().resource::<rig_ecs::restore::SavedKeys>();
    assert_eq!(keys.key(Totals::type_path()), Some("counter.totals"));
    assert_eq!(keys.path("counter.totals"), Some(Totals::type_path()));
    assert_eq!(keys.key(Count::type_path()), None);
    assert!(keys.is_listed("counter.totals"));
}

/// What two plugins' agent-loop systems write.
#[derive(Component)]
struct Tally;

/// Counts in the agent loop.
#[derive(Default)]
struct Counter;

impl Plugin for Counter {
    fn build(&self, app: &mut App) {
        app.add_systems(rig_ecs::stage::AgentPass, |_: Query<&mut Tally>| {});
    }
}

/// Resets in the agent loop, with no order against `Counter`.
#[derive(Default)]
struct Resetter;

impl Plugin for Resetter {
    fn build(&self, app: &mut App) {
        app.add_systems(rig_ecs::stage::AgentPass, |_: Query<&mut Tally>| {});
    }
}

#[test]
fn unordered_agent_loop_systems_name_both_plugins() {
    let mut app = App::new();
    app.init_schedule(rig_ecs::stage::AgentPass);
    load::<Counter>(&mut app, "counter", "path plugins/counter");
    load::<Resetter>(&mut app, "resetter", "version 1");
    let conflicts = super::stage_conflicts(app.world_mut());
    assert!(
        matches!(conflicts.as_slice(), [conflict]
            if conflict.contains("Counter`")
                && conflict.contains("Resetter`")
                && conflict.contains("Tally")),
        "{conflicts:?}"
    );
}

/// Each system a plugin added is listed with that plugin.
#[test]
fn plugin_systems_list_each_system_with_its_plugin() {
    let mut app = App::new();
    app.init_schedule(rig_ecs::stage::AgentPass);
    load::<Counter>(&mut app, "counter", "path plugins/counter");
    let counter = app.world().resource::<PluginEntity<Counter>>().entity();
    let systems = app.world().resource::<super::PluginSystems>();
    let listed: Vec<_> = systems
        .iter()
        .filter(|(.., plugin)| *plugin == Some(counter))
        .collect();
    assert!(!listed.is_empty());
    assert!(
        listed
            .iter()
            .all(|&(system, schedule, plugin)| systems.plugin(system, schedule) == plugin)
    );
}
