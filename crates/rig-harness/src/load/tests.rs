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

/// Stands in for `rig_basics::budget::BudgetPlugin`, off by default.
#[derive(Default)]
struct OffByDefault;

impl Plugin for OffByDefault {
    fn build(&self, _: &mut App) {}
}

/// Stands in for `rig_basics::skills::SkillsPlugin`, on by default.
#[derive(Default)]
struct OnByDefault;

impl Plugin for OnByDefault {
    fn build(&self, _: &mut App) {}
}

const BUDGET: &str = "rig_basics::budget::BudgetPlugin";
const SKILLS: &str = "rig_basics::skills::SkillsPlugin";

/// Run by [`a_session_adds_the_plugins_its_settings_or_its_parent_turn_on`]
/// in a process of its own: builds an app in the home `RIG_TEST_CHOSEN`
/// names, from its `work` directory, and prints which of the two plugins
/// it added and the list every process it starts gets.
#[test]
fn chosen_in_a_process_of_its_own() {
    use crate::services::{Processes, ProjectRoot, RigHome};
    use rig_harness_protocol::Home;

    let Ok(root) = std::env::var("RIG_TEST_CHOSEN") else {
        return;
    };
    let root = std::path::PathBuf::from(root);
    let mut app = App::new();
    app.insert_resource(RigHome(Home::at(root.join("home"))))
        .insert_resource(ProjectRoot(root.join("work")))
        .init_resource::<Processes>();
    super::load_chosen(
        &mut app,
        &[
            super::Compiled::new::<OffByDefault>(BUDGET, "rig-basics", ""),
            super::Compiled::new::<OnByDefault>(SKILLS, "rig-basics", ""),
        ],
    );
    let launcher = app.world().resource::<Processes>().launcher();
    let handed = (launcher.env.iter())
        .find(|(name, _)| name == "RIG_PLUGINS")
        .map(|(_, value)| value.to_string_lossy().into_owned())
        .unwrap_or_default();
    println!(
        "budget={} skills={} handed={handed}",
        app.is_plugin_added::<OffByDefault>(),
        app.is_plugin_added::<OnByDefault>()
    );
}

#[test]
fn a_session_adds_the_plugins_its_settings_or_its_parent_turn_on() {
    let root = std::env::temp_dir().join(format!("rig-load-chosen-{}", std::process::id()));
    std::fs::remove_dir_all(&root).ok();
    let project = root.join("work/.rig");
    assert!(std::fs::create_dir_all(root.join("home")).is_ok());
    assert!(std::fs::create_dir_all(&project).is_ok());
    let settings = format!("[plugins]\nenable = [\"{BUDGET}\"]\ndisable = [\"{SKILLS}\"]\n");
    assert!(std::fs::write(root.join("home/settings.toml"), settings).is_ok());
    let run = |parent: Option<&str>| {
        let test = std::env::current_exe().ok()?;
        let mut command = std::process::Command::new(test);
        command
            .args([
                "load::tests::chosen_in_a_process_of_its_own",
                "--exact",
                "--nocapture",
            ])
            .env("RIG_TEST_CHOSEN", &root)
            .env_remove("RIG_PLUGINS");
        if let Some(parent) = parent {
            command.env("RIG_PLUGINS", parent);
        }
        let started = rig_tools::process::Process::start(command, Default::default());
        let output = started.and_then(|child| child.output(None)).ok()?;
        let said = String::from_utf8_lossy(&output.stdout).into_owned();
        said.lines()
            .find(|line| line.starts_with("budget="))
            .map(str::to_owned)
    };
    let home = run(None).unwrap_or_default();
    // The project turns skills back on, with the same build.
    assert!(
        std::fs::write(
            project.join("settings.toml"),
            format!("[plugins]\nenable = [\"{SKILLS}\"]\n")
        )
        .is_ok()
    );
    let project = run(None).unwrap_or_default();
    // A child of a session runs its parent's list, whatever the settings.
    let child = run(Some(SKILLS)).unwrap_or_default();
    std::fs::remove_dir_all(&root).ok();
    assert!(
        home.starts_with("budget=true skills=false handed="),
        "{home}"
    );
    assert!(home.contains(BUDGET) && !home.contains(SKILLS), "{home}");
    assert!(project.starts_with("budget=true skills=true"), "{project}");
    assert_eq!(child, format!("budget=false skills=true handed={SKILLS}"));
}
