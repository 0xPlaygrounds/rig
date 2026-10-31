use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig_harness_protocol::Home;
use serde::Deserialize;

use super::{AppSettingsExt, ProjectSetting, Settings};
use crate::host::cli::Declared;
use crate::services::{ProjectRoot, RigHome};
use rig_ecs::agent::NoticeLevel;

#[derive(Resource, Deserialize, Default, Debug, PartialEq)]
#[serde(default, deny_unknown_fields)]
struct Limits {
    calls: u32,
    tools: Vec<String>,
}

#[derive(Resource, Deserialize, Default, Debug, PartialEq)]
#[serde(default)]
struct Quiet {
    on: bool,
}

#[test]
fn each_plugin_reads_its_table_and_a_bad_one_keeps_the_default() {
    let mut app = App::new();
    app.insert_resource(Settings::from_toml(
        "[limits]\ncalls = 3\ntools = [\"shell\"]\n\n[quiet]\non = \"yes\"\n",
    ))
    .add_setting::<Limits>("limits")
    .add_setting::<Quiet>("quiet");
    let world = app.world();
    assert_eq!(
        world.resource::<Limits>(),
        &Limits {
            calls: 3,
            tools: vec!["shell".to_owned()],
        }
    );
    assert_eq!(world.resource::<Quiet>(), &Quiet::default());
    let problems = world.resource::<Settings>().problems();
    assert_eq!(problems.len(), 1);
    assert!(
        problems.iter().all(|problem| problem.contains("[quiet]")),
        "{problems:?}"
    );
}

#[test]
fn the_file_is_read_from_the_apps_home_once() {
    let root = std::env::temp_dir().join(format!("rig-settings-{}", std::process::id()));
    std::fs::create_dir_all(&root).ok();
    let home = Home::at(&root);
    std::fs::write(home.settings(), "[limits]\ncalls = 9\n").ok();
    let mut app = App::new();
    app.insert_resource(RigHome(home.clone()))
        .add_setting::<Limits>("limits");
    // Later changes to the file are not read again while the app runs.
    std::fs::write(home.settings(), "not toml").ok();
    app.add_setting::<Quiet>("quiet");
    assert_eq!(app.world().resource::<Limits>().calls, 9);
    assert!(app.world().resource::<Settings>().problems().is_empty());
    let unreadable = Settings::read(&home.settings());
    assert!(
        unreadable
            .problems()
            .iter()
            .any(|problem| problem.contains("not TOML"))
    );
    assert!(
        Settings::read(&root.join("missing.toml"))
            .problems()
            .is_empty()
    );
    assert_eq!(Settings::read(&root).problems().len(), 1);
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn a_bad_file_names_itself() {
    let root = std::env::temp_dir().join(format!("rig-settings-named-{}", std::process::id()));
    std::fs::create_dir_all(&root).ok();
    let think = root.join("think.toml");
    std::fs::write(&think, "[keywords\n").ok();
    let unparsed = Settings::read(&think);
    std::fs::write(&think, "[limits]\ncalls = \"many\"\n").ok();
    let mistyped = Settings::read(&think).get::<Limits>("limits");
    std::fs::remove_dir_all(&root).ok();
    assert!(
        unparsed
            .problems()
            .iter()
            .all(|problem| problem.starts_with(&format!("{}: ", think.display()))),
        "{:?}",
        unparsed.problems()
    );
    assert_eq!(unparsed.problems().len(), 1);
    let problem = mistyped.err().unwrap_or_default();
    assert!(
        problem.starts_with(&format!("{} [limits]: ", think.display())),
        "{problem}"
    );
}

#[derive(Resource, Deserialize, Default, Debug, PartialEq)]
#[serde(default, deny_unknown_fields)]
struct Counts {
    n: u32,
    m: u32,
}

#[derive(Resource, Deserialize, Default, Debug, PartialEq)]
#[serde(default, deny_unknown_fields)]
struct Others {
    n: u32,
}

/// A project overrides, key by key, only the tables a plugin declared it
/// may: another table of its file is not read.
#[test]
fn a_project_overrides_only_the_tables_declared_for_it() {
    let root = std::env::temp_dir().join(format!("rig-settings-project-{}", std::process::id()));
    std::fs::create_dir_all(root.join(".rig")).ok();
    std::fs::write(
        root.join(".rig/settings.toml"),
        "[a]\nn = 2\n\n[b]\nn = 2\n",
    )
    .ok();
    let mut app = App::new();
    app.insert_resource(Settings::from_toml("[a]\nn = 1\nm = 5\n\n[b]\nn = 1\n"))
        .insert_resource(ProjectRoot(root.clone()))
        .add_project_setting::<Counts>("a")
        .add_setting::<Others>("b");
    assert_eq!(app.world().resource::<Counts>(), &Counts { n: 1, m: 5 });
    app.update();
    std::fs::remove_dir_all(&root).ok();
    assert_eq!(app.world().resource::<Counts>(), &Counts { n: 2, m: 5 });
    assert_eq!(app.world().resource::<Others>(), &Others { n: 1 });
    assert!(app.world().resource::<Settings>().problems().is_empty());
    // What a project may override is data a system can list.
    let world = app.world_mut();
    let declared: Vec<String> = world
        .query::<&ProjectSetting>()
        .iter(world)
        .map(|it| it.0.clone())
        .collect();
    assert_eq!(declared, ["a"]);
}

/// Tables read into one type apply in the order declared, the last
/// winning, even when a declaration was turned off while others were made.
#[test]
fn the_last_declared_project_table_wins_across_a_disabled_declaration() {
    use bevy_ecs::entity_disabling::Disabled;
    let root = std::env::temp_dir().join(format!("rig-settings-order-{}", std::process::id()));
    std::fs::create_dir_all(root.join(".rig")).ok();
    std::fs::write(
        root.join(".rig/settings.toml"),
        "[a]\nn = 1\n\n[b]\nn = 2\n",
    )
    .ok();
    let mut app = App::new();
    app.insert_resource(Settings::from_toml(""))
        .insert_resource(ProjectRoot(root.clone()))
        .add_project_setting::<Counts>("a");
    let world = app.world_mut();
    let first = world
        .query::<(Entity, &ProjectSetting)>()
        .iter(world)
        .find_map(|(entity, setting)| (setting.0 == "a").then_some(entity));
    let first = first.map(|entity| world.entity_mut(entity).insert(Disabled).id());
    app.add_project_setting::<Counts>("b");
    if let Some(first) = first {
        app.world_mut().entity_mut(first).remove::<Disabled>();
    }
    app.update();
    std::fs::remove_dir_all(&root).ok();
    assert!(first.is_some(), "the first declaration was found");
    assert_eq!(app.world().resource::<Counts>().n, 2);
    let world = app.world_mut();
    let mut places: Vec<(Declared, String)> = world
        .query::<(&Declared, &ProjectSetting)>()
        .iter(world)
        .map(|(&place, setting)| (place, setting.0.clone()))
        .collect();
    places.sort();
    let sections: Vec<&str> = places.iter().map(|(_, section)| section.as_str()).collect();
    assert_eq!(sections, ["a", "b"]);
    assert!(
        places
            .windows(2)
            .all(|pair| pair.first().map(|it| it.0) < pair.last().map(|it| it.0)),
        "{places:?}"
    );
}

/// A type read from a turned-off declaration and a later one is applied by
/// one system: a malformed table is one problem, not one per declaration.
#[test]
fn a_disabled_declaration_adds_no_second_apply_system() {
    use bevy_ecs::entity_disabling::Disabled;
    let root = std::env::temp_dir().join(format!("rig-settings-once-{}", std::process::id()));
    std::fs::create_dir_all(root.join(".rig")).ok();
    std::fs::write(root.join(".rig/settings.toml"), "[b]\nn = \"x\"\n").ok();
    let mut app = App::new();
    app.insert_resource(Settings::from_toml(""))
        .insert_resource(ProjectRoot(root.clone()))
        .add_project_setting::<Counts>("a");
    let world = app.world_mut();
    let first = world
        .query::<(Entity, &ProjectSetting)>()
        .iter(world)
        .find_map(|(entity, setting)| (setting.0 == "a").then_some(entity));
    if let Some(first) = first {
        world.entity_mut(first).insert(Disabled);
    }
    app.add_project_setting::<Counts>("b");
    app.update();
    std::fs::remove_dir_all(&root).ok();
    assert!(first.is_some(), "the first declaration was found");
    let problems = app.world().resource::<Settings>().problems();
    assert_eq!(problems.len(), 1, "{problems:?}");
}

#[test]
fn a_table_no_plugin_reads_is_named_with_the_one_it_likely_misspells() {
    let mut app = App::new();
    let file = "[trafic]\nlog = true\n[tuii]\n[shel]\n[mcp]\n[quiet]\non = true\n";
    app.insert_resource(Settings::from_toml(file))
        .add_setting::<Quiet>("quiet")
        .add_setting::<Quiet>("traffic")
        .add_setting::<Quiet>("tui");
    let unread = app.world().resource::<Settings>().unread();
    let mut said: Vec<(NoticeLevel, &str)> = unread
        .iter()
        .map(|notice| (notice.level, notice.text.as_str()))
        .collect();
    said.sort_by_key(|(_, text)| *text);
    assert_eq!(
        said,
        [
            (
                NoticeLevel::Info,
                "settings.toml: no enabled plugin reads [mcp]."
            ),
            (
                NoticeLevel::Info,
                "settings.toml: no enabled plugin reads [shel]."
            ),
            (
                NoticeLevel::Error,
                "settings.toml: no plugin reads [trafic]; did you mean [traffic]?"
            ),
            (
                NoticeLevel::Error,
                "settings.toml: no plugin reads [tuii]; did you mean [tui]?"
            ),
        ]
    );
}
