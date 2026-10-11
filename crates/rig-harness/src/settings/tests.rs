use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig::harness_protocol::Home;
use serde::Deserialize;

use super::{AppSettingsExt, Settings};
use crate::services::RigHome;

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
