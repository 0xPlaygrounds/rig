use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig_harness_protocol::Home;
use serde::Deserialize;

use super::{AppSettingsExt, Settings};
use crate::services::RigHome;
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
    // One line, saying where in the file.
    assert!(
        unparsed
            .problems()
            .iter()
            .all(|problem| !problem.contains('\n')
                && problem.contains("not TOML at line 1, column ")),
        "{:?}",
        unparsed.problems()
    );
    let problem = mistyped.err().unwrap_or_default();
    assert!(
        problem.starts_with(&format!("{} [limits]: ", think.display())),
        "{problem}"
    );
}

#[test]
fn a_key_the_table_does_not_have_says_to_delete_it() {
    let settings = Settings::from_toml("[limits]\ncalls = 3\nmax_depth = 2\n");
    let problem = settings.get::<Limits>("limits").err().unwrap_or_default();
    assert!(
        problem.contains("unknown field `max_depth`")
            && problem.ends_with("; delete `max_depth` from [limits]"),
        "{problem}"
    );
    let mistyped = Settings::from_toml("[limits]\ncalls = \"many\"\n");
    let problem = mistyped.get::<Limits>("limits").err().unwrap_or_default();
    assert!(!problem.contains("delete"), "{problem}");
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
