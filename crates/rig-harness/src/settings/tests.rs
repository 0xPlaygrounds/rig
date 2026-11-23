use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig_harness_protocol::Home;
use serde::Deserialize;

use super::{AppSettingsExt, Settings};
use crate::host::cli::{AppCliExt, Front};
use crate::host::launcher::Invoked;
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

#[derive(Resource, Deserialize, Default, Debug, PartialEq)]
#[serde(default, deny_unknown_fields)]
struct Outer {
    limits: Limits,
}

#[derive(Resource, Deserialize, Default, Debug, PartialEq)]
#[serde(default, deny_unknown_fields)]
struct Dotted {
    #[serde(rename = "limits.x")]
    limits: Limits,
}

#[test]
fn a_key_the_table_does_not_have_says_to_delete_it_from_its_table() {
    let settings = Settings::from_toml("[limits]\ncalls = 3\nmax_depth = 2\n");
    let problem = settings.get::<Limits>("limits").err().unwrap_or_default();
    assert!(
        problem.contains("unknown field `max_depth`")
            && problem.contains("delete `max_depth` from [limits]"),
        "{problem}"
    );
    // A key of a nested table names that table.
    let nested = Settings::from_toml("[remote]\n[remote.limits]\ncalls = 3\nmax_depth = 2\n");
    let problem = nested.get::<Outer>("remote").err().unwrap_or_default();
    assert!(
        problem.contains("delete `max_depth` from [remote.limits]"),
        "{problem}"
    );
    // A table name toml cannot give back, a key with a dot in it, is left out.
    let dotted = Settings::from_toml("[remote]\n[remote.\"limits.x\"]\nmax_depth = 2\n");
    let problem = (dotted.get::<Dotted>("remote").err()).unwrap_or_default();
    assert!(
        problem.contains("delete `max_depth`") && !problem.contains("from ["),
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

/// Reads `[limits]` only when the terminal front `tui` takes the run.
struct ReadsInItsFront;

impl Plugin for ReadsInItsFront {
    fn build(&self, _app: &mut App) {}

    fn finish(&self, app: &mut App) {
        if app.runs_front("tui") {
            app.add_setting::<Limits>("limits");
        }
    }
}

#[test]
fn a_table_read_once_the_front_is_chosen_refuses_the_run() {
    // A describing build chooses no front, so it would never read the
    // table a terminal start reads and refuses.
    let mut app = App::new();
    app.insert_resource(Settings::from_toml("[limits]\ncalls = 3\n"))
        .insert_resource(Invoked {
            terminal: true,
            ..Invoked::default()
        })
        .add_front(Front::new("tui").on_terminal())
        .add_plugins(ReadsInItsFront);
    app.finish();
    app.cleanup();
    assert!(app.runs_front("tui"));
    let refusal = app.world().resource::<Settings>().refusal();
    assert!(
        refusal
            .as_deref()
            .is_some_and(|why| why.contains("[limits]") && why.contains("`build`")),
        "{refusal:?}"
    );
}
