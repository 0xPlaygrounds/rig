use rig_harness::prelude::*;

use super::*;

/// Data two systems write.
#[derive(Component)]
struct Tally;

fn first_writer(_: Query<&mut Tally>) {}

fn second_writer(_: Query<&mut Tally>) {}

#[test]
fn unordered_writers_in_one_stage_fail_by_name() {
    assert_stages_ordered(|_: &mut App| {});
    let unordered = |app: &mut App| {
        app.add_systems(AgentPass, (first_writer, second_writer).in_set(Stage::Gate));
    };
    let failure = std::panic::catch_unwind(|| assert_stages_ordered(unordered)).err();
    let report = failure
        .and_then(|payload| payload.downcast::<String>().ok())
        .unwrap_or_default();
    assert!(
        report.contains("first_writer") && report.contains("second_writer"),
        "{report}"
    );
}

/// Each essential plugin is named as the default `plugins.toml` names it,
/// a path the generated `main.rs` compiles (`DEFAULTS`), which is what the
/// launcher and `/plugins` match: its type's own path
/// (`std::any::type_name`) may name a private module.
#[test]
fn each_essential_plugin_is_named_by_its_default_entry() {
    let defaults: Vec<&str> = DEFAULTS.iter().map(|(path, _)| *path).collect();
    let essential = rig_harness::harness_protocol::ESSENTIAL_PLUGINS;
    let unknown: Vec<&str> = (essential.iter().map(|(path, _)| *path))
        .filter(|path| !defaults.contains(path))
        .collect();
    assert!(unknown.is_empty(), "no default plugin is named {unknown:?}");
}
