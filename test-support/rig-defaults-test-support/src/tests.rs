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
