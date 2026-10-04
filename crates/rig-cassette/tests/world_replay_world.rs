//! Validate native program configurations and replay world goldens by effect id.

#![allow(clippy::expect_used, clippy::panic)]

#[path = "corpus/fixtures.rs"]
mod fixtures;

#[allow(
    dead_code,
    reason = "each world-replay target uses one half of the check"
)]
#[path = "corpus/replay_check.rs"]
mod replay_check;

use replay_check::{Programs, check_programs, replay_world_log as replay};
use rig_cassette::effect_log::EffectLog;
use rig_core::serve::ServingPolicy;

const EXPECTED_GOLDENS: usize = 9;

#[test]
#[should_panic(expected = "live task tool")]
fn a_live_tool_replacing_the_recorded_handler_is_detected() {
    let name = "openai_chat_long_task_inventory_restore";
    let text = std::fs::read_to_string(
        fixtures::effects_dir()
            .join("world")
            .join(format!("{name}.effects.json")),
    )
    .expect("task golden");
    let log: EffectLog = serde_json::from_str(&text).expect("task log");
    replay(name, &log, ServingPolicy::default(), true);
}

#[test]
fn every_world_golden_checks_its_programs_and_replays_by_id() {
    let directory = fixtures::effects_dir().join("world");
    let mut names: Vec<_> = std::fs::read_dir(&directory)
        .expect("world corpus")
        .map(|entry| {
            entry
                .expect("world fixture")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .filter_map(|name| name.strip_suffix(".effects.json").map(str::to_owned))
        .collect();
    names.sort();
    assert_eq!(
        names.len(),
        EXPECTED_GOLDENS,
        "update the count with the world corpus"
    );
    for name in &names {
        let text = std::fs::read_to_string(directory.join(format!("{name}.effects.json")))
            .expect("world golden");
        let log: EffectLog = serde_json::from_str(&text).expect("world log");
        let text = std::fs::read_to_string(directory.join(format!("{name}.programs.json")))
            .expect("world program scenes");
        let programs: Programs = serde_json::from_str(&text).expect("world program scenes decode");
        let policy = check_programs(name, &log, &programs);
        replay(name, &log, policy, false);
    }
    eprintln!("{} world goldens checked and replayed by id", names.len());
}
