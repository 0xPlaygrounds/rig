use rig_core::effect::{EffectId, EffectKind, Outcome};
use rig_harness::harness_protocol::{Home, SessionId};
use rig_harness_test_support::{app_on, scratch_home};

use super::*;

/// A record of effect `id`, a plugin's own of the kind `kind`, made by
/// the agent `agent` in its turn `turn`.
fn record(id: u64, agent: &str, kind: &str, turn: u64) -> EffectRecord {
    let kind = EffectKind::Custom {
        kind: kind.into(),
        payload: rig_core::serde_json::json!({ "id": id }),
    };
    let outcome = Ok(Outcome::Custom {
        payload: rig_core::serde_json::Value::Null,
    });
    let mut record = EffectRecord::new(EffectId::from_raw(id), "plugin".into(), kind, outcome);
    record.scope = Some(agent.into());
    record.turn = Some(turn);
    record
}

#[test]
fn the_records_of_one_turn_of_an_agent_read_back() {
    let home: Home = scratch_home("effect-log-home");
    let dir = home.session(&SessionId::generate());
    std::fs::create_dir_all(dir.path()).ok();
    let (mut app, _wakes) = app_on(&rig_cassette::journal::MemoryStore::default());
    app.insert_resource(SessionPaths(dir))
        .add_plugins((AgentPlugin, EffectLogPlugin));
    app.finish();
    app.update();
    let log = app.world().resource::<EffectLog>().clone();
    assert_eq!(log.read().map(|records| records.len()).ok(), Some(0));
    for record in [
        record(1, "a", "note", 0),
        record(2, "a", "subagent", 1),
        record(3, "a", "note", 1),
        record(4, "b", "note", 1),
    ] {
        app.world_mut().write_message(Recorded(record));
    }
    app.update();

    let ids = |filter: &EffectFilter| {
        let records = log.read_matching(filter).unwrap_or_default();
        records
            .iter()
            .map(|record| record.id.as_u64())
            .collect::<Vec<_>>()
    };
    let turn = EffectFilter::default().agent("a").turn(1);
    assert_eq!(ids(&turn), [2, 3]);
    assert_eq!(ids(&turn.clone().kind("subagent")), [2]);
    assert_eq!(ids(&EffectFilter::default().turn(1)), [2, 3, 4]);
    assert_eq!(ids(&EffectFilter::default()), [1, 2, 3, 4]);
}
