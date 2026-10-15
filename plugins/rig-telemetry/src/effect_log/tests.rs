use rig_core::effect::{Answered, EffectId, EffectKind};
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::app_on;

use super::*;

/// A record of effect `id`, a plugin's own of the kind `kind`, made by
/// the agent `agent` in its turn `turn`.
fn record(id: u64, agent: &str, kind: &str, turn: u64) -> EffectRecord {
    let kind = EffectKind::Custom {
        kind: kind.into(),
        payload: rig_core::serde_json::json!({ "id": id }),
    };
    let outcome = Ok(Answered::Custom {
        payload: rig_core::serde_json::Value::Null,
    });
    let mut record = EffectRecord::new(EffectId::from_raw(id), "plugin".into(), kind, outcome);
    record.scope = Some(agent.into());
    record.turn = Some(turn);
    record
}

/// The log is a file of the session's store, whatever keeps it.
#[test]
fn the_records_of_one_turn_of_an_agent_read_back_from_the_store() {
    let store = MemoryStore::default();
    let (mut app, _wakes) = app_on(&store);
    app.add_plugins((AgentPlugin, EffectLogPlugin));
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
        app.world_mut().write_message(EffectRecorded(record));
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
    assert!(
        store
            .read_file(SESSION_EFFECT_LOG)
            .is_ok_and(|log| !log.is_empty())
    );
}

/// A record of effect `id` whose payload says `text`.
fn saying(id: u64, text: &str) -> EffectRecord {
    let mut record = record(id, "a", "note", 0);
    record.kind = EffectKind::Custom {
        kind: "note".into(),
        payload: rig_core::serde_json::json!({ "text": text }),
    };
    record
}

/// A rewrite of the session reaches the records written before it, the
/// records settled in its frame and not written yet, and the writer's
/// memory of the file.
#[test]
fn a_rewrite_reaches_the_log_written_and_the_records_not_written_yet() {
    const SECRET: &str = "sk-123";
    let store = MemoryStore::default();
    let (mut app, _wakes) = app_on(&store);
    app.add_plugins((AgentPlugin, SessionLogPlugin, EffectLogPlugin));
    app.finish();
    app.update();
    app.world_mut()
        .write_message(EffectRecorded(saying(1, &format!("key {SECRET}"))));
    app.update();
    app.world_mut()
        .write_message(EffectRecorded(saying(2, &format!("again {SECRET}"))));
    app.world_mut().write_message(RewriteSession {
        by: Author::plugin("redact"),
        with: Arc::new(Strings(|text: &mut String| {
            let hit = text.contains(SECRET);
            if hit {
                *text = text.replace(SECRET, "[REDACTED]");
            }
            hit
        })),
        asker: None,
    });
    app.update();

    let file = store.read_file(SESSION_EFFECT_LOG).unwrap_or_default();
    let file = String::from_utf8_lossy(&file);
    assert!(!file.contains(SECRET), "{file}");
    let log = app.world().resource::<EffectLog>().clone();
    let texts: Vec<String> = log
        .read()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|record| match record.kind {
            EffectKind::Custom { payload, .. } => Some(payload.get("text")?.as_str()?.to_owned()),
            _ => None,
        })
        .collect();
    assert_eq!(texts, ["key [REDACTED]", "again [REDACTED]"]);
    let writer = format!("{:?}", app.world().resource::<EffectLogFile>().writer);
    assert!(!writer.contains(SECRET), "{writer}");
}

/// A resumed session's calls show what they started in the terminal view:
/// the log is read back once, off the main thread, and sent to it.
#[cfg(feature = "tui")]
#[test]
fn a_resumed_session_sends_its_log_to_the_terminal_view() {
    use rig_harness::prelude::Front;
    use rig_harness_test_support::{Collected, collect, run_until};
    use rig_tui::PastEffects;

    let store = MemoryStore::default();
    let (mut before, _wakes) = app_on(&store);
    before.add_plugins((AgentPlugin, SessionLogPlugin, EffectLogPlugin));
    before.finish();
    before.update();
    for record in [record(1, "a", "note", 0), record(2, "a", "subagent", 0)] {
        before.world_mut().write_message(EffectRecorded(record));
    }
    before.update();
    drop(before);

    let (mut app, wakes) = app_on(&store);
    // The view's front, taking this headless run.
    app.add_front(Front::new(rig_tui::FRONT).headless())
        .add_message::<PastEffects>()
        .add_plugins((AgentPlugin, SessionLogPlugin, EffectLogPlugin));
    collect::<PastEffects>(&mut app);
    app.finish();
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Collected<PastEffects>>().0.is_empty()
    });
    let sent = &app.world().resource::<Collected<PastEffects>>().0;
    let counts: Vec<usize> = sent.iter().map(PastEffects::len).collect();
    assert_eq!(counts, [2]);
}
