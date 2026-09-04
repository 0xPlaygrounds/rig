//! The corpus's third interpreter, the record-by-record one: every golden
//! log replayed through a Bevy `World` by id.
//!
//! A golden is a trace: each record names its id, its key, the effect, the
//! answer, its parent and — for a streamed dispatch the recorder kept
//! verbatim — its events. rig-ecs loads every record as an effect entity
//! under its recorded id (`Replay::load`), a child of its recorded parent,
//! and registers a by-id replayer per key (`Replay::register`); the
//! world's `Dispatch` re-issues them, the replayers answer each from the
//! record of its own id, and `Collect` lands the outcomes. The world's own
//! log of the replay must then be the golden again: the same ids, keys,
//! outcomes, parents and — through `Streamed` — the same events in order.
//!
//! No agent loop runs here: this is the bus half of the corpus, over every
//! golden the two agent interpreters produced, so a golden that neither
//! interpreter can replay without rig-agent still replays here. One row
//! per golden, counted.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::time::{Duration, Instant};

use bevy_app::App;
use bevy_ecs::schedule::LogLevel;
use rig_core::serve::ServingPolicy;
use rig_ecs::bus::{
    BusPlugin, EffectLogResource, EffectOutcome, Handlers, Issued, Replay, Streamed,
};
use rig_effect_log::{EffectLog, EffectLogRecorder};

/// The goldens the two agent interpreters replay: the same 207 files.
const EXPECTED_GOLDENS: usize = 207;

const GUARD: Duration = Duration::from_secs(30);

fn goldens() -> Vec<(String, EffectLog)> {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .expect("the fixtures directory")
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            name.strip_suffix(".effects.json").map(str::to_owned)
        })
        .collect();
    names.sort();
    names
        .into_iter()
        .map(|name| {
            let text = std::fs::read_to_string(dir.join(format!("{name}.effects.json")))
                .expect("the golden is committed");
            let log = serde_json::from_str(&text).expect("the golden loads");
            (name, log)
        })
        .collect()
}

/// A world under the golden's own serving policy (its header's `bus`, or
/// the default when the golden was recorded on a host's bus), with the
/// intake bound lifted so one tick takes every record it can.
fn world(log: &EffectLog) -> App {
    let mut app = App::new();
    let policy = ServingPolicy {
        command_capacity: 10_000,
        ..log.header.bus.unwrap_or_default()
    };
    app.add_plugins(BusPlugin::with_policy(policy).ambiguity_detection(LogLevel::Error));
    app.finish();
    app.cleanup();
    app
}

/// Replay one golden through a fresh world and compare, returning the
/// number of records replayed.
fn replay_through_a_world(name: &str, log: &EffectLog) -> usize {
    let mut app = world(log);
    Handlers::with(app.world_mut(), |handlers| {
        Replay::default().register(handlers, log)
    })
    .expect("a bus")
    .unwrap_or_else(|report| panic!("{name}: the golden registers: {report}"));
    // The golden's recorder kept events, or it did not: the world's does
    // the same, so the log it writes is comparable field for field.
    let recorder = if log.records.iter().any(|record| record.events.is_some()) {
        EffectLogRecorder::keeping_stream_events()
    } else {
        EffectLogRecorder::new()
    };
    EffectLogResource::install(app.world_mut(), recorder);
    let entities = Replay::load(app.world_mut(), log);
    assert_eq!(
        entities.len(),
        log.records.len(),
        "{name}: one entity per record"
    );

    let start = Instant::now();
    loop {
        app.update();
        let world = app.world();
        if entities
            .iter()
            .all(|entity| world.get::<EffectOutcome>(*entity).is_some())
        {
            break;
        }
        assert!(
            start.elapsed() < GUARD,
            "{name}: not replayed within {GUARD:?}"
        );
        std::thread::yield_now();
    }

    let world = app.world();
    let mut records: Vec<_> = log.records.iter().collect();
    records.sort_by_key(|record| record.id);
    for (entity, record) in entities.iter().zip(&records) {
        assert_eq!(
            world.get::<Issued>(*entity).expect("issued").0,
            record.id,
            "{name}: the recorded id"
        );
        let outcome = world.get::<EffectOutcome>(*entity).expect("answered");
        assert_eq!(
            serde_json::to_value(&outcome.0).expect("serde"),
            serde_json::to_value(&record.outcome).expect("serde"),
            "{name}: record {} replays its outcome",
            record.id
        );
        if let Some(events) = &record.events {
            let streamed = world
                .get::<Streamed>(*entity)
                .unwrap_or_else(|| panic!("{name}: record {} streamed", record.id));
            assert_eq!(
                serde_json::to_value(&streamed.events).expect("serde"),
                serde_json::to_value(events).expect("serde"),
                "{name}: record {} replays its events in order",
                record.id
            );
        }
    }
    // The world's log is in begin order, as rig-bus's is; under serial
    // serving that is not id order in either runtime, so both sides are
    // compared by id.
    let mut replayed = world.resource::<EffectLogResource>().log().records;
    replayed.sort_by_key(|record| record.id);
    assert_eq!(replayed.len(), records.len(), "{name}: the world's log");
    for (mine, theirs) in replayed.iter().zip(&records) {
        assert_eq!(mine.id, theirs.id, "{name}");
        assert_eq!(mine.key, theirs.key, "{name}");
        assert_eq!(mine.parent, theirs.parent, "{name}: causality survives");
        assert_eq!(
            serde_json::to_value(&mine.kind).expect("serde"),
            serde_json::to_value(&theirs.kind).expect("serde"),
            "{name}: the request is the record's"
        );
        assert_eq!(
            serde_json::to_value(&mine.outcome).expect("serde"),
            serde_json::to_value(&theirs.outcome).expect("serde"),
            "{name}"
        );
        assert_eq!(
            serde_json::to_value(&mine.events).expect("serde"),
            serde_json::to_value(&theirs.events).expect("serde"),
            "{name}: kept events are kept again"
        );
    }
    records.len()
}

#[test]
fn every_golden_replays_through_a_world_by_id() {
    let goldens = goldens();
    assert_eq!(
        goldens.len(),
        EXPECTED_GOLDENS,
        "the corpus has {EXPECTED_GOLDENS} goldens; update the count with the corpus"
    );
    let mut rows = Vec::with_capacity(goldens.len());
    let mut records = 0;
    let mut streamed = 0;
    for (name, log) in &goldens {
        let replayed = replay_through_a_world(name, log);
        records += replayed;
        streamed += log
            .records
            .iter()
            .filter(|record| record.events.is_some())
            .count();
        rows.push(format!("{name}: {replayed} records"));
    }
    eprintln!(
        "{} goldens, {records} records ({streamed} with kept events) replayed through a world by id",
        goldens.len()
    );
    assert_eq!(rows.len(), EXPECTED_GOLDENS);
}
