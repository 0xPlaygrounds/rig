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
//! log of the replay must then be the golden again, record for record: the
//! same ids, keys, requests, outcomes, parents, scopes and — through
//! `Streamed` — the same events in order. The header is not compared: the
//! world writes no hook list, no run spec and no program row, which are
//! the agent interpreters' (and rig-ecs PR 2's contract).
//!
//! No agent loop runs here: this is the bus half of the corpus, over every
//! golden the two agent interpreters produced, so a golden that neither
//! interpreter can replay without rig-agent still replays here. One row
//! per golden, counted.

#![allow(clippy::expect_used, clippy::panic)]

#[path = "corpus/fixtures.rs"]
mod fixtures;
#[allow(
    dead_code,
    reason = "each world-replay target uses one half of the check"
)]
#[path = "corpus/replay_check.rs"]
mod replay_check;

use replay_check::replay_through_a_world;
use rig_cassette::effect_log::EffectLog;

/// The goldens the two agent interpreters replay: the same files, the whole
/// corpus (the contract matrix on five more wires grew it past the original
/// 207; the failure rows on those wires past 685).
const EXPECTED_GOLDENS: usize = 206;

fn goldens() -> Vec<(String, EffectLog)> {
    let dir = fixtures::effects_dir();
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
