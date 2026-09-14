//! Two independent consumers of the public collected-item notification. The
//! recorder is an independent oracle; these traces are never reconstructed from
//! final text or used as the source of a provider request.
#![allow(dead_code, reason = "focused stream rows run on five provider columns")]

use std::collections::BTreeMap;

use bevy_app::App;
use bevy_ecs::prelude::*;
use rig::effect_log::EffectLog;
use rig::error::ErrorReport;
use rig::streaming::StreamEvent;
use rig_ecs::bus::{EffectOutcome, Issued, StreamItemsDelivered, Streamed};
use serde::Serialize;

use super::cells::Cell;

type Item = Result<StreamEvent, ErrorReport>;

#[derive(Clone, Debug, Serialize)]
pub(crate) struct Batch {
    effect: u64,
    entity: u64,
    turn_entity: Option<u64>,
    run_entity: Option<u64>,
    start: usize,
    items: Vec<Item>,
}

#[derive(Resource, Default)]
struct Consumer<const N: usize>(Vec<Batch>);

#[derive(Clone, Debug, Serialize)]
pub(crate) struct Snapshot {
    first: Vec<Batch>,
    second: Vec<Batch>,
}

pub(crate) fn applicable(cell: &Cell) -> bool {
    matches!(
        cell.name,
        "checkpoint_multi_turn_streamed"
            | "shaping_extra_context_streamed"
            | "serving_concurrent_concurrency_two_events"
    )
}

pub(crate) fn install(app: &mut App) {
    app.init_resource::<Consumer<0>>()
        .init_resource::<Consumer<1>>()
        .add_observer(consume::<0>)
        .add_observer(consume::<1>);
}

fn consume<const N: usize>(
    delivered: On<StreamItemsDelivered>,
    streams: Query<(&Streamed, Option<&EffectOutcome>, Option<&ChildOf>)>,
    parents: Query<&ChildOf>,
    mut consumer: ResMut<Consumer<N>>,
) {
    let (streamed, outcome, parent) = streams
        .get(delivered.effect)
        .expect("the observed effect is still present");
    assert!(
        outcome.is_none(),
        "delivery precedes effect-outcome publication"
    );
    assert!(!delivered.items.is_empty());
    let id = delivered.id.as_u64();
    let previous = consumer
        .0
        .iter()
        .filter(|batch| batch.effect == id)
        .map(|batch| batch.items.len())
        .sum::<usize>();
    assert_eq!(
        delivered.start, previous,
        "each consumer receives this effect without gaps or duplicates"
    );
    let durable = durable_items(streamed);
    let end = delivered.start + delivered.items.len();
    assert!(
        durable.len() >= end,
        "durable state is visible before notification"
    );
    assert_eq!(
        serde_json::to_value(&durable[delivered.start..end]).unwrap(),
        serde_json::to_value(&delivered.items).unwrap(),
        "notification is the actual newly collected durable slice, including errors"
    );
    let turn = parent.map(ChildOf::parent);
    let run = turn
        .and_then(|turn| parents.get(turn).ok())
        .map(ChildOf::parent);
    consumer.0.push(Batch {
        effect: id,
        entity: delivered.effect.to_bits(),
        turn_entity: turn.map(Entity::to_bits),
        run_entity: run.map(Entity::to_bits),
        start: delivered.start,
        items: delivered.items.clone(),
    });
}

fn durable_items(streamed: &Streamed) -> Vec<Item> {
    merge_items(
        &streamed.events,
        streamed
            .errors
            .iter()
            .map(|(position, error)| (*position, error)),
    )
}

fn merge_items<'a>(
    events: &[StreamEvent],
    errors: impl Iterator<Item = (usize, &'a ErrorReport)>,
) -> Vec<Item> {
    let errors: BTreeMap<_, _> = errors.collect();
    let mut events_iter = events.iter();
    (0..events.len() + errors.len())
        .map(|position| match errors.get(&position) {
            Some(error) => Err((*error).clone()),
            None => Ok(events_iter
                .next()
                .expect("event/error position partition")
                .clone()),
        })
        .collect()
}

fn snapshot(world: &World) -> Snapshot {
    Snapshot {
        first: world.resource::<Consumer<0>>().0.clone(),
        second: world.resource::<Consumer<1>>().0.clone(),
    }
}

/// Both observers exist before scene load. Hydration reads accumulated state;
/// the empty observer traces prove loading that history emits no live items.
pub(crate) fn assert_hydration(world: &mut World, cell: &Cell) {
    let traces = snapshot(world);
    assert!(
        traces.first.is_empty() && traces.second.is_empty(),
        "loading historical streams is not live delivery"
    );
    let hydrated: Vec<_> = world.query::<(&Issued, &Streamed)>().iter(world)
        .map(|(issued, streamed)| serde_json::json!({"effect":issued.0.as_u64(), "items":durable_items(streamed), "text":streamed.text})).collect();
    assert!(
        !hydrated.is_empty(),
        "the cut retains streams for explicit hydration"
    );
    write_evidence(
        cell,
        "hydrate",
        &serde_json::json!({"historical":hydrated,"live":traces}),
    );
}

pub(crate) fn save_head(world: &World, cell: &Cell) -> Snapshot {
    let traces = snapshot(world);
    assert_eq!(
        serde_json::to_value(&traces.first).unwrap(),
        serde_json::to_value(&traces.second).unwrap(),
        "independent consumers agree through the cut"
    );
    write_evidence(cell, "head", &serde_json::to_value(&traces).unwrap());
    traces
}

fn by_effect(batches: &[Batch]) -> BTreeMap<u64, Vec<Item>> {
    let mut result: BTreeMap<u64, Vec<Item>> = BTreeMap::new();
    for batch in batches {
        let items = result.entry(batch.effect).or_default();
        assert_eq!(batch.start, items.len());
        items.extend(batch.items.clone());
    }
    result
}

/// Compare complete item sequences against independent recorded adapter events
/// and positioned errors. Batch grouping is preserved as evidence, not asserted
/// identical across separate HTTP replay runs.
pub(crate) fn assert_complete(world: &World, cell: &Cell, log: &EffectLog, head: Option<Snapshot>) {
    let tail = snapshot(world);
    write_evidence(cell, "tail", &serde_json::to_value(&tail).unwrap());
    let mut traces = head.unwrap_or(Snapshot {
        first: Vec::new(),
        second: Vec::new(),
    });
    let head_ids: Vec<_> = traces.first.iter().map(|batch| batch.effect).collect();
    assert!(
        tail.first
            .iter()
            .all(|batch| !head_ids.contains(&batch.effect)),
        "restored consumers receive only new tail effects"
    );
    traces.first.extend(tail.first);
    traces.second.extend(tail.second);
    assert_eq!(
        serde_json::to_value(&traces.first).unwrap(),
        serde_json::to_value(&traces.second).unwrap(),
        "two independent observers see the same actual ordered delivery"
    );
    let mut expected = BTreeMap::new();
    for record in &log.records {
        if let Some(events) = &record.events {
            let errors = log
                .header
                .stream_errors
                .get(&record.id)
                .into_iter()
                .flatten()
                .map(|error| (error.item, &error.error));
            let items = merge_items(events, errors);
            if !items.is_empty() {
                expected.insert(record.id.as_u64(), items);
            }
        }
    }
    assert!(
        !expected.is_empty(),
        "a retained adapter stream supplies the independent oracle"
    );
    assert_eq!(
        serde_json::to_value(by_effect(&traces.first)).unwrap(),
        serde_json::to_value(&expected).unwrap(),
        "all text/tool/terminal/error items match the actual recorder, not just the answer"
    );
    write_evidence(
        cell,
        "complete",
        &serde_json::json!({"consumers":traces,"recorded_items":expected}),
    );
}

fn write_evidence(cell: &Cell, phase: &str, value: &serde_json::Value) {
    use sha2::{Digest, Sha256};
    let Some(directory) = std::env::var_os("RIG_STREAM_DELIVERY_EVIDENCE_DIR")
        .or_else(|| std::env::var_os("RIG_CHECKPOINT_ATTEMPT_DIR"))
    else {
        return;
    };
    let directory = std::path::PathBuf::from(directory);
    std::fs::create_dir_all(&directory).expect("consumer evidence directory");
    let bytes = serde_json::to_vec_pretty(&crate::cassettes::scrub_artifact(value))
        .expect("consumer trace JSON");
    let filename = format!("{}-consumer-{phase}.json", cell.name);
    let path = directory.join(&filename);
    assert!(
        crate::cassettes::artifact_safety_failures(
            &path,
            std::str::from_utf8(&bytes).expect("trace JSON UTF-8"),
        )
        .is_empty(),
        "external consumer evidence contains no sensitive data"
    );
    std::fs::write(path, &bytes).expect("write consumer trace");
    let metadata = serde_json::json!({"cell":cell.name,"phase":phase,"file":filename,"bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(&bytes)),"comparison":"ordered semantic items; collection batch boundaries are captured, not equated across independent runs"});
    std::fs::write(
        directory.join(format!("{}-consumer-{phase}.metadata.json", cell.name)),
        serde_json::to_vec_pretty(&metadata).unwrap(),
    )
    .expect("write trace metadata");
}
