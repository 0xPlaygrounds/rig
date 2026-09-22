//! Capture native configuration evidence for world-corpus compatibility checks.

use std::{cell::RefCell, collections::BTreeMap, future::Future, path::Path};

use bevy_ecs::prelude::*;
use rig_cassette::effect_log::EffectLog;
use rig_core::serve::ServingPolicy;
use rig_ecs::{
    agent::RunOf,
    bus::{Policy, Scope},
    checkpoint::Checkpoint,
};

type Programs = BTreeMap<String, (ServingPolicy, Checkpoint)>;

tokio::task_local! {
    static PROGRAMS: RefCell<Programs>;
}

/// Run a native producer with task-local pre-dispatch configuration capture.
/// Nested capture scopes are independent. The future's output is unchanged.
pub async fn capture_world_programs<F: Future>(future: F) -> F::Output {
    PROGRAMS.scope(RefCell::new(BTreeMap::new()), future).await
}

/// Save a run's declared configuration before dispatch for world-corpus replay.
/// Does nothing outside a capture scope. Panics if the run cannot be saved.
pub fn capture_world_program(world: &mut World, run: Entity, log: &EffectLog) {
    if PROGRAMS.try_with(|_| ()).is_err() {
        return;
    }
    assert!(
        world.get::<RunOf>(run).is_some(),
        "capture a run, not an agent"
    );
    let scope = world.get::<Scope>(run).expect("run scope").0.clone();
    let already_captured = PROGRAMS.with(|programs| programs.borrow().contains_key(&scope));
    if already_captured {
        return;
    }
    rig_cassette::ecs::identity::check_replayable(world, run, log)
        .expect("the producer's declared configuration is replay-compatible");
    let policy = world.resource::<Policy>().0;
    let mut scene =
        rig_ecs::checkpoint::save_world(world).expect("save pre-dispatch configuration");
    // This scene proves configuration, not execution. An intentionally unserved
    // startup intent must not require an implementation during configuration restore.
    for entity in &mut scene.entities {
        entity.remove(std::any::type_name::<rig_ecs::bus::PendingEffect>());
    }
    PROGRAMS.with(|programs| {
        programs.borrow_mut().insert(scope, (policy, scene));
    });
}

pub(super) fn programs(path: &Path, log: &EffectLog, regenerate: bool) {
    let scenes = PROGRAMS.with(|programs| programs.borrow().clone());
    assert_eq!(
        scenes.keys().collect::<Vec<_>>(),
        log.header.programs.keys().collect::<Vec<_>>(),
        "capture exactly the log's program scopes"
    );
    assert!(
        !scenes.is_empty(),
        "a native golden declares at least one program"
    );
    if regenerate {
        let text = serde_json::to_string_pretty(&scenes).expect("program scenes serialize");
        std::fs::write(path, format!("{text}\n")).expect("write program scenes");
    } else {
        let text = std::fs::read_to_string(path).expect("committed world program scenes");
        let committed: Programs = serde_json::from_str(&text).expect("valid world program scenes");
        assert_eq!(
            committed.keys().collect::<Vec<_>>(),
            scenes.keys().collect::<Vec<_>>()
        );
    }
}

/// Exclude only observed delivery boundaries from fresh native-run comparisons.
/// HTTP-backed unary, streamed, and concurrent cells race external readiness
/// against collection passes, so both batch numbers and stream groupings vary.
/// Raw deliveries remain in fixtures for replay; program identities, records,
/// stream errors, and all other header fields remain part of the comparison.
pub(super) fn comparison(mut value: serde_json::Value) -> serde_json::Value {
    value
        .get_mut("header")
        .and_then(serde_json::Value::as_object_mut)
        .expect("effect log header")
        .remove("deliveries");
    value
}

#[cfg(test)]
#[path = "world/tests.rs"]
mod tests;
