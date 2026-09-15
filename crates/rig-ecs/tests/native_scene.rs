//! Native construction defaults and checkpoint validation boundaries.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::indexing_slicing)]

use bevy_ecs::prelude::*;
use rig_core::{completion::CompletionRequestBuilder, effect::EffectKind, error::ErrorKind};
use rig_ecs::{
    agent::{
        Assembling, AwaitingModel, Cancelled, Completion, Cursor, DocumentId, Failed, Failure,
        InvalidRetries, OutputRetries, OutputToolName, Owner, ProviderRetried, Run, RunCounter,
        RunSeq, Settled, Turn, Usage, fork,
        scene::{RunScene, SceneEntity, SceneKind, WorldScene, load_world, save_world},
    },
    bus::{PendingEffect, Seq},
};

#[derive(Resource, Default)]
struct InsertedRuns(Vec<(usize, usize, usize, usize, Option<String>, u64)>);

fn observe_run(
    event: On<Add, Run>,
    runs: Query<(
        &Cursor,
        &OutputRetries,
        &InvalidRetries,
        &ProviderRetried,
        &OutputToolName,
        &Usage,
    )>,
    mut observed: ResMut<InsertedRuns>,
) {
    let (cursor, output, invalid, provider, name, usage) = runs.get(event.entity).unwrap();
    observed.0.push((
        cursor.turn,
        output.0,
        invalid.0,
        provider.0,
        name.0.clone(),
        usage.0.input_tokens,
    ));
}

fn observed_world() -> World {
    let mut world = World::new();
    world.init_resource::<InsertedRuns>();
    world.add_observer(observe_run);
    world
}

#[test]
fn explicit_required_values_are_visible_on_initial_run_insertion_and_load() {
    let mut source = observed_world();
    let bare = source.spawn(Run).id();
    assert_eq!(
        source.resource::<InsertedRuns>().0,
        vec![(0, 0, 0, 0, None, 0)]
    );
    source.entity_mut(bare).despawn();
    source.resource_mut::<InsertedRuns>().0.clear();
    source.spawn((
        Run,
        Cursor { turn: 7 },
        OutputRetries(2),
        InvalidRetries(3),
        ProviderRetried(4),
        OutputToolName(Some("chosen".into())),
        Usage(rig_core::completion::Usage {
            input_tokens: 13,
            ..Default::default()
        }),
    ));
    let expected = vec![(7, 2, 3, 4, Some("chosen".into()), 13)];
    assert_eq!(source.resource::<InsertedRuns>().0, expected);
    let scene = RunScene::save(&mut source).unwrap();
    let mut destination = observed_world();
    scene.load(&mut destination).unwrap();
    assert_eq!(destination.resource::<InsertedRuns>().0, expected);
}

#[test]
fn scene_and_fork_preserve_deliberately_absent_bookkeeping() {
    let mut source = World::new();
    source.init_resource::<RunCounter>();
    let run = source.spawn(Run).id();
    source.entity_mut(run).remove::<(
        Cursor,
        OutputRetries,
        InvalidRetries,
        ProviderRetried,
        OutputToolName,
        Usage,
    )>();
    let scene = RunScene::save(&mut source).unwrap();
    let clone = fork(&mut source, run).unwrap();
    let mut destination = World::new();
    let loaded = scene.load(&mut destination).unwrap()[0];
    for (world, entity) in [(&source, clone), (&destination, loaded)] {
        assert!(world.get::<Run>(entity).is_some());
        assert!(world.get::<Cursor>(entity).is_none());
        assert!(world.get::<OutputRetries>(entity).is_none());
        assert!(world.get::<InvalidRetries>(entity).is_none());
        assert!(world.get::<ProviderRetried>(entity).is_none());
        assert!(world.get::<OutputToolName>(entity).is_none());
        assert!(world.get::<Usage>(entity).is_none());
    }
}

#[test]
fn conflicting_raw_phases_never_reach_destination_observers() {
    let phases = [
        "assembling",
        "awaiting_model",
        "resolving_tools",
        "loading_memory",
        "settled",
        "failed",
    ];
    for (at, first) in phases.iter().enumerate() {
        for second in &phases[at + 1..] {
            let mut components = serde_json::Map::new();
            components.insert("run".into(), serde_json::Value::Null);
            for phase in [first, second] {
                let value = if *phase == "failed" {
                    serde_json::to_value(Failed(Failure::MaxTurns { limit: 1 })).unwrap()
                } else {
                    serde_json::Value::Null
                };
                components.insert((*phase).into(), value);
            }
            let scene = WorldScene {
                graph: RunScene {
                    entities: vec![SceneEntity {
                        kind: SceneKind::Run,
                        components,
                        parent: None,
                        relations: Vec::new(),
                    }],
                    ..Default::default()
                },
                ..Default::default()
            };
            let mut destination = observed_world();
            let initial = destination.entities().len();
            assert_eq!(
                scene.graph.load(&mut destination).unwrap_err().kind,
                ErrorKind::Request
            );
            assert_eq!(
                load_world(&scene, &mut destination).unwrap_err().kind,
                ErrorKind::Request
            );
            assert!(destination.resource::<InsertedRuns>().0.is_empty());
            assert_eq!(destination.entities().len(), initial);
        }
    }
}

#[test]
fn fork_refuses_conflicts_before_counter_or_observer_mutation() {
    let mut world = observed_world();
    world.init_resource::<RunCounter>();
    let run = world.spawn((Run, Assembling, AwaitingModel)).id();
    world.resource_mut::<InsertedRuns>().0.clear();
    let initial = world.entities().len();
    assert_eq!(fork(&mut world, run).unwrap_err().kind, ErrorKind::Request);
    assert_eq!(world.entities().len(), initial);
    assert!(world.resource::<InsertedRuns>().0.is_empty());
    world.entity_mut(run).remove::<AwaitingModel>();
    let clone = fork(&mut world, run).unwrap();
    assert_eq!(world.get::<RunSeq>(clone), Some(&RunSeq(0)));
    // A cancellation request may coexist with an ending until the runtime reads it.
    world
        .entity_mut(run)
        .remove::<Assembling>()
        .insert((Settled, Cancelled("late".into())));
    assert!(fork(&mut world, run).is_ok());
}

#[test]
fn completion_identity_restores_from_effect_kind_and_does_not_follow_forks() {
    let mut source = World::new();
    source.init_resource::<RunCounter>();
    source.init_resource::<rig_ecs::bus::SeqCounter>();
    let run = source.spawn((Run, Settled)).id();
    let turn = source.spawn((Turn, ChildOf(run))).id();
    source.spawn((
        PendingEffect::new(
            "model",
            EffectKind::Completion {
                request: CompletionRequestBuilder::unbound("hi").build(),
                stream: false,
            },
        ),
        Completion,
        Seq(0),
        ChildOf(turn),
    ));
    source.spawn((
        PendingEffect::new(
            "other",
            EffectKind::Custom {
                kind: "other".into(),
                payload: serde_json::Value::Null,
            },
        ),
        Seq(1),
        ChildOf(turn),
    ));
    let scene = save_world(&mut source).unwrap();
    let encoded = serde_json::to_vec(&scene).unwrap();
    let scene: WorldScene = serde_json::from_slice(&encoded).unwrap();
    let mut destination = World::new();
    destination.init_resource::<rig_ecs::bus::SeqCounter>();
    let loaded = load_world(&scene, &mut destination).unwrap();
    assert!(destination.get::<Completion>(loaded.effects[0]).is_some());
    assert!(destination.get::<Completion>(loaded.effects[1]).is_none());
    fork(&mut source, run).unwrap();
    assert_eq!(
        source
            .query_filtered::<Entity, With<Completion>>()
            .iter(&source)
            .count(),
        1
    );
}

#[test]
fn scene_indexes_keep_first_occurrence_and_original_vector_order() {
    let mut world = World::new();
    let shared = world
        .spawn((Owner("owner".into()), DocumentId("document".into())))
        .id();
    world.spawn((Run, ChildOf(shared)));
    let (scene, entities) = RunScene::take(&mut world).unwrap();
    assert_eq!(entities[..2], [shared, shared]);
    assert_eq!(
        scene
            .entities
            .iter()
            .map(|row| row.kind)
            .collect::<Vec<_>>(),
        [SceneKind::Agent, SceneKind::Document, SceneKind::Run]
    );
    assert_eq!(scene.entities[2].parent, Some(0));
}
