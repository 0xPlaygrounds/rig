//! Bounded relationship traversal and order-preserving scene lookup contracts.
#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use std::sync::Arc;

use bevy_ecs::{prelude::*, system::RunSystemOnce};
use rig_core::{
    effect::EffectId,
    error::ErrorKind,
    observe::{Action, ObservationLog},
};
use rig_ecs::bus::{
    EffectOutcome, Held, InFlight, Issued, PendingEffect, Reserved, Scene, Scope, Seq, SubjectWalk,
    Subjects, Witnessing, dispatch,
};

use crate::bus_support::{self, completion};

#[test]
fn subject_scope_includes_self_but_issued_parent_does_not() {
    let mut world = World::new();
    world.init_resource::<rig_ecs::bus::IdCounter>();
    let parent_id = EffectId::from_raw(4);
    let own_id = EffectId::from_raw(8);
    let parent = world.spawn((Issued(parent_id), Scope("outer".into()))).id();
    let child = world
        .spawn((
            ChildOf(parent),
            Issued(own_id),
            Scope("inner".into()),
            Seq(12),
        ))
        .id();
    let subject = world
        .run_system_once(move |walk: SubjectWalk| walk.of_entity(child))
        .unwrap();
    assert_eq!(subject.effect, Some(own_id));
    assert_eq!(subject.parent, Some(parent_id));
    assert_eq!(subject.scope.as_deref(), Some("inner"));
    assert_eq!(subject.order, Some(12));
    world.entity_mut(child).remove::<Scope>();
    let scope = world
        .run_system_once(move |walk: SubjectWalk| walk.of_scope(child))
        .unwrap();
    assert_eq!(scope.scope.as_deref(), Some("outer"));
    assert_eq!(scope.effect, None);
    assert_eq!(scope.parent, None);
}

#[test]
fn witness_preserves_scope_and_parent_walk_boundaries() {
    let mut world = World::new();
    world.init_resource::<rig_ecs::bus::IdCounter>();
    let id = EffectId::from_raw(9);
    let root = world
        .spawn((Issued(id), Scope("outside-scope-budget".into())))
        .id();
    let mut leaf = root;
    for _ in 0..4096 {
        leaf = world.spawn(ChildOf(leaf)).id();
    }
    let subject = world
        .run_system_once(move |walk: SubjectWalk| walk.of_entity(leaf))
        .unwrap();
    // The old witness inspected self plus 4095 ancestors for scope, but
    // followed 4096 parent edges when finding an issued ancestor.
    assert_eq!(subject.scope, None);
    assert_eq!(subject.parent, Some(id));
    let beyond = world.spawn(ChildOf(leaf)).id();
    let subject = world
        .run_system_once(move |walk: SubjectWalk| walk.of_entity(beyond))
        .unwrap();
    assert_eq!(subject.parent, None);
}

#[test]
fn dispatch_subjects_resolve_parents_issued_in_the_same_pass() {
    let mut app = bus_support::app();
    let log = Arc::new(ObservationLog::default());
    Witnessing::install(app.world_mut(), log.clone());
    bus_support::register(
        &mut app,
        "model",
        bus_support::MockModel::new(&Arc::new(bus_support::Counters::default())),
    );
    let parent_id = EffectId::from_raw(20);
    let child_id = EffectId::from_raw(21);
    let parent = app
        .world_mut()
        .spawn((
            PendingEffect::new("model", completion()),
            Reserved(parent_id),
            Scope("parent".into()),
        ))
        .id();
    app.world_mut().spawn((
        PendingEffect::new("model", completion()),
        Reserved(child_id),
        ChildOf(parent),
        Scope("child".into()),
    ));
    app.world_mut().run_system_once(dispatch).unwrap();
    let trace = log.trace();
    let issued: Vec<_> = trace
        .observations
        .iter()
        .filter(|fact| matches!(fact.action, Action::Issued))
        .map(|fact| &fact.subject)
        .collect();
    assert_eq!(issued.len(), 2);
    assert_eq!(issued[0].effect, Some(parent_id));
    assert_eq!(issued[0].parent, None);
    assert_eq!(issued[0].scope.as_deref(), Some("parent"));
    assert_eq!(issued[1].effect, Some(child_id));
    assert_eq!(issued[1].parent, Some(parent_id));
    assert_eq!(issued[1].scope.as_deref(), Some("child"));
}

#[test]
fn malformed_cycles_terminate_dispatch_and_never_make_self_an_issued_parent() {
    let mut app = bus_support::serial_app();
    let log = Arc::new(ObservationLog::default());
    Witnessing::install(app.world_mut(), log);
    let entity = app
        .world_mut()
        .spawn(PendingEffect::new("model", completion()))
        .id();
    let ancestor = app.world_mut().spawn(ChildOf(entity)).id();
    app.world_mut().entity_mut(entity).insert(ChildOf(ancestor));
    // A busy key outside the cycle must defer rather than hang or refuse.
    app.world_mut().spawn(InFlight {
        key: "model".into(),
    });
    app.world_mut().run_system_once(dispatch).unwrap();
    assert!(app.world().get::<EffectOutcome>(entity).is_none());
    assert!(app.world().get::<Issued>(entity).is_none());
    // An in-flight ancestor on the cycle really is a serial reentrant call.
    app.world_mut().entity_mut(ancestor).insert(InFlight {
        key: "model".into(),
    });
    app.world_mut().run_system_once(dispatch).unwrap();
    assert_eq!(
        app.world()
            .get::<EffectOutcome>(entity)
            .unwrap()
            .0
            .as_ref()
            .unwrap_err()
            .kind,
        ErrorKind::Request
    );
    app.world_mut()
        .entity_mut(entity)
        .insert(Issued(EffectId::from_raw(3)));
    let subject = app
        .world_mut()
        .run_system_once(move |subjects: Subjects| subjects.of(entity))
        .unwrap();
    assert_eq!(subject.effect, Some(EffectId::from_raw(3)));
    assert_eq!(subject.parent, None);
    assert_eq!(subject.scope, None);

    // A subject outside a cycle never revisits itself: the finite budget,
    // rather than the self check alone, must still terminate the walk.
    let descendant = app
        .world_mut()
        .spawn((
            PendingEffect::new("absent", completion()),
            ChildOf(ancestor),
        ))
        .id();
    app.world_mut().entity_mut(entity).remove::<Issued>();
    app.world_mut().run_system_once(dispatch).unwrap();
    assert_eq!(
        app.world()
            .get::<EffectOutcome>(descendant)
            .unwrap()
            .0
            .as_ref()
            .unwrap_err()
            .kind,
        ErrorKind::HandlerUnavailable
    );
}

#[test]
fn scene_indices_preserve_seq_ties_forward_parents_and_external_references() {
    let mut app = bus_support::app();
    let external = app.world_mut().spawn_empty().id();
    let parent = app
        .world_mut()
        .spawn((PendingEffect::new("parent", completion()), Held))
        .id();
    let child = app
        .world_mut()
        .spawn((PendingEffect::new("child", completion()), ChildOf(parent)))
        .id();
    let sibling = app
        .world_mut()
        .spawn((
            PendingEffect::new("external", completion()),
            ChildOf(external),
        ))
        .id();
    app.world_mut().entity_mut(parent).insert(Seq(7));
    app.world_mut().entity_mut(child).insert(Seq(2));
    app.world_mut().entity_mut(sibling).insert(Seq(2));
    let mut expected: Vec<_> = app
        .world_mut()
        .query::<(Entity, &Seq, &PendingEffect)>()
        .iter(app.world())
        .map(|(entity, seq, effect)| (entity, *seq, effect.key.clone()))
        .collect();
    expected.sort_by_key(|(_, seq, _)| *seq);
    let scene = Scene::save_with(app.world_mut(), |entity| (entity == external).then_some(42));
    assert_eq!(
        scene
            .effects
            .iter()
            .map(|effect| &effect.key)
            .collect::<Vec<_>>(),
        expected.iter().map(|(_, _, key)| key).collect::<Vec<_>>()
    );
    let parent_index = expected
        .iter()
        .position(|(entity, _, _)| *entity == parent)
        .unwrap();
    for (saved, (entity, _, _)) in scene.effects.iter().zip(&expected) {
        assert_eq!(saved.parent, (*entity == child).then_some(parent_index));
        assert_eq!(saved.parent_ref, (*entity == sibling).then_some(42));
    }
    let mut restored = bus_support::app();
    let external = restored.world_mut().spawn_empty().id();
    let loaded = scene
        .load_with(restored.world_mut(), |index| {
            (index == 42).then_some(external)
        })
        .unwrap();
    for (index, (entity, _, _)) in expected.iter().enumerate() {
        let actual = restored
            .world()
            .get::<ChildOf>(loaded[index])
            .map(ChildOf::parent);
        let wanted = if *entity == child {
            Some(loaded[parent_index])
        } else if *entity == sibling {
            Some(external)
        } else {
            None
        };
        assert_eq!(actual, wanted);
    }
}

#[cfg(feature = "replay")]
#[test]
fn replay_parent_lookup_keeps_first_loaded_match_and_ignores_forward_or_missing_ids() {
    use rig_core::{effect::EffectRecord, error::ErrorReport};
    use rig_ecs::bus::Replay;
    use rig_effect_log::EffectLog;

    let records = [
        (3, "last", 99),
        (1, "first", 3),
        (2, "child", 1),
        (1, "duplicate", 1),
    ]
    .into_iter()
    .map(|(id, key, parent)| EffectRecord {
        id: EffectId::from_raw(id),
        key: key.into(),
        kind: completion(),
        outcome: Err(ErrorReport::new(ErrorKind::Response, "recorded error")),
        parent: Some(EffectId::from_raw(parent)),
        scope: None,
        events: None,
        tool_output: None,
    })
    .collect();
    let mut app = bus_support::app();
    let entities = Replay::load(app.world_mut(), &EffectLog::from_records(records));
    assert_eq!(
        entities
            .iter()
            .map(|entity| app
                .world()
                .get::<PendingEffect>(*entity)
                .unwrap()
                .key
                .as_str())
            .collect::<Vec<_>>(),
        ["first", "duplicate", "child", "last"]
    );
    assert_eq!(
        entities
            .iter()
            .map(|entity| app.world().get::<ChildOf>(*entity).map(ChildOf::parent))
            .collect::<Vec<_>>(),
        [None, Some(entities[0]), Some(entities[0]), None]
    );
}
