//! Reconstruct and verify the actual replay world from serialized metadata.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic
)]

mod run_support;

use bevy_ecs::prelude::*;
use rig_core::{
    completion::{ModelRef, ProviderCapabilities},
    effect::{EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey},
    serve::{OutcomeSink, Serve},
};
use rig_ecs::{
    agent::{
        Failed, Grant, Order, Output, OutputKind, PolicyVersion, Settled, Temperature,
        scene::RunScene,
    },
    bus::{Bound, EffectLogResource, EffectOutcome, Handlers, PendingEffect, Replay},
    replay::{check_replayable, stamp_run},
    systems::spawn_run,
};
use rig_effect_log::{EffectLog, EffectLogRecorder};
use run_support::*;

const MODEL: &str = "t/model:default";
const TOOL: &str = "t/tool:unused";

struct Composing(Capturing);

impl Serve for Composing {
    type Family = rig_core::effect::family::Completion;
    fn descriptor(&self) -> HandlerDescriptor {
        let mut descriptor = self.0.descriptor();
        descriptor.family = FamilyDescriptor::Completion {
            model: ModelRef::new("composing"),
            capabilities: ProviderCapabilities::new().with_native_output_tool_composition(true),
        };
        descriptor
    }
    async fn serve(&self, kind: EffectKind, sink: OutcomeSink) {
        self.0.serve(kind, sink).await;
    }
}

fn program(world: &mut World, model: Entity, tool: Entity) -> Entity {
    let agent = spawn_agent(world, "t", model);
    world.entity_mut(agent).insert((PolicyVersion("metadata/v1".into()), Output {
        mode: OutputKind::Auto,
        schema: Some(serde_json::json!({"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]})),
    }));
    world.spawn((Grant(tool), Order(0), ChildOf(agent)));
    agent
}

fn bound(world: &mut World, key: &str) -> Entity {
    world
        .query::<(Entity, &Bound)>()
        .iter(world)
        .find(|(_, bound)| bound.key == HandlerKey::from(key))
        .unwrap()
        .0
}

#[test]
fn serialized_log_reconstructs_capabilities_identity_and_uncalled_grants() {
    let mut live = app();
    let recorder = EffectLogRecorder::new();
    EffectLogResource::install(live.world_mut(), recorder.clone());
    let (capturing, requests) = Capturing::new(MODEL, "{\"a\":1}");
    let model = register(&mut live, MODEL, Composing(capturing));
    let tool = register(
        &mut live,
        TOOL,
        NeverCalled {
            name: "unused".into(),
        },
    );
    let agent = program(live.world_mut(), model, tool);
    let run = spawn_run(live.world_mut(), agent, &[], "go", false, None);
    stamp_run(live.world_mut(), run, &recorder);
    tick_until(&mut live, "live settled", |world| {
        world.get::<Settled>(run).is_some()
    });
    assert!(requests.lock().unwrap()[0].output_schema.is_some());
    let log: EffectLog =
        serde_json::from_str(&serde_json::to_string(&recorder.log()).unwrap()).unwrap();
    assert!(log.header.required.is_empty());
    assert!(
        log.records
            .iter()
            .all(|record| record.key != HandlerKey::from(TOOL))
    );

    let mut replay = app();
    Handlers::with(replay.world_mut(), |handlers| {
        Replay::default().register(handlers, &log)
    })
    .unwrap()
    .unwrap();
    let model = bound(replay.world_mut(), MODEL);
    let tool = bound(replay.world_mut(), TOOL);
    let agent = program(replay.world_mut(), model, tool);
    let run = spawn_run(replay.world_mut(), agent, &[], "go", false, None);
    check_replayable(replay.world_mut(), run, &log).expect("same program in the replay world");

    let original = replay
        .world()
        .get::<Bound>(model)
        .unwrap()
        .descriptor
        .clone();
    for family in [
        FamilyDescriptor::Completion {
            model: ModelRef::new("different-model"),
            capabilities: ProviderCapabilities::new().with_native_output_tool_composition(true),
        },
        FamilyDescriptor::Completion {
            model: ModelRef::new("composing"),
            capabilities: ProviderCapabilities::default(),
        },
    ] {
        replay
            .world_mut()
            .get_mut::<Bound>(model)
            .unwrap()
            .descriptor
            .family = family;
        assert!(check_replayable(replay.world_mut(), run, &log).is_err());
    }
    replay
        .world_mut()
        .get_mut::<Bound>(model)
        .unwrap()
        .descriptor = original;
    replay
        .world_mut()
        .entity_mut(run)
        .insert(Temperature(Some(0.8)));
    assert!(check_replayable(replay.world_mut(), run, &log).is_err());
    replay.world_mut().entity_mut(run).remove::<Temperature>();
    replay
        .world_mut()
        .entity_mut(run)
        .insert(PolicyVersion("changed".into()));
    assert!(check_replayable(replay.world_mut(), run, &log).is_err());
    replay.world_mut().entity_mut(run).remove::<PolicyVersion>();
    replay
        .world_mut()
        .get_mut::<Bound>(model)
        .unwrap()
        .descriptor
        .layers
        .push("different-layer".into());
    assert!(check_replayable(replay.world_mut(), run, &log).is_err());
    replay
        .world_mut()
        .get_mut::<Bound>(model)
        .unwrap()
        .descriptor
        .layers
        .clear();
    let tool_descriptor = replay
        .world()
        .get::<Bound>(tool)
        .unwrap()
        .descriptor
        .clone();
    if let FamilyDescriptor::Tool { parameters, .. } = &mut replay
        .world_mut()
        .get_mut::<Bound>(tool)
        .unwrap()
        .descriptor
        .family
    {
        *parameters = serde_json::json!({"type": "object", "required": ["new_argument"]});
    }
    assert!(check_replayable(replay.world_mut(), run, &log).is_err());
    replay
        .world_mut()
        .get_mut::<Bound>(tool)
        .unwrap()
        .descriptor = tool_descriptor;
    check_replayable(replay.world_mut(), run, &log).unwrap();
    tick_until(&mut replay, "replay terminal", |world| {
        world.get::<Settled>(run).is_some() || world.get::<Failed>(run).is_some()
    });
    assert!(replay.world().get::<Failed>(run).is_none());
    assert!(replay.world().get::<Settled>(run).is_some());

    let scene = RunScene::save(live.world_mut()).unwrap();
    let scene: RunScene = serde_json::from_str(&serde_json::to_string(&scene).unwrap()).unwrap();
    let mut restored = app();
    Handlers::with(restored.world_mut(), |handlers| {
        Replay::default().register(handlers, &log)
    })
    .unwrap()
    .unwrap();
    scene
        .load(restored.world_mut())
        .expect("all scope dependencies were reconstructed");
    let unexpected = restored
        .world_mut()
        .spawn(PendingEffect::new(
            TOOL,
            EffectKind::ToolCall {
                name: "unused".into(),
                args: "{}".into(),
            },
        ))
        .id();
    tick_until(
        &mut restored,
        "uncalled required tool refuses unexpected request",
        |world| world.get::<EffectOutcome>(unexpected).is_some(),
    );
    let error = restored
        .world()
        .get::<EffectOutcome>(unexpected)
        .unwrap()
        .0
        .as_ref()
        .unwrap_err();
    assert_eq!(error.kind, rig_core::error::ErrorKind::Divergence);
    assert!(error.message.contains(TOOL));
}
