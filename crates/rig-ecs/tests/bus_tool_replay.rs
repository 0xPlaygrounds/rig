//! Durable published results are observable output, not ambient inputs.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

mod bus_support;

use bevy_ecs::prelude::*;
use bus_support::*;
use rig_core::{
    effect::{EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey, Outcome},
    error::{ErrorKind, ErrorReport},
    serve::{OutcomeSink, Serve},
    tool::{ContextValue, PublishedContext, ToolContext, ToolOutput, ToolResult},
};
use rig_ecs::bus::{
    EffectLogResource, EffectOutcome, Handlers, PendingEffect, Replay, ToolInputs, ToolOutputs,
};

#[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
struct Artifact(String);
impl ContextValue for Artifact {
    const KEY: &'static str = "test.artifact.v1";
}
#[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
struct Secret(String);
impl ContextValue for Secret {
    const KEY: &'static str = "test.secret";
}

struct Publish {
    fail: bool,
}
impl Serve for Publish {
    type Family = rig_core::effect::family::Tool;
    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("tool:publish"),
            family: FamilyDescriptor::Tool {
                name: "publish".into(),
                description: "publishes an artifact".into(),
                parameters: serde_json::json!({"type":"object"}),
                embedding: None,
            },
            layers: vec![],
        }
    }
    async fn serve(&self, _: EffectKind, sink: OutcomeSink) {
        let mut context = sink.scope::<ToolContext>().unwrap().for_dispatch();
        context
            .insert_result(Artifact("artifact-123".into()))
            .unwrap();
        sink.scope::<PublishedContext>().unwrap().publish(context);
        let answer = if self.fail {
            Err(ErrorReport::new(ErrorKind::Request, "tool refused"))
        } else {
            Ok(Outcome::ToolResult {
                result: ToolResult::success(ToolOutput::text("ok")),
            })
        };
        sink.resolve(answer).await;
    }
}

fn dispatch(world: &mut World, secret: &str) -> Entity {
    let mut context = ToolContext::new();
    context.insert(Secret(secret.into())).unwrap();
    // Opaque runtime capabilities must never be serialized by the recorder.
    context = context.with_scope(std::sync::Arc::new(std::sync::Mutex::new(7_u32)));
    world
        .spawn((
            PendingEffect::new(
                "tool:publish",
                EffectKind::ToolCall {
                    name: "publish".into(),
                    args: "{}".into(),
                },
            ),
            ToolInputs(context),
        ))
        .id()
}

#[test]
fn serialized_replay_preserves_results_on_success_and_error_without_recording_credentials() {
    for fail in [false, true] {
        let mut live = app();
        EffectLogResource::install(live.world_mut(), rig_effect_log::EffectLogRecorder::new());
        register(&mut live, "tool:publish", Publish { fail });
        let effect = dispatch(live.world_mut(), "recording-secret-do-not-log");
        tick_until(&mut live, "live publication", |w| {
            w.get::<EffectOutcome>(effect).is_some()
        });
        let answer = live.world().get::<EffectOutcome>(effect).unwrap().0.clone();
        let output = live
            .world()
            .get::<ToolOutputs>(effect)
            .unwrap()
            .0
            .result::<Artifact>()
            .unwrap();
        let json =
            serde_json::to_string(&live.world().resource::<EffectLogResource>().log()).unwrap();
        assert!(json.contains("artifact-123"));
        assert!(!json.contains("recording-secret-do-not-log"));
        assert!(!json.contains("test.secret"));
        let log = serde_json::from_str(&json).unwrap();
        let mut replay = app();
        Handlers::with(replay.world_mut(), |handlers| {
            Replay::default().register(handlers, &log)
        })
        .unwrap()
        .unwrap();
        let effect = dispatch(replay.world_mut(), "current-secret");
        tick_until(&mut replay, "replayed publication", |w| {
            w.get::<EffectOutcome>(effect).is_some()
        });
        assert_eq!(
            serde_json::to_value(&replay.world().get::<EffectOutcome>(effect).unwrap().0).unwrap(),
            serde_json::to_value(answer).unwrap()
        );
        let context = &replay.world().get::<ToolOutputs>(effect).unwrap().0;
        assert_eq!(context.result::<Artifact>().unwrap(), output);
        assert_eq!(
            context.get::<Secret>().unwrap(),
            Some(Secret("current-secret".into()))
        );
        // A downstream policy consumes the metadata, not just its wire bytes.
        let artifact = context.require_result::<Artifact>().unwrap();
        replay
            .world_mut()
            .insert_resource(SelectedArtifact(artifact.0));
        assert_eq!(
            replay.world().resource::<SelectedArtifact>().0,
            "artifact-123"
        );
    }
}

#[derive(Resource)]
struct SelectedArtifact(String);

struct ReplaceAnswer;
impl rig_core::serve::Intercept for ReplaceAnswer {
    fn name(&self) -> String {
        "replace-answer".into()
    }
    async fn before(
        &self,
        _: rig_core::effect::EffectId,
        _: &EffectKind,
    ) -> rig_core::serve::Decision {
        rig_core::serve::Decision::Proceed
    }
    async fn after(
        &self,
        _: rig_core::effect::EffectId,
        _: &EffectKind,
        _: &Result<Outcome, ErrorReport>,
    ) -> rig_core::serve::Verdict {
        rig_core::serve::Verdict::Replace(Err(ErrorReport::new(
            ErrorKind::Timeout,
            "outer verdict",
        )))
    }
}

#[test]
fn layered_verdict_keeps_nonempty_inner_output_in_the_record_and_replay() {
    let mut live = app();
    EffectLogResource::install(live.world_mut(), rig_effect_log::EffectLogRecorder::new());
    Handlers::with(live.world_mut(), |handlers| {
        handlers.register_erased(
            "tool:publish",
            rig_core::serve::ErasedHandler::new(Publish { fail: false }).layered(ReplaceAnswer),
        )
    })
    .unwrap()
    .unwrap();
    let effect = dispatch(live.world_mut(), "private");
    tick_until(&mut live, "layered answer", |world| {
        world.get::<EffectOutcome>(effect).is_some()
    });
    assert_eq!(
        live.world()
            .get::<EffectOutcome>(effect)
            .unwrap()
            .0
            .as_ref()
            .unwrap_err()
            .kind,
        ErrorKind::Timeout
    );
    assert_eq!(
        live.world()
            .get::<ToolOutputs>(effect)
            .unwrap()
            .0
            .result::<Artifact>()
            .unwrap(),
        Some(Artifact("artifact-123".into()))
    );
    let json = serde_json::to_string(&live.world().resource::<EffectLogResource>().log()).unwrap();
    let log: rig_effect_log::EffectLog = serde_json::from_str(&json).unwrap();
    assert!(
        log.records[0].outcome.is_ok(),
        "record is the inner handler, not the verdict"
    );
    assert_eq!(
        log.records[0]
            .tool_output
            .as_ref()
            .unwrap()
            .get::<Artifact>()
            .unwrap(),
        Some(Artifact("artifact-123".into()))
    );
    let mut replay = app();
    let replayer =
        rig_effect_log::EffectLogReplayer::for_key(&log, &HandlerKey::from("tool:publish"))
            .unwrap();
    Handlers::with(replay.world_mut(), |handlers| {
        handlers.register_erased(
            "tool:publish",
            rig_core::serve::ErasedHandler::new(replayer).layered(ReplaceAnswer),
        )
    })
    .unwrap()
    .unwrap();
    let effect = dispatch(replay.world_mut(), "new-private");
    tick_until(&mut replay, "layered replay", |world| {
        world.get::<EffectOutcome>(effect).is_some()
    });
    assert_eq!(
        replay
            .world()
            .get::<EffectOutcome>(effect)
            .unwrap()
            .0
            .as_ref()
            .unwrap_err()
            .kind,
        ErrorKind::Timeout
    );
    assert_eq!(
        replay
            .world()
            .get::<ToolOutputs>(effect)
            .unwrap()
            .0
            .result::<Artifact>()
            .unwrap(),
        Some(Artifact("artifact-123".into()))
    );
}

struct PublishThenWait;
impl Serve for PublishThenWait {
    type Family = rig_core::effect::family::Tool;
    fn descriptor(&self) -> HandlerDescriptor {
        Publish { fail: false }.descriptor()
    }
    async fn serve(&self, _: EffectKind, sink: OutcomeSink) {
        let mut context = ToolContext::new();
        context
            .insert_result(Artifact("before-cancel".into()))
            .unwrap();
        sink.scope::<PublishedContext>().unwrap().publish(context);
        std::future::pending::<()>().await;
        drop(sink);
    }
}

#[test]
fn cancellation_records_already_published_output() {
    let mut live = app();
    EffectLogResource::install(live.world_mut(), rig_effect_log::EffectLogRecorder::new());
    register(&mut live, "tool:publish", PublishThenWait);
    let effect = dispatch(live.world_mut(), "private");
    tick_until(&mut live, "published before cancellation", |world| {
        world
            .get::<rig_ecs::bus::Publishing>(effect)
            .is_some_and(|published| published.0.result_context().is_some())
    });
    live.world_mut().despawn(effect);
    let log = live.world().resource::<EffectLogResource>().log();
    assert_eq!(log.records.len(), 1);
    assert_eq!(
        log.records[0].outcome.as_ref().unwrap_err().kind,
        ErrorKind::Cancelled
    );
    assert_eq!(
        log.records[0]
            .tool_output
            .as_ref()
            .unwrap()
            .get::<Artifact>()
            .unwrap(),
        Some(Artifact("before-cancel".into()))
    );
    let mut replay = app();
    Handlers::with(replay.world_mut(), |handlers| {
        Replay::default().register(handlers, &log)
    })
    .unwrap()
    .unwrap();
    let effect = dispatch(replay.world_mut(), "new-private");
    tick_until(&mut replay, "cancelled output replay", |world| {
        world.get::<EffectOutcome>(effect).is_some()
    });
    assert_eq!(
        replay
            .world()
            .get::<EffectOutcome>(effect)
            .unwrap()
            .0
            .as_ref()
            .unwrap_err()
            .kind,
        ErrorKind::Cancelled
    );
    assert_eq!(
        replay
            .world()
            .get::<ToolOutputs>(effect)
            .unwrap()
            .0
            .result::<Artifact>()
            .unwrap(),
        Some(Artifact("before-cancel".into()))
    );
}

#[test]
fn world_native_cancellation_records_output_without_a_publishing_slot() {
    let mut live = app();
    EffectLogResource::install(live.world_mut(), rig_effect_log::EffectLogRecorder::new());
    Handlers::with(live.world_mut(), |handlers| {
        handlers.register_open("tool:publish", Publish { fail: false }.descriptor().family)
    })
    .unwrap()
    .unwrap();
    let effect = dispatch(live.world_mut(), "private");
    tick_until(&mut live, "world dispatch", |world| {
        world.get::<rig_ecs::bus::InFlight>(effect).is_some()
    });
    assert!(
        live.world()
            .get::<rig_ecs::bus::Publishing>(effect)
            .is_none()
    );
    let mut context = ToolContext::new();
    context
        .insert_result(Artifact("world-before-cancel".into()))
        .unwrap();
    live.world_mut()
        .entity_mut(effect)
        .insert(ToolOutputs(context));
    live.world_mut().despawn(effect);
    let log = live.world().resource::<EffectLogResource>().log();
    assert_eq!(log.records.len(), 1);
    assert_eq!(
        log.records[0].outcome.as_ref().unwrap_err().kind,
        ErrorKind::Cancelled
    );
    assert_eq!(
        log.records[0]
            .tool_output
            .as_ref()
            .unwrap()
            .get::<Artifact>()
            .unwrap(),
        Some(Artifact("world-before-cancel".into()))
    );
}

#[test]
fn world_native_published_output_is_recorded_and_mismatched_replay_does_not_publish() {
    let mut live = app();
    EffectLogResource::install(live.world_mut(), rig_effect_log::EffectLogRecorder::new());
    Handlers::with(live.world_mut(), |handlers| {
        handlers.register_open("tool:publish", Publish { fail: false }.descriptor().family)
    })
    .unwrap()
    .unwrap();
    let effect = dispatch(live.world_mut(), "private");
    tick_until(&mut live, "world dispatch", |world| {
        world.get::<rig_ecs::bus::InFlight>(effect).is_some()
    });
    let mut output = ToolContext::new();
    output
        .insert_result(Artifact("world-result".into()))
        .unwrap();
    live.world_mut().entity_mut(effect).insert((
        ToolOutputs(output),
        EffectOutcome(Ok(Outcome::ToolResult {
            result: ToolResult::success(ToolOutput::text("world-answer")),
        })),
    ));
    live.update();
    let log = live.world().resource::<EffectLogResource>().log();
    assert_eq!(
        log.records[0]
            .tool_output
            .as_ref()
            .unwrap()
            .get::<Artifact>()
            .unwrap(),
        Some(Artifact("world-result".into()))
    );

    for matches in [true, false] {
        let mut replay = app();
        Handlers::with(replay.world_mut(), |handlers| {
            Replay::default().register(handlers, &log)
        })
        .unwrap()
        .unwrap();
        let effect = dispatch(replay.world_mut(), "new-private");
        if !matches {
            replay
                .world_mut()
                .entity_mut(effect)
                .insert(PendingEffect::new(
                    "tool:publish",
                    EffectKind::ToolCall {
                        name: "publish".into(),
                        args: "{\"different\":true}".into(),
                    },
                ));
        }
        tick_until(&mut replay, "world output replay", |world| {
            world.get::<EffectOutcome>(effect).is_some()
        });
        if matches {
            assert_eq!(
                replay
                    .world()
                    .get::<ToolOutputs>(effect)
                    .unwrap()
                    .0
                    .result::<Artifact>()
                    .unwrap(),
                Some(Artifact("world-result".into()))
            );
        } else {
            assert!(
                replay
                    .world()
                    .get::<EffectOutcome>(effect)
                    .unwrap()
                    .0
                    .is_err()
            );
            assert!(replay.world().get::<ToolOutputs>(effect).is_none());
        }
    }
}
