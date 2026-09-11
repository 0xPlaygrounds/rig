//! Cancellation must replay at the private original-answer observation boundary.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "contract assertions"
)]
#[path = "../../../tests/bus_support/mod.rs"]
mod bus_support;
use crate::bus::{
    BusSet, EffectLogResource, EffectOutcome, Handlers, InFlight, PendingEffect, Replay,
    RigSchedule, Streamed,
};
use bevy_ecs::prelude::*;
use rig_core::{
    effect::{EffectKind, FamilyDescriptor, HandlerDescriptor, Outcome},
    serve::{Dispatch, Reply, Serve},
};
use rig_effect_log::EffectLogRecorder;
use std::sync::{Arc, atomic::Ordering};

struct AwaitVerdict(Arc<std::sync::atomic::AtomicBool>);
impl rig_core::serve::Intercept for AwaitVerdict {
    fn name(&self) -> String {
        "await-verdict".into()
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
        _: &Result<Outcome, rig_core::error::ErrorReport>,
    ) -> rig_core::serve::Verdict {
        self.0.store(true, Ordering::SeqCst);
        std::future::pending().await
    }
}

#[test]
fn cancellation_after_an_original_answer_has_a_replayable_visibility_trace() {
    struct Truncated;
    impl Serve for Truncated {
        type Family = rig_core::effect::family::Completion;
        fn descriptor(&self) -> HandlerDescriptor {
            HandlerDescriptor {
                key: "model".into(),
                family: FamilyDescriptor::Completion {
                    model: "truncated".into(),
                    capabilities: Default::default(),
                },
                layers: vec![],
            }
        }
        async fn serve(&self, _: EffectKind, _: Dispatch) -> Reply {
            Reply::Stream(Box::pin(futures::stream::empty()))
        }
    }
    for (streaming, truncated) in [(false, false), (true, false), (true, true)] {
        let mut app = bus_support::app();
        let recorder = EffectLogRecorder::keeping_stream_events();
        EffectLogResource::install(app.world_mut(), recorder.clone());
        let entered = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let counters = Arc::new(bus_support::Counters::default());
        let handler = if truncated {
            rig_core::serve::ErasedHandler::new(Truncated)
        } else {
            rig_core::serve::ErasedHandler::new(bus_support::MockModel {
                cap: 1,
                ..bus_support::MockModel::new(&counters)
            })
        }
        .layered(AwaitVerdict(entered.clone()));
        bus_support::register(&mut app, "model", handler);
        let effect = app
            .world_mut()
            .spawn(PendingEffect::new(
                "model",
                EffectKind::Completion {
                    request: bus_support::request(),
                    stream: streaming,
                },
            ))
            .id();
        bus_support::tick_until(&mut app, "verdict suspended", |_| {
            entered.load(Ordering::SeqCst)
        });
        assert!(app.world().get::<EffectOutcome>(effect).is_none());
        app.world_mut().despawn(effect);
        let log = recorder.log();
        assert!(
            if truncated {
                log.records[0]
                    .outcome
                    .as_ref()
                    .is_err_and(|error| error == &rig_core::serve::stream_truncated())
            } else {
                log.records[0].outcome.is_ok()
            },
            "cancellation must preserve the observed inner answer"
        );
        assert!(
            !log.header
                .deliveries
                .as_ref()
                .unwrap()
                .iter()
                .any(|delivery| matches!(delivery.kind, rig_core::effect::DeliveryKind::Outcome)),
            "no outcome reached the consumer"
        );
        rig_ecs::bus::delivery::ReplayDelivery::new(&log, true)
            .expect("a library-generated cancellation trace must remain replayable");
        assert!(matches!(
            log.header.deliveries.as_ref().unwrap().last().unwrap().kind,
            rig_core::effect::DeliveryKind::Cancelled
        ));
        for policy in [false, true] {
            let mut replay = bus_support::app();
            Handlers::with(replay.world_mut(), |handlers| {
                let mode = if policy {
                    Replay::policy_visible()
                } else {
                    Replay::default()
                };
                mode.register(handlers, &log)
            })
            .unwrap()
            .unwrap();
            let loaded = Replay::load(replay.world_mut(), &log)[0];
            bus_support::tick_until(
                &mut replay,
                "cancelled replay settled or refused",
                |world| world.get::<EffectOutcome>(loaded).is_some(),
            );
            let error = replay
                .world()
                .get::<EffectOutcome>(loaded)
                .unwrap()
                .0
                .as_ref()
                .unwrap_err();
            assert_eq!(
                error.kind,
                if policy {
                    rig_core::error::ErrorKind::Divergence
                } else {
                    rig_core::error::ErrorKind::Cancelled
                }
            );
            if policy {
                assert!(error.message.contains("did not reproduce cancellation"));
            }
            let delivered = replay.world().get::<Streamed>(loaded);
            assert!(
                delivered.is_none_or(|stream| stream.outcome.is_none()),
                "the original terminal was never delivered"
            );
        }
        let mut reproduced = bus_support::app();
        let rerecorder = EffectLogRecorder::keeping_stream_events();
        EffectLogResource::install(reproduced.world_mut(), rerecorder.clone());
        Handlers::with(reproduced.world_mut(), |handlers| {
            Replay::policy_visible().register(handlers, &log)
        })
        .unwrap()
        .unwrap();
        reproduced.world_mut().resource_mut::<Schedules>().add_systems(RigSchedule,
            (|mut commands: Commands, effects: Query<(Entity, &rig_ecs::bus::record::Observed), With<InFlight>>| {
                for (entity, observed) in &effects {
                    if observed.0.has_outcome() { commands.entity(entity).despawn(); }
                }
            }).in_set(BusSet::Judge));
        let loaded = Replay::load(reproduced.world_mut(), &log)[0];
        bus_support::tick_until(&mut reproduced, "policy reproduced cancellation", |world| {
            world.get_entity(loaded).is_err()
        });
        assert!(
            !reproduced
                .world()
                .contains_resource::<rig_ecs::bus::ReplayFailure>()
        );
        let rerecorded = rerecorder.log();
        assert_eq!(
            serde_json::to_value(&rerecorded.records[0].outcome).unwrap(),
            serde_json::to_value(&log.records[0].outcome).unwrap()
        );
        assert_eq!(
            serde_json::to_value(&rerecorded.records[0].events).unwrap(),
            serde_json::to_value(&log.records[0].events).unwrap()
        );
        assert_eq!(rerecorded.header.stream_errors, log.header.stream_errors);
        assert!(matches!(
            rerecorded
                .header
                .deliveries
                .as_ref()
                .unwrap()
                .last()
                .unwrap()
                .kind,
            rig_core::effect::DeliveryKind::Cancelled
        ));
        rig_ecs::bus::delivery::ReplayDelivery::new(&rerecorded, true)
            .expect("reproduced cancellation remains replayable");

        let mut invalid = log.clone();
        invalid
            .header
            .deliveries
            .as_mut()
            .unwrap()
            .push(rig_core::effect::Delivery {
                batch: log
                    .header
                    .deliveries
                    .as_ref()
                    .unwrap()
                    .last()
                    .unwrap()
                    .batch,
                id: log.records[0].id,
                kind: rig_core::effect::DeliveryKind::Outcome,
            });
        assert!(
            rig_ecs::bus::delivery::ReplayDelivery::new(&invalid, true).is_err(),
            "cancellation closes the delivery trace"
        );
    }
}
