//! The agent runtime's facts reach the witness: endings, cancellation
//! requests, invalid-call resolutions and the tool batch's holds, each with
//! the run's scope as its subject.

use crate::run_support;

use std::sync::Arc;

use rig_core::observe::{Action, ObservationLog};
use rig_ecs::{
    agent::{Cancelled, Failed, Settled},
    bus::Witnessing,
    systems::RunCommands,
};
use run_support::*;

fn witnessed(app: &mut bevy_app::App) -> Arc<ObservationLog> {
    let log = Arc::new(ObservationLog::default());
    Witnessing::install(app.world_mut(), log.clone());
    log
}

fn endings(log: &ObservationLog) -> Vec<(Option<String>, String)> {
    log.trace()
        .observations
        .iter()
        .filter_map(|o| match &o.action {
            Action::Ended { ending } => Some((o.subject.scope.clone(), ending.code.clone())),
            _ => None,
        })
        .collect()
}

#[derive(Default)]
struct RunClock(std::sync::atomic::AtomicU64);

impl rig_core::observe::Clock for RunClock {
    fn elapsed(&self) -> std::time::Duration {
        std::time::Duration::from_millis(self.0.load(std::sync::atomic::Ordering::SeqCst))
    }
}

#[test]
fn run_endings_cover_settlement_failure_cancellation_and_unfinished_removal() {
    for ending in [
        "settled",
        "max_turns",
        "cancelled",
        "despawned_without_ending",
    ] {
        let mut baseline = None;
        for timed in [false, true] {
            let mut app = app();
            let clock = Arc::new(RunClock(std::sync::atomic::AtomicU64::new(10)));
            let log = Arc::new(if timed {
                ObservationLog::default().with_clock(clock.clone())
            } else {
                ObservationLog::default()
            });
            Witnessing::install(app.world_mut(), log.clone());
            let (model, _) = Capturing::new("m", "fine");
            let model = register(&mut app, "m", model);
            let agent = spawn_agent(app.world_mut(), "app", model);
            let run = app.world_mut().spawn_run(agent, &[], "hi", false, Some(1));
            clock.0.store(60, std::sync::atomic::Ordering::SeqCst);
            match ending {
                "settled" => tick_until(&mut app, "settled", |world| {
                    world.get::<Settled>(run).is_some()
                }),
                "max_turns" => {
                    app.world_mut()
                        .entity_mut(run)
                        .insert(Failed(rig_ecs::agent::Failure::MaxTurns { limit: 1 }));
                }
                "cancelled" => {
                    app.world_mut()
                        .entity_mut(run)
                        .insert(Cancelled("stop".into()));
                    tick_until(&mut app, "cancelled", |world| {
                        world.get::<Failed>(run).is_some()
                    });
                }
                _ => {
                    app.world_mut().despawn(run);
                }
            }
            let trace = log.trace();
            let ending_facts: Vec<_> = trace
                .observations
                .iter()
                .filter(|o| matches!(o.action, Action::Ended { .. }))
                .collect();
            assert_eq!(ending_facts.len(), 1);
            let fact = ending_facts[0];
            assert!(
                matches!(&fact.action, Action::Ended { ending: reason } if reason.code == ending)
            );
            if timed {
                assert_eq!(
                    rig_core::test_utils::observations::compare(baseline.as_ref().unwrap(), &trace),
                    rig_core::test_utils::observations::Comparison::Equal
                );
            } else {
                baseline = Some(trace.clone());
            }
            // Removing an already ended run cannot add a second closure.
            if app.world().get_entity(run).is_ok() {
                app.world_mut().despawn(run);
            }
            assert_eq!(endings(&log).len(), 1);
        }
    }
}

#[test]
fn an_invalid_tool_call_ends_with_its_failure_reason() {
    let mut app = app();
    let log = witnessed(&mut app);
    let (model, _) = Scripted::new(
        "s",
        vec![vec![call("c1", "no_such_tool", serde_json::json!({}))]],
    );
    let model = register(&mut app, "s", model);
    let agent = spawn_agent(app.world_mut(), "app", model);
    let run = app.world_mut().spawn_run(agent, &[], "go", false, Some(2));
    tick_until(&mut app, "failed", |world| {
        world.get::<Failed>(run).is_some()
    });
    app.update();

    assert_eq!(
        endings(&log).last().map(|(_, code)| code.as_str()),
        Some("unknown_tool_call")
    );
}
