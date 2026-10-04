//! Effective configuration, explicit scope and declared custom policy identity.
use crate::run_support;
use bevy_ecs::prelude::*;
use rig_cassette::ecs::EffectLogResource;
use rig_cassette::ecs::identity::{check_replayable, stamp_run};
use rig_cassette::effect_log::{EffectLog, EffectLogRecorder};
use rig_ecs::{
    agent::{InvalidCalls, PolicyVersion, Unhandled},
    systems::RunCommands,
};
use run_support::*;

fn setup() -> (bevy_app::App, Entity, Entity, EffectLog) {
    let mut app = app();
    let (model, _) = Capturing::new("t/model:default", "ok");
    let model = register(&mut app, "t/model:default", model);
    let agent = spawn_agent(app.world_mut(), "t", model);
    app.world_mut()
        .entity_mut(agent)
        .insert(PolicyVersion("test/v1".into()));
    let run = app.world_mut().spawn_run(agent, &[], "go", false, None);
    let recorder = EffectLogRecorder::new();
    EffectLogResource::install(app.world_mut(), recorder.clone());
    stamp_run(app.world_mut(), run, &recorder).expect("the run stamps its program identity");
    (app, agent, run, recorder.log())
}

#[test]
fn retries_and_unhandled_policy_each_change_identity() {
    for policy in [
        InvalidCalls {
            retries: 2,
            unhandled: Unhandled::Fail,
        },
        InvalidCalls {
            retries: 0,
            unhandled: Unhandled::Ignore,
        },
    ] {
        for on_run in [false, true] {
            let (mut app, agent, run, log) = setup();
            check_replayable(app.world_mut(), run, &log).unwrap();
            app.world_mut()
                .entity_mut(if on_run { run } else { agent })
                .insert(policy);
            assert!(
                check_replayable(app.world_mut(), run, &log)
                    .unwrap_err()
                    .message
                    .contains("policy")
            );
        }
    }
}

#[test]
fn custom_policy_requires_an_explicit_version_and_detects_changes() {
    let (mut app, agent, run, log) = setup();
    app.world_mut()
        .entity_mut(agent)
        .insert(PolicyVersion("test/v2".into()));
    assert!(check_replayable(app.world_mut(), run, &log).is_err());
    app.world_mut().entity_mut(agent).remove::<PolicyVersion>();
    let recorder = EffectLogRecorder::new();
    EffectLogResource::install(app.world_mut(), recorder.clone());
    stamp_run(app.world_mut(), run, &recorder).expect("the run stamps its program identity");
    assert!(
        check_replayable(app.world_mut(), run, &recorder.log())
            .unwrap_err()
            .message
            .contains("unverified")
    );
}
