//! Effective configuration, explicit scope and declared custom policy identity.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
mod run_support;
use bevy_ecs::prelude::*;
use rig_ecs::{
    agent::{InvalidCalls, MaxTurns, PolicyVersion, Preamble, ToolPolicy, Unhandled},
    bus::{EffectLogResource, Scope},
    replay::{check_replayable, spec_hash, stamp_run},
    systems::spawn_run,
};
use rig_effect_log::{EffectLog, EffectLogRecorder};
use run_support::*;

fn setup() -> (bevy_app::App, Entity, Entity, EffectLog) {
    let mut app = app();
    let (model, _) = Capturing::new("t/model:default", "ok");
    let model = register(&mut app, "t/model:default", model);
    let agent = spawn_agent(app.world_mut(), "t", model);
    app.world_mut()
        .entity_mut(agent)
        .insert(PolicyVersion("test/v1".into()));
    let run = spawn_run(app.world_mut(), agent, &[], "go", false, None);
    let recorder = EffectLogRecorder::new();
    EffectLogResource::install(app.world_mut(), recorder.clone());
    stamp_run(app.world_mut(), run, &recorder);
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
fn run_overrides_win_and_mask_changes_to_agent_defaults() {
    let (mut app, agent, run, log) = setup();
    app.world_mut().entity_mut(run).insert((
        MaxTurns(7),
        Preamble(Some("run".into())),
        ToolPolicy { concurrency: 3 },
    ));
    assert!(check_replayable(app.world_mut(), run, &log).is_err());
    let before = spec_hash(app.world_mut(), run);
    app.world_mut().entity_mut(agent).insert((
        MaxTurns(9),
        Preamble(Some("different agent".into())),
        ToolPolicy { concurrency: 9 },
    ));
    assert_eq!(before, spec_hash(app.world_mut(), run));
}

#[test]
fn scope_selection_never_searches_for_another_matching_policy() {
    let (mut app, _, run, mut log) = setup();
    let scope = app.world().get::<Scope>(run).unwrap().0.clone();
    let own = log.header.programs.get(&scope).unwrap().clone();
    let mut other = own.clone();
    other
        .required
        .insert("tool:foreign".into(), rig_core::effect::EffectFamily::Tool);
    log.header.programs.insert("aaa/other".into(), other);
    check_replayable(app.world_mut(), run, &log).unwrap();
    log.header.programs.remove(&scope);
    assert!(
        check_replayable(app.world_mut(), run, &log)
            .unwrap_err()
            .message
            .contains(&scope)
    );
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
    stamp_run(app.world_mut(), run, &recorder);
    assert!(
        check_replayable(app.world_mut(), run, &recorder.log())
            .unwrap_err()
            .message
            .contains("unverified")
    );
}
