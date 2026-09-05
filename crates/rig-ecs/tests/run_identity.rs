//! Identity as data (CONTRACT §10): a world's log names each program by
//! its run's scope, and `replay::check_replayable` refuses a foreign log
//! by policy or by row before any dispatch.
//!
//! | claim | test |
//! |---|---|
//! | `stamp_run` writes `programs[scope] = { required, policy }`; rig-agent's goldens carry none | `a_worlds_log_names_its_program_by_scope` |
//! | a golden of another program is refused by policy, one with another row by the row's diff, one whose handlers do not serve the row by the gap | `check_replayable_refuses_a_foreign_log_by_name` |

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing
)]

mod run_support;

use bevy_ecs::prelude::*;
use rig_core::effect::{EffectFamily, HandlerKey};
use rig_ecs::{
    agent::{Grant, Order, PolicyVersion, Preamble},
    bus::{EffectLogResource, Scope},
    replay::{check_replayable, required_row, spec_hash, stamp_header, stamp_run},
    systems::spawn_run,
};
use rig_effect_log::{EffectLog, EffectLogRecorder};
use run_support::*;

const MODEL: &str = "t/model:default";
const ADD: &str = "t/tool:add#0";

fn golden(name: &str) -> EffectLog {
    let path = format!(
        "{}/../rig-verify/fixtures/{name}.effects.json",
        env!("CARGO_MANIFEST_DIR")
    );
    serde_json::from_str(&std::fs::read_to_string(path).expect("committed")).expect("loads")
}

#[test]
fn a_worlds_log_names_its_program_by_scope() {
    let mut app = app();
    EffectLogResource::install(app.world_mut(), EffectLogRecorder::new());
    let (model, _) = Capturing::new(MODEL, "ok");
    let model = register(&mut app, MODEL, model);
    let agent = spawn_agent(app.world_mut(), "t", model);
    let run = spawn_run(app.world_mut(), agent, &[], "go", false, None);
    let recorder = app.world().resource::<EffectLogResource>().0.clone();
    stamp_header(app.world_mut(), agent, &recorder, None, Vec::new());
    stamp_run(app.world_mut(), run, &recorder);
    let scope = app.world().get::<Scope>(run).expect("scoped").0.clone();
    let header = recorder.header();
    let identity = header.programs.get(&scope).expect("the run's identity");
    assert_eq!(Some(identity.policy), spec_hash(app.world_mut(), run));
    assert_eq!(identity.required, required_row(app.world_mut(), agent));
    assert_ne!(
        header.run_spec,
        Some(identity.policy),
        "builder identity is not effective run identity"
    );
    // rig-agent's goldens carry no `programs`.
    assert!(
        golden("anthropic_completion_smoke")
            .header
            .programs
            .is_empty()
    );
}

#[test]
fn check_replayable_refuses_a_foreign_log_by_name() {
    let mut app = app();
    let (model, _) = Capturing::new("golden/model:default", "ok");
    let model = register(&mut app, "golden/model:default", model);
    let agent = spawn_agent(app.world_mut(), "golden", model);
    app.world_mut().entity_mut(agent).insert(Preamble(Some(
        "You are a concise assistant. Answer directly.".to_owned(),
    )));
    let run = spawn_run(app.world_mut(), agent, &[], "go", false, None);
    app.world_mut()
        .entity_mut(agent)
        .insert(PolicyVersion("identity-test/v1".into()));
    let legacy = golden("anthropic_completion_smoke");
    assert!(
        check_replayable(app.world_mut(), run, &legacy)
            .expect_err("builder-only identity")
            .message
            .contains("unverified")
    );
    let recorder = EffectLogRecorder::new();
    EffectLogResource::install(app.world_mut(), recorder.clone());
    stamp_run(app.world_mut(), run, &recorder);
    let smoke = recorder.log();
    check_replayable(app.world_mut(), run, &smoke).expect("the run's own program");

    // Another policy: a different preamble.
    let other = app.world_mut();
    other
        .entity_mut(agent)
        .insert(Preamble(Some("Another program.".to_owned())));
    let refused = check_replayable(other, run, &smoke).expect_err("another policy");
    assert!(
        refused
            .message
            .starts_with("replay refused: the log was recorded under policy"),
        "{}",
        refused.message
    );
    app.world_mut().entity_mut(agent).insert(Preamble(Some(
        "You are a concise assistant. Answer directly.".to_owned(),
    )));

    // Another row: a tool granted the smoke never had.
    let tool = register(
        &mut app,
        ADD,
        NeverCalled {
            name: ADD.to_owned(),
        },
    );
    let grant = app
        .world_mut()
        .spawn((Grant(tool), Order(0), ChildOf(agent)))
        .id();
    let mut wrong_row = smoke.clone();
    let scope = app.world().get::<Scope>(run).unwrap().0.clone();
    wrong_row.header.programs.get_mut(&scope).unwrap().policy =
        spec_hash(app.world_mut(), run).unwrap();
    let refused = check_replayable(app.world_mut(), run, &wrong_row).expect_err("another row");
    assert!(
        refused
            .message
            .contains(&format!("`{ADD}` (tool_call) is missing")),
        "{}",
        refused.message
    );
    app.world_mut().despawn(grant);

    // A log whose programs name another policy: refused by the scopes it names.
    let mut foreign = smoke.clone();
    foreign.header.programs.clear();
    foreign.header.programs.insert(
        "other/run#0".to_owned(),
        rig_effect_log::ProgramIdentity {
            required: foreign.header.required.clone(),
            policy: 1,
        },
    );
    let refused = check_replayable(app.world_mut(), run, &foreign).expect_err("no such program");
    assert!(refused.message.contains(&scope), "{}", refused.message);

    // The row served: the log's handlers must serve every key of the row.
    let mut unserved = smoke.clone();
    unserved.header.handlers.clear();
    let refused = check_replayable(app.world_mut(), run, &unserved).expect_err("unserved");
    assert!(
        refused.kind == rig_core::error::ErrorKind::HandlerUnavailable
            && refused.message.contains("golden/model:default")
            && refused.message.contains("descriptor is missing"),
        "{}",
        refused.message
    );
    let _ = (HandlerKey::from(MODEL), EffectFamily::Completion);
}
