//! `agent::fork` (design §3.4): a run forked n − 1 times is n runs on one
//! agent, each with the prompt and its own number and scope, each
//! settling to its own answer.
//!
//! | claim | test |
//! |---|---|
//! | the forks share the prompt, take the next run numbers and scopes, and each settles | `a_forked_run_settles_beside_the_original` |

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing
)]

mod run_support;

use bevy_ecs::prelude::*;
use rig_core::message::AssistantContent;
use rig_ecs::{
    agent::{Run, RunResult, RunSeq, Runs, Settled, Utterance, fork},
    bus::Scope,
    systems::spawn_run,
};
use run_support::*;

#[test]
fn a_forked_run_settles_beside_the_original() {
    let mut app = app();
    let (model, _) = Scripted::new(
        "t/model:default",
        vec![
            vec![AssistantContent::text("one")],
            vec![AssistantContent::text("two")],
            vec![AssistantContent::text("three")],
        ],
    );
    let model = register(&mut app, "t/model:default", model);
    let agent = spawn_agent(app.world_mut(), "t", model);
    let run = spawn_run(app.world_mut(), agent, &[], "go", false, None);
    let second = fork(app.world_mut(), run);
    let third = fork(app.world_mut(), run);
    let world = app.world_mut();
    assert_eq!(world.get::<RunSeq>(second).map(|s| s.0), Some(1));
    assert_eq!(world.get::<RunSeq>(third).map(|s| s.0), Some(2));
    assert_eq!(
        world.get::<Scope>(third).map(|s| s.0.as_str()),
        Some("t/run#2")
    );
    assert_eq!(world.get::<Runs>(agent).map(|r| r.len()), Some(3));
    // Each fork has its own prompt utterance.
    let prompts = world
        .query_filtered::<&ChildOf, With<Utterance>>()
        .iter(world)
        .filter(|child_of| child_of.parent() == third)
        .count();
    assert_eq!(prompts, 1);
    tick_until(&mut app, "three runs", |world| {
        world
            .query_filtered::<(), (With<Run>, With<Settled>)>()
            .iter(world)
            .count()
            == 3
    });
    let mut answers: Vec<String> = app
        .world_mut()
        .query::<&RunResult>()
        .iter(app.world())
        .map(|r| r.0.clone())
        .collect();
    answers.sort();
    assert_eq!(answers, ["one", "three", "two"]);
}
