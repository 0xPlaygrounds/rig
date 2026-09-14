//! Best of n: one prompt, three runs, the judge picks — `agent::fork`
//! clones the run entity and its subtree, so the three runs share the
//! prompt and nothing after it; a system judges when all three settled.
//! In rig's root examples this is `agent_parallelization`'s shape with
//! one agent; here the runs are entities and the judge is a query.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::type_complexity,
    reason = "an example: user code, thirty lines, a mock behind it"
)]

mod support;

use bevy_app::{AppExit, Startup, Update};
use bevy_ecs::prelude::*;
use rig_core::message::AssistantContent;
use rig_ecs::{
    agent::{Run, fork},
    bus::Handlers,
    prelude::*,
    systems::spawn_run,
};

const N: usize = 3;

fn main() {
    support::app()
        .add_systems(Startup, ask)
        .add_systems(Update, judge)
        .run();
}

fn ask(mut handlers: Handlers, mut commands: Commands) {
    let model = support::Scripted::new(vec![
        vec![AssistantContent::text("Rust is a systems language.")],
        vec![AssistantContent::text(
            "Rust is a systems language with memory safety and no garbage collector.",
        )],
        vec![AssistantContent::text("Rust: fast, safe.")],
    ]);
    let (model, _) = support::register(&mut handlers, model, Vec::new());
    let agent = support::agent(&mut commands, model, "You are concise.", 1);
    commands.queue(move |world: &mut World| {
        let run = spawn_run(
            world,
            agent,
            &[],
            "What is Rust, in one sentence?",
            false,
            None,
        );
        for _ in 1..N {
            fork(world, run);
        }
    });
}

/// When every run settled, the longest answer wins.
fn judge(runs: Query<(&RunResult, Has<Settled>), With<Run>>, mut exit: MessageWriter<AppExit>) {
    if runs.iter().count() < N || runs.iter().any(|(_, settled)| !settled) {
        return;
    }
    let best = runs
        .iter()
        .map(|(result, _)| &result.0)
        .max_by_key(|answer| answer.len());
    println!("best of {N}: {}", best.map_or("", String::as_str));
    exit.write(AppExit::Success);
}
