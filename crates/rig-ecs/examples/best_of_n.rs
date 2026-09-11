//! Fork one prompt into three independent runs and select the longest successful
//! answer. Length is only a deterministic demonstration score, not a quality
//! metric. The judge watches this cohort; unrelated runs do not delay it.

mod support;

use bevy_app::{App, AppExit, ScheduleRunnerPlugin, Update};
use bevy_ecs::prelude::*;
use rig_core::{message::AssistantContent, serve::ServingPolicy};
use rig_ecs::{
    bus::{Handlers, run_to_quiescence},
    commands::{Agent, Prompt, install},
    inspect::RunView,
    lifecycle::fork,
};

#[derive(Resource)]
struct Cohort(Vec<Entity>);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut app = App::new();
    install(app.world_mut(), ServingPolicy::default())?;
    app.add_plugins(ScheduleRunnerPlugin::default())
        .add_systems(Update, (run_to_quiescence, judge).chain());

    let model = support::Scripted::new(vec![
        vec![AssistantContent::text("Rust is a systems language.")],
        vec![AssistantContent::text(
            "Rust is a systems language with memory safety and no garbage collector.",
        )],
        vec![AssistantContent::text("Rust: fast, safe.")],
    ]);
    let world = app.world_mut();
    let model = Handlers::register_in(world, support::MODEL, model)?;
    let agent = Agent::new(model)
        .preamble("You are concise.")
        .spawn(world)?;
    let first = Prompt::new(agent, "What is Rust, in one sentence?").spawn(world)?;
    let second = fork(world, first)?;
    let third = fork(world, first)?;
    app.insert_resource(Cohort(vec![first, second, third]));

    if app.run().is_success() {
        Ok(())
    } else {
        Err(std::io::Error::other("no successful candidate").into())
    }
}

fn judge(cohort: Res<Cohort>, runs: Query<RunView>, mut exit: MessageWriter<AppExit>) {
    let mut best: Option<&str> = None;
    for entity in &cohort.0 {
        let Ok(run) = runs.get(*entity) else {
            eprintln!("candidate {entity:?} was removed");
            continue;
        };
        if !run.is_finished() {
            return;
        }
        if let Some(answer) = run.answer() {
            if best.is_none_or(|current| answer.len() > current.len()) {
                best = Some(answer);
            }
        } else if let Some(failure) = run.failure {
            eprintln!("candidate {entity:?} failed: {:?}", failure.0);
        }
    }
    if let Some(answer) = best {
        println!("best of {}: {answer}", cohort.0.len());
        exit.write(AppExit::Success);
    } else {
        exit.write(AppExit::error());
    }
}
