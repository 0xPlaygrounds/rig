//! Display append-only model text using independent per-effect cursors.
//! The scripted model requires no credentials. Raw stream events, errors, and
//! usage remain available on the underlying components.

mod support;

use bevy_app::{App, AppExit, ScheduleRunnerPlugin, Update};
use bevy_ecs::prelude::*;
use rig_core::{
    message::{AssistantContent, Message},
    serve::ServingPolicy,
};
use rig_ecs::{
    agent::MessageParts,
    bus::{Handlers, RigSchedule, Streamed, run_to_quiescence},
    commands::{Agent, Prompt, install},
    inspect::RunView,
    stream::StreamText,
    systems::RigSet,
};
use std::io::Write;

#[derive(Resource)]
struct DisplayedRun(Entity);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut app = App::new();
    install(app.world_mut(), ServingPolicy::default())?;
    app.add_plugins(ScheduleRunnerPlugin::default())
        .add_systems(Update, (run_to_quiescence, finish).chain())
        .add_systems(RigSchedule, show.after(RigSet::Fold));

    let model = support::Scripted::new(vec![vec![AssistantContent::text(
        "Why did the Rustacean cross the road? To get to the other side — safely.",
    )]]);
    let history = [
        Message::user("Tell me a joke!"),
        Message::assistant("Why did the chicken cross the road?"),
    ]
    .iter()
    .filter_map(MessageParts::from_message)
    .collect::<Vec<_>>();
    let world = app.world_mut();
    let model = Handlers::register_in(world, support::MODEL, model)?;
    let agent = Agent::new(model)
        .preamble("You are a comedian here to entertain.")
        .spawn(world)?;
    let run = Prompt::new(agent, "Another one, about Rust.")
        .history(history)
        .streaming()
        .spawn(world)?;
    app.insert_resource(DisplayedRun(run));

    if app.run().is_success() {
        Ok(())
    } else {
        Err(std::io::Error::other("streaming example failed").into())
    }
}

fn show(
    streams: Query<(Entity, &Streamed), Changed<Streamed>>,
    mut removed: RemovedComponents<Streamed>,
    mut text: Local<StreamText>,
    mut exit: MessageWriter<AppExit>,
) {
    for entity in removed.read() {
        text.forget(entity);
    }
    for (entity, stream) in &streams {
        match text.read(entity, stream) {
            Ok(delta) => {
                print!("{delta}");
                if let Err(error) = std::io::stdout().flush() {
                    eprintln!("could not display stream: {error}");
                    exit.write(AppExit::error());
                }
            }
            Err(error) => {
                eprintln!("{error}");
                exit.write(AppExit::error());
            }
        }
    }
}

fn finish(displayed: Res<DisplayedRun>, runs: Query<RunView>, mut exit: MessageWriter<AppExit>) {
    let Ok(run) = runs.get(displayed.0) else {
        eprintln!("the displayed run was removed");
        exit.write(AppExit::error());
        return;
    };
    if !run.is_finished() {
        return;
    }
    println!();
    if let Some(failure) = run.failure {
        eprintln!("the run failed: {:?}", failure.0);
        exit.write(AppExit::error());
    } else {
        exit.write(AppExit::Success);
    }
}
