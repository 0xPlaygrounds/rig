//! `examples/agent_stream_chat` side by side: a streamed answer shown as
//! it arrives — there a `StreamingResult` polled in a loop, here a run
//! spawned streamed and a system after `RigSet::Fold` printing what the
//! effect's `Streamed` fold gained since the last tick.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::type_complexity,
    reason = "an example: user code, thirty lines, a mock behind it"
)]

mod support;

use std::io::Write;

use bevy_app::{AppExit, Startup};
use bevy_ecs::prelude::*;
use rig_core::message::{AssistantContent, Message};
use rig_ecs::{
    agent::MessageParts,
    bus::{Handlers, RigSchedule},
    prelude::*,
    systems::spawn_run,
};

fn main() {
    support::app()
        .add_systems(Startup, ask)
        .add_systems(RigSchedule, show.after(RigSet::Fold))
        .add_observer(|_: On<Add, Settled>, mut exit: MessageWriter<AppExit>| {
            println!();
            exit.write(AppExit::Success);
        })
        .run();
}

fn ask(mut handlers: Handlers, mut commands: Commands) {
    let model = support::Scripted::new(vec![vec![AssistantContent::text(
        "Why did the Rustacean cross the road? To get to the other side — safely.",
    )]]);
    let (model, _) = support::register(&mut handlers, model, Vec::new());
    let agent = support::agent(
        &mut commands,
        model,
        "You are a comedian here to entertain.",
        1,
    );
    let history: Vec<MessageParts> = [
        Message::user("Tell me a joke!"),
        Message::assistant("Why did the chicken cross the road?"),
    ]
    .iter()
    .filter_map(MessageParts::from_message)
    .collect();
    commands.queue(move |world: &mut World| {
        spawn_run(
            world,
            agent,
            &history,
            "Another one, about Rust.",
            true,
            None,
        );
    });
}

/// The UI: print what arrived since the last tick.
fn show(streams: Query<&Streamed, Changed<Streamed>>, mut shown: Local<usize>) {
    for stream in &streams {
        if stream.text.len() > *shown {
            print!("{}", &stream.text[*shown..]);
            let _ = std::io::stdout().flush();
            *shown = stream.text.len();
        }
    }
}
