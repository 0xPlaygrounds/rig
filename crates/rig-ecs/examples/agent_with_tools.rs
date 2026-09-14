//! `examples/agent_with_tools` side by side: an agent with two tools
//! answers "Calculate 2 - 5." — there the builder's `dynamic_tools`, here
//! a `Grant` link per tool entity and a run entity. The model is scripted
//! (`support`), so no key.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::type_complexity,
    reason = "an example: user code, thirty lines, a mock behind it"
)]

mod support;

use bevy_app::Startup;
use bevy_ecs::prelude::*;
use rig_core::message::AssistantContent;
use rig_ecs::{agent::Order, bus::Handlers, prelude::*, systems::spawn_run};

const PREAMBLE: &str = "You are a calculator here to help the user perform arithmetic operations. \
     You must use the provided tools before answering.";

fn main() {
    support::app()
        .add_systems(Startup, ask)
        .add_observer(support::print_the_answer_and_exit)
        .run();
}

fn ask(mut handlers: Handlers, mut commands: Commands) {
    let model = support::Scripted::new(vec![
        vec![support::call(
            "subtract",
            serde_json::json!({"x": 2, "y": 5}),
        )],
        vec![AssistantContent::text("-3")],
    ]);
    let tools = vec![support::add(), support::subtract()];
    let (model, tools) = support::register(&mut handlers, model, tools);
    let agent = support::agent(&mut commands, model, PREAMBLE, 2);
    for (order, tool) in tools.into_iter().enumerate() {
        commands.spawn((Grant(tool), Order(order as u64), ChildOf(agent)));
    }
    commands.queue(move |world: &mut World| {
        spawn_run(world, agent, &[], "Calculate 2 - 5.", false, None);
    });
}
