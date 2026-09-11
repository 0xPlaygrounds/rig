//! An agent with two tools answers "Calculate 2 - 5." using the public
//! construction API. The scripted model and tools require no credentials.

mod support;

use bevy_app::{App, ScheduleRunnerPlugin, Update};
use rig_core::{message::AssistantContent, serve::ServingPolicy};
use rig_ecs::{
    bus::{Handlers, run_to_quiescence},
    commands::{Agent, Prompt, install},
};

const PREAMBLE: &str = "You are a calculator here to help the user perform arithmetic operations. \
     You must use the provided tools before answering.";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut app = App::new();
    install(app.world_mut(), ServingPolicy::default())?;
    app.add_plugins(ScheduleRunnerPlugin::default())
        .add_systems(Update, run_to_quiescence)
        .add_observer(support::exit_when_failed)
        .add_observer(support::print_the_answer_and_exit);

    let model = support::Scripted::new(vec![
        vec![support::call(
            "subtract",
            serde_json::json!({"x": 2, "y": 5}),
        )],
        vec![AssistantContent::text("-3")],
    ]);
    let world = app.world_mut();
    let model = Handlers::register_in(world, support::MODEL, model)?;
    let add = Handlers::register_in(world, "demo/add", support::add())?;
    let subtract = Handlers::register_in(world, "demo/subtract", support::subtract())?;
    let agent = Agent::new(model)
        .owner("calculator")
        .preamble(PREAMBLE)
        .tools([add, subtract])
        .max_turns(2)
        .spawn(world)?;
    Prompt::new(agent, "Calculate 2 - 5.").spawn(world)?;

    if app.run().is_success() {
        Ok(())
    } else {
        Err(std::io::Error::other("the example run failed").into())
    }
}
