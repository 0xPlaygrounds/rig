use rig_harness::commands::RunCommand;
use rig_harness::prelude::*;

use super::BasicCommandsPlugin;

#[test]
fn retry_is_an_unknown_command() {
    let mut app = App::new();
    app.add_plugins((AgentPlugin, BasicCommandsPlugin));
    let world = app.world_mut();
    let agent = world.spawn(Agent).id();
    world.trigger(RunCommand::new(agent, "retry".to_owned()));
    world.flush();
    let why: Vec<Option<&str>> = app
        .world()
        .resource::<Messages<Recalled>>()
        .iter_current_update_messages()
        .map(|recalled| recalled.why.as_deref())
        .collect();
    assert_eq!(why, [Some("Unknown command /retry.")]);
}
