use rig_harness::harness_protocol::Invocation;

use super::*;

#[test]
fn the_agent_starts_on_the_model_named_with_model() {
    let mut app = App::new();
    app.insert_resource(Invoked {
        args: Invocation {
            print: None,
            model: Some("deepseek/deepseek-flash".to_owned()),
        },
        terminal: false,
    })
    .add_plugins((AgentPlugin, ModelsPlugin));
    app.update();
    app.update();
    let world = app.world_mut();
    let chosen: Vec<String> = world
        .query::<&ModelChoice>()
        .iter(world)
        .map(|choice| choice.0.clone())
        .collect();
    assert_eq!(chosen, ["deepseek/deepseek-flash"]);
}
