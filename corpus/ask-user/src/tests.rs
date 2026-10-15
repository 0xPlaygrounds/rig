use std::sync::mpsc::Receiver;

use rig_core::test_utils::MockCompletionModel;
use rig_harness::pick::PickPlugin;
use rig_harness_test_support::MemoryStore;
use serde_json::json;

use super::*;
use rig_harness_test_support::{
    Ended, ToolOutputs, app_on, calls, connect, first_agent, log_ended, log_tool_outputs, reply,
    run_until,
};

/// The kernel, this plugin and the harness's pickers, a front answering
/// questions when `someone_answers`, and its first agent, answered by a
/// model that asks `DB?` and then answers; the agent was asked to set up
/// storage.
fn start(someone_answers: bool) -> (App, Receiver<()>, MockCompletionModel) {
    let model = MockCompletionModel::from_stream_turns([
        calls(
            "1",
            "ask_user",
            json!({ "question": "DB?", "choices": ["pg", "sqlite"] }),
        ),
        reply("Using it."),
    ]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.add_plugins((AgentPlugin, SessionLogPlugin, PickPlugin, AskUserPlugin))
        .init_resource::<Ended>()
        .init_resource::<ToolOutputs>()
        .add_observer(log_ended)
        .add_observer(log_tool_outputs);
    if someone_answers {
        app.world_mut().spawn(Answerer::default());
    }
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = connect(&model) {
        world.entity_mut(agent).insert(connection);
    }
    world.trigger(Deliver::new(agent, "Set up storage.", DeliveryMode::Steer));
    (app, wakes, model)
}

/// The open questions.
fn open_asks(world: &mut World) -> Vec<(Entity, Ask)> {
    world
        .query_filtered::<(Entity, &Ask), Without<Outcome>>()
        .iter(world)
        .map(|(entity, ask)| (entity, ask.clone()))
        .collect()
}

/// What call 1 answered the model.
fn answered(world: &World) -> Option<&str> {
    world
        .resource::<ToolOutputs>()
        .0
        .get("1")
        .map(String::as_str)
}

#[test]
fn the_turn_waits_for_the_answer_a_front_gives() {
    let (mut app, wakes, model) = start(true);
    run_until(&mut app, &wakes, |world| {
        world.iter_entities().any(|entity| entity.contains::<Ask>())
    });
    for _ in 0..5 {
        app.update();
    }
    assert_eq!(model.requests().len(), 1, "the turn waits for the user");
    let asks = open_asks(app.world_mut());
    assert_eq!(asks.len(), 1, "one open question: {asks:?}");
    let Some((question, ask)) = asks.first() else {
        return;
    };
    assert_eq!(ask.title, "DB?");
    let ids: Vec<&str> = ask.choices.iter().map(|c| c.id.as_str()).collect();
    assert_eq!(ids, ["pg", "sqlite"]);
    // What every front's picker triggers when "sqlite" is chosen.
    let pick = Pick::person(*question, Some(Chosen::item(1)));
    app.world_mut().trigger(pick);
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 1
    });
    assert_eq!(answered(app.world()), Some("sqlite"));
    assert_eq!(model.requests().len(), 2);
}

#[test]
fn without_anyone_to_answer_the_tool_answers_at_once() {
    let (mut app, wakes, _model) = start(false);
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 1
    });
    assert!(open_asks(app.world_mut()).is_empty());
    assert_eq!(answered(app.world()), Some(NO_USER));
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(AskUserPlugin);
}
