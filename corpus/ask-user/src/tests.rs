use std::sync::mpsc::Receiver;

use rig_cassette::journal::MemoryStore;
use rig_core::completion::Message;
use rig_core::message::ToolResultContent;
use rig_core::test_utils::MockCompletionModel;
use serde_json::json;

use super::*;
use rig_harness_test_support::{Ended, app_on, calls, connect, log_ended, reply, run_until};

/// The kernel and this plugin, a front answering questions when
/// `someone_answers`, and its first agent, answered by a model that asks
/// `DB?` and then answers; the agent was asked to set up storage.
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
    app.add_plugins((AgentPlugin, JournalPlugin, AskUserPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    if someone_answers {
        app.world_mut().spawn(Answerer::default());
    }
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next()
        .unwrap_or(Entity::PLACEHOLDER);
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

/// The text of the tool result the second request ends with.
fn answered(model: &MockCompletionModel) -> Option<String> {
    let requests = model.requests();
    let Some(Message::User { content }) = requests.get(1)?.chat_history.last() else {
        return None;
    };
    content.iter().find_map(|part| match part {
        UserContent::ToolResult(result) => result.content.iter().find_map(|part| match part {
            ToolResultContent::Text(text) => Some(text.text.clone()),
            _ => None,
        }),
        _ => None,
    })
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
    let ids: Vec<&str> = ask
        .choices
        .iter()
        .map(|choice| choice.id.as_str())
        .collect();
    assert_eq!(ids, ["pg", "sqlite"]);
    // What the terminal view's picker inserts when "sqlite" is chosen.
    let answer = Answer::Choice("sqlite".to_owned()).outcome();
    app.world_mut().entity_mut(*question).insert(answer);
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 1
    });
    assert_eq!(answered(&model).as_deref(), Some("sqlite"));
}

#[test]
fn without_anyone_to_answer_the_tool_answers_at_once() {
    let (mut app, wakes, model) = start(false);
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 1
    });
    assert!(open_asks(app.world_mut()).is_empty());
    assert_eq!(answered(&model).as_deref(), Some(NO_USER));
}
