use rig_cassette::journal::MemoryStore;
use rig_core::test_utils::MockCompletionModel;

use super::*;
use rig_harness_test_support::{app_on, connect, reply, run_until};

/// The text of the user's messages to `agent`.
fn user_texts(world: &World, agent: Entity) -> Vec<String> {
    let Some(conversation) = world.get::<Conversation>(agent) else {
        return Vec::new();
    };
    conversation
        .messages()
        .iter()
        .filter_map(|message| match message {
            Message::User { content } => Some(content.iter()),
            _ => None,
        })
        .flatten()
        .filter_map(|part| match part {
            UserContent::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect()
}

/// The window's model without a window: what is typed in the input box
/// goes to the shown agent as the user's message, the answer lands in the
/// transcript, and the agent list shows the agent idle again.
#[test]
fn typing_in_the_input_box_reaches_the_agent_and_its_answer_the_transcript() {
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    app.set_error_handler(bevy_ecs::error::warn).add_plugins((
        AgentPlugin,
        JournalPlugin,
        WindowGuiPlugin,
    ));
    app.finish();
    app.update();
    let model = MockCompletionModel::from_stream_turns([reply("hello")]);
    let world = app.world_mut();
    let agent = world.resource::<GuiAgents>().shown;
    if let (Some(agent), Some(connection)) = (agent, connect(&model)) {
        world.entity_mut(agent).insert(connection);
    }

    world.resource_mut::<GuiInput>().text = "hi".to_owned();
    world.trigger(Submit);
    run_until(&mut app, &wakes, |world| {
        let idle = agent
            .and_then(|agent| world.resource::<GuiAgents>().row(agent).cloned())
            .is_some_and(|row| !row.busy);
        idle && world.resource::<GuiTranscript>().contains("hello")
    });

    let world = app.world();
    let agent = agent.unwrap_or(Entity::PLACEHOLDER);
    assert_eq!(user_texts(world, agent), ["hi"]);
    assert!(world.resource::<GuiInput>().text.is_empty());
    let transcript = world.resource::<GuiTranscript>();
    assert_eq!(transcript.agent, Some(agent));
    assert_eq!(
        transcript.blocks,
        [
            Block::User("hi".to_owned()),
            Block::Answer("hello".to_owned())
        ]
    );
    let row = world.resource::<GuiAgents>().row(agent).cloned();
    assert_eq!(
        row.map(|row| (row.status, row.busy)),
        Some(("idle".to_owned(), false))
    );
}
