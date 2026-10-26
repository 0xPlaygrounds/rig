use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::MemoryStore;

use super::*;
use rig_harness::ask::AskPlugin;
use rig_harness::pick::PickPlugin;
use rig_harness_test_support::{app_on, connect, reply, run_until};

/// The text of the user's messages to `agent`.
fn user_texts(world: &World, agent: Entity) -> Vec<String> {
    let Some(conversation) = world.get::<Conversation>(agent) else {
        return Vec::new();
    };
    conversation
        .messages()
        .iter()
        .filter_map(|message| match &**message {
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
        SessionLogPlugin,
        AskPlugin,
        PickPlugin,
        WindowGuiPlugin,
    ));
    app.finish();
    app.update();
    let model = MockCompletionModel::from_stream_turns([reply("hello")]);
    let world = app.world_mut();
    let agent = world.resource::<ShownAgent>().0;
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

/// A question from any plugin is a dialog; its button answers it, and the
/// answered question leaves the dialogs.
#[test]
fn a_question_is_a_dialog_its_buttons_answer() {
    let store = MemoryStore::default();
    let (mut app, _wakes) = app_on(&store);
    app.set_error_handler(bevy_ecs::error::warn).add_plugins((
        AgentPlugin,
        SessionLogPlugin,
        AskPlugin,
        PickPlugin,
        WindowGuiPlugin,
    ));
    app.finish();
    app.update();
    let agent = app.world().resource::<ShownAgent>().0;
    let agent = agent.unwrap_or(Entity::PLACEHOLDER);
    let ask = Ask::yes_no("Run `shell`?").body("cargo test");
    let ask = app.world_mut().spawn((ask, ForAgent(agent))).id();
    app.update();
    let shown: Vec<_> = app.world().resource::<GuiDialogs>().open.clone();
    assert_eq!(
        shown,
        [Dialog {
            entity: ask,
            title: "Run `shell`?".to_owned(),
            body: "cargo test".to_owned(),
            choices: vec!["Yes".to_owned(), "No".to_owned()],
            free_text: false,
        }]
    );

    app.world_mut().trigger(Choose { item: 0 });
    app.update();
    let answer = app.world().get::<Outcome>(ask).and_then(Answer::of);
    assert_eq!(answer, Some(Answer::choice("yes")));
    assert!(app.world().resource::<GuiDialogs>().open.is_empty());
}

/// Ends `conversation`'s turn as a stop does, at its last entry.
fn stop_turn(conversation: &mut Conversation) {
    conversation.halt(Halt::Stopped);
    let at = conversation.entries().next_back().map(|entry| entry.id);
    conversation.end_turn(TurnEnd::Stopped { why: None, at });
}

/// A stop follows the last message of the turn it cut off: the cut reply,
/// or the request nothing answered. The note it leaves the model draws
/// nothing.
#[test]
fn a_stop_follows_the_turn_it_cut_off() {
    let mut conversation = Conversation::default();
    conversation.push(Message::user("write"));
    conversation.open_turn();
    let text = vec![AssistantContent::text("half")];
    let aborted = AssistantMessage::aborted(None, text, "stopped");
    conversation.push(Message::Assistant(aborted));
    stop_turn(&mut conversation);
    conversation.push(Message::user("again"));
    conversation.open_turn();
    stop_turn(&mut conversation);
    conversation.push(Message::user("third"));
    assert_eq!(
        blocks_of(&conversation, &HashSet::new()),
        [
            Block::User("write".to_owned()),
            Block::Answer("half".to_owned()),
            Block::Stopped { replied: true },
            Block::User("again".to_owned()),
            Block::Stopped { replied: false },
            Block::User("third".to_owned()),
        ]
    );
}
