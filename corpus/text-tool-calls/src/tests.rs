use std::sync::Arc;
use std::sync::mpsc::Receiver;

use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{Ended, app_on, connect, log_ended, reply, run_until};
use schemars::JsonSchema;
use serde::Deserialize;

use super::*;

/// The model's first reply: a tool block split across chunks, mid-tag.
fn chunked_call() -> Vec<MockStreamEvent> {
    vec![
        MockStreamEvent::text("Let me look.<to"),
        MockStreamEvent::text("ol name=\"read\">{\"path\":"),
        MockStreamEvent::text(" \"a.txt\"}</tool>"),
        MockStreamEvent::final_response_with_default_usage(),
    ]
}

#[derive(Deserialize, JsonSchema)]
struct ReadArgs {
    /// The file.
    path: String,
}

/// The paths `read` was called with.
#[derive(Resource, Default)]
struct Reads(Vec<String>);

/// The text each model call streamed, as a front reads it, by call.
#[derive(Resource, Default)]
struct Shown(Vec<(Entity, String)>);

/// The kernel and the plugin on `store`, with a `read` tool that answers
/// "x", after the first frame restored the session.
fn start(store: &MemoryStore) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    app.add_plugins((AgentPlugin, SessionLogPlugin, TextToolCallsPlugin))
        .init_resource::<Reads>()
        .init_resource::<Shown>()
        .init_resource::<Ended>()
        .add_open_tool("read", "Read a file.", ToolOptions::default(), read)
        .add_systems(PostUpdate, watch)
        .add_observer(log_ended);
    app.finish();
    app.update();
    (app, wakes)
}

fn read(called: On<ToolCalled<ReadArgs>>, mut reads: ResMut<Reads>, mut commands: Commands) {
    reads.0.push(called.args.path.clone());
    let read = Outcome::from(ToolResult::success("x".into()));
    commands.entity(called.call).insert_if_new(read);
}

/// A front drawing each call's text as it streams in.
fn watch(mut deltas: MessageReader<StreamDelta>, mut shown: ResMut<Shown>) {
    for delta in deltas.read() {
        let StreamEvent::Text { text: fragment, .. } = &delta.event else {
            continue;
        };
        match shown.0.iter_mut().find(|(call, _)| *call == delta.effect) {
            Some((_, text)) => text.push_str(fragment),
            None => shown.0.push((delta.effect, fragment.clone())),
        }
    }
}

fn first_agent(world: &mut World) -> Option<Entity> {
    world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next()
}

/// The arguments of the `read` calls in `messages`.
fn read_calls(messages: &[std::sync::Arc<Message>]) -> Vec<String> {
    let calls = messages.iter().filter_map(|message| match &**message {
        Message::Assistant(assistant) => Some(assistant.tool_calls()),
        _ => None,
    });
    calls
        .flatten()
        .filter(|call| call.function.name.as_str() == "read")
        .map(|call| call.function.raw_arguments())
        .collect()
}

#[test]
fn a_streamed_text_block_runs_as_a_real_tool_call() {
    let store = MemoryStore::default();
    let (mut app, wakes) = start(&store);
    let model = MockCompletionModel::from_stream_turns([chunked_call(), reply("It says x.")]);
    // The catalog's DeepSeek model, flagged as calling no tools.
    let connection = connect(&model).map(|mut connection| {
        let mut spec = connection.spec.as_ref().clone();
        spec.tools = false;
        connection.spec = Arc::new(spec);
        connection
    });
    let agent = first_agent(app.world_mut());
    assert!(
        connection.is_some() && agent.is_some(),
        "a model and an agent"
    );
    let (Some(connection), Some(agent)) = (connection, agent) else {
        return;
    };
    app.world_mut().entity_mut(agent).insert(connection);
    app.world_mut().trigger(Deliver::new(
        agent,
        "What is in a.txt?",
        DeliveryMode::Queue,
    ));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });

    let requests = model.requests();
    assert_eq!(requests.len(), 2, "{requests:?}");
    let [first, second] = requests.as_slice() else {
        return;
    };
    assert!(first.tools.is_empty(), "no native tools: {:?}", first.tools);
    let preamble = first.system_instructions().unwrap_or_default();
    assert!(preamble.contains("<tool name="), "{preamble}");
    assert!(preamble.contains("## read"), "{preamble}");
    assert_eq!(app.world().resource::<Reads>().0, ["a.txt"]);

    // A front saw the text outside the block, and never a bit of it.
    let shown = &app.world().resource::<Shown>().0;
    let shown: Vec<&str> = shown.iter().map(|(_, text)| text.as_str()).collect();
    assert_eq!(shown, ["Let me look.", "It says x."]);

    // The result went back as user text, with no native call or result.
    let result_as_text = second.chat_history.iter().any(|message| match &**message {
        Message::User { content } => content.iter().any(
            |content| matches!(content, UserContent::Text(text) if text.text() == "Result of read: x"),
        ),
        _ => false,
    });
    assert!(result_as_text, "{second:?}");
    assert!(read_calls(&second.chat_history).is_empty());
    let sent = serde_json::to_string(second).unwrap_or_default();
    assert!(!sent.contains("tool_result"), "{sent}");
    drop(app);

    // The saved conversation keeps a real call and its result.
    let (mut restarted, _wakes) = start(&store);
    let world = restarted.world_mut();
    let restored = first_agent(world).and_then(|agent| world.get::<Conversation>(agent));
    let messages = restored
        .map(|conversation| conversation.messages().to_vec())
        .unwrap_or_default();
    assert_eq!(read_calls(&messages), [r#"{"path":"a.txt"}"#]);
    let has_result = messages.iter().any(|message| {
        matches!(&**message, Message::User { content }
            if content.iter().any(|content| matches!(content, UserContent::ToolResult(_))))
    });
    assert!(has_result, "{messages:?}");
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(TextToolCallsPlugin);
}
