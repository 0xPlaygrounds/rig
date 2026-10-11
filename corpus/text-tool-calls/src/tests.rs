use std::sync::Arc;
use std::sync::mpsc::Receiver;

use rig_cassette::journal::MemoryStore;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::{app_on, connect, reply, run_until};
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

/// `model` as a catalog model that cannot call tools.
fn without_tools(model: &MockCompletionModel) -> Option<Connection> {
    let mut connection = connect(model)?;
    let mut spec = connection.spec.as_ref().clone();
    spec.tools = false;
    connection.spec = Arc::new(spec);
    Some(connection)
}

#[derive(Deserialize, JsonSchema)]
struct ReadArgs {
    /// The file.
    path: String,
}

/// The paths `read` was called with.
#[derive(Resource, Default)]
struct Reads(Vec<String>);

/// Every streamed text of a model call a front could draw.
#[derive(Resource, Default)]
struct Shown(Vec<String>);

/// Turns ended.
#[derive(Resource, Default)]
struct Ended(usize);

/// The kernel and the plugin on `store`, with a `read` tool that answers
/// "x", after the first frame restored the session.
fn start(store: &MemoryStore) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    app.add_plugins((AgentPlugin, JournalPlugin, TextToolCallsPlugin))
        .init_resource::<Reads>()
        .init_resource::<Shown>()
        .init_resource::<Ended>()
        .add_open_tool("read", "Read a file.", ToolOptions::default(), read)
        .add_systems(Update, watch.after(PollCalls))
        .add_observer(|_: On<TurnEnded>, mut ended: ResMut<Ended>| ended.0 += 1);
    app.finish();
    app.update();
    (app, wakes)
}

fn read(called: On<ToolCalled<ReadArgs>>, mut reads: ResMut<Reads>, mut commands: Commands) {
    reads.0.push(called.args.path.clone());
    let output = ToolOutput(ToolResult::success("x".into()));
    commands.entity(called.call).insert(output);
}

fn watch(partials: Query<&Partial, Changed<Partial>>, mut shown: ResMut<Shown>) {
    shown
        .0
        .extend(partials.iter().map(|partial| partial.text.clone()));
}

fn first_agent(world: &mut World) -> Option<Entity> {
    world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next()
}

/// The `read` calls in `messages`, by their arguments.
fn read_calls(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .filter_map(|message| match message {
            Message::Assistant(assistant) => Some(assistant.tool_calls()),
            _ => None,
        })
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
    let connection = without_tools(&model);
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
    run_until(&mut app, &wakes, |world| world.resource::<Ended>().0 > 0);

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
    // Whatever text a front drew, it never showed any of the block.
    for shown in &app.world().resource::<Shown>().0 {
        let replies = ["Let me look.", "It says x."];
        assert!(
            replies
                .iter()
                .any(|reply| reply.starts_with(shown.as_str())),
            "{shown}"
        );
    }
    // The result went back as user text, and no message holds a native
    // tool call or result.
    let user_text = |message: &Message| {
        match message {
        Message::User { content } => content.iter().any(|content| {
            matches!(content, UserContent::Text(text) if text.text() == "Result of read: x")
        }),
        _ => false,
    }
    };
    assert!(second.chat_history.iter().any(user_text), "{second:?}");
    assert!(read_calls(&second.chat_history).is_empty());
    let sent = serde_json::to_string(second).unwrap_or_default();
    assert!(!sent.contains("tool_result"), "{sent}");

    // The conversation keeps a real call and its result, and the answer.
    let conversation = app
        .world()
        .get::<Conversation>(agent)
        .map(Conversation::messages);
    let messages = conversation.unwrap_or_default().to_vec();
    assert_eq!(read_calls(&messages), [r#"{"path":"a.txt"}"#]);
    let shown_first = messages.iter().find_map(|message| match message {
        Message::Assistant(assistant) => Some(assistant.content.clone()),
        _ => None,
    });
    let text = shown_first.and_then(|content| {
        content.into_iter().find_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text().to_owned()),
            _ => None,
        })
    });
    assert_eq!(text.as_deref(), Some("Let me look."));
    assert!(messages.iter().any(|message| matches!(message,
        Message::User { content } if content.iter().any(|c| matches!(c, UserContent::ToolResult(_))))));
    drop(app);

    // ...as saved: a restart restores the real call.
    let (mut restarted, _wakes) = start(&store);
    let world = restarted.world_mut();
    let agent = first_agent(world);
    let restored = agent.and_then(|agent| world.get::<Conversation>(agent));
    let restored = restored.map(|conversation| read_calls(conversation.messages()));
    assert_eq!(restored.unwrap_or_default(), [r#"{"path":"a.txt"}"#]);
}

/// The streamed text of the reply, as the turn reads it: the block never
/// reaches it, though the chunks cut it mid-tag.
#[test]
fn a_block_split_mid_tag_never_streams() {
    let model = MockCompletionModel::from_stream_turns([chunked_call()]);
    let connection = without_tools(&model);
    assert!(connection.is_some(), "the catalog lists the model");
    let Some(connection) = connection else {
        return;
    };
    let handler = adapter(&connection).handler.0;
    let request = CompletionRequest::new(Message::user("What is in a.txt?"));
    let kind = EffectKind::Completion {
        request,
        stream: true,
    };
    let items: Vec<_> = futures::executor::block_on(async {
        let reply = handler
            .handle(kind, Dispatch::new(EffectId::from_raw(1), true))
            .await;
        reply.into_stream().collect().await
    });
    let mut streamed = String::new();
    let mut calls = Vec::new();
    for item in items {
        match item {
            Ok(Relayed::Item(Item::Event(StreamEvent::Text { text, .. }))) => {
                streamed.push_str(&text);
            }
            Ok(Relayed::Done(response)) => {
                calls.extend(
                    response
                        .tool_calls()
                        .map(|call| call.function.raw_arguments()),
                );
            }
            _ => {}
        }
    }
    assert_eq!(streamed, "Let me look.");
    assert_eq!(calls, [r#"{"path":"a.txt"}"#]);
}
