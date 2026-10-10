use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, Instant};

use rig_cassette::journal::MemoryStore;
use rig_core::ProviderResponseError;
use rig_core::catalog::Catalog;
use rig_core::message::{AssistantContent, AssistantMessage, CallId, ToolName, UserContent};
use rig_core::operation::Completion;
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ModelAdapter;
use rig_core::test_utils::{MockCompletionModel, MockError, MockStreamEvent};
use rig_core::transcript::final_answer;
use rig_ecs::commands::RunCommand;
use rig_ecs::effects::Handler;
use rig_ecs::journal::{SessionStore, commit_message};

use super::*;

/// How long a test waits for a turn to end.
const DEADLINE: Duration = Duration::from_secs(10);

/// What a second `PrepareRequest` observer adds to every request.
const INJECTED: &str = "Context from another plugin.";

/// How each turn ended, in order.
#[derive(Resource, Default)]
struct Ended(Vec<TurnOutcome>);

/// The kernel and the compaction plugin on `store`, after the first frame
/// restored the session, with its loop's wakes.
fn start(store: &MemoryStore) -> (App, Receiver<()>) {
    let (sender, wakes) = channel();
    let mut app = App::new();
    app.set_error_handler(bevy_ecs::error::warn)
        .add_plugins(TaskPoolPlugin::default())
        .insert_resource(Wake::new(move || {
            sender.send(()).ok();
        }))
        .insert_resource(SessionStore::new(store.clone()))
        .add_plugins((AgentPlugin, JournalPlugin, CompactionPlugin))
        .init_resource::<Ended>()
        .add_observer(|ended: On<TurnEnded>, mut log: ResMut<Ended>| {
            log.0.push(ended.outcome.clone());
        });
    app.finish();
    app.update();
    (app, wakes)
}

/// The app's first agent, answered by `model` as the catalog's DeepSeek
/// model with a `window`-token context window, with `messages` in its
/// conversation.
fn agent(
    app: &mut App,
    model: &MockCompletionModel,
    window: u32,
    messages: Vec<Message>,
) -> Option<Entity> {
    let spec = Catalog::builtin()
        .resolve("deepseek/deepseek-flash")
        .ok()?
        .spec;
    let spec = Arc::new(spec.clone().with_context_window(window));
    let handler = ErasedHandler::new(ModelAdapter::<Completion>::new(
        spec.reference(),
        model.clone(),
    ));
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next()?;
    world.entity_mut(agent).insert(Connection {
        spec,
        handler: Handler(handler),
    });
    for message in messages {
        commit_message(world, agent, message);
    }
    Some(agent)
}

/// A call of the `read` tool for `path` and its `output`.
fn read(id: &str, path: &str, output: &str) -> Vec<Message> {
    let Ok(name) = ToolName::new("read") else {
        return Vec::new();
    };
    let arguments = serde_json::json!({ "path": path });
    let call = AssistantContent::tool_call(id, name.clone(), arguments);
    vec![
        Message::Assistant(AssistantMessage::new(vec![call])),
        Message::tool_result(CallId::from_wire(id), name, output),
    ]
}

/// A reply of `text`, then the end of the stream.
fn reply(text: &str) -> Vec<MockStreamEvent> {
    vec![
        MockStreamEvent::text(text),
        MockStreamEvent::final_response_with_default_usage(),
    ]
}

/// A refusal of the request as longer than the model's window.
fn too_long() -> Vec<MockStreamEvent> {
    let refusal = ProviderResponseError::without_status("prompt is too long: 210000 tokens");
    vec![MockStreamEvent::Error(MockError::ProviderResponse(refusal))]
}

/// Asks `agent` a question, then runs frames, sleeping until a wake
/// between them, until a turn ended or the deadline passed.
fn ask(app: &mut App, wakes: &Receiver<()>, agent: Entity) {
    app.world_mut().trigger(Deliver::user(
        agent,
        "And the next question?",
        DeliveryMode::Steer,
        Vec::new(),
    ));
    let started = Instant::now();
    while started.elapsed() < DEADLINE && app.world().resource::<Ended>().0.is_empty() {
        app.update();
        wakes.recv_timeout(Duration::from_millis(100)).ok();
    }
}

/// The answer the one turn of `app` ended with.
fn answer(app: &App) -> Option<String> {
    match app.world().resource::<Ended>().0.as_slice() {
        [TurnOutcome::Answered(message)] => final_answer(message),
        _ => None,
    }
}

/// Each request `model` got, as JSON.
fn sent(model: &MockCompletionModel) -> Vec<String> {
    model
        .requests()
        .iter()
        .map(|request| serde_json::to_string(request).unwrap_or_default())
        .collect()
}

#[test]
fn a_conversation_near_the_window_is_summarized_before_the_request_and_restored() {
    let store = MemoryStore::default();
    let (mut app, wakes) = start(&store);
    // A second observer of every request, beside the plugin's.
    app.add_observer(|mut prepare: On<PrepareRequest>| {
        if let Some(Message::User { content }) = prepare.event_mut().messages.last_mut() {
            content.push(UserContent::text(INJECTED));
        }
    });
    let mut messages = vec![Message::user("What does the crate do?")];
    messages.extend(read("1", "src/lib.rs", "pub fn answer() -> u32 { 42 }"));
    let long = "word ".repeat(4_000);
    for turn in 0..6 {
        messages.push(Message::assistant(format!("{turn}: {long}")));
        messages.push(Message::user(format!("question {turn}")));
    }
    messages.push(Message::assistant("The last answer."));
    let model = MockCompletionModel::from_stream_turns([
        reply("## Goal\nAnswer the user's questions about the crate."),
        reply("done"),
    ]);
    let agent = agent(&mut app, &model, 40_000, messages);
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    ask(&mut app, &wakes, agent);
    assert_eq!(answer(&app).as_deref(), Some("done"));
    let sent = sent(&model);
    assert_eq!(sent.len(), 2, "a summary, then the request: {sent:?}");
    let [summary, request] = sent.as_slice() else {
        return;
    };
    assert!(
        summary.contains("You summarize a conversation"),
        "{summary}"
    );
    assert!(!summary.contains(INJECTED), "the held request was not sent");
    assert!(request.contains("## Goal"), "{request}");
    assert!(request.contains("<read-files>\\nsrc/lib.rs"), "{request}");
    assert_eq!(request.matches(INJECTED).count(), 1, "{request}");
    let condensed = app.world().get::<Condensed>(agent).cloned();
    assert!(
        condensed
            .as_ref()
            .is_some_and(|condensed| condensed.upto > 2)
    );
    drop(app);

    // A restart starts from the messages the summary kept.
    let (mut restarted, _wakes) = start(&store);
    let world = restarted.world_mut();
    let agent = world.query_filtered::<Entity, With<Agent>>().single(world);
    assert!(agent.is_ok(), "one agent was restored");
    let Ok(agent) = agent else {
        return;
    };
    let restored = world.get::<Condensed>(agent);
    let summary = |condensed: Option<&Condensed>| condensed.map(|kept| kept.summary.clone());
    assert_eq!(summary(restored), summary(condensed.as_ref()));
    let tracked = world
        .get::<Summarized>(agent)
        .map(|summarized| &summarized.0.tracked);
    assert!(
        tracked.is_some_and(|tracked| tracked
            .iter()
            .any(|set| set.name == "read-files" && set.values.contains("src/lib.rs"))),
        "{tracked:?}"
    );
}

#[test]
fn a_request_too_long_clears_old_outputs_then_summarizes() {
    let (mut app, wakes) = start(&MemoryStore::default());
    // Three reads of 15k tokens each: the oldest is past the 40k kept.
    let big = "x".repeat(60_000);
    let mut messages = vec![Message::user("Read the three files.")];
    for (id, path) in [("1", "a.rs"), ("2", "b.rs"), ("3", "c.rs")] {
        messages.extend(read(id, path, &big));
    }
    messages.push(Message::assistant("Read them."));
    let model = MockCompletionModel::from_stream_turns([
        too_long(),
        too_long(),
        reply("## Goal\nRead three files."),
        reply("done"),
    ]);
    let agent = agent(&mut app, &model, 1_000_000, messages);
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    ask(&mut app, &wakes, agent);
    assert_eq!(answer(&app).as_deref(), Some("done"));
    let sent = sent(&model);
    assert_eq!(
        sent.len(),
        4,
        "the request, cleared, a summary, summarized: {sent:?}"
    );
    let [first, cleared, summary, last] = sent.as_slice() else {
        return;
    };
    let placeholder = "[output cleared to fit the context window";
    assert!(!first.contains(placeholder));
    assert!(cleared.contains(placeholder), "{cleared}");
    assert!(
        summary.contains("You summarize a conversation"),
        "{summary}"
    );
    assert!(last.contains("## Goal"), "{last}");
    assert!(app.world().get::<Condensed>(agent).is_some());
}

#[test]
fn an_interrupt_during_the_summary_cancels_the_compaction() {
    let (mut app, _wakes) = start(&MemoryStore::default());
    let messages = vec![
        Message::user("What is 2 + 2?"),
        Message::assistant("4"),
        Message::user("And 3 + 3?"),
        Message::assistant("6"),
    ];
    let model = MockCompletionModel::from_stream_turns([reply("## Goal\nArithmetic.")]);
    let agent = agent(&mut app, &model, 1_000_000, messages);
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    let world = app.world_mut();
    world.trigger(RunCommand {
        entity: agent,
        line: "compact keep the sums".to_owned(),
    });
    world.flush();
    let focus = world
        .query::<&ModelRequest>()
        .iter(world)
        .map(|call| serde_json::to_string(&call.request).unwrap_or_default())
        .collect::<Vec<_>>();
    assert!(
        matches!(focus.as_slice(), [request] if request.contains("Additional focus: keep the sums")),
        "{focus:?}"
    );
    world.trigger(Interrupt { entity: agent });
    world.flush();
    for _ in 0..5 {
        app.update();
        std::thread::sleep(Duration::from_millis(20));
    }
    let world = app.world_mut();
    assert!(world.get::<Condensed>(agent).is_none());
    assert_eq!(world.query::<&Summarizing>().iter(world).count(), 0);
    assert!(
        matches!(
            world.resource::<Ended>().0.as_slice(),
            [TurnOutcome::Stopped]
        ),
        "{:?}",
        world.resource::<Ended>().0
    );
}
