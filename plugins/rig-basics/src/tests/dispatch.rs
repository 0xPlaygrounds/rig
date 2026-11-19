//! A model's reply as the turn runs its tool calls, through real provider wires: a call the
//! output limit cut before its arguments never runs, and a call another took the stream index of
//! does.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::Receiver;

use rig_core::message::ToolName;
use rig_core::tool::{Tool, ToolContext, ToolExecutionError};
use rig_harness::prelude::*;
use rig_harness_test_support::{Ended, MemoryStore, app_on, log_ended, run_until};

/// The kernel on `store`, after the first frame restored the session, with its loop's wakes.
fn start(store: &MemoryStore) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    app.set_error_handler(bevy_ecs::error::warn)
        .add_plugins((AgentPlugin, SessionLogPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    (app, wakes)
}

/// [`start`] with the compaction plugin loaded as the agent loads it, whose recovery from a reply
/// cut short must leave a reply with tool calls to the scheduler.
fn start_with_compaction(store: &MemoryStore) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    app.set_error_handler(bevy_ecs::error::warn)
        .add_plugins((AgentPlugin, SessionLogPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    rig_harness::load::<crate::compaction::CompactionPlugin>(
        &mut app,
        "rig-basics",
        "path plugins/rig-basics",
    );
    app.finish();
    app.update();
    (app, wakes)
}

/// Asks `agent` a question, then runs frames until a turn ended.
fn ask(app: &mut App, wakes: &Receiver<()>, agent: Entity) {
    app.world_mut().trigger(Deliver::new(
        agent,
        "And the next question?",
        DeliveryMode::Steer,
    ));
    run_until(app, wakes, |world| !world.resource::<Ended>().0.is_empty());
}

/// The answer the one turn of `app` ended with.
fn answer(app: &App) -> Option<String> {
    match app.world().resource::<Ended>().0.as_slice() {
        [TurnOutcome::Answered(message)] => message.final_answer(),
        _ => None,
    }
}

/// A tool of the test that takes no arguments and counts its runs.
#[derive(Clone, Default)]
struct Ping(Arc<AtomicUsize>);

impl Tool for Ping {
    const NAME: &'static str = "ping";
    type Args = serde_json::Value;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Ping.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {}})
    }

    async fn call(
        &self,
        _: &mut ToolContext,
        _: serde_json::Value,
    ) -> Result<String, ToolExecutionError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok("pong".to_owned())
    }
}

/// A reply the output limit cut right after it named a no-argument tool,
/// its argument text still blank, on the Responses wire (the provider marks
/// the item incomplete) and on a chat wire: the tool never runs, and the
/// model reads that its reply reached the limit.
#[test]
fn a_call_cut_before_its_arguments_never_runs() {
    let sse = |events: Vec<(&str, serde_json::Value)>| {
        let body: String = events
            .iter()
            .map(|(event, data)| format!("event: {event}\ndata: {data}\n\n"))
            .collect();
        rig_core::test_utils::MockHttpResponse::success_typed(body, "text/event-stream")
    };
    let item = serde_json::json!({"type": "function_call", "id": "fc_1", "call_id": "call_1",
        "name": "ping", "arguments": "", "status": "incomplete"});
    let responses = sse(vec![
        (
            "response.output_item.done",
            serde_json::json!({"type": "response.output_item.done", "output_index": 0,
                "item": item}),
        ),
        (
            "response.incomplete",
            serde_json::json!({"type": "response.incomplete", "response": {"id": "resp_1",
                "object": "response", "status": "incomplete", "model": "gpt-5.4",
                "incomplete_details": {"reason": "max_output_tokens"}, "output": [item]}}),
        ),
    ]);
    let chat = |delta: serde_json::Value, finish: &str| {
        let chunk = serde_json::json!({"id": "c", "object": "chat.completion.chunk",
            "model": "m", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]});
        let body = format!("data: {chunk}\n\ndata: [DONE]\n\n");
        rig_core::test_utils::MockHttpResponse::success_typed(body, "text/event-stream")
    };
    let chat_cut = chat(
        serde_json::json!({"role": "assistant", "tool_calls": [{"index": 0, "id": "call_1",
            "type": "function", "function": {"name": "ping", "arguments": ""}}]}),
        "length",
    );
    let chat_done = chat(
        serde_json::json!({"role": "assistant", "content": "done"}),
        "stop",
    );
    for (dialect, model, cut, done) in [
        (
            &rig_core::providers::openai::wire::OPENAI,
            "gpt-5.4",
            responses,
            streamed(reasoned(2, "done"), None),
        ),
        (
            &rig_core::providers::openai::wire::DEEPSEEK,
            "deepseek-v4-flash",
            chat_cut,
            chat_done,
        ),
    ] {
        let (mut app, wakes) = start(&MemoryStore::default());
        let ping = Ping::default();
        app.add_tool(ping.clone());
        let http = rig_core::test_utils::SequencedHttpClient::new([cut, done]);
        let agent = on_responses(&mut app, (dialect, model), 1_000_000, &http);
        assert!(agent.is_some(), "an agent on {}", dialect.name);
        let Some(agent) = agent else { return };
        ask(&mut app, &wakes, agent);
        assert_eq!(answer(&app).as_deref(), Some("done"), "{}", dialect.name);
        assert_eq!(ping.0.load(Ordering::SeqCst), 0, "{}", dialect.name);
        let bodies = bodies(&http);
        assert!(
            matches!(bodies.as_slice(), [_, next]
                if next.contains("Not run: your reply reached the output limit before `ping`")),
            "{}: {bodies:?}",
            dialect.name
        );
    }
}

/// A call whose stream index another call takes over, in a reply that ends
/// with `tool_calls` on DeepSeek's chat wire: no limit cut it, so the turn
/// goes on, the call that took the index runs, and the first is answered as
/// malformed.
#[test]
fn a_call_after_a_takeover_in_a_finished_reply_runs() {
    let chunk = |delta: serde_json::Value, finish: Option<&str>| {
        serde_json::json!({"id": "c", "object": "chat.completion.chunk", "model": "m",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
    };
    let sse = |chunks: Vec<serde_json::Value>| {
        let mut body: String = chunks
            .iter()
            .map(|chunk| format!("data: {chunk}\n\n"))
            .collect();
        body.push_str("data: [DONE]\n\n");
        rig_core::test_utils::MockHttpResponse::success_typed(body, "text/event-stream")
    };
    let calls = sse(vec![
        chunk(
            serde_json::json!({"role": "assistant", "tool_calls": [{"index": 0, "id": "call_1",
                "type": "function", "function": {"name": "shell", "arguments": "{\"command\": \"ec"}}]}),
            None,
        ),
        chunk(
            serde_json::json!({"tool_calls": [{"index": 0, "id": "call_2", "type": "function",
                "function": {"name": "ping", "arguments": "{}"}}]}),
            None,
        ),
        chunk(serde_json::json!({}), Some("tool_calls")),
    ]);
    let done = sse(vec![chunk(
        serde_json::json!({"role": "assistant", "content": "done"}),
        Some("stop"),
    )]);
    let (mut app, wakes) = start(&MemoryStore::default());
    let ping = Ping::default();
    app.add_tool(ping.clone());
    let http = rig_core::test_utils::SequencedHttpClient::new([calls, done]);
    let deepseek = &rig_core::providers::openai::wire::DEEPSEEK;
    let agent = on_responses(&mut app, (deepseek, "deepseek-v4-flash"), 1_000_000, &http);
    assert!(agent.is_some(), "an agent on the DeepSeek chat wire");
    let Some(agent) = agent else { return };
    ask(&mut app, &wakes, agent);
    assert_eq!(answer(&app).as_deref(), Some("done"));
    assert_eq!(
        ping.0.load(Ordering::SeqCst),
        1,
        "the call that took the index runs"
    );
}

/// A reply on a real wire, DeepSeek's chat stream, that the output limit cut in the middle of a
/// call's arguments: the decoder closes the call unfinished, so it is answered as cut off and never
/// run, with compaction loaded, which leaves a reply with tool calls to the scheduler.
#[test]
fn a_call_a_real_wire_cut_at_the_output_limit_is_answered_so() {
    let (mut app, wakes) = start_with_compaction(&MemoryStore::default());
    app.add_tool(rig_tools::Read::default());
    let chunk = |delta: serde_json::Value, finish: Option<&str>| {
        serde_json::json!({"id": "c", "object": "chat.completion.chunk", "model": "m",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
    };
    let sse = |chunks: Vec<serde_json::Value>| {
        let mut body: String = chunks
            .iter()
            .map(|chunk| format!("data: {chunk}\n\n"))
            .collect();
        body.push_str("data: [DONE]\n\n");
        rig_core::test_utils::MockHttpResponse::success_typed(body, "text/event-stream")
    };
    let cut = chunk(
        serde_json::json!({"role": "assistant", "tool_calls": [{"index": 0, "id": "call_1",
            "type": "function", "function": {"name": "read", "arguments": "{\"path\": \"Cargo.to"}}]}),
        Some("length"),
    );
    let done = chunk(
        serde_json::json!({"role": "assistant", "content": "done"}),
        Some("stop"),
    );
    let http = rig_core::test_utils::SequencedHttpClient::new([sse(vec![cut]), sse(vec![done])]);
    let deepseek = &rig_core::providers::openai::wire::DEEPSEEK;
    let agent = on_responses(&mut app, (deepseek, "deepseek-v4-flash"), 1_000_000, &http);
    assert!(agent.is_some(), "an agent on the DeepSeek chat wire");
    let Some(agent) = agent else { return };
    ask(&mut app, &wakes, agent);
    assert_eq!(answer(&app).as_deref(), Some("done"));
    let bodies = bodies(&http);
    assert!(
        matches!(bodies.as_slice(), [_, next]
            if next.contains("Not run: your reply reached the output limit before `read`")),
        "{bodies:?}"
    );
}

/// A Responses reply streamed with `items` as its output, reporting
/// `input` tokens read, or no usage with `None`, so the next request is
/// measured by the estimate alone.
fn streamed(
    items: Vec<serde_json::Value>,
    input: Option<u64>,
) -> rig_core::test_utils::MockHttpResponse {
    let mut body = String::new();
    for (index, item) in items.iter().enumerate() {
        let event = serde_json::json!({
            "type": "response.output_item.done", "output_index": index, "item": item,
        });
        body.push_str(&format!(
            "event: response.output_item.done\ndata: {event}\n\n"
        ));
    }
    let mut response = serde_json::json!({
        "id": "resp_1", "object": "response", "status": "completed", "model": "gpt-5.4",
        "output": items,
    });
    if let (Some(input), Some(response)) = (input, response.as_object_mut()) {
        let usage = serde_json::json!({
            "input_tokens": input, "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 5, "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": input + 5,
        });
        response.insert("usage".to_owned(), usage);
    }
    let done = serde_json::json!({"type": "response.completed", "response": response});
    body.push_str(&format!("event: response.completed\ndata: {done}\n\n"));
    rig_core::test_utils::MockHttpResponse::success_typed(body, "text/event-stream")
}

/// A reply's reasoning item `n`, with its ciphertext, and its text.
fn reasoned(n: usize, text: &str) -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({
            "type": "reasoning", "id": format!("rs_{n}"), "summary": [], "content": [],
            "encrypted_content": format!("ciphertext-{n}"),
        }),
        serde_json::json!({
            "type": "message", "id": format!("msg_{n}"), "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }),
    ]
}

/// The first agent of `app` on the catalog's `model` with a `window`-token
/// context window, through the real Responses wire of `dialect` over
/// `http`, a scripted transport, offered `read` and `shell` (stand-ins
/// for those the app does not have).
fn on_responses(
    app: &mut App,
    (dialect, model): (&rig_core::providers::openai::wire::Dialect, &str),
    window: u32,
    http: &rig_core::test_utils::SequencedHttpClient,
) -> Option<Entity> {
    for name in ["read", "shell"] {
        let world = app.world_mut();
        if world
            .query::<&ToolDef>()
            .iter(world)
            .any(|def| def.0.name.as_str() == name)
        {
            continue;
        }
        let definition = rig_core::completion::ToolDefinition {
            name: ToolName::new(name).ok()?,
            description: "A tool.".to_owned(),
            parameters: serde_json::json!({"type": "object"}),
            annotations: Default::default(),
        };
        app.world_mut().spawn(ToolDef(definition.into()));
    }
    let served = rig_core::providers::openai::OpenAIConfig::with_key(dialect, "key")
        .connect(http.clone())
        .completion(model);
    let spec = rig_core::catalog::Catalog::builtin()
        .resolve(&format!("{}/{model}", dialect.name))
        .ok()?
        .shared();
    let spec = Arc::new(spec.as_ref().clone().with_context_window(window));
    let handler = rig_core::serve::adapters::ModelAdapter::<rig_core::operation::Completion>::new(
        spec.reference(),
        served,
    );
    let connection = Connection {
        spec,
        handler: rig_ecs::effects::Handler(rig_core::serve::ErasedHandler::new(handler)),
    };
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next()?;
    world.entity_mut(agent).insert(connection);
    Some(agent)
}

/// The request bodies `http` got.
fn bodies(http: &rig_core::test_utils::SequencedHttpClient) -> Vec<String> {
    http.requests()
        .iter()
        .map(|request| String::from_utf8_lossy(&request.body).into_owned())
        .collect()
}
