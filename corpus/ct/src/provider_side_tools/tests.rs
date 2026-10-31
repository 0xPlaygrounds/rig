use std::collections::VecDeque;
use std::future::{self, Future};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use rig_core::catalog::Catalog;
use rig_core::http_client::{
    self, Bytes, DynHttpClient, HttpClientExt, LazyBody, MultipartForm, Request, Response,
    StatusCode, StreamingResponse,
};
use rig_core::providers::registry::ConnectOptions;
use rig_core::test_utils::MockCompletionModel;
use rig_core::wasm_compat::WasmCompatSend;
use rig_harness::settings::Settings;
use rig_harness_test_support::{
    Ended, MemoryStore, app_on, first_agent, log_ended, pass_wall_clock, reply, scripted_models,
};
use serde_json::json;

use super::*;

/// The model with server tools, reached through the fake Messages API.
const CLAUDE: &str = "anthropic/claude-haiku-4-5";
const CHANGELOG: &str = "https://github.com/hyperium/http/blob/master/CHANGELOG.md";

/// One streamed reply: its SSE chunks, the last maybe a cut connection.
type Script = Vec<Result<String, &'static str>>;

/// An HTTP client serving Anthropic's Messages API from scripted streams,
/// keeping each request body.
#[derive(Clone, Default)]
struct Fake(Arc<Mutex<(VecDeque<Script>, Vec<Value>)>>);

impl Fake {
    fn new(replies: impl IntoIterator<Item = Script>) -> Self {
        Self(Arc::new(Mutex::new((
            replies.into_iter().collect(),
            Vec::new(),
        ))))
    }

    fn sent(&self) -> Vec<Value> {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .1
            .clone()
    }
}

impl HttpClientExt for Fake {
    fn send<T, U>(
        &self,
        _: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        future::ready(Err(http_client::Error::NoHeaders))
    }

    fn send_multipart<U>(
        &self,
        _: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        future::ready(Err(http_client::Error::NoHeaders))
    }

    fn send_streaming<T>(
        &self,
        request: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        let body: Bytes = request.into_body().into();
        let script = {
            let mut api = self.0.lock().unwrap_or_else(PoisonError::into_inner);
            api.1
                .push(serde_json::from_slice(&body).unwrap_or_default());
            api.0.pop_front().unwrap_or_default()
        };
        let chunks = script.into_iter().map(|chunk| match chunk {
            Ok(text) => Ok(Bytes::from(text)),
            Err(why) => Err(http_client::Error::instance(std::io::Error::new(
                std::io::ErrorKind::ConnectionReset,
                why,
            ))),
        });
        let stream: http_client::BoxedStream = Box::pin(futures::stream::iter(chunks));
        let response = Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "text/event-stream")
            .body(stream);
        future::ready(response.map_err(http_client::Error::Protocol))
    }
}

fn event(data: &Value) -> Result<String, &'static str> {
    let kind = data.get("type").and_then(Value::as_str).unwrap_or_default();
    Ok(format!("event: {kind}\ndata: {data}\n\n"))
}

fn start() -> Result<String, &'static str> {
    event(&json!({"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-4-5",
        "content": [], "stop_reason": null, "usage": {"input_tokens": 10, "output_tokens": 1}}}))
}

fn block(index: usize, block: &Value) -> Result<String, &'static str> {
    event(&json!({"type": "content_block_start", "index": index, "content_block": block}))
}

fn delta(index: usize, delta: &Value) -> Result<String, &'static str> {
    event(&json!({"type": "content_block_delta", "index": index, "delta": delta}))
}

fn stop(index: usize) -> Result<String, &'static str> {
    event(&json!({"type": "content_block_stop", "index": index}))
}

fn text(index: usize, text: &str) -> Vec<Result<String, &'static str>> {
    vec![
        block(index, &json!({"type": "text", "text": ""})),
        delta(index, &json!({"type": "text_delta", "text": text})),
        stop(index),
    ]
}

fn end(reason: &str, searches: u64) -> Vec<Result<String, &'static str>> {
    vec![
        event(
            &json!({"type": "message_delta", "delta": {"stop_reason": reason},
            "usage": {"output_tokens": 20, "server_tool_use": {"web_search_requests": searches}}}),
        ),
        event(&json!({"type": "message_stop"})),
    ]
}

/// Text, a provider-run search streaming its query, its result, a cited
/// text and a local `read`; cut inside the search when `cut`.
fn searching(cut: bool) -> Script {
    let mut script = vec![start()];
    script.extend(text(0, "Let me check."));
    script.push(block(
        1,
        &json!({"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {}}),
    ));
    script.push(delta(
        1,
        &json!({"type": "input_json_delta", "partial_json": "{\"query\": \"http crate"}),
    ));
    if cut {
        script.push(Err("connection reset inside the search"));
        return script;
    }
    script.push(delta(
        1,
        &json!({"type": "input_json_delta", "partial_json": " timeouts\"}"}),
    ));
    script.push(stop(1));
    let result = json!({"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1",
        "content": [{"type": "web_search_result", "url": CHANGELOG, "title": "http CHANGELOG",
            "encrypted_content": "opaque-page", "page_age": null}]});
    script.extend([block(2, &result), stop(2)]);
    script.push(block(3, &json!({"type": "text", "text": ""})));
    let citation = json!({"type": "web_search_result_location", "url": CHANGELOG,
        "title": "http CHANGELOG", "cited_text": "1.3.0: no timeout changes",
        "encrypted_index": "opaque-index"});
    script.push(delta(
        3,
        &json!({"type": "citations_delta", "citation": citation}),
    ));
    script.push(delta(
        3,
        &json!({"type": "text_delta", "text": "The latest release did not change timeouts."}),
    ));
    script.push(stop(3));
    script.push(block(
        4,
        &json!({"type": "tool_use", "id": "toolu_1", "name": "read", "input": {}}),
    ));
    script.push(delta(
        4,
        &json!({"type": "input_json_delta", "partial_json": "{\"path\": \"Cargo.lock\"}"}),
    ));
    script.push(stop(4));
    script.extend(end("tool_use", 1));
    script
}

fn answering(answer: &str) -> Script {
    let mut script = vec![start()];
    script.extend(text(0, answer));
    script.extend(end("end_turn", 0));
    script
}

/// The names of the tool calls the kernel was asked to run, in order.
#[derive(Resource, Default)]
struct Ran(Vec<String>);

fn log_ran(added: On<Add<ToolRequest>>, calls: Query<&ToolRequest>, mut ran: ResMut<Ran>) {
    if let Ok(request) = calls.get(added.entity) {
        ran.0.push(request.call.function.name.as_str().to_owned());
    }
}

/// A local `read`.
fn read(called: On<ToolCalled<Value>>, mut calls: Query<&mut Decision, ToolCalls>) {
    if let Ok(mut call) = calls.get_mut(called.call) {
        let lock = "[[package]]\nname = \"http\"\nversion = \"1.3.0\"";
        call.decide(Decider::plugin("test"), ToolResult::success(lock.into()));
    }
}

fn app(fake: &Fake, other: &MockCompletionModel) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    let settings = format!(
        "[provider_tools.models.\"{CLAUDE}\"]\nserver_tools = [\"web_search\"]\n\
         [provider_tools.models.\"{CLAUDE}\".options.web_search]\nmax_uses = 3\n\
         allowed_domains = [\"github.com\"]"
    );
    let Models(mut connector): Models = scripted_models([("other/model", other)]);
    if let Ok(claude) = Catalog::builtin().resolve(CLAUDE) {
        connector.catalog_mut().insert(claude.spec.clone());
        let options = ConnectOptions::new().api_key("test-key");
        connector.provide(claude.spec.provider, options);
    }
    let fake = fake.clone();
    connector.wrap_http(move |_| DynHttpClient::new(fake.clone()));
    app.insert_resource(Settings::from_toml(&settings))
        .insert_resource(Models(connector))
        .add_plugins((AgentPlugin, SessionLogPlugin, ProviderToolsPlugin))
        .add_open_tool("read", "Read a file.", ToolOptions::default(), read)
        .init_resource::<Ran>()
        .add_observer(log_ran)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let model = CLAUDE.to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
    });
    (app, wakes, agent)
}

/// Sends `text` and runs frames, each after any retry's wait passed,
/// until the turn ended.
fn turn(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str) {
    let ended = app.world().resource::<Ended>().0.len();
    let message = Deliver::new(agent, text, DeliveryMode::Queue);
    app.world_mut().trigger(message);
    let started = std::time::Instant::now();
    while started.elapsed() < Duration::from_secs(20)
        && app.world().resource::<Ended>().0.len() == ended
    {
        pass_wall_clock(app.world_mut(), Duration::from_secs(60));
        app.update();
        wakes.recv_timeout(Duration::from_millis(20)).ok();
    }
}

fn kinds(content: &[AssistantContent]) -> Vec<String> {
    content
        .iter()
        .map(|block| match block {
            AssistantContent::Text(_) => "text".to_owned(),
            AssistantContent::ToolCall(call) => format!("call:{}", call.function.name.as_str()),
            AssistantContent::Opaque(opaque) => opaque.kind().unwrap_or_default().to_owned(),
            other => format!("{other:?}"),
        })
        .collect()
}

#[test]
fn provider_run_search_is_kept_replayed_natively_and_sent_as_text_elsewhere() {
    let fake = Fake::new([
        searching(true),
        searching(false),
        answering("Timeouts are unchanged in http 1.3.0 [1]."),
        answering("You're welcome."),
    ]);
    let other = MockCompletionModel::from_stream_turns([reply("No timeout changes.")]);
    let (mut app, wakes, agent) = app(&fake, &other);
    turn(
        &mut app,
        &wakes,
        agent,
        "Check whether the latest release of the HTTP crate changed timeouts.",
    );

    // Cut inside the search, the reply was sent again as a whole; only the
    // local `read` reached the kernel, and it ran.
    let ended = format!("{:?}", app.world().resource::<Ended>().0);
    assert!(ended.contains("Timeouts are unchanged"), "{ended}");
    assert_eq!(fake.sent().len(), 3, "the cut reply was retried once");
    assert_eq!(app.world().resource::<Ran>().0, ["read"]);

    // The conversation keeps the provider-run call and its result in stream
    // order, once, with the cited text's citation.
    let world = app.world_mut();
    let conversation = world.query::<&Conversation>().get(world, agent);
    assert!(conversation.is_ok(), "the agent has a conversation");
    let Ok(conversation) = conversation else {
        return;
    };
    let replies: Vec<&AssistantMessage> = conversation
        .messages()
        .iter()
        .filter_map(|message| match &**message {
            ChatMessage::Assistant(reply) => Some(reply),
            _ => None,
        })
        .collect();
    let first = replies.first().map(|reply| kinds(&reply.content));
    let expected = [
        "text",
        "server_tool_use",
        "web_search_tool_result",
        "text",
        "call:read",
    ];
    assert_eq!(first.as_deref(), Some(&expected.map(str::to_owned)[..]));
    let uses = replies
        .iter()
        .flat_map(|reply| &reply.content)
        .filter(|block| run(block).is_some());
    assert_eq!(uses.count(), 1, "stored once");

    // The transcript shows the search as provider-run, and the sources.
    let notes: Vec<Note> = conversation
        .asides()
        .filter_map(|(_, aside)| Note::of(aside))
        .collect();
    let titles: Vec<&str> = notes.iter().map(|note| note.title.as_str()).collect();
    assert_eq!(
        titles,
        [
            "web_search ran on provider: http crate timeouts",
            "Sources (1)"
        ]
    );
    let shown = notes
        .iter()
        .map(|note| note.text.clone())
        .collect::<Vec<_>>();
    assert!(
        shown.iter().all(|text| text.contains(CHANGELOG)),
        "{shown:?}"
    );
    let usage = app.world().resource::<ProviderToolUsage>();
    assert_eq!(usage.web_searches, 1, "the provider-reported count");

    // The next turn replays the search to the same provider natively, with
    // its opaque ids, and enables the configured search with its options.
    turn(&mut app, &wakes, agent, "Thanks.");
    let sent = fake.sent();
    let last = sent.last().map(Value::to_string).unwrap_or_default();
    for native in [
        "\"type\":\"server_tool_use\"",
        "\"id\":\"srvtoolu_1\"",
        "\"type\":\"web_search_tool_result\"",
        "\"encrypted_content\":\"opaque-page\"",
        "\"encrypted_index\":\"opaque-index\"",
        "\"type\":\"web_search_20250305\"",
        "\"max_uses\":3",
    ] {
        assert!(last.contains(native), "{native} not in {last}");
    }

    // Another provider gets the search as text and no provider-run call.
    app.world_mut().trigger(SetModel {
        entity: agent,
        model: "other/model".to_owned(),
    });
    turn(&mut app, &wakes, agent, "Summarize what you found.");
    let requests = other.requests();
    let history = requests
        .last()
        .map(|request| request.chat_history.clone())
        .unwrap_or_default();
    let blocks: Vec<&AssistantContent> = history
        .iter()
        .filter_map(|message| match &**message {
            ChatMessage::Assistant(reply) => Some(&reply.content),
            _ => None,
        })
        .flatten()
        .collect();
    assert!(
        !blocks
            .iter()
            .any(|block| matches!(block, AssistantContent::Opaque(_))),
        "{blocks:?}"
    );
    let rendered = format!("[web search: http crate timeouts] → http CHANGELOG <{CHANGELOG}>");
    assert!(
        blocks
            .iter()
            .any(|block| matches!(block, AssistantContent::Text(text) if text.text == rendered)),
        "{blocks:?}"
    );
    assert!(!format!("{:?}", requests.last()).contains("web_search_20250305"));
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ProviderToolsPlugin);
}
