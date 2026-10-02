//! A chat-completions tool turn whose assistant message carries both
//! `reasoning` and `reasoning_content`, as a gateway relaying a
//! `reasoning_content` upstream behind a `reasoning` surface sends it.
//!
//! Not live traffic: the fixture is recorded from a local server that
//! answers in that shape. No configured endpoint was found sending both
//! keys. The two keys carry different text: rig reads `reasoning_content`
//! as the turn's reasoning and, as pi does, sends it back under that key
//! alone; the server refuses a continuation that does not echo
//! `reasoning_content` back, as reasoning endpoints do.

use std::future::Future;
use std::net::SocketAddr;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::{Json, Router, routing::post};
use futures::FutureExt;
use rig::completion::Message;
use rig::providers::openai::{OpenAIConfig, Route};
use rig_test_support::cassette_models::OpenAiModels;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use crate::cassettes::ProviderCassette;
use crate::reasoning::{self, WeatherTool};

const SCENARIO: &str = "openai_compatible/dual_reasoning_keys_tool_roundtrip";
const MODEL: &str = "gateway-relayed-reasoning-model";
const UPSTREAM_REASONING: &str =
    "The user wants Tokyo's current weather, so I will call get_weather for Tokyo, Japan.";
const SURFACE_REASONING: &str = "Calling get_weather for Tokyo.";

/// The first reply decodes with its tool call whole and its reasoning taken
/// from `reasoning_content`, and the continuation sends that reasoning back
/// under that key alone.
#[tokio::test]
async fn dual_reasoning_keys_tool_roundtrip() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let invocations = call_count.clone();
    with_local_dual_reasoning_keys_cassette(
        "openai_compatible/dual_reasoning_keys_tool_roundtrip",
        |client| async move {
            let agent = rig::AgentBuilder::new(client.completion(MODEL))
                .preamble(reasoning::TOOL_SYSTEM_PROMPT)
                .tool(WeatherTool::new(invocations))
                .default_max_turns(2)
                .build();
            let result = agent
                .chat(reasoning::TOOL_USER_PROMPT, &mut Vec::<Message>::new())
                .await
                .expect("a reply carrying both reasoning keys decodes")
                .output();
            reasoning::assert_nonstreaming_universal(&result, &call_count, "dual-reasoning-keys");
        },
    )
    .await;

    let turns = crate::cassettes::recorded_json_turns("openai", SCENARIO);
    assert_eq!(turns.len(), 2, "the tool call, then the continuation");
    let replied = &turns[0].1["choices"][0]["message"];
    let (Some(upstream), Some(_surface)) = (
        replied["reasoning_content"].as_str(),
        replied["reasoning"].as_str(),
    ) else {
        panic!("the recorded reply carries both reasoning keys: {replied}");
    };
    let echoed = turns[1].0["messages"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|message| message["role"] == "assistant")
        .expect("the continuation replays the assistant turn");
    // As pi does, the reasoning goes back under the first key that carried
    // it, once.
    assert_eq!(
        (
            echoed["reasoning_content"].as_str(),
            echoed["reasoning"].as_str()
        ),
        (Some(upstream), None),
        "the reasoning goes back under the key it arrived in: {echoed}"
    );
    let call = |message: &serde_json::Value| {
        let call = &message["tool_calls"][0];
        let arguments: serde_json::Value =
            serde_json::from_str(call["function"]["arguments"].as_str().unwrap_or("{}"))
                .unwrap_or_default();
        (
            call["id"].clone(),
            call["function"]["name"].clone(),
            arguments,
        )
    };
    assert_eq!(call(echoed), call(replied), "the tool call is replayed");
}

async fn with_local_dual_reasoning_keys_cassette<F, Fut>(scenario: &'static str, test_body: F)
where
    F: FnOnce(OpenAiModels) -> Fut,
    Fut: Future<Output = ()>,
{
    let server = LocalGatewayServer::start().await;
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "openai",
        scenario,
        &server.base_url(),
    )
    .await;
    let client = OpenAIConfig::new("dummy-openai-compatible-key")
        .with_base_url(cassette.base_url())
        .with_route(Route::Chat);
    let result = AssertUnwindSafe(test_body(OpenAiModels::new(
        client,
        rig::rig_reqwest::shared(),
    )))
    .catch_unwind()
    .await;
    cassette.finish_after_test(result).await;
}

struct LocalGatewayServer {
    addr: SocketAddr,
    shutdown: Option<oneshot::Sender<()>>,
    task: JoinHandle<()>,
}

impl LocalGatewayServer {
    async fn start() -> Self {
        let app = Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(Arc::new(AtomicUsize::new(0)));
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("the local gateway binds");
        let addr = listener
            .local_addr()
            .expect("the local gateway has an address");
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let task = tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = shutdown_rx.await;
                })
                .await
                .expect("the local gateway serves");
        });
        Self {
            addr,
            shutdown: Some(shutdown_tx),
            task,
        }
    }

    fn base_url(&self) -> String {
        format!("http://{}/v1", self.addr)
    }
}

impl Drop for LocalGatewayServer {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        self.task.abort();
    }
}

async fn chat_completions(
    State(requests): State<Arc<AtomicUsize>>,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    match requests.fetch_add(1, Ordering::SeqCst) {
        0 => (StatusCode::OK, Json(tool_call_reply())),
        1 if echoes_reasoning_content(&body) => (StatusCode::OK, Json(final_reply())),
        1 => bad_request(
            "The reasoning_content in the thinking mode must be passed back to the API.",
        ),
        _ => bad_request("unexpected extra request"),
    }
}

fn bad_request(message: &str) -> (StatusCode, Json<Value>) {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({ "error": { "message": message, "type": "invalid_request_error" } })),
    )
}

fn tool_call_reply() -> Value {
    json!({
        "id": "gen_dual_1",
        "object": "chat.completion",
        "created": 0,
        "model": MODEL,
        "choices": [{
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": null,
                "reasoning": SURFACE_REASONING,
                "reasoning_content": UPSTREAM_REASONING,
                "tool_calls": [{
                    "id": "call_dual_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"city\":\"Tokyo, Japan\"}"
                    }
                }]
            }
        }],
        "usage": { "prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20 }
    })
}

fn final_reply() -> Value {
    json!({
        "id": "gen_dual_2",
        "object": "chat.completion",
        "created": 0,
        "model": MODEL,
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "Tokyo, Japan is sunny at 72F (22C). Pack sunscreen; an umbrella is not needed."
            }
        }],
        "usage": { "prompt_tokens": 25, "completion_tokens": 18, "total_tokens": 43 }
    })
}

/// Whether the continuation replays the assistant turn with its reasoning
/// under `reasoning_content`.
fn echoes_reasoning_content(body: &Value) -> bool {
    body["messages"]
        .as_array()
        .into_iter()
        .flatten()
        .any(|message| {
            message["role"] == "assistant" && message["reasoning_content"] == UPSTREAM_REASONING
        })
}
