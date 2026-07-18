//! OpenAI-compatible Responses API reasoning-content roundtrip regression tests.
//!
//! This covers providers that return non-streaming reasoning items with a
//! `content` array containing `reasoning_text`, as llama.cpp does.

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
use rig::completion::{Chat, Message};
use rig::prelude::AgentClientExt;
use rig::providers::openai;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use crate::cassettes::{self, ProviderCassette};
use crate::reasoning::{self, WeatherTool};

const SCENARIO: &str = "openai_compatible/reasoning_content_tool_roundtrip";
const REASONING_TEXT: &str =
    "The user asked for current weather, so I need to call get_weather before answering.";

#[tokio::test]
async fn nonstreaming_reasoning_content_tool_roundtrip() {
    with_local_reasoning_content_cassette(
        "openai_compatible/reasoning_content_tool_roundtrip",
        |client| async move {
            let call_count = Arc::new(AtomicUsize::new(0));
            let agent = client
                .agent("llama-cpp-reasoning-model")
                .preamble(reasoning::TOOL_SYSTEM_PROMPT)
                .tool(WeatherTool::new(call_count.clone()))
                .additional_params(json!({
                    "reasoning": { "effort": "medium" }
                }))
                .default_max_turns(2)
                .build();

            let result = agent
                .chat(reasoning::TOOL_USER_PROMPT, &mut Vec::<Message>::new())
                .await
                .expect("OpenAI-compatible provider should accept replayed reasoning content");

            reasoning::assert_nonstreaming_universal(&result, &call_count, "openai-compatible");
        },
    )
    .await;

    assert_cassette_preserves_reasoning_content(SCENARIO);
}

async fn with_local_reasoning_content_cassette<F, Fut>(scenario: &'static str, test_body: F)
where
    F: FnOnce(openai::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let server = LocalReasoningContentServer::start().await;
    let cassette = ProviderCassette::start("openai", scenario, &server.base_url()).await;
    let client = openai::Client::builder()
        .api_key("dummy-openai-compatible-key")
        .base_url(cassette.base_url())
        .build()
        .expect("OpenAI-compatible cassette client should build");

    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

struct LocalReasoningContentServer {
    addr: SocketAddr,
    shutdown: Option<oneshot::Sender<()>>,
    task: JoinHandle<()>,
}

impl LocalReasoningContentServer {
    async fn start() -> Self {
        let state = Arc::new(LocalState {
            request_count: AtomicUsize::new(0),
        });
        let app = Router::new()
            .route("/v1/responses", post(local_responses_api))
            .with_state(state);
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("local OpenAI-compatible server should bind");
        let addr = listener
            .local_addr()
            .expect("local OpenAI-compatible server address should be available");
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let task = tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = shutdown_rx.await;
                })
                .await
                .expect("local OpenAI-compatible server should run");
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

impl Drop for LocalReasoningContentServer {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        self.task.abort();
    }
}

struct LocalState {
    request_count: AtomicUsize,
}

async fn local_responses_api(
    State(state): State<Arc<LocalState>>,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    match state.request_count.fetch_add(1, Ordering::SeqCst) {
        0 => first_tool_call_response(),
        1 => continuation_response(body),
        _ => (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": {
                    "message": "unexpected extra request",
                    "type": "invalid_request_error"
                }
            })),
        ),
    }
}

fn first_tool_call_response() -> (StatusCode, Json<Value>) {
    (
        StatusCode::OK,
        Json(json!({
            "id": "resp_llamacpp_1",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "llama-cpp-reasoning-model",
            "output": [
                {
                    "id": "rs_llamacpp_1",
                    "summary": [],
                    "type": "reasoning",
                    "content": [
                        {
                            "text": REASONING_TEXT,
                            "type": "reasoning_text"
                        }
                    ],
                    "encrypted_content": "",
                    "status": "completed"
                },
                {
                    "arguments": "{\"city\":\"Tokyo, Japan\"}",
                    "call_id": "call_llamacpp_1",
                    "id": "fc_llamacpp_1",
                    "name": "get_weather",
                    "status": "completed",
                    "type": "function_call"
                }
            ],
            "usage": {
                "input_tokens": 12,
                "output_tokens": 8,
                "total_tokens": 20
            }
        })),
    )
}

fn continuation_response(body: Value) -> (StatusCode, Json<Value>) {
    if !request_preserves_reasoning_content(&body) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": {
                    "code": 400,
                    "message": "item['content'] is not an array",
                    "type": "invalid_request_error"
                }
            })),
        );
    }

    (
        StatusCode::OK,
        Json(json!({
            "id": "resp_llamacpp_2",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "llama-cpp-reasoning-model",
            "output": [
                {
                    "id": "msg_llamacpp_1",
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                    "content": [
                        {
                            "annotations": [],
                            "text": "Tokyo, Japan is sunny at 72F (22C). Pack sunscreen; an umbrella is not needed based on the current weather.",
                            "type": "output_text"
                        }
                    ]
                }
            ],
            "usage": {
                "input_tokens": 25,
                "output_tokens": 18,
                "total_tokens": 43
            }
        })),
    )
}

fn request_preserves_reasoning_content(body: &Value) -> bool {
    body.get("input")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("reasoning"))
        .any(reasoning_item_preserves_content)
}

fn reasoning_item_preserves_content(item: &Value) -> bool {
    item.get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .any(|content| {
            content.get("type").and_then(Value::as_str) == Some("reasoning_text")
                && content.get("text").and_then(Value::as_str) == Some(REASONING_TEXT)
        })
}

fn assert_cassette_preserves_reasoning_content(scenario: &str) {
    let cassette_path = cassettes::cassette_path("openai", scenario);
    let contents = std::fs::read_to_string(&cassette_path).unwrap_or_else(|error| {
        panic!(
            "provider cassette {} should be readable after recording: {error}",
            cassette_path.display()
        )
    });
    let interactions = serde_yaml::Deserializer::from_str(&contents)
        .map(|document| serde_yaml::Value::deserialize(document).expect("cassette interaction"))
        .collect::<Vec<_>>();

    assert!(
        interactions.iter().any(|interaction| {
            let Some(body) = interaction
                .get("when")
                .and_then(|when| when.get("body"))
                .and_then(serde_yaml::Value::as_str)
            else {
                return false;
            };
            let Ok(body) = serde_json::from_str::<Value>(body) else {
                return false;
            };
            request_preserves_reasoning_content(&body)
        }),
        "expected cassette {} to contain a continuation request preserving reasoning.content",
        cassette_path.display()
    );
}
