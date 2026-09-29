//! An agent runs over the OpenAI Responses websocket transport like over
//! any other model: its tool loop sends each turn on the one connection, and
//! with chaining off each turn carries the whole conversation.

#![cfg(not(target_arch = "wasm32"))]
#![allow(clippy::expect_used, clippy::indexing_slicing)]

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use rig_agent::AgentBuilder;
use rig_agent::tool::{Tool, ToolContext, ToolExecutionError};
use rig_core::driver::Model;
use rig_core::http_client;
use rig_core::providers::openai::OpenAIConfig;
use rig_core::providers::openai::responses_api::websocket::{ResponsesSocket, ResponsesWebSocket};
use rig_core::test_utils::RecordingHttpClient;
use rig_core::wasm_compat::WasmBoxedFuture;
use rig_core::ws_client::{CloseFrame, Frame, WebSocketConnection};
use serde_json::json;

/// Server events per turn, released by each `response.create`; what the
/// transport wrote.
#[derive(Clone, Default)]
struct Script(Arc<Mutex<(VecDeque<Vec<String>>, VecDeque<String>, Vec<String>)>>);

impl Script {
    fn turns(turns: Vec<Vec<String>>) -> Self {
        Self(Arc::new(Mutex::new((
            turns.into(),
            VecDeque::new(),
            Vec::new(),
        ))))
    }

    fn sent(&self) -> Vec<serde_json::Value> {
        self.0
            .lock()
            .expect("script")
            .2
            .iter()
            .map(|text| serde_json::from_str(text).expect("the transport sends JSON"))
            .collect()
    }
}

impl WebSocketConnection for Script {
    fn send(&mut self, frame: Frame) -> WasmBoxedFuture<'_, http_client::Result<()>> {
        let mut state = self.0.lock().expect("script");
        if let Frame::Text(text) = frame {
            state.2.push(text);
        }
        if let Some(turn) = state.0.pop_front() {
            state.1.extend(turn);
        }
        Box::pin(std::future::ready(Ok(())))
    }

    fn recv(&mut self) -> WasmBoxedFuture<'_, http_client::Result<Option<Frame>>> {
        let next = self.0.lock().expect("script").1.pop_front();
        Box::pin(std::future::ready(Ok(next.map(Frame::Text))))
    }

    fn close(
        &mut self,
        _frame: Option<CloseFrame>,
    ) -> WasmBoxedFuture<'_, http_client::Result<()>> {
        Box::pin(std::future::ready(Ok(())))
    }
}

fn completed(id: &str, output: serde_json::Value) -> String {
    json!({
        "type": "response.completed",
        "sequence_number": 9,
        "response": {
            "id": id,
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "error": null,
            "incomplete_details": null,
            "instructions": null,
            "max_output_tokens": null,
            "model": "gpt-5.4",
            "usage": null,
            "output": output,
            "tools": []
        }
    })
    .to_string()
}

struct GetWeather(Arc<AtomicUsize>);

impl Tool for GetWeather {
    const NAME: &'static str = "get_weather";
    type Error = ToolExecutionError;
    type Args = serde_json::Value;
    type Output = String;

    fn description(&self) -> String {
        "The weather in a city".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({"type": "object", "properties": {"city": {"type": "string"}}})
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok("sunny".to_owned())
    }
}

#[tokio::test]
async fn an_agent_runs_its_tool_loop_over_one_connection() {
    let call = json!({
        "type": "function_call",
        "id": "fc_1",
        "arguments": "{\"city\":\"Tokyo\"}",
        "call_id": "call_1",
        "name": "get_weather",
        "status": "completed",
    });
    let script = Script::turns(vec![
        vec![
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "sequence_number": 1,
                "item": call,
            })
            .to_string(),
            completed("resp_1", json!([call])),
        ],
        vec![
            json!({
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": "Sunny in Tokyo",
                "item_id": "msg_2",
                "logprobs": [],
                "output_index": 0,
                "sequence_number": 1,
            })
            .to_string(),
            completed("resp_2", json!([])),
        ],
    ]);
    let wire = OpenAIConfig::new("test-key")
        .connect(RecordingHttpClient::new("{}"))
        .responses("gpt-5.4")
        .wire;
    let model = Model::new(
        ResponsesSocket::new(wire),
        ResponsesWebSocket::from_connection(Box::new(script.clone())),
    );
    let calls = Arc::new(AtomicUsize::new(0));

    let output = AgentBuilder::new(model)
        .tool(GetWeather(calls.clone()))
        .build()
        .prompt("What is the weather in Tokyo?")
        .max_turns(2)
        .await
        .expect("the agent run completes over the websocket");

    assert_eq!(output.output, "Sunny in Tokyo");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let sent = script.sent();
    assert_eq!(sent.len(), 2);
    // Chaining is off: the second turn is the whole conversation, the
    // call and its result included, and names no previous response.
    assert!(sent[1].get("previous_response_id").is_none(), "{}", sent[1]);
    let input = sent[1]["input"].as_array().expect("input items");
    assert!(
        input.iter().any(|item| item["type"] == "function_call"),
        "{}",
        sent[1]
    );
    assert!(
        input
            .iter()
            .any(|item| item["type"] == "function_call_output" && item["call_id"] == "call_1"),
        "{}",
        sent[1]
    );
}
