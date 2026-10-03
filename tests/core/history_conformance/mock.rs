//! The scripted mock wire's history suite: it proves the harness drives a
//! wire end to end, with provider items carried in the scripted request.

use rig_core::completion::{CompletionRequest, CompletionResponse, FinishReason, Usage};
use rig_core::error::EncodeError;
use rig_core::message::{AssistantContent, Message, Opaque, Origin, Reasoning, Text};
use rig_core::test_utils::{MockFrame, MockScript, MockStreamEvent, mock_final};
use rig_core::wire::{Mode, Wire};
use rig_history_conformance::{Ending, HistoryFixture, Shape, decode};
use serde_json::{Value, json};

pub struct MockHistory;

const MODEL: &str = "mock-1";

fn streamed(shape: Shape) -> Option<Vec<MockFrame>> {
    let events = match shape {
        Shape::Rich => vec![
            MockStreamEvent::reasoning("plan the lookup").with_reasoning_id("rs_1"),
            MockStreamEvent::text_start("msg_1", Some(json!({"id": "msg_1", "phase": "final"}))),
            MockStreamEvent::text("looking it up"),
            MockStreamEvent::tool_call("call_1", "lookup", json!({"q": "rig"})),
            MockStreamEvent::final_response(Usage::default()),
        ],
        Shape::Interleaved => vec![
            MockStreamEvent::reasoning("first").with_reasoning_id("rs_1"),
            MockStreamEvent::text_start("msg_1", Some(json!({"id": "msg_1"}))),
            MockStreamEvent::text("between"),
            MockStreamEvent::reasoning("second").with_reasoning_id("rs_2"),
            MockStreamEvent::tool_call("call_1", "lookup", json!({"q": "rig"})),
            MockStreamEvent::final_response(Usage::default()),
        ],
        Shape::Unknown => return None,
    };
    Some(events.into_iter().map(MockFrame::Event).collect())
}

/// The whole reply of `shape`: what its stream folds into, answered at once.
fn whole(shape: Shape) -> Option<Vec<MockFrame>> {
    let response = match shape {
        Shape::Unknown => {
            let text = AssistantContent::Text(Text::new("noted"))
                .with_native(json!({"id": "msg_1", "x_rig_field": true}));
            let invented = AssistantContent::Opaque(Opaque {
                item: json!({"type": "x_rig_invented", "id": "x_1"}),
                replay: true,
            });
            CompletionResponse::new(
                vec![text, invented],
                Usage::default(),
                Origin::new("mock.script", "mock", MODEL),
                Value::Null,
            )
            .with_finish_reason(FinishReason::Stop)
        }
        shape => decode(
            &MockHistory.wire(MODEL),
            &CompletionRequest::new("restate"),
            Mode::Streaming,
            streamed(shape)?,
        )
        .ok()?,
    };
    Some(vec![MockFrame::Response(Box::new(response))])
}

fn finished(reason: FinishReason) -> Vec<MockFrame> {
    let response = CompletionResponse::new(
        vec![AssistantContent::Reasoning(Reasoning::new("done"))],
        Usage::default(),
        Origin::new("mock.script", "mock", MODEL),
        Value::Null,
    )
    .with_finish_reason(reason);
    vec![MockFrame::Response(Box::new(response))]
}

impl HistoryFixture for MockHistory {
    type Wire = MockScript;

    fn wire(&self, model: &str) -> MockScript {
        MockScript::default().with_id(model)
    }

    fn model(&self) -> &'static str {
        MODEL
    }

    fn other_model(&self) -> &'static str {
        "mock-2"
    }

    /// What a wire that sends current provider items verbatim would send:
    /// each assistant block as its item while it is current, else its
    /// canonical fields.
    fn body(
        &self,
        wire: &MockScript,
        request: CompletionRequest,
        mode: Mode,
    ) -> Result<Value, EncodeError> {
        let request = wire.encode(request, mode)?;
        let messages: Vec<Value> = request
            .chat_history
            .iter()
            .map(|message| match message {
                Message::Assistant(turn) => Value::Array(
                    turn.content
                        .iter()
                        .map(|block| match block {
                            AssistantContent::Opaque(opaque) => opaque.item.clone(),
                            block => block.native_item().cloned().unwrap_or_else(|| {
                                serde_json::to_value(block.canonical()).unwrap_or_default()
                            }),
                        })
                        .collect(),
                ),
                message => serde_json::to_value(message).unwrap_or_default(),
            })
            .collect();
        Ok(json!({"model": request.model, "messages": messages}))
    }

    fn reply(&self, shape: Shape, mode: Mode) -> Option<Vec<MockFrame>> {
        match mode {
            Mode::Streaming => streamed(shape),
            Mode::Unary => whole(shape),
        }
    }

    fn call_reply(&self, arguments: &str, mode: Mode) -> Option<Vec<MockFrame>> {
        let events = vec![
            MockStreamEvent::tool_call_name_delta("call_1", "lookup"),
            MockStreamEvent::tool_call_arguments_delta("call_1", arguments),
            MockStreamEvent::tool_call_end("call_1"),
            MockStreamEvent::FinalResponse(mock_final(Usage::default())),
        ];
        let frames: Vec<MockFrame> = events.into_iter().map(MockFrame::Event).collect();
        match mode {
            Mode::Streaming => Some(frames),
            Mode::Unary => {
                let response = decode(
                    &self.wire(MODEL),
                    &CompletionRequest::new("restate"),
                    Mode::Streaming,
                    frames,
                )
                .ok()?;
                Some(vec![MockFrame::Response(Box::new(response))])
            }
        }
    }

    fn finishes(&self) -> Vec<(&'static str, Vec<MockFrame>, Ending)> {
        vec![
            ("stop", finished(FinishReason::Stop), Ending::Success),
            ("length", finished(FinishReason::Length), Ending::Success),
            (
                "tool_calls",
                finished(FinishReason::ToolCalls),
                Ending::Success,
            ),
            (
                "content_filter",
                finished(FinishReason::ContentFilter),
                Ending::Failure,
            ),
            (
                "error",
                finished(FinishReason::Other("error".to_owned())),
                Ending::Failure,
            ),
        ]
    }
}

rig_history_conformance::history_conformance_suite! {
    wire: "mock",
    fixture: MockHistory,
}
