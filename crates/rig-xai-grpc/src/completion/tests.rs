use super::*;
use crate::proto::{
    self, CompletionOutput, CompletionOutputChunk, Delta, FunctionCall, SamplingUsage, ToolCall,
    ToolCallType, content, tool, tool_call, tool_choice,
};
use futures::StreamExt;
use rig_core::Model;
use rig_core::completion::{CompletionResponse, ToolDefinition};
use rig_core::message::{
    AssistantContent, CallId, ImageDetail, ImageMediaType, Message, ToolName, UserContent,
};
use rig_core::providers::xai::extension::{XaiExt, XaiOptions};
use rig_core::streaming::{Item, StreamEvent};
use serde_json::json;

const MODEL: &str = "grok-4.3";

fn add() -> ToolName {
    ToolName::new("add").expect("a tool name")
}

fn encode(request: CompletionRequest) -> Result<GetCompletionsRequest, EncodeError> {
    GetCompletion::new(MODEL).encode(request, Mode::Unary)
}

/// Answers every request with scripted frames.
#[derive(Clone)]
struct Scripted(std::sync::Arc<std::sync::Mutex<Vec<XaiFrame>>>);

impl Transport<GetCompletion> for Scripted {
    fn send(&self, _request: GetCompletionsRequest, _exchange: Exchange) -> Opening<XaiFrame> {
        let frames = std::mem::take(&mut *self.0.lock().expect("script lock"));
        Opening::ready(Opened::new(futures::stream::iter(
            frames.into_iter().map(Ok),
        )))
    }
}

fn scripted(frames: Vec<XaiFrame>) -> Model<GetCompletion, Scripted> {
    Model::new(
        GetCompletion::new(MODEL),
        Scripted(std::sync::Arc::new(std::sync::Mutex::new(frames))),
    )
}

fn call(id: &str, kind: ToolCallType, arguments: &str) -> ToolCall {
    ToolCall {
        id: id.to_owned(),
        r#type: kind as i32,
        status: proto::ToolCallStatus::Completed as i32,
        tool: Some(tool_call::Tool::Function(FunctionCall {
            name: "add".to_owned(),
            arguments: arguments.to_owned(),
        })),
        ..ToolCall::default()
    }
}

fn usage() -> SamplingUsage {
    SamplingUsage {
        prompt_tokens: 10,
        completion_tokens: 4,
        reasoning_tokens: 3,
        total_tokens: 17,
        cached_prompt_text_tokens: 6,
        cost_in_usd_ticks: Some(2_500_000),
        ..SamplingUsage::default()
    }
}

fn chunk(delta: Delta, finish: proto::FinishReason, usage: Option<SamplingUsage>) -> XaiFrame {
    XaiFrame::Chunk(proto::GetChatCompletionChunk {
        id: "resp-1".to_owned(),
        model: MODEL.to_owned(),
        outputs: vec![CompletionOutputChunk {
            delta: Some(Delta {
                role: proto::MessageRole::RoleAssistant as i32,
                ..delta
            }),
            finish_reason: finish as i32,
            ..CompletionOutputChunk::default()
        }],
        usage,
        ..proto::GetChatCompletionChunk::default()
    })
}

fn text(text: &str) -> Delta {
    Delta {
        content: text.to_owned(),
        ..Delta::default()
    }
}

/// The history, tools and options reach the proto request: roles as the
/// enum, text and images as content parts, the call and its result paired
/// by id, and the tool's schema as JSON text.
#[test]
fn a_conversation_with_tools_encodes_as_the_proto_request() {
    let mut request = CompletionRequest::new("What is 2 + 3?").preamble("Be brief.");
    request.chat_history.extend([
        Message::from(AssistantContent::tool_call(
            "call_1",
            add(),
            json!({"x": 2, "y": 3}),
        )),
        Message::tool_result(CallId::from_wire("call_1"), add(), "5"),
        Message::from(vec![
            UserContent::text("And this?"),
            UserContent::image_url(
                "https://example.com/sum.png",
                Some(ImageMediaType::PNG),
                Some(ImageDetail::Low),
            ),
        ]),
    ]);
    request.tools = vec![ToolDefinition::new(
        add(),
        "Add two integers.",
        json!({"type": "object", "properties": {"x": {"type": "integer"}}}),
    )];
    request.tool_choice = Some(rig_core::message::ToolChoice::Required);
    request.temperature = Some(0.5);
    request.max_tokens = Some(64);

    let encoded = encode(request).expect("the request encodes");

    assert_eq!(encoded.model, MODEL);
    assert_eq!(encoded.max_tokens, Some(64));
    assert_eq!(encoded.temperature, Some(0.5));
    let roles: Vec<i32> = encoded
        .messages
        .iter()
        .map(|message| message.role)
        .collect();
    use proto::MessageRole as Role;
    assert_eq!(
        roles,
        [
            Role::RoleSystem,
            Role::RoleUser,
            Role::RoleAssistant,
            Role::RoleTool,
            Role::RoleUser
        ]
        .map(|role| role as i32)
    );
    let assistant = encoded.messages.get(2).expect("the assistant turn");
    let sent = assistant.tool_calls.first().expect("the call");
    assert_eq!(sent.id, "call_1");
    assert_eq!(sent.r#type(), ToolCallType::ClientSideTool);
    let tool_turn = encoded.messages.get(3).expect("the result");
    assert_eq!(tool_turn.tool_call_id.as_deref(), Some("call_1"));
    let image = encoded
        .messages
        .get(4)
        .and_then(|message| message.content.get(1))
        .and_then(|part| part.content.clone());
    let Some(content::Content::ImageUrl(image)) = image else {
        panic!("the image part: {image:?}");
    };
    assert_eq!(image.detail(), proto::ImageDetail::DetailLow);
    let Some(tool::Tool::Function(function)) =
        encoded.tools.first().and_then(|tool| tool.tool.clone())
    else {
        panic!("the function tool");
    };
    let schema: serde_json::Value =
        serde_json::from_str(&function.parameters).expect("the schema is JSON text");
    assert_eq!(
        schema
            .pointer("/properties/x/type")
            .and_then(serde_json::Value::as_str),
        Some("integer")
    );
    assert_eq!(
        encoded.tool_choice.and_then(|choice| choice.tool_choice),
        Some(tool_choice::ToolChoice::Mode(
            proto::ToolMode::Required as i32
        ))
    );
}

/// Raw parameters reach any field the proto declares, by its proto3 JSON,
/// and one it does not declare is refused rather than dropped.
#[test]
fn raw_parameters_reach_proto_fields_and_unknown_ones_are_refused() {
    let mut request = CompletionRequest::new("hi");
    request.additional_params = Some(json!({
        "include": ["INCLUDE_OPTION_INLINE_CITATIONS"],
        "reasoning_effort": "low",
        "tools": [{"web_search": {}}],
    }));
    let encoded = encode(request).expect("the request encodes");
    assert_eq!(
        encoded.include,
        [proto::IncludeOption::InlineCitations as i32]
    );
    assert_eq!(
        encoded.reasoning_effort,
        Some(proto::ReasoningEffort::EffortLow as i32)
    );
    assert!(matches!(
        encoded.tools.first().and_then(|tool| tool.tool.clone()),
        Some(tool::Tool::WebSearch(_))
    ));

    let mut request = CompletionRequest::new("hi");
    request.additional_params = Some(json!({"no_such_field": true}));
    let error = encode(request).expect_err("an undeclared field is refused");
    assert!(error.to_string().contains("no_such_field"), "{error}");
}

/// The gRPC request has no prompt cache key, typed or as xAI's option.
#[test]
fn a_prompt_cache_key_is_refused() {
    let mut request = CompletionRequest::new("hi");
    request.options.cache_key = Some("conversation-1".to_owned());
    let model = scripted(Vec::new());
    let error = futures::executor::block_on(model.call(request)).expect_err("refused");
    assert!(error.to_string().contains("prompt cache key"), "{error}");

    let request =
        CompletionRequest::new("hi").provider_option(XaiOptions::new().prompt_cache_key("k"));
    let error = encode(request).expect_err("refused");
    assert!(error.to_string().contains("prompt_cache_key"), "{error}");
}

fn complete(frames: Vec<XaiFrame>) -> CompletionResponse {
    futures::executor::block_on(scripted(frames).call(CompletionRequest::new("hi")))
        .expect("the reply completes")
}

/// A unary reply reads its reasoning, text, client-side calls and usage;
/// a server-side call is the API's own and is not surfaced; `raw` is the
/// reply's proto3 JSON, which xAI's extras read.
#[test]
fn a_unary_reply_decodes_its_message_usage_and_raw_document() {
    let response = proto::GetChatCompletionResponse {
        id: "resp-1".to_owned(),
        model: MODEL.to_owned(),
        outputs: vec![CompletionOutput {
            finish_reason: proto::FinishReason::ReasonToolCalls as i32,
            message: Some(proto::CompletionMessage {
                content: "Adding.".to_owned(),
                reasoning_content: "Use the tool.".to_owned(),
                role: proto::MessageRole::RoleAssistant as i32,
                tool_calls: vec![
                    call("search_1", ToolCallType::WebSearchTool, "{}"),
                    call("call_1", ToolCallType::ClientSideTool, r#"{"x":2,"y":3}"#),
                ],
                ..proto::CompletionMessage::default()
            }),
            ..CompletionOutput::default()
        }],
        usage: Some(usage()),
        ..proto::GetChatCompletionResponse::default()
    };
    let reply = complete(vec![XaiFrame::Whole(response)]);

    let content = &reply.choice;
    assert!(
        matches!(content.first(), Some(AssistantContent::Reasoning(_))),
        "{content:?}"
    );
    let calls: Vec<_> = content
        .iter()
        .filter_map(|block| match block {
            AssistantContent::ToolCall(call) => Some(call.function.arguments_value()),
            _ => None,
        })
        .collect();
    assert_eq!(calls, [json!({"x": 2, "y": 3})]);
    assert_eq!(reply.usage.input_tokens, Some(10));
    assert_eq!(reply.usage.cached_input_tokens, Some(6));
    assert_eq!(
        reply.raw.pointer("/outputs/0/finish_reason"),
        Some(&json!("REASON_TOOL_CALLS"))
    );
    let extras = reply
        .extras::<XaiExt>()
        .expect("xAI extras")
        .expect("they read");
    assert_eq!(extras.cost_in_usd_ticks, Some(2_500_000));
}

/// A stream's text arrives as it is sent, its calls keep one index each,
/// a call streamed in fragments is not read again whole, and `raw` folds
/// the chunks into one response.
#[tokio::test]
async fn a_stream_decodes_text_and_calls_and_folds_its_document() {
    let fragment = |arguments: &str| ToolCall {
        index: Some(0),
        status: proto::ToolCallStatus::InProgress as i32,
        ..call("call_2", ToolCallType::ClientSideTool, arguments)
    };
    let frames = vec![
        chunk(text("Hel"), proto::FinishReason::ReasonInvalid, None),
        chunk(text("lo"), proto::FinishReason::ReasonInvalid, None),
        chunk(
            Delta {
                tool_calls: vec![call(
                    "call_1",
                    ToolCallType::ClientSideTool,
                    r#"{"x":1,"y":1}"#,
                )],
                ..Delta::default()
            },
            proto::FinishReason::ReasonInvalid,
            None,
        ),
        chunk(
            Delta {
                tool_calls: vec![fragment(r#"{"x":2,"#), fragment(r#""y":2}"#)],
                ..Delta::default()
            },
            proto::FinishReason::ReasonInvalid,
            None,
        ),
        chunk(
            Delta {
                tool_calls: vec![call(
                    "call_2",
                    ToolCallType::ClientSideTool,
                    r#"{"x":2,"y":2}"#,
                )],
                ..Delta::default()
            },
            proto::FinishReason::ReasonToolCalls,
            Some(usage()),
        ),
    ];
    let mut stream = scripted(frames)
        .stream(CompletionRequest::new("hi"))
        .expect("the stream opens");
    let mut streamed = String::new();
    while let Some(item) = stream.next().await {
        if let Item::Event(StreamEvent::Text { text, .. }) = item.expect("an event") {
            streamed.push_str(&text);
        }
    }
    let reply = stream.finish().await.expect("the stream ends");

    assert_eq!(streamed, "Hello");
    let calls: Vec<_> = reply
        .choice
        .iter()
        .filter_map(|block| match block {
            AssistantContent::ToolCall(call) => Some(call.function.arguments_value()),
            _ => None,
        })
        .collect();
    assert_eq!(calls, [json!({"x": 1, "y": 1}), json!({"x": 2, "y": 2})]);
    assert_eq!(reply.usage.total_tokens, Some(17));
    assert_eq!(
        reply.raw.pointer("/outputs/0/message/content"),
        Some(&json!("Hello"))
    );
    assert_eq!(
        reply.raw.pointer("/usage/cost_in_usd_ticks"),
        Some(&json!(2_500_000))
    );
}

/// A gRPC failure keeps its status code as the provider code.
#[test]
fn an_rpc_failure_keeps_its_code() {
    let error = rpc_error(&tonic::Status::unavailable("overloaded"));
    assert_eq!(grpc_code_name(tonic::Code::Unavailable), "UNAVAILABLE");
    assert!(error.to_string().contains("overloaded"), "{error}");
}
