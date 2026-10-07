use super::*;
use crate::completion::{CompletionRequest, Message, ToolDefinition};
use crate::message::{self, ToolChoice as MessageToolChoice};
use serde_json::json;

/// The body `request` builds on the Gemini 2.5 wire.
fn body_of(request: CompletionRequest) -> Value {
    let wire =
        crate::providers::gemini::GeminiConfig::new("test-key").interactions("gemini-2.5-flash");
    json_of(&wire, request, None).expect("request should build")
}

/// The body `wire` builds for `request`, as JSON.
fn json_of(
    wire: &Interactions,
    request: CompletionRequest,
    stream: Option<bool>,
) -> Result<Value, EncodeError> {
    Ok(serde_json::to_value(create_request_body(
        wire, &request, stream,
    )?)?)
}

/// The first `function_result` step of `body`.
fn function_result(body: &Value) -> &Value {
    body["input"]
        .as_array()
        .and_then(|steps| steps.iter().find(|step| step["type"] == "function_result"))
        .expect("a function result step")
}

fn tool_result_request(content: Vec<message::ToolResultContent>) -> CompletionRequest {
    CompletionRequest::new(Message::from(message::UserContent::tool_result(
        crate::message::CallId::from_wire("call-123"),
        crate::message::ToolName::new("get_weather").expect("tool name"),
        content,
    )))
}

#[test]
fn test_create_request_body_simple() {
    let prompt = Message::User {
        content: vec![message::UserContent::text("Hello")],
    };

    let request = CompletionRequest::from(vec![Message::system("Be precise."), prompt])
        .temperature(0.7)
        .max_tokens(128)
        .tool_choice(MessageToolChoice::Required);

    let wire =
        crate::providers::gemini::GeminiConfig::new("test-key").interactions("gemini-2.5-flash");
    let result = json_of(&wire, request, Some(false)).expect("request should build");

    assert_eq!(result["model"], "gemini-2.5-flash");
    assert!(result.get("agent").is_none());
    assert_eq!(result["stream"], false);
    assert_eq!(result["system_instruction"], "Be precise.");

    let config = &result["generation_config"];
    assert_eq!(config["temperature"], 0.7);
    assert_eq!(config["max_output_tokens"], 128);
    assert_eq!(config["tool_choice"], "any");

    assert_eq!(
        result["input"],
        json!([{"type": "user_input", "content": [{"type": "text", "text": "Hello"}]}])
    );
}

/// `functionResponse.name` is the executed function's name: read from
/// the required `ToolResult::name`, never an identifier.
#[test]
fn tool_result_serializes_the_executed_name_not_an_identifier() {
    use message::{AssistantContent, ToolCall, ToolFunction, ToolResultContent};

    let call = |call_id: &str, name: &str| {
        let function = ToolFunction::new(
            crate::message::ToolName::new(name.to_owned()).expect("tool name"),
            json!({}),
        );
        Message::Assistant(message::AssistantMessage::new(vec![
            AssistantContent::ToolCall(ToolCall::from_wire(call_id, function)),
        ]))
    };
    let result = |call_id: &str, name: &str| Message::User {
        content: vec![message::UserContent::tool_result(
            crate::message::CallId::from_wire(call_id),
            crate::message::ToolName::new(name).expect("tool name"),
            vec![ToolResultContent::text("out")],
        )],
    };

    let request = CompletionRequest::from(vec![
        // A driver-built result carries the executed name (a repair
        // hook renamed the call: `sum` ran, not `add`).
        call("call_1", "sum"),
        result("call_1", "sum"),
        // An OpenAI-shaped correlator travels as the call id while the
        // required `name` field carries the executed name: `call_abc` must
        // never reach the wire as a name.
        call("call_abc", "get_weather"),
        result("call_abc", "get_weather"),
        call("call_9", "get_time"),
        result("call_9", "get_time"),
    ]);

    let body = body_of(request);
    let results: Vec<&Value> = body["input"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|step| step["type"] == "function_result")
        .collect();
    let names: Vec<&str> = results
        .iter()
        .filter_map(|step| step["name"].as_str())
        .collect();
    let call_ids: Vec<&str> = results
        .iter()
        .filter_map(|step| step["call_id"].as_str())
        .collect();

    assert_eq!(names, ["sum", "get_weather", "get_time"]);
    assert_eq!(call_ids, ["call_1", "call_abc", "call_9"]);
}

#[test]
fn test_tool_result_without_provider_id_sends_minted_call_id() {
    // A call id is always available: the wire gets the provider-issued id
    // when one exists, else the spelling of the id rig issued.
    let call = message::CallId::from_wire("");
    let history = vec![Message::from(message::UserContent::ToolResult(
        message::ToolResult {
            is_error: false,
            call: call.clone(),
            name: crate::message::ToolName::new("get_weather".to_string()).expect("tool name"),
            content: vec![message::ToolResultContent::text("ok")],
        },
    ))];
    let wire =
        crate::providers::gemini::GeminiConfig::new("test-key").interactions("gemini-2.5-flash");
    let spelled = crate::providers::internal::wire_ids::WireIds::for_target(
        &history,
        &wire,
        "gemini-2.5-flash",
    );
    let body = body_of(CompletionRequest::from(history));

    let result = function_result(&body);
    let sent = result["call_id"].as_str().expect("a call id is sent");
    assert!(!sent.is_empty());
    assert_eq!(Some(sent), spelled.of(&call));
    assert_eq!(result["name"], "get_weather");
}

#[test]
fn test_tool_result_preserves_text_and_json_types() {
    let body = body_of(tool_result_request(vec![
        message::ToolResultContent::text(r#"{"status":"literal"}"#),
        message::ToolResultContent::json(json!({ "status": "structured" })),
    ]));

    let expected_result = json!([
        {
            "type": "text",
            "text": "{\"status\":\"literal\"}"
        },
        {
            "type": "text",
            "text": "{\"status\":\"structured\"}"
        }
    ]);
    assert_eq!(
        function_result(&body),
        &json!({
            "type": "function_result",
            "name": "get_weather",
            "result": expected_result,
            "call_id": "call-123"
        })
    );
}

#[test]
fn test_tool_result_images_and_text_serialize_as_ordered_tagged_content() {
    let tool_result = message::UserContent::ToolResult(message::ToolResult {
        is_error: false,
        call: crate::message::CallId::from_wire("call-image"),
        name: crate::message::ToolName::new("render".to_string()).expect("tool name"),
        content: vec![
            message::ToolResultContent::image_base64(
                "first-image",
                Some(message::ImageMediaType::PNG),
                None,
            ),
            message::ToolResultContent::text("between-images"),
            message::ToolResultContent::Image(message::Image {
                data: message::DocumentSourceKind::Url(
                    "https://example.com/second.jpg".to_string(),
                ),
                media_type: Some(message::ImageMediaType::JPEG),
                detail: None,
                native: None,
            }),
        ],
    });
    let serialized = body_of(CompletionRequest::new(Message::User {
        content: vec![tool_result],
    }));

    // The result is a step of its own on the wire (not nested in a
    // `user_input` step, which the API refuses on a round trip).
    assert_eq!(
        serialized.pointer("/input/0"),
        Some(&json!({
            "type": "function_result",
            "name": "render",
            "result": [
                {
                    "type": "image",
                    "data": "first-image",
                    "mime_type": "image/png"
                },
                {
                    "type": "text",
                    "text": "between-images"
                },
                {
                    "type": "image",
                    "uri": "https://example.com/second.jpg",
                    "mime_type": "image/jpeg"
                }
            ],
            "call_id": "call-image"
        }))
    );
}

/// A canonical reasoning block, with no provider step to send, is rebuilt
/// as a thought whose summary items carry their `type`, as every content
/// item on this wire does.
#[test]
fn a_rebuilt_thought_tags_its_summary() {
    let wire = interactions_wire();
    let ids = WireIds::for_target(&[], &wire, &wire.model);
    let step = assistant_step(
        &message::AssistantContent::reasoning("add them"),
        &wire,
        &ids,
    )
    .expect("reasoning converts");
    assert_eq!(
        step,
        Some(json!({
            "type": "thought",
            "summary": [{ "type": "text", "text": "add them" }]
        }))
    );
}

/// A turn decoded from this model, as the recorded tool turn of
/// `gemini/interactions_api/tool_result_roundtrip.yaml` has it: a
/// signature-only thought, then a call.
fn decoded_tool_turn() -> Message {
    let thought = json!({"signature": "c2ln", "type": "thought"});
    let call = json!({"arguments": {"x": 7, "y": 11}, "id": "call_217140", "name": "add", "type": "function_call"});
    let tool_call = message::ToolCall::from_wire(
        "call_217140",
        message::ToolFunction::new(
            crate::message::ToolName::new("add").expect("tool name"),
            json!({"x": 7, "y": 11}),
        ),
    );
    let mut turn = message::AssistantMessage::new(vec![
        message::AssistantContent::Reasoning(message::Reasoning::new("")).with_native(thought),
        message::AssistantContent::ToolCall(tool_call).with_native(call),
    ]);
    turn.origin = Some(crate::message::Origin::new(
        API,
        PROVIDER_NAME,
        "gemini-3-flash-preview",
    ));
    turn.stop = Some(crate::message::StopReason::ToolUse);
    Message::Assistant(turn)
}

fn tool_result() -> Message {
    Message::from(message::UserContent::tool_result(
        crate::message::CallId::from_wire("call_217140"),
        crate::message::ToolName::new("add").expect("tool name"),
        vec![message::ToolResultContent::json(json!({"sum": 18}))],
    ))
}

/// The steps `history` encodes to for `wire`, adapted at the request
/// boundary as the driver adapts it.
/// The request declares `add`, so the history's calls and results replay as
/// steps rather than text.
fn encoded_steps(wire: &Interactions, history: Vec<Message>) -> Vec<Value> {
    let add = ToolDefinition {
        name: crate::message::ToolName::new("add").expect("tool name"),
        description: "Add two numbers.".to_owned(),
        parameters: json!({"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}}}),
    };
    let request = <crate::operation::Completion as crate::wire::Operation>::prepare(
        CompletionRequest::from(history).tools(vec![add]),
        &wire.describe(),
    )
    .expect("the request is valid");
    match json_of(wire, request, None).expect("the request builds") {
        Value::Object(mut body) => match body.shift_remove("input") {
            Some(Value::Array(steps)) => steps,
            other => panic!("the body's input is a list: {other:?}"),
        },
        other => panic!("the body is an object: {other}"),
    }
}

/// Another model gets no thought: the signature-only reasoning is dropped
/// and the call is rebuilt from its canonical fields.
#[test]
fn another_model_gets_the_canonical_turn() {
    let wire = crate::providers::gemini::GeminiConfig::new("test-key")
        .interactions("gemini-3-pro-preview");
    let steps = encoded_steps(
        &wire,
        vec![
            Message::user("Add 7 and 11."),
            decoded_tool_turn(),
            tool_result(),
        ],
    );
    assert_eq!(
        steps.get(1),
        Some(
            &json!({"type": "function_call", "name": "add", "arguments": {"x": 7, "y": 11}, "id": "call_217140"})
        )
    );
    assert_eq!(steps.len(), 3);
}

/// A tool round trip in hand-built history: each block and the result are
/// steps of their own, in the message's order.
#[test]
fn a_tool_round_trip_is_top_level_steps() {
    let call = message::ToolCall::from_wire(
        "fc_1",
        message::ToolFunction::new(
            crate::message::ToolName::new("add".to_owned()).expect("tool name"),
            json!({"x": 17, "y": 25}),
        ),
    );
    let assistant = Message::Assistant(message::AssistantMessage::new(vec![
        message::AssistantContent::text("Adding."),
        message::AssistantContent::ToolCall(call.clone()),
    ]));
    let result = Message::from(message::UserContent::tool_result(
        call.id.clone(),
        crate::message::ToolName::new("add").expect("tool name"),
        vec![message::ToolResultContent::json(json!({"sum": 42}))],
    ));
    let steps = encoded_steps(
        &interactions_wire(),
        vec![Message::user("Add 17 and 25."), assistant, result],
    );
    let kinds: Vec<&str> = steps
        .iter()
        .filter_map(|step| step["type"].as_str())
        .collect();
    assert_eq!(
        kinds,
        [
            "user_input",
            "model_output",
            "function_call",
            "function_result"
        ]
    );
}

// ── the Interactions wire ───────────────────────────────────────────────
//
// Bodies pasted verbatim from committed cassettes, named at each constant.
// This family's unary reply is a different *document* from its stream
// events (a whole interaction resource), so the decoder names it as one
// more event of the wire and replays the resource's steps through the
// streamed content mapping. These tests pin that the two transports agree
// on the SHAPE of a turn (block kinds and signature placement), which is
// what the recorded traffic allows: no two committed cassettes record the
// same interaction both ways.

use crate::test_utils::RecordingHttpClient;
use crate::wire::{Mode, Wire};

/// `crates/rig-cassette/fixtures/cassettes/gemini/interactions_api/basic_interaction_returns_id.yaml`
const UNARY_INTERACTION: &str = r#"{"created":"1970-01-01T00:00:00Z","id":"v1_REDACTED_1","model":"gemini-3-flash-preview","object":"interaction","service_tier":"standard","status":"completed","steps":[{"signature":"signature_REDACTED_1","type":"thought"},{"content":[{"text":"1. Hummingbirds are the only birds capable of flying **backwards**.\n2. Their hearts can beat up to **1,260 times per minute**.","type":"text"}],"type":"model_output"}],"updated":"1970-01-01T00:00:00Z","usage":{"input_tokens_by_modality":[{"modality":"text","tokens":14}],"raw_prompt_token":39,"total_cached_tokens":0,"total_input_tokens":14,"total_output_tokens":34,"total_thought_tokens":222,"total_tokens":270,"total_tool_use_tokens":0}}"#;

fn interactions_wire() -> Interactions {
    crate::providers::gemini::GeminiConfig::new("test-key").interactions("gemini-3-flash-preview")
}

fn probe() -> CompletionRequest {
    CompletionRequest::new("probe")
}

/// The block kinds a folded turn carries, and the signature on its
/// reasoning block: the shape two transports must agree on.
fn shape(response: &crate::completion::CompletionResponse) -> (Vec<&'static str>, Option<String>) {
    let kinds = response
        .choice
        .iter()
        .map(|item| match item {
            message::AssistantContent::Text(_) => "text",
            message::AssistantContent::Reasoning(_) => "reasoning",
            message::AssistantContent::ToolCall(_) => "tool_call",
            message::AssistantContent::Image(_) => "image",
            message::AssistantContent::Opaque(_) => "opaque",
        })
        .collect();
    let signature = response.choice.iter().find_map(|item| match item {
        message::AssistantContent::Reasoning(reasoning) => reasoning
            .native
            .as_ref()
            .and_then(|native| native.item["signature"].as_str())
            .map(str::to_owned),
        _ => None,
    });
    (kinds, signature)
}

/// The one request an `Encoded` carries: this wire sends one per call.
fn sole(encoded: &crate::wire::Encoded) -> &http::Request<crate::wire::Body> {
    &encoded.request
}

/// A `background: true` interaction outlives its create request and a
/// dropped stream resumes from the last event seen. Both are the same
/// interaction read again, so both are requests on one wire: the poll GETs
/// the resource, the resume GETs the event stream from `last_event_id`
/// onward. These are byte for byte the two requests the client layer sent.
#[test]
fn one_interaction_is_polled_unary_and_resumed_streamed() {
    let gemini = crate::providers::gemini::GeminiConfig::new("test-key");

    let poll = InteractionResume::new(gemini.clone(), "v1_REDACTED_1")
        .encode(probe(), Mode::Unary)
        .expect("the poll request encodes");
    assert_eq!(sole(&poll).method(), http::Method::GET);
    assert_eq!(
        sole(&poll).uri().path(),
        "/v1beta/interactions/v1_REDACTED_1"
    );
    assert_eq!(sole(&poll).uri().query(), None);
    assert_eq!(poll.framing, crate::http_client::framing::Framing::Whole);
    assert_eq!(
        sole(&poll)
            .headers()
            .get("x-goog-api-key")
            .and_then(|value| value.to_str().ok()),
        Some("test-key")
    );

    let resumed = InteractionResume::new(gemini.clone(), "v1_REDACTED_1")
        .after_event("42")
        .encode(probe(), Mode::Streaming)
        .expect("the resume request encodes");
    assert_eq!(sole(&resumed).method(), http::Method::GET);
    assert_eq!(
        sole(&resumed).uri().path(),
        "/v1beta/interactions/v1_REDACTED_1"
    );
    assert_eq!(
        sole(&resumed).uri().query(),
        Some("stream=true&last_event_id=42&alt=sse")
    );
    assert_eq!(resumed.framing, crate::http_client::framing::Framing::Sse);

    // Resuming without a cursor asks for the stream from its beginning,
    // which is the API's own default and what the client layer sent.
    let from_start = InteractionResume::new(gemini, "v1_REDACTED_1")
        .encode(probe(), Mode::Streaming)
        .expect("the resume request encodes");
    assert_eq!(sole(&from_start).uri().query(), Some("stream=true&alt=sse"));
}

/// The poll's reply is the whole interaction resource, so it decodes
/// through the same decoder as the stream and the document survives on the
/// response's `raw` for a caller that wants the provider's own vocabulary.
#[tokio::test]
async fn a_polled_interaction_folds_its_steps_and_keeps_the_document() {
    let response = crate::driver::Model::new(
        InteractionResume::new(
            crate::providers::gemini::GeminiConfig::new("test-key"),
            "v1_REDACTED_1",
        ),
        RecordingHttpClient::new(UNARY_INTERACTION),
    )
    .call(probe())
    .await
    .expect("the recorded interaction resource decodes");

    assert_eq!(
        shape(&response),
        (
            vec!["reasoning", "text"],
            Some("signature_REDACTED_1".to_owned())
        )
    );
    assert_eq!(response.response_id(), Some("v1_REDACTED_1"));
    assert_eq!(
        response.raw["id"], "v1_REDACTED_1",
        "`raw` is the interaction document"
    );
    assert_eq!(response.raw["status"], "completed");
}
