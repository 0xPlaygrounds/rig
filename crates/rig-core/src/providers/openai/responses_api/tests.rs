use super::*;
use crate::error::EncodeError;
use crate::error::ProviderError;
use crate::message;
use serde_json::json;
use std::collections::HashMap;

/// A whole Responses body decoded as the OpenAI wire's unary reply.
fn fold_body(response: Value) -> Result<completion::CompletionResponse, ProviderError> {
    let body = serde_json::to_string(&response)?;
    crate::test_utils::decode_reply(
        &openai_wire("gpt-5-mini"),
        &crate::completion::CompletionRequest::new("hello"),
        crate::wire::Mode::Unary,
        [crate::wire::WireFrame::Text(body)],
        serde_json::Value::Null,
    )
}

/// The OpenAI Responses wire, for the request-shaping assertions.
fn openai_wire(model: &str) -> wire::Responses {
    crate::providers::openai::OpenAIConfig::new("dummy-key").responses(model)
}

/// The Responses request a wire builds for a Rig request — the one
/// conversion every caller reaches, whatever opened the socket.
fn wire_request(wire: &wire::Responses, request: completion::CompletionRequest) -> Value {
    wire.responses_request(request, false)
        .expect("request should convert")
}

/// The request the OpenAI wire for `model` builds for `request`.
fn convert(model: &str, request: completion::CompletionRequest) -> Result<Value, EncodeError> {
    openai_wire(model).responses_request(request, false)
}

/// The request the OpenAI wire for `model` sends for `request`, prepared as
/// the driver prepares it.
fn prepared(model: &str, request: completion::CompletionRequest) -> Value {
    use crate::wire::{Operation, Wire};
    let wire = openai_wire(model);
    let request = crate::operation::Completion::prepare(request, &wire.describe())
        .expect("the request prepares");
    wire.responses_request(request, false)
        .expect("request should convert")
}

/// The input items `message` alone becomes.
fn input_items(message: completion::Message) -> Result<Vec<Value>, EncodeError> {
    let wire = openai_wire("gpt-5");
    let mut custom = Custom {
        tools: Default::default(),
        calls: Default::default(),
    };
    super::input(&[message], &wire, "gpt-5", &mut custom, false)
}

fn test_document(id: &str, text: &str) -> crate::completion::Document {
    crate::completion::Document {
        id: id.to_string(),
        text: text.to_string(),
        additional_props: HashMap::new(),
    }
}

fn weather_tool_definition() -> completion::ToolDefinition {
    completion::ToolDefinition {
        name: crate::message::ToolName::new("get_weather").expect("tool name"),
        description: "Get the weather".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }),
    }
}

fn rig_tool_result(content: message::ToolResultContent) -> message::Message {
    rig_tool_result_of(vec![content])
}

fn rig_tool_result_of(content: Vec<message::ToolResultContent>) -> message::Message {
    message::Message::User {
        content: vec![message::UserContent::ToolResult(message::ToolResult {
            is_error: false,
            call: crate::message::CallId::from_wire("call-id"),
            name: crate::message::ToolName::new("tool".to_string()).expect("tool name"),
            content,
        })],
    }
}

#[test]
fn tool_result_file_id_image_uses_the_native_wire_field() {
    let input = rig_tool_result(message::ToolResultContent::Image(message::Image {
        data: message::DocumentSourceKind::FileId("file-image-123".to_string()),
        media_type: None,
        detail: None,
        native: None,
    }));

    let items = input_items(input).expect("input item conversion");
    assert_eq!(
        items[0],
        json!({
            "type": "function_call_output",
            "call_id": "call-id",
            "output": [{
                "type": "input_image",
                "file_id": "file-image-123",
                "detail": "auto"
            }],
            "status": "completed"
        })
    );
}

fn weather_tool_request() -> completion::CompletionRequest {
    completion::CompletionRequest::new("what's the weather?").tools(vec![weather_tool_definition()])
}

#[test]
fn responses_tool_choice_modes_serialize_as_plain_strings() {
    for (choice, expected) in [
        (message::ToolChoice::Auto, json!("auto")),
        (message::ToolChoice::None, json!("none")),
        (message::ToolChoice::Required, json!("required")),
    ] {
        assert_eq!(
            super::tool_choice(choice).expect("mode should convert"),
            expected
        );
    }
}

#[test]
fn responses_tool_choice_specific_single_name_serializes_as_named_function() {
    let converted = super::tool_choice(message::ToolChoice::Specific {
        function_names: vec![crate::message::ToolName::new("get_weather").expect("tool name")],
    })
    .expect("single specific tool should convert");

    assert_eq!(
        converted,
        json!({"type": "function", "name": "get_weather"})
    );
}

#[test]
fn responses_tool_choice_specific_multiple_names_serialize_as_allowed_tools() {
    let converted = super::tool_choice(message::ToolChoice::Specific {
        function_names: vec![
            crate::message::ToolName::new("add").expect("tool name"),
            crate::message::ToolName::new("subtract").expect("tool name"),
        ],
    })
    .expect("multiple specific tools should convert");

    assert_eq!(
        converted,
        json!({
            "type": "allowed_tools",
            "mode": "required",
            "tools": [
                {"type": "function", "name": "add"},
                {"type": "function", "name": "subtract"}
            ]
        })
    );
}

#[test]
fn responses_tool_choice_specific_empty_names_error() {
    let converted = super::tool_choice(message::ToolChoice::Specific {
        function_names: vec![],
    });

    assert!(matches!(
        converted.map_err(ProviderError::from),
        Err(ProviderError::Request(error))
            if error.to_string().contains("at least one function name")
    ));
}

#[test]
fn responses_request_with_specific_tool_choice_serializes_named_function() {
    let mut request = weather_tool_request();
    request.tool_choice = Some(message::ToolChoice::Specific {
        function_names: vec![crate::message::ToolName::new("get_weather").expect("tool name")],
    });

    let request_json = convert("gpt-test", request).expect("convert");

    assert_eq!(
        request_json.get("tool_choice"),
        Some(&json!({"type": "function", "name": "get_weather"}))
    );
}

#[test]
fn responses_strict_function_tools_sanitize_schema() {
    let tool = ResponsesToolDefinition::strict_function(
        "get_weather",
        "Get the weather",
        weather_tool_definition().parameters,
    );

    assert!(tool.strict);
    assert_eq!(tool.parameters["additionalProperties"], json!(false));
    assert_eq!(tool.parameters["required"], json!(["location", "unit"]));
}

fn request_with_preamble(preamble: &str) -> completion::CompletionRequest {
    completion::CompletionRequest::from(vec![
        message::Message::system(preamble),
        message::Message::user("Hello"),
    ])
}

fn system_only_request(system_text: &str) -> completion::CompletionRequest {
    completion::CompletionRequest::new(completion::Message::system(system_text))
}

#[test]
fn responses_request_with_only_system_messages_keeps_them_in_input() {
    let req = convert("gpt-4o-mini", system_only_request("System only"))
        .expect("request conversion should succeed");
    let serialized = serde_json::to_value(&req).expect("request should serialize");
    let input = serialized["input"]
        .as_array()
        .expect("input should be array");

    assert!(
        serialized.get("instructions").is_none(),
        "lifting a system-only history would leave input empty, so it stays in input"
    );
    assert_eq!(input.len(), 1);
    assert_eq!(input[0]["role"], "system");
    assert!(input[0].to_string().contains("System only"));
}

#[test]
fn responses_wire_can_fallback_to_system_messages_in_input() {
    let wire = openai_wire("gpt-4o-mini").with_system_instructions_as_messages();

    let req = wire_request(&wire, request_with_preamble("You are concise."));
    let serialized = serde_json::to_value(&req).expect("request should serialize");
    let input = serialized["input"]
        .as_array()
        .expect("input should be array");

    assert!(serialized.get("instructions").is_none());
    assert_eq!(input.len(), 2);
    assert_eq!(input[0]["role"], "system");
    assert!(input[0].to_string().contains("You are concise."));
    assert_eq!(input[1]["role"], "user");
}

#[test]
fn all_instructions_system_only_input_reports_non_system_requirement() {
    for (case, system) in [
        ("system-only input", "System only"),
        (
            "whitespace-only system input, which produces no `instructions` field",
            "   ",
        ),
    ] {
        let Err(err) = openai_wire("gpt-4o-mini")
            .with_system_instructions_placement(SystemInstructionsPlacement::AllInstructions)
            .responses_request(system_only_request(system), false)
        else {
            panic!("{case}: should fail once every item is lifted");
        };

        assert!(
            err.to_string().contains("non-system item"),
            "{case}: the error should explain that lifted system messages left input empty: {err}"
        );
    }
}

#[test]
fn responses_request_conversion_keeps_tools_non_strict_by_default() {
    let req = convert("gpt-4o-mini", weather_tool_request()).expect("request should convert");

    let tool = &req["tools"][0];
    assert_eq!(tool["strict"], json!(false));
    assert_eq!(tool["parameters"]["required"], json!(["location"]));
    assert!(tool["parameters"].get("additionalProperties").is_none());
}

#[test]
fn responses_wire_strict_tools_opt_in_sanitizes_all_function_tools() {
    let wire =
        openai_wire("gpt-4o-mini")
            .with_strict_tools()
            .with_tool(completion::ToolDefinition {
                name: crate::message::ToolName::new("lookup").expect("tool name"),
                description: "Look something up".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {"q": {"type": "string"}}
                }),
            });

    let mut request = weather_tool_request();
    request.additional_params = Some(json!({
        "tools": [{
            "type": "function",
            "name": "extra",
            "description": "An additional_params tool",
            "parameters": {"type": "object", "properties": {"x": {"type": "string"}}}
        }]
    }));

    let req = wire_request(&wire, request);

    let tools = req["tools"].as_array().expect("tools");
    assert_eq!(tools.len(), 3);
    for tool in tools {
        assert_eq!(
            tool["strict"],
            json!(true),
            "{} should be strict",
            tool["name"]
        );
        assert_eq!(tool["parameters"]["additionalProperties"], json!(false));
    }
}

#[test]
fn responses_request_keeps_documents_after_lifted_system_messages() {
    let request = crate::completion::CompletionRequest::new("Prompt")
        .message(completion::Message::system("System prompt"))
        .message(completion::Message::user("Earlier user turn"))
        .message(completion::Message::assistant("Earlier assistant turn"))
        .document(test_document("doc1", "Document text."));

    let responses_request = prepared("gpt-4o-mini", request);

    let serialized = serde_json::to_value(&responses_request).expect("request should serialize");
    let input = serialized["input"]
        .as_array()
        .expect("input should be an array");

    assert_eq!(serialized["instructions"], json!("System prompt"));
    assert_eq!(input.len(), 4);
    assert_eq!(input[0]["role"], "user");
    assert!(
        input[0].to_string().contains("<file id: doc1>"),
        "document input should be first after system instructions are lifted: {input:?}"
    );
    assert_eq!(input[1]["role"], "user");
    assert!(
        input[1].to_string().contains("Earlier user turn"),
        "prior user history should follow document input: {input:?}"
    );
    assert_eq!(input[2]["role"], "assistant");
    assert!(
        input[2].to_string().contains("Earlier assistant turn"),
        "prior assistant history should follow prior user history: {input:?}"
    );
    assert_eq!(input[3]["role"], "user");
    assert!(
        input[3].to_string().contains("Prompt"),
        "prompt should remain last: {input:?}"
    );
}

#[test]
fn responses_direct_request_keeps_mid_conversation_system_messages_in_input() {
    let request = crate::completion::CompletionRequest::from(vec![
        completion::Message::system("System prompt"),
        completion::Message::assistant("Earlier assistant turn"),
        completion::Message::system("Mid-conversation instruction"),
        completion::Message::user("Prompt"),
    ])
    .documents(vec![test_document("doc1", "Document text.")]);

    let responses_request = prepared("gpt-4o-mini", request);

    let serialized = serde_json::to_value(&responses_request).expect("request should serialize");
    let input = serialized["input"]
        .as_array()
        .expect("input should be an array");

    assert_eq!(
        serialized["instructions"],
        json!("System prompt"),
        "only the leading run of system messages should be lifted"
    );
    assert_eq!(input.len(), 4);
    assert_eq!(input[0]["role"], "user");
    assert!(
        input[0].to_string().contains("<file id: doc1>"),
        "document input should follow lifted system instructions: {input:?}"
    );
    assert_eq!(input[1]["role"], "assistant");
    assert_eq!(input[2]["role"], "system");
    assert!(
        input[2]
            .to_string()
            .contains("Mid-conversation instruction"),
        "mid-conversation system messages should keep their position: {input:?}"
    );
    assert_eq!(input[3]["role"], "user");
    assert_eq!(
        input
            .iter()
            .filter(|message| message.to_string().contains("<file id: doc1>"))
            .count(),
        1,
        "document input should appear exactly once: {input:?}"
    );
}

#[test]
fn completion_response_incomplete_reports_the_truncation_reason() {
    let response = json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "incomplete",
        "incomplete_details": { "reason": "max_output_tokens" },
        "model": "gpt-5.4",
        "output": [{
            "type": "message",
            "id": "msg_456",
            "status": "incomplete",
            "role": "assistant",
            "content": [{ "type": "output_text", "annotations": [], "text": "half an ans" }]
        }],
        "tools": []
    });

    let completion: completion::CompletionResponse =
        fold_body(response).expect("response should convert");

    assert_eq!(
        completion.finish_reason(),
        Some(completion::FinishReason::Length)
    );
}

#[test]
fn file_id_document_serializes_as_input_item_content() {
    let message = completion::Message::User {
        content: vec![message::UserContent::Document(message::Document {
            data: DocumentSourceKind::FileId("file_abc".to_string()),
            media_type: None,
            additional_params: None,
        })],
    };

    let converted = input_items(message).expect("conversion should succeed");
    let json = serde_json::to_value(&converted[0]).expect("serialize input item");

    assert_eq!(json["type"], "message");
    assert_eq!(json["role"], "user");
    assert_eq!(json["content"][0]["type"], "input_file");
    assert_eq!(json["content"][0]["file_id"], "file_abc");
    assert!(json["content"][0].get("file_data").is_none());
    assert!(json["content"][0].get("file_url").is_none());
}

#[test]
fn completion_response_with_unknown_output_keeps_usage() {
    // Guards the original reason the catch-all exists: an unknown item must
    // not break decoding of the whole response or drop token usage.
    let response = json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-5.4",
        "output": [
            {
                "type": "web_search_call",
                "id": "ws_001",
                "status": "completed",
            },
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [ { "type": "output_text", "text": "hi", "annotations": [] } ],
            },
        ],
        "usage": {
            "input_tokens": 100,
            "input_tokens_details": { "cached_tokens": 25 },
            "output_tokens": 50,
            "output_tokens_details": { "reasoning_tokens": 15 },
            "total_tokens": 150,
        },
    });

    let completion = fold_body(response).expect("an unknown item decodes");
    assert!(matches!(
        completion.choice.first(),
        Some(completion::AssistantContent::Opaque(_))
    ));
    assert_eq!(completion.usage.total_tokens, Some(150));
    assert_eq!(completion.usage.cached_input_tokens, Some(25));
    assert_eq!(completion.usage.reasoning_tokens, Some(15));
}

// Regression tests for issue #1429: `file_url` and `filename` are mutually
// exclusive on OpenAI's Responses API (400 `mutually_exclusive_parameters`),
// so URL-backed PDFs must not carry the hardcoded `filename`. These tests
// cover the `TryFrom<crate::completion::Message> for Vec<InputItem>` path
// that `Model::call` requests actually go through.
//
// See <https://platform.openai.com/docs/guides/pdf-files> for the
// `input_file` content part and its `file_url` / `file_data` / `file_id`
// input variants.

const PDF_URL: &str = "https://example.com/resume.pdf";

fn url_pdf_message() -> message::Message {
    message::Message::User {
        content: vec![message::UserContent::document_url(
            PDF_URL,
            Some(message::DocumentMediaType::PDF),
        )],
    }
}

/// Recursively collect every JSON object with `"type": "input_file"`.
fn find_input_files(value: &serde_json::Value, out: &mut Vec<serde_json::Value>) {
    match value {
        serde_json::Value::Object(map) => {
            if map.get("type").and_then(|t| t.as_str()) == Some("input_file") {
                out.push(value.clone());
            }
            map.values().for_each(|v| find_input_files(v, out));
        }
        serde_json::Value::Array(items) => {
            items.iter().for_each(|v| find_input_files(v, out));
        }
        _ => {}
    }
}

fn sole_input_file(value: &serde_json::Value) -> serde_json::Value {
    let mut found = Vec::new();
    find_input_files(value, &mut found);
    assert_eq!(
        found.len(),
        1,
        "expected exactly one input_file item in {value:#}"
    );
    found.pop().unwrap()
}

fn assert_url_only_input_file(input_file: &serde_json::Value) {
    assert_eq!(
        input_file.get("file_url").and_then(|v| v.as_str()),
        Some(PDF_URL),
        "URL PDF should carry file_url: {input_file:#}"
    );
    assert_eq!(
        input_file.get("filename"),
        None,
        "filename must be absent for URL PDFs (issue #1429): {input_file:#}"
    );
    assert_eq!(
        input_file.get("file_data"),
        None,
        "file_data must be absent for URL PDFs: {input_file:#}"
    );
}

#[test]
fn url_pdf_via_input_item_path_omits_filename() {
    let items = input_items(url_pdf_message()).expect("URL PDF should convert to input items");
    let json = serde_json::to_value(&items).expect("input items should serialize");
    assert_url_only_input_file(&sole_input_file(&json));
}

#[test]
fn base64_pdf_via_input_item_path_keeps_filename() {
    let input = message::Message::User {
        content: vec![message::UserContent::Document(message::Document {
            data: DocumentSourceKind::base64("dGVzdA=="),
            media_type: Some(message::DocumentMediaType::PDF),
            additional_params: None,
        })],
    };

    let items = input_items(input).expect("base64 PDF should convert to input items");
    let json = serde_json::to_value(&items).expect("input items should serialize");
    let input_file = sole_input_file(&json);

    assert_eq!(
        input_file.get("file_data").and_then(|v| v.as_str()),
        Some("data:application/pdf;base64,dGVzdA=="),
        "base64 PDF should carry file_data: {input_file:#}"
    );
    assert_eq!(
        input_file.get("filename").and_then(|v| v.as_str()),
        Some("document.pdf"),
        "base64 PDF should keep the default filename: {input_file:#}"
    );
    assert_eq!(
        input_file.get("file_url"),
        None,
        "base64 PDF should not carry file_url: {input_file:#}"
    );
}

/// Raw-capture test: the reply document, driven end to end over a mock
/// transport that hands back a Responses body *and* an `x-request-id`
/// response header. The captured value is the body as parsed; the
/// transport id lives on the normalized response, beside the capture, not
/// inside it. `with_error_response_headers` with `200 OK` is the one unary
/// double that carries response headers.
mod raw_capture {
    use super::*;

    use crate::test_utils::RecordingHttpClient;

    const REQUEST_ID: &str = "req_unit_responses_0001";

    /// A Responses body carrying `service_tier`, which the normalized
    /// response provably lacks.
    const BODY: &str = r#"{
            "id": "resp_raw_1",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "error": null,
            "incomplete_details": null,
            "instructions": null,
            "max_output_tokens": null,
            "model": "gpt-4o-mini-2024-07-18",
            "service_tier": "default",
            "usage": {
                "input_tokens": 4,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 3,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 7
            },
            "output": [{
                "type": "message",
                "id": "msg_raw_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "hello", "annotations": []}]
            }],
            "tools": []
        }"#;

    /// The load-bearing capture property: `raw` is the provider's body
    /// exactly as it arrived, and folding that capture again (with the
    /// header id reattached, since the capture is body only) reproduces
    /// every normalized field. The transport id is not part of the capture,
    /// while the normalized response beside it carries the header.
    #[tokio::test]
    async fn completion_captures_the_provider_body_as_raw() {
        let mut headers = http::HeaderMap::new();
        headers.insert("x-request-id", http::HeaderValue::from_static(REQUEST_ID));
        let model = crate::driver::Model::new(
            openai_wire("gpt-4o-mini"),
            RecordingHttpClient::with_error_response_headers(http::StatusCode::OK, BODY, headers),
        );

        let response = model
            .call(crate::completion::CompletionRequest::new("hello"))
            .await
            .expect("completion");

        let raw = &response.raw;
        let body: Value = serde_json::from_str(BODY).expect("the body is JSON");
        assert_eq!(*raw, body, "the capture is the provider's body");
        assert_eq!(raw["service_tier"], "default");
        assert!(raw.get("provider_request_id").is_none());

        let mut refolded = fold_body(raw.clone()).expect("re-fold the capture");
        refolded.provider_request_id = Some(REQUEST_ID.to_string());
        assert_eq!(response.identity(), refolded.identity());
        assert_eq!(response.finish_reason(), refolded.finish_reason());
        assert_eq!(response.model(), refolded.model());
        assert_eq!(response.usage, refolded.usage);
        assert_eq!(response.choice, refolded.choice);
        assert_eq!(response.provider_request_id.as_deref(), Some(REQUEST_ID));
    }
}

/// The `lookup` tool the histories below call.
fn lookup_tool() -> completion::ToolDefinition {
    completion::ToolDefinition {
        name: message::ToolName::new("lookup").expect("tool name"),
        description: "Look something up".to_owned(),
        parameters: json!({"type": "object"}),
    }
}

#[test]
fn foreign_call_ids_are_normalized_as_pi_normalizes_them() {
    use crate::completion::ReplayTarget;
    let wire = openai_wire("gpt-5.4");
    let long = format!("call_{}", "a".repeat(80));
    for (id, expected) in [
        ("call_1", "call_1".to_owned()),
        ("toolu_01|fc_abc", "toolu_01_fc_abc".to_owned()),
        ("call.with spaces..", "call_with_spaces".to_owned()),
        (long.as_str(), long[..64].to_owned()),
    ] {
        assert_eq!(
            wire.normalize_tool_call_id(id, wire.model(), None),
            expected,
            "{id}"
        );
    }
}

/// A request continuing a stored response sends the results of that
/// response's calls alone: the adapter keeps them, since the provider holds
/// the calls they answer.
#[test]
fn a_stored_continuation_keeps_the_results_of_the_stored_calls() {
    use crate::wire::{Operation, Wire};
    let wire = openai_wire("gpt-5.4");
    let result = completion::Message::tool_result(
        message::CallId::from_wire("call_1"),
        message::ToolName::new("lookup").expect("tool name"),
        "sunny",
    );
    let mut request =
        crate::completion::CompletionRequest::from(vec![result]).tools(vec![lookup_tool()]);
    request.additional_params = Some(json!({"previous_response_id": "resp_1", "store": true}));
    let prepared = crate::operation::Completion::prepare(request.clone(), &wire.describe())
        .expect("the continuation prepares");
    let body = serde_json::to_value(wire.responses_request(prepared, false).expect("encodes"))
        .expect("serializes");
    assert_eq!(body["input"][0]["type"], "function_call_output");
    assert_eq!(body["input"][0]["call_id"], "call_1");
    assert_eq!(body["previous_response_id"], "resp_1");

    // Without a stored response, the same result answers nothing.
    request.additional_params = None;
    let error = crate::operation::Completion::prepare(request, &wire.describe())
        .and_then(|prepared| wire.responses_request(prepared, false).map_err(Into::into));
    assert!(error.is_err(), "{error:?}");
}
