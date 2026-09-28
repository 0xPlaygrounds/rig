use serde_json::{Value, json};

use super::super::completion::GenerateContent;
use super::super::{GeminiConfig, api};
use crate::NonEmpty;
use crate::completion::{CompletionRequest, CompletionResponse, FinishReason, ToolDefinition};
use crate::message::{AssistantContent, Message, ToolChoice, UserContent};
use crate::wire::{Mode, Wire, WireFrame};

fn wire() -> GenerateContent {
    GenerateContent::new(GeminiConfig::new("test-key"), "gemini-3.8-flash")
}

fn body(wire: &GenerateContent, request: CompletionRequest) -> Value {
    let encoded = wire
        .encode(request, Mode::Unary)
        .expect("the request encodes");
    let crate::wire::Body::Bytes(bytes) = encoded.request.body() else {
        panic!("a JSON body");
    };
    serde_json::from_slice(bytes).expect("JSON")
}

fn body_text(wire: &GenerateContent, request: CompletionRequest) -> String {
    let encoded = wire
        .encode(request, Mode::Unary)
        .expect("the request encodes");
    let crate::wire::Body::Bytes(bytes) = encoded.request.body() else {
        panic!("a JSON body");
    };
    String::from_utf8(bytes.to_vec()).expect("UTF-8")
}

fn decode(frames: &[&str], mode: Mode) -> Result<CompletionResponse, crate::error::ProviderError> {
    let wire = wire();
    crate::test_utils::decode_reply(
        &wire,
        &CompletionRequest::new("hi"),
        mode,
        frames
            .iter()
            .map(|frame| WireFrame::Text((*frame).to_owned())),
        Value::Null,
    )
}

fn tool(name: &str) -> ToolDefinition {
    ToolDefinition {
        name: name.to_owned(),
        description: format!("{name} tool"),
        parameters: json!({
            "type": "object",
            "properties": {
                "kind": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                "limit": {"type": "integer", "minimum": 1}
            },
            "required": ["kind"],
            "additionalProperties": false
        }),
    }
}

const NATIVE: &str = r#"{"executableCode":{"language":"PYTHON","code":"print(21 * 2)","id":"c1"},"thoughtSignature":"Y29kZQ=="}"#;

fn unary_reply() -> String {
    format!(
        r#"{{"candidates":[{{"content":{{"role":"model","parts":[{{"text":"weighing it","thought":true,"thoughtSignature":"dGhvdWdodA=="}},{NATIVE},{{"functionCall":{{"id":"fc1","name":"lookup","args":{{"q":"a"}}}},"thoughtSignature":"Y2FsbA=="}},{{"functionCall":{{"id":"fc2","name":"lookup","args":{{"q":"b"}}}}}}]}},"finishReason":"STOP"}}],"usageMetadata":{{"promptTokenCount":100,"cachedContentTokenCount":40,"candidatesTokenCount":10,"thoughtsTokenCount":5,"toolUsePromptTokenCount":7,"totalTokenCount":122}},"modelVersion":"gemini-3.8-flash","responseId":"r1"}}"#
    )
}

#[test]
fn a_unary_reply_keeps_every_part_in_order() {
    let response = decode(&[&unary_reply()], Mode::Unary).expect("decodes");
    let kinds: Vec<&str> = response
        .choice
        .iter()
        .map(|part| match part {
            AssistantContent::Reasoning(_) => "thought",
            AssistantContent::Native(_) => "native",
            AssistantContent::ToolCall(_) => "call",
            AssistantContent::Text(_) => "text",
            AssistantContent::Image(_) => "image",
        })
        .collect();
    assert_eq!(kinds, ["thought", "native", "call", "call"]);
    let calls: Vec<_> = response.tool_calls().collect();
    let signature = |index: usize| {
        calls.get(index).and_then(|call| {
            call.signature
                .as_ref()
                .and_then(|signature| signature.open(&super::super::ISSUER))
                .map(|signature| signature.signature.clone())
        })
    };
    assert_eq!(signature(0).as_deref(), Some("Y2FsbA=="));
    assert_eq!(signature(1), None, "parallel calls carry one signature");
    assert_eq!(response.finish_reason(), Some(FinishReason::ToolCalls));
    assert_eq!(response.model.as_deref(), Some("gemini-3.8-flash"));
}

#[test]
fn usage_counts_tool_prompts_as_input_and_thoughts_as_output() {
    let response = decode(&[&unary_reply()], Mode::Unary).expect("decodes");
    let usage = response.usage;
    assert_eq!(usage.input_tokens, Some(107));
    assert_eq!(usage.output_tokens, Some(15));
    assert_eq!(usage.total_tokens, Some(122));
    assert_eq!(usage.cached_input_tokens, Some(40));
    assert_eq!(usage.reasoning_tokens, Some(5));
    assert_eq!(usage.tool_use_prompt_tokens, Some(7));
}

#[test]
fn a_native_part_is_re_sent_byte_for_byte() {
    let response = decode(&[&unary_reply()], Mode::Unary).expect("decodes");
    let content = NonEmpty::from_vec(response.choice).expect("parts");
    let request = CompletionRequest::new("go on").message(Message::Assistant { id: None, content });
    let text = body_text(&wire(), request);
    assert!(text.contains(NATIVE), "{text}");
    assert!(text.contains(r#""thoughtSignature":"dGhvdWdodA==""#));
    assert!(text.contains(r#""thoughtSignature":"Y2FsbA==""#));
}

#[test]
fn streamed_text_joins_across_chunks_and_a_trailing_signature_is_its_own_part() {
    let frames = [
        r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"Hello"}]}}],"responseId":"r2"}"#,
        r#"{"candidates":[{"content":{"role":"model","parts":[{"text":", world"}]}}]}"#,
        r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"","thoughtSignature":"ZW5k"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":2,"totalTokenCount":5}}"#,
    ];
    let response = decode(&frames, Mode::Streaming).expect("decodes");
    let texts: Vec<(String, Option<String>)> = response
        .choice
        .iter()
        .filter_map(|part| match part {
            AssistantContent::Text(text) => Some((
                text.text.clone(),
                text.signature
                    .as_ref()
                    .and_then(|signature| signature.open(&super::super::ISSUER))
                    .map(|signature| signature.signature.clone()),
            )),
            _ => None,
        })
        .collect();
    assert_eq!(
        texts,
        [
            ("Hello, world".to_owned(), None),
            (String::new(), Some("ZW5k".to_owned()))
        ]
    );
    assert_eq!(response.usage.total_tokens, Some(5));
}

#[test]
fn a_reply_without_a_finish_reason_is_truncated() {
    let error = decode(
        &[r#"{"candidates":[{"content":{"parts":[{"text":"cut"}]}}]}"#],
        Mode::Streaming,
    )
    .expect_err("truncated");
    assert!(matches!(error, crate::error::ProviderError::Truncated));
}

#[test]
fn a_tool_protocol_finish_is_an_error() {
    let error = decode(
        &[r#"{"candidates":[{"finishReason":"MALFORMED_FUNCTION_CALL","finishMessage":"bad call"}]}"#],
        Mode::Unary,
    )
    .expect_err("an error");
    assert!(
        error.to_string().contains("MALFORMED_FUNCTION_CALL"),
        "{error}"
    );
}

#[test]
fn a_blocked_prompt_is_a_refusal() {
    let error = decode(
        &[r#"{"promptFeedback":{"blockReason":"SAFETY"}}"#],
        Mode::Unary,
    )
    .expect_err("blocked");
    let crate::error::ProviderError::ProviderResponse(response) = error else {
        panic!("a provider response");
    };
    assert!(response.refusal);
}

#[test]
fn an_empty_reply_needs_a_truncating_reason() {
    assert!(
        decode(
            &[r#"{"candidates":[{"content":{"parts":[]},"finishReason":"STOP"}]}"#],
            Mode::Unary
        )
        .is_err()
    );
    let response = decode(
        &[r#"{"candidates":[{"content":{"parts":[]},"finishReason":"MAX_TOKENS"}]}"#],
        Mode::Unary,
    )
    .expect("a truncated reply is not an error");
    assert!(response.choice.is_empty());
}

#[test]
fn additional_params_are_refused() {
    let error = wire()
        .encode(
            CompletionRequest::new("hi").additional_params(json!({"generationConfig": {}})),
            Mode::Unary,
        )
        .expect_err("refused");
    assert!(
        error
            .to_string()
            .contains("additional_params is not read by Gemini; use GenerateContent::settings"),
        "{error}"
    );
}

#[test]
fn tool_schemas_are_sent_as_written() {
    let body = body(&wire(), CompletionRequest::new("hi").tool(tool("search")));
    assert_eq!(
        body["tools"],
        json!([{"functionDeclarations": [{
            "name": "search",
            "description": "search tool",
            "parametersJsonSchema": tool("search").parameters
        }]}])
    );
    assert!(body.get("toolConfig").is_none());
}

#[test]
fn function_and_hosted_tools_together_turn_on_server_side_invocations() {
    let hosted = wire().with_settings(api::RequestSettings {
        tools: vec![api::HostedTool {
            code_execution: Some(api::CodeExecution::default()),
            ..Default::default()
        }],
        ..Default::default()
    });
    let mixed = body(&hosted, CompletionRequest::new("hi").tool(tool("search")));
    assert_eq!(
        mixed["toolConfig"],
        json!({"includeServerSideToolInvocations": true})
    );
    assert_eq!(mixed["tools"][1], json!({"codeExecution": {}}));

    let hosted_only = body(&hosted, CompletionRequest::new("hi"));
    assert!(hosted_only.get("toolConfig").is_none());
    let functions_only = body(&wire(), CompletionRequest::new("hi").tool(tool("search")));
    assert!(functions_only.get("toolConfig").is_none());
}

#[test]
fn rig_settings_and_model_settings_meet_in_one_body() {
    let wire = wire().with_settings(api::RequestSettings {
        generation_config: api::GenerationSettings {
            thinking_config: Some(api::ThinkingConfig {
                thinking_level: Some(api::ThinkingLevel::Low),
                ..Default::default()
            }),
            ..Default::default()
        },
        service_tier: Some(api::ServiceTier::Flex),
        ..Default::default()
    });
    let body = body(
        &wire,
        CompletionRequest::new("hi")
            .preamble("be brief")
            .max_tokens(256)
            .tool(tool("search"))
            .tool_choice(ToolChoice::Specific {
                function_names: vec!["search".to_owned()],
            }),
    );
    assert_eq!(
        body["generationConfig"],
        json!({"maxOutputTokens": 256, "thinkingConfig": {"thinkingLevel": "LOW"}})
    );
    assert_eq!(body["serviceTier"], "flex");
    assert_eq!(
        body["systemInstruction"],
        json!({"parts": [{"text": "be brief"}]})
    );
    assert_eq!(
        body["toolConfig"],
        json!({"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["search"]}})
    );
    assert_eq!(
        body["contents"],
        json!([{"role": "user", "parts": [{"text": "hi"}]}])
    );
}

#[test]
fn a_tool_result_answers_its_call_by_id_and_name() {
    let call = crate::message::ToolCall::from_wire(
        "fc1",
        crate::message::ToolFunction::new(
            crate::message::ToolName::new("lookup").expect("name"),
            json!({"q": "a"}),
        ),
    );
    let request = CompletionRequest::new(Message::tool_results(NonEmpty::new(call.result(
        crate::message::ToolResultContent::json(json!({"status": "ok"})),
    ))))
    .message(Message::Assistant {
        id: None,
        content: NonEmpty::new(AssistantContent::ToolCall(call)),
    });
    let body = body(&wire(), request);
    assert_eq!(
        body["contents"][1],
        json!({"role": "user", "parts": [{"functionResponse": {
            "id": "fc1",
            "name": "lookup",
            "response": {"result": {"status": "ok"}}
        }}]})
    );
}

#[test]
fn media_detail_becomes_the_parts_resolution() {
    let image = UserContent::image_base64(
        "aGk=",
        Some(crate::message::ImageMediaType::PNG),
        Some(crate::message::MediaDetail::Low),
    );
    let request = CompletionRequest::new(Message::User {
        content: NonEmpty::with_rest(UserContent::text("what is this?"), [image]),
    });
    let body = body(&wire(), request);
    assert_eq!(
        body["contents"][0]["parts"][1],
        json!({
            "inlineData": {"data": "aGk=", "mimeType": "image/png"},
            "mediaResolution": {"level": "MEDIA_RESOLUTION_LOW"}
        })
    );
}

#[test]
fn another_services_native_part_is_refused() {
    let native = crate::message::Sealed::new(
        "anthropic",
        crate::message::NativePart::new(
            "x",
            serde_json::value::RawValue::from_string("{}".to_owned()).expect("raw"),
        ),
    );
    let request = CompletionRequest::new("hi").message(Message::Assistant {
        id: None,
        content: NonEmpty::new(AssistantContent::Native(native)),
    });
    assert!(wire().encode(request, Mode::Unary).is_err());
}
