use serde_json::{Value, json};

use super::*;
use crate::NonEmpty;
use crate::completion::ToolDefinition;
use crate::message::AssistantContent;
use crate::providers::gemini::GeminiConfig;
use crate::providers::gemini::api::Unmodeled;
use crate::wire::{Mode, Wire, WireFrame};

fn wire() -> Interactions {
    Interactions::new(GeminiConfig::new("test-key"), "gemini-3.8-flash")
}

fn body(wire: &Interactions, request: CompletionRequest, mode: Mode) -> Value {
    let encoded = wire.encode(request, mode).expect("encodes");
    let crate::wire::Body::Bytes(bytes) = encoded.request.body() else {
        panic!("a JSON body");
    };
    serde_json::from_slice(bytes).expect("JSON")
}

#[test]
fn a_request_is_a_list_of_steps_with_rigs_fields() {
    let tool = ToolDefinition {
        name: "holdings".into(),
        description: "Positions in an account.".into(),
        parameters: json!({"type": "object", "properties": {"account": {"type": "string"}}}),
    };
    let body = body(
        &wire(),
        CompletionRequest::new("What is ACC-7 worth?")
            .preamble("You are a portfolio analyst.")
            .temperature(0.2)
            .max_tokens(512)
            .tool(tool)
            .tool_choice(ToolChoice::Required),
        Mode::Unary,
    );
    assert_eq!(
        body,
        json!({
            "model": "gemini-3.8-flash",
            "input": [{"type": "user_input", "content": [{"type": "text", "text": "What is ACC-7 worth?"}]}],
            "system_instruction": "You are a portfolio analyst.",
            "tools": [{
                "type": "function",
                "name": "holdings",
                "description": "Positions in an account.",
                "parameters": {"type": "object", "properties": {"account": {"type": "string"}}}
            }],
            "generation_config": {"temperature": 0.2, "max_output_tokens": 512, "tool_choice": "any"},
            "stream": false
        })
    );
}

#[test]
fn settings_carry_interactions_own_options() {
    let wire = wire().with_settings(api::RequestSettings {
        agent: Some("deep-research-preview-04-2026".into()),
        background: Some(true),
        service_tier: Some(api::ServiceTier::Deferred),
        tools: vec![api::HostedTool::new("google_search")],
        generation_config: api::GenerationSettings {
            thinking_level: Some(api::ThinkingLevel::Low),
            ..Default::default()
        },
        ..Default::default()
    });
    let body = body(&wire, CompletionRequest::new("hi"), Mode::Streaming);
    assert!(body.get("model").is_none(), "an agent replaces the model");
    assert_eq!(body["agent"], "deep-research-preview-04-2026");
    assert_eq!(body["service_tier"], "deferred");
    assert_eq!(body["tools"], json!([{"type": "google_search"}]));
    assert_eq!(body["generation_config"], json!({"thinking_level": "low"}));
    assert_eq!(body["stream"], true);
}

#[test]
fn settings_refuse_what_interactions_rejects_or_rig_owns() {
    for key in [
        "safety_settings",
        "safetySettings",
        "system_instruction",
        "input",
    ] {
        assert!(
            Unmodeled::<api::RequestSettings>::new()
                .with(key, json!([]))
                .is_err(),
            "{key}"
        );
    }
    for key in [
        "media_resolution",
        "mediaResolution",
        "temperature",
        "tool_choice",
    ] {
        assert!(
            Unmodeled::<api::GenerationSettings>::new()
                .with(key, json!("high"))
                .is_err(),
            "{key}"
        );
    }
}

#[test]
fn additional_params_are_refused() {
    assert!(
        wire()
            .encode(
                CompletionRequest::new("hi").additional_params(json!({"store": false})),
                Mode::Unary
            )
            .is_err()
    );
}

const STEPS: &str = r#"{"id":"v1_1","status":"requires_action","model":"gemini-3.8-flash","object":"interaction","usage":{"total_tokens":157227,"total_input_tokens":4163,"total_cached_tokens":90503,"total_output_tokens":1124,"total_tool_use_tokens":147650,"total_thought_tokens":4290},"steps":[
{"id":"call_1","signature":"c2VhcmNo","type":"google_search_call","arguments":{"queries":["NVDA close"]},"search_type":"web_search"},
{"call_id":"call_1","signature":"cmVzdWx0","type":"google_search_result","result":[{"search_suggestions":"<b>x</b>"}]},
{"signature":"dGhvdWdodA==","type":"thought"},
{"id":"call_2","type":"function_call","name":"holdings","arguments":{"account":"ACC-7"}},
{"type":"model_output","content":[{"type":"text","text":"Checking.","annotations":[{"type":"url_citation","url":"https://example.com","start_index":0,"end_index":4}]}]}
]}"#;

fn decode(frames: &[&str], mode: Mode) -> crate::completion::CompletionResponse {
    crate::test_utils::decode_reply(
        &wire(),
        &CompletionRequest::new("hi"),
        mode,
        frames
            .iter()
            .map(|frame| WireFrame::Text((*frame).to_owned())),
        Value::Null,
    )
    .expect("decodes")
}

#[test]
fn a_whole_interaction_keeps_hosted_steps_and_replays_them() {
    let response = decode(&[STEPS], Mode::Unary);
    let kinds: Vec<&str> = response
        .choice
        .iter()
        .map(|part| match part {
            AssistantContent::Native(_) => "native",
            AssistantContent::Reasoning(_) => "thought",
            AssistantContent::ToolCall(_) => "call",
            AssistantContent::Text(_) => "text",
            AssistantContent::Image(_) => "image",
        })
        .collect();
    assert_eq!(
        kinds,
        ["native", "native", "thought", "call", "text", "native"]
    );
    let usage = response.usage;
    assert_eq!(usage.input_tokens, Some(4163 + 147650));
    assert_eq!(usage.output_tokens, Some(1124 + 4290));
    assert_eq!(usage.total_tokens, Some(157227));
    assert!(usage.cached_input_tokens <= usage.input_tokens);

    let content = NonEmpty::from_vec(response.choice).expect("parts");
    let replay = body(
        &wire(),
        CompletionRequest::new("go on").message(Message::Assistant { id: None, content }),
        Mode::Unary,
    );
    let input = replay["input"].as_array().expect("steps");
    assert_eq!(input[0]["type"], "google_search_call");
    assert_eq!(input[0]["signature"], "c2VhcmNo");
    assert_eq!(input[0]["search_type"], "web_search");
    assert_eq!(input[1]["signature"], "cmVzdWx0");
    assert_eq!(
        input[2],
        json!({"type": "thought", "signature": "dGhvdWdodA=="})
    );
    assert_eq!(input[3]["id"], "call_2");
    assert_eq!(
        input[4],
        json!({"type": "model_output", "content": [{
            "type": "text",
            "text": "Checking.",
            "annotations": [{"type": "url_citation", "url": "https://example.com", "start_index": 0, "end_index": 4}]
        }]})
    );
}

#[test]
fn a_streamed_search_call_gets_its_search_type_back() {
    let frames = [
        r#"{"interaction":{"id":"","status":"in_progress","object":"interaction","model":"gemini-3.8-flash"},"event_type":"interaction.created"}"#,
        r#"{"index":0,"step":{"id":"call_9","signature":"","type":"google_search_call"},"event_type":"step.start"}"#,
        r#"{"index":0,"delta":{"signature":"c2VhcmNo","type":"google_search_call","arguments":{"queries":["q"]}},"event_type":"step.delta"}"#,
        r#"{"index":0,"event_type":"step.stop"}"#,
        r#"{"index":1,"step":{"type":"thought"},"event_type":"step.start"}"#,
        r#"{"index":1,"delta":{"signature":"dGg=","type":"thought_signature"},"event_type":"step.delta"}"#,
        r#"{"index":1,"event_type":"step.stop"}"#,
        r#"{"index":2,"step":{"type":"model_output"},"event_type":"step.start"}"#,
        r#"{"index":2,"delta":{"text":"The close ","type":"text"},"event_type":"step.delta"}"#,
        r#"{"index":2,"delta":{"text":"was 225.","type":"text"},"event_type":"step.delta"}"#,
        r#"{"index":2,"event_type":"step.stop"}"#,
        r#"{"interaction":{"id":"v1_2","status":"completed","usage":{"total_tokens":30,"total_input_tokens":20,"total_output_tokens":6,"total_thought_tokens":4}},"event_type":"interaction.completed"}"#,
    ];
    let response = decode(&frames, Mode::Streaming);
    assert_eq!(response.text(), "The close was 225.");
    let native = response
        .choice
        .iter()
        .find_map(|part| match part {
            AssistantContent::Native(native) => Some(native),
            _ => None,
        })
        .expect("the search step");
    let Ok(api::Step::Hosted(step)) = api::Step::try_from(native) else {
        panic!("a hosted step");
    };
    assert_eq!(step.search_type.as_deref(), Some("web_search"));
    assert_eq!(step.signature.as_deref(), Some("c2VhcmNo"));
    assert_eq!(step.id.as_deref(), Some("call_9"));
    assert_eq!(response.usage.total_tokens, Some(30));
}

#[test]
fn a_streamed_call_assembles_its_arguments() {
    let frames = [
        r#"{"index":0,"step":{"id":"call_3","type":"function_call","name":"holdings"},"event_type":"step.start"}"#,
        r#"{"index":0,"delta":{"type":"arguments_delta","arguments":"{\"account\":"},"event_type":"step.delta"}"#,
        r#"{"index":0,"delta":{"type":"arguments_delta","arguments":"\"ACC-7\"}"},"event_type":"step.delta"}"#,
        r#"{"index":0,"event_type":"step.stop"}"#,
        r#"{"interaction":{"id":"v1_3","status":"requires_action"},"event_type":"interaction.completed"}"#,
    ];
    let response = decode(&frames, Mode::Streaming);
    let call = response.tool_calls().next().expect("a call");
    assert_eq!(call.function.arguments, json!({"account": "ACC-7"}));
    assert_eq!(call.id.to_string(), "call_3");
}

#[test]
fn a_tool_result_names_its_call_even_when_rig_issued_the_id() {
    let call = crate::message::ToolCall::from_wire(
        "",
        crate::message::ToolFunction::new(
            crate::message::ToolName::new("holdings").expect("name"),
            json!({}),
        ),
    );
    let request = CompletionRequest::new(Message::tool_results(NonEmpty::new(
        call.result(crate::message::ToolResultContent::text("12 GOOGL")),
    )))
    .message(Message::Assistant {
        id: None,
        content: NonEmpty::new(AssistantContent::ToolCall(call)),
    });
    let body = body(&wire(), request, Mode::Unary);
    let input = body["input"].as_array().expect("steps");
    assert_eq!(input[0]["id"], input[1]["call_id"]);
    assert_eq!(input[1]["result"], "12 GOOGL");
}
