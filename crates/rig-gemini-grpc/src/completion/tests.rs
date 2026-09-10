use super::*;

// ============================================================
// rpc_error — pins the from_provider_body usage on the RPC error path
// ============================================================

#[test]
fn rpc_error_preserves_status_text_without_http_status() {
    let status = tonic::Status::unavailable("boom");
    let expected = status.to_string();

    let err = rpc_error(&status);

    // The raw provider error text is preserved verbatim, and there is no
    // HTTP status because gRPC is a non-HTTP transport.
    assert_eq!(err.provider_response_body(), Some(expected.as_str()));
    assert_eq!(err.provider_response_status(), None);
}

#[test]
fn test_decode_base64_bytes_accepts_url_safe_with_padding() {
    assert!(matches!(
        decode_base64_bytes("_-wgVQA="),
        Ok(bytes) if bytes == vec![0xFF, 0xEC, 0x20, 0x55, 0x00]
    ));
}

#[test]
fn test_decode_base64_bytes_accepts_url_safe_no_pad() {
    assert!(matches!(
        decode_base64_bytes("_-wgVQA"),
        Ok(bytes) if bytes == vec![0xFF, 0xEC, 0x20, 0x55, 0x00]
    ));
}

#[test]
fn test_decode_base64_bytes_accepts_standard_no_pad() {
    assert!(matches!(
        decode_base64_bytes("Zg"),
        Ok(bytes) if bytes == b"f".to_vec()
    ));
}

#[test]
fn test_decode_base64_bytes_accepts_data_uri_prefix() {
    assert!(matches!(
        decode_base64_bytes("data:text/plain;base64,Zm9v"),
        Ok(bytes) if bytes == b"foo".to_vec()
    ));
}

// ============================================================
// tool_parameters_to_proto_schema — regression coverage for #1710
// ============================================================

#[test]
fn tool_params_empty_object_maps_to_none() {
    let v = serde_json::json!({"type": "object", "properties": {}});
    assert!(tool_parameters_to_proto_schema(&v).unwrap().is_none());
}

#[test]
fn tool_params_null_maps_to_none() {
    assert!(
        tool_parameters_to_proto_schema(&serde_json::Value::Null)
            .unwrap()
            .is_none()
    );
}

#[test]
fn tool_params_object_with_scalar_properties_round_trips() {
    let v = serde_json::json!({
        "type": "object",
        "properties": {
            "city":      { "type": "string",  "description": "City name" },
            "max_price": { "type": "integer", "description": "Cap, USD"  }
        },
        "required": ["city"]
    });

    let schema = tool_parameters_to_proto_schema(&v)
        .expect("schema conversion")
        .expect("schema");
    assert_eq!(schema.r#type, proto::Type::Object as i32);
    assert_eq!(schema.required, vec!["city".to_string()]);
    assert_eq!(schema.properties.len(), 2);

    let city = schema.properties.get("city").expect("city prop");
    assert_eq!(city.r#type, proto::Type::String as i32);
    assert_eq!(city.description, "City name");

    let max_price = schema.properties.get("max_price").expect("max_price prop");
    assert_eq!(max_price.r#type, proto::Type::Integer as i32);
}

#[test]
fn tool_params_array_with_typed_items() {
    let v = serde_json::json!({
        "type": "array",
        "items": { "type": "string" }
    });

    let schema = tool_parameters_to_proto_schema(&v)
        .expect("schema conversion")
        .expect("schema");
    assert_eq!(schema.r#type, proto::Type::Array as i32);
    let items = schema.items.expect("items");
    assert_eq!(items.r#type, proto::Type::String as i32);
}

#[test]
fn tool_params_enum_strings_preserved() {
    let v = serde_json::json!({
        "type": "string",
        "enum": ["celsius", "fahrenheit"]
    });

    let schema = tool_parameters_to_proto_schema(&v)
        .expect("schema conversion")
        .expect("schema");
    assert_eq!(schema.r#type, proto::Type::String as i32);
    assert_eq!(
        schema.r#enum,
        vec!["celsius".to_string(), "fahrenheit".to_string()]
    );
}

#[test]
fn tool_params_resolves_defs_ref_properties() {
    let v = serde_json::json!({
        "type": "object",
        "properties": {
            "destination": { "$ref": "#/$defs/Destination" }
        },
        "required": ["destination"],
        "$defs": {
            "Destination": {
                "type": "object",
                "properties": {
                    "city": { "type": "string" },
                    "country_code": { "type": "string" }
                },
                "required": ["city"]
            }
        }
    });

    let schema = tool_parameters_to_proto_schema(&v)
        .expect("schema conversion")
        .expect("schema");
    let destination = schema
        .properties
        .get("destination")
        .expect("destination prop");

    assert_eq!(destination.r#type, proto::Type::Object as i32);
    assert_eq!(destination.required, vec!["city".to_string()]);
    assert_eq!(
        destination
            .properties
            .get("city")
            .expect("city prop")
            .r#type,
        proto::Type::String as i32
    );
}

#[test]
fn tool_params_nullable_type_array_preserves_non_null_type() {
    let v = serde_json::json!({
        "type": "object",
        "properties": {
            "nickname": { "type": ["null", "string"] }
        }
    });

    let schema = tool_parameters_to_proto_schema(&v)
        .expect("schema conversion")
        .expect("schema");
    let nickname = schema.properties.get("nickname").expect("nickname prop");

    assert_eq!(nickname.r#type, proto::Type::String as i32);
    assert!(nickname.nullable);
}

#[test]
fn tool_params_any_of_uses_non_null_schema() {
    let v = serde_json::json!({
        "anyOf": [
            { "type": "null" },
            {
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                },
                "required": ["query"]
            }
        ]
    });

    let schema = tool_parameters_to_proto_schema(&v)
        .expect("schema conversion")
        .expect("schema");

    assert_eq!(schema.r#type, proto::Type::Object as i32);
    assert!(schema.nullable);
    assert_eq!(schema.required, vec!["query".to_string()]);
    assert_eq!(
        schema.properties.get("query").expect("query prop").r#type,
        proto::Type::String as i32
    );
}

#[test]
fn tool_params_array_without_items_defaults_to_string_items() {
    let v = serde_json::json!({ "type": "array" });

    let schema = tool_parameters_to_proto_schema(&v)
        .expect("schema conversion")
        .expect("schema");

    assert_eq!(schema.r#type, proto::Type::Array as i32);
    assert_eq!(
        schema.items.expect("items").r#type,
        proto::Type::String as i32
    );
}

/// `FunctionResponse.name` is the executed function's name: read from
/// the required `ToolResult::name` — never an identifier, no matter how
/// identifier-shaped the correlation handles are.
#[test]
fn create_grpc_request_sends_the_executed_name_not_an_identifier() {
    use rig_core::message::{
        AssistantContent, ProviderCallId, ToolCall, ToolCallId, ToolFunction, ToolResult,
        ToolResultContent,
    };

    let call = |wire_id: &str, name: &str| message::Message::Assistant {
        id: None,
        content: vec![AssistantContent::ToolCall(ToolCall::from_wire(
            wire_id,
            ToolFunction {
                name: name.to_owned(),
                arguments: serde_json::json!({}),
            },
        ))],
    };
    let result = |wire_id: &str, name: &str| message::Message::User {
        content: vec![message::UserContent::ToolResult(ToolResult {
            call: ToolCallId::new_or_mint(wire_id),
            provider: ProviderCallId::new(wire_id),
            name: name.to_owned(),
            content: vec![ToolResultContent::text("out")],
        })],
    };

    let req = create_grpc_request(
        "gemini-2.5-flash",
        CompletionRequest {
            model: None,
            chat_history: vec![
                // Driver-built: the executed name travels as data (a
                // repair hook renamed the call: `sum` ran, not `add`).
                call("call_1", "add"),
                result("call_1", "sum"),
                // Cross-provider history with an OpenAI-shaped id —
                // `call_abc` must never travel as the name.
                call("call_abc", "get_weather"),
                result("call_abc", "get_weather"),
            ],
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        },
    )
    .expect("request build");

    // The name is the executed tool's name; the proto `id` is the
    // provider-issued call id (never rig's minted handle).
    let responses: Vec<(&str, &str)> = req
        .contents
        .iter()
        .flat_map(|content| content.parts.iter())
        .filter_map(|part| match &part.data {
            Some(proto::part::Data::FunctionResponse(fr)) => {
                Some((fr.id.as_str(), fr.name.as_str()))
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        responses,
        vec![("call_1", "sum"), ("call_abc", "get_weather")]
    );
}

#[test]
fn create_grpc_request_populates_tool_parameters() {
    use rig_core::completion::ToolDefinition;

    let tool = ToolDefinition {
        name: "get_weather".to_string(),
        description: "Look up the current weather for a city.".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "City name" }
            },
            "required": ["city"]
        }),
    };

    let req = create_grpc_request(
        "gemini-2.5-flash",
        CompletionRequest {
            model: None,
            chat_history: vec![message::Message::user("forecast in Berlin?")],
            documents: Vec::new(),
            tools: vec![tool],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        },
    )
    .expect("request build");

    assert_eq!(req.tools.len(), 1);
    let tool = req.tools.first().expect("tool entry");
    let decl = tool
        .function_declarations
        .first()
        .expect("function declaration");
    assert_eq!(decl.name, "get_weather");

    // The regression in #1710 was `parameters: None` here.
    let params = decl.parameters.as_ref().expect("parameters populated");
    assert_eq!(params.r#type, proto::Type::Object as i32);
    assert_eq!(params.required, vec!["city".to_string()]);
    assert!(params.properties.contains_key("city"));
}

/// The gRPC wire carries the model's chain-of-thought in the same `parts`
/// array as the answer, flagged by `thought` — same shape as the REST
/// wire, where reading it as output text was a live-confirmed defect.
/// There is no cassette harness for this transport (it is protobuf over
/// gRPC, not HTTP), so the wire shape is stated directly.
#[test]
fn text_response_skips_thought_parts() {
    let response = proto::GenerateContentResponse {
        candidates: vec![proto::Candidate {
            content: Some(proto::Content {
                parts: vec![
                    proto::Part {
                        data: Some(proto::part::Data::Text(
                            "Let me work through this...".to_string(),
                        )),
                        thought: true,
                        ..Default::default()
                    },
                    proto::Part {
                        data: Some(proto::part::Data::Text("The answer is 42.".to_string())),
                        thought: false,
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };

    assert_eq!(
        response.text_response().as_deref(),
        Some("The answer is 42."),
        "reasoning must not be reported as the response text"
    );
}

/// The wire hangs a `thoughtSignature` on a trailing part with no
/// `thought` flag. This crate's streaming adapter has always kept it; the
/// unary mapper dropped it, the same asymmetry the REST wire carried. The
/// signature belongs to the chain-of-thought block that precedes it.
#[test]
fn a_trailing_thought_signature_signs_the_reasoning_before_it() {
    let response = proto::GenerateContentResponse {
        candidates: vec![proto::Candidate {
            content: Some(proto::Content {
                parts: vec![
                    proto::Part {
                        data: Some(proto::part::Data::Text("the chain".to_string())),
                        thought: true,
                        ..Default::default()
                    },
                    proto::Part {
                        data: Some(proto::part::Data::Text("answer".to_string())),
                        thought: false,
                        thought_signature: b"sig-bytes".to_vec(),
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };

    let normalized: completion::CompletionResponse =
        response.try_into().expect("payload should normalize");
    assert_eq!(
        normalized.choice.len(),
        2,
        "no empty sibling; got {:?}",
        normalized.choice
    );
    assert!(
        matches!(
            normalized.choice.first(),
            Some(completion::AssistantContent::Reasoning(reasoning))
                if matches!(reasoning.content.first(),
                    Some(message::ReasoningContent::Text { text, signature })
                        if text == "the chain" && signature.is_some())
        ),
        "the reasoning block must carry the trailing signature, got {:?}",
        normalized.choice
    );
}

/// The load-bearing property behind `CompletionResponse::raw` for the
/// gRPC provider: the captured value is
/// `serde_json::to_value(&GenerateContentResponse)` — the prost message
/// `raw_completion` returns, with the serde derives `build.rs` attaches to
/// every generated type — and a consumer must be able to read it back as
/// the same message and get the same JSON. There is no cassette harness
/// for gRPC, so this is the unit-form pin. Fields rig never normalizes
/// (`cached_content_token_count` under `usage_metadata`, the candidate's
/// `finish_message`) survive both directions, and normalizing the
/// restored message agrees with normalizing the original.
#[test]
fn generate_content_response_round_trips_through_serde_json_value() {
    let raw = proto::GenerateContentResponse {
        candidates: vec![proto::Candidate {
            content: Some(proto::Content {
                parts: vec![proto::Part {
                    data: Some(proto::part::Data::Text("hello".to_string())),
                    ..Default::default()
                }],
                role: "model".to_string(),
            }),
            finish_reason: proto::candidate::FinishReason::Stop as i32,
            index: Some(0),
            finish_message: Some("done".to_string()),
        }],
        usage_metadata: Some(proto::UsageMetadata {
            prompt_token_count: 10,
            candidates_token_count: 20,
            total_token_count: 30,
            cached_content_token_count: 4,
        }),
        model_version: "gemini-2.5-flash".to_string(),
        response_id: "resp-grpc-1".to_string(),
        prompt_feedback: None,
    };

    let value = serde_json::to_value(&raw).expect("serialize");
    assert_eq!(
        value.pointer("/usage_metadata/cached_content_token_count"),
        Some(&serde_json::json!(4))
    );
    assert_eq!(
        value.pointer("/candidates/0/finish_message"),
        Some(&serde_json::json!("done"))
    );
    assert_eq!(
        value.pointer("/model_version"),
        Some(&serde_json::json!("gemini-2.5-flash"))
    );

    let back: proto::GenerateContentResponse =
        serde_json::from_value(value.clone()).expect("deserialize");
    assert_eq!(
        serde_json::to_value(&back).expect("re-serialize"),
        value,
        "the capture must read back into GenerateContentResponse and re-serialize identically"
    );
    assert_eq!(back, raw);

    let original: completion::CompletionResponse = raw.try_into().expect("original converts");
    let restored: completion::CompletionResponse = back.try_into().expect("restored converts");
    assert_eq!(restored.identity(), original.identity());
    assert_eq!(restored.finish_reason(), original.finish_reason());
    assert_eq!(restored.model, original.model);
    assert_eq!(restored.usage, original.usage);
    assert_eq!(restored.choice, original.choice);
    assert_eq!(
        restored.identity().response_id.as_deref(),
        Some("resp-grpc-1")
    );
    assert_eq!(
        restored.finish_reason(),
        Some(completion::FinishReason::Stop)
    );
}
