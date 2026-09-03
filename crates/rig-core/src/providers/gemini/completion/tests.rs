use crate::{
    message,
    providers::gemini::completion::gemini_api_types::{
        BlockReason, CitationMetadata, ContentCandidate, FinishReason, FunctionCall,
        GenerateContentResponse, LogprobsResult, ModalityTokenCount, PromptFeedback, Schema,
        TopCandidate, UsageMetadata, flatten_schema, tool_parameters_to_schema,
    },
};

use super::*;
use serde_json::json;

#[test]
fn test_usage_metadata_deserializes_without_total_token_count() {
    // Gemini's proto3-JSON encoding omits fields whose value is the default (0),
    // so `totalTokenCount` is absent on short/empty/blocked generations.
    let usage: UsageMetadata =
        serde_json::from_str(r#"{"promptTokenCount": 12}"#).expect("should deserialize");
    assert_eq!(usage.total_token_count, 0);
    assert_eq!(usage.prompt_token_count, 12);
}

#[test]
fn test_generate_content_response_deserializes_without_candidates_or_response_id() {
    // Blocked prompt responses can omit default-valued proto fields, including
    // empty repeated `candidates` and empty string `responseId`.
    let response: GenerateContentResponse = serde_json::from_value(json!({
        "promptFeedback": {
            "blockReason": "SAFETY"
        }
    }))
    .expect("blocked prompt response should deserialize");

    assert!(response.response_id.is_empty());
    assert!(response.candidates.is_empty());

    let error = completion::CompletionResponse::try_from(response)
        .expect_err("empty candidates should become a response error");
    assert!(error.to_string().contains("No response candidates"));
}

#[test]
fn test_modality_token_count_deserializes_without_zero_token_count() {
    let count: ModalityTokenCount = serde_json::from_value(json!({
        "modality": "TEXT"
    }))
    .expect("zero tokenCount may be omitted");

    assert_eq!(count.token_count, 0);
}

#[test]
fn test_response_metadata_repeated_fields_deserialize_when_omitted() {
    let citation_metadata: CitationMetadata =
        serde_json::from_value(json!({})).expect("empty citation metadata should deserialize");
    assert!(citation_metadata.citation_sources.is_empty());

    let logprobs: LogprobsResult =
        serde_json::from_value(json!({})).expect("empty logprobs result should deserialize");
    assert!(logprobs.top_candidates.is_empty());
    assert_eq!(logprobs.log_probability_sum, None);
    assert!(logprobs.chosen_candidates.is_empty());

    let top_candidate: TopCandidate =
        serde_json::from_value(json!({})).expect("empty top candidate should deserialize");
    assert!(top_candidate.candidates.is_empty());
}

#[test]
fn test_logprobs_result_deserializes_official_json_field_names() {
    let logprobs: LogprobsResult = serde_json::from_value(json!({
        "topCandidates": [
            {
                "candidates": [
                    {
                        "token": "Hello",
                        "tokenId": 123,
                        "logProbability": -0.1
                    },
                    {
                        "token": "Hi",
                        "tokenId": 124,
                        "logProbability": -1.25
                    }
                ]
            }
        ],
        "logProbabilitySum": -0.1,
        "chosenCandidates": [
            {
                "token": "Hello",
                "tokenId": 123,
                "logProbability": -0.1
            }
        ]
    }))
    .expect("official Gemini logprobs result should deserialize");

    assert_eq!(logprobs.top_candidates.len(), 1);
    assert_eq!(logprobs.top_candidates[0].candidates.len(), 2);
    assert_eq!(
        logprobs.top_candidates[0].candidates[0].token.as_deref(),
        Some("Hello")
    );
    assert_eq!(logprobs.top_candidates[0].candidates[0].token_id, Some(123));
    assert_eq!(
        logprobs.top_candidates[0].candidates[0].log_probability,
        Some(-0.1)
    );
    assert_eq!(logprobs.log_probability_sum, Some(-0.1));
    assert_eq!(logprobs.chosen_candidates.len(), 1);
    assert_eq!(
        logprobs.chosen_candidates[0].token.as_deref(),
        Some("Hello")
    );
    assert_eq!(logprobs.chosen_candidates[0].token_id, Some(123));
    assert_eq!(logprobs.chosen_candidates[0].log_probability, Some(-0.1));
}

#[test]
fn test_resolve_request_model_uses_override() {
    let request = CompletionRequest {
        model: Some("gemini-2.5-flash".to_string()),
        chat_history: vec!["Hello".into()],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    let request_model = resolve_request_model("gemini-2.0-flash", &request);
    assert_eq!(request_model, "gemini-2.5-flash");
    assert_eq!(
        completion_endpoint(&request_model),
        "/v1beta/models/gemini-2.5-flash:generateContent"
    );
    assert_eq!(
        streaming_endpoint(&request_model),
        "/v1beta/models/gemini-2.5-flash:streamGenerateContent"
    );
}

#[test]
fn test_resolve_request_model_uses_default_when_unset() {
    let request = CompletionRequest {
        model: None,
        chat_history: vec!["Hello".into()],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    assert_eq!(
        resolve_request_model("gemini-2.0-flash", &request),
        "gemini-2.0-flash"
    );
}

#[test]
fn test_deserialize_message_user() {
    let raw_message = r#"{
            "parts": [
                {"text": "Hello, world!"},
                {"inlineData": {"mimeType": "image/png", "data": "base64encodeddata"}},
                {"functionCall": {"name": "test_function", "args": {"arg1": "value1"}}},
                {"functionResponse": {"name": "test_function", "response": {"result": "success"}}},
                {"fileData": {"mimeType": "application/pdf", "fileUri": "http://example.com/file.pdf"}},
                {"executableCode": {"code": "print('Hello, world!')", "language": "PYTHON"}},
                {"codeExecutionResult": {"output": "Hello, world!", "outcome": "OUTCOME_OK"}}
            ],
            "role": "user"
        }"#;

    let content: Content = {
        let jd = &mut serde_json::Deserializer::from_str(raw_message);
        serde_path_to_error::deserialize(jd).unwrap_or_else(|err| {
            panic!("Deserialization error at {}: {}", err.path(), err);
        })
    };
    assert_eq!(content.role, Some(Role::User));
    assert_eq!(content.parts.len(), 7);

    let parts: Vec<Part> = content.parts.into_iter().collect();

    if let Part {
        part: PartKind::Text(text),
        ..
    } = &parts[0]
    {
        assert_eq!(text, "Hello, world!");
    } else {
        panic!("Expected text part");
    }

    if let Part {
        part: PartKind::InlineData(inline_data),
        ..
    } = &parts[1]
    {
        assert_eq!(inline_data.mime_type, "image/png");
        assert_eq!(inline_data.data, "base64encodeddata");
    } else {
        panic!("Expected inline data part");
    }

    if let Part {
        part: PartKind::FunctionCall(function_call),
        ..
    } = &parts[2]
    {
        assert_eq!(function_call.name, "test_function");
        assert_eq!(
            function_call.args.as_object().unwrap().get("arg1").unwrap(),
            "value1"
        );
    } else {
        panic!("Expected function call part");
    }

    if let Part {
        part: PartKind::FunctionResponse(function_response),
        ..
    } = &parts[3]
    {
        assert_eq!(function_response.name, "test_function");
        assert_eq!(
            function_response
                .response
                .as_ref()
                .unwrap()
                .get("result")
                .unwrap(),
            "success"
        );
    } else {
        panic!("Expected function response part");
    }

    if let Part {
        part: PartKind::FileData(file_data),
        ..
    } = &parts[4]
    {
        assert_eq!(file_data.mime_type.as_ref().unwrap(), "application/pdf");
        assert_eq!(file_data.file_uri, "http://example.com/file.pdf");
    } else {
        panic!("Expected file data part");
    }

    if let Part {
        part: PartKind::ExecutableCode(executable_code),
        ..
    } = &parts[5]
    {
        assert_eq!(executable_code.code, "print('Hello, world!')");
    } else {
        panic!("Expected executable code part");
    }

    if let Part {
        part: PartKind::CodeExecutionResult(code_execution_result),
        ..
    } = &parts[6]
    {
        assert_eq!(
            code_execution_result.clone().output.unwrap(),
            "Hello, world!"
        );
    } else {
        panic!("Expected code execution result part");
    }
}

#[test]
fn test_deserialize_message_model() {
    let json_data = json!({
        "parts": [{"text": "Hello, user!"}],
        "role": "model"
    });

    let content: Content = serde_json::from_value(json_data).unwrap();
    assert_eq!(content.role, Some(Role::Model));
    assert_eq!(content.parts.len(), 1);
    if let Some(Part {
        part: PartKind::Text(text),
        ..
    }) = content.parts.first()
    {
        assert_eq!(text, "Hello, user!");
    } else {
        panic!("Expected text part");
    }
}

#[test]
fn test_message_conversion_user() {
    let msg = message::Message::user("Hello, world!");
    let content: Content = msg.try_into().unwrap();
    assert_eq!(content.role, Some(Role::User));
    assert_eq!(content.parts.len(), 1);
    if let Some(Part {
        part: PartKind::Text(text),
        ..
    }) = &content.parts.first()
    {
        assert_eq!(text, "Hello, world!");
    } else {
        panic!("Expected text part");
    }
}

#[test]
fn test_message_conversion_model() {
    let msg = message::Message::assistant("Hello, user!");

    let content: Content = msg.try_into().unwrap();
    assert_eq!(content.role, Some(Role::Model));
    assert_eq!(content.parts.len(), 1);
    if let Some(Part {
        part: PartKind::Text(text),
        ..
    }) = &content.parts.first()
    {
        assert_eq!(text, "Hello, user!");
    } else {
        panic!("Expected text part");
    }
}

#[test]
fn test_thought_signature_is_preserved_from_response_reasoning_part() {
    let response = GenerateContentResponse {
        response_id: "resp_1".to_string(),
        candidates: vec![ContentCandidate {
            content: Some(Content {
                parts: vec![Part {
                    thought: Some(true),
                    thought_signature: Some("thought_sig_123".to_string()),
                    part: PartKind::Text("thinking text".to_string()),
                    additional_params: None,
                }],
                role: Some(Role::Model),
            }),
            finish_reason: Some(FinishReason::Stop),
            safety_ratings: None,
            citation_metadata: None,
            token_count: None,
            avg_logprobs: None,
            logprobs_result: None,
            index: Some(0),
            finish_message: None,
        }],
        prompt_feedback: None,
        usage_metadata: None,
        model_version: None,
    };

    let converted: crate::completion::CompletionResponse =
        response.try_into().expect("convert response");
    let first = converted.choice.first();
    assert!(matches!(
        first,
        Some(message::AssistantContent::Reasoning(message::Reasoning { content, .. }))
            if matches!(
                content.first(),
                Some(message::ReasoningContent::Text {
                    text,
                    signature: Some(signature)
                }) if text == "thinking text" && signature == "thought_sig_123"
            )
    ));
}

#[test]
fn test_tool_protocol_finish_reason_returns_response_error() {
    for (reason, finish_message) in [
        (
            FinishReason::MalformedFunctionCall,
            "malformed function call: default_api",
        ),
        (
            FinishReason::UnexpectedToolCall,
            "unexpected tool call: default_api",
        ),
        (
            FinishReason::MissingThoughtSignature,
            "missing thought signature for tool call",
        ),
        (
            FinishReason::TooManyToolCalls,
            "too many tool calls in response",
        ),
        (
            FinishReason::MalformedResponse,
            "malformed response from provider",
        ),
    ] {
        let reason_name = format!("{reason:?}");
        let response = GenerateContentResponse {
            response_id: "resp_tool_protocol_error".to_string(),
            candidates: vec![ContentCandidate {
                content: Some(Content {
                    parts: vec![Part {
                        thought: None,
                        thought_signature: None,
                        part: PartKind::FunctionCall(FunctionCall {
                            name: "default_api".to_string(),
                            args: json!({"x": 1}),
                            id: None,
                        }),
                        additional_params: None,
                    }],
                    role: Some(Role::Model),
                }),
                finish_reason: Some(reason),
                safety_ratings: None,
                citation_metadata: None,
                token_count: None,
                avg_logprobs: None,
                logprobs_result: None,
                index: Some(0),
                finish_message: Some(finish_message.to_string()),
            }],
            prompt_feedback: None,
            usage_metadata: None,
            model_version: None,
        };

        let err = crate::completion::CompletionResponse::try_from(response)
            .expect_err("tool protocol finish reason should fail");

        assert!(matches!(
            err,
            CompletionError::ResponseError(message)
                if message.contains(&reason_name)
                    && message.contains(finish_message)
        ));
    }
}

#[test]
fn test_completion_response_usage_preserves_cached_and_reasoning_tokens() {
    let response = GenerateContentResponse {
        response_id: "resp_1".to_string(),
        candidates: vec![ContentCandidate {
            content: Some(Content {
                parts: vec![Part {
                    thought: None,
                    thought_signature: None,
                    part: PartKind::Text("answer".to_string()),
                    additional_params: None,
                }],
                role: Some(Role::Model),
            }),
            finish_reason: Some(FinishReason::Stop),
            safety_ratings: None,
            citation_metadata: None,
            token_count: None,
            avg_logprobs: None,
            logprobs_result: None,
            index: Some(0),
            finish_message: None,
        }],
        prompt_feedback: None,
        usage_metadata: Some(UsageMetadata {
            prompt_token_count: 40,
            cached_content_token_count: Some(20),
            candidates_token_count: Some(30),
            total_token_count: 100,
            thoughts_token_count: Some(10),
            prompt_tokens_details: None,
            cache_tokens_details: None,
            candidates_tokens_details: None,
            tool_use_prompt_token_count: Some(12),
            tool_use_prompt_tokens_details: None,
            traffic_type: None,
        }),
        model_version: Some("gemini-2.0-flash-001".to_string()),
    };

    let converted: crate::completion::CompletionResponse =
        response.try_into().expect("convert response");

    assert_eq!(converted.usage.input_tokens, 40);
    assert_eq!(converted.usage.cached_input_tokens, 20);
    assert_eq!(converted.usage.output_tokens, 30);
    assert_eq!(converted.usage.reasoning_tokens, 10);
    assert_eq!(converted.usage.tool_use_prompt_tokens, 12);
    assert_eq!(converted.usage.total_tokens, 100);
}

#[test]
fn test_finish_reason_maps_every_wire_variant() {
    use crate::completion::FinishReason as Normalized;

    for (wire, expected) in [
        (FinishReason::Stop, Normalized::Stop),
        (FinishReason::MaxTokens, Normalized::Length),
        (FinishReason::Safety, Normalized::ContentFilter),
        (FinishReason::Blocklist, Normalized::ContentFilter),
        (FinishReason::ProhibitedContent, Normalized::ContentFilter),
        (FinishReason::Spii, Normalized::ContentFilter),
        // Everything Gemini reports that rig does not model survives in the
        // provider's own SCREAMING_SNAKE_CASE spelling.
        (
            FinishReason::Recitation,
            Normalized::Other("RECITATION".to_string()),
        ),
        (
            FinishReason::Language,
            Normalized::Other("LANGUAGE".to_string()),
        ),
        (FinishReason::Other, Normalized::Other("OTHER".to_string())),
        (
            FinishReason::MalformedFunctionCall,
            Normalized::Other("MALFORMED_FUNCTION_CALL".to_string()),
        ),
        (
            FinishReason::UnexpectedToolCall,
            Normalized::Other("UNEXPECTED_TOOL_CALL".to_string()),
        ),
        (
            FinishReason::MissingThoughtSignature,
            Normalized::Other("MISSING_THOUGHT_SIGNATURE".to_string()),
        ),
        (
            FinishReason::TooManyToolCalls,
            Normalized::Other("TOO_MANY_TOOL_CALLS".to_string()),
        ),
        (
            FinishReason::MalformedResponse,
            Normalized::Other("MALFORMED_RESPONSE".to_string()),
        ),
    ] {
        assert_eq!(
            map_finish_reason(&wire),
            Some(expected),
            "wire reason {wire:?}"
        );
    }

    // The proto default means Gemini reported no reason; both the REST and
    // gRPC mappers treat it as absent rather than an `Other` value.
    assert_eq!(
        map_finish_reason(&FinishReason::FinishReasonUnspecified),
        None
    );
}

#[test]
fn test_finish_reason_wire_spelling_matches_serde() {
    // `as_wire_str` is hand-written; keep it honest against the serde
    // representation the same enum deserializes from.
    for reason in [
        FinishReason::FinishReasonUnspecified,
        FinishReason::Stop,
        FinishReason::MaxTokens,
        FinishReason::Safety,
        FinishReason::Recitation,
        FinishReason::Language,
        FinishReason::Other,
        FinishReason::Blocklist,
        FinishReason::ProhibitedContent,
        FinishReason::Spii,
        FinishReason::MalformedFunctionCall,
        FinishReason::UnexpectedToolCall,
        FinishReason::MissingThoughtSignature,
        FinishReason::TooManyToolCalls,
        FinishReason::MalformedResponse,
    ] {
        let serialized = serde_json::to_value(&reason).expect("reason should serialize");
        assert_eq!(serialized, json!(reason.as_wire_str()));
    }
}

#[test]
fn test_unknown_finish_reason_round_trips_verbatim() {
    // A wire value this crate does not know must land in `Unknown` with
    // the provider's spelling intact — and serialize back to the same
    // string — so nothing is lost between deserialize and re-serialize.
    let reason: FinishReason = serde_json::from_value(json!("FINISH_REASON_FUTURE"))
        .expect("unknown finish reason should deserialize");
    assert!(matches!(&reason, FinishReason::Unknown(s) if s == "FINISH_REASON_FUTURE"));
    assert_eq!(reason.as_wire_str(), "FINISH_REASON_FUTURE");
    assert_eq!(
        serde_json::to_value(&reason).expect("reason should serialize"),
        json!("FINISH_REASON_FUTURE")
    );
    assert_eq!(
        map_finish_reason(&reason),
        Some(crate::completion::FinishReason::Other(
            "FINISH_REASON_FUTURE".to_string()
        ))
    );
}

#[test]
fn test_unknown_block_reason_deserializes_verbatim() {
    // Same contract for prompt feedback: a new block reason must not fail
    // the payload, and the spelling is preserved.
    let feedback: PromptFeedback = serde_json::from_value(json!({
        "blockReason": "BLOCK_REASON_FUTURE"
    }))
    .expect("unknown block reason should deserialize");
    assert!(matches!(
        feedback.block_reason,
        Some(BlockReason::Unknown(ref s)) if s == "BLOCK_REASON_FUTURE"
    ));
}

#[test]
fn test_unary_response_with_unknown_finish_reason_stays_parseable() {
    // A finish reason Google ships tomorrow must not fail the whole
    // payload: content and usage stay intact, and the reason maps to
    // `Other` verbatim — matching the gRPC crate's handling of unknowns.
    let response: GenerateContentResponse = serde_json::from_value(json!({
        "responseId": "resp-future",
        "candidates": [{
            "content": {
                "parts": [{"text": "hi"}],
                "role": "model"
            },
            "finishReason": "FINISH_REASON_FUTURE"
        }],
        "usageMetadata": {
            "promptTokenCount": 3,
            "candidatesTokenCount": 2,
            "totalTokenCount": 5
        }
    }))
    .expect("unknown finish reason should not fail the payload");

    let converted: crate::completion::CompletionResponse =
        response.try_into().expect("convert response");

    assert!(matches!(
        converted.choice.first(),
        Some(message::AssistantContent::Text(text)) if text.text == "hi"
    ));
    assert_eq!(converted.usage.total_tokens, 5);
    assert_eq!(
        converted.finish_reason(),
        Some(crate::completion::FinishReason::Other(
            "FINISH_REASON_FUTURE".to_string()
        ))
    );
}

#[test]
fn test_streaming_candidate_with_unknown_finish_reason_stays_parseable() {
    // Streaming terminal chunks embed the same `ContentCandidate`; an
    // unknown reason must leave the chunk deserializable so the terminal
    // record is still produced.
    let candidate: ContentCandidate = serde_json::from_value(json!({
        "content": {
            "parts": [{"text": "done"}],
            "role": "model"
        },
        "finishReason": "FINISH_REASON_FUTURE"
    }))
    .expect("unknown finish reason should not fail the chunk");

    let reason = candidate.finish_reason.expect("finish reason present");
    assert_eq!(
        map_finish_reason(&reason),
        Some(crate::completion::FinishReason::Other(
            "FINISH_REASON_FUTURE".to_string()
        ))
    );
}

#[test]
fn test_completion_response_carries_normalized_metadata() {
    let response: GenerateContentResponse = serde_json::from_value(json!({
        "responseId": "resp-meta",
        "modelVersion": "gemini-2.0-flash-001",
        "candidates": [{
            "content": {
                "parts": [{"text": "hi"}],
                "role": "model"
            },
            "finishReason": "MAX_TOKENS"
        }]
    }))
    .expect("response should deserialize");

    let converted: crate::completion::CompletionResponse =
        response.try_into().expect("convert response");

    assert_eq!(converted.provider, PROVIDER_NAME);
    assert_eq!(converted.model.as_deref(), Some("gemini-2.0-flash-001"));
    assert_eq!(converted.response_id.as_deref(), Some("resp-meta"));
    assert_eq!(converted.message_id, None);
    assert_eq!(
        converted.finish_reason(),
        Some(crate::completion::FinishReason::Length)
    );
}

#[test]
fn test_completion_response_upgrades_stop_to_tool_calls() {
    // Gemini reports STOP on turns that only emitted a function call; the
    // normalized response must still say `ToolCalls`.
    let response: GenerateContentResponse = serde_json::from_value(json!({
        "responseId": "resp-tool",
        "candidates": [{
            "content": {
                "parts": [{
                    "functionCall": {
                        "name": "get_weather",
                        "args": {"city": "Paris"}
                    }
                }],
                "role": "model"
            },
            "finishReason": "STOP"
        }]
    }))
    .expect("response should deserialize");

    let converted: crate::completion::CompletionResponse =
        response.try_into().expect("convert response");

    assert_eq!(
        converted.finish_reason(),
        Some(crate::completion::FinishReason::ToolCalls)
    );
    assert_eq!(converted.model, None);
}

#[test]
fn test_reasoning_signature_is_emitted_in_gemini_part() {
    let msg = message::Message::Assistant {
        id: None,
        content: vec![message::AssistantContent::Reasoning(
            message::Reasoning::new_with_signature(
                "structured thought",
                Some("reuse_sig_456".to_string()),
            ),
        )],
    };

    let converted: Content = msg.try_into().expect("convert message");
    let first = converted.parts.first().expect("reasoning part");
    assert_eq!(first.thought, Some(true));
    assert_eq!(first.thought_signature.as_deref(), Some("reuse_sig_456"));
    assert!(matches!(
        &first.part,
        PartKind::Text(text) if text == "structured thought"
    ));
}

#[test]
fn test_message_conversion_tool_call() {
    let tool_call = message::ToolCall::from_wire(
        "call-123",
        message::ToolFunction {
            name: "test_function".to_string(),
            arguments: json!({"arg1": "value1"}),
        },
    );

    let msg = message::Message::Assistant {
        id: None,
        content: vec![message::AssistantContent::ToolCall(tool_call)],
    };

    let content: Content = msg.try_into().unwrap();
    assert_eq!(content.role, Some(Role::Model));
    assert_eq!(content.parts.len(), 1);
    if let Some(Part {
        part: PartKind::FunctionCall(function_call),
        ..
    }) = content.parts.first()
    {
        assert_eq!(function_call.name, "test_function");
        assert_eq!(
            function_call.args.as_object().unwrap().get("arg1").unwrap(),
            "value1"
        );
        assert_eq!(function_call.id.as_deref(), Some("call-123"));
    } else {
        panic!("Expected function call part");
    }
}

#[test]
fn test_response_function_call_preserves_correlation_id() {
    let response: GenerateContentResponse = serde_json::from_value(json!({
        "responseId": "response-123",
        "candidates": [{
            "content": {
                "parts": [{
                    "functionCall": {
                        "name": "test_function",
                        "args": {"arg1": "value1"},
                        "id": "call-123"
                    }
                }],
                "role": "model"
            },
            "finishReason": "STOP"
        }]
    }))
    .expect("response should deserialize");

    let converted: crate::completion::CompletionResponse =
        response.try_into().expect("response should convert");
    let Some(message::AssistantContent::ToolCall(tool_call)) = converted.choice.first() else {
        panic!("expected a tool call");
    };
    assert_eq!(tool_call.id, "call-123");
    assert_eq!(
        tool_call.provider.as_ref().expect("wire id").call_id,
        "call-123"
    );
}

#[test]
fn test_vec_schema_conversion() {
    let schema_with_ref = json!({
        "type": "array",
        "items": {
            "$ref": "#/$defs/Person"
        },
        "$defs": {
            "Person": {
                "type": "object",
                "properties": {
                    "first_name": {
                        "type": ["string", "null"],
                        "description": "The person's first name, if provided (null otherwise)"
                    },
                    "last_name": {
                        "type": ["string", "null"],
                        "description": "The person's last name, if provided (null otherwise)"
                    },
                    "job": {
                        "type": ["string", "null"],
                        "description": "The person's job, if provided (null otherwise)"
                    }
                },
                "required": []
            }
        }
    });

    let result: Result<Schema, _> = schema_with_ref.try_into();

    match result {
        Ok(schema) => {
            assert_eq!(schema.r#type, "array");

            if let Some(items) = schema.items {
                println!("item types: {}", items.r#type);

                assert_ne!(items.r#type, "", "Items type should not be empty string!");
                assert_eq!(items.r#type, "object", "Items should be object type");
            } else {
                panic!("Schema should have items field for array type");
            }
        }
        Err(e) => println!("Schema conversion failed: {e:?}"),
    }
}

#[test]
fn test_object_schema() {
    let simple_schema = json!({
        "type": "object",
        "properties": {
            "name": {
                "type": "string"
            }
        }
    });

    let schema: Schema = simple_schema.try_into().unwrap();
    assert_eq!(schema.r#type, "object");
    assert!(schema.properties.is_some());
}

#[test]
fn test_array_with_inline_items() {
    let inline_schema = json!({
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string"
                }
            }
        }
    });

    let schema: Schema = inline_schema.try_into().unwrap();
    assert_eq!(schema.r#type, "array");

    if let Some(items) = schema.items {
        assert_eq!(items.r#type, "object");
        assert!(items.properties.is_some());
    } else {
        panic!("Schema should have items field");
    }
}
#[test]
fn test_flattened_schema() {
    let ref_schema = json!({
        "type": "array",
        "items": {
            "$ref": "#/$defs/Person"
        },
        "$defs": {
            "Person": {
                "type": "object",
                "properties": {
                    "name": { "type": "string" }
                }
            }
        }
    });

    let flattened = flatten_schema(ref_schema).unwrap();
    let schema: Schema = flattened.try_into().unwrap();

    assert_eq!(schema.r#type, "array");

    if let Some(items) = schema.items {
        println!("Flattened items type: '{}'", items.r#type);

        assert_eq!(items.r#type, "object");
        assert!(items.properties.is_some());
    }
}

#[test]
fn test_array_without_items_gets_default() {
    let schema_json = json!({
        "type": "object",
        "properties": {
            "service_ids": {
                "type": "array",
                "description": "A list of service IDs"
            }
        }
    });

    let schema: Schema = schema_json.try_into().unwrap();
    let props = schema.properties.unwrap();
    let service_ids = props.get("service_ids").unwrap();
    assert_eq!(service_ids.r#type, "array");
    let items = service_ids
        .items
        .as_ref()
        .expect("array schema missing items should get a default");
    assert_eq!(items.r#type, "string");
}

#[test]
fn test_tool_parameters_to_schema_maps_no_arg_tool_to_none() {
    let schema = tool_parameters_to_schema(json!({"type": "object", "properties": {}}))
        .expect("schema conversion");

    assert!(schema.is_none());
}

#[test]
fn test_tool_parameters_to_schema_resolves_defs_ref() {
    let schema_json = json!({
        "type": "object",
        "properties": {
            "destination": { "$ref": "#/$defs/Destination" }
        },
        "required": ["destination"],
        "$defs": {
            "Destination": {
                "type": "object",
                "properties": {
                    "city": { "type": "string" }
                },
                "required": ["city"]
            }
        }
    });

    let schema = tool_parameters_to_schema(schema_json)
        .expect("schema conversion")
        .expect("schema");
    let props = schema.properties.expect("properties");
    let destination = props.get("destination").expect("destination prop");

    assert_eq!(destination.r#type, "object");
    assert_eq!(destination.required, Some(vec!["city".to_string()]));
}

#[test]
fn test_tool_parameters_to_schema_handles_nullable_type_arrays() {
    let schema_json = json!({
        "type": "object",
        "properties": {
            "nickname": { "type": ["null", "string"] }
        }
    });

    let schema = tool_parameters_to_schema(schema_json)
        .expect("schema conversion")
        .expect("schema");
    let props = schema.properties.expect("properties");
    let nickname = props.get("nickname").expect("nickname prop");

    assert_eq!(nickname.r#type, "string");
    assert_eq!(nickname.nullable, Some(true));
}

#[test]
fn test_txt_document_conversion_to_text_part() {
    // Test that TXT documents are converted to plain text parts, not inline data
    use crate::message::{DocumentMediaType, UserContent};

    let doc = UserContent::document(
        "Note: test.md\nPath: /test.md\nContent: Hello World!",
        Some(DocumentMediaType::TXT),
    );

    let content: Content = message::Message::User { content: vec![doc] }
        .try_into()
        .unwrap();

    if let Part {
        part: PartKind::Text(text),
        ..
    } = &content.parts[0]
    {
        assert!(text.contains("Note: test.md"));
        assert!(text.contains("Hello World!"));
    } else {
        panic!(
            "Expected text part for TXT document, got: {:?}",
            content.parts[0]
        );
    }
}

#[test]
fn test_tool_result_with_image_content() {
    // Test that a ToolResult with image content converts correctly to Gemini's Part format
    use crate::message::{
        DocumentSourceKind, Image, ImageMediaType, ToolResult, ToolResultContent,
    };

    // Create a tool result with both text and image content
    let tool_result = ToolResult {
        call: message::ToolCallId::new_or_minted("call-123", 0),
        provider: message::ProviderCallId::new("call-123"),
        name: "test_tool".to_string(),
        content: vec![
            ToolResultContent::Text(message::Text::new(r#"{"status": "success"}"#.to_string())),
            ToolResultContent::Image(Image {
                data: DocumentSourceKind::Base64("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==".to_string()),
                media_type: Some(ImageMediaType::PNG),
                detail: None,
                additional_params: None,
            }),
        ],
    };

    let user_content = message::UserContent::ToolResult(tool_result);
    let msg = message::Message::User {
        content: vec![user_content],
    };

    // Convert to Gemini Content
    let content: Content = msg.try_into().expect("Should convert to Gemini Content");
    assert_eq!(content.role, Some(Role::User));
    assert_eq!(content.parts.len(), 1);

    // Verify the part is a FunctionResponse with both response and parts
    if let Some(Part {
        part: PartKind::FunctionResponse(function_response),
        ..
    }) = content.parts.first()
    {
        assert_eq!(function_response.name, "test_tool");
        assert_eq!(function_response.id.as_deref(), Some("call-123"));

        // Check that response JSON is present
        assert!(function_response.response.is_some());
        let response = function_response.response.as_ref().unwrap();
        assert_eq!(
            response,
            &json!({
                "result": r#"{"status": "success"}"#
            })
        );

        // Check that parts with image data are present
        assert!(function_response.parts.is_some());
        let parts = function_response.parts.as_ref().unwrap();
        assert_eq!(parts.len(), 1);

        let image_part = &parts[0];
        assert!(image_part.inline_data.is_some());
        let inline_data = image_part.inline_data.as_ref().unwrap();
        assert_eq!(inline_data.mime_type, "image/png");
        assert!(!inline_data.data.is_empty());
        assert_eq!(inline_data.display_name, None);
    } else {
        panic!("Expected FunctionResponse part");
    }
}

#[test]
fn mixed_inline_images_and_text_keep_text_response_and_ordered_parts() {
    use crate::message::{ImageMediaType, ToolResult, ToolResultContent};

    let message = message::Message::User {
        content: vec![message::UserContent::ToolResult(ToolResult {
            call: message::ToolCallId::minted(0),
            provider: None,
            name: "ordered_tool".to_string(),
            content: vec![
                ToolResultContent::image_base64("first-image", Some(ImageMediaType::PNG), None),
                ToolResultContent::text("between-images"),
                ToolResultContent::image_base64("second-image", Some(ImageMediaType::JPEG), None),
            ],
        })],
    };

    let content: Content = message.try_into().expect("tool result should convert");
    let PartKind::FunctionResponse(response) = &content.parts[0].part else {
        panic!("expected a function response");
    };

    assert_eq!(
        response.response,
        Some(json!({ "result": "between-images" }))
    );

    let parts = response
        .parts
        .as_ref()
        .expect("images should be inline parts");
    assert_eq!(parts.len(), 2);
    let first = parts[0].inline_data.as_ref().expect("first inline image");
    assert_eq!(first.mime_type, "image/png");
    assert_eq!(first.data, "first-image");
    assert_eq!(first.display_name, None);
    let second = parts[1].inline_data.as_ref().expect("second inline image");
    assert_eq!(second.mime_type, "image/jpeg");
    assert_eq!(second.data, "second-image");
    assert_eq!(second.display_name, None);
}

#[test]
fn mixed_inline_image_and_json_keep_structured_value_and_media_part() {
    use crate::message::{ImageMediaType, ToolResult, ToolResultContent};

    let message = message::Message::User {
        content: vec![message::UserContent::ToolResult(ToolResult {
            call: message::ToolCallId::minted(0),
            provider: None,
            name: "ordered_tool".to_string(),
            content: vec![
                ToolResultContent::json(json!({ "status": "ok" })),
                ToolResultContent::image_base64("image-data", Some(ImageMediaType::PNG), None),
            ],
        })],
    };

    let content: Content = message.try_into().expect("tool result should convert");
    let PartKind::FunctionResponse(response) = &content.parts[0].part else {
        panic!("expected a function response");
    };

    assert_eq!(
        response.response,
        Some(json!({ "result": { "status": "ok" } }))
    );
    let parts = response
        .parts
        .as_ref()
        .expect("image should be an inline part");
    assert_eq!(parts.len(), 1);
    let inline_data = parts[0].inline_data.as_ref().expect("inline image data");
    assert_eq!(inline_data.data, "image-data");
    assert_eq!(inline_data.display_name, None);
}

#[test]
fn mixed_url_image_and_response_value_is_rejected() {
    use crate::message::{DocumentSourceKind, Image, ImageMediaType, ToolResultContent};

    let tool_result = message::Message::User {
        content: vec![message::UserContent::ToolResult(message::ToolResult {
            call: message::ToolCallId::minted(0),
            provider: None,
            name: "url_tool".to_string(),
            content: vec![
                ToolResultContent::Image(Image {
                    data: DocumentSourceKind::Url("https://example.com/image.png".to_string()),
                    media_type: Some(ImageMediaType::PNG),
                    detail: None,
                    additional_params: None,
                }),
                ToolResultContent::text("after-image"),
            ],
        })],
    };

    let error = Content::try_from(tool_result)
        .expect_err("URL-backed tool result images should be rejected");
    assert!(
        error
            .to_string()
            .contains("URL-backed images are not supported"),
        "unexpected error: {error}"
    );
}

#[test]
fn tool_result_rejects_unsupported_image_media_types() {
    use crate::message::{ImageMediaType, ToolResult, ToolResultContent};

    for media_type in [
        ImageMediaType::GIF,
        ImageMediaType::HEIC,
        ImageMediaType::HEIF,
        ImageMediaType::SVG,
    ] {
        let message = message::Message::User {
            content: vec![message::UserContent::ToolResult(ToolResult {
                call: message::ToolCallId::minted(0),
                provider: None,
                name: "image_tool".to_string(),
                content: vec![ToolResultContent::image_base64(
                    "image-data",
                    Some(media_type),
                    None,
                )],
            })],
        };

        let error = Content::try_from(message)
            .expect_err("unsupported tool result image type should be rejected");
        assert!(
            error
                .to_string()
                .contains("supported types are JPEG, PNG, and WEBP"),
            "unexpected error: {error}"
        );
    }
}

#[test]
fn structured_json_refs_remain_literal_with_unreferenced_image_parts() {
    use crate::message::{ImageMediaType, ToolResult, ToolResultContent};

    let message = message::Message::User {
        content: vec![message::UserContent::ToolResult(ToolResult {
            call: message::ToolCallId::minted(0),
            provider: None,
            name: "collision_tool".to_string(),
            content: vec![
                ToolResultContent::json(json!({
                    "literal": {
                        "$ref": "tool_result_image_0"
                    }
                })),
                ToolResultContent::image_base64("image-data", Some(ImageMediaType::PNG), None),
            ],
        })],
    };

    let content: Content = message.try_into().expect("tool result should convert");
    let PartKind::FunctionResponse(response) = &content.parts[0].part else {
        panic!("expected a function response");
    };

    assert_eq!(
        response.response,
        Some(json!({
            "result": {
                "literal": {
                    "$ref": "tool_result_image_0"
                }
            }
        }))
    );
    assert_eq!(
        response.parts.as_ref().and_then(|parts| {
            parts
                .first()
                .and_then(|part| part.inline_data.as_ref())
                .and_then(|part| part.display_name.as_deref())
        }),
        None
    );
}

#[test]
fn tool_result_literal_text_and_structured_json_remain_distinct() {
    use crate::message::{ToolResult, ToolResultContent};

    let cases = [
        (
            ToolResultContent::text(r#"{"status":"ok"}"#),
            json!({ "result": "{\"status\":\"ok\"}" }),
        ),
        (
            ToolResultContent::json(json!({ "status": "ok" })),
            json!({ "result": { "status": "ok" } }),
        ),
    ];

    for (tool_content, expected) in cases {
        let message = message::Message::User {
            content: vec![message::UserContent::ToolResult(ToolResult {
                call: message::ToolCallId::minted(0),
                provider: None,
                name: "test_tool".to_string(),
                content: vec![tool_content],
            })],
        };
        let content: Content = message.try_into().expect("tool result should convert");

        let PartKind::FunctionResponse(response) = &content.parts[0].part else {
            panic!("expected a function response");
        };
        assert_eq!(response.response.as_ref(), Some(&expected));
    }
}

/// A consumer echoing a minted `ToolCall::id` through
/// `tool_result()` must not put that handle on Gemini's wire: the
/// paired functionCall omitted its id (the provider issued none), and
/// an asymmetric functionCall/functionResponse id pair is rejected.
#[test]
fn echoed_minted_handle_never_reaches_the_function_response_id() {
    use crate::message::{ToolCall, ToolCallId, ToolFunction, ToolResultContent};

    // An id-less wire minted the handle (Gemini REST issued no id).
    let call = ToolCall::new(
        ToolCallId::minted(0),
        ToolFunction {
            name: "lookup".to_string(),
            arguments: json!({}),
        },
    );

    let message = message::Message::User {
        content: vec![message::UserContent::tool_result(
            call.id.as_str(),
            "lookup",
            vec![ToolResultContent::text("out")],
        )],
    };
    let content: Content = message.try_into().expect("tool result should convert");
    let PartKind::FunctionResponse(response) = &content.parts[0].part else {
        panic!("expected a function response");
    };
    assert_eq!(response.id, None);
}

/// A cross-provider ingested transcript (rig's inbound converters
/// stamp `name: ""` — Anthropic/OpenAI-chat/Cohere/Bedrock wires carry
/// no name) must reach Gemini with the name resolved from the paired
/// call: `functionResponse.name: ""` is INVALID_ARGUMENT.
#[test]
fn ingested_nameless_results_resolve_their_name_at_request_assembly() {
    use crate::completion::request::CompletionRequest;
    use crate::message::{AssistantContent, ToolCall, ToolFunction, ToolResultContent};

    let request = CompletionRequest {
        chat_history: vec![
            message::Message::user("weather?"),
            message::Message::Assistant {
                id: None,
                content: vec![AssistantContent::ToolCall(ToolCall::from_wire(
                    "toolu_abc",
                    ToolFunction {
                        name: "get_weather".to_owned(),
                        arguments: json!({"city": "Paris"}),
                    },
                ))],
            },
            message::Message::User {
                content: vec![message::UserContent::tool_result_from_wire(
                    "toolu_abc",
                    "",
                    vec![ToolResultContent::text("sunny")],
                )],
            },
        ],
        documents: vec![],
        tools: vec![],
        temperature: None,
        model: None,
        output_schema: None,
        record_telemetry_content: false,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
    };

    let body = create_request_body(request).expect("request should build");
    let response_names: Vec<_> = body
        .contents
        .iter()
        .flat_map(|content| &content.parts)
        .filter_map(|part| match &part.part {
            PartKind::FunctionResponse(response) => Some(response.name.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(response_names, ["get_weather"]);
}

/// A wire-derived result keeps its provider-issued id on replay.
#[test]
fn wire_derived_tool_result_keeps_the_provider_id_on_the_wire() {
    use crate::message::ToolResultContent;

    let message = message::Message::User {
        content: vec![message::UserContent::tool_result_from_wire(
            "gemini-issued-id",
            "lookup",
            vec![ToolResultContent::text("out")],
        )],
    };
    let content: Content = message.try_into().expect("tool result should convert");
    let PartKind::FunctionResponse(response) = &content.parts[0].part else {
        panic!("expected a function response");
    };
    assert_eq!(response.id.as_deref(), Some("gemini-issued-id"));
}

#[test]
fn test_markdown_document_conversion_to_text_part() {
    // Test that MARKDOWN documents are converted to plain text parts
    use crate::message::{DocumentMediaType, UserContent};

    let doc = UserContent::document(
        "# Heading\n\n* List item",
        Some(DocumentMediaType::MARKDOWN),
    );

    let content: Content = message::Message::User { content: vec![doc] }
        .try_into()
        .unwrap();

    if let Part {
        part: PartKind::Text(text),
        ..
    } = &content.parts[0]
    {
        assert_eq!(text, "# Heading\n\n* List item");
    } else {
        panic!(
            "Expected text part for MARKDOWN document, got: {:?}",
            content.parts[0]
        );
    }
}

#[test]
fn test_markdown_url_document_conversion_to_file_data_part() {
    // URL-backed MARKDOWN documents should be represented as file_data.
    use crate::message::{DocumentMediaType, DocumentSourceKind, UserContent};

    let doc = UserContent::Document(message::Document {
        data: DocumentSourceKind::Url(
            "https://generativelanguage.googleapis.com/v1beta/files/test-markdown".to_string(),
        ),
        media_type: Some(DocumentMediaType::MARKDOWN),
        additional_params: None,
    });

    let content: Content = message::Message::User { content: vec![doc] }
        .try_into()
        .unwrap();

    if let Part {
        part: PartKind::FileData(file_data),
        ..
    } = &content.parts[0]
    {
        assert_eq!(
            file_data.file_uri,
            "https://generativelanguage.googleapis.com/v1beta/files/test-markdown"
        );
        assert_eq!(file_data.mime_type.as_deref(), Some("text/markdown"));
    } else {
        panic!(
            "Expected file_data part for URL MARKDOWN document, got: {:?}",
            content.parts[0]
        );
    }
}

#[test]
fn test_tool_result_with_url_image_is_rejected() {
    use crate::message::{
        DocumentSourceKind, Image, ImageMediaType, ToolResult, ToolResultContent,
    };

    let tool_result = ToolResult {
        call: message::ToolCallId::minted(0),
        provider: None,
        name: "screenshot_tool".to_string(),
        content: vec![ToolResultContent::Image(Image {
            data: DocumentSourceKind::Url("https://example.com/image.png".to_string()),
            media_type: Some(ImageMediaType::PNG),
            detail: None,
            additional_params: None,
        })],
    };

    let user_content = message::UserContent::ToolResult(tool_result);
    let msg = message::Message::User {
        content: vec![user_content],
    };

    let error =
        Content::try_from(msg).expect_err("URL-backed tool result images should be rejected");
    assert!(
        error
            .to_string()
            .contains("URL-backed images are not supported"),
        "unexpected error: {error}"
    );
}

#[test]
fn test_create_request_body_with_documents() {
    // Test that documents are injected into chat history
    use crate::completion::request::{CompletionRequest, Document};
    use crate::message::Message;

    let documents = vec![
        Document {
            id: "doc1".to_string(),
            text: "Note: first.md\nContent: First note".to_string(),
            additional_props: std::collections::HashMap::new(),
        },
        Document {
            id: "doc2".to_string(),
            text: "Note: second.md\nContent: Second note".to_string(),
            additional_props: std::collections::HashMap::new(),
        },
    ];

    let documents_message = CompletionRequest {
        chat_history: vec![Message::user("placeholder")],
        documents,
        tools: vec![],
        temperature: None,
        model: None,
        output_schema: None,
        record_telemetry_content: false,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
    }
    .normalized_documents()
    .unwrap();

    let completion_request = CompletionRequest {
        chat_history: vec![
            Message::system("You are a helpful assistant"),
            documents_message,
            Message::user("What are my notes about?"),
        ],
        documents: vec![],
        tools: vec![],
        temperature: None,
        model: None,
        output_schema: None,
        record_telemetry_content: false,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
    };

    let request = create_request_body(completion_request).unwrap();

    // Should have 2 contents: 1 for documents, 1 for user message
    assert_eq!(
        request.contents.len(),
        2,
        "Expected 2 contents (documents + user message)"
    );

    // First content should be documents with role User
    assert_eq!(request.contents[0].role, Some(Role::User));
    assert_eq!(
        request.contents[0].parts.len(),
        2,
        "Expected 2 document parts"
    );

    // Check that documents are text parts
    for part in &request.contents[0].parts {
        if let Part {
            part: PartKind::Text(text),
            ..
        } = part
        {
            assert!(
                text.contains("Note:") && text.contains("Content:"),
                "Document should contain note metadata"
            );
        } else {
            panic!("Document parts should be text, not {part:?}");
        }
    }

    // Second content should be the user message
    assert_eq!(request.contents[1].role, Some(Role::User));
    if let Part {
        part: PartKind::Text(text),
        ..
    } = &request.contents[1].parts[0]
    {
        assert_eq!(text, "What are my notes about?");
    } else {
        panic!("Expected user message to be text");
    }
}

#[test]
fn test_create_request_body_without_documents() {
    // Test backward compatibility: requests without documents work as before
    use crate::completion::request::CompletionRequest;
    use crate::message::Message;

    let completion_request = CompletionRequest {
        chat_history: vec![
            Message::system("You are a helpful assistant"),
            Message::user("Hello"),
        ],
        documents: vec![], // No documents
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        model: None,
        output_schema: None,
        record_telemetry_content: false,
        additional_params: None,
    };

    let request = create_request_body(completion_request).unwrap();

    // Should have only 1 content (the user message)
    assert_eq!(request.contents.len(), 1, "Expected only user message");
    assert_eq!(request.contents[0].role, Some(Role::User));

    if let Part {
        part: PartKind::Text(text),
        ..
    } = &request.contents[0].parts[0]
    {
        assert_eq!(text, "Hello");
    } else {
        panic!("Expected user message to be text");
    }
}

#[tokio::test]
async fn completion_non_success_preserves_status_and_body() {
    use crate::client::completion::CompletionClient;
    use crate::completion::CompletionModel as _;
    use crate::providers::gemini::Client;
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"error":{"code":503,"message":"boom","status":"UNAVAILABLE"}}"#;
    let http_client =
        RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let client = Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.completion_model(super::GEMINI_3_FLASH_PREVIEW);
    let request = model.completion_request("hello").build();

    let error = model
        .completion(request)
        .await
        .expect_err("should fail with non-success status");

    assert!(matches!(error, CompletionError::HttpError(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}
