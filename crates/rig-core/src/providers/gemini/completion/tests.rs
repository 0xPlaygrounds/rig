use crate::{
    message,
    providers::gemini::completion::gemini_api_types::{
        BlockReason, CitationMetadata, ContentCandidate, FinishReason, GenerateContentResponse,
        LogprobsResult, ModalityTokenCount, PartKind, PromptFeedback, Schema, TopCandidate,
        UsageMetadata, flatten_schema, map_finish_reason, tool_parameters_to_schema,
    },
};

use super::*;
use serde_json::json;

/// `message` as the content the REST wire sends Gemini 2.5 for it.
fn to_content(message: impl Into<message::Message>) -> Result<Content, EncodeError> {
    to_content_for("gemini-2.5-flash", message)
}

/// The contents of `request`, typed.
fn typed(request: &GenerateContentRequest) -> Vec<Content> {
    request
        .contents
        .iter()
        .map(|content| serde_json::from_value(content.clone()).expect("a Gemini content"))
        .collect()
}

/// `message` as the content the REST wire sends `model` for it.
fn to_content_for(
    model: &str,
    message: impl Into<message::Message>,
) -> Result<Content, EncodeError> {
    let contents = contents(vec![message.into()], model)?;
    let content = contents
        .into_iter()
        .next()
        .ok_or_else(|| EncodeError::request("no content"))?;
    Ok(serde_json::from_value(content)?)
}

#[test]
fn test_usage_metadata_deserializes_without_total_token_count() {
    // Gemini's proto3-JSON encoding omits fields whose value is the default (0),
    // so `totalTokenCount` is absent on short/empty/blocked generations.
    let usage: UsageMetadata =
        serde_json::from_str(r#"{"promptTokenCount": 12}"#).expect("should deserialize");
    assert_eq!(usage.total_token_count, 0);
    assert_eq!(usage.prompt_token_count, 12);
}

#[tokio::test]
async fn test_generate_content_response_deserializes_without_candidates_or_response_id() {
    // Blocked prompt responses can omit default-valued proto fields, including
    // empty repeated `candidates` and empty string `responseId`.
    let body = json!({
        "promptFeedback": {
            "blockReason": "SAFETY"
        }
    });
    let response: GenerateContentResponse =
        serde_json::from_value(body.clone()).expect("blocked prompt response should deserialize");

    assert!(response.response_id.is_empty());
    assert!(response.candidates.is_empty());

    // A set `blockReason` is the provider's verdict on the prompt: the
    // error names it instead of reporting a generic missing-candidate parse
    // failure.
    let error = fold_unary("gemini-2.5-flash", body.to_string())
        .await
        .expect_err("a blocked prompt is an error");
    assert!(
        matches!(&error, ProviderError::ProviderResponse(response) if response.body.contains("blocked the prompt") && response.body.contains("SAFETY") && response.refusal && response.code.as_deref() == Some("SAFETY")),
        "{error}"
    );
    let report = crate::error::ErrorReport::from(&error);
    assert!(
        report.refusal,
        "the verdict is a refusal on the report: {report:?}"
    );
    assert!(!report.retryable, "{report:?}");
    assert_eq!(report.code.as_deref(), Some("SAFETY"));
}

#[tokio::test]
async fn test_blocked_prompt_error_carries_the_safety_ratings() {
    let error = fold_unary(
        "gemini-2.5-flash",
        json!({
            "promptFeedback": {
                "blockReason": "PROHIBITED_CONTENT",
                "safetyRatings": [
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "probability": "HIGH"}
                ]
            },
            "usageMetadata": {"promptTokenCount": 12, "totalTokenCount": 12}
        })
        .to_string(),
    )
    .await
    .expect_err("blocked");
    let message = error.to_string();
    assert!(message.contains("PROHIBITED_CONTENT"), "{message}");
    assert!(
        message.contains("HARM_CATEGORY_DANGEROUS_CONTENT"),
        "{message}"
    );
    assert!(message.contains("HIGH"), "{message}");
}

#[tokio::test]
async fn test_unknown_block_reason_reaches_the_error_verbatim() {
    let error = fold_unary(
        "gemini-2.5-flash",
        json!({"promptFeedback": {"blockReason": "SOMETHING_NEW"}}).to_string(),
    )
    .await
    .expect_err("blocked");
    assert!(
        error.to_string().contains("block_reason=SOMETHING_NEW"),
        "{error}"
    );
    // Unknown means unknown: not a verdict on the content, so retryable.
    assert!(error.is_retryable(), "{error:?}");
}

// A whole reply that names no block and carries no candidate never names
// the provider's end: it is truncated, not a block.
#[tokio::test]
async fn test_unspecified_block_reason_is_not_a_block() {
    let error = fold_unary(
        "gemini-2.5-flash",
        json!({"promptFeedback": {"blockReason": "BLOCK_REASON_UNSPECIFIED"}}).to_string(),
    )
    .await
    .expect_err("no candidates");
    assert!(matches!(error, ProviderError::Truncated), "{error}");
}

#[tokio::test]
async fn test_no_candidates_without_prompt_feedback_is_still_a_response_error() {
    let error = fold_unary("gemini-2.5-flash", "{}")
        .await
        .expect_err("empty candidates should become a response error");
    assert!(matches!(error, ProviderError::Truncated), "{error}");
    assert_eq!(error.kind(), crate::error::ErrorKind::Response);
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
    let request = CompletionRequest::new("Hello").model("gemini-2.5-flash".to_string());

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
    let request = CompletionRequest::new("Hello");

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
    let content: Content = to_content(msg).unwrap();
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

    let content: Content = to_content(msg).unwrap();
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

#[tokio::test]
async fn test_thought_signature_is_preserved_from_response_reasoning_part() {
    let converted = unary(
        "gemini-2.5-flash",
        r#"{"responseId":"resp_1","candidates":[{"content":{"parts":[{"text":"thinking text","thought":true,"thoughtSignature":"thought_sig_123"}],"role":"model"},"finishReason":"STOP","index":0}]}"#,
    )
    .await;
    let first = converted.choice.first();
    assert!(
        matches!(
            first,
            Some(message::AssistantContent::Reasoning(reasoning))
                if reasoning.text == "thinking text"
                    && reasoning.native.as_ref().map(|native| &native.item["thoughtSignature"])
                        == Some(&json!("thought_sig_123"))
        ),
        "{first:?}"
    );
}

#[tokio::test]
async fn a_tool_protocol_finish_keeps_the_call_in_a_failed_turn() {
    for reason in [
        "MALFORMED_FUNCTION_CALL",
        "UNEXPECTED_TOOL_CALL",
        "MISSING_THOUGHT_SIGNATURE",
        "TOO_MANY_TOOL_CALLS",
        "MALFORMED_RESPONSE",
    ] {
        let body = json!({
            "responseId": "resp_tool_protocol_error",
            "candidates": [{
                "content": {
                    "parts": [{"functionCall": {"name": "default_api", "args": {"x": 1}}}],
                    "role": "model"
                },
                "finishReason": reason,
                "finishMessage": "the call was malformed",
                "index": 0
            }]
        });

        let response = fold_unary("gemini-2.5-flash", body.to_string())
            .await
            .unwrap_or_else(|error| panic!("{reason} is a turn, not a failed reply: {error}"));

        assert_eq!(response.tool_calls().count(), 1, "{reason}");
        assert_eq!(
            response.finish_reason(),
            Some(crate::completion::FinishReason::Other(reason.to_owned()))
        );
        assert!(
            response.stop().is_failure(),
            "{reason}: {:?}",
            response.stop()
        );
    }
}

#[tokio::test]
async fn test_completion_response_usage_preserves_cached_and_reasoning_tokens() {
    let converted = unary(
        "gemini-2.5-flash",
        r#"{"responseId":"resp_1","candidates":[{"content":{"parts":[{"text":"answer"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":40,"cachedContentTokenCount":20,"candidatesTokenCount":30,"totalTokenCount":92,"thoughtsTokenCount":10,"toolUsePromptTokenCount":12},"modelVersion":"gemini-2.0-flash-001"}"#,
    )
    .await;

    // Input is the prompt plus the hosted-tool prompt, output the candidates
    // plus the thoughts, and the total their sum, as `totalTokenCount` is.
    assert_eq!(converted.usage.input_tokens, Some(52));
    assert_eq!(converted.usage.cached_input_tokens, Some(20));
    assert_eq!(converted.usage.output_tokens, Some(40));
    assert_eq!(converted.usage.reasoning_tokens, Some(10));
    assert_eq!(converted.usage.tool_use_prompt_tokens, Some(12));
    assert_eq!(converted.usage.total_tokens, Some(92));
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
        assert_eq!(map_finish_reason(&wire), expected, "wire reason {wire:?}");
    }

    // The unused zero value names no clean stop, so it is a failure.
    assert_eq!(
        map_finish_reason(&FinishReason::FinishReasonUnspecified),
        Normalized::Other("FINISH_REASON_UNSPECIFIED".to_owned())
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
        crate::completion::FinishReason::Other("FINISH_REASON_FUTURE".to_string())
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

#[tokio::test]
async fn test_unary_response_with_unknown_finish_reason_stays_parseable() {
    // A finish reason Google ships tomorrow must not fail the whole
    // payload: content and usage stay intact, and the reason maps to
    // `Other` verbatim — matching the gRPC crate's handling of unknowns.
    let converted = unary(
        "gemini-2.5-flash",
        r#"{"responseId":"resp-future","candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"},"finishReason":"FINISH_REASON_FUTURE"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":2,"totalTokenCount":5}}"#,
    )
    .await;

    assert!(matches!(
        converted.choice.first(),
        Some(message::AssistantContent::Text(text)) if text.text == "hi"
    ));
    assert_eq!(converted.usage.total_tokens, Some(5));
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
        crate::completion::FinishReason::Other("FINISH_REASON_FUTURE".to_string())
    );
}

#[tokio::test]
async fn test_completion_response_carries_normalized_metadata() {
    let converted = unary(
        "gemini-2.5-flash",
        r#"{"responseId":"resp-meta","modelVersion":"gemini-2.0-flash-001","candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"},"finishReason":"MAX_TOKENS"}]}"#,
    )
    .await;

    assert_eq!(converted.provider(), PROVIDER_NAME);
    assert_eq!(converted.model(), Some("gemini-2.0-flash-001"));
    assert_eq!(converted.response_id(), Some("resp-meta"));
    assert_eq!(
        converted.finish_reason(),
        Some(crate::completion::FinishReason::Length)
    );
}

#[tokio::test]
async fn test_completion_response_upgrades_stop_to_tool_calls() {
    // Gemini reports STOP on turns that only emitted a function call; the
    // normalized response must still say `ToolCalls`.
    let converted = unary(
        "gemini-2.5-flash",
        r#"{"responseId":"resp-tool","candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"city":"Paris"}}}],"role":"model"},"finishReason":"STOP"}]}"#,
    )
    .await;

    assert_eq!(
        converted.finish_reason(),
        Some(crate::completion::FinishReason::ToolCalls)
    );
    assert_eq!(converted.model(), None);
}

#[test]
fn test_reasoning_signature_is_emitted_in_gemini_part() {
    let msg = message::Message::from(vec![
        message::AssistantContent::reasoning("structured thought").with_native(json!({
            "text": "structured thought",
            "thought": true,
            "thoughtSignature": "cmV1c2Vfc2lnXzQ1Ng==",
        })),
    ]);

    let converted: Content = to_content(msg).expect("convert message");
    let first = converted.parts.first().expect("reasoning part");
    assert_eq!(first.thought, Some(true));
    assert_eq!(
        first.thought_signature.as_deref(),
        Some("cmV1c2Vfc2lnXzQ1Ng==")
    );
    assert!(matches!(
        &first.part,
        PartKind::Text(text) if text == "structured thought"
    ));
}

#[test]
fn test_message_conversion_tool_call() {
    let tool_call = message::ToolCall::from_wire(
        "call-123",
        message::ToolFunction::new(
            crate::message::ToolName::new("test_function".to_string()).expect("tool name"),
            json!({"arg1": "value1"}),
        ),
    );

    let msg = message::Message::from(tool_call);

    // Gemini 3 takes call ids; Gemini 2.5 is sent none.
    let content: Content = to_content_for("gemini-3-flash-preview", msg.clone()).unwrap();
    let Some(Part {
        part: PartKind::FunctionCall(function_call),
        ..
    }) = to_content(msg).unwrap().parts.into_iter().next()
    else {
        panic!("Expected function call part");
    };
    assert_eq!(function_call.id, None);
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

#[tokio::test]
async fn test_response_function_call_preserves_correlation_id() {
    let converted = unary(
        "gemini-2.5-flash",
        r#"{"responseId":"response-123","candidates":[{"content":{"parts":[{"functionCall":{"name":"test_function","args":{"arg1":"value1"},"id":"call-123"}}],"role":"model"},"finishReason":"STOP"}]}"#,
    )
    .await;
    let Some(message::AssistantContent::ToolCall(tool_call)) = converted.choice.first() else {
        panic!("expected a tool call");
    };
    assert_eq!(
        tool_call.id.provider().map(message::ProviderCallId::as_str),
        Some("call-123")
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

    let doc = UserContent::document_text(
        "Note: test.md\nPath: /test.md\nContent: Hello World!",
        Some(DocumentMediaType::TXT),
    );

    let content: Content = to_content(message::Message::User { content: vec![doc] }).unwrap();

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
    let tool_result = ToolResult { is_error: false,
        call: crate::message::CallId::from_wire("call-123"),
        name: crate::message::ToolName::new("test_tool".to_string()).expect("tool name"),
        content: vec![ToolResultContent::Text(message::Text::new(r#"{"status": "success"}"#.to_string())),ToolResultContent::Image(Image {
                data: DocumentSourceKind::Base64("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==".to_string()),
                media_type: Some(ImageMediaType::PNG),
                detail: None,
                native: None,
            })],
    };

    let user_content = message::UserContent::ToolResult(tool_result);
    let msg = message::Message::User {
        content: vec![user_content],
    };

    // Convert to Gemini Content
    let content: Content =
        to_content_for("gemini-3-flash-preview", msg).expect("Should convert to Gemini Content");
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
            is_error: false,
            call: crate::message::CallId::from_wire(""),
            name: crate::message::ToolName::new("ordered_tool".to_string()).expect("tool name"),
            content: vec![
                ToolResultContent::image_base64("first-image", Some(ImageMediaType::PNG), None),
                ToolResultContent::text("between-images"),
                ToolResultContent::image_base64("second-image", Some(ImageMediaType::JPEG), None),
            ],
        })],
    };

    let content: Content = to_content(message).expect("tool result should convert");
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
            is_error: false,
            call: crate::message::CallId::from_wire(""),
            name: crate::message::ToolName::new("ordered_tool".to_string()).expect("tool name"),
            content: vec![
                ToolResultContent::json(json!({ "status": "ok" })),
                ToolResultContent::image_base64("image-data", Some(ImageMediaType::PNG), None),
            ],
        })],
    };

    let content: Content = to_content(message).expect("tool result should convert");
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
                is_error: false,
                call: crate::message::CallId::from_wire(""),
                name: crate::message::ToolName::new("image_tool".to_string()).expect("tool name"),
                content: vec![ToolResultContent::image_base64(
                    "image-data",
                    Some(media_type),
                    None,
                )],
            })],
        };

        let error =
            to_content(message).expect_err("unsupported tool result image type should be rejected");
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
            is_error: false,
            call: crate::message::CallId::from_wire(""),
            name: crate::message::ToolName::new("collision_tool".to_string()).expect("tool name"),
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

    let content: Content = to_content(message).expect("tool result should convert");
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
                is_error: false,
                call: crate::message::CallId::from_wire(""),
                name: crate::message::ToolName::new("test_tool".to_string()).expect("tool name"),
                content: vec![tool_content],
            })],
        };
        let content: Content = to_content(message).expect("tool result should convert");

        let PartKind::FunctionResponse(response) = &content.parts[0].part else {
            panic!("expected a function response");
        };
        assert_eq!(response.response.as_ref(), Some(&expected));
    }
}

/// A consumer echoing a minted `ToolCall::id` through `tool_result()` does
/// not put that handle on the wire of a model that takes no ids: the paired
/// functionCall omitted its id, and an asymmetric functionCall and
/// functionResponse id pair is rejected.
#[test]
fn echoed_minted_handle_never_reaches_the_function_response_id() {
    use crate::message::{CallId, ToolCall, ToolFunction, ToolResultContent};

    // An id-less wire: rig issued the id (Gemini REST issued none).
    let call = ToolCall::new(
        CallId::from_wire(""),
        ToolFunction::new(
            crate::message::ToolName::new("lookup".to_string()).expect("tool name"),
            json!({}),
        ),
    );

    let message = message::Message::User {
        content: vec![message::UserContent::ToolResult(message::ToolResult {
            is_error: false,
            call: call.id.clone(),
            name: call.function.name.clone(),
            content: vec![ToolResultContent::text("out")],
        })],
    };
    let content: Content = to_content(message).expect("tool result should convert");
    let PartKind::FunctionResponse(response) = &content.parts[0].part else {
        panic!("expected a function response");
    };
    assert_eq!(response.id, None);
}

/// A wire-derived result keeps its provider-issued id on replay to a model
/// that takes ids.
#[test]
fn wire_derived_tool_result_keeps_the_provider_id_on_the_wire() {
    use crate::message::ToolResultContent;

    let message = message::Message::User {
        content: vec![message::UserContent::tool_result(
            crate::message::CallId::from_wire("gemini-issued-id"),
            crate::message::ToolName::new("lookup").expect("tool name"),
            vec![ToolResultContent::text("out")],
        )],
    };
    let content: Content =
        to_content_for("gemini-3-flash-preview", message).expect("tool result should convert");
    let PartKind::FunctionResponse(response) = &content.parts[0].part else {
        panic!("expected a function response");
    };
    assert_eq!(response.id.as_deref(), Some("gemini-issued-id"));
}

#[test]
fn test_markdown_document_conversion_to_text_part() {
    // Test that MARKDOWN documents are converted to plain text parts
    use crate::message::{DocumentMediaType, UserContent};

    let doc = UserContent::document_text(
        "# Heading\n\n* List item",
        Some(DocumentMediaType::MARKDOWN),
    );

    let content: Content = to_content(message::Message::User { content: vec![doc] }).unwrap();

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

    let content: Content = to_content(message::Message::User { content: vec![doc] }).unwrap();

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
fn test_user_image_url_renders_as_file_data() {
    // A URL-sourced user image is a Files API / Cloud Storage reference on
    // this wire (`fileData`), never fetched inline: an arbitrary HTTPS image
    // is not a Gemini capability, and the adapter does not convert it.
    use crate::message::{DocumentSourceKind, Image, ImageMediaType, UserContent};

    let image = UserContent::Image(Image {
        data: DocumentSourceKind::Url("https://example.com/red_square.png".to_string()),
        media_type: Some(ImageMediaType::PNG),
        detail: None,
        native: None,
    });

    let content: Content = to_content(message::Message::User {
        content: vec![image],
    })
    .unwrap();

    match &content.parts[0] {
        Part {
            part: PartKind::FileData(file_data),
            ..
        } => {
            assert_eq!(file_data.file_uri, "https://example.com/red_square.png");
            assert_eq!(file_data.mime_type.as_deref(), Some("image/png"));
        }
        other => panic!("Expected file_data part for a URL image, got: {other:?}"),
    }
}

#[test]
fn a_url_tool_result_image_is_sent_as_file_data() {
    use crate::message::{
        DocumentSourceKind, Image, ImageMediaType, ToolResult, ToolResultContent,
    };

    let tool_result = ToolResult {
        is_error: false,
        call: crate::message::CallId::from_wire(""),
        name: crate::message::ToolName::new("screenshot_tool".to_string()).expect("tool name"),
        content: vec![
            ToolResultContent::Image(Image {
                data: DocumentSourceKind::Url("https://example.com/image.png".to_string()),
                media_type: Some(ImageMediaType::PNG),
                detail: None,
                native: None,
            }),
            ToolResultContent::text("after-image"),
        ],
    };

    let content = to_content_for(
        "gemini-3-flash-preview",
        message::Message::User {
            content: vec![message::UserContent::ToolResult(tool_result)],
        },
    )
    .expect("a URL image encodes");
    let PartKind::FunctionResponse(response) = &content.parts[0].part else {
        panic!("a function response: {content:?}");
    };
    assert_eq!(response.response, Some(json!({"result": "after-image"})));
    let parts = response.parts.as_ref().expect("the image is a part");
    let file = parts[0].file_data.as_ref().expect("the image is file data");
    assert_eq!(file.file_uri, "https://example.com/image.png");
    assert_eq!(file.mime_type.as_deref(), Some("image/png"));
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

    let documents_message = CompletionRequest::new("placeholder")
        .documents(documents)
        .normalized_documents()
        .unwrap();

    let completion_request = CompletionRequest::from(vec![
        Message::system("You are a helpful assistant"),
        documents_message,
        Message::user("What are my notes about?"),
    ]);

    let request = create_request_body(completion_request, "gemini-2.5-flash").unwrap();
    let contents = typed(&request);

    // Should have 2 contents: 1 for documents, 1 for user message
    assert_eq!(
        contents.len(),
        2,
        "Expected 2 contents (documents + user message)"
    );

    // First content should be documents with role User
    assert_eq!(contents[0].role, Some(Role::User));
    assert_eq!(contents[0].parts.len(), 2, "Expected 2 document parts");

    // Check that documents are text parts
    for part in &contents[0].parts {
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
    assert_eq!(contents[1].role, Some(Role::User));
    if let Part {
        part: PartKind::Text(text),
        ..
    } = &contents[1].parts[0]
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

    let completion_request =
        CompletionRequest::new("Hello").preamble("You are a helpful assistant");

    let request = create_request_body(completion_request, "gemini-2.5-flash").unwrap();
    let contents = typed(&request);

    // Should have only 1 content (the user message)
    assert_eq!(contents.len(), 1, "Expected only user message");
    assert_eq!(contents[0].role, Some(Role::User));

    if let Part {
        part: PartKind::Text(text),
        ..
    } = &contents[0].parts[0]
    {
        assert_eq!(text, "Hello");
    } else {
        panic!("Expected user message to be text");
    }
}

/// A non-success reply is reported with the provider's own status and body
/// preserved: the envelope shape is Gemini's business, so nothing on the
/// path may narrow it by parsing before the caller sees it.
#[tokio::test]
async fn completion_non_success_preserves_status_and_body() {
    let body = r#"{"error":{"code":503,"message":"boom","status":"UNAVAILABLE"}}"#;
    let error = crate::driver::Model::new(
        wire(super::GEMINI_3_FLASH_PREVIEW),
        RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body),
    )
    .call(wire_request("hello"))
    .await
    .expect_err("should fail with non-success status");

    assert!(matches!(error, ProviderError::ProviderResponse(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}

#[tokio::test]
async fn block_reasons_split_into_final_refusals_and_transient_blocks() {
    // `SAFETY`, `BLOCKLIST` and `PROHIBITED_CONTENT` judge the content and
    // are final. `OTHER` is Google's "blocked due to unknown reasons"; the
    // same prompt is answered on the next call, so it is retryable and it
    // is not a refusal. The classification is typed: the report's
    // `retryable` and `kind` say it, no caller reads the message. The block
    // arrives under a 200, which the driver keeps on the error for the
    // caller to see; a success status is no retry verdict, so the
    // decoder's own stands.
    for (reason, retryable) in [
        ("SAFETY", false),
        ("BLOCKLIST", false),
        ("PROHIBITED_CONTENT", false),
        ("OTHER", true),
        ("SOMETHING_NEW", true),
    ] {
        let error = fold_unary(
            "gemini-2.5-flash",
            json!({"promptFeedback": {"blockReason": reason}}).to_string(),
        )
        .await
        .expect_err(reason);
        assert_eq!(error.is_retryable(), retryable, "{reason}: {error:?}");
        let report = crate::error::ErrorReport::from(&error);
        assert_eq!(report.retryable, retryable, "{reason}: {report:?}");
        assert!(
            report.message.contains(&format!("block_reason={reason}")),
            "{reason}: {}",
            report.message
        );
        assert!(
            matches!(&error, ProviderError::ProviderResponse(response) if response.status == Some(http::StatusCode::OK) && response.refusal != retryable && response.code.as_deref() == Some(reason)),
            "{reason}: {error:?}"
        );
        assert_eq!(report.kind, crate::error::ErrorKind::ProviderResponse);
        assert_eq!(report.refusal, !retryable, "{reason}: {report:?}");
    }
}

// ── the GenerateContent wire ────────────────────────────────────────────
//
// Bodies below are pasted verbatim from committed cassettes, named at each
// constant. The point of the pairs is the property the wire model exists
// for: the unary reply and the streamed reply of the SAME turn, decoded by
// the SAME decoder, fold to the same answer.

use crate::test_utils::{MockStreamingClient, RecordingHttpClient};
use crate::wire::{Mode, Wire};
use futures::StreamExt;

/// `crates/rig-cassette/fixtures/cassettes/gemini/turn_termination_matrix/blocking_completed_turn_reports_stop_and_cap.yaml`
const CEDAR_UNARY: &str = r#"{"candidates":[{"content":{"parts":[{"text":"cedar"}],"role":"model"},"finishReason":"STOP","index":0}],"modelVersion":"gemini-2.5-flash","responseId":"id_REDACTED_1","usageMetadata":{"candidatesTokenCount":2,"promptTokenCount":22,"promptTokensDetails":[{"modality":"TEXT","tokenCount":22}],"serviceTier":"standard","totalTokenCount":24}}"#;

/// `crates/rig-cassette/fixtures/cassettes/gemini/turn_termination_matrix/streaming_completed_turn_reports_stop_and_cap.yaml`
/// — the same turn, streamed. Gemini delivered it as one event.
const CEDAR_STREAM: &str = concat!(
    r#"data: {"candidates":[{"content":{"parts":[{"text":"cedar"}],"role":"model"},"finishReason":"STOP","index":0}],"modelVersion":"gemini-2.5-flash","responseId":"id_REDACTED_1","usageMetadata":{"candidatesTokenCount":2,"promptTokenCount":22,"promptTokensDetails":[{"modality":"TEXT","tokenCount":22}],"serviceTier":"standard","totalTokenCount":24}}"#,
    "\r\n\r\n",
);

/// `crates/rig-cassette/fixtures/cassettes/gemini/thought_text_matrix/blocking_keeps_a_trailing_thought_signature.yaml`
const SIGNED_UNARY: &str = r#"{"candidates":[{"content":{"parts":[{"text":"289","thoughtSignature":"c2lnbmF0dXJl"}],"role":"model"},"finishReason":"STOP","index":0}],"modelVersion":"gemini-3-flash-preview","responseId":"id_REDACTED_1","usageMetadata":{"candidatesTokenCount":2,"promptTokenCount":14,"promptTokensDetails":[{"modality":"TEXT","tokenCount":14}],"serviceTier":"standard","thoughtsTokenCount":43,"totalTokenCount":59}}"#;

/// `crates/rig-cassette/fixtures/cassettes/gemini/thought_text_matrix/streaming_twin_agrees_on_a_trailing_thought_signature.yaml`
/// — the same turn, streamed across two events, the signature riding a
/// trailing part that carries no `thought` flag.
const SIGNED_STREAM: &str = concat!(
    r#"data: {"candidates":[{"content":{"parts":[{"text":"289"}],"role":"model"},"index":0}],"modelVersion":"gemini-3-flash-preview","responseId":"id_REDACTED_1","usageMetadata":{"candidatesTokenCount":3,"promptTokenCount":14,"promptTokensDetails":[{"modality":"TEXT","tokenCount":14}],"serviceTier":"standard","thoughtsTokenCount":43,"totalTokenCount":60}}"#,
    "\r\n\r\n",
    r#"data: {"candidates":[{"content":{"parts":[{"text":"","thoughtSignature":"c2lnbmF0dXJl"}],"role":"model"},"finishReason":"STOP","index":0}],"modelVersion":"gemini-3-flash-preview","responseId":"id_REDACTED_1","usageMetadata":{"candidatesTokenCount":3,"promptTokenCount":14,"promptTokensDetails":[{"modality":"TEXT","tokenCount":14}],"serviceTier":"standard","thoughtsTokenCount":43,"totalTokenCount":60}}"#,
    "\r\n\r\n",
);

fn wire_request(prompt: &str) -> CompletionRequest {
    CompletionRequest::new(prompt)
}

fn wire(model: &str) -> GenerateContent {
    crate::providers::gemini::GeminiConfig::new("test-key").completion(model)
}

/// What a folded response says, for comparing two transports.
fn folded(
    response: &crate::completion::CompletionResponse,
) -> (
    Vec<message::AssistantContent>,
    crate::completion::Usage,
    Option<crate::completion::FinishReason>,
    Option<String>,
) {
    (
        response.choice.to_vec(),
        response.usage,
        response.finish_reason(),
        response.model().map(str::to_owned),
    )
}

/// Fold one `generateContent` reply body through the bound wire, the way a
/// caller's `completion()` does — errors included.
async fn fold_unary(
    model: &str,
    body: impl Into<bytes::Bytes>,
) -> Result<crate::completion::CompletionResponse, ProviderError> {
    crate::driver::Model::new(wire(model), RecordingHttpClient::new(body))
        .call(wire_request("probe"))
        .await
}

async fn unary(model: &str, body: &'static str) -> crate::completion::CompletionResponse {
    fold_unary(model, body)
        .await
        .expect("the recorded unary reply decodes")
}

async fn streamed(model: &str, body: &'static str) -> crate::completion::CompletionResponse {
    let mut stream = crate::driver::Model::new(
        wire(model),
        MockStreamingClient {
            sse_bytes: bytes::Bytes::from_static(body.as_bytes()),
        },
    )
    .stream(wire_request("probe"))
    .expect("the stream opens");
    while let Some(item) = stream.next().await {
        item.expect("the recorded stream carries no in-band error");
    }
    stream
        .finish()
        .await
        .expect("the stream produced a terminal record")
}

#[tokio::test]
async fn a_unary_reply_and_a_streamed_reply_fold_to_the_same_answer() {
    let buffered = unary("gemini-2.5-flash", CEDAR_UNARY).await;
    let streamed = streamed("gemini-2.5-flash", CEDAR_STREAM).await;
    assert_eq!(folded(&buffered), folded(&streamed));
    assert_eq!(
        buffered
            .choice
            .first()
            .map(message::AssistantContent::canonical),
        Some(message::AssistantContent::text("cedar"))
    );
    assert_eq!(
        buffered.finish_reason(),
        Some(crate::completion::FinishReason::Stop)
    );
    assert_eq!(buffered.usage.output_tokens, Some(2));
    assert_eq!(buffered.usage.total_tokens, Some(24));
}

/// Gemini hangs `thoughtSignature` on an answer part. The unary reply signs
/// its one answer part; the streamed twin sends the text, then an empty
/// signed part, which continues the same text block. Both decode to one
/// text whose provider item is the signed part, and both replay it as is.
#[tokio::test]
async fn a_trailing_thought_signature_joins_the_text_it_follows() {
    let buffered = unary("gemini-3-flash-preview", SIGNED_UNARY).await;
    let streamed = streamed("gemini-3-flash-preview", SIGNED_STREAM).await;
    assert_eq!(buffered.message(), streamed.message());
    let signed = json!({ "text": "289", "thoughtSignature": "c2lnbmF0dXJl" });
    let [message::AssistantContent::Text(text)] = buffered.choice.as_slice() else {
        panic!("one answer text: {:?}", buffered.choice);
    };
    assert_eq!(text.text, "289");
    assert_eq!(
        text.native.as_ref().map(|native| &native.item),
        Some(&signed)
    );

    let replayed = to_content(buffered.choice.clone()).expect("the turn replays");
    assert_eq!(
        serde_json::to_value(&replayed.parts).unwrap(),
        json!([signed])
    );
}

/// One part carrying both non-empty text and its `thoughtSignature`, in a
/// single frame: both transports keep the signature on that text.
/// Shaped after the effect corpus
/// (`crates/rig-cassette/fixtures/effects/gemini_tool_call_turns.effects.json`).
const SIGNED_ONE_PART: &str = r#"{"candidates":[{"content":{"parts":[{"text":"done","thoughtSignature":"c2lnbmF0dXJl"}],"role":"model"},"finishReason":"STOP","index":0}],"modelVersion":"gemini-3-flash-preview","responseId":"id_REDACTED_1","usageMetadata":{"candidatesTokenCount":1,"promptTokenCount":14,"promptTokensDetails":[{"modality":"TEXT","tokenCount":14}],"thoughtsTokenCount":12,"totalTokenCount":27}}"#;

/// The same document as one SSE event: the streamed twin of [`SIGNED_ONE_PART`].
const SIGNED_ONE_PART_STREAM: &str = concat!(
    r#"data: {"candidates":[{"content":{"parts":[{"text":"done","thoughtSignature":"c2lnbmF0dXJl"}],"role":"model"},"finishReason":"STOP","index":0}],"modelVersion":"gemini-3-flash-preview","responseId":"id_REDACTED_1","usageMetadata":{"candidatesTokenCount":1,"promptTokenCount":14,"promptTokensDetails":[{"modality":"TEXT","tokenCount":14}],"thoughtsTokenCount":12,"totalTokenCount":27}}"#,
    "\r\n\r\n",
);

#[tokio::test]
async fn a_signature_on_its_own_text_part_stays_on_that_text_on_both_transports() {
    let buffered = unary("gemini-3-flash-preview", SIGNED_ONE_PART).await;
    let streamed = streamed("gemini-3-flash-preview", SIGNED_ONE_PART_STREAM).await;

    let expected = vec![
        message::AssistantContent::text("done")
            .with_native(json!({ "text": "done", "thoughtSignature": "c2lnbmF0dXJl" })),
    ];
    assert_eq!(buffered.choice, expected);
    assert_eq!(streamed.choice, expected);
}

/// The one request an `Encoded` carries.
fn sole(encoded: &crate::wire::Encoded) -> &http::Request<crate::wire::Body> {
    &encoded.request
}

#[test]
fn the_mode_chooses_the_endpoint_and_the_framing() {
    let wire = wire("gemini-2.5-flash");

    let unary = wire
        .encode(wire_request("probe"), Mode::Unary)
        .expect("the unary request encodes");
    assert_eq!(
        sole(&unary).uri().path(),
        "/v1beta/models/gemini-2.5-flash:generateContent"
    );
    // The key is a query parameter on this family, appended last.
    assert_eq!(sole(&unary).uri().query(), Some("key=test-key"));
    assert_eq!(unary.framing, crate::http_client::framing::Framing::Whole);

    let streaming = wire
        .encode(wire_request("probe"), Mode::Streaming)
        .expect("the streaming request encodes");
    assert_eq!(
        sole(&streaming).uri().path(),
        "/v1beta/models/gemini-2.5-flash:streamGenerateContent"
    );
    assert_eq!(sole(&streaming).uri().query(), Some("alt=sse&key=test-key"));
    assert_eq!(streaming.framing, crate::http_client::framing::Framing::Sse);
    // Gemini reports no transport request-id header.
    assert_eq!(streaming.request_id_header, None);
}

/// The span names this wire has always recorded, per mode. Telemetry
/// equivalence is part of the port's contract.
#[test]
fn the_wire_keeps_its_span_names() {
    let wire = wire("gemini-2.5-flash");
    assert_eq!(
        wire.describe()
            .telemetry
            .map(|telemetry| telemetry(crate::wire::Mode::Unary)),
        Some(GenAiOperation::GenerateContent)
    );
    assert_eq!(
        wire.describe()
            .telemetry
            .map(|telemetry| telemetry(crate::wire::Mode::Streaming)),
        Some(GenAiOperation::ChatStreaming)
    );
}

/// An `inlineData` image part is an image block holding the part, and it
/// replays as the part Gemini sent.
///
/// Shape taken from
/// `crates/rig-cassette/fixtures/cassettes/gemini/image_generation/nano_banana_image_generation_smoke.yaml`,
/// whose recorded `data` is a 1 MB PNG; the payload here is shortened
/// because only its survival is under test.
#[tokio::test]
async fn an_inline_data_part_decodes_to_an_image_holding_the_part() {
    const IMAGE_REPLY: &str = r#"{"candidates":[{"content":{"parts":[{"inlineData":{"data":"iVBORw0KGgoAAAANSUhEUg==","mimeType":"image/png"}}],"role":"model"},"finishReason":"STOP","index":0}],"modelVersion":"gemini-2.5-flash-image","responseId":"id_REDACTED_1","usageMetadata":{"candidatesTokenCount":1290,"promptTokenCount":15,"totalTokenCount":1305}}"#;

    let response = unary("gemini-2.5-flash-image", IMAGE_REPLY).await;
    let part =
        json!({ "inlineData": { "data": "iVBORw0KGgoAAAANSUhEUg==", "mimeType": "image/png" } });
    let [message::AssistantContent::Image(image)] = response.choice.as_slice() else {
        panic!("one image: {:?}", response.choice);
    };
    assert_eq!(
        image.data,
        message::DocumentSourceKind::Base64("iVBORw0KGgoAAAANSUhEUg==".to_owned())
    );
    assert_eq!(image.media_type, Some(message::ImageMediaType::PNG));
    assert_eq!(
        image.native.as_ref().map(|native| &native.item),
        Some(&part)
    );
    let replayed = to_content(response.choice.clone()).expect("the turn replays");
    assert_eq!(
        serde_json::to_value(&replayed.parts).unwrap(),
        json!([part])
    );
}

/// Consecutive answer parts in one document continue one text block, as a
/// stream's do: the block holds the merged part, with the last signature.
#[tokio::test]
async fn answer_parts_in_one_document_continue_one_text() {
    let body = r#"{"candidates":[{"content":{"parts":[{"text":"first"},{"text":"second"},{"text":"","thoughtSignature":"c2lnLTk="}],"role":"model"},"finishReason":"STOP","index":0}],"modelVersion":"gemini-3-flash-preview","responseId":"r","usageMetadata":{"candidatesTokenCount":1,"promptTokenCount":1,"totalTokenCount":2}}"#;
    let response = unary("gemini-3-flash-preview", body).await;
    assert_eq!(
        response.choice,
        vec![
            message::AssistantContent::text("firstsecond")
                .with_native(json!({ "text": "firstsecond", "thoughtSignature": "c2lnLTk=" }))
        ]
    );
}

/// A Gemini answer-text signature reaches only the model that issued it:
/// adapted for another Gemini model, the text replays without it.
#[test]
fn a_text_signature_reaches_no_other_model() {
    let signed = message::AssistantMessage {
        origin: Some(message::Origin::new(
            "gemini.generate_content",
            PROVIDER_NAME,
            "gemini-3-flash-preview",
        )),
        ..message::AssistantMessage::new(vec![
            message::AssistantContent::text("the answer")
                .with_native(json!({ "text": "the answer", "thoughtSignature": "c2lnbmVk" })),
        ])
    };
    let history = vec![message::Message::user("q"), signed.into()];
    for (model, part) in [
        (
            "gemini-3-flash-preview",
            json!({ "text": "the answer", "thoughtSignature": "c2lnbmVk" }),
        ),
        ("gemini-2.5-flash", json!({ "text": "the answer" })),
    ] {
        let adapted = crate::completion::adapt(&history, &wire(model));
        let contents = contents(adapted, model).expect("the history encodes");
        assert_eq!(contents[1]["parts"], json!([part]), "{model}");
    }
}

/// A signed answer text survives serde, which is how a history persists, and
/// still replays its part as Gemini sent it.
#[test]
fn a_signed_answer_text_round_trips_through_serde() {
    let message = message::Message::from(vec![
        message::AssistantContent::text("the answer")
            .with_native(json!({ "text": "the answer", "thoughtSignature": "c2lnbmVk" })),
    ]);
    let json = serde_json::to_string(&message).expect("the message serializes");
    let loaded: message::Message = serde_json::from_str(&json).expect("the message loads");
    assert_eq!(loaded, message);
    let content: Content = to_content(loaded).expect("the turn replays");
    let [part] = content.parts.as_slice() else {
        panic!("one answer part: {:?}", content.parts);
    };
    assert_eq!(part.thought_signature.as_deref(), Some("c2lnbmVk"));
    assert_ne!(part.thought, Some(true));
}

/// An empty thought part that carries a signature is kept and replays its
/// signature on a thought part.
#[test]
fn an_empty_signed_thought_replays_its_signature() {
    let part = json!({ "text": "", "thought": true, "thoughtSignature": "c2lnbmVk" });
    let message = message::Message::from(vec![
        message::AssistantContent::text("289"),
        message::AssistantContent::reasoning("").with_native(part.clone()),
    ]);
    let content = to_content(message).expect("the turn replays");
    assert_eq!(
        serde_json::to_value(&content.parts).unwrap(),
        json!([{ "text": "289" }, part])
    );
}

/// A part rebuilt from canonical fields follows pi's rebuild: text alone, a
/// thought flag on reasoning, a call id only for a model that takes ids,
/// and nothing for blank text or redacted reasoning. Gemini 3 also gets
/// Google's placeholder signature on the call, which it requires.
#[test]
fn canonical_blocks_rebuild_as_pi_rebuilds_them() {
    let call = message::AssistantContent::tool_call(
        "call-1",
        message::ToolName::new("lookup").expect("tool name"),
        json!({ "q": 1 }),
    );
    let message = message::Message::from(vec![
        message::AssistantContent::text("  "),
        message::AssistantContent::reasoning("why"),
        message::AssistantContent::Reasoning(message::Reasoning {
            redacted: true,
            ..message::Reasoning::default()
        }),
        message::AssistantContent::text("answer"),
        call,
    ]);
    for (model, call) in [
        (
            "gemini-2.5-flash",
            json!({ "functionCall": { "name": "lookup", "args": { "q": 1 } } }),
        ),
        (
            "gemini-3-flash-preview",
            json!({
                "functionCall": { "name": "lookup", "args": { "q": 1 }, "id": "call-1" },
                "thoughtSignature": "skip_thought_signature_validator",
            }),
        ),
    ] {
        let contents = contents(vec![message.clone()], model).expect("the turn encodes");
        assert_eq!(
            contents[0]["parts"],
            json!([
                { "thought": true, "text": "why" },
                { "text": "answer" },
                call,
            ]),
            "{model}"
        );
    }
}

/// A `thoughtSignature` that is not base64 is left out of the replayed part:
/// Gemini rejects the whole request over one ("Invalid value at
/// 'contents[1].parts[0].thought_signature' (TYPE_BYTES), Base64 decoding
/// failed"). Every other field is sent as received.
#[test]
fn a_signature_that_is_not_base64_is_left_out() {
    let message = message::Message::from(vec![
        message::AssistantContent::text("a")
            .with_native(json!({ "text": "a", "thoughtSignature": "not base64!", "extra": 1 })),
        message::AssistantContent::text("b")
            .with_native(json!({ "text": "b", "thoughtSignature": "c2ln" })),
    ]);
    let contents = contents(vec![message], "gemini-2.5-flash").expect("the turn encodes");
    assert_eq!(
        contents[0]["parts"],
        json!([
            { "text": "a", "extra": 1 },
            { "text": "b", "thoughtSignature": "c2ln" },
        ])
    );
}

/// Call ids travel only to models that take them (pi's
/// `requiresToolCallId`), and another model's id is normalized for them.
#[test]
fn call_ids_follow_the_models_that_take_them() {
    for (model, takes) in [
        ("gemini-2.5-flash", false),
        ("gemini-2.0-flash", false),
        ("gemini-3-flash-preview", true),
        ("gemini-3.8-flash", true),
        ("gemini-live-3-flash", true),
        ("claude-sonnet-4-5", true),
        ("gpt-oss-120b", true),
        ("gemma-3-27b-it", false),
    ] {
        assert_eq!(requires_tool_call_id(model), takes, "{model}");
    }
    let foreign = format!("call.{}|x", "y".repeat(80));
    let normalized = normalize_tool_call_id("gemini-3-flash-preview", &foreign);
    assert_eq!(normalized.len(), 64);
    assert!(normalized.starts_with("call_yyy"));
    assert_eq!(
        normalize_tool_call_id("gemini-2.5-flash", &foreign),
        foreign
    );
}

/// `ThoughtReplay::CurrentTurn` leaves the signatures out of the turns
/// before the newest user text, and only the signatures.
#[test]
fn current_turn_replay_drops_only_finished_signatures() {
    let signed = |text: &str| {
        message::AssistantContent::text(text)
            .with_native(json!({ "text": text, "thoughtSignature": "c2ln" }))
    };
    let mut body = create_request_body(
        CompletionRequest::from(vec![
            message::Message::user("one"),
            message::Message::from(vec![signed("first")]),
            message::Message::user("two"),
            message::Message::from(vec![signed("second")]),
        ]),
        "gemini-2.5-flash",
    )
    .expect("the request encodes");
    drop_finished_signatures(&mut body.contents);
    assert_eq!(body.contents[1]["parts"], json!([{ "text": "first" }]));
    assert_eq!(
        body.contents[3]["parts"],
        json!([{ "text": "second", "thoughtSignature": "c2ln" }])
    );
}

/// Every part kind of the wire, an invented kind, and invented fields on
/// known kinds survive decode and same-model replay, whole and streamed.
#[test]
#[deny(clippy::wildcard_enum_match_arm)]
fn every_part_kind_survives_decode_and_replay() {
    use crate::test_utils::history::{assert_every_variant, assert_restated_agrees, decode};
    use crate::wire::WireFrame;

    let parts = vec![
        json!({ "thought": true, "text": "why", "thoughtSignature": "c2ln" }),
        json!({ "text": "plain", "futureField": 1 }),
        json!({
            "functionCall": { "name": "lookup", "args": { "q": 1 }, "id": "c1", "futureField": true },
            "thoughtSignature": "c2ln",
        }),
        json!({ "inlineData": { "mimeType": "image/png", "data": "iVBORw0KGgo=" } }),
        json!({ "functionResponse": { "name": "lookup", "response": { "ok": true } } }),
        json!({ "fileData": { "mimeType": "application/pdf", "fileUri": "gs://bucket/a.pdf" } }),
        json!({ "executableCode": { "language": "PYTHON", "code": "print(1)" } }),
        json!({ "codeExecutionResult": { "outcome": "OUTCOME_OK", "output": "1" } }),
        json!({ "futureKind": { "x": 1 } }),
    ];
    let known: Vec<Part> = parts
        .iter()
        .take(8)
        .map(|part| serde_json::from_value(part.clone()).expect("a known part"))
        .collect();
    assert_every_variant(
        &known,
        |part| match part.part {
            PartKind::Text(_) => 0,
            PartKind::InlineData(_) => 1,
            PartKind::FunctionCall(_) => 2,
            PartKind::FunctionResponse(_) => 3,
            PartKind::FileData(_) => 4,
            PartKind::ExecutableCode(_) => 5,
            PartKind::CodeExecutionResult(_) => 6,
        },
        7,
    );

    let end = json!({ "finishReason": "STOP", "index": 0 });
    let document = |parts: &[serde_json::Value], end: Option<&serde_json::Value>| {
        let mut candidate = json!({ "content": { "parts": parts, "role": "model" } });
        if let (Some(candidate), Some(serde_json::Value::Object(end))) =
            (candidate.as_object_mut(), end)
        {
            candidate.extend(end.clone());
        }
        WireFrame::Text(json!({ "candidates": [candidate], "responseId": "r" }).to_string())
    };
    let whole = vec![document(&parts, Some(&end))];
    let mut streamed: Vec<WireFrame> = parts
        .iter()
        .map(|part| document(std::slice::from_ref(part), None))
        .collect();
    streamed.push(document(&[], Some(&end)));

    let wire = wire("gemini-3-flash-preview");
    assert_restated_agrees(&wire, whole.clone(), streamed.clone());
    for (mode, frames) in [(Mode::Unary, whole), (Mode::Streaming, streamed)] {
        let response = decode(&wire, mode, frames).expect("the reply decodes");
        assert_eq!(response.choice.len(), parts.len(), "{mode:?}");
        let history = crate::completion::adapt(&[response.message().expect("a turn")], &wire);
        let replayed = contents(history, "gemini-3-flash-preview").expect("the turn replays");
        assert_eq!(replayed[0]["parts"], json!(parts), "{mode:?}");
    }
}

/// Grounding, URL context, safety ratings and citations stay with the turn
/// as its message-level native: the candidate without its content.
#[test]
fn candidate_metadata_is_the_turns_native() {
    use crate::wire::WireFrame;

    let metadata = json!({
        "finishReason": "STOP",
        "index": 0,
        "safetyRatings": [{ "category": "HARM_CATEGORY_HARASSMENT", "probability": "NEGLIGIBLE" }],
        "citationMetadata": { "citationSources": [{ "uri": "https://example.com", "startIndex": 0, "endIndex": 4 }] },
        "groundingMetadata": { "webSearchQueries": ["rig"], "groundingChunks": [{ "web": { "uri": "https://example.com" } }] },
        "urlContextMetadata": { "urlMetadata": [{ "retrievedUrl": "https://example.com" }] },
    });
    let mut candidate = metadata.clone();
    candidate["content"] = json!({ "parts": [{ "text": "rig" }], "role": "model" });
    let frame = WireFrame::Text(json!({ "candidates": [candidate] }).to_string());
    let response =
        crate::test_utils::history::decode(&wire("gemini-2.5-flash"), Mode::Unary, [frame])
            .expect("the reply decodes");
    assert_eq!(response.native.map(|native| native.item), Some(metadata));
}

/// A call rig issued the id for is spelled `tool-<n>` for a model that
/// takes ids, on the call and its response alike, so a history always
/// encodes to the same bytes.
#[test]
fn a_rig_issued_call_id_is_spelled_as_a_request_local_alias() {
    let call = message::ToolCall::from_wire(
        "",
        message::ToolFunction::new(
            message::ToolName::new("lookup").expect("tool name"),
            json!({}),
        ),
    );
    let history = vec![
        message::Message::from(call.clone()),
        message::Message::tool_results(vec![
            call.result(vec![message::ToolResultContent::text("out")]),
        ]),
    ];
    let contents = contents(history, "gemini-3-flash-preview").expect("the history encodes");
    assert_eq!(
        contents[0]["parts"][0]["functionCall"]["id"],
        json!("tool-0")
    );
    assert_eq!(
        contents[1]["parts"][0]["functionResponse"]["id"],
        json!("tool-0")
    );
}

/// The model a request addresses decides whether a foreign call id is
/// normalized, not the wire's own model.
#[test]
fn a_request_model_override_decides_call_id_normalization() {
    use crate::completion::ReplayTarget;

    let wire = GenerateContent::new(
        crate::providers::gemini::GeminiConfig::new("k"),
        "gemini-2.5-flash",
    );
    assert_eq!(
        wire.normalize_tool_call_id("a.b", "gemini-2.5-flash", None),
        "a.b"
    );
    assert_eq!(
        wire.normalize_tool_call_id("a.b", "gemini-3-pro-preview", None),
        "a_b"
    );
}

/// The body `request` sends to `model` once prepared, as the driver sends it.
fn prepared_body(model: &str, request: CompletionRequest) -> Value {
    use crate::wire::Operation;
    let wire = wire(model);
    let request = crate::operation::Completion::prepare(request, &wire.describe())
        .expect("the request prepares");
    let encoded = wire
        .encode(request, Mode::Unary)
        .expect("the request encodes");
    let crate::wire::Body::Bytes(bytes) = sole(&encoded).body() else {
        panic!("a JSON body");
    };
    serde_json::from_slice(bytes).expect("JSON")
}

#[tokio::test]
async fn an_unknown_traffic_type_or_modality_never_fails_the_reply() {
    let response = fold_unary(
        "gemini-2.5-flash",
        r#"{"candidates":[{"content":{"parts":[{"text":"kept"}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":2,"totalTokenCount":5,"trafficType":"ON_DEMAND_PRIORITY","promptTokensDetails":[{"modality":"HOLOGRAM","tokenCount":3}],"serviceTier":"flex"}}"#,
    )
    .await
    .expect("a usage label never fails the reply");
    assert_eq!(response.text(), "kept");
    assert_eq!(response.usage.total_tokens, Some(5));
}

#[tokio::test]
async fn a_function_call_without_args_decodes_with_empty_arguments() {
    let response = fold_unary(
        "gemini-2.5-flash",
        r#"{"candidates":[{"content":{"parts":[{"functionCall":{"name":"now"}}],"role":"model"},"finishReason":"STOP"}]}"#,
    )
    .await
    .expect("a call without args decodes");
    let calls: Vec<_> = response.tool_calls().collect();
    let [call] = calls.as_slice() else {
        panic!("one call: {:?}", response.choice);
    };
    assert_eq!(call.function.arguments_value(), json!({}));
    assert!(call.function.invalid_arguments.is_none());
}

#[tokio::test]
async fn every_documented_finish_reason_ends_the_turn_as_documented() {
    for (reason, failed) in [("STOP", false), ("MAX_TOKENS", false)]
        .into_iter()
        .chain(
            gemini_api_types::FAILURE_FINISHES
                .iter()
                .map(|reason| (*reason, true)),
        )
        .chain([("A_REASON_FROM_TOMORROW", true)])
    {
        let body = json!({
            "candidates": [{
                "content": {"parts": [{"text": "partial"}], "role": "model"},
                "finishReason": reason
            }]
        });
        let response = fold_unary("gemini-2.5-flash", body.to_string())
            .await
            .unwrap_or_else(|error| panic!("{reason} is a turn: {error}"));
        assert_eq!(
            response.stop().is_failure(),
            failed,
            "{reason} ends as {:?}",
            response.stop()
        );
    }
}

#[test]
fn a_failed_tool_result_is_sent_under_error() {
    let mut result = message::ToolCall::new(
        message::CallId::from_wire("call_1"),
        message::ToolFunction::new(
            message::ToolName::new("lookup").expect("a tool name"),
            json!({}),
        ),
    )
    .error_result(vec![message::ToolResultContent::text("no such file")]);
    let content = to_content(message::Message::User {
        content: vec![message::UserContent::ToolResult(result.clone())],
    })
    .expect("a failed result encodes");
    let PartKind::FunctionResponse(response) = &content.parts[0].part else {
        panic!("a function response: {content:?}");
    };
    assert_eq!(response.response, Some(json!({"error": "no such file"})));

    result.is_error = false;
    let content = to_content(message::Message::User {
        content: vec![message::UserContent::ToolResult(result)],
    })
    .expect("a result encodes");
    let PartKind::FunctionResponse(response) = &content.parts[0].part else {
        panic!("a function response: {content:?}");
    };
    assert_eq!(response.response, Some(json!({"result": "no such file"})));
}

/// A tool-result image as Gemini 2 and Gemini 3 get it: Gemini 3 reads it
/// inside the function response, Gemini 2 in a user message after it.
#[test]
fn tool_result_images_reach_gemini_2_in_a_following_user_message() {
    let call = message::ToolCall::new(
        message::CallId::from_wire("call_shot"),
        message::ToolFunction::new(
            message::ToolName::new("shot").expect("a tool name"),
            json!({}),
        ),
    );
    let image = message::Image {
        data: message::DocumentSourceKind::base64("aW1hZ2U="),
        media_type: Some(message::ImageMediaType::PNG),
        ..message::Image::default()
    };
    let history = vec![
        message::Message::user("look"),
        message::Message::Assistant(message::AssistantMessage::new(vec![
            message::AssistantContent::ToolCall(call.clone()),
        ])),
        message::Message::User {
            content: vec![message::UserContent::ToolResult(call.result(vec![
                message::ToolResultContent::text("see"),
                message::ToolResultContent::Image(image),
            ]))],
        },
    ];
    let request = |history: Vec<message::Message>| {
        let mut request = CompletionRequest::new("next");
        request.chat_history = history;
        request
    };

    let body = prepared_body("gemini-2.5-flash", request(history.clone()));
    let contents = body["contents"].as_array().expect("contents");
    let response = contents
        .iter()
        .flat_map(|content| content["parts"].as_array().into_iter().flatten())
        .find_map(|part| part.get("functionResponse"))
        .expect("the result is sent");
    assert!(response.get("parts").is_none(), "{response}");
    assert!(
        contents.iter().any(|content| content["role"] == "user"
            && content["parts"]
                .as_array()
                .is_some_and(|parts| parts.iter().any(|part| part.get("inlineData").is_some()))),
        "the image follows in a user message: {body}"
    );

    let body = prepared_body("gemini-3-flash-preview", request(history));
    let response = body["contents"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|content| content["parts"].as_array().into_iter().flatten())
        .find_map(|part| part.get("functionResponse"))
        .expect("the result is sent");
    assert!(response.get("parts").is_some(), "{response}");
}

/// #1179: documents open the first user message, so the contents keep
/// alternating roles.
#[test]
fn documents_open_the_first_user_message_on_gemini() {
    let request =
        CompletionRequest::new("the prompt").documents(vec![crate::completion::Document {
            id: "doc1".to_owned(),
            text: "first note".to_owned(),
            additional_props: Default::default(),
        }]);
    let body = prepared_body("gemini-2.5-flash", request);
    let contents = body["contents"].as_array().expect("contents");
    assert_eq!(contents.len(), 1, "one user content: {body}");
    let text = contents[0].to_string();
    assert!(
        text.contains("first note") && text.contains("the prompt"),
        "{body}"
    );
}

/// A user message that holds tool results and then the user's next words
/// (the adapter merges adjacent user messages) goes out as two contents,
/// as pi sends them: Gemini answers text sharing a content with function
/// responses poorly, often with an empty reply.
#[test]
fn function_responses_and_user_text_go_in_separate_contents() {
    let call = message::ToolCall::new(
        message::CallId::from_wire("call_1"),
        message::ToolFunction::new(
            message::ToolName::new("lookup").expect("a tool name"),
            json!({}),
        ),
    );
    let contents = contents(
        vec![message::Message::User {
            content: vec![
                message::UserContent::ToolResult(
                    call.result(vec![message::ToolResultContent::text("found")]),
                ),
                message::UserContent::text("now answer"),
            ],
        }],
        "gemini-2.5-flash",
    )
    .expect("encodes");
    assert_eq!(
        contents,
        vec![
            json!({"role": "user", "parts": [{"thought": false, "functionResponse": {"name": "lookup", "response": {"result": "found"}}}]}),
            json!({"role": "user", "parts": [{"thought": false, "text": "now answer"}]}),
        ]
    );
}
