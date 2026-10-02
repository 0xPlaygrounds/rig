use super::*;
use serde_json::json;

/// The escape hatch's contract is that nothing the provider sent was
/// dropped, and the SDK's output type is `#[non_exhaustive]`, so the
/// conversion's rest pattern hides every field added upstream. This pins
/// the ones known today — `trace`, `performance_config` and `service_tier`
/// were all silently discarded before.
#[test]
fn converse_output_carries_every_sdk_field() {
    let sdk_output = aws_sdk_bedrockruntime::operation::converse::ConverseOutput::builder()
        .stop_reason(aws_bedrock::StopReason::GuardrailIntervened)
        .output(aws_bedrock::ConverseOutput::Message(
            aws_bedrock::Message::builder()
                .role(aws_bedrock::ConversationRole::Assistant)
                .content(aws_bedrock::ContentBlock::Text("blocked".into()))
                .build()
                .unwrap(),
        ))
        .usage(
            aws_bedrock::TokenUsage::builder()
                .input_tokens(1)
                .output_tokens(2)
                .total_tokens(3)
                .build()
                .unwrap(),
        )
        .metrics(
            aws_bedrock::ConverseMetrics::builder()
                .latency_ms(4)
                .build()
                .unwrap(),
        )
        .trace(
            aws_bedrock::ConverseTrace::builder()
                .guardrail(aws_bedrock::GuardrailTraceAssessment::builder().build())
                .build(),
        )
        .performance_config(
            aws_bedrock::PerformanceConfiguration::builder()
                .latency(aws_bedrock::PerformanceConfigLatency::Standard)
                .build(),
        )
        .service_tier(
            aws_bedrock::ServiceTier::builder()
                .r#type(aws_bedrock::ServiceTierType::Default)
                .build()
                .unwrap(),
        )
        .build()
        .unwrap();

    let mirrored = InternalConverseOutput::try_from(sdk_output).unwrap();

    assert!(mirrored.output.is_some());
    assert_eq!(mirrored.stop_reason, StopReason::GuardrailIntervened);
    assert!(mirrored.usage.is_some());
    assert!(mirrored.metrics.is_some());
    assert!(
        mirrored.trace().is_some(),
        "the guardrail trace must survive the conversion"
    );
    assert!(
        mirrored.performance_config.is_some(),
        "the performance configuration must survive the conversion"
    );
    assert!(
        mirrored.service_tier.is_some(),
        "the service tier must survive the conversion"
    );
}

/// The SDK types behind `trace`, `performance_config` and `service_tier`
/// are not `Serialize`, so they are `#[serde(skip)]`: serializing must
/// still succeed and must not invent values on the way back.
#[test]
fn skipped_provider_fields_round_trip_as_absent() {
    let output = InternalConverseOutput {
        output: None,
        stop_reason: StopReason::EndTurn,
        usage: None,
        metrics: None,
        additional_model_response_fields: None,
        request_id: Some("req-1".to_string()),
        trace: None,
        performance_config: None,
        service_tier: None,
    };

    let json = serde_json::to_string(&output).unwrap();
    let restored: InternalConverseOutput = serde_json::from_str(&json).unwrap();

    assert_eq!(restored.request_id(), Some("req-1"));
    assert!(restored.trace().is_none());
    assert!(restored.performance_config.is_none());
    assert!(restored.service_tier.is_none());
}

#[test]
fn mirror_enum_converts_known_variants() {
    assert_eq!(
        StopReason::from(aws_bedrock::StopReason::EndTurn),
        StopReason::EndTurn
    );
    // Borrowed impl.
    assert_eq!(
        StopReason::from(&aws_bedrock::StopReason::ModelContextWindowExceeded),
        StopReason::ModelContextWindowExceeded
    );
    // A renamed pairing (aws `Error` -> ours `IsError`).
    assert_eq!(
        ToolResultStatus::from(aws_bedrock::ToolResultStatus::Error),
        ToolResultStatus::IsError
    );
}

/// A variant this crate does not name keeps its wire spelling rather than
/// failing the reply.
#[test]
fn mirror_enum_unknown_variant_keeps_its_spelling() {
    let reason = StopReason::from(aws_bedrock::StopReason::from("weird_stop"));
    assert_eq!(reason.as_str(), "weird_stop");
    assert_eq!(
        ConversationRole::from(aws_bedrock::ConversationRole::from("nope")),
        ConversationRole::Unknown(UnknownVariantValue("nope".to_owned()))
    );
}

/// A union variant this crate does not mirror is `Unknown`, not an error.
#[test]
fn a_search_result_block_is_kept_as_a_marker() {
    let block = aws_bedrock::ContentBlock::SearchResult(
        aws_bedrock::SearchResultBlock::builder()
            .source("s")
            .title("t")
            .content(
                aws_bedrock::SearchResultContentBlock::builder()
                    .text("x")
                    .build()
                    .unwrap(),
            )
            .build()
            .unwrap(),
    );
    assert_eq!(
        ContentBlock::try_from(block).unwrap(),
        ContentBlock::SearchResult
    );
}

#[test]
fn additional_model_response_fields_survive_as_json() {
    let doc = json::to_document(json!({"reasoning_effort": "low", "depth": 3}));
    let output = aws_sdk_bedrockruntime::operation::converse::ConverseOutput::builder()
        .stop_reason(aws_bedrock::StopReason::EndTurn)
        .additional_model_response_fields(doc)
        .build()
        .unwrap();

    let internal = InternalConverseOutput::try_from(output).unwrap();
    assert_eq!(
        internal.additional_model_response_fields,
        Some(json!({"reasoning_effort": "low", "depth": 3}))
    );

    // The whole normalized output stays serializable and the extras
    // survive a serde round trip.
    let value = serde_json::to_value(&internal).unwrap();
    assert_eq!(
        value.get("additional_model_response_fields"),
        Some(&json!({"reasoning_effort": "low", "depth": 3}))
    );
    let back: InternalConverseOutput = serde_json::from_value(value).unwrap();
    assert_eq!(back, internal);
}

#[test]
fn tool_use_input_decodes_into_json_value() {
    let aws_block = aws_bedrock::ToolUseBlock::builder()
        .tool_use_id("call_1")
        .name("add")
        .input(json::to_document(json!({"x": 1, "y": 2})))
        .build()
        .unwrap();

    let ours = ToolUseBlock::try_from(aws_block).unwrap();
    assert_eq!(ours.tool_use_id, "call_1");
    assert_eq!(ours.name, "add");
    assert_eq!(ours.input, json!({"x": 1, "y": 2}));
}
