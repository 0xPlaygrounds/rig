use crate::types::{
    assistant_content::RigAssistantContent,
    converse_output::{ContentBlock, InternalConverseOutput},
    errors::TypeConversionError,
    json::AwsDocument,
};

use super::AwsConverseOutput;
use aws_sdk_bedrockruntime::types as aws_bedrock;
use base64::{Engine as _, prelude::BASE64_STANDARD};
use rig_core::{
    completion,
    message::{AssistantContent, ReasoningContent},
    telemetry::ProviderResponseExt,
};
use serde_json::json;

/// The inbound path reads the mirror, but what Bedrock sends is the SDK
/// block, so the tests still start there and mirror it first.
fn mirrored(block: aws_bedrock::ContentBlock) -> ContentBlock {
    block.try_into().expect("the SDK block mirrors")
}

/// Helper: build an AwsConverseOutput with text content and optional usage.
fn make_output(text: &str, usage: Option<aws_bedrock::TokenUsage>) -> AwsConverseOutput {
    make_output_with_content(vec![aws_bedrock::ContentBlock::Text(text.into())], usage)
}

fn make_output_with_content(
    content: Vec<aws_bedrock::ContentBlock>,
    usage: Option<aws_bedrock::TokenUsage>,
) -> AwsConverseOutput {
    let message = aws_bedrock::Message::builder()
        .role(aws_bedrock::ConversationRole::Assistant)
        .set_content(Some(content))
        .build()
        .unwrap();
    let mut builder = aws_sdk_bedrockruntime::operation::converse::ConverseOutput::builder()
        .output(aws_bedrock::ConverseOutput::Message(message))
        .stop_reason(aws_bedrock::StopReason::EndTurn);
    if let Some(u) = usage {
        builder = builder.usage(u);
    }
    let internal: InternalConverseOutput = builder.build().unwrap().try_into().unwrap();
    AwsConverseOutput(internal)
}

fn make_usage(input: i32, output: i32, total: i32) -> aws_bedrock::TokenUsage {
    aws_bedrock::TokenUsage::builder()
        .input_tokens(input)
        .output_tokens(output)
        .total_tokens(total)
        .build()
        .unwrap()
}

#[test]
fn provider_response_ext_text_response() {
    let out = make_output("hello world", None);
    assert_eq!(out.text_response(), Some("hello world".to_string()));
}

#[test]
fn provider_response_ext_response_id_is_none() {
    let out = make_output("x", None);
    assert!(out.response_id().is_none());
    assert!(out.response_model_name().is_none());
}

#[test]
fn provider_response_ext_usage_with_tokens() {
    let out = make_output("x", Some(make_usage(100, 50, 150)));
    let usage = out.usage().unwrap();
    assert_eq!(usage.input_tokens, 100);
    assert_eq!(usage.output_tokens, 50);
    assert_eq!(usage.total_tokens, 150);
}

#[test]
fn provider_response_ext_usage_none_when_missing() {
    let out = make_output("x", None);
    assert!(out.usage().is_none());
}

#[test]
fn token_usage_delegates_to_provider_response_ext() {
    let out = make_output("x", Some(make_usage(10, 20, 30)));
    assert_eq!(
        out.usage().unwrap_or_default(),
        completion::Usage {
            input_tokens: 10,
            output_tokens: 20,
            total_tokens: 30,
            ..completion::Usage::new()
        }
    );
}

#[test]
fn token_usage_zero_when_no_usage() {
    let out = make_output("x", None);
    // Zero-valued usage is rig's documented sentinel for "the provider
    // reported no usage metrics".
    assert_eq!(out.usage().unwrap_or_default(), completion::Usage::new());
    assert!(!out.usage().unwrap_or_default().has_values());
}

#[test]
fn aws_converse_output_to_completion_response() {
    let message = aws_bedrock::Message::builder()
        .role(aws_bedrock::ConversationRole::Assistant)
        .content(aws_bedrock::ContentBlock::Text("txt".into()))
        .build()
        .unwrap();
    let output = aws_bedrock::ConverseOutput::Message(message);
    let converse_output = aws_sdk_bedrockruntime::operation::converse::ConverseOutput::builder()
        .output(output)
        .stop_reason(aws_bedrock::StopReason::EndTurn)
        .build()
        .unwrap();
    let converse_output: Result<InternalConverseOutput, TypeConversionError> =
        converse_output.try_into();
    assert!(converse_output.is_ok());
    let converse_output = converse_output.unwrap();
    let completion: Result<completion::CompletionResponse, _> =
        AwsConverseOutput(converse_output).try_into();
    assert!(completion.is_ok());
    let completion = completion.unwrap();
    assert_eq!(
        completion.choice,
        vec![AssistantContent::Text("txt".into())]
    );
}

#[test]
fn aws_converse_output_preserves_parallel_tool_calls_in_completion_response() {
    let content = vec![
        aws_bedrock::ContentBlock::Text("preface".into()),
        aws_bedrock::ContentBlock::ToolUse(
            aws_bedrock::ToolUseBlock::builder()
                .tool_use_id("call_1")
                .name("add")
                .input(AwsDocument::from(json!({"x": 1, "y": 2})).0)
                .build()
                .unwrap(),
        ),
        aws_bedrock::ContentBlock::ToolUse(
            aws_bedrock::ToolUseBlock::builder()
                .tool_use_id("call_2")
                .name("subtract")
                .input(AwsDocument::from(json!({"x": 4, "y": 3})).0)
                .build()
                .unwrap(),
        ),
    ];

    let completion: completion::CompletionResponse = make_output_with_content(content, None)
        .try_into()
        .expect("conversion should succeed");

    let choice: Vec<_> = completion.choice.into_iter().collect();
    assert_eq!(choice.len(), 3);
    assert_eq!(choice[0], AssistantContent::Text("preface".into()));

    let AssistantContent::ToolCall(first_tool) = &choice[1] else {
        panic!("expected first tool call");
    };
    assert_eq!(first_tool.id, "call_1");
    assert_eq!(first_tool.function.name, "add");
    assert_eq!(first_tool.function.arguments, json!({"x": 1, "y": 2}));

    let AssistantContent::ToolCall(second_tool) = &choice[2] else {
        panic!("expected second tool call");
    };
    assert_eq!(second_tool.id, "call_2");
    assert_eq!(second_tool.function.name, "subtract");
    assert_eq!(second_tool.function.arguments, json!({"x": 4, "y": 3}));
}

#[test]
fn tool_use_echo_prefers_provider_call_id_like_the_result_leg() {
    // Both Converse legs must send the same toolUseId. The result leg
    // (user_content.rs) sends provider-else-handle; a diverged history
    // (e.g. JSON-restored with independent id/provider fields) must not
    // orphan the pair.
    let tool_call = rig_core::message::ToolCall::new(
        rig_core::message::ToolCallId::new("minted-handle").unwrap(),
        rig_core::message::ToolFunction {
            name: "add".into(),
            arguments: json!({"x": 1}),
        },
    )
    .with_provider(rig_core::message::ProviderCallId::new("call_abc").unwrap());

    let block = RigAssistantContent(AssistantContent::ToolCall(tool_call))
        .into_content_block()
        .unwrap()
        .expect("tool calls never degrade away");
    let aws_bedrock::ContentBlock::ToolUse(tool_use) = block else {
        panic!("expected a toolUse block");
    };
    assert_eq!(tool_use.tool_use_id(), "call_abc");
}

#[test]
fn tool_use_echo_falls_back_to_minted_handle_without_provider_id() {
    let tool_call = rig_core::message::ToolCall::new(
        rig_core::message::ToolCallId::new("minted-handle").unwrap(),
        rig_core::message::ToolFunction {
            name: "add".into(),
            arguments: json!({"x": 1}),
        },
    );

    let block = RigAssistantContent(AssistantContent::ToolCall(tool_call))
        .into_content_block()
        .unwrap()
        .expect("tool calls never degrade away");
    let aws_bedrock::ContentBlock::ToolUse(tool_use) = block else {
        panic!("expected a toolUse block");
    };
    assert_eq!(tool_use.tool_use_id(), "minted-handle");
}

#[test]
fn aws_content_block_to_assistant_content() {
    let content_block = mirrored(aws_bedrock::ContentBlock::Text("text".into()));
    let rig_assistant_content: Result<RigAssistantContent, _> = content_block.try_into();
    assert!(rig_assistant_content.is_ok());
    assert_eq!(
        rig_assistant_content.unwrap().0,
        AssistantContent::Text("text".into())
    );
}

#[test]
fn aws_reasoning_content_to_assistant_content_without_signature() {
    // Test conversion from AWS ReasoningContent to Rig AssistantContent without signature
    let reasoning_text_block = aws_bedrock::ReasoningTextBlock::builder()
        .text("This is my reasoning")
        .build()
        .unwrap();

    let content_block = mirrored(aws_bedrock::ContentBlock::ReasoningContent(
        aws_bedrock::ReasoningContentBlock::ReasoningText(reasoning_text_block),
    ));

    let rig_assistant_content: Result<RigAssistantContent, _> = content_block.try_into();
    assert!(rig_assistant_content.is_ok());

    match rig_assistant_content.unwrap().0 {
        AssistantContent::Reasoning(reasoning) => {
            assert_eq!(reasoning.first_text(), Some("This is my reasoning"));
            assert_eq!(reasoning.first_signature(), None);
            assert!(matches!(
                reasoning.content.first(),
                Some(ReasoningContent::Text { text, signature: None }) if text == "This is my reasoning"
            ));
        }
        _ => panic!("Expected AssistantContent::Reasoning"),
    }
}

#[test]
fn aws_reasoning_content_to_assistant_content_with_signature() {
    // Test conversion from AWS ReasoningContent to Rig AssistantContent with signature
    let reasoning_text_block = aws_bedrock::ReasoningTextBlock::builder()
        .text("This is my reasoning with signature")
        .signature("test_signature_123")
        .build()
        .unwrap();

    let content_block = mirrored(aws_bedrock::ContentBlock::ReasoningContent(
        aws_bedrock::ReasoningContentBlock::ReasoningText(reasoning_text_block),
    ));

    let rig_assistant_content: Result<RigAssistantContent, _> = content_block.try_into();
    assert!(rig_assistant_content.is_ok());

    match rig_assistant_content.unwrap().0 {
        AssistantContent::Reasoning(reasoning) => {
            assert_eq!(
                reasoning.first_text(),
                Some("This is my reasoning with signature")
            );
            assert_eq!(reasoning.first_signature(), Some("test_signature_123"));
            assert!(matches!(
                reasoning.content.first(),
                Some(ReasoningContent::Text { text, signature: Some(sig) })
                    if text == "This is my reasoning with signature" && sig == "test_signature_123"
            ));
        }
        _ => panic!("Expected AssistantContent::Reasoning"),
    }
}

#[test]
fn rig_reasoning_to_aws_content_block_without_signature() {
    // Test conversion from Rig Reasoning to AWS ContentBlock without signature
    let reasoning = rig_core::message::Reasoning::new("My reasoning content");
    let rig_content = RigAssistantContent(AssistantContent::Reasoning(reasoning));

    let aws_content_block = rig_content
        .into_content_block()
        .expect("conversion should succeed")
        .expect("the item converts to a block");

    match aws_content_block {
        aws_bedrock::ContentBlock::ReasoningContent(
            aws_bedrock::ReasoningContentBlock::ReasoningText(reasoning_text),
        ) => {
            assert_eq!(reasoning_text.text, "My reasoning content");
            assert_eq!(reasoning_text.signature, None);
        }
        _ => panic!("Expected ContentBlock::ReasoningContent"),
    }
}

#[test]
fn rig_reasoning_to_aws_content_block_with_signature() {
    // Test conversion from Rig Reasoning to AWS ContentBlock with signature
    let reasoning = rig_core::message::Reasoning::new_with_signature(
        "My reasoning content",
        Some("sig_abc_123".to_string()),
    );
    let rig_content = RigAssistantContent(AssistantContent::Reasoning(reasoning));

    let aws_content_block = rig_content
        .into_content_block()
        .expect("conversion should succeed")
        .expect("the item converts to a block");

    match aws_content_block {
        aws_bedrock::ContentBlock::ReasoningContent(
            aws_bedrock::ReasoningContentBlock::ReasoningText(reasoning_text),
        ) => {
            assert_eq!(reasoning_text.text, "My reasoning content");
            assert_eq!(reasoning_text.signature, Some("sig_abc_123".to_string()));
        }
        _ => panic!("Expected ContentBlock::ReasoningContent"),
    }
}

#[test]
fn rig_reasoning_with_multiple_strings_to_aws_content_block() {
    // Test that multiple reasoning strings are joined correctly
    let reasoning = rig_core::message::Reasoning::multi(vec![
        "First part".to_string(),
        " Second part".to_string(),
        " Third part".to_string(),
    ]);

    let rig_content = RigAssistantContent(AssistantContent::Reasoning(reasoning));

    let aws_content_block = rig_content
        .into_content_block()
        .expect("conversion should succeed")
        .expect("the item converts to a block");

    match aws_content_block {
        aws_bedrock::ContentBlock::ReasoningContent(
            aws_bedrock::ReasoningContentBlock::ReasoningText(reasoning_text),
        ) => {
            assert_eq!(reasoning_text.text, "First part\n Second part\n Third part");
        }
        _ => panic!("Expected ContentBlock::ReasoningContent"),
    }
}

#[test]
fn rig_reasoning_with_empty_text_and_signature_is_converted() {
    // Adaptive thinking on Bedrock can emit a reasoning block whose
    // plaintext body is empty but with a real cryptographic signature
    // attached. Verify we forward this as a `ReasoningTextBlock` with
    // empty text + signature instead of rejecting it.
    let reasoning =
        rig_core::message::Reasoning::new_with_signature("", Some("sig_empty_text".to_string()));
    let rig_content = RigAssistantContent(AssistantContent::Reasoning(reasoning));

    let aws_content_block = rig_content
        .into_content_block()
        .expect("conversion should succeed")
        .expect("the item converts to a block");

    match aws_content_block {
        aws_bedrock::ContentBlock::ReasoningContent(
            aws_bedrock::ReasoningContentBlock::ReasoningText(reasoning_text),
        ) => {
            assert_eq!(reasoning_text.text, "");
            assert_eq!(reasoning_text.signature, Some("sig_empty_text".to_string()));
        }
        _ => panic!("Expected ContentBlock::ReasoningContent"),
    }
}

#[test]
fn rig_reasoning_with_empty_text_and_no_signature_returns_error() {
    let reasoning = rig_core::message::Reasoning::new_with_signature("", None);
    let rig_content = RigAssistantContent(AssistantContent::Reasoning(reasoning));

    let aws_content_block = rig_content.into_content_block();
    assert!(matches!(
        aws_content_block,
        Err(completion::CompletionError::ProviderError(message))
            if message.contains("at least one text or summary block")
    ));
}

#[test]
fn rig_reasoning_with_multiple_signed_text_blocks_returns_error() {
    let mut reasoning =
        rig_core::message::Reasoning::new_with_signature("part one", Some("sig_1".to_string()));
    reasoning.content.push(ReasoningContent::Text {
        text: "part two".to_string(),
        signature: Some("sig_2".to_string()),
    });
    let rig_content = RigAssistantContent(AssistantContent::Reasoning(reasoning));

    let aws_content_block = rig_content.into_content_block();
    assert!(matches!(
        aws_content_block,
        Err(completion::CompletionError::ProviderError(message))
            if message.contains("multiple signed reasoning text blocks")
    ));
}

// ---- Redacted (safety-encrypted) reasoning: #2258 F2 ----

const REDACTED_BLOB: &[u8] = b"\x00\x01opaque-ciphertext\xff";

#[test]
fn aws_redacted_reasoning_content_becomes_redacted_reasoning_not_an_error() {
    // Previously this failed the WHOLE response with "unsupported
    // ReasoningContentBlock variant".
    let content_block = mirrored(aws_bedrock::ContentBlock::ReasoningContent(
        aws_bedrock::ReasoningContentBlock::RedactedContent(aws_smithy_types::Blob::new(
            REDACTED_BLOB.to_vec(),
        )),
    ));

    let rig_content: RigAssistantContent = content_block
        .try_into()
        .expect("redacted reasoning must convert, not error");

    match rig_content.0 {
        AssistantContent::Reasoning(reasoning) => assert_eq!(
            reasoning.content,
            vec![ReasoningContent::Redacted {
                data: BASE64_STANDARD.encode(REDACTED_BLOB),
            }]
        ),
        other => panic!("Expected redacted reasoning, got {other:?}"),
    }
}

#[test]
fn rig_redacted_reasoning_replays_as_redacted_content_never_as_plaintext() {
    // The bug this pins: `display_text()` folds `Redacted` into the
    // flattened text, so the blob used to be replayed as an unsigned
    // `reasoningText` body.
    let reasoning = rig_core::message::Reasoning::redacted(BASE64_STANDARD.encode(REDACTED_BLOB));
    let rig_content = RigAssistantContent(AssistantContent::Reasoning(reasoning));

    let aws_content_block = rig_content
        .into_content_block()
        .expect("redacted reasoning should convert outbound")
        .expect("a native redacted blob replays");

    match aws_content_block {
        aws_bedrock::ContentBlock::ReasoningContent(
            aws_bedrock::ReasoningContentBlock::RedactedContent(blob),
        ) => assert_eq!(blob.as_ref(), REDACTED_BLOB),
        other => panic!("Expected RedactedContent, got {other:?}"),
    }
}

#[test]
fn redacted_reasoning_round_trips_byte_for_byte() {
    let inbound = mirrored(aws_bedrock::ContentBlock::ReasoningContent(
        aws_bedrock::ReasoningContentBlock::RedactedContent(aws_smithy_types::Blob::new(
            REDACTED_BLOB.to_vec(),
        )),
    ));

    let rig_content: RigAssistantContent =
        inbound.try_into().expect("inbound conversion should work");
    let outbound = rig_content
        .into_content_block()
        .expect("outbound conversion should work")
        .expect("a native redacted blob replays");

    match outbound {
        aws_bedrock::ContentBlock::ReasoningContent(
            aws_bedrock::ReasoningContentBlock::RedactedContent(blob),
        ) => assert_eq!(blob.as_ref(), REDACTED_BLOB),
        other => panic!("Expected RedactedContent, got {other:?}"),
    }
}

/// `Encrypted` never originates on this wire — it is another
/// provider's ciphertext (OpenAI Responses `encrypted_content`,
/// OpenRouter `reasoning.encrypted`, Anthropic). It must never ship as
/// Bedrock's own `redactedContent`, even when it happens to decode:
/// the block degrades away entirely.
#[test]
fn foreign_encrypted_reasoning_is_dropped_never_shipped_as_redacted() {
    let reasoning = rig_core::message::Reasoning::encrypted(BASE64_STANDARD.encode(REDACTED_BLOB));
    let rig_content = RigAssistantContent(AssistantContent::Reasoning(reasoning));

    let converted = rig_content
        .into_content_block()
        .expect("foreign ciphertext must degrade, not fail the request");
    assert!(
        converted.is_none(),
        "another provider's ciphertext must not reach Converse: {converted:?}"
    );
}

/// A mixed block degrades: the un-representable opaque part drops (with
/// a warning), the text replays — the request must not fail locally.
/// Never as flattened plaintext: the ciphertext must not reach the
/// `reasoningText` body.
#[test]
fn opaque_reasoning_mixed_with_text_drops_the_opaque_part_not_the_request() {
    let mut reasoning = rig_core::message::Reasoning::new("visible thinking");
    reasoning.content.push(ReasoningContent::Redacted {
        data: BASE64_STANDARD.encode(REDACTED_BLOB),
    });
    let rig_content = RigAssistantContent(AssistantContent::Reasoning(reasoning));

    let aws_content_block = rig_content
        .into_content_block()
        .expect("mixed block must degrade")
        .expect("the representable text replays");
    match aws_content_block {
        aws_bedrock::ContentBlock::ReasoningContent(
            aws_bedrock::ReasoningContentBlock::ReasoningText(text),
        ) => {
            assert_eq!(text.text(), "visible thinking");
        }
        other => panic!("Expected ReasoningText, got {other:?}"),
    }
}

/// The exact shape the OpenAI Responses path builds when
/// `encrypted_content` is requested: `[Summary, Encrypted]`. Replaying
/// that history to Bedrock must degrade to the summary text, never fail
/// the whole request (#2258 B3).
#[test]
fn responses_summary_plus_encrypted_reasoning_replays_as_text() {
    let mut reasoning = rig_core::message::Reasoning::new("");
    reasoning.content.clear();
    reasoning
        .content
        .push(ReasoningContent::Summary("the summary".to_owned()));
    reasoning
        .content
        .push(ReasoningContent::Encrypted("enc-opaque-blob".to_owned()));
    let rig_content = RigAssistantContent(AssistantContent::Reasoning(reasoning));

    let aws_content_block = rig_content
        .into_content_block()
        .expect("the Responses encrypted shape must degrade, not fail the request")
        .expect("the summary text replays");
    match aws_content_block {
        aws_bedrock::ContentBlock::ReasoningContent(
            aws_bedrock::ReasoningContentBlock::ReasoningText(text),
        ) => {
            assert_eq!(text.text(), "the summary");
            assert!(
                !text.text().contains("enc-opaque-blob"),
                "ciphertext must never flatten into the plaintext body"
            );
        }
        other => panic!("Expected ReasoningText, got {other:?}"),
    }
}

/// All-opaque with several payloads keeps the first and drops the rest
/// (Converse carries one `redactedContent` blob per block).
#[test]
fn multiple_opaque_reasoning_payloads_keep_the_first() {
    let mut reasoning =
        rig_core::message::Reasoning::redacted(BASE64_STANDARD.encode(REDACTED_BLOB));
    reasoning.content.push(ReasoningContent::Redacted {
        data: BASE64_STANDARD.encode(b"second"),
    });
    let rig_content = RigAssistantContent(AssistantContent::Reasoning(reasoning));

    let aws_content_block = rig_content
        .into_content_block()
        .expect("multiple opaque payloads must degrade")
        .expect("the first native blob replays");
    match aws_content_block {
        aws_bedrock::ContentBlock::ReasoningContent(
            aws_bedrock::ReasoningContentBlock::RedactedContent(blob),
        ) => {
            assert_eq!(blob.as_ref(), REDACTED_BLOB);
        }
        other => panic!("Expected RedactedContent, got {other:?}"),
    }
}

/// An all-encrypted block whose token is unpadded/URL-safe — the shape
/// OpenAI and OpenRouter actually store (verbatim, never
/// base64-standard) — must not fail the whole request locally, and
/// must never reach the decode at all.
#[test]
fn foreign_encrypted_reasoning_never_fails_the_request() {
    let reasoning = rig_core::message::Reasoning::encrypted("gAAAA-non_base64-token_");
    let rig_content = RigAssistantContent(AssistantContent::Reasoning(reasoning));
    let converted = rig_content
        .into_content_block()
        .expect("foreign ciphertext must degrade, not fail the request");
    assert!(converted.is_none());
}

/// A native redacted blob that no longer decodes cannot be replayed:
/// it degrades away (with a warning) rather than shipping corrupt
/// bytes OR failing the request — consistent with the mixed-block
/// policy stated above.
#[test]
fn non_base64_redacted_reasoning_is_dropped_rather_than_sent_corrupt() {
    let reasoning = rig_core::message::Reasoning::redacted("not*valid*base64");
    let rig_content = RigAssistantContent(AssistantContent::Reasoning(reasoning));

    let converted = rig_content
        .into_content_block()
        .expect("an un-decodable blob must degrade, not fail the request");
    assert!(converted.is_none());
}

/// The load-bearing property behind `CompletionResponse::raw` for Bedrock:
/// the captured value is `serde_json::to_value(&AwsConverseOutput)`, and a
/// consumer must be able to read it back as the same type and get the
/// same JSON — including `metrics` and `additional_model_response_fields`,
/// which the normalized response never carries. `AwsConverseOutput` is
/// `Serialize + Deserialize`, so both halves are pinned here. The
/// SDK-typed extras (`trace`, `performance_config`, `service_tier`) are
/// `#[serde(skip)]` and so are absent from the capture by construction;
/// they read back as `None`, which is why the assertion is on the JSON
/// and on the normalized response, not on struct equality with the
/// in-process original.
#[test]
fn aws_converse_output_round_trips_through_serde_json_value() {
    let mut builder = aws_sdk_bedrockruntime::operation::converse::ConverseOutput::builder()
        .output(aws_bedrock::ConverseOutput::Message(
            aws_bedrock::Message::builder()
                .role(aws_bedrock::ConversationRole::Assistant)
                .content(aws_bedrock::ContentBlock::Text("hello".into()))
                .build()
                .expect("message should build"),
        ))
        .stop_reason(aws_bedrock::StopReason::EndTurn)
        .usage(make_usage(3, 1, 4))
        .metrics(
            aws_bedrock::ConverseMetrics::builder()
                .latency_ms(42)
                .build()
                .expect("metrics should build"),
        );
    builder = builder.additional_model_response_fields(aws_smithy_types::Document::Object(
        std::collections::HashMap::from([(
            "stop_sequence".to_string(),
            aws_smithy_types::Document::String("alpha".to_string()),
        )]),
    ));
    let internal: InternalConverseOutput = builder
        .build()
        .expect("converse output should build")
        .try_into()
        .expect("the SDK output mirrors");
    let raw = AwsConverseOutput(internal);

    let value = serde_json::to_value(&raw).expect("serialize");
    assert_eq!(value["metrics"]["latency_ms"], 42);
    assert_eq!(
        value["additional_model_response_fields"]["stop_sequence"],
        "alpha"
    );
    assert!(value.get("trace").is_none(), "SDK-typed extras are skipped");

    let back: AwsConverseOutput = serde_json::from_value(value.clone()).expect("deserialize");
    assert_eq!(
        serde_json::to_value(&back).expect("re-serialize"),
        value,
        "the capture must read back into AwsConverseOutput and re-serialize identically"
    );

    let original: completion::CompletionResponse = raw.try_into().expect("original converts");
    let restored: completion::CompletionResponse = back.try_into().expect("restored converts");
    assert_eq!(restored.identity(), original.identity());
    assert_eq!(restored.finish_reason(), original.finish_reason());
    assert_eq!(restored.usage, original.usage);
    assert_eq!(restored.choice, original.choice);
    assert_eq!(
        restored.finish_reason(),
        Some(completion::FinishReason::Stop)
    );
}
