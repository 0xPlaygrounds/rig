use aws_sdk_bedrockruntime::types as aws_bedrock;
use base64::{Engine, prelude::BASE64_STANDARD};

use rig_core::{
    completion::CompletionError,
    message::{AssistantContent, Text},
};
use serde::{Deserialize, Serialize};

use crate::types::message::RigMessage;

use super::{
    converse_output::{
        ContentBlock, InternalConverseOutput, ReasoningContentBlock, StopReason, TokenUsage,
    },
    json::AwsDocument,
};
use rig_core::completion::{self};
use rig_core::telemetry::ProviderResponseExt;

#[derive(Clone, Deserialize, Serialize)]
pub struct AwsConverseOutput(pub InternalConverseOutput);

/// Normalize Bedrock token counts into rig's usage record. Shared by the
/// unary response path and the streaming terminal record.
pub(crate) fn normalize_usage(usage: &TokenUsage) -> completion::Usage {
    completion::Usage {
        input_tokens: usage.input_tokens as u64,
        output_tokens: usage.output_tokens as u64,
        total_tokens: usage.total_tokens as u64,
        cached_input_tokens: usage.cache_read_input_tokens.unwrap_or_default() as u64,
        cache_creation_input_tokens: usage.cache_write_input_tokens.unwrap_or_default() as u64,
        tool_use_prompt_tokens: 0,
        reasoning_tokens: 0,
    }
}

impl ProviderResponseExt for AwsConverseOutput {
    type Usage = completion::Usage;

    fn get_response_id(&self) -> Option<String> {
        None // Bedrock Converse API doesn't return a response ID
    }

    fn get_response_model_name(&self) -> Option<String> {
        None // Bedrock doesn't echo model name in response
    }

    fn get_text_response(&self) -> Option<String> {
        let output = self.0.output.as_ref()?;
        let message = output.as_message().ok()?;
        let response = message
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text(text) => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");

        if response.is_empty() {
            None
        } else {
            Some(response)
        }
    }

    fn get_usage(&self) -> Option<Self::Usage> {
        self.0.usage().map(normalize_usage)
    }
}

/// Stable descriptor name reported on normalized Bedrock responses.
pub const PROVIDER_NAME: &str = "aws_bedrock";

/// Map Bedrock's `stopReason` onto rig's normalized vocabulary.
///
/// `StopSequence` is a natural stop, not a truncation — the model emitted a
/// configured stop string and finished. Guardrail intervention is a content
/// filter by another name. Anything the SDK surfaced as unknown is carried
/// verbatim.
pub fn map_stop_reason(stop_reason: &StopReason) -> completion::FinishReason {
    match stop_reason {
        StopReason::EndTurn | StopReason::StopSequence => completion::FinishReason::Stop,
        StopReason::MaxTokens => completion::FinishReason::Length,
        StopReason::ToolUse => completion::FinishReason::ToolCalls,
        StopReason::ContentFiltered | StopReason::GuardrailIntervened => {
            completion::FinishReason::ContentFilter
        }
        StopReason::Unknown(value) => completion::FinishReason::Other(value.to_string()),
    }
}

impl TryFrom<AwsConverseOutput> for completion::CompletionResponse {
    type Error = CompletionError;

    fn try_from(value: AwsConverseOutput) -> Result<Self, Self::Error> {
        let message: RigMessage = value
            .to_owned()
            .0
            .output
            .ok_or(CompletionError::ProviderError(
                "Model didn't return any output".into(),
            ))?
            .as_message()
            .map_err(|_| {
                CompletionError::ProviderError(
                    "Failed to extract message from converse output".into(),
                )
            })?
            .to_owned()
            .try_into()?;

        // This arm rejects a *role* mismatch, not an empty choice — the
        // empty-converted-content case is rejected upstream in the message
        // conversion — so it carries its own diagnostic rather than the
        // shared empty-response wording.
        let choice = match message.0 {
            completion::Message::Assistant { content, .. } => Ok(content),
            _ => Err(CompletionError::ResponseError(
                "Converse output message was not an assistant message".to_owned(),
            )),
        }?;

        let usage = value.0.usage().map(normalize_usage).unwrap_or_default();

        let finish_reason = map_stop_reason(&value.0.stop_reason);

        // Bedrock's transport request id comes from the AWS SDK's response
        // metadata (`x-amzn-RequestId`), captured when the SDK output was
        // converted into `InternalConverseOutput`.
        let provider_request_id = value.0.request_id().map(str::to_string);

        Ok(
            completion::CompletionResponse::new(choice, usage, PROVIDER_NAME)
                .with_optional_provider_request_id(provider_request_id)
                .with_finish_reason(finish_reason),
        )
    }
}

pub struct RigAssistantContent(pub AssistantContent);

impl TryFrom<ContentBlock> for RigAssistantContent {
    type Error = CompletionError;

    fn try_from(value: ContentBlock) -> Result<Self, Self::Error> {
        match value {
            ContentBlock::Text(text) => {
                Ok(RigAssistantContent(AssistantContent::Text(Text::new(text))))
            }
            ContentBlock::ToolUse(call) => Ok(RigAssistantContent(
                completion::AssistantContent::tool_call(&call.tool_use_id, &call.name, call.input),
            )),
            ContentBlock::ReasoningContent(reasoning_block) => match reasoning_block {
                ReasoningContentBlock::ReasoningText(reasoning_text) => Ok(RigAssistantContent(
                    AssistantContent::Reasoning(rig_core::message::Reasoning::new_with_signature(
                        &reasoning_text.text,
                        reasoning_text.signature,
                    )),
                )),
                // Content the safety classifier encrypted. It is normal model
                // output, not a protocol violation: erroring here failed the
                // whole response over a block the streaming path and the
                // direct-Anthropic adapter both carry. The blob is bytes and
                // rig's canonical reasoning content is a string, so it travels
                // base64-encoded and decodes on the way back out.
                ReasoningContentBlock::RedactedContent(blob) => {
                    Ok(RigAssistantContent(AssistantContent::Reasoning(
                        rig_core::message::Reasoning::redacted(BASE64_STANDARD.encode(blob.inner)),
                    )))
                }
                _ => Err(CompletionError::ProviderError(
                    "AWS Bedrock returned unsupported ReasoningContentBlock variant".into(),
                )),
            },
            _ => Err(CompletionError::ProviderError(
                "AWS Bedrock returned unsupported ContentBlock".into(),
            )),
        }
    }
}

impl RigAssistantContent {
    /// Convert one assistant content item for the Converse request.
    ///
    /// `Ok(None)` means the item degrades away entirely: opaque reasoning
    /// Bedrock cannot carry — another provider's ciphertext, or a redacted
    /// blob that no longer decodes — drops with a warning instead of
    /// failing the request.
    pub(crate) fn into_content_block(
        self,
    ) -> Result<Option<aws_bedrock::ContentBlock>, CompletionError> {
        match self.0 {
            AssistantContent::Text(text) => Ok(Some(aws_bedrock::ContentBlock::Text(text.text))),
            AssistantContent::ToolCall(tool_call) => {
                // Both Converse legs must agree on `toolUseId`: the result
                // leg (user_content.rs) sends the provider-issued call id
                // when one exists, so the assistant echo does too — a bare
                // minted handle here would orphan the paired toolResult
                // whenever the two diverge.
                let tool_use_id = tool_call.wire_call_id().to_owned();
                let doc: AwsDocument = tool_call.function.arguments.into();
                Ok(Some(aws_bedrock::ContentBlock::ToolUse(
                    aws_bedrock::ToolUseBlock::builder()
                        .tool_use_id(tool_use_id)
                        .name(tool_call.function.name)
                        .input(doc.0)
                        .build()
                        .map_err(|e| CompletionError::ProviderError(e.to_string()))?,
                )))
            }
            AssistantContent::Reasoning(mut reasoning) => {
                // Opaque payloads are ciphertext, not prose — and their
                // provenance is in the variant. `Redacted` is
                // Bedrock-native: this file's own inbound legs base64-encode
                // Converse's `redactedContent` bytes, so only it may decode
                // back onto the wire. `Encrypted` NEVER originates here — it
                // is OpenAI Responses `encrypted_content`, OpenRouter
                // `reasoning.encrypted`, or Anthropic ciphertext, stored
                // verbatim (not base64) — and Bedrock can neither verify nor
                // use another provider's ciphertext. Shipping it as
                // Bedrock's own `redactedContent` hands Converse a body its
                // models never wrote, and its token shapes routinely fail
                // strict base64, which used to fail the whole request.
                // Degrade, don't fail: drop what Converse cannot carry.
                let foreign = reasoning
                    .content
                    .iter()
                    .filter(|content| {
                        matches!(content, rig_core::message::ReasoningContent::Encrypted(_))
                    })
                    .count();
                if foreign > 0 {
                    tracing::warn!(
                        dropped = foreign,
                        "dropping foreign encrypted reasoning payload(s); Bedrock cannot \
                         verify another provider's ciphertext"
                    );
                    reasoning.content.retain(|content| {
                        !matches!(content, rig_core::message::ReasoningContent::Encrypted(_))
                    });
                    if reasoning.content.is_empty() {
                        return Ok(None);
                    }
                }

                let redacted: Vec<&str> = reasoning
                    .content
                    .iter()
                    .filter_map(|content| match content {
                        rig_core::message::ReasoningContent::Redacted { data } => {
                            Some(data.as_str())
                        }
                        _ => None,
                    })
                    .collect();

                if !redacted.is_empty() {
                    if redacted.len() != reasoning.content.len() {
                        // A mixed block cannot be represented on Converse
                        // (one block is either `reasoningText` or
                        // `redactedContent`). Cross-provider replay must
                        // degrade, not fail the whole request locally: drop
                        // the un-representable opaque part(s), keep the
                        // representable text.
                        tracing::warn!(
                            dropped = redacted.len(),
                            "dropping redacted reasoning payloads Bedrock cannot carry \
                             alongside reasoning text; replaying the text only"
                        );
                        reasoning.content.retain(|content| {
                            !matches!(
                                content,
                                rig_core::message::ReasoningContent::Redacted { .. }
                            )
                        });
                    } else {
                        if redacted.len() > 1 {
                            // All-redacted with several payloads: keep the
                            // first, drop the rest — same degrade-don't-fail
                            // policy.
                            tracing::warn!(
                                dropped = redacted.len() - 1,
                                "dropping extra redacted reasoning payloads; Bedrock carries \
                                 one redactedContent blob per block"
                            );
                        }

                        // Round-trips the encoding the inbound legs apply:
                        // the wire carries bytes, rig's canonical content is
                        // a string. A blob that no longer decodes cannot be
                        // replayed — degrade like the mixed case rather than
                        // failing the request over history Bedrock will not
                        // miss.
                        let data = redacted.first().copied().unwrap_or_default();
                        return match BASE64_STANDARD.decode(data) {
                            Ok(bytes) => Ok(Some(aws_bedrock::ContentBlock::ReasoningContent(
                                aws_bedrock::ReasoningContentBlock::RedactedContent(
                                    aws_smithy_types::Blob::new(bytes),
                                ),
                            ))),
                            Err(error) => {
                                tracing::warn!(
                                    %error,
                                    "dropping redacted reasoning content that is not valid \
                                     base64"
                                );
                                Ok(None)
                            }
                        };
                    }
                }

                let signed_text_count = reasoning
                    .content
                    .iter()
                    .filter(|content| {
                        matches!(
                            content,
                            rig_core::message::ReasoningContent::Text {
                                signature: Some(_),
                                ..
                            }
                        )
                    })
                    .count();
                if signed_text_count > 1 {
                    return Err(CompletionError::ProviderError(
                        "AWS Bedrock does not support multiple signed reasoning text blocks"
                            .to_owned(),
                    ));
                }
                if signed_text_count == 1 && reasoning.content.len() > 1 {
                    return Err(CompletionError::ProviderError(
                        "AWS Bedrock requires a single signed reasoning text block without additional reasoning parts"
                            .to_owned(),
                    ));
                }

                let flattened_text = reasoning.display_text();
                let has_signature = reasoning.first_signature().is_some();
                // Adaptive thinking on Bedrock can emit a reasoning block whose
                // plaintext body is empty but with a real cryptographic
                // signature attached. The signature is what Anthropic uses to
                // verify tool_use round-trips, so we must preserve it. Only
                // reject when there's neither text nor signature to send.
                if flattened_text.is_empty() && !has_signature {
                    return Err(CompletionError::ProviderError(
                        "AWS Bedrock reasoning conversion requires at least one text or summary block"
                            .to_owned(),
                    ));
                }

                let mut reasoning_block =
                    aws_bedrock::ReasoningTextBlock::builder().text(flattened_text);

                if let Some(sig) = reasoning.first_signature().map(str::to_owned) {
                    reasoning_block = reasoning_block.signature(sig);
                }

                let reasoning_text_block = reasoning_block.build().map_err(|e| {
                    CompletionError::ProviderError(format!(
                        "Failed to build reasoning block: {}",
                        e
                    ))
                })?;

                Ok(Some(aws_bedrock::ContentBlock::ReasoningContent(
                    aws_bedrock::ReasoningContentBlock::ReasoningText(reasoning_text_block),
                )))
            }
            AssistantContent::Image(_) => Err(CompletionError::ProviderError(
                "AWS Bedrock does not support image content in assistant messages".to_owned(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
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
    fn provider_response_ext_get_text_response() {
        let out = make_output("hello world", None);
        assert_eq!(out.get_text_response(), Some("hello world".to_string()));
    }

    #[test]
    fn provider_response_ext_response_id_is_none() {
        let out = make_output("x", None);
        assert!(out.get_response_id().is_none());
        assert!(out.get_response_model_name().is_none());
    }

    #[test]
    fn provider_response_ext_get_usage_with_tokens() {
        let out = make_output("x", Some(make_usage(100, 50, 150)));
        let usage = out.get_usage().unwrap();
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
    }

    #[test]
    fn provider_response_ext_get_usage_none_when_missing() {
        let out = make_output("x", None);
        assert!(out.get_usage().is_none());
    }

    #[test]
    fn get_token_usage_delegates_to_provider_response_ext() {
        let out = make_output("x", Some(make_usage(10, 20, 30)));
        assert_eq!(
            out.get_usage().unwrap_or_default(),
            completion::Usage {
                input_tokens: 10,
                output_tokens: 20,
                total_tokens: 30,
                ..completion::Usage::new()
            }
        );
    }

    #[test]
    fn get_token_usage_zero_when_no_usage() {
        let out = make_output("x", None);
        // Zero-valued usage is rig's documented sentinel for "the provider
        // reported no usage metrics".
        assert_eq!(
            out.get_usage().unwrap_or_default(),
            completion::Usage::new()
        );
        assert!(!out.get_usage().unwrap_or_default().has_values());
    }

    #[test]
    fn aws_converse_output_to_completion_response() {
        let message = aws_bedrock::Message::builder()
            .role(aws_bedrock::ConversationRole::Assistant)
            .content(aws_bedrock::ContentBlock::Text("txt".into()))
            .build()
            .unwrap();
        let output = aws_bedrock::ConverseOutput::Message(message);
        let converse_output =
            aws_sdk_bedrockruntime::operation::converse::ConverseOutput::builder()
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
        let reasoning = rig_core::message::Reasoning::new_with_signature(
            "",
            Some("sig_empty_text".to_string()),
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
        let reasoning =
            rig_core::message::Reasoning::redacted(BASE64_STANDARD.encode(REDACTED_BLOB));
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
        let reasoning =
            rig_core::message::Reasoning::encrypted(BASE64_STANDARD.encode(REDACTED_BLOB));
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
}
