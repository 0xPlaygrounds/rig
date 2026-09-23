use aws_sdk_bedrockruntime::types as aws_bedrock;
use base64::{Engine, prelude::BASE64_STANDARD};

use rig_core::error::ProviderError;
use rig_core::message::{AssistantContent, Text};
use serde::{Deserialize, Serialize};

use crate::types::message::RigMessage;

use super::{
    converse_output::{
        ContentBlock, InternalConverseOutput, ReasoningContentBlock, StopReason, TokenUsage,
    },
    json::AwsDocument,
};
use rig_core::completion;
use rig_core::telemetry::ProviderResponseExt;

#[derive(Clone, Deserialize, Serialize)]
pub struct AwsConverseOutput(pub InternalConverseOutput);

/// Normalize Bedrock token counts into rig's usage record. Shared by the
/// unary response path and the streaming terminal record.
pub(crate) fn normalize_usage(usage: &TokenUsage) -> completion::Usage {
    completion::Usage {
        input_tokens: Some(usage.input_tokens as u64),
        output_tokens: Some(usage.output_tokens as u64),
        total_tokens: Some(usage.total_tokens as u64),
        cached_input_tokens: usage.cache_read_input_tokens.map(|n| n as u64),
        cache_creation_input_tokens: usage.cache_write_input_tokens.map(|n| n as u64),
        tool_use_prompt_tokens: None,
        reasoning_tokens: None,
    }
}

impl ProviderResponseExt for AwsConverseOutput {
    type Usage = completion::Usage;

    fn response_id(&self) -> Option<&str> {
        None // Bedrock Converse API doesn't return a response ID
    }

    fn response_model_name(&self) -> Option<&str> {
        None // Bedrock doesn't echo model name in response
    }

    fn text_response(&self) -> Option<String> {
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

    fn usage(&self) -> Option<Self::Usage> {
        self.0.usage().map(normalize_usage)
    }
}

/// Stable descriptor name reported on normalized Bedrock responses.
pub const PROVIDER_NAME: &str = "aws_bedrock";

/// Convert a Converse output for `model` into a completion response whose
/// reasoning records its issuer ([`reasoning_issuer`]).
pub(crate) fn completion_response(
    output: AwsConverseOutput,
    model: &str,
) -> Result<completion::CompletionResponse, ProviderError> {
    let mut response: completion::CompletionResponse = output.try_into()?;
    response.choice =
        rig_core::streaming::stamp_reasoning(response.choice, reasoning_issuer(model));
    Ok(response)
}

/// The issuer Bedrock's reasoning records for `model`. Anthropic documents
/// Claude thinking signatures as valid across the Claude API, Bedrock and
/// Vertex AI, so Claude reasoning records `anthropic`; other models' reasoning
/// records [`PROVIDER_NAME`]. An inference-profile ARN that does not name the
/// model counts as another model.
pub fn reasoning_issuer(model: &str) -> &'static str {
    if model.contains("anthropic.claude") {
        "anthropic"
    } else {
        PROVIDER_NAME
    }
}

/// Normalizes stop reasons, preserving unknown values in `Other`.
/// Stop sequences map to `Stop`; guardrail intervention maps to `ContentFilter`.
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
    type Error = ProviderError;

    fn try_from(value: AwsConverseOutput) -> Result<Self, Self::Error> {
        // The provider's own document, captured before the output is
        // consumed into normalized content.
        let raw = serde_json::to_value(&value.0)?;
        let message: RigMessage = value
            .clone()
            .0
            .output
            .ok_or(ProviderError::Provider(
                "Model didn't return any output".into(),
            ))?
            .as_message()
            .map_err(|_| {
                ProviderError::Provider("Failed to extract message from converse output".into())
            })?
            .to_owned()
            .try_into()?;

        let choice = match message.0 {
            completion::Message::Assistant { content, .. } => Ok(content),
            _ => Err(ProviderError::Response(
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
            completion::CompletionResponse::new(choice, usage, PROVIDER_NAME, raw)
                .with_optional_provider_request_id(provider_request_id)
                .with_finish_reason(finish_reason),
        )
    }
}

pub struct RigAssistantContent(pub AssistantContent);

impl TryFrom<ContentBlock> for RigAssistantContent {
    type Error = ProviderError;

    fn try_from(value: ContentBlock) -> Result<Self, Self::Error> {
        match value {
            ContentBlock::Text(text) => {
                Ok(RigAssistantContent(AssistantContent::Text(Text::new(text))))
            }
            ContentBlock::ToolUse(call) => Ok(RigAssistantContent(
                completion::AssistantContent::tool_call(&call.tool_use_id, &call.name, call.input),
            )),
            ContentBlock::ReasoningContent(reasoning_block) => match reasoning_block {
                ReasoningContentBlock::ReasoningText(reasoning_text) => {
                    Ok(RigAssistantContent(AssistantContent::Reasoning(
                        // The issuer depends on the model, which the Converse
                        // output does not name: `completion_response` stamps it.
                        rig_core::message::Reasoning::new_with_signature(
                            &reasoning_text.text,
                            reasoning_text.signature,
                        ),
                    )))
                }
                // Base64 preserves opaque redacted bytes for request replay.
                ReasoningContentBlock::RedactedContent(blob) => {
                    Ok(RigAssistantContent(AssistantContent::Reasoning(
                        rig_core::message::Reasoning::redacted(BASE64_STANDARD.encode(blob.inner)),
                    )))
                }
                _ => Err(ProviderError::Provider(
                    "AWS Bedrock returned unsupported ReasoningContentBlock variant".into(),
                )),
            },
            _ => Err(ProviderError::Provider(
                "AWS Bedrock returned unsupported ContentBlock".into(),
            )),
        }
    }
}

impl RigAssistantContent {
    /// Converts assistant content for a Converse request.
    /// Returns `Ok(None)` after dropping unsupported ciphertext or invalid base64
    /// with a warning. Rejects images and unrepresentable signed reasoning.
    pub(crate) fn into_content_block(
        self,
    ) -> Result<Option<aws_bedrock::ContentBlock>, ProviderError> {
        match self.0 {
            AssistantContent::Text(text) => Ok(Some(aws_bedrock::ContentBlock::Text(text.text))),
            AssistantContent::ToolCall(tool_call) => {
                // Calls and results must use the same provider-issued identity,
                // not a potentially different local assembly handle.
                let tool_use_id = tool_call.wire_call_id().into_owned();
                let doc: AwsDocument = tool_call.function.arguments.into();
                Ok(Some(aws_bedrock::ContentBlock::ToolUse(
                    aws_bedrock::ToolUseBlock::builder()
                        .tool_use_id(tool_use_id)
                        .name(tool_call.function.name)
                        .input(doc.0)
                        .build()
                        .map_err(|e| ProviderError::Provider(e.to_string()))?,
                )))
            }
            AssistantContent::Reasoning(mut reasoning) => {
                // Only Redacted payloads represent base64-encoded Converse bytes.
                // Drop Encrypted payloads rather than reinterpret foreign ciphertext.
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
                        // Converse cannot mix redacted bytes and text in one
                        // block; retain only the representable text.
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
                            tracing::warn!(
                                dropped = redacted.len() - 1,
                                "dropping extra redacted reasoning payloads; Bedrock carries \
                                 one redactedContent blob per block"
                            );
                        }

                        // Invalid base64 cannot reconstruct wire bytes; omit it
                        // rather than fail the whole history conversion.
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
                    return Err(ProviderError::Provider(
                        "AWS Bedrock does not support multiple signed reasoning text blocks"
                            .to_owned(),
                    ));
                }
                if signed_text_count == 1 && reasoning.content.len() > 1 {
                    return Err(ProviderError::Provider(
                        "AWS Bedrock requires a single signed reasoning text block without additional reasoning parts"
                            .to_owned(),
                    ));
                }

                let flattened_text = reasoning.display_text();
                let has_signature = reasoning.first_signature().is_some();
                // Signature-only reasoning must survive tool-call replay even
                // when its plaintext is empty.
                if flattened_text.is_empty() && !has_signature {
                    return Err(ProviderError::Provider(
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
                    ProviderError::Provider(format!("Failed to build reasoning block: {e}"))
                })?;

                Ok(Some(aws_bedrock::ContentBlock::ReasoningContent(
                    aws_bedrock::ReasoningContentBlock::ReasoningText(reasoning_text_block),
                )))
            }
            AssistantContent::Image(_) => Err(ProviderError::Provider(
                "AWS Bedrock does not support image content in assistant messages".to_owned(),
            )),
        }
    }
}

#[cfg(test)]
mod tests;
