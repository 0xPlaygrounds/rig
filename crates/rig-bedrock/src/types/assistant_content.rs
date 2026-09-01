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
            .clone()
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
                    CompletionError::ProviderError(format!("Failed to build reasoning block: {e}"))
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
mod tests;
