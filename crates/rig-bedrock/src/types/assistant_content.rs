use aws_sdk_bedrockruntime::types as aws_bedrock;
use aws_smithy_types::Blob;
use base64::{Engine, prelude::BASE64_STANDARD};
use serde_json::Value;

use rig_core::error::ProviderError;
use rig_core::message::{AssistantContent, Reasoning};
use rig_core::operation::Finish;

use super::{
    converse_output::{StopReason, TokenUsage},
    json,
};
use rig_core::completion;

/// Rig's input is Bedrock's `inputTokens` plus its cache reads and writes,
/// and its total input plus output, which is `totalTokens`. Shared by the
/// unary response path and the streaming terminal record.
pub(crate) fn normalize_usage(usage: &TokenUsage) -> completion::Usage {
    let cache_read = usage.cache_read_input_tokens.map(|n| n as u64);
    let cache_write = usage.cache_write_input_tokens.map(|n| n as u64);
    let input = usage.input_tokens as u64 + cache_read.unwrap_or(0) + cache_write.unwrap_or(0);
    let output = usage.output_tokens as u64;
    completion::Usage {
        input_tokens: Some(input),
        output_tokens: Some(output),
        total_tokens: Some(input + output),
        cached_input_tokens: cache_read,
        cache_creation_input_tokens: cache_write,
        tool_use_prompt_tokens: None,
        reasoning_tokens: None,
    }
}

/// Stable descriptor name reported on normalized Bedrock responses.
pub const PROVIDER_NAME: &str = "aws_bedrock";

/// Whether `model` reads the `signature` of replayed reasoning. Only Claude
/// does; other models reject the field. An inference-profile ARN that does
/// not name the model counts as another model.
pub(crate) fn reads_signatures(model: &str) -> bool {
    model.contains("anthropic.claude")
}

/// Normalizes stop reasons, preserving the others' wire spelling in `Other`.
/// Stop sequences map to `Stop`, an exceeded context window to `Length`, and
/// guardrail intervention to `ContentFilter`.
pub fn map_stop_reason(stop_reason: &StopReason) -> completion::FinishReason {
    match stop_reason {
        StopReason::EndTurn | StopReason::StopSequence => completion::FinishReason::Stop,
        StopReason::MaxTokens | StopReason::ModelContextWindowExceeded => {
            completion::FinishReason::Length
        }
        StopReason::ToolUse => completion::FinishReason::ToolCalls,
        StopReason::ContentFiltered | StopReason::GuardrailIntervened => {
            completion::FinishReason::ContentFilter
        }
        StopReason::MalformedModelOutput
        | StopReason::MalformedToolUse
        | StopReason::Unknown(_) => {
            completion::FinishReason::Other(stop_reason.as_str().to_owned())
        }
    }
}

/// The provider's end of a reply that stopped for `stop_reason`. A
/// malformed output or an unknown reason fails the turn, which is then
/// never replayed.
pub(crate) fn finish(usage: Option<&TokenUsage>, stop_reason: Option<&StopReason>) -> Finish {
    let error = stop_reason
        .filter(|reason| {
            matches!(
                reason,
                StopReason::MalformedModelOutput
                    | StopReason::MalformedToolUse
                    | StopReason::Unknown(_)
            )
        })
        .map(|reason| format!("Provider stopped with: {}", reason.as_str()));
    Finish {
        usage: usage.map(normalize_usage).unwrap_or_default(),
        reason: stop_reason.map(map_stop_reason),
        error,
        ..Finish::default()
    }
}

/// The Converse block for one assistant block, or `None` when there is
/// nothing to send. `signatures` is [`reads_signatures`] of the target model.
#[deny(clippy::wildcard_enum_match_arm)]
pub(crate) fn to_aws(
    content: AssistantContent,
    signatures: bool,
) -> Result<Option<aws_bedrock::ContentBlock>, ProviderError> {
    let native = content.native_item().cloned();
    match content {
        // Converse rejects blank text.
        AssistantContent::Text(text) if text.text.trim().is_empty() => Ok(None),
        AssistantContent::Text(text) => Ok(Some(aws_bedrock::ContentBlock::Text(text.text))),
        AssistantContent::ToolCall(call) => aws_bedrock::ToolUseBlock::builder()
            .tool_use_id(call.id.wire())
            .name(call.function.name)
            .input(json::to_document(call.function.arguments))
            .build()
            .map(|call| Some(aws_bedrock::ContentBlock::ToolUse(call)))
            .map_err(ProviderError::request),
        AssistantContent::Reasoning(reasoning) => reasoning_to_aws(reasoning, native, signatures),
        AssistantContent::Image(image) => Ok(Some(aws_bedrock::ContentBlock::Image(
            super::image::to_aws(image)?,
        ))),
        // Every opaque item this wire decodes is marked not to replay, so
        // the adapter never hands one back.
        AssistantContent::Opaque(_) => Ok(None),
    }
}

/// Reasoning with its provider item `native`: redacted bytes go back as
/// they came; signed text goes back signed to a model that reads
/// signatures, even when the text is empty.
fn reasoning_to_aws(
    reasoning: Reasoning,
    native: Option<Value>,
    signatures: bool,
) -> Result<Option<aws_bedrock::ContentBlock>, ProviderError> {
    let field = |key| native.as_ref()?.get(key)?.as_str();
    if let Some(data) = field("redacted") {
        // A payload that is not base64 cannot be the bytes Bedrock sent.
        return Ok(match BASE64_STANDARD.decode(data) {
            Ok(bytes) => Some(aws_bedrock::ContentBlock::ReasoningContent(
                aws_bedrock::ReasoningContentBlock::RedactedContent(Blob::new(bytes)),
            )),
            Err(error) => {
                tracing::warn!(%error, "dropping redacted reasoning that is not base64");
                None
            }
        });
    }
    let signature = field("signature").filter(|signature| !signature.trim().is_empty());
    let text = reasoning.text;
    let block = match (signatures, signature) {
        (true, Some(signature)) => aws_bedrock::ReasoningTextBlock::builder()
            .text(text)
            .signature(signature),
        _ if text.trim().is_empty() => return Ok(None),
        // Claude rejects unsigned reasoning: it goes back as text.
        (true, None) => return Ok(Some(aws_bedrock::ContentBlock::Text(text))),
        (false, _) => aws_bedrock::ReasoningTextBlock::builder().text(text),
    };
    block
        .build()
        .map(|block| {
            Some(aws_bedrock::ContentBlock::ReasoningContent(
                aws_bedrock::ReasoningContentBlock::ReasoningText(block),
            ))
        })
        .map_err(ProviderError::request)
}

#[cfg(test)]
mod tests;
