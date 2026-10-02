use aws_sdk_bedrockruntime::types as aws_bedrock;

use rig_core::completion;
use rig_core::error::ProviderError;
use rig_core::message::AssistantContent;
use rig_core::operation::Finish;

use super::{block, json};
use crate::completion::Family;

/// Rig's input is Bedrock's `inputTokens` plus its cache reads and writes,
/// and its total input plus output, which is `totalTokens`. Shared by the
/// unary response path and the streaming terminal record.
pub(crate) fn normalize_usage(usage: &aws_bedrock::TokenUsage) -> completion::Usage {
    let count = |tokens: i32| u64::try_from(tokens).unwrap_or(0);
    let cache_read = usage.cache_read_input_tokens.map(count);
    let cache_write = usage.cache_write_input_tokens.map(count);
    let input = count(usage.input_tokens) + cache_read.unwrap_or(0) + cache_write.unwrap_or(0);
    let output = count(usage.output_tokens);
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

/// Normalizes every stop reason Converse documents. An end of turn or a
/// stop sequence is `Stop`, the token limit and an exceeded context window
/// are `Length`, guardrail and content-filter interventions are
/// `ContentFilter`, and a malformed output or a reason this SDK version
/// does not name keeps its wire spelling in `Other`. Only the first three
/// are successful stops.
pub fn map_stop_reason(stop_reason: &aws_bedrock::StopReason) -> completion::FinishReason {
    use aws_bedrock::StopReason as Reason;
    use completion::FinishReason;
    match stop_reason {
        Reason::EndTurn | Reason::StopSequence => FinishReason::Stop,
        Reason::MaxTokens | Reason::ModelContextWindowExceeded => FinishReason::Length,
        Reason::ToolUse => FinishReason::ToolCalls,
        Reason::ContentFiltered | Reason::GuardrailIntervened => FinishReason::ContentFilter,
        Reason::MalformedModelOutput | Reason::MalformedToolUse => {
            FinishReason::Other(stop_reason.as_str().to_owned())
        }
        unknown => FinishReason::Other(unknown.as_str().to_owned()),
    }
}

/// The provider's end of a reply that stopped for `stop_reason`. A
/// malformed output or an unknown reason fails the turn, which is then
/// never replayed.
pub(crate) fn finish(
    usage: Option<&aws_bedrock::TokenUsage>,
    stop_reason: Option<&aws_bedrock::StopReason>,
) -> Finish {
    let reason = stop_reason.map(map_stop_reason);
    let error = match (&reason, stop_reason) {
        (Some(completion::FinishReason::Other(_)), Some(stop_reason)) => {
            Some(format!("Provider stopped with: {}", stop_reason.as_str()))
        }
        _ => None,
    };
    Finish {
        usage: usage.map(normalize_usage).unwrap_or_default(),
        reason,
        error,
        ..Finish::default()
    }
}

/// The Converse block for one assistant block, or `None` when there is
/// nothing to send. A provider item still current goes back exactly as it
/// came; otherwise the block is built from its canonical fields for a
/// model of `family`.
#[deny(clippy::wildcard_enum_match_arm)]
pub(crate) fn to_aws(
    content: AssistantContent,
    family: Family,
) -> Result<Option<aws_bedrock::ContentBlock>, ProviderError> {
    if let Some(block) = content.native_item().and_then(block::from_json) {
        return Ok(Some(block));
    }
    match content {
        // Converse rejects blank text.
        AssistantContent::Text(text) if text.text.trim().is_empty() => Ok(None),
        AssistantContent::Text(text) => Ok(Some(aws_bedrock::ContentBlock::Text(text.text))),
        AssistantContent::ToolCall(call) => aws_bedrock::ToolUseBlock::builder()
            .tool_use_id(call.id.wire())
            .name(call.function.name)
            .input(json::to_document(serde_json::Value::Object(
                call.function.arguments,
            )))
            .build()
            .map(|call| Some(aws_bedrock::ContentBlock::ToolUse(call)))
            .map_err(ProviderError::request),
        // Redacted reasoning is nothing without its bytes.
        AssistantContent::Reasoning(reasoning)
            if reasoning.redacted || reasoning.text.trim().is_empty() =>
        {
            Ok(None)
        }
        // Claude rejects unsigned reasoning: it goes back as text.
        AssistantContent::Reasoning(reasoning) if family == Family::Claude => {
            Ok(Some(aws_bedrock::ContentBlock::Text(reasoning.text)))
        }
        AssistantContent::Reasoning(reasoning) => aws_bedrock::ReasoningTextBlock::builder()
            .text(reasoning.text)
            .build()
            .map(|block| {
                Some(aws_bedrock::ContentBlock::ReasoningContent(
                    aws_bedrock::ReasoningContentBlock::ReasoningText(block),
                ))
            })
            .map_err(ProviderError::request),
        AssistantContent::Image(image) => Ok(Some(aws_bedrock::ContentBlock::Image(
            super::image::to_aws(image)?,
        ))),
        // A hosted tool's use or result goes back to the model that ran it.
        AssistantContent::Opaque(opaque) if opaque.replay => Ok(block::from_json(&opaque.item)),
        AssistantContent::Opaque(_) => Ok(None),
    }
}

#[cfg(test)]
mod tests;
