//! Crate-internal streaming profile helpers for compatible provider tests.

use crate::{
    completion::CompletionError,
    providers::internal::openai_chat_completions_compatible::{
        CompatibleChoice, CompatibleChunk, CompatibleFinishReason, CompatibleToolCallChunk,
        NormalizedCompatibleChunk,
    },
};

use super::streaming::MOCK_PROVIDER;

fn test_chunk(choice: CompatibleChoice) -> CompatibleChunk {
    CompatibleChunk {
        response_id: None,
        response_model: None,
        choice: Some(choice),
        usage: None,
    }
}

fn tool_call_choice(
    finish_reason: CompatibleFinishReason,
    tool_calls: Vec<CompatibleToolCallChunk>,
) -> CompatibleChoice {
    CompatibleChoice {
        finish_reason,
        text: None,
        reasoning: None,
        tool_calls,
        details: Vec::new(),
    }
}

fn tool_call_chunk(
    index: usize,
    id: Option<&str>,
    name: Option<&str>,
    arguments: Option<&str>,
) -> CompatibleToolCallChunk {
    CompatibleToolCallChunk {
        index,
        id: id.map(ToOwned::to_owned),
        name: name.map(ToOwned::to_owned),
        arguments: arguments.map(ToOwned::to_owned),
    }
}

/// Scripted chunk normalizers that drive the shared compatible stream state
/// machine directly, without a wire dialect.
///
/// The state-machine tests script chunk *outcomes* (a pending tool call, a
/// normalize error, an eviction sequence) rather than provider JSON, so they
/// pin the machine's own semantics. As a plain enum they are an arm of
/// [`ChunkNormalizer`](crate::providers::internal::openai_chat_completions_compatible::ChunkNormalizer)
/// instead of trait impls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TestNormalizer {
    /// Yields a pending tool call, then errors — the "errors terminate without
    /// flushing" case.
    ErrorAfterPendingToolCall,
    /// Same-index tool calls with distinct ids, which must evict.
    DistinctToolCallEviction,
    /// An unfinished tool call finalized by a `tool_calls` finish reason.
    FinishReasonCleanup,
}

impl TestNormalizer {
    /// The scripted provider name on the terminal record.
    pub(crate) fn provider_name(&self) -> &'static str {
        MOCK_PROVIDER
    }

    /// Whether same-index tool calls with distinct ids evict.
    pub(crate) fn uses_distinct_tool_call_eviction(&self) -> bool {
        matches!(self, Self::DistinctToolCallEviction)
    }

    /// Map a scripted SSE `data` token to a normalized chunk.
    pub(crate) fn normalize(&self, data: &str) -> NormalizedCompatibleChunk {
        match self {
            Self::ErrorAfterPendingToolCall => match data {
                "start" => Ok(Some(test_chunk(tool_call_choice(
                    CompatibleFinishReason::Other,
                    vec![tool_call_chunk(0, Some("call_123"), Some("ping"), Some(""))],
                )))),
                "bad" => Err(CompletionError::ProviderError(
                    "normalize failed".to_owned(),
                )),
                _ => Ok(None),
            },
            Self::DistinctToolCallEviction => {
                let choice = match data {
                    "first_start" => Some(tool_call_choice(
                        CompatibleFinishReason::Other,
                        vec![tool_call_chunk(
                            0,
                            Some("call_aaa"),
                            Some("search"),
                            Some(""),
                        )],
                    )),
                    "first_args" => Some(tool_call_choice(
                        CompatibleFinishReason::Other,
                        vec![tool_call_chunk(0, None, None, Some("{\"query\":\"one\"}"))],
                    )),
                    // A partial (still-streaming) argument fragment: the
                    // accumulator holds it as a bare `Value::String` because it
                    // is not yet a complete `{...}` object. If the first call is
                    // evicted at this point, its arguments are a non-object
                    // string — the exact #1958 leak condition.
                    "first_args_partial" => Some(tool_call_choice(
                        CompatibleFinishReason::Other,
                        vec![tool_call_chunk(0, None, None, Some("{\"query\":"))],
                    )),
                    "second_start" => Some(tool_call_choice(
                        CompatibleFinishReason::Other,
                        vec![tool_call_chunk(
                            0,
                            Some("call_bbb"),
                            Some("search"),
                            Some(""),
                        )],
                    )),
                    "second_args" => Some(tool_call_choice(
                        CompatibleFinishReason::Other,
                        vec![tool_call_chunk(0, None, None, Some("{\"query\":\"two\"}"))],
                    )),
                    "finish" => Some(tool_call_choice(
                        CompatibleFinishReason::ToolCalls,
                        Vec::new(),
                    )),
                    _ => None,
                };
                Ok(choice.map(test_chunk))
            }
            Self::FinishReasonCleanup => {
                let choice = match data {
                    "start" => Some(tool_call_choice(
                        CompatibleFinishReason::Other,
                        vec![tool_call_chunk(
                            0,
                            Some("call_123"),
                            Some("ping"),
                            Some("{\"x\":"),
                        )],
                    )),
                    "finish" => Some(tool_call_choice(
                        CompatibleFinishReason::ToolCalls,
                        Vec::new(),
                    )),
                    _ => None,
                };
                Ok(choice.map(test_chunk))
            }
        }
    }
}
