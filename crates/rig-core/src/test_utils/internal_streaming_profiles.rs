//! Crate-internal streaming profile helpers for compatible provider tests.

use crate::{
    completion::{CompletionError, CompletionTerminalMetadata},
    providers::internal::openai_chat_completions_compatible::{
        CompatibleChoice, CompatibleChunk, CompatibleFinishReason, CompatibleStreamProfile,
        CompatibleToolCallChunk,
    },
};

use super::MockResponse;

fn test_chunk(choice: CompatibleChoice<()>) -> CompatibleChunk<MockResponse, ()> {
    CompatibleChunk {
        response_id: None,
        response_model: None,
        choice: Some(choice),
        usage: None,
    }
}

fn tool_call_choice(
    finish_reason: Option<CompatibleFinishReason>,
    tool_calls: Vec<CompatibleToolCallChunk>,
) -> CompatibleChoice<()> {
    CompatibleChoice {
        terminal_metadata: finish_reason.map(|reason| {
            CompletionTerminalMetadata::new(reason.into()).with_raw_reason(format!("{reason:?}"))
        }),
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

/// Streaming profile that yields a pending tool call and then errors.
#[derive(Clone, Copy)]
pub(crate) struct ErrorAfterPendingToolCallProfile;

impl CompatibleStreamProfile for ErrorAfterPendingToolCallProfile {
    type Usage = MockResponse;
    type Detail = ();
    type FinalResponse = MockResponse;

    fn normalize_chunk(
        &self,
        data: &str,
    ) -> Result<Option<CompatibleChunk<Self::Usage, Self::Detail>>, CompletionError> {
        match data {
            "start" => Ok(Some(test_chunk(tool_call_choice(
                None,
                vec![tool_call_chunk(0, Some("call_123"), Some("ping"), Some(""))],
            )))),
            "bad" => Err(CompletionError::ProviderError(
                "normalize failed".to_owned(),
            )),
            _ => Ok(None),
        }
    }

    fn build_final_response(
        &self,
        _usage: Self::Usage,
        terminal_metadata: Option<CompletionTerminalMetadata>,
    ) -> Self::FinalResponse {
        let response = MockResponse::new();
        if let Some(metadata) = terminal_metadata {
            response.with_terminal_metadata(metadata)
        } else {
            response
        }
    }
}

/// Streaming profile whose same-index tool calls should evict by distinct IDs.
#[derive(Clone, Copy)]
pub(crate) struct DistinctToolCallEvictionProfile;

impl CompatibleStreamProfile for DistinctToolCallEvictionProfile {
    type Usage = MockResponse;
    type Detail = ();
    type FinalResponse = MockResponse;

    fn normalize_chunk(
        &self,
        data: &str,
    ) -> Result<Option<CompatibleChunk<Self::Usage, Self::Detail>>, CompletionError> {
        let choice = match data {
            "first_start" => Some(tool_call_choice(
                None,
                vec![tool_call_chunk(
                    0,
                    Some("call_aaa"),
                    Some("search"),
                    Some(""),
                )],
            )),
            "first_args" => Some(tool_call_choice(
                None,
                vec![tool_call_chunk(0, None, None, Some("{\"query\":\"one\"}"))],
            )),
            "second_start" => Some(tool_call_choice(
                None,
                vec![tool_call_chunk(
                    0,
                    Some("call_bbb"),
                    Some("search"),
                    Some(""),
                )],
            )),
            "second_args" => Some(tool_call_choice(
                None,
                vec![tool_call_chunk(0, None, None, Some("{\"query\":\"two\"}"))],
            )),
            "finish" => Some(tool_call_choice(
                Some(CompatibleFinishReason::ToolCalls),
                Vec::new(),
            )),
            _ => None,
        };

        Ok(choice.map(test_chunk))
    }

    fn build_final_response(
        &self,
        _usage: Self::Usage,
        terminal_metadata: Option<CompletionTerminalMetadata>,
    ) -> Self::FinalResponse {
        let response = MockResponse::new();
        if let Some(metadata) = terminal_metadata {
            response.with_terminal_metadata(metadata)
        } else {
            response
        }
    }

    fn uses_distinct_tool_call_eviction(&self) -> bool {
        true
    }
}

/// Streaming profile with an unfinished tool call finalized by tool-calls finish reason.
#[derive(Clone, Copy)]
pub(crate) struct FinishReasonCleanupProfile;

impl CompatibleStreamProfile for FinishReasonCleanupProfile {
    type Usage = MockResponse;
    type Detail = ();
    type FinalResponse = MockResponse;

    fn normalize_chunk(
        &self,
        data: &str,
    ) -> Result<Option<CompatibleChunk<Self::Usage, Self::Detail>>, CompletionError> {
        let choice = match data {
            "start" => Some(tool_call_choice(
                None,
                vec![tool_call_chunk(
                    0,
                    Some("call_123"),
                    Some("ping"),
                    Some("{\"x\":"),
                )],
            )),
            "finish" => Some(tool_call_choice(
                Some(CompatibleFinishReason::ToolCalls),
                Vec::new(),
            )),
            _ => None,
        };

        Ok(choice.map(test_chunk))
    }

    fn build_final_response(
        &self,
        _usage: Self::Usage,
        terminal_metadata: Option<CompletionTerminalMetadata>,
    ) -> Self::FinalResponse {
        let response = MockResponse::new();
        if let Some(metadata) = terminal_metadata {
            response.with_terminal_metadata(metadata)
        } else {
            response
        }
    }
}
