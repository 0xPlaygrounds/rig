//! Crate-internal streaming profile helpers for compatible provider tests.

use crate::{
    completion::Usage,
    providers::internal::openai_chat_completions_compatible::{
        CompatibleChoice, CompatibleChunk, CompatibleFinishReason, CompatibleStreamProfile,
        CompatibleTerminal, CompatibleToolCallChunk,
    },
    providers::internal::wire::WireEvent,
    streaming::StreamFinal,
};

use super::MOCK_PROVIDER;

fn test_chunk(choice: CompatibleChoice<()>) -> CompatibleChunk<Usage, ()> {
    CompatibleChunk {
        response_id: None,
        response_model: None,
        choice: Some(choice),
        usage: None,
    }
}

/// Classify an unmatched mock frame as `Unknown`, mirroring how the real
/// classifier treats unrecognizable JSON (driver policy: warn + skip).
fn unknown_frame<U, D>(data: &str) -> WireEvent<CompatibleChunk<U, D>> {
    WireEvent::Unknown {
        event_type: data.to_owned(),
        value: serde_json::Value::String(data.to_owned()),
    }
}

fn tool_call_choice(
    finish_reason: CompatibleFinishReason,
    tool_calls: Vec<CompatibleToolCallChunk>,
) -> CompatibleChoice<()> {
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

/// Streaming profile that yields a pending tool call and then a corrupt frame.
#[derive(Clone, Copy)]
pub(crate) struct ErrorAfterPendingToolCallProfile;

impl CompatibleStreamProfile for ErrorAfterPendingToolCallProfile {
    type Usage = Usage;
    type Detail = ();
    type FinalResponse = StreamFinal;

    fn classify_chunk(&self, data: &str) -> WireEvent<CompatibleChunk<Self::Usage, Self::Detail>> {
        match data {
            "start" => WireEvent::Known(test_chunk(tool_call_choice(
                CompatibleFinishReason::Absent,
                vec![tool_call_chunk(0, Some("call_123"), Some("ping"), Some(""))],
            ))),
            "bad" => WireEvent::Corrupt(<serde_json::Error as serde::de::Error>::custom(
                "normalize failed",
            )),
            _ => unknown_frame(data),
        }
    }

    fn build_final_response(
        &self,
        terminal: CompatibleTerminal<Self::Usage>,
    ) -> Self::FinalResponse {
        StreamFinal::new(MOCK_PROVIDER, terminal.usage)
            .with_optional_finish_reason(terminal.finish_reason)
    }
}

/// Streaming profile whose same-index tool calls should evict by distinct IDs.
#[derive(Clone, Copy)]
pub(crate) struct DistinctToolCallEvictionProfile;

impl CompatibleStreamProfile for DistinctToolCallEvictionProfile {
    type Usage = Usage;
    type Detail = ();
    type FinalResponse = StreamFinal;

    fn classify_chunk(&self, data: &str) -> WireEvent<CompatibleChunk<Self::Usage, Self::Detail>> {
        let choice = match data {
            "first_start" => Some(tool_call_choice(
                CompatibleFinishReason::Absent,
                vec![tool_call_chunk(
                    0,
                    Some("call_aaa"),
                    Some("search"),
                    Some(""),
                )],
            )),
            "first_args" => Some(tool_call_choice(
                CompatibleFinishReason::Absent,
                vec![tool_call_chunk(0, None, None, Some("{\"query\":\"one\"}"))],
            )),
            // A partial (still-streaming) argument fragment: the accumulator
            // holds it as a bare `Value::String` because it is not yet a complete
            // `{...}` object. If the first call is evicted at this point, its
            // arguments are a non-object string — the exact #1958 leak condition.
            "first_args_partial" => Some(tool_call_choice(
                CompatibleFinishReason::Absent,
                vec![tool_call_chunk(0, None, None, Some("{\"query\":"))],
            )),
            "second_start" => Some(tool_call_choice(
                CompatibleFinishReason::Absent,
                vec![tool_call_chunk(
                    0,
                    Some("call_bbb"),
                    Some("search"),
                    Some(""),
                )],
            )),
            "second_args" => Some(tool_call_choice(
                CompatibleFinishReason::Absent,
                vec![tool_call_chunk(0, None, None, Some("{\"query\":\"two\"}"))],
            )),
            "finish" => Some(tool_call_choice(
                CompatibleFinishReason::Reported(crate::completion::FinishReason::ToolCalls),
                Vec::new(),
            )),
            _ => None,
        };

        match choice {
            Some(choice) => WireEvent::Known(test_chunk(choice)),
            None => unknown_frame(data),
        }
    }

    fn build_final_response(
        &self,
        terminal: CompatibleTerminal<Self::Usage>,
    ) -> Self::FinalResponse {
        StreamFinal::new(MOCK_PROVIDER, terminal.usage)
            .with_optional_finish_reason(terminal.finish_reason)
    }

    fn uses_distinct_tool_call_eviction(&self) -> bool {
        true
    }
}

/// Streaming profile with an unfinished tool call finalized by tool-calls finish reason.
#[derive(Clone, Copy)]
pub(crate) struct FinishReasonCleanupProfile;

impl CompatibleStreamProfile for FinishReasonCleanupProfile {
    type Usage = Usage;
    type Detail = ();
    type FinalResponse = StreamFinal;

    fn classify_chunk(&self, data: &str) -> WireEvent<CompatibleChunk<Self::Usage, Self::Detail>> {
        let choice = match data {
            "start" => Some(tool_call_choice(
                CompatibleFinishReason::Absent,
                vec![tool_call_chunk(
                    0,
                    Some("call_123"),
                    Some("ping"),
                    Some("{\"x\":"),
                )],
            )),
            "finish" => Some(tool_call_choice(
                CompatibleFinishReason::Reported(crate::completion::FinishReason::ToolCalls),
                Vec::new(),
            )),
            _ => None,
        };

        match choice {
            Some(choice) => WireEvent::Known(test_chunk(choice)),
            None => unknown_frame(data),
        }
    }

    fn build_final_response(
        &self,
        terminal: CompatibleTerminal<Self::Usage>,
    ) -> Self::FinalResponse {
        StreamFinal::new(MOCK_PROVIDER, terminal.usage)
            .with_optional_finish_reason(terminal.finish_reason)
    }
}
