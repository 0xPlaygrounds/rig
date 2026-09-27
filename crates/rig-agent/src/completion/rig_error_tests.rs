//! Each rig-agent error converts with the kind, retry verdict, message and
//! source chain written out here.

use rig_core::error::{ErrorKind, RigError};
use rig_core::memory::MemoryError;

use super::*;
use crate::agent::StreamingError;
use crate::run::prepare::PrepareError;
use crate::tool::server::ToolServerError;

/// `error`'s classification, as the fields a caller routes on.
fn assert_converts(
    error: RigError,
    kind: ErrorKind,
    retryable: bool,
    message: &str,
    source_chain: &[&str],
) {
    assert_eq!(error.kind, kind, "{error:?}");
    assert_eq!(error.retryable, retryable, "{error:?}");
    assert_eq!(error.message, message, "{error:?}");
    assert_eq!(error.source_chain, source_chain, "{error:?}");
}

/// A provider's refusal, as a failed run carries it.
fn refusal() -> RigError {
    RigError::new(ErrorKind::ProviderResponse, "ProviderResponseError: boom")
        .with_http_status(503)
        .with_retryable(true)
}

fn cancelled() -> PromptError {
    PromptError::prompt_cancelled(Vec::new(), "stopped by a hook")
}

#[test]
fn a_prompt_error_converts_by_what_ended_the_run() {
    assert_eq!(RigError::from(PromptError::Failed(refusal())), refusal());
    assert_converts(
        PromptError::MemoryError(MemoryError::Policy("history too long".to_owned())).into(),
        ErrorKind::MemoryPolicy,
        false,
        "Memory policy error: history too long",
        &[],
    );
    assert_converts(
        cancelled().into(),
        ErrorKind::Cancelled,
        false,
        "PromptCancelled: stopped by a hook",
        &[],
    );
    assert_converts(
        PromptError::UnknownToolCall {
            tool_name: "search".to_owned(),
            available_tools: vec!["add".to_owned()],
            allowed_tools: vec!["add".to_owned()],
            chat_history: Vec::new(),
        }
        .into(),
        ErrorKind::Response,
        false,
        "UnknownToolCall: model attempted to call unknown or disallowed tool `search`. \
         Available tools: [\"add\"]. Allowed tools for this turn: [\"add\"]",
        &[],
    );
    assert_converts(
        PromptError::MaxTurnsError {
            max_turns: 3,
            chat_history: Vec::new(),
            prompt: Message::user("go"),
        }
        .into(),
        ErrorKind::Other,
        false,
        "MaxTurnsError: reached max turns limit: 3",
        &[],
    );
}

#[test]
fn a_streaming_error_converts_as_its_failure_or_prompt_error() {
    assert_eq!(RigError::from(StreamingError::Failed(refusal())), refusal());
    assert_converts(
        StreamingError::Prompt(cancelled()).into(),
        ErrorKind::Cancelled,
        false,
        "PromptCancelled: stopped by a hook",
        &[],
    );
}

#[test]
fn a_structured_output_error_is_a_response_rig_cannot_use() {
    assert_eq!(
        RigError::from(StructuredOutputError::PromptError(PromptError::Failed(
            refusal()
        ))),
        refusal()
    );
    let json = serde_json::from_str::<serde_json::Value>("{").expect_err("truncated JSON");
    assert_converts(
        StructuredOutputError::DeserializationError(json).into(),
        ErrorKind::Response,
        false,
        "DeserializationError: EOF while parsing an object at line 1 column 1",
        &["EOF while parsing an object at line 1 column 1"],
    );
    assert_converts(
        StructuredOutputError::EmptyResponse.into(),
        ErrorKind::Response,
        false,
        "EmptyResponse: model returned no content",
        &[],
    );
}

#[test]
fn a_prepare_error_reports_as_the_agent_reports_it() {
    assert_converts(
        PrepareError::Request("`lookup` is not an available tool".to_owned()).into(),
        ErrorKind::Request,
        false,
        "RequestError: `lookup` is not an available tool",
        &["`lookup` is not an available tool"],
    );
}

#[test]
fn a_tool_server_error_is_its_failure() {
    assert_eq!(
        RigError::from(ToolServerError::DefinitionError(refusal())),
        refusal()
    );
}
