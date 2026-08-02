//! Owned response data for one agent run, plus the history/tool-result
//! helpers the drivers share.
//!
//! [`PromptResponse`] is the unified result of a run on either driver: the
//! blocking [`AgentSession`](crate::session::AgentSession) returns it from
//! [`SessionEvent::Done`](crate::session::SessionEvent::Done), and the
//! streaming [`AgentStream`](crate::stream::AgentStream) surfaces it as the
//! terminal [`AgentStreamItem::Final`](crate::stream::AgentStreamItem::Final)
//! host item; a fully driven [`AgentRunStream`](crate::stream::AgentRunStream)
//! emits it as [`AgentRunItem::Final`](crate::stream::AgentRunItem::Final).

use rig_core::{
    OneOrMany,
    message::{AssistantContent, ToolResultContent, UserContent},
};

use crate::{
    completion::{Message, Usage},
    tool::ToolOutput,
};
use serde::{Deserialize, Serialize};

/// Details for one successfully completed completion request made by an agent run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CompletionCall {
    /// Zero-based index of the completion request within this agent run.
    pub call_index: usize,
    /// Token usage reported for this completion request.
    ///
    /// Zero-valued usage is [`Usage`]'s documented sentinel for missing
    /// provider usage metrics; rig does not distinguish "reported all zeros"
    /// from "unreported".
    #[serde(default, deserialize_with = "usage_null_as_default")]
    pub usage: Usage,
}

impl CompletionCall {
    /// Create details for one completion request in an agent run.
    pub fn new(call_index: usize, usage: Usage) -> Self {
        Self { call_index, usage }
    }
}

/// Tolerate `null` usage from data serialized before rig dropped the
/// `Option<Usage>` encoding of missing provider usage metrics.
///
/// This tolerance requires a self-describing format such as JSON; data
/// serialized with non-self-describing formats (e.g. bincode) from before the
/// change cannot round-trip.
fn usage_null_as_default<'de, D>(deserializer: D) -> Result<Usage, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<Usage>::deserialize(deserializer)?.unwrap_or_default())
}

/// The result of an agent run, returned by **both** the blocking
/// ([`AgentSession`]) and streaming ([`AgentStream`]) drivers so a call site
/// reads identically whether it used `.run()` or `.stream_prompt()`.
///
/// On the streaming driver this is the payload of the terminal host
/// [`AgentStreamItem::Final`] or driven [`AgentRunItem::Final`] item.
///
/// [`AgentSession`]: crate::session::AgentSession
/// [`AgentStream`]: crate::stream::AgentStream
/// [`AgentStreamItem::Final`]: crate::stream::AgentStreamItem::Final
/// [`AgentRunItem::Final`]: crate::stream::AgentRunItem::Final
#[derive(Debug, Clone, Serialize, Deserialize)]
// Serialize *and* deserialize both go through `PromptResponseRepr` so the two
// directions agree on `content`'s wire shape (an `Option`). Routing only
// deserialize through the shadow would make serialize write a bare `OneOrMany`
// while deserialize expects an `Option`, breaking round-trips for positional /
// non-self-describing formats (e.g. bincode). The repr carries the field serde
// attributes, so the JSON shape is unchanged.
#[serde(from = "PromptResponseRepr", into = "PromptResponseRepr")]
#[non_exhaustive]
pub struct PromptResponse {
    /// Concatenated assistant text for the final turn.
    pub output: String,
    /// Aggregated token usage across the whole run.
    pub usage: Usage,
    /// Successfully completed completion requests made by this agent run.
    ///
    /// `usage` remains the aggregate across the whole run. Use the last
    /// entry's usage to inspect the final completion request's prompt/context
    /// length. Zero-valued entry usage means the provider reported no usage
    /// metrics for that request.
    pub completion_calls: Vec<CompletionCall>,
    /// Accumulated message history for the run: the run's committed transcript
    /// (prompt, assistant turns, and tool-call/result pairs). Append these to
    /// a conversation store to persist the turn — see the
    /// [`agent_api`](crate::agent_api) module docs for the host recipe.
    pub messages: Option<Vec<Message>>,
    /// Structured assistant content for the final turn.
    ///
    /// Where [`output`](Self::output) is the concatenated text, this preserves
    /// the individual content parts (text, reasoning, images, …).
    pub content: OneOrMany<AssistantContent>,
    /// Number of synthetic output-tool calls in the turn that finalized this
    /// response. Kept crate-private because it is runner bookkeeping rather
    /// than provider-facing response content.
    output_tool_calls: usize,
}

/// Serde shadow for [`PromptResponse`]. `content` is an `Option` here so runs
/// serialized before the field existed still deserialize: a missing `content`
/// reconstructs the structured final turn from `output` (a single text part),
/// keeping [`PromptResponse::output`] and [`PromptResponse::content`] consistent
/// for legacy data rather than defaulting to empty text. It carries the field
/// serde attributes for both directions, keeping the serialized shape identical
/// (`completion_calls` omitted when empty; `messages`/`content` always present).
#[derive(Serialize, Deserialize)]
struct PromptResponseRepr {
    output: String,
    usage: Usage,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    completion_calls: Vec<CompletionCall>,
    messages: Option<Vec<Message>>,
    #[serde(default)]
    content: Option<OneOrMany<AssistantContent>>,
    #[serde(skip)]
    output_tool_calls: usize,
}

impl From<PromptResponseRepr> for PromptResponse {
    fn from(repr: PromptResponseRepr) -> Self {
        let content = repr
            .content
            .unwrap_or_else(|| OneOrMany::one(AssistantContent::text(repr.output.clone())));
        Self {
            output: repr.output,
            usage: repr.usage,
            completion_calls: repr.completion_calls,
            messages: repr.messages,
            content,
            output_tool_calls: repr.output_tool_calls,
        }
    }
}

impl From<PromptResponse> for PromptResponseRepr {
    fn from(response: PromptResponse) -> Self {
        Self {
            output: response.output,
            usage: response.usage,
            completion_calls: response.completion_calls,
            messages: response.messages,
            content: Some(response.content),
            output_tool_calls: response.output_tool_calls,
        }
    }
}

impl std::fmt::Display for PromptResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.output.fmt(f)
    }
}

impl PromptResponse {
    pub fn new(output: impl Into<String>, usage: Usage) -> Self {
        let output = output.into();
        Self {
            content: OneOrMany::one(AssistantContent::text(output.clone())),
            output,
            usage,
            completion_calls: Vec::new(),
            messages: None,
            output_tool_calls: 0,
        }
    }

    /// An empty run result (empty output, zero usage, no history).
    pub fn empty() -> Self {
        Self::new(String::new(), Usage::new())
    }

    pub fn with_messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = Some(messages);
        self
    }

    /// Attach completion call details to this response.
    pub fn with_completion_calls(mut self, completion_calls: Vec<CompletionCall>) -> Self {
        self.completion_calls = completion_calls;
        self
    }

    /// Set the structured assistant content for the final turn.
    pub fn with_content(mut self, content: OneOrMany<AssistantContent>) -> Self {
        self.content = content;
        self
    }

    pub(crate) fn with_output_tool_calls(mut self, count: usize) -> Self {
        self.output_tool_calls = count;
        self
    }

    /// How many synthetic output-tool calls this run's finalizing turn made.
    ///
    /// Structured-output runs in [`OutputMode::Tool`](crate::agent::OutputMode)
    /// finalize through a synthetic output-tool call; this is the marker an
    /// extraction protocol checks to distinguish "the model answered in prose"
    /// (`0`) from a real submission. Never serialized — it is a per-run
    /// observation, not persisted state.
    pub fn output_tool_calls(&self) -> usize {
        self.output_tool_calls
    }

    /// The concatenated assistant text for the final turn.
    pub fn output(&self) -> &str {
        &self.output
    }

    /// Aggregated token usage across the whole run.
    pub fn usage(&self) -> Usage {
        self.usage
    }

    /// The run's accumulated message history, if tracked.
    pub fn messages(&self) -> Option<&[Message]> {
        self.messages.as_deref()
    }

    /// The structured assistant content for the final turn.
    pub fn content(&self) -> &OneOrMany<AssistantContent> {
        &self.content
    }

    /// Returns successfully completed completion requests made by this agent run.
    ///
    /// Zero-valued entry usage means the provider reported no usage metrics
    /// for that request.
    pub fn completion_calls(&self) -> &[CompletionCall] {
        &self.completion_calls
    }

    /// Number of completion requests this agent run made.
    pub fn requests(&self) -> usize {
        self.completion_calls.len()
    }
}

pub(crate) const TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER: &str =
    "Tool not executed because another tool call in the same assistant turn was invalid.";

/// Combine input history with new messages for building completion requests.
pub(crate) fn build_history_for_request(
    chat_history: Option<&[Message]>,
    new_messages: &[Message],
) -> Vec<Message> {
    let input = chat_history.unwrap_or(&[]);
    input.iter().chain(new_messages.iter()).cloned().collect()
}

/// Build the full history for error reporting (input + new messages).
pub(crate) fn build_full_history(
    chat_history: Option<&[Message]>,
    new_messages: Vec<Message>,
) -> Vec<Message> {
    let input = chat_history.unwrap_or(&[]);
    input.iter().cloned().chain(new_messages).collect()
}

/// Wrap already-shaped tool-result content for the model (see
/// [`tool_result_output`] / [`tool_result_message`]).
fn tool_result_with(
    id: String,
    call_id: Option<String>,
    content: OneOrMany<ToolResultContent>,
) -> UserContent {
    match call_id {
        Some(call_id) => UserContent::tool_result_with_call_id(id, call_id, content),
        None => UserContent::tool_result(id, content),
    }
}

/// Shape a canonical real tool output as a tool result without reparsing text.
pub(crate) fn tool_result_output(
    id: String,
    call_id: Option<String>,
    output: ToolOutput,
) -> UserContent {
    tool_result_with(id, call_id, output.into_content())
}

/// Shape a **synthetic message** (a hook skip reason, recovery feedback, or a
/// "not executed" notice) as a tool result. Emitted **verbatim as text** and
/// never re-parsed as structured tool output, so a JSON-shaped message is not
/// silently reinterpreted as an image/multimodal result. Used identically by the
/// blocking and streaming drivers so synthetic results match across both.
pub(crate) fn tool_result_message(
    id: String,
    call_id: Option<String>,
    message: String,
) -> UserContent {
    tool_result_with(
        id,
        call_id,
        OneOrMany::one(ToolResultContent::text(message)),
    )
}

pub(crate) fn invalid_tool_retry_user_message(
    assistant_content: &OneOrMany<AssistantContent>,
    invalid_tool_call_id: &str,
    feedback: String,
) -> Option<Message> {
    let retry_results = assistant_content
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(tool_call) if tool_call.id == invalid_tool_call_id => {
                Some(tool_result_message(
                    tool_call.id.clone(),
                    tool_call.call_id.clone(),
                    feedback.clone(),
                ))
            }
            AssistantContent::ToolCall(tool_call) => Some(tool_result_message(
                tool_call.id.clone(),
                tool_call.call_id.clone(),
                TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER.to_string(),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();

    Some(Message::User {
        content: OneOrMany::from_iter_optional(retry_results)?,
    })
}

pub(crate) fn is_empty_assistant_turn(choice: &OneOrMany<AssistantContent>) -> bool {
    choice.len() == 1
        && matches!(
            choice.first(),
            AssistantContent::Text(text) if text.text.is_empty() && text.additional_params.is_none()
        )
}

pub(crate) fn assistant_text_from_choice(choice: &OneOrMany<AssistantContent>) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{CompletionCall, PromptResponse, PromptResponseRepr};
    use crate::completion::{AssistantContent, Usage};
    use serde_json::json;

    fn usage(input_tokens: u64, output_tokens: u64) -> Usage {
        Usage {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
            tool_use_prompt_tokens: 0,
            reasoning_tokens: 0,
        }
    }

    #[test]
    fn prompt_response_serializes_completion_calls_with_missing_usage() {
        let reported_usage = usage(3, 4);
        let response = PromptResponse::new("ok", reported_usage).with_completion_calls(vec![
            CompletionCall::new(0, Usage::new()),
            CompletionCall::new(1, reported_usage),
        ]);

        let value = serde_json::to_value(&response).expect("serialize prompt response");

        // Unreported usage serializes as a plain zero-valued object: zero is
        // Usage's documented sentinel for missing provider metrics, so there
        // is no null encoding to keep in sync.
        assert_eq!(
            value.get("completion_calls"),
            Some(&json!([
                {
                    "call_index": 0,
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "cached_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "tool_use_prompt_tokens": 0,
                        "reasoning_tokens": 0,
                    }
                },
                {
                    "call_index": 1,
                    "usage": {
                        "input_tokens": 3,
                        "output_tokens": 4,
                        "total_tokens": 7,
                        "cached_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "tool_use_prompt_tokens": 0,
                        "reasoning_tokens": 0,
                    }
                }
            ]))
        );

        let response: PromptResponse =
            serde_json::from_value(value).expect("deserialize prompt response");
        assert_eq!(
            response.completion_calls(),
            &[
                CompletionCall::new(0, Usage::new()),
                CompletionCall::new(1, reported_usage)
            ]
        );
        assert_eq!(response.requests(), 2);
    }

    #[test]
    fn prompt_response_output_tool_marker_is_never_serialized() {
        let response = PromptResponse::new("ok", usage(1, 2)).with_output_tool_calls(3);

        let value = serde_json::to_value(&response).expect("serialize prompt response");
        assert!(value.get("output_tool_calls").is_none());

        let decoded: PromptResponse =
            serde_json::from_value(value).expect("deserialize prompt response");
        assert_eq!(decoded.output_tool_calls(), 0);
    }

    #[test]
    fn prompt_response_deserializes_pre_monoid_null_usage_format() {
        // Fixture captured from rig before CompletionCall.usage dropped its
        // Option encoding; `"usage": null` must map to zero-valued usage.
        let fixture = r#"{"output":"ok","usage":{"input_tokens":3,"output_tokens":4,"total_tokens":7,"cached_input_tokens":0,"cache_creation_input_tokens":0,"tool_use_prompt_tokens":0,"reasoning_tokens":0},"completion_calls":[{"call_index":0,"usage":null},{"call_index":1,"usage":{"input_tokens":3,"output_tokens":4,"total_tokens":7,"cached_input_tokens":0,"cache_creation_input_tokens":0,"tool_use_prompt_tokens":0,"reasoning_tokens":0}}],"messages":[{"role":"user","content":[{"type":"text","text":"add things"}]}]}"#;

        let response: PromptResponse =
            serde_json::from_str(fixture).expect("old-format response should deserialize");
        assert_eq!(
            response.completion_calls(),
            &[
                CompletionCall::new(0, Usage::new()),
                CompletionCall::new(1, usage(3, 4))
            ]
        );
    }

    #[test]
    fn prompt_response_missing_content_reconstructs_from_output() {
        // Runs serialized before `content` existed must not deserialize to empty
        // text: the structured final turn is reconstructed from `output`, so
        // `output()` and `content()` stay consistent for legacy data.
        let mut value = serde_json::to_value(PromptResponse::new("hello", Usage::new()))
            .expect("serialize prompt response");
        value
            .as_object_mut()
            .expect("prompt response serializes to a JSON object")
            .remove("content");
        assert!(
            value.get("content").is_none(),
            "fixture must omit the content field to model legacy data"
        );

        let response: PromptResponse = serde_json::from_value(value)
            .expect("legacy response without content should deserialize");

        assert_eq!(response.output(), "hello");
        assert_eq!(response.content().iter().count(), 1);
        assert_eq!(response.content().first(), AssistantContent::text("hello"));
    }

    #[test]
    fn prompt_response_missing_content_empty_output_stays_empty_text() {
        let mut value =
            serde_json::to_value(PromptResponse::empty()).expect("serialize prompt response");
        value
            .as_object_mut()
            .expect("prompt response serializes to a JSON object")
            .remove("content");

        let response: PromptResponse = serde_json::from_value(value)
            .expect("legacy empty response without content should deserialize");

        assert_eq!(response.output(), "");
        assert_eq!(response.content().first(), AssistantContent::text(""));
    }

    #[test]
    fn prompt_response_roundtrip_preserves_explicit_content() {
        // An explicitly-set `content` (e.g. the streaming surface's structured
        // final turn) must survive a serialize/deserialize round-trip and is not
        // clobbered by the output-derived fallback.
        let response = PromptResponse::new("visible text", Usage::new()).with_content(
            rig_core::OneOrMany::one(AssistantContent::text("structured")),
        );

        let value = serde_json::to_value(&response).expect("serialize prompt response");
        assert!(
            value.get("content").is_some(),
            "content is part of the serialized shape"
        );

        let round: PromptResponse =
            serde_json::from_value(value).expect("deserialize prompt response");
        assert_eq!(round.output(), "visible text");
        // The stored content is "structured" — distinct from `output` — proving the
        // output-derived fallback only fills a genuinely absent `content`.
        let AssistantContent::Text(text) = round.content().first() else {
            panic!("expected text content, got {:?}", round.content().first());
        };
        assert_eq!(text.text, "structured");
    }

    #[test]
    fn prompt_response_serialize_and_deserialize_agree_on_wire_shape() {
        // Serialize *and* deserialize both route through `PromptResponseRepr`, so
        // the two directions agree on `content`'s wire shape (an `Option`).
        let response = PromptResponse::new("hi", usage(1, 2))
            .with_completion_calls(vec![CompletionCall::new(0, usage(1, 2))]);

        let from_response = serde_json::to_value(&response).expect("serialize response");
        let from_shadow = serde_json::to_value(PromptResponseRepr::from(response.clone()))
            .expect("serialize shadow");
        assert_eq!(
            from_response, from_shadow,
            "serialize must route through the same shadow as deserialize"
        );

        let round: PromptResponse =
            serde_json::from_value(from_response).expect("deserialize response");
        assert_eq!(round.output(), "hi");
        assert_eq!(round.usage(), usage(1, 2));
        assert_eq!(
            round.completion_calls(),
            &[CompletionCall::new(0, usage(1, 2))]
        );
    }
}
