pub mod streaming;

use super::{Agent, hook::AgentHook, run::OutputMode, runner::AgentRunner};
use rig_core::{
    completion::FinishReason,
    message::{
        AssistantContent, ProviderCallId, ToolCallId, ToolResultContent, UserContent, non_empty,
    },
    wasm_compat::{WasmBoxedFuture, WasmCompatSend},
};

use crate::{
    completion::{Message, PromptError, Usage},
    tool::{ToolContext, ToolOutput},
};
use serde::{Deserialize, Serialize};
use std::{future::IntoFuture, marker::PhantomData};

/// The provider-neutral identity carrier, re-exported from rig-core so agent
/// callers name one type across core responses, stream terminals, completion
/// calls, and hook events.
pub use rig_core::completion::ResponseIdentity;

/// Generate the request-builder setters that forward verbatim to an inner
/// receiver — `AgentRunner` for the blocking builder, the wrapped
/// `PromptRequest` for the typed builder, and the `AgentRunner` for the
/// streaming builder. Only the setters whose signature *and* documentation are
/// identical across all three builders live here; `max_turns`, `add_hook`, and
/// `tool_concurrency`, whose docs are builder-specific, stay hand-written (the
/// blocking builders share `tool_concurrency` via [`forward_tool_concurrency`]).
/// `$recv` is the field name to delegate through (`runner` or `inner`).
macro_rules! forward_prompt_setters {
    ($recv:ident) => {
        /// Attach a per-call [`ToolContext`] for this request.
        ///
        /// Every tool the agent executes during this request can read the
        /// caller-provided values (auth tokens, session IDs, conversation state, …)
        /// through the tool's [`ToolContext`](crate::tool::ToolContext),
        /// without the model ever seeing them.
        pub fn tool_context(mut self, context: ToolContext) -> Self {
            self.$recv = self.$recv.tool_context(context);
            self
        }

        /// Add chat history to the prompt request.
        pub fn history<H, Item>(mut self, history: H) -> Self
        where
            H: IntoIterator<Item = Item>,
            Item: Into<Message>,
        {
            self.$recv = self.$recv.history(history);
            self
        }

        /// Override the agent preamble for this request.
        pub fn preamble(mut self, preamble: impl Into<String>) -> Self {
            self.$recv = self.$recv.preamble(preamble);
            self
        }

        /// Remove the agent's configured preamble for this request.
        pub fn without_preamble(mut self) -> Self {
            self.$recv = self.$recv.without_preamble();
            self
        }

        /// Append one static context document for this request.
        pub fn document(mut self, document: crate::completion::Document) -> Self {
            self.$recv = self.$recv.document(document);
            self
        }

        /// Append static context documents for this request.
        pub fn documents(
            mut self,
            documents: impl IntoIterator<Item = crate::completion::Document>,
        ) -> Self {
            self.$recv = self.$recv.documents(documents);
            self
        }

        /// Override the model temperature for this request.
        pub fn temperature(mut self, temperature: f64) -> Self {
            self.$recv = self.$recv.temperature(temperature);
            self
        }

        /// Remove the agent's configured temperature for this request.
        pub fn without_temperature(mut self) -> Self {
            self.$recv = self.$recv.without_temperature();
            self
        }

        /// Override the maximum completion token count for this request.
        pub fn max_tokens(mut self, max_tokens: u64) -> Self {
            self.$recv = self.$recv.max_tokens(max_tokens);
            self
        }

        /// Remove the agent's configured maximum token count for this request.
        pub fn without_max_tokens(mut self) -> Self {
            self.$recv = self.$recv.without_max_tokens();
            self
        }

        /// Shallow-merge object fields into the provider-specific parameters
        /// for this request. Later fields win.
        pub fn merge_additional_params(
            mut self,
            params: serde_json::Map<String, serde_json::Value>,
        ) -> Self {
            self.$recv = self.$recv.merge_additional_params(params);
            self
        }

        /// Replace all provider-specific parameters for this request.
        pub fn replace_additional_params(mut self, params: serde_json::Value) -> Self {
            self.$recv = self.$recv.replace_additional_params(params);
            self
        }

        /// Remove the agent's configured provider-specific parameters for this request.
        pub fn without_additional_params(mut self) -> Self {
            self.$recv = self.$recv.without_additional_params();
            self
        }

        /// Override the tool-choice policy for this request.
        pub fn tool_choice(mut self, tool_choice: rig_core::message::ToolChoice) -> Self {
            self.$recv = self.$recv.tool_choice(tool_choice);
            self
        }

        /// Remove the agent's configured tool-choice policy for this request.
        pub fn without_tool_choice(mut self) -> Self {
            self.$recv = self.$recv.without_tool_choice();
            self
        }

        /// Opt in or out of recording sensitive request, response, and tool
        /// content on GenAI telemetry spans for this request.
        ///
        /// Defaults to the agent's setting, which defaults to `false`. Enabling
        /// this can expose prompts, retrieved context, tool results, model
        /// responses, and other sensitive or high-cardinality data through
        /// OpenTelemetry span attributes, which can increase observability
        /// backend storage and query costs. Only enable it when content
        /// telemetry is acceptable for this request. Structural metadata and
        /// token usage remain available when disabled.
        pub fn record_content_telemetry(mut self, enabled: bool) -> Self {
            self.$recv = self.$recv.record_content_telemetry(enabled);
            self
        }

        /// Set the conversation id used to load and persist memory for this request.
        ///
        /// Overrides any default conversation id set on the agent. If memory is not
        /// configured on the agent, this has no effect.
        pub fn conversation(mut self, id: impl Into<String>) -> Self {
            self.$recv = self.$recv.conversation(id);
            self
        }

        /// Disable conversation memory for this request.
        ///
        /// History will neither be loaded from nor saved to the agent's memory backend.
        pub fn without_memory(mut self) -> Self {
            self.$recv = self.$recv.without_memory();
            self
        }

        /// Set the retry budget for invalid tool-call recovery.
        ///
        /// Invalid tool-call retries also consume the total model-call budget.
        pub fn max_invalid_tool_call_retries(mut self, retries: usize) -> Self {
            self.$recv = self.$recv.max_invalid_tool_call_retries(retries);
            self
        }

        /// Set the default model candidate for this run.
        ///
        /// This does not suppress registered model-selection hooks, which may
        /// replace this candidate before each model call (including retries).
        pub fn using_model(mut self, model: $crate::agent::ModelHandle) -> Self {
            self.$recv = self.$recv.using_model(model);
            self
        }

        /// Erase and set a typed default model for this run.
        pub fn using_model_value<M>(mut self, model: M) -> Self
        where
            M: $crate::completion::CompletionModel + 'static,
        {
            self.$recv = self.$recv.using_model_value(model);
            self
        }
    };
}
pub(crate) use forward_prompt_setters;

/// Generate the `tool_concurrency` setter for the blocking builders, whose doc
/// is identical to each other but differs from the streaming builder's (the
/// streaming version documents how tool items are ordered in the emitted
/// stream). `$recv` is the field name to delegate through (`runner` or `inner`).
macro_rules! forward_tool_concurrency {
    ($recv:ident) => {
        /// Execute up to `concurrency` of a turn's tool calls at once.
        ///
        /// See [`AgentRunner::tool_concurrency`] for ordering guarantees: the tool
        /// batch commits and surfaces atomically, so persisted history and streamed
        /// tool results are both in tool-call order (results are surfaced only after
        /// the whole batch settles successfully).
        pub fn tool_concurrency(mut self, concurrency: usize) -> Self {
            self.$recv = self.$recv.tool_concurrency(concurrency);
            self
        }
    };
}

pub trait PromptType {}
pub struct Standard;
pub struct Extended;

impl PromptType for Standard {}
impl PromptType for Extended {}

/// A builder for creating prompt requests with customizable options.
/// Uses generics to track which options have been set during the build process.
///
/// When the agent has no configured `default_max_turns`, the implicit budget is
/// one model call. Use [`.max_turns()`](Self::max_turns) to override the agent's
/// configured or implicit budget; a tool call followed by a model-authored final
/// answer generally requires at least two model calls.
pub struct PromptRequest<S>
where
    S: PromptType,
{
    /// The hook-aware driver this request configures and runs.
    pub(crate) runner: AgentRunner,
    /// Phantom data to track the type of the request (Standard vs Extended).
    state: PhantomData<S>,
}

impl PromptRequest<Standard> {
    /// Create a new PromptRequest from an agent, cloning the agent's data and
    /// default hook stack.
    pub fn from_agent(agent: &Agent, prompt: impl Into<Message>) -> Self {
        PromptRequest {
            runner: AgentRunner::from_agent(agent, prompt),
            state: PhantomData,
        }
    }
}

impl<S> PromptRequest<S>
where
    S: PromptType,
{
    /// Enable returning extended details for responses (includes aggregated token usage
    /// and the full message history accumulated during the agent loop).
    ///
    /// Note: This changes the type of the response from `.send` to return a `PromptResponse` struct
    /// instead of a simple `String`. This is useful for tracking token usage across multiple turns
    /// of conversation and inspecting the full message exchange.
    pub fn extended_details(self) -> PromptRequest<Extended> {
        PromptRequest {
            runner: self.runner,
            state: PhantomData,
        }
    }

    /// Set the total model-call budget, including the initial call and every
    /// retry or continuation. Zero emits no model calls; one permits only the
    /// initial call. Exceeding the budget returns
    /// [`crate::completion::PromptError::MaxTurnsError`].
    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.runner = self.runner.max_turns(max_turns);
        self
    }

    /// Append a hook for this request (on top of any the agent already carries).
    /// Hooks run in registration order; how their results compose is
    /// event-dependent (model selections and `ToolCall`/`ToolResult` rewrites
    /// chain, `CompletionCall` request patches accumulate and merge, while
    /// model-turn steering and observe-only/recovery events use
    /// first-non-`Continue`-wins). See the [`hook`](crate::agent::hook) module
    /// docs.
    pub fn add_hook<H>(mut self, hook: H) -> Self
    where
        H: AgentHook + 'static,
    {
        self.runner = self.runner.add_hook(hook);
        self
    }

    forward_prompt_setters!(runner);
    forward_tool_concurrency!(runner);
}

/// Due to: [RFC 2515](https://github.com/rust-lang/rust/issues/63063), we have to use a `BoxFuture`
///  for the `IntoFuture` implementation. In the future, we should be able to use `impl Future<...>`
///  directly via the associated type.
impl IntoFuture for PromptRequest<Standard> {
    type Output = Result<String, PromptError>;
    type IntoFuture = WasmBoxedFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.send())
    }
}

impl IntoFuture for PromptRequest<Extended> {
    type Output = Result<PromptResponse, PromptError>;
    type IntoFuture = WasmBoxedFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.send())
    }
}

impl PromptRequest<Standard> {
    async fn send(self) -> Result<String, PromptError> {
        self.extended_details().send().await.map(|resp| resp.output)
    }
}

/// Details for one successfully completed completion request made by an agent run.
// No longer `Copy`: the identity fields carry owned strings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
    /// Provider-assigned assistant message ID for this call, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
    /// Provider-assigned response-scoped ID for this call, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// The provider's transport request id for this call (HTTP response
    /// header, e.g. Anthropic `request-id`) — the id provider support asks
    /// for. `None` means the provider did not report one, never an error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_request_id: Option<String>,
    /// Why the model stopped generating on this call, when the provider
    /// reported it. `None` means the provider reported no reason.
    ///
    /// Recorded **per call** rather than once per run: a multi-turn run makes N
    /// completion requests, each with its own terminal reason, and collapsing
    /// them to a single run-level value would lose exactly the information that
    /// makes a truncated turn diagnosable — which turn hit the limit. A caller
    /// that wants the run's last reason reads it off the final entry.
    ///
    /// This is the field whose absence hid rig#2322: the provider layer carried
    /// [`FinishReason::Length`] on the stream's terminal record, but the agent
    /// assembler dropped it, so a turn truncated at the output-token limit was
    /// indistinguishable from a turn that simply had nothing to say.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
}

impl CompletionCall {
    /// Create details for one completion request in an agent run; identity
    /// metadata starts unset and is attached with [`Self::with_identity`].
    pub fn new(call_index: usize, usage: Usage) -> Self {
        Self {
            call_index,
            usage,
            message_id: None,
            response_id: None,
            provider_request_id: None,
            finish_reason: None,
        }
    }

    /// Attach the response identity metadata this call's attempt reported.
    pub fn with_identity(mut self, identity: ResponseIdentity) -> Self {
        self.message_id = identity.message_id;
        self.response_id = identity.response_id;
        self.provider_request_id = identity.provider_request_id;
        self
    }

    /// Attach the terminal finish reason this call's attempt reported.
    ///
    /// Kept separate from [`Self::with_identity`] because a finish reason is
    /// not identity: [`ResponseIdentity`] answers "which response was this",
    /// while this answers "why did it stop".
    pub fn with_finish_reason(mut self, finish_reason: Option<FinishReason>) -> Self {
        self.finish_reason = finish_reason;
        self
    }

    /// This call's identity metadata as one [`ResponseIdentity`] carrier.
    pub fn identity(&self) -> ResponseIdentity {
        ResponseIdentity {
            message_id: self.message_id.clone(),
            response_id: self.response_id.clone(),
            provider_request_id: self.provider_request_id.clone(),
        }
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
/// ([`PromptRequest`]) and streaming ([`StreamingPromptRequest`]) surfaces so a
/// call site reads identically whether it used `.prompt()` or `.stream_prompt()`.
///
/// On the streaming surface this is the payload of the terminal
/// [`MultiTurnStreamItem::FinalResponse`] item.
///
/// [`StreamingPromptRequest`]: crate::agent::StreamingPromptRequest
/// [`MultiTurnStreamItem::FinalResponse`]: crate::agent::MultiTurnStreamItem::FinalResponse
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub completion_calls: Vec<CompletionCall>,
    /// Accumulated message history for the run (the run's persisted transcript),
    /// unless memory/history bookkeeping was disabled for the request.
    pub messages: Option<Vec<Message>>,
    /// Structured assistant content for the final turn.
    ///
    /// Where [`output`](Self::output) is the concatenated text, this preserves
    /// the individual content parts (text, reasoning, images, …).
    pub content: Vec<AssistantContent>,
    /// Number of synthetic output-tool calls in the turn that finalized this
    /// response. Kept crate-private because it is runner bookkeeping rather
    /// than provider-facing response content.
    #[serde(skip)]
    output_tool_calls: usize,
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
            content: vec![AssistantContent::text(output.clone())],
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
    pub fn with_content(mut self, content: Vec<AssistantContent>) -> Self {
        self.content = content;
        self
    }

    pub(crate) fn with_output_tool_calls(mut self, count: usize) -> Self {
        self.output_tool_calls = count;
        self
    }

    pub(crate) fn output_tool_calls(&self) -> usize {
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
    pub fn content(&self) -> &[AssistantContent] {
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TypedPromptResponse<T> {
    pub output: T,
    pub usage: Usage,
    /// Successfully completed completion requests made by this agent run.
    ///
    /// `usage` remains the aggregate across the whole run. Use the last
    /// entry's usage to inspect the final completion request's prompt/context
    /// length. Zero-valued entry usage means the provider reported no usage
    /// metrics for that request.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub completion_calls: Vec<CompletionCall>,
}

impl<T> TypedPromptResponse<T> {
    pub fn new(output: T, usage: Usage) -> Self {
        Self {
            output,
            usage,
            completion_calls: Vec::new(),
        }
    }

    /// Attach completion call details to this response.
    pub fn with_completion_calls(mut self, completion_calls: Vec<CompletionCall>) -> Self {
        self.completion_calls = completion_calls;
        self
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
    call: ToolCallId,
    provider: Option<ProviderCallId>,
    name: String,
    content: Vec<ToolResultContent>,
) -> UserContent {
    // The *executed* tool's name travels as data on the result: several
    // wires require it on replay (Gemini `functionResponse.name`, Ollama
    // tool messages), and an identifier is not a name.
    UserContent::tool_result_for(call, provider, name, content)
}

/// Shape a canonical real tool output as a tool result without reparsing text.
pub(crate) fn tool_result_output(
    call: ToolCallId,
    provider: Option<ProviderCallId>,
    name: String,
    output: ToolOutput,
) -> UserContent {
    tool_result_with(call, provider, name, output.into_content())
}

/// Shape a **synthetic message** (a hook skip reason, recovery feedback, or a
/// "not executed" notice) as a tool result. Emitted **verbatim as text** and
/// never re-parsed as structured tool output, so a JSON-shaped message is not
/// silently reinterpreted as an image/multimodal result. Used identically by the
/// blocking and streaming drivers so synthetic results match across both.
pub(crate) fn tool_result_message(
    call: ToolCallId,
    provider: Option<ProviderCallId>,
    name: String,
    message: String,
) -> UserContent {
    tool_result_with(call, provider, name, vec![ToolResultContent::text(message)])
}

pub(crate) fn invalid_tool_retry_user_message(
    assistant_content: &[AssistantContent],
    invalid_tool_call_id: &ToolCallId,
    feedback: String,
) -> Option<Message> {
    // Selecting the invalid call by id is correct by construction:
    // `ToolCallId` is unique and non-empty (minted at the provider boundary
    // when the wire issued none), so id-less wires can no longer collapse
    // every peer onto the first match arm.
    let retry_results = assistant_content
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(tool_call) if tool_call.id == *invalid_tool_call_id => {
                Some(tool_result_message(
                    tool_call.id.clone(),
                    tool_call.provider.clone(),
                    tool_call.function.name.clone(),
                    feedback.clone(),
                ))
            }
            AssistantContent::ToolCall(tool_call) => Some(tool_result_message(
                tool_call.id.clone(),
                tool_call.provider.clone(),
                tool_call.function.name.clone(),
                TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER.to_string(),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();

    Some(Message::User {
        content: non_empty(retry_results)?,
    })
}

/// Whether an assistant turn carried nothing the caller should see.
///
/// Two shapes mean the same thing, and both must be recognised:
///
/// - **Zero parts.** A turn that produced no text and no tool call is an
///   empty list — the shape the streaming path produces (its assembler
///   filters empty text deltas out of the canonical order).
/// - **One empty, unannotated text block.** A blocking wire can deliver an
///   assistant message whose only part is an empty text block; it carries
///   nothing, and the agent curates it out of history exactly as it curates
///   a zero-part turn. The annotation guard is load-bearing: an *annotated*
///   empty text block carries data and must not read as empty. Annotation is
///   a plain `is_some()`: [`rig_core::message::AdditionalParams`] is
///   non-empty by construction, so `Some` always carries data, live and
///   restored alike (pinned by
///   `empty_turn_classification_survives_a_serde_round_trip`).
///
/// This runs on turns flowing through the agent loop only. Caller-supplied
/// `chat_history` is never filtered: an empty text block you replay goes to
/// the wire as-is.
pub(crate) fn is_empty_assistant_turn(choice: &[AssistantContent]) -> bool {
    if choice.is_empty() {
        return true;
    }

    choice.len() == 1
        && matches!(
            choice.first(),
            Some(AssistantContent::Text(text))
                if text.text.is_empty() && text.additional_params.is_none()
        )
}

/// Whether a turn delivered **no answer**: no tool call, and no non-empty text
/// block.
///
/// Deliberately *not* [`is_empty_assistant_turn`], which answers a different
/// question — "does this turn belong in history". They diverge on the shapes
/// that are **worth recording yet answer nothing**, of which there are two:
///
/// 1. a turn carrying only [`AssistantContent::Reasoning`] — the reasoning is
///    real content worth replaying, but it is not an answer;
/// 2. a turn carrying only an **empty text block with `additional_params`** —
///    the annotation (citations, encrypted reasoning references, and other
///    provider metadata some wires require on replay) is worth recording, but
///    the caller still receives no text.
///
/// Metadata-only text therefore does **not** count as an answer. That follows
/// from what the caller actually gets: [`assistant_text_from_choice`]
/// concatenates `text.text` alone, so such a turn yields `""` — the annotation
/// is metadata *about* an answer, never the answer itself.
///
/// Reasoning is not an answer. It is the model's scratch work, it is often not
/// even replayable across turns, and a caller asked a question rather than for
/// the thinking. Treating it as output is how a thinking model that burned its
/// whole budget mid-thought used to report success with an empty string
/// (rig#2322): Gemini counts thinking tokens against `maxOutputTokens`, so a
/// truncated thinking turn *typically* carries reasoning and no text — the
/// common case, not a corner one.
///
/// Tool calls count as delivered: they are an answer in progress, and a
/// truncated tool-call turn must still route to execution. So do images —
/// ten providers emit assistant images, and an image *is* the answer for an
/// image-generation turn.
///
/// The match is **exhaustive on purpose**: no `_` arm. Every content variant
/// must be classified explicitly, so adding one to [`AssistantContent`] breaks
/// this build and forces a decision instead of silently inheriting a default.
/// The first version of this predicate had a `_ => false` catch-all and so
/// classified image-only turns as "no answer" — a truncated image-generation
/// turn would have errored despite delivering an image, which matters because
/// image tokens count against the same output budget.
pub(crate) fn turn_delivered_no_answer(choice: &[AssistantContent]) -> bool {
    !choice.iter().any(|content| match content {
        // Real text is an answer; an empty block delivers nothing.
        AssistantContent::Text(text) => !text.text.is_empty(),
        AssistantContent::ToolCall(_) => true,
        AssistantContent::Image(_) => true,
        // The one exclusion: scratch work, not an answer.
        AssistantContent::Reasoning(_) => false,
    })
}

pub(crate) fn assistant_text_from_choice(choice: &[AssistantContent]) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

impl PromptRequest<Extended> {
    async fn send(self) -> Result<PromptResponse, PromptError> {
        self.runner.run().await
    }
}

// ================================================================
// TypedPromptRequest - for structured output with automatic deserialization
// ================================================================

use crate::completion::StructuredOutputError;
use schemars::{JsonSchema, schema_for};
use serde::de::DeserializeOwned;

/// A builder for creating typed prompt requests that return deserialized structured output.
///
/// This struct wraps a standard `PromptRequest` and adds:
/// - Automatic JSON schema generation from the target type `T`
/// - Automatic deserialization of the response into `T`
///
/// The type parameter `S` represents the state of the request (Standard or Extended).
/// Use `.extended_details()` to transition to Extended state for usage tracking.
///
/// # Example
/// ```rust,ignore
/// let forecast: WeatherForecast = agent
///     .prompt_typed("What's the weather in NYC?")
///     .max_turns(3)
///     .await?;
/// ```
pub struct TypedPromptRequest<T, S>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend,
    S: PromptType,
{
    inner: PromptRequest<S>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> TypedPromptRequest<T, Standard>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend,
{
    /// Create a new TypedPromptRequest from an agent.
    ///
    /// This automatically sets the output schema based on the type parameter `T`.
    pub fn from_agent(agent: &Agent, prompt: impl Into<Message>) -> Self {
        let mut inner = PromptRequest::from_agent(agent, prompt);
        // Override the output schema with the schema for T
        inner.runner.config.output_schema = Some(schema_for!(T));
        // Typed prompts deserialize the model's final string, so they pin
        // `Native` structured output to keep the typed API's behavior unchanged
        // across all providers (#1928). Routing the typed path through `Tool`
        // output mode for tool-using agents on non-composing providers is a
        // follow-up; use the untyped `output_schema`/`output_mode` API for
        // tool-composing structured output today.
        inner.runner.config.output_mode = OutputMode::Native;
        Self {
            inner,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, S> TypedPromptRequest<T, S>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend,
    S: PromptType,
{
    /// Enable returning extended details for responses (includes aggregated token usage).
    ///
    /// Note: This changes the type of the response from `.send()` to return a `TypedPromptResponse<T>` struct
    /// instead of just `T`. This is useful for tracking token usage across multiple turns
    /// of conversation.
    pub fn extended_details(self) -> TypedPromptRequest<T, Extended> {
        TypedPromptRequest {
            inner: self.inner.extended_details(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set the total model-call budget, including the initial call and every
    /// retry or continuation. Zero emits no model calls; one permits only the
    /// initial call. Exceeding the budget returns a
    /// [`StructuredOutputError::PromptError`] wrapping a `MaxTurnsError`.
    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.inner = self.inner.max_turns(max_turns);
        self
    }

    /// Append a hook to this request's hook stack (on top of any the agent
    /// already carries).
    pub fn add_hook<H>(mut self, hook: H) -> Self
    where
        H: AgentHook + 'static,
    {
        self.inner = self.inner.add_hook(hook);
        self
    }

    forward_prompt_setters!(inner);
    forward_tool_concurrency!(inner);
}

/// Deserialize a typed structured response from the model's final text.
///
/// Tries a direct parse first (the common path — native and tool-call output is
/// already clean JSON), then falls back to the first balanced JSON value in the
/// text so prose or markdown code fences around the JSON don't break weaker
/// `Prompted`/best-effort output (#1928).
fn deserialize_structured_output<T: DeserializeOwned>(text: &str) -> Result<T, serde_json::Error> {
    let trimmed = text.trim();
    match serde_json::from_str::<T>(trimmed) {
        Ok(value) => Ok(value),
        Err(direct_err) => {
            let Some(start) = trimmed.find(['{', '[']) else {
                return Err(direct_err);
            };
            serde_json::Deserializer::from_str(&trimmed[start..])
                .into_iter::<T>()
                .next()
                .unwrap_or(Err(direct_err))
        }
    }
}

impl<T> TypedPromptRequest<T, Standard>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend,
{
    /// Send the typed prompt request and deserialize the response.
    async fn send(self) -> Result<T, StructuredOutputError> {
        let response = self.inner.send().await.map_err(Box::new)?;

        if response.is_empty() {
            return Err(StructuredOutputError::EmptyResponse);
        }

        let parsed: T = deserialize_structured_output(&response)?;
        Ok(parsed)
    }
}

impl<T> TypedPromptRequest<T, Extended>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend,
{
    /// Send the typed prompt request with extended details and deserialize the response.
    async fn send(self) -> Result<TypedPromptResponse<T>, StructuredOutputError> {
        let response = self.inner.send().await.map_err(Box::new)?;

        if response.output.is_empty() {
            return Err(StructuredOutputError::EmptyResponse);
        }

        let parsed: T = deserialize_structured_output(&response.output)?;
        Ok(TypedPromptResponse::new(parsed, response.usage)
            .with_completion_calls(response.completion_calls))
    }
}

impl<T> IntoFuture for TypedPromptRequest<T, Standard>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend + 'static,
{
    type Output = Result<T, StructuredOutputError>;
    type IntoFuture = WasmBoxedFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.send())
    }
}

impl<T> IntoFuture for TypedPromptRequest<T, Extended>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend + 'static,
{
    type Output = Result<TypedPromptResponse<T>, StructuredOutputError>;
    type IntoFuture = WasmBoxedFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.send())
    }
}
#[cfg(test)]
mod tests {
    use super::ResponseIdentity;
    use super::{
        CompletionCall, PromptResponse, TypedPromptResponse, assistant_text_from_choice,
        is_empty_assistant_turn, turn_delivered_no_answer,
    };
    use crate::{
        agent::{
            AgentBuilder,
            hook::{
                AgentHook, CompletionResponse as CompletionResponseEvent, HookContext,
                InvalidToolCallAction, InvalidToolCallContext, ObservationAction,
                ToolCall as ToolCallEvent, ToolCallAction,
            },
        },
        completion::{
            AssistantContent, CompletionError, CompletionRequest, FinishReason, Message, Prompt,
            PromptError, StructuredOutputError, TypedPrompt, Usage,
        },
        test_utils::{
            AppendFailingMemory, CountingMemory, FailingMemory, MockAddTool, MockCompletionModel,
            MockContextProbeTool, MockOperationArgs, MockSubtractTool, MockToolError, MockTurn,
            SessionId,
        },
        tool::{Tool, ToolContext},
    };
    use rig_core::message::ProviderCallId;
    use rig_core::message::{Text, ToolCall, ToolChoice, ToolFunction, UserContent};
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};
    use serde_json::json;
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicU32, Ordering},
    };

    /// rig#2322 — the **blocking** surface enforces the same truncation
    /// contract as the streamed one.
    ///
    /// The premise of the whole fix is that the two surfaces disagreeing is
    /// what let truncation surface as a blank answer, yet every other guard
    /// test drives `stream_prompt`. Until `MockTurn::with_finish_reason`
    /// existed the blocking mock could not report a reason at all, so
    /// `runner.rs`'s `.with_finish_reason(resp.finish_reason())` — and its
    /// propagation through `model_response` → `record_completion_call` → this
    /// guard — was never exercised. Deleting that one line failed nothing.
    #[tokio::test]
    async fn blocking_prompt_rejects_an_empty_truncated_turn() {
        let agent = AgentBuilder::new(MockCompletionModel::from_turns([MockTurn::from_contents(
            [],
        )
        .with_finish_reason(FinishReason::Length)]))
        .build();

        let err = agent
            .prompt("write a long essay")
            .await
            .expect_err("a content-less truncated turn must not return an empty answer");

        let rendered = format!("{err:?}");
        assert!(
            rendered.contains("Length") && rendered.contains("max_tokens"),
            "the blocking error must name the reason and the remedy: {rendered}"
        );
    }

    /// rig#2322 — blocking counterpart of the reasoning-only case: the shape
    /// that motivated the predicate fix must be caught on both surfaces.
    #[tokio::test]
    async fn blocking_prompt_rejects_a_reasoning_only_truncated_turn() {
        let agent = AgentBuilder::new(MockCompletionModel::from_turns([MockTurn::from_content(
            AssistantContent::Reasoning(rig_core::message::Reasoning::new(
                "thinking, never answering",
            )),
        )
        .with_finish_reason(FinishReason::Length)]))
        .build();

        let err = agent
            .prompt("solve this")
            .await
            .expect_err("a reasoning-only truncated turn must not return an empty answer");

        assert!(format!("{err:?}").contains("Length"));
    }

    /// rig#2322 — and the blocking surface keeps a truncated turn that *did*
    /// answer, with the reason reaching `completion_calls`.
    ///
    /// This is the half that proves the blocking plumbing carries the reason
    /// rather than merely erroring: the value has to survive into the returned
    /// `PromptResponse`.
    #[tokio::test]
    async fn blocking_prompt_keeps_partial_output_and_records_the_reason() {
        let agent = AgentBuilder::new(MockCompletionModel::from_turns([MockTurn::text(
            "a partial ans",
        )
        .with_finish_reason(FinishReason::Length)]))
        .build();

        let response = agent
            .prompt("write a long essay")
            .extended_details()
            .await
            .expect("a truncated turn that produced text must still succeed");

        assert_eq!(response.output, "a partial ans");
        assert_eq!(
            response
                .completion_calls
                .last()
                .and_then(|call| call.finish_reason.clone()),
            Some(FinishReason::Length),
            "the terminal reason must reach the caller on the blocking surface too"
        );
    }

    /// rig#2322 follow-up — every `AssistantContent` variant's answer
    /// classification, pinned one variant at a time.
    ///
    /// A unit test rather than a cassette because this is a pure
    /// classification decision over a rig-owned enum: no provider traffic is
    /// involved, and the failure it guards against (a new variant silently
    /// inheriting the wrong bucket) is invisible at the wire level.
    ///
    /// The predicate originally used a `_ => false` catch-all, which made
    /// image-only turns read as "no answer" — so a truncated image-generation
    /// turn would have errored despite delivering an image. The match is now
    /// exhaustive; this test pins each arm so a reclassification is deliberate.
    #[test]
    fn answer_classification_covers_every_assistant_content_variant() {
        let image = AssistantContent::image_base64(
            "iVBORw0KGgo=",
            Some(rig_core::message::ImageMediaType::PNG),
            Some(rig_core::message::ImageDetail::default()),
        );
        let reasoning = AssistantContent::Reasoning(rig_core::message::Reasoning::new("thinking"));
        let tool_call = AssistantContent::ToolCall(ToolCall::from_wire(
            "call_1".to_string(),
            ToolFunction::new("add".to_string(), json!({})),
        ));

        // Delivered an answer.
        assert!(!turn_delivered_no_answer(&[AssistantContent::text("hi")]));
        assert!(!turn_delivered_no_answer(std::slice::from_ref(&tool_call)));
        assert!(
            !turn_delivered_no_answer(std::slice::from_ref(&image)),
            "an image IS the answer for an image-generation turn; classifying it \
             as 'no answer' makes a truncated image turn error despite delivering one"
        );

        // Delivered nothing.
        assert!(turn_delivered_no_answer(&[]));
        assert!(turn_delivered_no_answer(&[AssistantContent::text("")]));
        assert!(
            turn_delivered_no_answer(std::slice::from_ref(&reasoning)),
            "reasoning is scratch work, not an answer"
        );

        // Mixed: one real item is enough.
        assert!(!turn_delivered_no_answer(&[
            reasoning.clone(),
            AssistantContent::text("answer")
        ]));
        assert!(
            !turn_delivered_no_answer(&[reasoning.clone(), image.clone()]),
            "a thinking image model that produced an image has answered"
        );
        assert!(turn_delivered_no_answer(&[
            reasoning,
            AssistantContent::text("")
        ]));
    }

    /// rig#2322 follow-up — the two predicates diverge on exactly the shapes
    /// that are **worth recording yet answer nothing**, and that is intentional.
    ///
    /// `is_empty_assistant_turn` governs the history push ("does this belong in
    /// the transcript"); `turn_delivered_no_answer` governs the truncation
    /// guard ("did this answer the question"). Two shapes disagree, and for the
    /// same reason: reasoning-only, and an empty text block carrying
    /// `additional_params`. Both are worth recording; neither delivers text.
    ///
    /// An earlier version of this test claimed the divergence was reasoning-only
    /// and checked just three agreeing shapes — none of them annotated — so the
    /// second case went unnoticed. The agreeing set is now enumerated
    /// explicitly alongside it.
    #[test]
    fn the_two_turn_predicates_diverge_on_recordable_but_answerless_turns() {
        let reasoning_only = vec![AssistantContent::Reasoning(
            rig_core::message::Reasoning::new("t"),
        )];
        let annotated_empty_text = vec![AssistantContent::Text(Text {
            text: String::new(),
            additional_params: rig_core::message::AdditionalParams::try_from_value(
                json!({"citations": ["ref"]}),
            )
            .expect("citation params should be a JSON object"),
        })];

        // Divergent: recordable, but no answer was delivered.
        for choice in [&reasoning_only, &annotated_empty_text] {
            assert!(
                !is_empty_assistant_turn(choice),
                "should be recorded in history: {choice:?}"
            );
            assert!(
                turn_delivered_no_answer(choice),
                "should count as no answer — the caller receives no text: {choice:?}"
            );
        }

        // Agreeing: everything else.
        for choice in [
            vec![],
            vec![AssistantContent::text("")],
            vec![AssistantContent::text("real")],
            vec![AssistantContent::image_base64(
                "iVBORw0KGgo=",
                Some(rig_core::message::ImageMediaType::PNG),
                Some(rig_core::message::ImageDetail::default()),
            )],
            vec![AssistantContent::ToolCall(ToolCall::from_wire(
                "call_1".to_string(),
                ToolFunction::new("add".to_string(), json!({})),
            ))],
        ] {
            assert_eq!(
                is_empty_assistant_turn(&choice),
                turn_delivered_no_answer(&choice),
                "unexpected divergence on {choice:?}"
            );
        }
    }

    /// rig#2322 follow-up — metadata-only text is **not** an answer, stated as
    /// its own decision rather than left implied by the predicate's code.
    ///
    /// A text block with `additional_params` and no text carries provider
    /// metadata (citations, encrypted reasoning references) that is worth
    /// keeping in history, but `assistant_text_from_choice` concatenates
    /// `text.text` alone — so the caller receives `""`. Nothing was answered,
    /// and a turn truncated in that state is a failed turn.
    ///
    /// If this is ever reclassified, the run-level consequence is the point to
    /// weigh: it decides whether a truncated annotation-only turn errors or
    /// silently returns an empty string.
    #[test]
    fn metadata_only_text_is_not_an_answer() {
        let annotated_empty = AssistantContent::Text(Text {
            text: String::new(),
            additional_params: rig_core::message::AdditionalParams::try_from_value(
                json!({"citations": ["ref"]}),
            )
            .expect("citation params should be a JSON object"),
        });

        assert!(turn_delivered_no_answer(std::slice::from_ref(
            &annotated_empty
        )));
        assert_eq!(
            assistant_text_from_choice(std::slice::from_ref(&annotated_empty)),
            "",
            "the caller receives no text, which is why this is not an answer"
        );

        // The annotation does not suppress a real answer beside it.
        assert!(!turn_delivered_no_answer(&[
            annotated_empty,
            AssistantContent::text("real"),
        ]));
    }

    #[derive(Serialize)]
    struct SerializeOnly {
        value: &'static str,
    }

    #[derive(Deserialize)]
    struct DeserializeOnly {
        value: String,
    }

    #[derive(Debug, Deserialize, JsonSchema, PartialEq)]
    struct TypedAnswer {
        value: String,
    }

    #[test]
    fn deserialize_structured_output_tolerates_fences_and_prose() {
        // Clean JSON (native / output-tool path).
        assert_eq!(
            super::deserialize_structured_output::<TypedAnswer>(r#"{"value":"x"}"#).unwrap(),
            TypedAnswer { value: "x".into() }
        );
        // Markdown-fenced JSON (weak Prompted-mode models).
        assert_eq!(
            super::deserialize_structured_output::<TypedAnswer>("```json\n{\"value\":\"y\"}\n```")
                .unwrap(),
            TypedAnswer { value: "y".into() }
        );
        // Prose around the JSON object.
        assert_eq!(
            super::deserialize_structured_output::<TypedAnswer>(
                "Here you go: {\"value\":\"z\"} — hope that helps!"
            )
            .unwrap(),
            TypedAnswer { value: "z".into() }
        );
        // No JSON at all still errors.
        assert!(super::deserialize_structured_output::<TypedAnswer>("no json here").is_err());
    }

    #[derive(Clone)]
    struct PanicOnUnknownToolHook;

    impl AgentHook for PanicOnUnknownToolHook {
        async fn on_completion_response(
            &self,
            _ctx: &HookContext,
            _event: CompletionResponseEvent<'_>,
        ) -> ObservationAction {
            panic!("unknown tool response should fail before response hooks run")
        }
        async fn on_tool_call(
            &self,
            _ctx: &HookContext,
            _event: ToolCallEvent<'_>,
        ) -> ToolCallAction {
            panic!("unknown tool call should fail before tool hooks run")
        }
    }

    #[derive(Clone)]
    struct PanicOnToolCallHook;

    impl AgentHook for PanicOnToolCallHook {
        async fn on_tool_call(
            &self,
            _ctx: &HookContext,
            _event: ToolCallEvent<'_>,
        ) -> ToolCallAction {
            panic!("recovered invalid turn should not invoke normal tool hooks")
        }
    }

    #[derive(Clone)]
    struct SkipDefaultApiAndPanicOnToolCallHook;

    impl AgentHook for SkipDefaultApiAndPanicOnToolCallHook {
        async fn on_invalid_tool_call(
            &self,
            ctx: &HookContext,
            event: &InvalidToolCallContext,
        ) -> Option<InvalidToolCallAction> {
            SkipDefaultApiHook.on_invalid_tool_call(ctx, event).await
        }
        async fn on_tool_call(
            &self,
            ctx: &HookContext,
            event: ToolCallEvent<'_>,
        ) -> ToolCallAction {
            PanicOnToolCallHook.on_tool_call(ctx, event).await
        }
    }

    #[derive(Clone)]
    struct RepairDefaultApiHook;

    impl AgentHook for RepairDefaultApiHook {
        async fn on_invalid_tool_call(
            &self,
            _ctx: &HookContext,
            event: &InvalidToolCallContext,
        ) -> Option<InvalidToolCallAction> {
            assert_eq!(event.tool_name, "default_api");
            Some(InvalidToolCallAction::repair("add"))
        }
    }

    #[derive(Clone)]
    struct RepairToSubtractHook;

    impl AgentHook for RepairToSubtractHook {
        async fn on_invalid_tool_call(
            &self,
            _ctx: &HookContext,
            _event: &InvalidToolCallContext,
        ) -> Option<InvalidToolCallAction> {
            Some(InvalidToolCallAction::repair("subtract"))
        }
    }

    #[derive(Clone)]
    struct RetryDefaultApiHook;

    impl AgentHook for RetryDefaultApiHook {
        async fn on_invalid_tool_call(
            &self,
            _ctx: &HookContext,
            event: &InvalidToolCallContext,
        ) -> Option<InvalidToolCallAction> {
            Some(InvalidToolCallAction::retry(format!(
                "Use one of these tools instead: {:?}",
                event.allowed_tools
            )))
        }
    }

    #[derive(Clone)]
    struct SkipDefaultApiHook;

    impl AgentHook for SkipDefaultApiHook {
        async fn on_invalid_tool_call(
            &self,
            _ctx: &HookContext,
            _event: &InvalidToolCallContext,
        ) -> Option<InvalidToolCallAction> {
            Some(InvalidToolCallAction::skip("default_api is not available"))
        }
    }

    #[derive(Clone, Default)]
    struct RecordingInvalidToolCallHook {
        contexts: Arc<Mutex<Vec<InvalidToolCallContext>>>,
    }

    impl RecordingInvalidToolCallHook {
        fn observed(&self) -> Vec<InvalidToolCallContext> {
            self.contexts
                .lock()
                .expect("invalid tool context records mutex was poisoned")
                .clone()
        }
    }

    impl AgentHook for RecordingInvalidToolCallHook {
        async fn on_invalid_tool_call(
            &self,
            _ctx: &HookContext,
            event: &InvalidToolCallContext,
        ) -> Option<InvalidToolCallAction> {
            self.contexts
                .lock()
                .expect("invalid tool context records mutex was poisoned")
                .push(event.clone());
            None
        }
    }

    #[derive(Clone)]
    struct CountingAddTool {
        calls: Arc<AtomicU32>,
    }

    impl Tool for CountingAddTool {
        const NAME: &'static str = "add";
        type Error = MockToolError;
        type Args = MockOperationArgs;
        type Output = i32;

        fn description(&self) -> String {
            MockAddTool.description()
        }

        fn parameters(&self) -> serde_json::Value {
            MockAddTool.parameters()
        }

        async fn call(
            &self,
            _context: &mut crate::tool::ToolContext,
            _args: Self::Args,
        ) -> Result<Self::Output, Self::Error> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(0)
        }
    }

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
    fn typed_prompt_response_serializes_with_serialize_only_output() {
        let response = TypedPromptResponse::new(
            SerializeOnly { value: "ok" },
            Usage {
                input_tokens: 1,
                output_tokens: 2,
                total_tokens: 3,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
                tool_use_prompt_tokens: 0,
                reasoning_tokens: 0,
            },
        );

        let json = serde_json::to_string(&response).expect("serialize typed prompt response");
        assert!(json.contains("\"value\":\"ok\""));
    }

    #[test]
    fn typed_prompt_response_deserializes_with_deserialize_only_output() {
        let response: TypedPromptResponse<DeserializeOnly> = serde_json::from_str(
            r#"{"output":{"value":"ok"},"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3,"cached_input_tokens":0,"cache_creation_input_tokens":0,"reasoning_tokens":0}}"#,
        )
        .expect("deserialize typed prompt response");

        assert_eq!(response.requests(), 0);
        assert_eq!(response.output.value, "ok");
        assert_eq!(response.usage.input_tokens, 1);
        assert_eq!(response.usage.output_tokens, 2);
        assert_eq!(response.usage.total_tokens, 3);
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
    fn empty_turn_classification_survives_a_serde_round_trip() {
        // A suspended run restored from JSON must classify its empty-text
        // turn exactly like the live run did, whatever spelling of "no
        // extras" the JSON carries, and an *annotated* empty block must
        // still read as content either way. The serde canonicalization
        // mechanics behind this (`{}`/`null` decode to `None`, empty params
        // never serialize) are pinned where they live, by rig-core's
        // `empty_params_canonicalize_to_none_in_both_serde_directions` —
        // this test asserts classification only.
        let live = vec![AssistantContent::text("")];
        assert!(is_empty_assistant_turn(&live));

        let round: Vec<AssistantContent> =
            serde_json::from_str(&serde_json::to_string(&live).expect("serialize"))
                .expect("deserialize");
        assert!(
            is_empty_assistant_turn(&round),
            "restored turn must classify like the live one: {round:?}"
        );

        // An explicit `{}` or `null` in the JSON — the shape a mechanical
        // migration script writes — classifies exactly like an absent field.
        for empty_spelling in [serde_json::json!({}), serde_json::Value::Null] {
            let migrated: Vec<AssistantContent> = serde_json::from_value(serde_json::json!([
                {"type": "text", "text": "", "additional_params": empty_spelling}
            ]))
            .expect("deserialize migrated");
            assert!(is_empty_assistant_turn(&migrated));
        }

        // The old uncanonicalized-`Some({})` hazard is unrepresentable:
        // `AdditionalParams` has no empty value, so the only way to spell
        // "no extras" in memory is `None` and live/restored classification
        // agree by construction.
        let canonical_absent = vec![AssistantContent::Text(rig_core::message::Text {
            text: String::new(),
            additional_params: rig_core::message::AdditionalParams::try_from_value(
                serde_json::json!({}),
            )
            .expect("object params"),
        })];
        assert!(is_empty_assistant_turn(&canonical_absent));
        let restored: Vec<AssistantContent> =
            serde_json::from_value(serde_json::to_value(&canonical_absent).expect("serialize"))
                .expect("deserialize");
        assert!(is_empty_assistant_turn(&restored));

        let annotated: Vec<AssistantContent> = serde_json::from_value(serde_json::json!([
            {"type": "text", "text": "", "additional_params": {"signature": "sig"}}
        ]))
        .expect("deserialize annotated");
        assert!(
            !is_empty_assistant_turn(&annotated),
            "an annotated empty block carries data: {annotated:?}"
        );
    }

    #[test]
    fn prompt_response_deserializes_pre_monoid_null_usage_format() {
        // Pins `CompletionCall.usage`'s null tolerance: `"usage": null` (the
        // pre-monoid Option encoding) must map to zero-valued usage. The
        // fixture otherwise uses the current shape — `content` is a required
        // field since the missing-`content` reconstruction was dropped.
        let fixture = r#"{"output":"ok","usage":{"input_tokens":3,"output_tokens":4,"total_tokens":7,"cached_input_tokens":0,"cache_creation_input_tokens":0,"tool_use_prompt_tokens":0,"reasoning_tokens":0},"completion_calls":[{"call_index":0,"usage":null},{"call_index":1,"usage":{"input_tokens":3,"output_tokens":4,"total_tokens":7,"cached_input_tokens":0,"cache_creation_input_tokens":0,"tool_use_prompt_tokens":0,"reasoning_tokens":0}}],"messages":[{"role":"user","content":[{"type":"text","text":"add things"}]}],"content":[{"type":"text","text":"ok"}]}"#;

        let response: PromptResponse =
            serde_json::from_str(fixture).expect("old-format response should deserialize");
        assert_eq!(
            response.completion_calls(),
            &[
                CompletionCall::new(0, Usage::new()),
                CompletionCall::new(1, usage(3, 4))
            ]
        );
        // `content` uses the tagged shape — assistant content is tagged like
        // user content (see `the_type_key_is_the_tag_and_the_untagged_shape_does_not_load`).
        let [AssistantContent::Text(text)] = response.content() else {
            panic!("expected one text block, got {:?}", response.content());
        };
        assert_eq!(text.text, "ok");
    }

    #[test]
    fn the_type_key_is_the_tag_and_the_untagged_shape_does_not_load() {
        // Assistant content is tagged like user content: `"type"` is consumed
        // as the discriminant, never captured into `additional_params`. And
        // there is deliberately no untagged fallback — the bare shape 0.41
        // serialized fails to deserialize (MIGRATING carries the recipe),
        // pinned here so removing the tag requirement is a visible decision,
        // not an accident.
        let tagged: Vec<AssistantContent> =
            serde_json::from_value(serde_json::json!([{"type": "text", "text": "ok"}]))
                .expect("deserialize");
        let [AssistantContent::Text(text)] = tagged.as_slice() else {
            panic!("expected one text block, got {tagged:?}");
        };
        assert_eq!(text.text, "ok");
        assert_eq!(text.additional_params, None, "the tag is not data");

        serde_json::from_value::<Vec<AssistantContent>>(serde_json::json!([{"text": "ok"}]))
            .expect_err("the untagged shape must not deserialize");
    }

    #[test]
    fn prompt_response_roundtrip_preserves_explicit_content() {
        // An explicitly-set `content` (e.g. the streaming surface's structured
        // final turn) must survive a serialize/deserialize round-trip intact —
        // `content` and `output` are independent fields.
        let response = PromptResponse::new("visible text", Usage::new())
            .with_content(vec![AssistantContent::text("structured")]);

        let value = serde_json::to_value(&response).expect("serialize prompt response");
        assert!(
            value.get("content").is_some(),
            "content is part of the serialized shape"
        );

        let round: PromptResponse =
            serde_json::from_value(value).expect("deserialize prompt response");
        assert_eq!(round.output(), "visible text");
        // The stored content is "structured" — distinct from `output` — so the
        // round trip demonstrably carried `content` itself rather than anything
        // derived from `output`. (Compare the text directly to sidestep the
        // `Text::additional_params` serde round-trip asymmetry.)
        let Some(AssistantContent::Text(text)) = round.content().first() else {
            panic!("expected text content, got {:?}", round.content().first());
        };
        assert_eq!(text.text, "structured");
    }

    #[test]
    fn prompt_response_serialize_and_deserialize_agree_on_wire_shape() {
        // `content` is a required, bare list in both serde directions — the
        // pre-`content` reconstruction (and the shadow repr that carried it)
        // is gone, so serialize and deserialize agree by construction. Pin
        // the shape: `content` present, `completion_calls` omitted only when
        // empty, and the value round-trips.
        let response = PromptResponse::new("hi", usage(1, 2))
            .with_completion_calls(vec![CompletionCall::new(0, usage(1, 2))]);

        let from_response = serde_json::to_value(&response).expect("serialize response");
        assert!(from_response.get("content").is_some());
        assert!(from_response.get("completion_calls").is_some());

        let round: PromptResponse =
            serde_json::from_value(from_response).expect("deserialize response");
        assert_eq!(round.output(), "hi");
        assert_eq!(round.usage(), usage(1, 2));
        assert_eq!(
            round.completion_calls(),
            &[CompletionCall::new(0, usage(1, 2))]
        );

        // The omission direction of `completion_calls`' skip-when-empty:
        // an empty list serializes without the key (the shadow-era wire
        // shape), and the keyless JSON still deserializes.
        let bare = serde_json::to_value(PromptResponse::new("hi", usage(1, 2)))
            .expect("serialize bare response");
        assert!(bare.get("completion_calls").is_none());
        let round: PromptResponse =
            serde_json::from_value(bare).expect("deserialize keyless response");
        assert!(round.completion_calls().is_empty());
    }

    #[tokio::test]
    async fn prompt_response_records_completion_call_without_reported_usage() {
        let model = MockCompletionModel::new([MockTurn::text("ok")]);
        let agent = AgentBuilder::new(model).build();

        let response = agent
            .prompt("say ok")
            .extended_details()
            .await
            .expect("prompt should succeed");

        assert_eq!(response.output, "ok");
        assert_eq!(response.usage, Usage::new());
        assert_eq!(
            response.completion_calls(),
            &[CompletionCall::new(0, Usage::new())]
        );
    }

    #[tokio::test]
    async fn typed_prompt_response_preserves_completion_calls() {
        let call_usage = Usage {
            input_tokens: 4,
            output_tokens: 6,
            total_tokens: 10,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
            tool_use_prompt_tokens: 0,
            reasoning_tokens: 0,
        };
        let model =
            MockCompletionModel::new([MockTurn::text(r#"{"value":"ok"}"#).with_usage(call_usage)]);
        let agent = AgentBuilder::new(model).build();

        let response = agent
            .prompt_typed::<TypedAnswer>("return typed json")
            .extended_details()
            .await
            .expect("typed prompt should succeed");

        assert_eq!(
            response.output,
            TypedAnswer {
                value: "ok".to_string()
            }
        );
        assert_eq!(response.usage, call_usage);
        assert_eq!(
            response.completion_calls(),
            &[CompletionCall::new(0, call_usage)]
        );
    }

    fn validate_follow_up_tool_history(request: &CompletionRequest) {
        let history = request.chat_history.clone();
        assert_eq!(
            history.len(),
            3,
            "follow-up request should contain the prompt, assistant tool call, and user tool result: {history:?}"
        );

        assert!(matches!(
            history.first(),
            Some(Message::User { content })
                if matches!(
                    content.first(),
                    Some(UserContent::Text(text)) if text.text == "do tool work"
                )
        ));

        // The wire issued "tool_call_1" (adopted as rig's durable id) and the
        // provider-specific correlator was overridden to "call_1"; the result
        // answers the durable id and echoes the provider correlator.
        assert!(matches!(
            history.get(1),
            Some(Message::Assistant { content, .. })
                if matches!(
                    content.first(),
                    Some(AssistantContent::ToolCall(tool_call))
                        if tool_call.id == "tool_call_1"
                            && tool_call.provider.as_ref().is_some_and(
                                |provider| provider.call_id == "call_1"
                            )
                )
        ));

        assert!(matches!(
            history.get(2),
            Some(Message::User { content })
                if matches!(
                    content.first(),
                    Some(UserContent::ToolResult(tool_result))
                        if tool_result.call == "tool_call_1"
                            && tool_result.provider.as_ref().is_some_and(
                                |provider| provider.call_id == "call_1"
                            )
                )
        ));
    }

    fn history_contains_tool_call(history: &[Message], tool_name: &str) -> bool {
        history.iter().any(|message| {
            matches!(
                message,
                Message::Assistant { content, .. }
                    if content.iter().any(|item| matches!(
                        item,
                        AssistantContent::ToolCall(tool_call)
                            if tool_call.function.name == tool_name
                    ))
            )
        })
    }

    /// The invalid-call retry transcript pairs 1:1 by construction: every tool
    /// call in the assistant turn carries a unique non-empty id (minted at the
    /// provider boundary when the wire issued none), and the retry results
    /// answer exactly those ids.
    fn assert_retry_transcript_ids_pair(assistant: &Message, results: &Message) {
        use std::collections::BTreeSet;

        let Message::Assistant { content, .. } = assistant else {
            panic!("expected the assistant tool-call turn, got {assistant:?}");
        };
        let call_ids: Vec<&str> = content
            .iter()
            .filter_map(|item| match item {
                AssistantContent::ToolCall(tool_call) => Some(tool_call.id.as_str()),
                _ => None,
            })
            .collect();
        let Message::User { content } = results else {
            panic!("expected the user retry-result turn, got {results:?}");
        };
        let result_ids: Vec<&str> = content
            .iter()
            .filter_map(|item| match item {
                UserContent::ToolResult(result) => Some(result.call.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            call_ids.iter().all(|id| !id.is_empty()),
            "every tool call carries a non-empty id: {call_ids:?}"
        );
        let unique_calls: BTreeSet<&str> = call_ids.iter().copied().collect();
        assert_eq!(
            unique_calls.len(),
            call_ids.len(),
            "tool-call ids must be unique: {call_ids:?}"
        );
        let unique_results: BTreeSet<&str> = result_ids.iter().copied().collect();
        assert_eq!(
            unique_results.len(),
            result_ids.len(),
            "retry-result ids must be unique: {result_ids:?}"
        );
        assert_eq!(
            unique_calls, unique_results,
            "retry results must answer exactly the turn's tool calls"
        );
    }

    #[tokio::test]
    async fn unknown_tool_call_fails_before_non_streaming_second_request() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 1, "y": 2})),
            MockTurn::text("should not be requested"),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let err = agent
            .prompt("use the tool")
            .add_hook(PanicOnUnknownToolHook)
            .max_turns(3)
            .await
            .expect_err("unknown model-emitted tool should fail");

        match err {
            PromptError::UnknownToolCall {
                tool_name,
                available_tools,
                allowed_tools,
                chat_history,
            } => {
                assert_eq!(tool_name, "default_api");
                assert_eq!(available_tools, vec!["add".to_string()]);
                assert_eq!(allowed_tools, vec!["add".to_string()]);
                assert!(history_contains_tool_call(&chat_history, "default_api"));
            }
            other => panic!("expected UnknownToolCall, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    /// The motivating use-case: a `ToolContext` set on the prompt request is
    /// threaded all the way to the tool the agent loop executes.
    #[tokio::test]
    async fn tool_context_reaches_tool_through_agent_loop() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "context_probe", json!({})),
            MockTurn::text("done"),
        ]);
        let probe = MockContextProbeTool::default();
        let agent = AgentBuilder::new(model).tool(probe.clone()).build();

        let mut context = ToolContext::new();
        context.insert(SessionId("abc-123".to_string()));

        let out = agent
            .prompt("use the tool")
            .tool_context(context)
            .max_turns(3)
            .await
            .expect("run succeeds");

        assert_eq!(out, "done");
        assert_eq!(probe.observed().as_deref(), Some("session:abc-123"));
    }

    /// Context values persist for the whole run, across *multiple* tool-call rounds
    /// (the headline value prop). The model calls the probe in two consecutive
    /// rounds; both must observe the same injected value, not just the first.
    #[tokio::test]
    async fn tool_context_persists_across_multiple_rounds() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("c1", "context_probe", json!({})),
            MockTurn::tool_call("c2", "context_probe", json!({})),
            MockTurn::text("done"),
        ]);
        let probe = MockContextProbeTool::default();
        let agent = AgentBuilder::new(model).tool(probe.clone()).build();

        let mut context = ToolContext::new();
        context.insert(SessionId("abc-123".to_string()));

        let out = agent
            .prompt("use the tool twice")
            .tool_context(context)
            .max_turns(5)
            .await
            .expect("run succeeds");

        assert_eq!(out, "done");
        assert_eq!(
            probe.observations(),
            vec!["session:abc-123".to_string(), "session:abc-123".to_string()],
        );
    }

    /// Without a context, the same tool runs with an empty one (no panic, no
    /// stale value) — the backward-compatible default path.
    #[tokio::test]
    async fn tool_runs_with_empty_context_when_none_supplied() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "context_probe", json!({})),
            MockTurn::text("done"),
        ]);
        let probe = MockContextProbeTool::default();
        let agent = AgentBuilder::new(model).tool(probe.clone()).build();

        let out = agent
            .prompt("use the tool")
            .max_turns(3)
            .await
            .expect("run succeeds");

        assert_eq!(out, "done");
        // The single call path receives an empty context and observes no session.
        assert_eq!(probe.observed().as_deref(), Some("no-session"));
    }

    /// Direct typed calls use the same context contract as dispatched calls.
    #[tokio::test]
    async fn probe_direct_call_uses_context() {
        let probe = MockContextProbeTool::default();
        let out = probe
            .call(&mut ToolContext::new(), json!({}))
            .await
            .expect("call succeeds");
        assert_eq!(out, "no-session");
        assert_eq!(probe.observed().as_deref(), Some("no-session"));
    }

    #[tokio::test]
    async fn invalid_tool_call_context_uses_completed_tool_call_provider_id() {
        let invalid_hook = RecordingInvalidToolCallHook::default();
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 1, "y": 2}))
                .with_call_id("provider_call_1"),
            MockTurn::text("should not be requested"),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let err = agent
            .prompt("use the tool")
            .add_hook(invalid_hook.clone())
            .max_turns(3)
            .await
            .expect_err("invalid tool should fail");

        assert!(matches!(err, PromptError::UnknownToolCall { .. }));
        assert_eq!(recorded.request_count(), 1);
        let contexts = invalid_hook.observed();
        assert_eq!(contexts.len(), 1);
        let context = &contexts[0];
        assert_eq!(context.tool_name, "default_api");
        assert_eq!(context.tool_call_id.as_deref(), Some("tool_call_1"));
        assert_eq!(context.internal_call_id, None);
        assert!(!context.is_streaming);
    }

    #[tokio::test]
    async fn disallowed_specific_tool_call_fails_before_non_streaming_second_request() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "subtract", json!({"x": 3, "y": 1})),
            MockTurn::text("should not be requested"),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model)
            .tool(MockAddTool)
            .tool(MockSubtractTool)
            .tool_choice(ToolChoice::Specific {
                function_names: vec!["add".to_string()],
            })
            .build();

        let err = agent
            .prompt("use the allowed tool")
            .add_hook(PanicOnUnknownToolHook)
            .max_turns(3)
            .await
            .expect_err("disallowed model-emitted tool should fail");

        match err {
            PromptError::UnknownToolCall {
                tool_name,
                available_tools,
                allowed_tools,
                chat_history,
            } => {
                assert_eq!(tool_name, "subtract");
                assert_eq!(
                    available_tools,
                    vec!["add".to_string(), "subtract".to_string()]
                );
                assert_eq!(allowed_tools, vec!["add".to_string()]);
                assert!(history_contains_tool_call(&chat_history, "subtract"));
            }
            other => panic!("expected UnknownToolCall, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn tool_choice_none_rejects_non_streaming_tool_call() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "add", json!({"x": 1, "y": 2})),
            MockTurn::text("should not be requested"),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model)
            .tool(MockAddTool)
            .tool_choice(ToolChoice::None)
            .build();

        let err = agent
            .prompt("do not use tools")
            .add_hook(PanicOnUnknownToolHook)
            .max_turns(3)
            .await
            .expect_err("ToolChoice::None should reject returned tool calls");

        match err {
            PromptError::UnknownToolCall {
                tool_name,
                available_tools,
                allowed_tools,
                chat_history,
            } => {
                assert_eq!(tool_name, "add");
                assert_eq!(available_tools, vec!["add".to_string()]);
                assert!(allowed_tools.is_empty());
                assert!(history_contains_tool_call(&chat_history, "add"));
            }
            other => panic!("expected UnknownToolCall, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn invalid_tool_call_hook_can_repair_non_streaming_tool_name() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text("done"),
        ]);
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let response = agent
            .prompt("add")
            .add_hook(RepairDefaultApiHook)
            .max_turns(3)
            .extended_details()
            .await
            .expect("repaired tool call should execute");

        assert_eq!(response.output, "done");
        let messages = response.messages.expect("messages should be present");
        assert!(history_contains_tool_call(&messages, "add"));
        assert!(!history_contains_tool_call(&messages, "default_api"));
        assert!(messages.iter().any(|message| {
            matches!(
                message,
                Message::User { content }
                    if content.iter().any(|content| {
                        matches!(
                            content,
                            UserContent::ToolResult(result)
                                if result.content.iter().any(|content| {
                                    matches!(
                                        content,
                                        rig_core::message::ToolResultContent::Json { value }
                                            if value == &serde_json::json!(5)
                                    )
                                })
                        )
                    })
            )
        }));
    }

    #[tokio::test]
    async fn invalid_tool_call_hook_retry_adds_feedback_and_retries_non_streaming() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text("retried"),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let response = agent
            .prompt("add")
            .add_hook(RetryDefaultApiHook)
            .max_invalid_tool_call_retries(1)
            .max_turns(3)
            .extended_details()
            .await
            .expect("retry should recover");

        assert_eq!(response.output, "retried");
        assert_eq!(recorded.request_count(), 2);
        let messages = response.messages.expect("messages should be present");
        assert!(messages.iter().any(|message| {
            matches!(
                message,
                Message::User { content }
                    if content.iter().any(|content| {
                        matches!(
                            content,
                            UserContent::ToolResult(result)
                                if result.content.iter().any(|content| {
                                    matches!(
                                        content,
                                        rig_core::message::ToolResultContent::Text(text)
                                            if text.text.contains("Use one of these tools instead")
                                    )
                                })
                        )
                    })
            )
        }));
    }

    #[tokio::test]
    async fn invalid_tool_call_hook_retries_mixed_non_streaming_turn_without_executing_valid_call()
    {
        let add_calls = Arc::new(AtomicU32::new(0));
        let valid_tool_call = ToolCall::from_wire(
            "tool_call_1",
            ToolFunction::new("add".to_string(), json!({"x": 2, "y": 3})),
        )
        .with_provider(ProviderCallId::new("call_1").expect("non-empty provider id"));
        let invalid_tool_call = ToolCall::from_wire(
            "tool_call_2",
            ToolFunction::new("default_api".to_string(), json!({"x": 4, "y": 5})),
        )
        .with_provider(ProviderCallId::new("call_2").expect("non-empty provider id"));
        let model = MockCompletionModel::new([
            MockTurn::from_contents([
                AssistantContent::ToolCall(valid_tool_call),
                AssistantContent::ToolCall(invalid_tool_call),
            ]),
            MockTurn::text("retried"),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model)
            .tool(CountingAddTool {
                calls: add_calls.clone(),
            })
            .build();

        let response = agent
            .prompt("add")
            .add_hook(RetryDefaultApiHook)
            .max_invalid_tool_call_retries(1)
            .max_turns(3)
            .extended_details()
            .await
            .expect("retry should recover");

        assert_eq!(response.output, "retried");
        assert_eq!(add_calls.load(Ordering::SeqCst), 0);
        let requests = recorded.requests();
        assert_eq!(requests.len(), 2);
        let retry_history = requests[1].chat_history.clone();
        assert_eq!(retry_history.len(), 3);
        assert!(matches!(
            retry_history.get(1),
            Some(Message::Assistant { content, .. })
                if content.iter().any(|item| matches!(
                    item,
                    AssistantContent::ToolCall(tool_call)
                        if tool_call.id == "tool_call_1"
                            && tool_call.function.name == "add"
                ))
                    && content.iter().any(|item| matches!(
                        item,
                        AssistantContent::ToolCall(tool_call)
                            if tool_call.id == "tool_call_2"
                                && tool_call.function.name == "default_api"
                    ))
        ));
        assert!(matches!(
            retry_history.get(2),
            Some(Message::User { content })
                if content.iter().filter(|item| matches!(item, UserContent::ToolResult(_))).count() == 2
                    && content.iter().any(|item| matches!(
                        item,
                        UserContent::ToolResult(result)
                            if result.call == "tool_call_1"
                                && result.provider.as_ref().is_some_and(
                                    |provider| provider.call_id == "call_1"
                                )
                                && result.content.iter().any(|content| matches!(
                                    content,
                                    rig_core::message::ToolResultContent::Text(text)
                                        if text.text == super::TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER
                                ))
                    ))
                    && content.iter().any(|item| matches!(
                        item,
                        UserContent::ToolResult(result)
                            if result.call == "tool_call_2"
                                && result.provider.as_ref().is_some_and(
                                    |provider| provider.call_id == "call_2"
                                )
                                && result.content.iter().any(|content| matches!(
                                    content,
                                    rig_core::message::ToolResultContent::Text(text)
                                        if text.text.contains("Use one of these tools instead")
                                ))
            ))
        ));
        assert_retry_transcript_ids_pair(
            retry_history.get(1).expect("assistant tool-call turn"),
            retry_history.get(2).expect("retry-result turn"),
        );
    }

    #[tokio::test]
    async fn invalid_tool_call_hook_skips_mixed_non_streaming_turn_without_executing_valid_call() {
        let add_calls = Arc::new(AtomicU32::new(0));
        let valid_tool_call = ToolCall::from_wire(
            "tool_call_1",
            ToolFunction::new("add".to_string(), json!({"x": 2, "y": 3})),
        )
        .with_provider(ProviderCallId::new("call_1").expect("non-empty provider id"));
        let invalid_tool_call = ToolCall::from_wire(
            "tool_call_2",
            ToolFunction::new("default_api".to_string(), json!({"x": 4, "y": 5})),
        )
        .with_provider(ProviderCallId::new("call_2").expect("non-empty provider id"));
        let model = MockCompletionModel::new([
            MockTurn::from_contents([
                AssistantContent::ToolCall(valid_tool_call),
                AssistantContent::ToolCall(invalid_tool_call),
            ]),
            MockTurn::text("skipped"),
        ]);
        let agent = AgentBuilder::new(model)
            .tool(CountingAddTool {
                calls: add_calls.clone(),
            })
            .build();

        let response = agent
            .prompt("add")
            .add_hook(SkipDefaultApiAndPanicOnToolCallHook)
            .max_turns(3)
            .extended_details()
            .await
            .expect("skip should recover without executing peer tools");

        assert_eq!(response.output, "skipped");
        assert_eq!(add_calls.load(Ordering::SeqCst), 0);
        let messages = response.messages.expect("messages should be present");
        assert!(history_contains_tool_call(&messages, "add"));
        assert!(history_contains_tool_call(&messages, "default_api"));
        assert!(matches!(
            messages.get(2),
            Some(Message::User { content })
                if content.iter().filter(|item| matches!(item, UserContent::ToolResult(_))).count() == 2
                    && content.iter().any(|item| matches!(
                        item,
                        UserContent::ToolResult(result)
                            if result.call == "tool_call_1"
                                && result.provider.as_ref().is_some_and(
                                    |provider| provider.call_id == "call_1"
                                )
                                && result.content.iter().any(|content| matches!(
                                    content,
                                    rig_core::message::ToolResultContent::Text(text)
                                        if text.text == super::TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER
                                ))
                    ))
                    && content.iter().any(|item| matches!(
                        item,
                        UserContent::ToolResult(result)
                            if result.call == "tool_call_2"
                                && result.provider.as_ref().is_some_and(
                                    |provider| provider.call_id == "call_2"
                                )
                                && result.content.iter().any(|content| matches!(
                                    content,
                                    rig_core::message::ToolResultContent::Text(text)
                                        if text.text == "default_api is not available"
                                ))
                    ))
        ));
        assert_retry_transcript_ids_pair(
            messages.get(1).expect("assistant tool-call turn"),
            messages.get(2).expect("skip-result turn"),
        );
    }

    #[tokio::test]
    async fn invalid_tool_call_hook_retry_budget_exhaustion_fails() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text("should not be requested"),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let err = agent
            .prompt("add")
            .add_hook(RetryDefaultApiHook)
            .max_invalid_tool_call_retries(0)
            .max_turns(3)
            .await
            .expect_err("retry without budget should fail");

        match err {
            PromptError::UnknownToolCall {
                tool_name,
                chat_history,
                ..
            } => {
                assert_eq!(tool_name, "default_api");
                assert!(history_contains_tool_call(&chat_history, "default_api"));
            }
            other => panic!("expected UnknownToolCall, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn invalid_tool_call_hook_can_skip_structured_non_streaming_call() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text("skipped"),
        ]);
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let response = agent
            .prompt("add")
            .add_hook(SkipDefaultApiHook)
            .max_turns(3)
            .extended_details()
            .await
            .expect("skip should continue with synthetic tool result");

        assert_eq!(response.output, "skipped");
        let messages = response.messages.expect("messages should be present");
        assert!(history_contains_tool_call(&messages, "default_api"));
        assert!(messages.iter().any(|message| {
            matches!(
                message,
                Message::User { content }
                    if content.iter().any(|content| {
                        matches!(
                            content,
                            UserContent::ToolResult(result)
                                if result.content.iter().any(|content| {
                                    matches!(
                                        content,
                                        rig_core::message::ToolResultContent::Text(text)
                                            if text.text == "default_api is not available"
                                    )
                                })
                        )
                    })
            )
        }));
    }

    #[tokio::test]
    async fn skip_under_specific_tool_choice_returns_synthetic_feedback() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text("skipped"),
        ]);
        let agent = AgentBuilder::new(model)
            .tool(MockAddTool)
            .tool_choice(ToolChoice::Specific {
                function_names: vec!["add".to_string()],
            })
            .build();

        let response = agent
            .prompt("add")
            .add_hook(SkipDefaultApiHook)
            .max_turns(3)
            .extended_details()
            .await
            .expect("skip should produce synthetic feedback under Specific");

        assert_eq!(response.output, "skipped");
        let messages = response.messages.expect("messages should be present");
        assert!(history_contains_tool_call(&messages, "default_api"));
        assert!(messages.iter().any(|message| {
            matches!(
                message,
                Message::User { content }
                    if content.iter().any(|content| {
                        matches!(
                            content,
                            UserContent::ToolResult(result)
                                if result.call == "tool_call_1"
                                    && result.content.iter().any(|content| {
                                        matches!(
                                            content,
                                            rig_core::message::ToolResultContent::Text(text)
                                                if text.text == "default_api is not available"
                                        )
                                    })
                        )
                    })
            )
        }));
    }

    #[tokio::test]
    async fn repair_to_disallowed_specific_tool_fails() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text("should not be requested"),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model)
            .tool(MockAddTool)
            .tool(MockSubtractTool)
            .tool_choice(ToolChoice::Specific {
                function_names: vec!["add".to_string()],
            })
            .build();

        let err = agent
            .prompt("add")
            .add_hook(RepairToSubtractHook)
            .max_turns(3)
            .await
            .expect_err("repair to a disallowed tool should fail");

        match err {
            PromptError::UnknownToolCall { tool_name, .. } => {
                assert_eq!(tool_name, "subtract");
            }
            other => panic!("expected UnknownToolCall, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn repair_under_tool_choice_none_fails() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text("should not be requested"),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model)
            .tool(MockAddTool)
            .tool_choice(ToolChoice::None)
            .build();

        let err = agent
            .prompt("do not use tools")
            .add_hook(RepairDefaultApiHook)
            .max_turns(3)
            .await
            .expect_err("ToolChoice::None should reject repaired tool calls");

        match err {
            PromptError::UnknownToolCall { tool_name, .. } => {
                assert_eq!(tool_name, "add");
            }
            other => panic!("expected UnknownToolCall, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn skip_under_tool_choice_none_fails() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text("should not be requested"),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model)
            .tool(MockAddTool)
            .tool_choice(ToolChoice::None)
            .build();

        let err = agent
            .prompt("do not use tools")
            .add_hook(SkipDefaultApiHook)
            .max_turns(3)
            .await
            .expect_err("ToolChoice::None should reject skipped tool calls");

        match err {
            PromptError::UnknownToolCall { tool_name, .. } => {
                assert_eq!(tool_name, "default_api");
            }
            other => panic!("expected UnknownToolCall, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn typed_prompt_default_invalid_tool_call_fails_fast() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text(r#"{"value":"should not be requested"}"#),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let err = agent
            .prompt_typed::<TypedAnswer>("return typed json")
            .add_hook(PanicOnUnknownToolHook)
            .max_turns(3)
            .await
            .expect_err("typed prompt should preserve fail-fast default");

        match err {
            StructuredOutputError::PromptError(err) => match *err {
                PromptError::UnknownToolCall { tool_name, .. } => {
                    assert_eq!(tool_name, "default_api");
                }
                other => panic!("expected UnknownToolCall, got {other:?}"),
            },
            other => panic!("expected prompt error, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn typed_prompt_invalid_tool_call_hook_can_repair_tool_name() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text(r#"{"value":"repaired"}"#),
        ]);
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let response = agent
            .prompt_typed::<TypedAnswer>("return typed json")
            .add_hook(RepairDefaultApiHook)
            .max_turns(3)
            .await
            .expect("typed prompt should repair invalid tool call");

        assert_eq!(
            response,
            TypedAnswer {
                value: "repaired".to_string()
            }
        );
    }

    #[tokio::test]
    async fn typed_prompt_invalid_tool_call_hook_can_retry_and_parse_response() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text(r#"{"value":"retried"}"#),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let response = agent
            .prompt_typed::<TypedAnswer>("return typed json")
            .add_hook(RetryDefaultApiHook)
            .max_invalid_tool_call_retries(1)
            .max_turns(3)
            .await
            .expect("typed prompt should retry invalid tool call");

        assert_eq!(
            response,
            TypedAnswer {
                value: "retried".to_string()
            }
        );
        assert_eq!(recorded.request_count(), 2);
    }

    #[tokio::test]
    async fn typed_prompt_invalid_tool_call_retry_budget_exhaustion_fails() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text(r#"{"value":"should not be requested"}"#),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let err = agent
            .prompt_typed::<TypedAnswer>("return typed json")
            .add_hook(RetryDefaultApiHook)
            .max_invalid_tool_call_retries(0)
            .max_turns(3)
            .await
            .expect_err("typed prompt should fail when retry budget is exhausted");

        match err {
            StructuredOutputError::PromptError(err) => match *err {
                PromptError::UnknownToolCall { tool_name, .. } => {
                    assert_eq!(tool_name, "default_api");
                }
                other => panic!("expected UnknownToolCall, got {other:?}"),
            },
            other => panic!("expected prompt error, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn invalid_specific_tool_choice_fails_before_non_streaming_provider_request() {
        let model = MockCompletionModel::text("should not be requested");
        let recorded = model.clone();
        let agent = AgentBuilder::new(model)
            .tool(MockAddTool)
            .tool_choice(ToolChoice::Specific {
                function_names: vec!["missing".to_string()],
            })
            .build();

        let err = agent
            .prompt("use the missing tool")
            .await
            .expect_err("invalid ToolChoice::Specific should fail before provider request");

        match err {
            PromptError::CompletionError(CompletionError::RequestError(err)) => {
                let msg = err.to_string();
                assert!(msg.contains("missing"), "got: {msg}");
                assert!(msg.contains("add"), "got: {msg}");
            }
            other => panic!("expected CompletionError::RequestError, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 0);
    }

    #[tokio::test]
    async fn allowed_specific_tool_call_executes_normally() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "add", json!({"x": 1, "y": 2})),
            MockTurn::text("done"),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model)
            .tool(MockAddTool)
            .tool_choice(ToolChoice::Specific {
                function_names: vec!["add".to_string()],
            })
            .build();

        let response = agent
            .prompt("use the allowed tool")
            .max_turns(3)
            .await
            .expect("allowed specific tool should execute");

        assert_eq!(response, "done");
        assert_eq!(recorded.request_count(), 2);
    }

    #[tokio::test]
    async fn prompt_request_stops_cleanly_on_empty_terminal_turn() {
        let first_call_usage = Usage {
            input_tokens: 1,
            output_tokens: 1,
            total_tokens: 2,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
            tool_use_prompt_tokens: 0,
            reasoning_tokens: 0,
        };
        let second_call_usage = Usage {
            input_tokens: 1,
            output_tokens: 1,
            total_tokens: 2,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
            tool_use_prompt_tokens: 0,
            reasoning_tokens: 0,
        };
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "add", json!({"x": 1, "y": 2}))
                .with_call_id("call_1")
                .with_usage(first_call_usage),
            MockTurn::text("").with_usage(second_call_usage),
        ]);
        let agent = AgentBuilder::new(model.clone()).tool(MockAddTool).build();

        let response = agent
            .prompt("do tool work")
            .max_turns(3)
            .extended_details()
            .await
            .expect("empty terminal turn should not error");

        assert!(response.output.is_empty());
        assert_eq!(
            response.usage,
            Usage {
                input_tokens: 2,
                output_tokens: 2,
                total_tokens: 4,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
                tool_use_prompt_tokens: 0,
                reasoning_tokens: 0,
            }
        );
        assert_eq!(
            response.completion_calls(),
            &[
                CompletionCall::new(0, first_call_usage),
                CompletionCall::new(1, second_call_usage)
            ]
        );

        let history = response
            .messages
            .expect("extended response should include history");
        assert_eq!(history.len(), 3);
        assert!(matches!(
            history.first(),
            Some(Message::User { content })
                if matches!(
                    content.first(),
                    Some(UserContent::Text(text)) if text.text == "do tool work"
                )
        ));
        assert!(history.iter().any(|message| matches!(
            message,
            Message::Assistant { content, .. }
                if matches!(
                    content.first(),
                    Some(AssistantContent::ToolCall(tool_call))
                        if tool_call.id == "tool_call_1"
                            && tool_call.provider.as_ref().is_some_and(
                                |provider| provider.call_id == "call_1"
                            )
                )
        )));
        assert!(history.iter().any(|message| matches!(
            message,
            Message::User { content }
                if matches!(
                    content.first(),
                    Some(UserContent::ToolResult(tool_result))
                        if tool_result.call == "tool_call_1"
                            && tool_result.provider.as_ref().is_some_and(
                                |provider| provider.call_id == "call_1"
                            )
                )
        )));
        assert!(!history.iter().any(|message| matches!(
            message,
            Message::Assistant { content, .. }
                if content.iter().any(|item| matches!(
                    item,
                    AssistantContent::Text(text) if text.text.is_empty()
                ))
        )));
        let requests = model.requests();
        assert_eq!(requests.len(), 2);
        validate_follow_up_tool_history(&requests[1]);
    }

    #[tokio::test]
    async fn prompt_request_concatenates_text_blocks_without_inserted_newlines() {
        let model = MockCompletionModel::new([MockTurn::from_contents([
            AssistantContent::Text(Text::new("According to the document, ")),
            AssistantContent::Text(Text::new("the grass is green")),
            AssistantContent::Text(Text::new(" and the sky is blue.")),
        ])]);
        let agent = AgentBuilder::new(model).build();

        let response = agent
            .prompt("answer with cited spans")
            .await
            .expect("prompt should succeed");

        assert_eq!(
            response,
            "According to the document, the grass is green and the sky is blue."
        );
    }

    #[tokio::test]
    async fn prompt_request_preserves_metadata_only_text_turn_in_history() {
        let metadata = rig_core::message::AdditionalParams::try_from_value(json!({
            "citations": [{
                "type": "web_search_result_location",
                "cited_text": "Claude Shannon was born in 1916.",
                "url": "https://example.com/shannon",
                "title": null,
                "encrypted_index": "encrypted-reference"
            }]
        }))
        .expect("object params")
        .expect("params carry data");
        let model =
            MockCompletionModel::new([MockTurn::from_content(AssistantContent::Text(Text {
                text: String::new(),
                additional_params: Some(metadata.clone()),
            }))]);
        let agent = AgentBuilder::new(model).build();

        let response = agent
            .prompt("answer with cited metadata")
            .extended_details()
            .await
            .expect("metadata-only text turn should succeed");

        assert!(response.output.is_empty());
        let history = response
            .messages
            .expect("extended response should include history");
        assert!(history.iter().any(|message| matches!(
            message,
            Message::Assistant { content, .. }
                if matches!(
                    content.first(),
                    Some(AssistantContent::Text(text))
                        if text.text.is_empty()
                            && text.additional_params.as_ref() == Some(&metadata)
                )
        )));
    }

    // ----- Conversation memory integration tests -----

    use rig_core::memory::{ConversationMemory, InMemoryConversationMemory};

    #[tokio::test]
    async fn memory_loads_into_request_history() {
        let memory = InMemoryConversationMemory::new();
        memory
            .append(
                "thread-1",
                vec![Message::user("hello"), Message::assistant("hi there")],
            )
            .await
            .unwrap();

        let model = MockCompletionModel::text("ack");
        let recorded = model.clone();

        let agent = AgentBuilder::new(model).memory(memory).build();
        let _ = agent
            .prompt("ping")
            .conversation("thread-1")
            .await
            .expect("prompt should succeed");

        let received = recorded.requests()[0].chat_history.clone();
        assert_eq!(
            received.len(),
            3,
            "loaded memory (2) + current prompt should appear in request: {received:?}"
        );
    }

    #[tokio::test]
    async fn memory_appends_full_turn_after_success() {
        let memory = InMemoryConversationMemory::new();
        let model = MockCompletionModel::text("ack");
        let agent = AgentBuilder::new(model).memory(memory.clone()).build();

        let _ = agent
            .prompt("hello")
            .conversation("t1")
            .await
            .expect("prompt should succeed");

        let stored = memory.load("t1").await.unwrap();
        assert_eq!(stored.len(), 2, "user prompt + assistant response saved");
    }

    #[tokio::test]
    async fn explicit_with_history_overrides_memory() {
        let memory = CountingMemory::default();
        memory
            .inner()
            .append("t1", vec![Message::user("from-memory")])
            .await
            .unwrap();

        let model = MockCompletionModel::text("ack");
        let recorded = model.clone();

        let agent = AgentBuilder::new(model).memory(memory.clone()).build();
        let _ = agent
            .prompt("hello")
            .conversation("t1")
            .history(vec![Message::user("from-caller")])
            .await
            .expect("prompt should succeed");

        assert_eq!(memory.load_count(), 0, "load skipped");
        let appends = memory.append_count();
        assert_eq!(appends, 0, "append skipped");

        let received = recorded.requests()[0].chat_history.clone();
        assert_eq!(received.len(), 2, "caller history (1) + current prompt");
        assert!(matches!(
            received.first(),
            Some(Message::User { content })
                if matches!(content.first(), Some(UserContent::Text(t)) if t.text == "from-caller")
        ));
    }

    #[tokio::test]
    async fn memory_unchanged_on_provider_error() {
        let memory = InMemoryConversationMemory::new();
        let model = MockCompletionModel::new([MockTurn::error("boom")]);

        let agent = AgentBuilder::new(model).memory(memory.clone()).build();
        let result = agent.prompt("hello").conversation("t1").await;
        assert!(result.is_err());

        let stored = memory.load("t1").await.unwrap();
        assert!(stored.is_empty(), "no append on error");
    }

    #[tokio::test]
    async fn multi_step_tool_run_appends_committed_turn_exactly_once() {
        // A tool round-trip is two model calls (tool call -> final text) but one
        // run: the committed turn must be appended to memory exactly once, not
        // once per model call.
        let memory = CountingMemory::default();
        let model = MockCompletionModel::new([
            MockTurn::tool_call("call-1", "add", json!({"x": 2, "y": 3})),
            MockTurn::text("sum is 5"),
        ]);

        let agent = AgentBuilder::new(model)
            .memory(memory.clone())
            .tool(MockAddTool)
            .default_max_turns(2)
            .build();

        let _ = agent
            .prompt("add 2 and 3")
            .conversation("t1")
            .await
            .expect("multi-step run should succeed");

        assert_eq!(
            memory.append_count(),
            1,
            "one append for the whole run, not one per model call"
        );

        let stored = memory.load("t1").await.unwrap();
        // user prompt + assistant tool call + tool result + final assistant text.
        assert_eq!(
            stored.len(),
            4,
            "the full committed turn is persisted once: {stored:?}"
        );
        assert!(
            matches!(
                stored.last(),
                Some(Message::Assistant { content, .. })
                    if content
                        .iter()
                        .any(|item| matches!(item, AssistantContent::Text(t) if t.text == "sum is 5"))
            ),
            "final assistant text is persisted: {stored:?}"
        );
    }

    #[tokio::test]
    async fn append_persists_only_newly_committed_messages() {
        // With pre-loaded history, a run must append only the new turn's
        // messages, never re-append the loaded history (which would duplicate
        // it). Pre-load directly through `inner()` so it does not count as an
        // append by the run.
        let memory = CountingMemory::default();
        memory
            .inner()
            .append(
                "t1",
                vec![Message::user("old-q"), Message::assistant("old-a")],
            )
            .await
            .unwrap();

        let model = MockCompletionModel::text("new-a");
        let agent = AgentBuilder::new(model).memory(memory.clone()).build();

        let _ = agent
            .prompt("new-q")
            .conversation("t1")
            .await
            .expect("prompt should succeed");

        assert_eq!(memory.append_count(), 1, "one append for the run");

        let stored = memory.load("t1").await.unwrap();
        // preloaded [old-q, old-a] + new [new-q, new-a]; re-appending the loaded
        // history would instead make this 6.
        assert_eq!(
            stored.len(),
            4,
            "only the new turn is appended, loaded history is not duplicated: {stored:?}"
        );
        assert!(
            matches!(
                stored.first(),
                Some(Message::User { content })
                    if matches!(content.first(), Some(UserContent::Text(t)) if t.text == "old-q")
            ),
            "loaded history is preserved once at the front: {stored:?}"
        );
    }

    #[tokio::test]
    async fn hook_stopped_run_does_not_append() {
        // A run stopped by a hook before it completes must not append.
        struct StopOnCompletion;
        impl AgentHook for StopOnCompletion {
            async fn on_completion_call(
                &self,
                _ctx: &HookContext,
                _event: crate::agent::CompletionCallEvent<'_>,
            ) -> crate::agent::CompletionCallAction {
                crate::agent::CompletionCallAction::stop("stop")
            }
        }

        let memory = CountingMemory::default();
        let model = MockCompletionModel::text("unreached");
        let agent = AgentBuilder::new(model)
            .memory(memory.clone())
            .add_hook(StopOnCompletion)
            .build();

        let result = agent.prompt("hello").conversation("t1").await;
        assert!(result.is_err(), "a stop hook terminates the run");

        assert_eq!(memory.append_count(), 0, "stopped runs do not append");
        let stored = memory.load("t1").await.unwrap();
        assert!(stored.is_empty(), "nothing persisted on stop: {stored:?}");
    }

    #[tokio::test]
    async fn committed_transcript_roles_form_a_valid_sequence() {
        // The committed history of a tool round-trip must be a well-formed
        // role sequence: it starts with a user message, never commits two
        // consecutive assistant messages, and pairs each assistant tool call
        // with a following user tool-result message.
        let memory = CountingMemory::default();
        let model = MockCompletionModel::new([
            MockTurn::tool_call("call-1", "add", json!({"x": 1, "y": 1})),
            MockTurn::text("done"),
        ]);

        let agent = AgentBuilder::new(model)
            .memory(memory.clone())
            .tool(MockAddTool)
            .default_max_turns(2)
            .build();

        let _ = agent
            .prompt("go")
            .conversation("t1")
            .await
            .expect("run should succeed");

        let stored = memory.load("t1").await.unwrap();

        assert!(
            matches!(stored.first(), Some(Message::User { .. })),
            "committed transcript begins with a user message: {stored:?}"
        );
        assert!(
            !stored
                .windows(2)
                .any(|pair| matches!(pair, [Message::Assistant { .. }, Message::Assistant { .. }])),
            "no two assistant messages are committed back to back: {stored:?}"
        );
        // Each assistant turn carrying a tool call is followed by a user
        // tool-result message.
        for (index, message) in stored.iter().enumerate() {
            let has_tool_call = matches!(
                message,
                Message::Assistant { content, .. }
                    if content.iter().any(|item| matches!(item, AssistantContent::ToolCall(_)))
            );
            if has_tool_call {
                assert!(
                    matches!(stored.get(index + 1), Some(Message::User { content })
                        if content
                            .iter()
                            .any(|item| matches!(item, UserContent::ToolResult(_)))),
                    "assistant tool call at {index} is followed by a user tool result: {stored:?}"
                );
            }
        }
    }

    #[tokio::test]
    async fn missing_conversation_id_behaves_as_no_memory() {
        let memory = CountingMemory::default();
        let model = MockCompletionModel::text("ack");
        let agent = AgentBuilder::new(model).memory(memory.clone()).build();

        let _ = agent.prompt("hello").await.expect("prompt should succeed");

        assert_eq!(memory.load_count(), 0);
        assert_eq!(memory.append_count(), 0);
    }

    #[tokio::test]
    async fn default_conversation_id_is_used_when_none_per_request() {
        let memory = InMemoryConversationMemory::new();
        let model = MockCompletionModel::text("ack");
        let agent = AgentBuilder::new(model)
            .memory(memory.clone())
            .conversation("default-thread")
            .build();

        let _ = agent.prompt("hello").await.expect("prompt should succeed");
        let stored = memory.load("default-thread").await.unwrap();
        assert_eq!(stored.len(), 2);
    }

    #[tokio::test]
    async fn with_filter_truncates_loaded_history() {
        let memory = InMemoryConversationMemory::new()
            .with_filter(|msgs: Vec<Message>| msgs.into_iter().rev().take(2).rev().collect());
        memory
            .append(
                "t1",
                vec![
                    Message::user("1"),
                    Message::assistant("2"),
                    Message::user("3"),
                    Message::assistant("4"),
                ],
            )
            .await
            .unwrap();

        let model = MockCompletionModel::text("ack");
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).memory(memory).build();

        let _ = agent
            .prompt("ping")
            .conversation("t1")
            .await
            .expect("prompt should succeed");

        let received = recorded.requests()[0].chat_history.clone();
        assert_eq!(
            received.len(),
            3,
            "window-truncated history (2) + current prompt"
        );
    }

    #[tokio::test]
    async fn without_memory_disables_for_request() {
        let memory = CountingMemory::default();
        let model = MockCompletionModel::text("ack");
        let agent = AgentBuilder::new(model)
            .memory(memory.clone())
            .conversation("t1")
            .build();

        let _ = agent
            .prompt("hello")
            .without_memory()
            .await
            .expect("prompt should succeed");

        assert_eq!(memory.load_count(), 0);
        assert_eq!(memory.append_count(), 0);
    }

    #[tokio::test]
    async fn memory_load_error_surfaces_as_prompt_error() {
        let model = MockCompletionModel::text("ack");
        let agent = AgentBuilder::new(model)
            .memory(FailingMemory::default())
            .build();
        let result = agent.prompt("hello").conversation("t1").await;

        match result {
            Err(PromptError::MemoryError(err)) => {
                let msg = err.to_string();
                assert!(msg.contains("load boom"), "got: {msg}");
            }
            other => panic!("expected PromptError::MemoryError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn memory_append_error_does_not_drop_response() {
        let model = MockCompletionModel::text("ack");
        let agent = AgentBuilder::new(model)
            .memory(AppendFailingMemory::default())
            .build();
        let response: String = agent
            .prompt("hello")
            .conversation("t1")
            .await
            .expect("append failure must not block successful completion");

        assert!(!response.is_empty());
    }

    /// Serde compatibility (rig#2265): run records persisted before the
    /// identity fields existed still load, with every identity field `None`.
    #[test]
    fn completion_call_without_identity_fields_still_deserializes() {
        let call: CompletionCall = serde_json::from_str(
            r#"{"call_index": 3, "usage": {"input_tokens": 1, "output_tokens": 2,
                "total_tokens": 3, "cached_input_tokens": 0,
                "cache_creation_input_tokens": 0, "reasoning_tokens": 0}}"#,
        )
        .expect("pre-identity CompletionCall JSON should load");
        assert_eq!(call.call_index, 3);
        assert_eq!(call.identity(), ResponseIdentity::default());
    }

    /// And a populated record round-trips the identity losslessly.
    #[test]
    fn completion_call_identity_round_trips() {
        let call = CompletionCall::new(0, crate::completion::Usage::new()).with_identity(
            ResponseIdentity {
                message_id: Some("msg_1".into()),
                response_id: Some("resp_1".into()),
                provider_request_id: Some("req_1".into()),
            },
        );
        let json = serde_json::to_string(&call).expect("serialize");
        let restored: CompletionCall = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored, call);
    }
}
