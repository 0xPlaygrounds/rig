//! This module provides functionality for working with streaming completion models.
//! It provides traits and types for generating streaming completion requests and
//! handling streaming completion responses.
//!
//! Provider implementations use these types to expose raw streamed completion
//! events without depending on a runtime.

use crate::OneOrMany;
use crate::completion::{CompletionError, CompletionResponse, Usage};
use crate::message::{
    AssistantContent, Reasoning, ReasoningContent, Text, ToolCall, ToolFunction, ToolResult,
};
use crate::wasm_compat::WasmCompatSend;
use futures::stream::{AbortHandle, Abortable};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::sync::atomic::AtomicBool;
use std::task::{Context, Poll};
use tokio::sync::watch;

/// Control for pausing and resuming a streaming response
pub struct PauseControl {
    pub(crate) paused_tx: watch::Sender<bool>,
    pub(crate) paused_rx: watch::Receiver<bool>,
}

impl PauseControl {
    /// Create a pause controller in the running state.
    pub fn new() -> Self {
        let (paused_tx, paused_rx) = watch::channel(false);
        Self {
            paused_tx,
            paused_rx,
        }
    }

    /// Pause polling of the public stream until [`PauseControl::resume`] is called.
    pub fn pause(&self) {
        let _ = self.paused_tx.send(true);
    }

    /// Resume polling after a pause.
    pub fn resume(&self) {
        let _ = self.paused_tx.send(false);
    }

    /// Returns whether the stream is currently paused.
    pub fn is_paused(&self) -> bool {
        *self.paused_rx.borrow()
    }
}

impl Default for PauseControl {
    fn default() -> Self {
        Self::new()
    }
}

/// The content of a tool call delta - either the tool name or argument data
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum ToolCallDeltaContent {
    /// Tool/function name emitted by the provider.
    Name(String),
    /// Partial JSON argument data emitted by the provider.
    Delta(String),
}

/// Discriminant for [`StreamFinal`].
///
/// [`StreamedAssistantContent`] is `#[serde(untagged)]` and its
/// [`StreamedAssistantContent::Unknown`] variant matches any JSON value, so the
/// terminal record needs a field that identifies it structurally.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamFinalKind {
    /// The provider's terminal stream event.
    Final,
}

/// The provider's terminal stream record, normalized.
///
/// This replaces the provider-typed final payload that streams used to carry:
/// usage is a plain field rather than a trait method, and the finish reason is
/// normalized exactly as on the unary [`CompletionResponse`].
///
/// Providers that want their own terminal type keep it behind
/// [`RawStreamingResult`] and map it once with [`normalize_stream`].
///
/// # Emission contract
///
/// A terminal record is emitted only when the provider signaled genuine
/// completion — its own end-of-response event (an Anthropic `message_delta`
/// with a stop reason, an OpenAI `[DONE]` / `response.completed`, a Gemini
/// chunk carrying `finishReason`, and so on). Three failure shapes reach a
/// consumer, and they are distinct:
///
/// | Shape | `Err` item | Stream continues | Terminal record |
/// |---|---|---|---|
/// | Transport error (connection lost, HTTP failure) | yes | no | never |
/// | Malformed frame (recoverable parse error) | yes | yes | if a genuine terminal later arrives |
/// | Truncation (EOF without the provider's end event) | no | — | never |
///
/// On a terminal error (a transport failure or the provider's own failure
/// event), tool calls that were fully delivered before the failure are yielded
/// *before* the terminal `Err`; nothing follows the error — the stream then
/// ends without a terminal record.
///
/// Consequently an `Err` item is **not** by itself terminal: a malformed frame
/// is surfaced and the stream keeps consuming, so a later genuine terminal
/// still completes it. Consumers must drain the stream to `None` rather than
/// stop at the first `Err`, and must treat the absence of a terminal record as
/// truncation, never as a successful zero-usage completion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(from = "StreamFinalRepr")]
#[non_exhaustive]
pub struct StreamFinal {
    /// Discriminating field; always [`StreamFinalKind::Final`].
    pub kind: StreamFinalKind,
    /// Token usage reported by the provider for this streamed completion.
    /// Zero-valued usage is the documented sentinel for missing metrics.
    pub usage: Usage,
    /// Why the model stopped generating, when the provider reported it.
    ///
    /// [`normalize_stream`] applies
    /// [`FinishReason::reconcile_with_output`](crate::completion::FinishReason::reconcile_with_output)
    /// to this value using the tool calls actually seen on the stream, so a
    /// provider mapper does not need to (and cannot — it has no view of the
    /// preceding events).
    #[serde(default)]
    pub finish_reason: Option<crate::completion::FinishReason>,
    /// Provider-assigned *assistant message* ID, when available — only IDs the
    /// provider would recognize on a replayed assistant message. Response-scoped
    /// identifiers belong in [`StreamFinal::response_id`].
    #[serde(default)]
    pub message_id: Option<String>,
    /// Provider-assigned response-scoped ID, when available — e.g. an OpenAI
    /// chat `chatcmpl-` ID. Never replayed to a provider as a message ID.
    #[serde(default)]
    pub response_id: Option<String>,
    /// Stable descriptor name of the provider that produced this stream.
    pub provider: String,
    /// Provider-reported model identifier, when available.
    #[serde(default)]
    pub model: Option<String>,
}

impl StreamFinal {
    /// Create a terminal record for `provider` with `usage`; optional metadata
    /// starts unset and is filled in with the `with_*` helpers.
    pub fn new(provider: impl Into<String>, usage: Usage) -> Self {
        Self {
            kind: StreamFinalKind::Final,
            usage,
            finish_reason: None,
            message_id: None,
            response_id: None,
            provider: provider.into(),
            model: None,
        }
    }

    /// Attach the normalized finish reason.
    pub fn with_finish_reason(self, finish_reason: crate::completion::FinishReason) -> Self {
        self.with_optional_finish_reason(Some(finish_reason))
    }

    /// Attach the normalized finish reason when the provider reported one.
    pub fn with_optional_finish_reason(
        mut self,
        finish_reason: Option<crate::completion::FinishReason>,
    ) -> Self {
        self.finish_reason = finish_reason;
        self
    }

    /// Attach the provider-assigned message ID.
    ///
    /// An empty string is treated as absent, matching the unary
    /// [`CompletionResponse`](crate::completion::CompletionResponse) setters:
    /// the invariant lives in the setters so no provider call site can
    /// diverge.
    pub fn with_message_id(self, message_id: impl Into<String>) -> Self {
        self.with_optional_message_id(Some(message_id.into()))
    }

    /// Attach the provider-assigned message ID when the provider reported one.
    pub fn with_optional_message_id(mut self, message_id: Option<impl Into<String>>) -> Self {
        self.message_id = message_id.map(Into::into).filter(|id| !id.is_empty());
        self
    }

    /// Attach the provider-assigned response-scoped ID.
    pub fn with_response_id(self, response_id: impl Into<String>) -> Self {
        self.with_optional_response_id(Some(response_id.into()))
    }

    /// Attach the provider-assigned response-scoped ID when the provider
    /// reported one.
    pub fn with_optional_response_id(mut self, response_id: Option<impl Into<String>>) -> Self {
        self.response_id = response_id.map(Into::into).filter(|id| !id.is_empty());
        self
    }

    /// Attach the provider-reported model identifier.
    pub fn with_model(self, model: impl Into<String>) -> Self {
        self.with_optional_model(Some(model.into()))
    }

    /// Attach the provider-reported model identifier when the stream reported
    /// one.
    pub fn with_optional_model(mut self, model: Option<impl Into<String>>) -> Self {
        self.model = model.map(Into::into).filter(|model| !model.is_empty());
        self
    }
}

/// Wire-shape mirror of [`StreamFinal`], used only for deserialization.
///
/// Serde must never construct an invariant-bearing value structurally: a plain
/// derive would let `"message_id":""` skip the empty-string filtering the
/// `with_*` setters apply. This mirror deserializes the exact wire shape —
/// including the discriminating `kind` field — and [`From`] funnels it through
/// [`StreamFinal::new`] and the setters, so every deserialized value satisfies
/// the same invariants as a constructed one. Serialization stays derived on
/// [`StreamFinal`] itself, so the wire format is unchanged.
#[derive(Deserialize)]
struct StreamFinalRepr {
    kind: StreamFinalKind,
    usage: Usage,
    #[serde(default)]
    finish_reason: Option<crate::completion::FinishReason>,
    #[serde(default)]
    message_id: Option<String>,
    #[serde(default)]
    response_id: Option<String>,
    provider: String,
    #[serde(default)]
    model: Option<String>,
}

impl From<StreamFinalRepr> for StreamFinal {
    fn from(repr: StreamFinalRepr) -> Self {
        let StreamFinalRepr {
            kind,
            usage,
            finish_reason,
            message_id,
            response_id,
            provider,
            model,
        } = repr;
        // `StreamFinal::new` sets the only possible discriminant; the
        // irrefutable pattern consumes the mirrored field.
        let StreamFinalKind::Final = kind;
        Self::new(provider, usage)
            .with_optional_finish_reason(finish_reason)
            .with_optional_message_id(message_id)
            .with_optional_response_id(response_id)
            .with_optional_model(model)
    }
}

/// Enum representing a streaming chunk from the model.
///
/// `R` is the terminal record type. Ordinary streams use the normalized
/// [`StreamFinal`] default; a provider's inherent `raw_stream` method
/// substitutes its own native terminal type over the same event vocabulary,
/// which is what keeps [`crate::completion::CompletionModel`] free of response
/// associated types.
#[derive(Debug, Clone)]
pub enum RawStreamingChoice<R = StreamFinal> {
    /// A text chunk from a message response
    Message(String),

    /// Start a new text content block in the accumulated final choice.
    ///
    /// This is an internal provider-normalization event. It is not yielded to
    /// public stream consumers, but lets providers preserve metadata boundaries
    /// for final aggregated assistant text blocks.
    TextStart {
        /// Provider-specific metadata attached to this text block.
        additional_params: Option<serde_json::Value>,
    },

    /// Provider-specific metadata for the current text content block.
    ///
    /// This is not yielded to public stream consumers. The metadata is merged
    /// into the current aggregated [`Text`] block.
    TextAdditionalParams(serde_json::Value),

    /// A tool call response (in its entirety)
    ToolCall(RawStreamingToolCall),
    /// A tool call partial/delta
    ToolCallDelta {
        /// Provider-supplied tool call ID.
        id: String,
        /// Rig-generated unique identifier for this tool call.
        internal_call_id: String,
        content: ToolCallDeltaContent,
    },
    /// A reasoning (in its entirety)
    Reasoning {
        /// Identity of the reasoning item this block belongs to.
        ///
        /// Required: reasoning interleaves with other output on real wires
        /// (OpenAI Responses emits the completed item after tool calls), so
        /// the accumulator must key by identity rather than guess by
        /// adjacency. Providers propagate the wire's item id (`item_id` on
        /// Responses events, the content-block index on Anthropic/Bedrock)
        /// or mint a stream-stable id at the boundary when the wire has
        /// none — core never invents identity. Deltas and the full block
        /// for the same item MUST carry the same id.
        id: String,
        /// Complete reasoning content block.
        content: ReasoningContent,
    },
    /// A reasoning partial/delta
    ReasoningDelta {
        /// Identity of the reasoning item this delta extends. Same contract
        /// as [`RawStreamingChoice::Reasoning::id`]; all deltas of one block
        /// share one id.
        id: String,
        /// Partial reasoning text.
        reasoning: String,
    },

    /// The final response object, must be yielded if you want the
    /// `response` field to be populated on the `StreamingCompletionResponse`
    FinalResponse(R),

    /// Provider-assigned message ID (e.g. OpenAI Responses API `msg_` ID).
    /// Captured silently into `StreamingCompletionResponse::message_id`.
    MessageId(String),

    /// A provider-native output item this version does not model — e.g. an
    /// OpenAI Responses hosted-tool result (`web_search_call`, `file_search_call`,
    /// `computer_call`, `code_interpreter_call`). Carries the raw item object
    /// verbatim. Forwarded to the stream consumer as
    /// [`StreamedAssistantContent::Unknown`] but not folded into the accumulated
    /// assistant message (there is no `AssistantContent::Unknown` history slot).
    Unknown(serde_json::Value),
}

impl<R> RawStreamingChoice<R> {
    /// Convert only the terminal record, preserving every incremental content
    /// event unchanged.
    pub fn try_map_final<S>(
        self,
        map: impl FnOnce(R) -> Result<S, CompletionError>,
    ) -> Result<RawStreamingChoice<S>, CompletionError> {
        Ok(match self {
            Self::Message(text) => RawStreamingChoice::Message(text),
            Self::TextStart { additional_params } => {
                RawStreamingChoice::TextStart { additional_params }
            }
            Self::TextAdditionalParams(params) => RawStreamingChoice::TextAdditionalParams(params),
            Self::ToolCall(call) => RawStreamingChoice::ToolCall(call),
            Self::ToolCallDelta {
                id,
                internal_call_id,
                content,
            } => RawStreamingChoice::ToolCallDelta {
                id,
                internal_call_id,
                content,
            },
            Self::Reasoning { id, content } => RawStreamingChoice::Reasoning { id, content },
            Self::ReasoningDelta { id, reasoning } => {
                RawStreamingChoice::ReasoningDelta { id, reasoning }
            }
            Self::FinalResponse(response) => RawStreamingChoice::FinalResponse(map(response)?),
            Self::MessageId(id) => RawStreamingChoice::MessageId(id),
            Self::Unknown(value) => RawStreamingChoice::Unknown(value),
        })
    }
}

/// Describes a streaming tool call response (in its entirety)
#[derive(Debug, Clone)]
pub struct RawStreamingToolCall {
    /// Provider-supplied tool call ID.
    pub id: String,
    /// Rig-generated unique identifier for this tool call.
    pub internal_call_id: String,
    /// Provider-specific call ID used by some APIs for tool result correlation.
    pub call_id: Option<String>,
    /// Tool/function name.
    pub name: String,
    /// Parsed tool arguments.
    pub arguments: serde_json::Value,
    /// Optional provider signature associated with the tool call.
    pub signature: Option<String>,
    /// Additional provider-specific tool call metadata.
    pub additional_params: Option<serde_json::Value>,
}

impl RawStreamingToolCall {
    /// Create an empty tool call accumulator for provider streaming parsers.
    pub fn empty() -> Self {
        Self {
            id: String::new(),
            internal_call_id: crate::id::generate(),
            call_id: None,
            name: String::new(),
            arguments: serde_json::Value::Null,
            signature: None,
            additional_params: None,
        }
    }

    /// Create a complete tool call with a generated internal call ID.
    pub fn new(id: String, name: String, arguments: serde_json::Value) -> Self {
        Self {
            id,
            internal_call_id: crate::id::generate(),
            call_id: None,
            name,
            arguments,
            signature: None,
            additional_params: None,
        }
    }

    /// Override the generated internal call ID.
    pub fn with_internal_call_id(mut self, internal_call_id: String) -> Self {
        self.internal_call_id = internal_call_id;
        self
    }

    /// Attach a provider-specific call ID.
    pub fn with_call_id(mut self, call_id: String) -> Self {
        self.call_id = Some(call_id);
        self
    }

    /// Attach or clear a provider signature.
    pub fn with_signature(mut self, signature: Option<String>) -> Self {
        self.signature = signature;
        self
    }

    /// Attach provider-specific metadata.
    pub fn with_additional_params(mut self, additional_params: Option<serde_json::Value>) -> Self {
        self.additional_params = additional_params;
        self
    }
}

impl From<RawStreamingToolCall> for ToolCall {
    fn from(tool_call: RawStreamingToolCall) -> Self {
        ToolCall {
            id: tool_call.id,
            call_id: tool_call.call_id,
            function: ToolFunction {
                name: tool_call.name,
                arguments: tool_call.arguments,
            },
            signature: tool_call.signature,
            additional_params: tool_call.additional_params,
        }
    }
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
/// Provider stream whose terminal record is the provider-native `R`, on native
/// targets.
pub type RawStreamingResult<R> =
    Pin<Box<dyn Stream<Item = Result<RawStreamingChoice<R>, CompletionError>> + Send>>;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
/// Provider stream whose terminal record is the provider-native `R`, on wasm
/// targets.
pub type RawStreamingResult<R> =
    Pin<Box<dyn Stream<Item = Result<RawStreamingChoice<R>, CompletionError>>>>;

/// Normalized provider stream, as consumed by [`StreamingCompletionResponse`].
pub type StreamingResult = RawStreamingResult<StreamFinal>;

/// Normalize the terminal record of a provider-native stream.
///
/// Every incremental event passes through untouched; only
/// [`RawStreamingChoice::FinalResponse`] is converted, by `map`. On the way
/// through, the stream remembers whether it emitted any tool call and applies
/// [`FinishReason::reconcile_with_output`](crate::completion::FinishReason::reconcile_with_output)
/// to the mapped record — the streaming counterpart of what
/// [`CompletionResponse::with_finish_reason`] does on the unary path, so both
/// paths agree about a `stop` that was really a tool call.
pub fn normalize_stream<R, F>(stream: RawStreamingResult<R>, mut map: F) -> StreamingResult
where
    R: 'static,
    F: FnMut(R) -> Result<StreamFinal, CompletionError> + WasmCompatSend + 'static,
{
    let mut emitted_tool_call = false;
    Box::pin(stream.map(move |item| {
        item.and_then(|choice| {
            // Only a completed `ToolCall` counts, because only that becomes an
            // `AssistantContent::ToolCall` in the aggregated choice — which is
            // exactly what the unary path reconciles against. Counting deltas
            // here would make a stream whose tool call never assembled report
            // `ToolCalls` while the same data converted to a unary response
            // reported `Stop`.
            if matches!(&choice, RawStreamingChoice::ToolCall(_)) {
                emitted_tool_call = true;
            }
            choice.try_map_final(|response| {
                let mut response = map(response)?;
                response.finish_reason = response
                    .finish_reason
                    .map(|reason| reason.reconcile_with_output(emitted_tool_call));
                Ok(response)
            })
        })
    }))
}

/// The response from a streaming completion request;
/// message and response are populated at the end of the
/// `inner` stream.
pub struct StreamingCompletionResponse {
    pub(crate) inner: Abortable<StreamingResult>,
    pub(crate) abort_handle: AbortHandle,
    pub(crate) pause_control: PauseControl,
    assistant_items: Vec<AssistantContent>,
    text_item_index: Option<usize>,
    reasoning_item_index: Option<usize>,
    /// Stable descriptor name of the provider producing this stream.
    ///
    /// Known when the stream is opened rather than when it terminates, so a
    /// stream that errors or is cancelled before its terminal record still
    /// names its provider.
    provider: String,
    /// The final aggregated message from the stream
    /// contains all text and tool calls generated
    pub choice: OneOrMany<AssistantContent>,
    /// The provider's normalized terminal record, may be `None`
    /// if the provider didn't yield it during the stream
    pub response: Option<StreamFinal>,
    pub final_response_yielded: AtomicBool,
    /// Provider-assigned message ID (e.g. OpenAI Responses API `msg_` ID).
    pub message_id: Option<String>,
}

impl StreamingCompletionResponse {
    /// Wrap a provider stream and initialize aggregation state.
    ///
    /// `provider` is the stable descriptor name of the provider producing the
    /// stream; it is recorded up front so it is available even when the stream
    /// never reaches its terminal record.
    pub fn stream(provider: impl Into<String>, inner: StreamingResult) -> Self {
        let (abort_handle, abort_registration) = AbortHandle::new_pair();
        let abortable_stream = Abortable::new(inner, abort_registration);
        let pause_control = PauseControl::new();
        Self {
            inner: abortable_stream,
            abort_handle,
            pause_control,
            assistant_items: vec![],
            text_item_index: None,
            reasoning_item_index: None,
            provider: provider.into(),
            choice: OneOrMany::one(AssistantContent::text("")),
            response: None,
            final_response_yielded: AtomicBool::new(false),
            message_id: None,
        }
    }

    /// Stable descriptor name of the provider producing this stream.
    pub fn provider(&self) -> &str {
        &self.provider
    }

    /// Cancel the stream and immediately drop the provider's inner stream.
    /// Cancellation is surfaced as normal stream termination.
    pub fn cancel(&mut self) {
        self.abort_handle.abort();
        let (abort_handle, abort_registration) = AbortHandle::new_pair();
        let empty: StreamingResult = Box::pin(futures::stream::poll_fn(|_| Poll::Ready(None)));
        self.inner = Abortable::new(empty, abort_registration);
        self.abort_handle = abort_handle;
    }

    /// Pause stream polling.
    pub fn pause(&self) {
        self.pause_control.pause();
    }

    /// Resume stream polling after a pause.
    pub fn resume(&self) {
        self.pause_control.resume();
    }

    /// Returns whether the stream is currently paused.
    pub fn is_paused(&self) -> bool {
        self.pause_control.is_paused()
    }

    /// Token usage reported by the provider for this response.
    ///
    /// Returns the usage carried by the final response once the stream has
    /// produced it. Until then — or when the provider does not report streamed
    /// usage — this returns [`Usage::new`], the zero-valued sentinel for missing
    /// usage metrics.
    pub fn usage(&self) -> Usage {
        self.response
            .as_ref()
            .map(|response| response.usage)
            .unwrap_or_default()
    }

    fn append_text_chunk(&mut self, text: &str) {
        if let Some(index) = self.text_item_index
            && let Some(AssistantContent::Text(existing_text)) = self.assistant_items.get_mut(index)
        {
            existing_text.text.push_str(text);
            return;
        }

        self.assistant_items
            .push(AssistantContent::text(text.to_owned()));
        self.text_item_index = Some(self.assistant_items.len() - 1);
    }

    fn append_text_additional_params(&mut self, additional_params: serde_json::Value) {
        if additional_params.is_null() {
            return;
        }

        let index = if let Some(index) = self.text_item_index
            && matches!(
                self.assistant_items.get(index),
                Some(AssistantContent::Text(_))
            ) {
            index
        } else {
            self.assistant_items.push(AssistantContent::text(""));
            let index = self.assistant_items.len() - 1;
            self.text_item_index = Some(index);
            index
        };

        let Some(AssistantContent::Text(text)) = self.assistant_items.get_mut(index) else {
            return;
        };

        match text.additional_params.as_mut() {
            Some(existing) => merge_text_additional_params(existing, additional_params),
            None => text.additional_params = Some(additional_params),
        }
    }

    /// Accumulate streaming reasoning delta text into assistant_items.
    /// Providers that only emit ReasoningDelta (not full Reasoning blocks)
    /// need this so the aggregated response includes reasoning content.
    fn append_reasoning_chunk(&mut self, id: &str, text: &str) {
        // Deltas key strictly by item id: a delta for a different item never
        // merges into the active block (ids are mandatory on the raw grammar,
        // so this is exact, not heuristic).
        if let Some(index) = self.reasoning_item_index
            && let Some(AssistantContent::Reasoning(existing)) = self.assistant_items.get_mut(index)
            && existing.id.as_deref() == Some(id)
            && let Some(ReasoningContent::Text {
                text: existing_text,
                ..
            }) = existing.content.last_mut()
        {
            existing_text.push_str(text);
            return;
        }

        self.assistant_items
            .push(AssistantContent::Reasoning(Reasoning {
                id: Some(id.to_string()),
                content: vec![ReasoningContent::Text {
                    text: text.to_string(),
                    signature: None,
                }],
            }));
        self.reasoning_item_index = Some(self.assistant_items.len() - 1);
    }
}

fn merge_text_additional_params(existing: &mut serde_json::Value, incoming: serde_json::Value) {
    match (existing, incoming) {
        (serde_json::Value::Object(existing_map), serde_json::Value::Object(incoming_map)) => {
            for (key, incoming_value) in incoming_map {
                match existing_map.get_mut(&key) {
                    Some(existing_value) => match (existing_value, incoming_value) {
                        (
                            serde_json::Value::Array(existing_array),
                            serde_json::Value::Array(mut incoming_array),
                        ) => existing_array.append(&mut incoming_array),
                        (existing_value, incoming_value) => {
                            merge_text_additional_params(existing_value, incoming_value);
                        }
                    },
                    None => {
                        existing_map.insert(key, incoming_value);
                    }
                }
            }
        }
        (existing, incoming) => {
            *existing = incoming;
        }
    }
}

impl From<StreamingCompletionResponse> for CompletionResponse {
    fn from(value: StreamingCompletionResponse) -> CompletionResponse {
        // Usage is the zero sentinel (`Usage::new`) when the stream produced no
        // terminal record. `provider` comes from the stream itself rather than
        // the terminal record, so it is populated even then.
        let terminal = value.response.as_ref();
        CompletionResponse::new(
            value.choice,
            terminal.map(|response| response.usage).unwrap_or_default(),
            value.provider,
        )
        // An explicit `MessageId` event outranks the terminal record's ID.
        .with_optional_message_id(
            value
                .message_id
                .or_else(|| terminal.and_then(|response| response.message_id.clone())),
        )
        .with_optional_response_id(terminal.and_then(|response| response.response_id.clone()))
        .with_optional_finish_reason(terminal.and_then(|response| response.finish_reason.clone()))
        .with_optional_model(terminal.and_then(|response| response.model.clone()))
    }
}

impl Stream for StreamingCompletionResponse {
    type Item = Result<StreamedAssistantContent, CompletionError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let stream = self.get_mut();

        if stream.is_paused() {
            cx.waker().wake_by_ref();
            return Poll::Pending;
        }

        match Pin::new(&mut stream.inner).poll_next(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(None) => {
                // This is run at the end of the inner stream to collect all tokens into
                // a single unified `Message`.
                if stream.assistant_items.is_empty() {
                    stream.assistant_items.push(AssistantContent::text(""));
                }

                if let Some(choice) =
                    OneOrMany::from_iter_optional(std::mem::take(&mut stream.assistant_items))
                {
                    stream.choice = choice;
                }

                Poll::Ready(None)
            }
            Poll::Ready(Some(Err(err))) => {
                if matches!(err, CompletionError::ProviderError(ref e) if e.to_string().contains("aborted"))
                {
                    return Poll::Ready(None); // Treat cancellation as stream termination
                }
                Poll::Ready(Some(Err(err)))
            }
            Poll::Ready(Some(Ok(choice))) => match choice {
                RawStreamingChoice::Message(text) => {
                    stream.reasoning_item_index = None;
                    stream.append_text_chunk(&text);
                    Poll::Ready(Some(Ok(StreamedAssistantContent::text(&text))))
                }
                RawStreamingChoice::TextStart { additional_params } => {
                    stream.reasoning_item_index = None;
                    stream.text_item_index = None;
                    if let Some(additional_params) = additional_params {
                        stream.append_text_additional_params(additional_params);
                    }
                    stream.poll_next_unpin(cx)
                }
                RawStreamingChoice::TextAdditionalParams(additional_params) => {
                    stream.append_text_additional_params(additional_params);
                    stream.poll_next_unpin(cx)
                }
                RawStreamingChoice::ToolCallDelta {
                    id,
                    internal_call_id,
                    content,
                } => Poll::Ready(Some(Ok(StreamedAssistantContent::ToolCallDelta {
                    id,
                    internal_call_id,
                    content,
                }))),
                RawStreamingChoice::Reasoning { id, content } => {
                    let reasoning = Reasoning {
                        id: Some(id),
                        content: vec![content],
                    };
                    stream.text_item_index = None;
                    // A full reasoning block supersedes its own delta
                    // accumulation: the deltas are only a fallback for
                    // providers that never send the completed block, so the
                    // delta-built item is *replaced*, not kept alongside a
                    // duplicate. Identity is decided by ID: matching IDs (or
                    // neither side carrying one) replace; mismatched IDs —
                    // including an ID on only one side — belong to a
                    // different reasoning item and append. The active-index
                    // slot is only a fast path: providers may emit the
                    // completed block after other output cleared the index
                    // (reasoning → tool call → completed block), so a miss
                    // falls back to a by-ID scan of the aggregated items.
                    // Ids are mandatory on the raw grammar, so identity is
                    // exact equality — no heuristics. The active-index slot is
                    // only a fast path; a miss (other output interleaved since
                    // the deltas) falls back to a by-id scan.
                    let same_item = |existing: &Reasoning| existing.id == reasoning.id;
                    let replace_index = stream
                        .reasoning_item_index
                        .filter(|&index| {
                            matches!(
                                stream.assistant_items.get(index),
                                Some(AssistantContent::Reasoning(existing)) if same_item(existing)
                            )
                        })
                        .or_else(|| {
                            stream.assistant_items.iter().rposition(|item| {
                                matches!(item, AssistantContent::Reasoning(existing) if same_item(existing))
                            })
                        });
                    match replace_index.and_then(|index| stream.assistant_items.get_mut(index)) {
                        Some(item) => *item = AssistantContent::Reasoning(reasoning.clone()),
                        None => stream
                            .assistant_items
                            .push(AssistantContent::Reasoning(reasoning.clone())),
                    }
                    stream.reasoning_item_index = None;
                    Poll::Ready(Some(Ok(StreamedAssistantContent::Reasoning(reasoning))))
                }
                RawStreamingChoice::ReasoningDelta { id, reasoning } => {
                    stream.text_item_index = None;
                    stream.append_reasoning_chunk(&id, &reasoning);
                    Poll::Ready(Some(Ok(StreamedAssistantContent::ReasoningDelta {
                        id,
                        reasoning,
                    })))
                }
                RawStreamingChoice::ToolCall(raw_tool_call) => {
                    let internal_call_id = raw_tool_call.internal_call_id.clone();
                    let tool_call: ToolCall = raw_tool_call.into();
                    stream.text_item_index = None;
                    stream.reasoning_item_index = None;
                    stream
                        .assistant_items
                        .push(AssistantContent::ToolCall(tool_call.clone()));
                    Poll::Ready(Some(Ok(StreamedAssistantContent::ToolCall {
                        tool_call,
                        internal_call_id,
                    })))
                }
                RawStreamingChoice::FinalResponse(response) => {
                    if stream
                        .final_response_yielded
                        .load(std::sync::atomic::Ordering::SeqCst)
                    {
                        stream.poll_next_unpin(cx)
                    } else {
                        // Set the final response field and return the next item in the stream.
                        // An explicit `MessageId` event keeps precedence; the
                        // terminal record only fills a gap.
                        if stream.message_id.is_none() {
                            stream.message_id = response.message_id.clone();
                        }
                        stream.response = Some(response.clone());
                        stream
                            .final_response_yielded
                            .store(true, std::sync::atomic::Ordering::SeqCst);
                        let final_response = StreamedAssistantContent::final_response(response);
                        Poll::Ready(Some(Ok(final_response)))
                    }
                }
                RawStreamingChoice::MessageId(id) => {
                    stream.message_id = Some(id);
                    stream.poll_next_unpin(cx)
                }
                RawStreamingChoice::Unknown(value) => {
                    // Pass an unmodeled provider item straight through to the
                    // consumer; it is intentionally not pushed into
                    // `assistant_items` (no `AssistantContent::Unknown` exists).
                    Poll::Ready(Some(Ok(StreamedAssistantContent::Unknown(value))))
                }
            },
        }
    }
}

// Test module
#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::completion::FinishReason;
    use async_stream::stream;
    use tokio::time::sleep;

    /// Provider descriptor used by the mock streams in this module.
    const TEST_PROVIDER: &str = "test-provider";

    /// Terminal record with a known total-token count.
    fn mock_final_with_total_tokens(total_tokens: u64) -> StreamFinal {
        let mut usage = Usage::new();
        usage.total_tokens = total_tokens;
        StreamFinal::new(TEST_PROVIDER, usage)
    }

    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    fn to_stream_result(
        stream: impl futures::Stream<Item = Result<RawStreamingChoice, CompletionError>>
        + Send
        + 'static,
    ) -> StreamingResult {
        Box::pin(stream)
    }

    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    fn to_stream_result(
        stream: impl futures::Stream<Item = Result<RawStreamingChoice, CompletionError>> + 'static,
    ) -> StreamingResult {
        Box::pin(stream)
    }

    fn create_mock_stream() -> StreamingCompletionResponse {
        let stream = stream! {
            yield Ok(RawStreamingChoice::Message("hello 1".to_string()));
            sleep(Duration::from_millis(100)).await;
            yield Ok(RawStreamingChoice::Message("hello 2".to_string()));
            sleep(Duration::from_millis(100)).await;
            yield Ok(RawStreamingChoice::Message("hello 3".to_string()));
            sleep(Duration::from_millis(100)).await;
            yield Ok(RawStreamingChoice::FinalResponse(mock_final_with_total_tokens(15)));
        };

        StreamingCompletionResponse::stream(TEST_PROVIDER, to_stream_result(stream))
    }

    fn create_reasoning_stream() -> StreamingCompletionResponse {
        let stream = stream! {
            yield Ok(RawStreamingChoice::Reasoning {                id: "rs_1".to_string(),
                content: ReasoningContent::Text {
                    text: "step one".to_string(),
                    signature: Some("sig_1".to_string()),
                },
            });
            yield Ok(RawStreamingChoice::Message("final answer".to_string()));
            yield Ok(RawStreamingChoice::FinalResponse(mock_final_with_total_tokens(5)));
        };

        StreamingCompletionResponse::stream(TEST_PROVIDER, to_stream_result(stream))
    }

    fn create_reasoning_only_stream() -> StreamingCompletionResponse {
        let stream = stream! {
            yield Ok(RawStreamingChoice::Reasoning {                id: "rs_only".to_string(),
                content: ReasoningContent::Summary("hidden summary".to_string()),
            });
            yield Ok(RawStreamingChoice::FinalResponse(mock_final_with_total_tokens(2)));
        };

        StreamingCompletionResponse::stream(TEST_PROVIDER, to_stream_result(stream))
    }

    fn create_interleaved_stream() -> StreamingCompletionResponse {
        let stream = stream! {
            yield Ok(RawStreamingChoice::Reasoning {                id: "rs_interleaved".to_string(),
                content: ReasoningContent::Text {
                    text: "chain-of-thought".to_string(),
                    signature: None,
                },
            });
            yield Ok(RawStreamingChoice::Message("final-text".to_string()));
            yield Ok(RawStreamingChoice::ToolCall(
                RawStreamingToolCall::new(
                    "tool_1".to_string(),
                    "mock_tool".to_string(),
                    serde_json::json!({"arg": 1}),
                ),
            ));
            yield Ok(RawStreamingChoice::FinalResponse(mock_final_with_total_tokens(3)));
        };

        StreamingCompletionResponse::stream(TEST_PROVIDER, to_stream_result(stream))
    }

    fn create_text_tool_text_stream() -> StreamingCompletionResponse {
        let stream = stream! {
            yield Ok(RawStreamingChoice::Message("first".to_string()));
            yield Ok(RawStreamingChoice::ToolCall(
                RawStreamingToolCall::new(
                    "tool_split".to_string(),
                    "mock_tool".to_string(),
                    serde_json::json!({"arg": "x"}),
                ),
            ));
            yield Ok(RawStreamingChoice::Message("second".to_string()));
            yield Ok(RawStreamingChoice::FinalResponse(mock_final_with_total_tokens(3)));
        };

        StreamingCompletionResponse::stream(TEST_PROVIDER, to_stream_result(stream))
    }

    fn create_text_metadata_stream() -> StreamingCompletionResponse {
        let stream = stream! {
            yield Ok(RawStreamingChoice::TextStart {
                additional_params: None,
            });
            yield Ok(RawStreamingChoice::Message("first".to_string()));
            yield Ok(RawStreamingChoice::TextAdditionalParams(serde_json::json!({
                "citations": [{
                    "type": "char_location",
                    "cited_text": "First citation.",
                    "document_index": 0,
                    "start_char_index": 0,
                    "end_char_index": 15
                }]
            })));
            yield Ok(RawStreamingChoice::TextAdditionalParams(serde_json::json!({
                "citations": [{
                    "type": "char_location",
                    "cited_text": "Second citation.",
                    "document_index": 0,
                    "start_char_index": 16,
                    "end_char_index": 32
                }]
            })));
            yield Ok(RawStreamingChoice::TextStart {
                additional_params: Some(serde_json::json!({
                    "block": 2
                })),
            });
            yield Ok(RawStreamingChoice::Message("second".to_string()));
            yield Ok(RawStreamingChoice::FinalResponse(mock_final_with_total_tokens(3)));
        };

        StreamingCompletionResponse::stream(TEST_PROVIDER, to_stream_result(stream))
    }

    #[tokio::test]
    async fn into_completion_response_derives_usage_from_final_response() {
        let mut stream = create_mock_stream();

        // Drain the stream so the final response (and its usage) is captured.
        while stream.next().await.is_some() {}

        // usage() surfaces the final response's token usage...
        assert_eq!(stream.usage().total_tokens, 15);

        // ...and the From conversion carries it instead of a zero sentinel.
        let response: CompletionResponse = stream.into();
        assert_eq!(response.usage.total_tokens, 15);
        assert_eq!(response.provider, TEST_PROVIDER);
    }

    #[tokio::test]
    async fn a_stream_without_a_terminal_record_still_names_its_provider() {
        // The provider is known when the stream is opened, so a stream that
        // errors or is truncated before its terminal record must not degrade
        // `provider` to an empty string — every other missing value has a
        // documented sentinel (`Usage::new`, `None`) and this one should too.
        let mut stream = StreamingCompletionResponse::stream(
            TEST_PROVIDER,
            to_stream_result(stream! {
                yield Ok(RawStreamingChoice::Message("truncated".to_string()));
            }),
        );
        while stream.next().await.is_some() {}

        // No terminal record was ever yielded, so none may be synthesized.
        assert!(stream.response.is_none());

        let response: CompletionResponse = stream.into();
        assert_eq!(response.provider, TEST_PROVIDER);
        assert_eq!(response.usage, Usage::new());
        assert_eq!(response.finish_reason(), None);
        assert_eq!(response.model, None);
    }

    #[tokio::test]
    async fn a_stream_that_errors_mid_stream_keeps_content_and_omits_the_terminal() {
        // A transport error after some content must forward the error, keep
        // the content already aggregated, and never fabricate a terminal
        // record the provider did not send.
        let mut stream = StreamingCompletionResponse::stream(
            TEST_PROVIDER,
            to_stream_result(stream! {
                yield Ok(RawStreamingChoice::Message("partial".to_string()));
                yield Err(CompletionError::ProviderError(
                    "connection reset".to_string(),
                ));
            }),
        );

        let mut saw_error = false;
        while let Some(item) = stream.next().await {
            if item.is_err() {
                saw_error = true;
            }
        }
        assert!(saw_error, "the mid-stream error must be forwarded");

        // No StreamFinal may be synthesized for the aborted stream...
        assert!(stream.response.is_none());

        // ...but the content delivered before the error is preserved.
        assert_eq!(
            stream.choice.first(),
            AssistantContent::text("partial".to_string()),
        );
    }

    #[tokio::test]
    async fn normalize_stream_upgrades_a_stop_that_carried_a_tool_call() {
        // Several gateways report a plain `stop` on a tool-calling turn. The
        // streaming path must reconcile it exactly as the unary path does.
        let raw: RawStreamingResult<Usage> = Box::pin(stream! {
            yield Ok(RawStreamingChoice::ToolCall(RawStreamingToolCall {
                id: "call_1".to_string(),
                call_id: None,
                internal_call_id: "internal_1".to_string(),
                name: "lookup".to_string(),
                arguments: serde_json::json!({}),
                signature: None,
                additional_params: None,
            }));
            yield Ok(RawStreamingChoice::FinalResponse(Usage::new()));
        });

        let normalized = normalize_stream(raw, |usage| {
            Ok(StreamFinal::new(TEST_PROVIDER, usage).with_finish_reason(FinishReason::Stop))
        });

        let mut stream = StreamingCompletionResponse::stream(TEST_PROVIDER, normalized);
        while stream.next().await.is_some() {}

        assert_eq!(
            stream
                .response
                .as_ref()
                .and_then(|final_record| final_record.finish_reason.clone()),
            Some(FinishReason::ToolCalls),
        );
    }

    #[tokio::test]
    async fn normalize_stream_leaves_a_stop_without_tool_calls_alone() {
        let raw: RawStreamingResult<Usage> = Box::pin(stream! {
            yield Ok(RawStreamingChoice::Message("done".to_string()));
            yield Ok(RawStreamingChoice::FinalResponse(Usage::new()));
        });

        let normalized = normalize_stream(raw, |usage| {
            Ok(StreamFinal::new(TEST_PROVIDER, usage).with_finish_reason(FinishReason::Stop))
        });

        let mut stream = StreamingCompletionResponse::stream(TEST_PROVIDER, normalized);
        while stream.next().await.is_some() {}

        assert_eq!(
            stream
                .response
                .as_ref()
                .and_then(|final_record| final_record.finish_reason.clone()),
            Some(FinishReason::Stop),
        );
    }

    #[test]
    fn stream_final_round_trips_and_is_distinguishable_from_unknown_content() {
        let final_record = StreamFinal::new(
            "example",
            Usage {
                input_tokens: 4,
                output_tokens: 6,
                total_tokens: 10,
                cached_input_tokens: 1,
                cache_creation_input_tokens: 2,
                tool_use_prompt_tokens: 3,
                reasoning_tokens: 4,
            },
        )
        .with_finish_reason(FinishReason::Other("future_reason".to_owned()))
        .with_message_id("msg_123")
        .with_model("provider-model-v2");

        let encoded = serde_json::to_value(StreamedAssistantContent::Final(final_record.clone()))
            .expect("serialize final item");
        assert_eq!(encoded["kind"], serde_json::json!("final"));

        let decoded = serde_json::from_value::<StreamedAssistantContent>(encoded)
            .expect("deserialize final item");
        assert_eq!(decoded, StreamedAssistantContent::Final(final_record));

        // An unmodeled provider item must still land in `Unknown` rather than
        // being mistaken for a terminal record.
        let provider_item = serde_json::json!({
            "provider_native_event": "future_terminal",
            "usage": {"total_tokens": 10}
        });
        let decoded = serde_json::from_value::<StreamedAssistantContent>(provider_item.clone())
            .expect("deserialize unknown item");
        assert_eq!(decoded, StreamedAssistantContent::Unknown(provider_item));
    }

    /// Deserialization funnels through `new` + the setters, so the invariants
    /// hold on persisted values too: a `""` identifier comes back as `None`.
    #[test]
    fn deserializing_stream_final_filters_empty_identifiers() {
        let decoded = serde_json::from_value::<StreamFinal>(serde_json::json!({
            "kind": "final",
            "usage": Usage::new(),
            "message_id": "",
            "response_id": "",
            "model": "",
            "provider": "example",
        }))
        .expect("deserialize terminal record");

        assert_eq!(decoded.message_id, None);
        assert_eq!(decoded.response_id, None);
        assert_eq!(decoded.model, None);
    }

    /// The deserialization mirror must not change the wire format: a fully
    /// populated terminal record round-trips to byte-identical JSON.
    #[test]
    fn stream_final_serde_round_trip_is_identity() {
        let final_record = StreamFinal::new(
            "example",
            Usage {
                input_tokens: 4,
                output_tokens: 6,
                total_tokens: 10,
                cached_input_tokens: 1,
                cache_creation_input_tokens: 2,
                tool_use_prompt_tokens: 3,
                reasoning_tokens: 4,
            },
        )
        .with_finish_reason(FinishReason::Stop)
        .with_message_id("msg_123")
        .with_response_id("resp_456")
        .with_model("provider-model-v2");

        let encoded = serde_json::to_value(&final_record).expect("serialize terminal record");
        assert_eq!(encoded["kind"], serde_json::json!("final"));

        let decoded = serde_json::from_value::<StreamFinal>(encoded.clone()).expect("deserialize");
        assert_eq!(decoded, final_record);
        assert_eq!(
            serde_json::to_value(&decoded).expect("re-serialize"),
            encoded
        );
    }

    #[tokio::test]
    async fn usage_is_zero_sentinel_before_final_response() {
        // A stream that never yields a FinalResponse reports the zero sentinel.
        let stream = StreamingCompletionResponse::stream(
            TEST_PROVIDER,
            to_stream_result(stream! {
                yield Ok(RawStreamingChoice::Message("no final response".to_string()));
            }),
        );
        assert_eq!(stream.usage().total_tokens, 0);
    }

    #[tokio::test]
    async fn test_stream_cancellation() {
        let mut stream = create_mock_stream();

        println!("Response: ");
        let mut chunk_count = 0;
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(StreamedAssistantContent::Text(text)) => {
                    print!("{}", text.text);
                    std::io::Write::flush(&mut std::io::stdout()).unwrap();
                    chunk_count += 1;
                }
                Ok(StreamedAssistantContent::ToolCall {
                    tool_call,
                    internal_call_id,
                }) => {
                    println!("\nTool Call: {tool_call:?}, internal_call_id={internal_call_id:?}");
                    chunk_count += 1;
                }
                Ok(StreamedAssistantContent::ToolCallDelta {
                    id,
                    internal_call_id,
                    content,
                }) => {
                    println!(
                        "\nTool Call delta: id={id:?}, internal_call_id={internal_call_id:?}, content={content:?}"
                    );
                    chunk_count += 1;
                }
                Ok(StreamedAssistantContent::Final(res)) => {
                    println!("\nFinal response: {res:?}");
                }
                Ok(StreamedAssistantContent::Reasoning(reasoning)) => {
                    let reasoning = reasoning.display_text();
                    print!("{reasoning}");
                    std::io::Write::flush(&mut std::io::stdout()).unwrap();
                }
                Ok(StreamedAssistantContent::ReasoningDelta { reasoning, .. }) => {
                    println!("Reasoning delta: {reasoning}");
                    chunk_count += 1;
                }
                Ok(StreamedAssistantContent::Unknown(value)) => {
                    println!("\nUnknown item: {value:?}");
                    chunk_count += 1;
                }
                Err(e) => {
                    eprintln!("Error: {e:?}");
                    break;
                }
            }

            if chunk_count >= 2 {
                println!("\nCancelling stream...");
                stream.cancel();
                println!("Stream cancelled.");
                break;
            }
        }

        let next_chunk = stream.next().await;
        assert!(
            next_chunk.is_none(),
            "Expected no further chunks after cancellation, got {next_chunk:?}"
        );
    }

    #[tokio::test]
    async fn test_stream_pause_resume() {
        let stream = create_mock_stream();

        // Test pause
        stream.pause();
        assert!(stream.is_paused());

        // Test resume
        stream.resume();
        assert!(!stream.is_paused());
    }

    #[tokio::test]
    async fn test_stream_aggregates_reasoning_content() {
        let mut stream = create_reasoning_stream();
        while stream.next().await.is_some() {}

        let choice_items: Vec<AssistantContent> = stream.choice.clone().into_iter().collect();

        assert!(choice_items.iter().any(|item| matches!(
            item,
            AssistantContent::Reasoning(Reasoning {
                id: Some(id),
                content
            }) if id == "rs_1"
                && matches!(
                    content.first(),
                    Some(ReasoningContent::Text {
                        text,
                        signature: Some(signature)
                    }) if text == "step one" && signature == "sig_1"
                )
        )));
    }

    /// A full reasoning block replaces its own delta accumulation, so the
    /// aggregated choice matches unary normalization of the same turn: one
    /// reasoning item carrying the completed block, not delta-plus-duplicate.
    #[tokio::test]
    async fn full_reasoning_block_supersedes_its_accumulated_deltas() {
        let mut stream = StreamingCompletionResponse::stream(
            TEST_PROVIDER,
            to_stream_result(stream! {
                yield Ok(RawStreamingChoice::ReasoningDelta {
                    id: "rs_1".to_string(),
                    reasoning: "partial ".to_string(),
                });
                yield Ok(RawStreamingChoice::Reasoning {                    id: "rs_1".to_string(),
                    content: ReasoningContent::Text {
                        text: "the complete chain".to_string(),
                        signature: Some("sig_1".to_string()),
                    },
                });
                yield Ok(RawStreamingChoice::FinalResponse(mock_final_with_total_tokens(2)));
            }),
        );
        while stream.next().await.is_some() {}

        let choice_items: Vec<AssistantContent> = stream.choice.clone().into_iter().collect();
        let reasoning_items: Vec<&Reasoning> = choice_items
            .iter()
            .filter_map(|item| match item {
                AssistantContent::Reasoning(reasoning) => Some(reasoning),
                _ => None,
            })
            .collect();

        assert_eq!(reasoning_items.len(), 1, "got {choice_items:?}");
        let reasoning = reasoning_items.first().expect("one reasoning item");
        assert_eq!(reasoning.id.as_deref(), Some("rs_1"));
        assert!(matches!(
            reasoning.content.first(),
            Some(ReasoningContent::Text { text, signature: Some(signature) })
                if text == "the complete chain" && signature == "sig_1"
        ));
    }

    /// A full block whose ID differs from the accumulating item's ID is a
    /// distinct reasoning item and is appended, not a replacement.
    #[tokio::test]
    async fn full_reasoning_block_with_a_different_id_appends() {
        let mut stream = StreamingCompletionResponse::stream(
            TEST_PROVIDER,
            to_stream_result(stream! {
                yield Ok(RawStreamingChoice::ReasoningDelta {
                    id: "rs_1".to_string(),
                    reasoning: "first item deltas".to_string(),
                });
                yield Ok(RawStreamingChoice::Reasoning {                    id: "rs_2".to_string(),
                    content: ReasoningContent::Text {
                        text: "a different item".to_string(),
                        signature: None,
                    },
                });
                yield Ok(RawStreamingChoice::FinalResponse(mock_final_with_total_tokens(2)));
            }),
        );
        while stream.next().await.is_some() {}

        let choice_items: Vec<AssistantContent> = stream.choice.clone().into_iter().collect();
        let reasoning_ids: Vec<Option<&str>> = choice_items
            .iter()
            .filter_map(|item| match item {
                AssistantContent::Reasoning(reasoning) => Some(reasoning.id.as_deref()),
                _ => None,
            })
            .collect();

        assert_eq!(reasoning_ids, vec![Some("rs_1"), Some("rs_2")]);
    }

    #[tokio::test]
    async fn full_reasoning_block_supersedes_deltas_across_interleaved_output() {
        // Providers may emit the completed reasoning item after other output
        // (reasoning -> tool call -> completed block). The tool call clears
        // the active reasoning index, so replacement must fall back to the
        // by-ID scan rather than appending a duplicate.
        let mut stream = StreamingCompletionResponse::stream(
            TEST_PROVIDER,
            to_stream_result(stream! {
                yield Ok(RawStreamingChoice::ReasoningDelta {
                    id: "rs_1".to_string(),
                    reasoning: "partial ".to_string(),
                });
                yield Ok(RawStreamingChoice::ToolCall(RawStreamingToolCall::new(
                    "call_1".to_string(),
                    "probe".to_string(),
                    serde_json::json!({}),
                )));
                yield Ok(RawStreamingChoice::Reasoning {                    id: "rs_1".to_string(),
                    content: ReasoningContent::Text {
                        text: "the full block".to_string(),
                        signature: None,
                    },
                });
                yield Ok(RawStreamingChoice::FinalResponse(mock_final_with_total_tokens(2)));
            }),
        );
        while stream.next().await.is_some() {}

        let choice_items: Vec<AssistantContent> = stream.choice.clone().into_iter().collect();
        let reasoning_items: Vec<&Reasoning> = choice_items
            .iter()
            .filter_map(|item| match item {
                AssistantContent::Reasoning(reasoning) => Some(reasoning),
                _ => None,
            })
            .collect();

        assert_eq!(
            reasoning_items.len(),
            1,
            "the full block must replace the delta-built item, not join it"
        );
        let only = reasoning_items.first().expect("one reasoning item");
        assert_eq!(only.id.as_deref(), Some("rs_1"));
        assert!(
            only.content.iter().any(|content| matches!(
                content,
                ReasoningContent::Text { text, .. } if text == "the full block"
            )),
            "the surviving item must carry the full block's content"
        );
    }

    #[tokio::test]
    async fn minted_id_full_reasoning_block_does_not_clobber_a_wire_id_item() {
        // Ids are mandatory on the grammar; a provider-minted id (the
        // "reasoning-0"-style boundary fallback) is a distinct identity from
        // a wire-supplied one, so the block appends rather than overwriting
        // an unrelated item's deltas.
        let mut stream = StreamingCompletionResponse::stream(
            TEST_PROVIDER,
            to_stream_result(stream! {
                yield Ok(RawStreamingChoice::ReasoningDelta {
                    id: "rs_1".to_string(),
                    reasoning: "identified deltas".to_string(),
                });
                yield Ok(RawStreamingChoice::Reasoning {
                    id: "reasoning-0".to_string(),
                    content: ReasoningContent::Text {
                        text: "anonymous block".to_string(),
                        signature: None,
                    },
                });
                yield Ok(RawStreamingChoice::FinalResponse(mock_final_with_total_tokens(2)));
            }),
        );
        while stream.next().await.is_some() {}

        let choice_items: Vec<AssistantContent> = stream.choice.clone().into_iter().collect();
        let reasoning_ids: Vec<Option<&str>> = choice_items
            .iter()
            .filter_map(|item| match item {
                AssistantContent::Reasoning(reasoning) => Some(reasoning.id.as_deref()),
                _ => None,
            })
            .collect();

        assert_eq!(reasoning_ids, vec![Some("rs_1"), Some("reasoning-0")]);
    }

    #[tokio::test]
    async fn test_stream_reasoning_only_does_not_inject_empty_text() {
        let mut stream = create_reasoning_only_stream();
        while stream.next().await.is_some() {}

        let choice_items: Vec<AssistantContent> = stream.choice.clone().into_iter().collect();
        assert_eq!(choice_items.len(), 1);
        assert!(matches!(
            choice_items.first(),
            Some(AssistantContent::Reasoning(Reasoning { id: Some(id), .. })) if id == "rs_only"
        ));
    }

    #[tokio::test]
    async fn test_stream_aggregates_assistant_items_in_arrival_order() {
        let mut stream = create_interleaved_stream();
        while stream.next().await.is_some() {}

        let choice_items: Vec<AssistantContent> = stream.choice.clone().into_iter().collect();
        assert_eq!(choice_items.len(), 3);
        assert!(matches!(
            choice_items.first(),
            Some(AssistantContent::Reasoning(Reasoning { id: Some(id), .. })) if id == "rs_interleaved"
        ));
        assert!(matches!(
            choice_items.get(1),
            Some(AssistantContent::Text(Text { text, .. })) if text == "final-text"
        ));
        assert!(matches!(
            choice_items.get(2),
            Some(AssistantContent::ToolCall(ToolCall { id, .. })) if id == "tool_1"
        ));
    }

    #[tokio::test]
    async fn unknown_choice_reaches_consumer_but_not_aggregated_choice() {
        let unknown = serde_json::json!({
            "type": "web_search_call",
            "id": "ws_1",
            "status": "completed",
        });
        let yielded = unknown.clone();
        let stream = stream! {
            yield Ok(RawStreamingChoice::Unknown(yielded));
            yield Ok(RawStreamingChoice::Message("done".to_string()));
            yield Ok(RawStreamingChoice::FinalResponse(mock_final_with_total_tokens(1)));
        };
        let mut stream =
            StreamingCompletionResponse::stream(TEST_PROVIDER, to_stream_result(stream));

        let mut consumer_unknown = None;
        let mut consumer_text = String::new();
        while let Some(item) = stream.next().await {
            match item.expect("stream item should be Ok") {
                StreamedAssistantContent::Unknown(value) => consumer_unknown = Some(value),
                StreamedAssistantContent::Text(text) => consumer_text.push_str(&text.text),
                _ => {}
            }
        }

        // The consumer receives the unmodeled item verbatim ...
        assert_eq!(consumer_unknown.as_ref(), Some(&unknown));
        assert_eq!(consumer_text, "done");

        // ... but it is structurally absent from the aggregated assistant choice
        // (the sole source of persisted history): only the text item remains.
        let choice_items: Vec<AssistantContent> = stream.choice.clone().into_iter().collect();
        assert_eq!(choice_items.len(), 1);
        assert!(matches!(
            choice_items.first(),
            Some(AssistantContent::Text(Text { text, .. })) if text == "done"
        ));
    }

    #[tokio::test]
    async fn test_stream_keeps_non_contiguous_text_chunks_split_by_tool_call() {
        let mut stream = create_text_tool_text_stream();
        while stream.next().await.is_some() {}

        let choice_items: Vec<AssistantContent> = stream.choice.clone().into_iter().collect();
        assert_eq!(choice_items.len(), 3);
        assert!(matches!(
            choice_items.first(),
            Some(AssistantContent::Text(Text { text, .. })) if text == "first"
        ));
        assert!(matches!(
            choice_items.get(1),
            Some(AssistantContent::ToolCall(ToolCall { id, .. })) if id == "tool_split"
        ));
        assert!(matches!(
            choice_items.get(2),
            Some(AssistantContent::Text(Text { text, .. })) if text == "second"
        ));
    }

    #[tokio::test]
    async fn test_stream_preserves_text_additional_params() {
        let mut stream = create_text_metadata_stream();
        while stream.next().await.is_some() {}

        let choice_items: Vec<AssistantContent> = stream.choice.clone().into_iter().collect();
        assert_eq!(choice_items.len(), 2);

        let Some(AssistantContent::Text(Text {
            text,
            additional_params: Some(additional_params),
        })) = choice_items.first()
        else {
            panic!("expected first text item with metadata");
        };
        assert_eq!(text, "first");
        assert_eq!(
            additional_params["citations"]
                .as_array()
                .expect("citations should be an array")
                .len(),
            2
        );

        let Some(AssistantContent::Text(Text {
            text,
            additional_params: Some(additional_params),
        })) = choice_items.get(1)
        else {
            panic!("expected second text item with metadata");
        };
        assert_eq!(text, "second");
        assert_eq!(additional_params["block"], 2);
    }
}

/// Describes responses from a streamed provider response which is either text, a tool call or a final usage response.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum StreamedAssistantContent {
    /// Text delta emitted by the assistant.
    Text(Text),
    /// Complete tool call emitted by the assistant.
    ToolCall {
        tool_call: ToolCall,
        /// Rig-generated unique identifier for this tool call.
        /// Use this to correlate with ToolCallDelta events.
        internal_call_id: String,
    },
    /// Partial tool call data emitted by the assistant.
    ToolCallDelta {
        /// Provider-supplied tool call ID.
        id: String,
        /// Rig-generated unique identifier for this tool call.
        internal_call_id: String,
        content: ToolCallDeltaContent,
    },
    /// Complete reasoning block emitted by the assistant.
    ///
    /// Supersedes any prior [`StreamedAssistantContent::ReasoningDelta`]s
    /// carrying the same reasoning `id`: render it as a *replacement* for
    /// the accumulated delta text, not an addition. The aggregated
    /// [`StreamingCompletionResponse::choice`] already applies this
    /// replacement.
    Reasoning(Reasoning),
    /// Partial reasoning text emitted by the assistant.
    ReasoningDelta {
        /// Identity of the reasoning item this delta extends. Always
        /// populated: providers propagate the wire's item identity or mint a
        /// stream-stable id at the boundary, so consumers can correlate
        /// deltas with the full [`StreamedAssistantContent::Reasoning`]
        /// block that supersedes them.
        id: String,
        /// Partial reasoning text.
        reasoning: String,
    },
    /// The provider's normalized terminal record, if yielded by the stream.
    Final(StreamFinal),
    /// A provider-native output item rig does not model, preserved verbatim —
    /// e.g. an OpenAI Responses hosted-tool result (`web_search_call`,
    /// `file_search_call`, `computer_call`, `code_interpreter_call`). It is
    /// yielded to the consumer for inspection/forwarding but is not added to the
    /// accumulated assistant message or persisted history. Kept last because the
    /// enum is `#[serde(untagged)]` and a raw [`Value`](serde_json::Value)
    /// matches anything, so earlier (typed) variants must be tried first.
    Unknown(serde_json::Value),
}

impl StreamedAssistantContent {
    /// Create a text stream item.
    pub fn text(text: &str) -> Self {
        Self::Text(Text::new(text.to_string()))
    }

    /// Create a final response stream item.
    pub fn final_response(res: StreamFinal) -> Self {
        Self::Final(res)
    }
}

/// Streamed user content. This content is primarily used to represent tool results from tool calls made during a multi-turn/step agent prompt.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum StreamedUserContent {
    /// Tool result emitted during a multi-turn streaming agent loop.
    ToolResult {
        tool_result: ToolResult,
        /// Rig-generated unique identifier for the tool call this result
        /// belongs to. Use this to correlate with the originating
        /// [`StreamedAssistantContent::ToolCall::internal_call_id`].
        internal_call_id: String,
    },
}

impl StreamedUserContent {
    /// Create a streamed tool result correlated to an internal tool call ID.
    pub fn tool_result(tool_result: ToolResult, internal_call_id: String) -> Self {
        Self::ToolResult {
            tool_result,
            internal_call_id,
        }
    }
}
