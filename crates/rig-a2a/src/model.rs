//! A remote A2A agent as a Rig [`CompletionModel`].
//!
//! [`A2AModel`] lets a remote agent back a Rig [`Agent`], so `prompt`, `chat`,
//! streaming, hooks, and conversation memory all work against it, and
//! [`Agent::into_tool`] projects it as a sub-agent.
//!
//! # What A2A cannot carry
//!
//! A [`CompletionRequest`] describes a chat-completion API. A2A is a different
//! contract: it exchanges messages with an agent that owns its own instructions
//! and hides its own tools — the specification's phrasing is that agents
//! collaborate "without needing to share their internal thoughts, plans, or
//! tool implementations". Several request fields therefore have no A2A
//! equivalent, and this model refuses rather than ignoring the ones whose
//! absence would silently change the result:
//!
//! | Request field | Behavior |
//! | --- | --- |
//! | `tools` | **Error.** A remote agent never emits a tool call, so registering local tools on an A2A-backed agent would advertise tools that can never be invoked. |
//! | `output_schema` | **Error.** A2A cannot constrain the remote's output, so an [`Extractor`] over this model would silently return unvalidated text. |
//! | `tool_choice` | **Error** when it demands a tool; ignored when it merely permits one. |
//! | `preamble` | Sent as a leading text part. The remote already has its own instructions and is free to disregard it. |
//! | `documents` | Sent as one text part each, rendered by [`Document`]'s `Display`. |
//! | `additional_params` | Reads only the [`THREAD_PARAMS_KEY`] threading block; ignores other fields (logged at debug). |
//! | `temperature`, `max_tokens`, `model` | Ignored (logged at debug). Sampling belongs to the remote agent. |
//!
//! `usage` is reported as zero, which is Rig's documented sentinel for "the
//! provider supplied no metrics" — A2A has no token accounting.
//!
//! # Conversation threading
//!
//! Three ways to thread, in increasing precedence:
//!
//! 1. **Bind the model.** [`A2AClient::model_for_conversation`] fixes one
//!    conversation for the model's lifetime.
//! 2. **Name it on a builder or run.** [`A2AConversationExt`] sets both Rig's
//!    memory key and the A2A conversation request parameter. Binding the
//!    builder also works before [`Agent::into_tool`].
//! 3. **Supply the ids.** Put the server-issued `contextId` — and a paused
//!    task's `taskId` — under the [`THREAD_PARAMS_KEY`] block of
//!    `additional_params` to resume a conversation this process never opened,
//!    or one restored from your own storage.
//!
//! The first two look the conversation up in the client's store; the third
//! bypasses it. A model that sees none of them is single-turn.
//!
//! Threaded models send only the newest prompt, because the remote agent holds
//! the history for that `contextId`; an unthreaded model renders the whole chat
//! history into the outbound message so a single-shot call still carries its
//! context.
//!
//! The preamble and the context documents get the same treatment. Rig rebuilds
//! both on every turn — a stateless provider sees only the current request —
//! but a threaded remote keeps everything sent under one `contextId`, so a
//! later turn carries only the chunks that remote has not already been given.
//! Documents are tracked individually, so per-turn retrieval still delivers
//! each newly retrieved document exactly once.
//!
//! [`Agent`]: rig_agent::agent::Agent
//! [`Agent::into_tool`]: rig_agent::agent::Agent::into_tool
//! [`Document`]: rig_core::completion::Document
//! [`A2AClient::agent`]: crate::A2AClient::agent
//! [`A2AClient::model_for_conversation`]: crate::A2AClient::model_for_conversation
//! [`A2AConversationExt`]: crate::A2AConversationExt
//! [`Extractor`]: rig_agent::extractor::Extractor

use std::sync::Arc;

use a2a::{Message as A2AMessage, Part, Role, SendMessageRequest, SendMessageResponse};
use a2a_client::{A2AClient as InnerClient, Transport};
use futures::StreamExt;
use rig_core::completion::{
    AssistantContent, CompletionError, CompletionModel, CompletionRequest, CompletionResponse,
    FinishReason, Usage,
};
use rig_core::message::{Message as RigMessage, ToolChoice, UserContent};
use rig_core::streaming::{
    RawStreamingChoice, StreamFinal, StreamingCompletionResponse, StreamingResult,
};

use crate::error::A2AError;
use crate::parts::{
    DEFAULT_TEXT_LIMIT, message_body_limited, state_label, status_text_limited, task_body_limited,
};
use crate::thread::{A2AThreadInfo, ConversationId, ThreadStore};

/// Provider name reported on responses and streams from this model.
pub const PROVIDER: &str = "a2a";

/// A remote A2A agent, usable wherever Rig expects a [`CompletionModel`].
///
/// Build one with [`A2AClient::model`](crate::A2AClient::model) or
/// [`A2AClient::model_for_conversation`](crate::A2AClient::model_for_conversation).
#[derive(Clone)]
pub struct A2AModel {
    inner: Arc<InnerClient<Box<dyn Transport>>>,
    tenant: Option<String>,
    threads: ThreadStore,
    conversation: Option<ConversationId>,
    agent_name: String,
}

impl std::fmt::Debug for A2AModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("A2AModel")
            .field("agent_name", &self.agent_name)
            .field("conversation", &self.conversation)
            .finish_non_exhaustive()
    }
}

impl A2AModel {
    pub(crate) fn new(
        inner: Arc<InnerClient<Box<dyn Transport>>>,
        tenant: Option<String>,
        threads: ThreadStore,
        conversation: Option<ConversationId>,
        agent_name: String,
    ) -> Self {
        Self {
            inner,
            tenant,
            threads,
            conversation,
            agent_name,
        }
    }

    /// The remote agent's card name.
    pub fn agent_name(&self) -> &str {
        &self.agent_name
    }

    /// The conversation this model threads into, if any.
    pub fn conversation(&self) -> Option<&ConversationId> {
        self.conversation.as_ref()
    }

    /// Bind (or rebind) this model to a conversation.
    ///
    /// Conversation state lives in the shared store on the originating
    /// [`A2AClient`](crate::A2AClient), so models using the same id continue the
    /// same remote conversation.
    pub fn with_conversation(mut self, id: impl Into<ConversationId>) -> Self {
        self.conversation = Some(id.into());
        self
    }

    /// Build the outbound A2A request for a Rig completion request.
    fn build_request(&self, request: CompletionRequest) -> Result<Outbound, A2AError> {
        reject_unsupported(&request)?;
        log_ignored(&request);

        let params = thread_params(request.additional_params.as_ref())?;
        // A conversation named on the run outranks the model's own binding, so
        // one agent serves many conversations instead of one agent per
        // conversation. `A2AConversationExt` puts the run's id here.
        let conversation = params
            .conversation
            .map(ConversationId::new)
            .or_else(|| self.conversation.clone());

        let mut thread = conversation
            .as_ref()
            .map(|id| self.threads.get(id))
            .unwrap_or_default();
        // Ids the caller threaded onto this request outrank the store: an
        // explicit caller owns the conversation and may be resuming one this
        // process never saw.
        if let Some(context_id) = params.context_id {
            thread.context_id = Some(context_id);
            thread.task_id = params.task_id;
        }
        // A threaded remote already holds everything sent under this
        // `contextId`; replaying it would duplicate it on the remote side. An
        // unthreaded call has no server-side memory, so the whole request is
        // rendered into this one message.
        let threaded = thread.context_id.is_some();

        let mut parts = Vec::new();
        let mut context_fingerprints = Vec::new();

        // The preamble and the context documents are *standing* context: Rig
        // rebuilds them on every turn for a stateless provider, but a thread
        // needs each chunk only once. Documents are fingerprinted one by one
        // rather than as a block, so a turn that retrieves one new document
        // sends that document alone.
        let mut push_standing_context = |text: String| {
            if text.trim().is_empty() {
                return;
            }
            let fingerprint = fingerprint(&text);
            if threaded && thread.sent_context.contains(&fingerprint) {
                return;
            }
            context_fingerprints.push(fingerprint);
            parts.push(Part::text(text));
        };

        if let Some(preamble) = request.preamble.as_ref() {
            push_standing_context(preamble.clone());
        }
        for document in &request.documents {
            push_standing_context(document.to_string());
        }

        let history = request.chat_history.into_iter().collect::<Vec<_>>();
        let messages: &[RigMessage] = if threaded {
            history.last().map(std::slice::from_ref).unwrap_or_default()
        } else {
            &history
        };
        for message in messages {
            push_message_text(&mut parts, message);
        }

        if parts.is_empty() {
            return Err(A2AError::EmptyRequest);
        }

        let mut message = A2AMessage::new(Role::User, parts);
        message.context_id = thread.context_id;
        message.task_id = thread.task_id;
        Ok(Outbound {
            request: SendMessageRequest {
                message,
                configuration: None,
                metadata: None,
                tenant: self.tenant.clone(),
            },
            context_fingerprints,
            conversation,
        })
    }

    /// Record the identifiers a response carried against the conversation this
    /// request threaded into, if any.
    fn record(&self, conversation: Option<&ConversationId>, info: &A2AThreadInfo) {
        if let Some(conversation) = conversation {
            self.threads.record(conversation, info);
        }
    }

    /// Record the standing context a delivered request carried, so later turns
    /// in the same conversation do not repeat it.
    fn record_sent_context(&self, conversation: Option<&ConversationId>, fingerprints: &[u64]) {
        if let Some(conversation) = conversation {
            self.threads.record_sent_context(conversation, fingerprints);
        }
    }
}

/// An outbound request together with the standing context it carries.
///
/// The fingerprints are committed to the conversation only once the remote has
/// answered, so a request that never landed offers its context again.
struct Outbound {
    request: SendMessageRequest,
    context_fingerprints: Vec<u64>,
    /// Conversation this request threads into, after per-run overrides.
    conversation: Option<ConversationId>,
}

/// Key under which a caller threads A2A identifiers onto one request through
/// [`CompletionRequest::additional_params`].
///
/// ```no_run
/// # use rig_agent::completion::Prompt;
/// # use serde_json::json;
/// # async fn run(agent: rig_agent::agent::Agent, context_id: String) -> anyhow::Result<()> {
/// let reply = agent
///     .prompt("turn 2")
///     .merge_additional_params(
///         json!({ "a2a": { "context_id": context_id } })
///             .as_object()
///             .cloned()
///             .unwrap_or_default(),
///     )
///     .await?;
/// # let _ = reply;
/// # Ok(()) }
/// ```
pub const THREAD_PARAMS_KEY: &str = "a2a";

/// A2A threading a caller attached to one request.
#[derive(Debug, Default, PartialEq, Eq)]
struct ThreadParams {
    /// Conversation whose tracked identifiers this request continues.
    conversation: Option<String>,
    context_id: Option<String>,
    task_id: Option<String>,
}

/// Read the identifiers a caller threaded onto this request.
///
/// A caller who owns the conversation — because they persisted the ids, or
/// resumed one this process never opened — supplies them here instead of
/// binding the model to a [`ConversationId`]. Malformed directives fail the
/// request rather than being dropped: a typo that silently opened a fresh
/// remote conversation would read as the remote losing its memory.
fn thread_params(additional_params: Option<&serde_json::Value>) -> Result<ThreadParams, A2AError> {
    let Some(params) = additional_params.and_then(|params| params.get(THREAD_PARAMS_KEY)) else {
        return Ok(ThreadParams::default());
    };
    let Some(fields) = params.as_object() else {
        return Err(A2AError::InvalidThreadParams {
            detail: "must be an object with optional `context_id` and `task_id` string fields",
        });
    };
    if fields
        .keys()
        .any(|key| !matches!(key.as_str(), "conversation" | "context_id" | "task_id"))
    {
        return Err(A2AError::InvalidThreadParams {
            detail: "accepts only the `conversation`, `context_id` and `task_id` fields",
        });
    }

    let conversation = id_field(
        fields,
        "conversation",
        A2AError::InvalidThreadParams {
            detail: "`conversation` must be a non-empty string",
        },
    )?;
    let context_id = id_field(fields, "context_id", A2AError::InvalidContextId)?;
    let task_id = id_field(fields, "task_id", A2AError::InvalidTaskId)?;
    // The two are different claims — "look up what you tracked" versus "use
    // these ids" — and honoring one would silently ignore the other.
    if conversation.is_some() && (context_id.is_some() || task_id.is_some()) {
        return Err(A2AError::InvalidThreadParams {
            detail: "sets `conversation` together with explicit ids; supply one or the other",
        });
    }
    // A task lives inside a context. Accepting a lone taskId would send a
    // request the caller believes is threaded while the history replay that
    // keys off the contextId still fires.
    if task_id.is_some() && context_id.is_none() {
        return Err(A2AError::InvalidThreadParams {
            detail: "a `task_id` must be accompanied by the `context_id` that owns it",
        });
    }
    Ok(ThreadParams {
        conversation,
        context_id,
        task_id,
    })
}

fn id_field(
    fields: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    invalid: A2AError,
) -> Result<Option<String>, A2AError> {
    match fields.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(id)) if !id.trim().is_empty() => Ok(Some(id.clone())),
        Some(_) => Err(invalid),
    }
}

/// Identify a chunk of standing context by content.
///
/// Content rather than position or document id: Rig merges static context with
/// whatever passive or active retrieval produced this turn, and only the text
/// itself says whether the remote has seen it.
fn fingerprint(text: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

impl CompletionModel for A2AModel {
    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, CompletionError> {
        let outbound = self.build_request(request)?;
        let response = self
            .inner
            .send_message(&outbound.request)
            .await
            .map_err(A2AError::Protocol)?;

        let info = A2AThreadInfo::from_response(&response);
        let conversation = outbound.conversation.as_ref();
        self.record(conversation, &info);
        self.record_sent_context(conversation, &outbound.context_fingerprints);

        let (text, message_id) = match &response {
            SendMessageResponse::Task(task) => {
                if let Some(error) = task_failure(&response) {
                    return Err(error);
                }
                let body = task_body_limited(task, DEFAULT_TEXT_LIMIT)?;
                (task_text(&task.status.state, body)?, None)
            }
            SendMessageResponse::Message(message) => {
                let body = message_body_limited(message, DEFAULT_TEXT_LIMIT)?;
                if body.is_empty() {
                    return Err(CompletionError::ResponseError(
                        "remote A2A agent replied with a message carrying no text".to_string(),
                    ));
                }
                (body, Some(message.message_id.clone()))
            }
        };

        Ok(
            CompletionResponse::new(text_choice(text), Usage::new(), PROVIDER)
                .with_model(self.agent_name.clone())
                .with_optional_message_id(message_id)
                // Response-scoped, so the task id rather than the conversation's
                // `contextId`, which is identical on every turn of a thread.
                .with_optional_response_id(info.task_id.clone())
                .with_finish_reason(finish_reason(&info)),
        )
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse, CompletionError> {
        let outbound = self.build_request(request)?;
        let upstream = self
            .inner
            .send_streaming_message(&outbound.request)
            .await
            .map_err(A2AError::Protocol)?;

        // The request is on the wire by the time the transport hands back a
        // stream, so its standing context has reached the remote whatever the
        // stream goes on to yield.
        self.record_sent_context(
            outbound.conversation.as_ref(),
            &outbound.context_fingerprints,
        );

        let threads = self.threads.clone();
        let conversation = outbound.conversation;
        let model = self.agent_name.clone();

        let stream: StreamingResult = Box::pin(async_stream_events(upstream, model, move |info| {
            if let Some(conversation) = &conversation {
                threads.record(conversation, info);
            }
        }));

        Ok(StreamingCompletionResponse::stream(PROVIDER, stream))
    }
}

/// Translate an A2A event stream into Rig streaming events.
///
/// Text arrives as artifact deltas and status messages; the terminal record is
/// emitted when the upstream stream ends, carrying the finish reason implied by
/// the last observed task state. A failure state ends the stream with an error
/// rather than a terminal record, matching the non-streaming surface.
fn async_stream_events(
    upstream: futures::stream::BoxStream<'static, Result<a2a::StreamResponse, a2a::A2AError>>,
    model: String,
    mut record: impl FnMut(&A2AThreadInfo) + Send + 'static,
) -> impl futures::Stream<Item = Result<RawStreamingChoice<StreamFinal>, CompletionError>> + Send {
    async_stream::stream! {
        let mut upstream = upstream;
        let mut last: Option<A2AThreadInfo> = None;

        while let Some(event) = upstream.next().await {
            let event = match event {
                Ok(event) => event,
                Err(error) => {
                    yield Err(CompletionError::from(A2AError::Protocol(error)));
                    return;
                }
            };

            match event {
                a2a::StreamResponse::Task(task) => {
                    let info = A2AThreadInfo::for_task(
                        &task.context_id,
                        &task.id,
                        &task.status.state,
                    );
                    record(&info);
                    last = Some(info);

                    if let Some(error) = failure_for(&task.status.state, || {
                        status_text_limited(&task, DEFAULT_TEXT_LIMIT)
                    }) {
                        yield Err(error);
                        return;
                    }
                    match task_body_limited(&task, DEFAULT_TEXT_LIMIT) {
                        Ok(text) if !text.is_empty() => {
                            yield Ok(RawStreamingChoice::Message(text));
                        }
                        Ok(_) => {}
                        Err(error) => {
                            yield Err(CompletionError::from(error));
                            return;
                        }
                    }
                }
                a2a::StreamResponse::Message(message) => {
                    let info = A2AThreadInfo {
                        context_id: message.context_id.clone(),
                        task_id: message.task_id.clone(),
                        state: None,
                        resumable: false,
                    };
                    record(&info);
                    last = Some(info);
                    match message_body_limited(&message, DEFAULT_TEXT_LIMIT) {
                        Ok(text) if !text.is_empty() => {
                            yield Ok(RawStreamingChoice::Message(text));
                        }
                        Ok(_) => {}
                        Err(error) => {
                            yield Err(CompletionError::from(error));
                            return;
                        }
                    }
                }
                a2a::StreamResponse::StatusUpdate(update) => {
                    let info = A2AThreadInfo::for_task(
                        &update.context_id,
                        &update.task_id,
                        &update.status.state,
                    );
                    record(&info);
                    last = Some(info);

                    // A status update's message is where many A2A agents put
                    // their answer, so project it once and use it both to
                    // explain a failure and as streamed text.
                    let status = match update.status.message.as_ref() {
                        Some(message) => match crate::parts::parts_to_text_limited(
                            &message.parts,
                            DEFAULT_TEXT_LIMIT,
                            "stream status",
                        ) {
                            Ok(text) => text,
                            Err(error) => {
                                yield Err(CompletionError::from(error));
                                return;
                            }
                        },
                        None => String::new(),
                    };

                    if let Some(error) =
                        failure_for(&update.status.state, || Ok(status.clone()))
                    {
                        yield Err(error);
                        return;
                    }
                    if !status.is_empty() {
                        yield Ok(RawStreamingChoice::Message(status));
                    }
                }
                a2a::StreamResponse::ArtifactUpdate(update) => {
                    // An artifact belongs to a task without reporting its state,
                    // so it must not disturb the resumability the last status
                    // update established — clearing a paused task's id here
                    // would strand the conversation.
                    let info = A2AThreadInfo {
                        context_id: Some(update.context_id.clone()),
                        task_id: Some(update.task_id.clone()),
                        state: last.as_ref().and_then(|info| info.state.clone()),
                        resumable: last.as_ref().is_some_and(|info| info.resumable),
                    };
                    record(&info);
                    match crate::parts::parts_to_text_limited(
                        &update.artifact.parts,
                        DEFAULT_TEXT_LIMIT,
                        "stream artifact",
                    ) {
                        Ok(text) if !text.is_empty() => {
                            yield Ok(RawStreamingChoice::Message(text));
                        }
                        Ok(_) => {}
                        Err(error) => {
                            yield Err(CompletionError::from(error));
                            return;
                        }
                    }
                }
            }
        }

        let final_record = StreamFinal::new(PROVIDER, Usage::new())
            .with_model(model)
            .with_optional_response_id(last.as_ref().and_then(|info| info.task_id.clone()))
            .with_finish_reason(last.as_ref().map_or(FinishReason::Stop, finish_reason));
        yield Ok(RawStreamingChoice::FinalResponse(final_record));
    }
}

/// Refuse request fields whose absence would silently change the result.
fn reject_unsupported(request: &CompletionRequest) -> Result<(), A2AError> {
    if !request.tools.is_empty() {
        return Err(A2AError::Unsupported {
            what: "tools",
            detail: "an A2A agent runs its own tools and never returns a tool call, so tools registered on this agent could never be invoked",
        });
    }
    if request.output_schema.is_some() {
        return Err(A2AError::Unsupported {
            what: "output_schema",
            detail: "A2A cannot constrain a remote agent's output, so structured extraction would return unvalidated text",
        });
    }
    if matches!(
        request.tool_choice,
        Some(ToolChoice::Required) | Some(ToolChoice::Specific { .. })
    ) {
        return Err(A2AError::Unsupported {
            what: "tool_choice",
            detail: "an A2A agent cannot be made to call a Rig tool",
        });
    }
    Ok(())
}

/// Note the sampling knobs the remote agent owns instead of the caller.
///
/// The `a2a` block of `additional_params` is threading, not sampling, and is
/// read by [`thread_params`] — only the rest of that object is ignored.
fn log_ignored(request: &CompletionRequest) {
    let ignored_params = request.additional_params.as_ref().is_some_and(|params| {
        params
            .as_object()
            .is_none_or(|fields| fields.keys().any(|key| key != THREAD_PARAMS_KEY))
    });
    if request.temperature.is_some()
        || request.max_tokens.is_some()
        || ignored_params
        || request.model.is_some()
    {
        tracing::debug!(
            target: "rig_a2a",
            "sampling parameters are ignored: an A2A agent owns its own decoding settings"
        );
    }
}

/// Append a Rig message's text to the outbound parts.
///
/// Prefixes assistant turns so a rendered history reads as a dialogue rather
/// than one undifferentiated block. Non-text content has no A2A projection and
/// is skipped.
fn push_message_text(parts: &mut Vec<Part>, message: &RigMessage) {
    let (prefix, text) = match message {
        RigMessage::User { content } => ("", user_text(content)),
        RigMessage::Assistant { content, .. } => ("assistant: ", assistant_text(content)),
        RigMessage::System { content, .. } => ("", content.clone()),
    };
    if text.trim().is_empty() {
        return;
    }
    parts.push(Part::text(format!("{prefix}{text}")));
}

fn user_text(content: &[UserContent]) -> String {
    content
        .iter()
        .filter_map(|item| match item {
            UserContent::Text(text) => Some(text.text.clone()),
            UserContent::ToolResult(result) => Some(
                result
                    .content
                    .iter()
                    .filter_map(|content| match content {
                        rig_core::message::ToolResultContent::Text(text) => Some(text.text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
            ),
            other => {
                log_dropped_content(content_label(other));
                None
            }
        })
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn assistant_text(content: &[AssistantContent]) -> String {
    content
        .iter()
        .filter_map(|item| match item {
            AssistantContent::Text(text) => Some(text.text.clone()),
            AssistantContent::ToolCall(_) => None,
            _ => {
                log_dropped_content("assistant");
                None
            }
        })
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn content_label(content: &UserContent) -> &'static str {
    match content {
        UserContent::Image(_) => "image",
        UserContent::Audio(_) => "audio",
        UserContent::Document(_) => "document",
        _ => "user",
    }
}

/// Non-text history has no A2A projection. Say so rather than dropping it
/// silently, mirroring the binary-part log in [`crate::parts`].
fn log_dropped_content(kind: &'static str) {
    tracing::debug!(
        target: "rig_a2a",
        kind,
        "skipping non-text message content while projecting to an A2A message"
    );
}

fn text_choice(text: String) -> Vec<AssistantContent> {
    vec![AssistantContent::text(text)]
}

/// A task's assistant text, or the error a content-free reply deserves.
///
/// Rig's providers normalize a *legitimate* empty turn to empty text — the
/// Anthropic provider does this for a documented empty `end_turn` — and report
/// any other content-free response as a [`CompletionError::ResponseError`].
/// A2A's legitimate empty turn is a completed task with no artifacts and no
/// closing status message: a task whose result was an action rather than text.
/// Any other state answering with nothing has neither produced a result nor
/// finished, so the caller has nothing to act on and is told so.
///
fn task_text(state: &a2a::TaskState, body: String) -> Result<String, CompletionError> {
    if !body.is_empty() {
        return Ok(body);
    }
    if matches!(state, a2a::TaskState::Completed) {
        tracing::debug!(
            target: "rig_a2a",
            "remote A2A task completed with no artifacts and no status message; reporting an empty assistant turn"
        );
        return Ok(body);
    }
    Err(CompletionError::ResponseError(format!(
        "remote A2A agent returned no content and its task is {}",
        state_label(state)
    )))
}

/// A2A has no stop-reason concept; a paused task is the one case that is not a
/// plain stop, and it maps to "the model wants more input".
fn finish_reason(info: &A2AThreadInfo) -> FinishReason {
    match info.state {
        Some(a2a::TaskState::InputRequired) => FinishReason::Other("input_required".to_string()),
        _ => FinishReason::Stop,
    }
}

fn task_failure(response: &SendMessageResponse) -> Option<CompletionError> {
    let SendMessageResponse::Task(task) = response else {
        return None;
    };
    failure_for(&task.status.state, || {
        status_text_limited(task, DEFAULT_TEXT_LIMIT)
    })
}

/// Map a task state the caller cannot act on to a completion error.
///
/// A completion model has one provider-error channel; the state name is kept
/// in the message so callers can still distinguish task states.
fn failure_for(
    state: &a2a::TaskState,
    status: impl FnOnce() -> Result<String, A2AError>,
) -> Option<CompletionError> {
    use a2a::TaskState::{AuthRequired, Canceled, Failed, Rejected};
    if !matches!(state, Failed | Rejected | Canceled | AuthRequired) {
        return None;
    }
    Some(match status() {
        Ok(text) => {
            let text = if text.is_empty() {
                "no status message".to_string()
            } else {
                text
            };
            CompletionError::ProviderError(format!(
                "remote A2A agent {}: {text}",
                state_label(state)
            ))
        }
        Err(error) => CompletionError::from(error),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig_core::completion::CompletionRequestBuilder;

    fn request() -> CompletionRequest {
        CompletionRequest {
            model: None,
            preamble: None,
            chat_history: vec![RigMessage::user("hello")],
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        }
    }

    #[test]
    fn tools_are_rejected_rather_than_ignored() {
        let mut req = request();
        req.tools.push(rig_core::completion::ToolDefinition {
            name: "add".to_string(),
            description: "adds".to_string(),
            parameters: serde_json::json!({}),
        });
        let error = reject_unsupported(&req).expect_err("tools must be refused");
        assert!(error.to_string().contains("tools"), "{error}");
    }

    #[test]
    fn output_schema_is_rejected() {
        let mut req = request();
        req.output_schema = Some(schemars::json_schema!({"type": "object"}));
        let error = reject_unsupported(&req).expect_err("output_schema must be refused");
        assert!(error.to_string().contains("output_schema"), "{error}");
    }

    #[test]
    fn demanding_tool_choice_is_rejected_but_permissive_is_not() {
        let mut req = request();
        req.tool_choice = Some(ToolChoice::Required);
        assert!(reject_unsupported(&req).is_err());

        req.tool_choice = Some(ToolChoice::Auto);
        assert!(reject_unsupported(&req).is_ok());

        req.tool_choice = Some(ToolChoice::None);
        assert!(reject_unsupported(&req).is_ok());
    }

    #[test]
    fn thread_params_are_read_off_additional_params() {
        let params = serde_json::json!({"a2a": {"context_id": "ctx-1", "task_id": "task-1"}});
        assert_eq!(
            thread_params(Some(&params)).expect("well-formed directive"),
            ThreadParams {
                conversation: None,
                context_id: Some("ctx-1".to_string()),
                task_id: Some("task-1".to_string()),
            }
        );

        let params = serde_json::json!({"a2a": {"conversation": "user-42"}});
        assert_eq!(
            thread_params(Some(&params)).expect("well-formed directive"),
            ThreadParams {
                conversation: Some("user-42".to_string()),
                ..ThreadParams::default()
            }
        );

        assert_eq!(
            thread_params(None).expect("absent directive"),
            ThreadParams::default()
        );
        let unrelated = serde_json::json!({"temperature": 0.5});
        assert_eq!(
            thread_params(Some(&unrelated)).expect("unrelated params"),
            ThreadParams::default()
        );
    }

    #[test]
    fn malformed_thread_params_fail_rather_than_opening_a_new_conversation() {
        for params in [
            serde_json::json!({"a2a": "ctx-1"}),
            serde_json::json!({"a2a": {"contextId": "ctx-1"}}),
            serde_json::json!({"a2a": {"context_id": ""}}),
            serde_json::json!({"a2a": {"context_id": 7}}),
            // A task id cannot stand on its own: history replay keys off the
            // context id, so this would silently send an unthreaded request.
            serde_json::json!({"a2a": {"task_id": "task-1"}}),
            // Two different claims; honoring one would ignore the other.
            serde_json::json!({"a2a": {"conversation": "c", "context_id": "ctx-1"}}),
        ] {
            assert!(
                thread_params(Some(&params)).is_err(),
                "must be rejected: {params}"
            );
        }
    }

    #[test]
    fn sampling_parameters_do_not_fail_the_request() {
        let mut req = request();
        req.temperature = Some(0.9);
        req.max_tokens = Some(128);
        assert!(reject_unsupported(&req).is_ok());
    }

    #[test]
    fn history_renders_with_assistant_turns_prefixed() {
        let mut parts = Vec::new();
        push_message_text(&mut parts, &RigMessage::user("hi"));
        push_message_text(&mut parts, &RigMessage::assistant("hello there"));
        push_message_text(&mut parts, &RigMessage::user("and again"));

        let rendered: Vec<String> = parts
            .iter()
            .filter_map(|part| match &part.content {
                a2a::PartContent::Text(text) => Some(text.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(rendered, ["hi", "assistant: hello there", "and again"]);
    }

    #[test]
    fn empty_messages_are_skipped() {
        let mut parts = Vec::new();
        push_message_text(&mut parts, &RigMessage::user("   "));
        assert!(parts.is_empty());
    }

    #[test]
    fn input_required_reports_a_distinct_finish_reason() {
        let info = A2AThreadInfo {
            context_id: None,
            task_id: None,
            state: Some(a2a::TaskState::InputRequired),
            resumable: true,
        };
        assert_eq!(
            finish_reason(&info),
            FinishReason::Other("input_required".to_string())
        );

        let info = A2AThreadInfo {
            state: Some(a2a::TaskState::Completed),
            ..info
        };
        assert_eq!(finish_reason(&info), FinishReason::Stop);
    }

    #[test]
    fn failure_states_become_provider_errors_naming_the_state() {
        for state in [
            a2a::TaskState::Failed,
            a2a::TaskState::Rejected,
            a2a::TaskState::Canceled,
            a2a::TaskState::AuthRequired,
        ] {
            let error = failure_for(&state, || Ok("quota exceeded".to_string()))
                .expect("state must fail the completion");
            let text = error.to_string();
            assert!(text.contains(state_label(&state)), "{text}");
            assert!(text.contains("quota exceeded"), "{text}");
        }
        assert!(failure_for(&a2a::TaskState::Completed, || Ok(String::new())).is_none());
        assert!(failure_for(&a2a::TaskState::InputRequired, || Ok(String::new())).is_none());
    }

    /// `completion_request` gates on `Self: Clone`, and agent construction
    /// erases through `CompletionModel + 'static`; keep both paths compiling.
    #[allow(dead_code)]
    fn model_satisfies_the_rig_model_bounds(model: A2AModel) -> CompletionRequestBuilder<A2AModel> {
        fn erasable<M: CompletionModel + 'static>(_: &M) {}
        erasable(&model);
        model.completion_request("hi")
    }
}
