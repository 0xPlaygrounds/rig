//! Streamed agent events, bounded event feeds, and terminal response shaping.
//!
//! ```no_run
//! # fn example(agent: &rig_agent::Agent) {
//! let stream = agent.prompt("Explain ownership.").stream();
//! # }
//! ```

use rig_core::error::ProviderError;
use rig_core::{
    message::AssistantContent,
    wasm_compat::{WasmBoxedStream, WasmCompatSend},
};

use crate::{
    agent::engine::{DriveItem, StreamingTurnSource, drive_agent},
    agent::hook::{AgentHook, RunSettled, SettledOutcome, StepEventKind},
    agent::runner::{AgentRunner, RunOrigin},
    streaming::{Item, StreamEvent},
};
use futures::{SinkExt, Stream, StreamExt, channel::mpsc, stream::FusedStream};
use serde::Serialize;
use std::io::Write;
use std::pin::Pin;
use tracing_futures::Instrument;

use crate::completion::PromptError;
use crate::run::response::{CompletionCall, PromptResponse};
use rig_core::message::Message;

/// The stream a streamed run yields: its items, then its ending.
pub type StreamingResult = WasmBoxedStream<'static, Result<MultiTurnStreamItem, PromptError>>;

#[derive(Serialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "camelCase")]
/// One item of a streamed run: a provider stream event, a committed tool
/// call, a lifecycle marker, or the run's final response.
pub enum MultiTurnStreamItem {
    /// A provider stream item containing model-emitted content: part
    /// starts and ends, text and reasoning fragments, the call parts of a
    /// validated tool call, and unmodeled passthrough payloads. The model's
    /// completed calls are also reported as [`ToolCall`](Self::ToolCall)
    /// when the turn commits.
    StreamAssistantItem(Item<StreamEvent>),
    /// A tool call the **model emitted**, reported when the model turn is
    /// committed, for each call Rig routes to execution. Such a call is
    /// reported whether or not the tool body ultimately runs (a hook skip
    /// still reports it); it is **not** an execution-lifecycle event (see
    /// [`ToolExecutionCommitted`](Self::ToolExecutionCommitted)).
    ///
    /// Two kinds of model tool call are **not** reported here: a call rejected and
    /// handled by invalid-tool-call recovery (surfaced via that recovery
    /// path), and a structured-output Tool-mode output-tool call, which
    /// finalizes the run directly; its structured result is surfaced in
    /// the [`FinalResponse`](Self::FinalResponse) rather than as a completed
    /// call.
    ToolCall {
        /// The call as the model emitted it. Its id is equal on its
        /// execution commit and its result.
        tool_call: rig_core::message::ToolCall,
    },
    /// Confirmation that Rig **executed and committed** a tool call. This is not
    /// a real-time start notification: it is surfaced together with its
    /// `ToolResult` only after the whole batch settles successfully. Use tool
    /// hooks for live host-side start/result observation.
    ///
    /// This item is emitted only for a tool whose body actually ran (it passed
    /// its `ToolCall` hook checks), never for a call dropped by a sibling's
    /// termination, skipped by a hook, or resolved by invalid-call recovery.
    /// Correlate it with the model call and result through the call's id.
    ToolExecutionCommitted {
        /// The tool call as **executed**: the model's call with any
        /// [`DispatchAction::Patch`](crate::agent::DispatchAction::Patch) hook rewrite
        /// applied (so a redaction rewrite is reflected here, not leaked). The
        /// model's *original* call is reported via
        /// [`StreamAssistantItem`](Self::StreamAssistantItem).
        tool_call: rig_core::message::ToolCall,
    },
    /// The **result** of an executed (or hook-skipped) tool call. The tool
    /// batch commits and surfaces atomically at every `tool_concurrency`
    /// (including the sequential default): results are surfaced in call order
    /// only after the whole batch settles successfully; a run that terminates
    /// mid-batch surfaces no successful tool results.
    ToolResult {
        /// The result; `tool_result.call` is the id of the call it answers.
        tool_result: rig_core::message::ToolResult,
    },
    /// Details for one successfully completed completion request made by this agent stream.
    ///
    /// This is emitted when a provider call finishes. Usage is the provider's
    /// final usage for that completion request when available; it is not
    /// incremental per streamed token.
    ///
    /// ```
    /// use rig_agent::agent::MultiTurnStreamItem;
    /// fn input_tokens(item: &MultiTurnStreamItem) -> Option<u64> {
    ///     match item {
    ///         MultiTurnStreamItem::CompletionCall(call) => call.usage.input_tokens,
    ///         _ => None,
    ///     }
    /// }
    /// ```
    CompletionCall(CompletionCall),
    /// The completed model turn was rejected by a hook for retry.
    ///
    /// Text and reasoning deltas emitted for this turn were provisional. A
    /// consumer should discard or visually reset output associated with `turn`.
    /// A subsequent attempt is made only if the run's total model-call budget
    /// permits it.
    ModelTurnRetried {
        /// One-based model-call index of the rejected turn.
        turn: usize,
    },
    /// The final result from the stream: the unified [`PromptResponse`] shared
    /// with the blocking surface.
    ///
    /// Terminal for the run: no retry, further turn, or tool execution follows.
    /// This item is the stream-side
    /// counterpart of the `on_run_settled` hook's success outcome. Error
    /// termination surfaces as the stream's `Err` item instead, which is
    /// equally terminal.
    FinalResponse(PromptResponse),
}

/// Build the unified [`PromptResponse`] for the streaming surface from the
/// final turn's structured content.
fn final_response_from_content(
    content: Vec<AssistantContent>,
    aggregated_usage: crate::completion::Usage,
    completion_calls: Vec<CompletionCall>,
    history: Vec<Message>,
) -> PromptResponse {
    PromptResponse::from_content(content, aggregated_usage)
        .with_completion_calls(completion_calls)
        .with_messages(history)
}

impl MultiTurnStreamItem {
    pub(crate) fn stream_item(item: Item<StreamEvent>) -> Self {
        Self::StreamAssistantItem(item)
    }

    /// Stamp a `FinalResponse` item with how the run's memory append
    /// settled; any other item is returned unchanged.
    pub(crate) fn with_memory_append(
        self,
        memory_append: Option<crate::run::MemoryAppend>,
    ) -> Self {
        match self {
            Self::FinalResponse(response) => {
                Self::FinalResponse(response.with_memory_append(memory_append))
            }
            other => other,
        }
    }

    /// Build a final response from structured content and aggregate usage.
    /// Concatenates text for output; completion details and history remain empty.
    pub fn final_response(
        content: Vec<AssistantContent>,
        aggregated_usage: crate::completion::Usage,
    ) -> Self {
        Self::FinalResponse(final_response_from_content(
            content,
            aggregated_usage,
            Vec::new(),
            Vec::new(),
        ))
    }

    pub(crate) fn final_response_with_completion_calls(
        content: Vec<AssistantContent>,
        aggregated_usage: crate::completion::Usage,
        completion_calls: Vec<CompletionCall>,
        history: Vec<Message>,
    ) -> Self {
        Self::FinalResponse(final_response_from_content(
            content,
            aggregated_usage,
            completion_calls,
            history,
        ))
    }
}

impl AgentRunner {
    /// Drive the agent loop as a stream of assistant content, tool activity
    /// and, last, the [`FinalResponse`](MultiTurnStreamItem::FinalResponse).
    /// Hooks fire at every observable point, including streamed text and
    /// tool-call deltas.
    ///
    /// Like [`run`](AgentRunner::run), this is lazy: memory loading and agent
    /// span creation begin only when the stream is first polled,
    /// and a stream that is dropped unpolled has done nothing. A memory-load
    /// failure is the stream's first (and only) item. The stream is `Send` on
    /// native targets, so it can be built in synchronous code and handed to
    /// whatever polls it; it runs under the span it was built in (a caller's
    /// enabled span is adopted, otherwise a root `invoke_agent` is created),
    /// not under whichever span first polls it.
    ///
    /// ```rust,no_run
    /// # use rig_agent::{Agent, agent::StreamingResult};
    /// fn start(agent: &Agent, prompt: &str) -> StreamingResult {
    ///     // Nothing runs until whoever holds this polls it.
    ///     agent.prompt(prompt).stream()
    /// }
    /// ```
    ///
    /// Shares the drive loop, run construction, tool execution and fail-closed
    /// hook handling with the blocking [`run`](AgentRunner::run) via
    /// `drive_agent`, so the two behave identically apart from the streamed
    /// delta events.
    #[must_use = "a stream does nothing until polled"]
    pub fn stream(self) -> StreamingResult {
        // The span the stream is built under is the one it runs under, not
        // whichever span first polls it: a host may build the stream in a
        // request span and hand it to a task of its own.
        self.stream_under(tracing::Span::current())
    }

    /// Build a lazy stream under an explicitly captured ambient span.
    fn stream_under(self, ambient: tracing::Span) -> StreamingResult {
        let run_under = ambient.clone();
        let stream = async_stream::stream! {
            let mut inner = self.start_stream(run_under).await;
            while let Some(item) = inner.next().await {
                yield item;
            }
        };
        Box::pin(stream.instrument(ambient))
    }

    /// The eager half of [`stream`](Self::stream): resolve memory, build the
    /// run and return the driver as a stream. Called on the first poll,
    /// under `ambient`, the span the stream was built in.
    async fn start_stream(self, ambient: tracing::Span) -> StreamingResult {
        let (agent_span, created_agent_span) = self.open_agent_span(ambient);

        let bus = self.config.bus.clone();
        let hook_ctx = self.hook_context(true);
        // A resumed run loads nothing and appends what it adds (see `run`).
        let resolved = match &self.origin {
            RunOrigin::Resume(_) => self.resumed_memory(),
            RunOrigin::Prompt(_) => {
                let resolve = self.resolve_history_and_memory(&hook_ctx);
                futures::pin_mut!(resolve);
                let mut driven = bus.drive(futures::stream::once(resolve));
                driven.next().await.unwrap_or(Ok((None, None)))
            }
        };
        let (history_override, memory_handle) = match resolved {
            Ok(resolved) => resolved,
            Err(err) => {
                // Notify settlement before yielding the error so consumers stopping
                // at the first failure cannot suppress the lifecycle notification.
                let hooks = self.config.hooks.clone();
                let stream = async_stream::stream! {
                    let err = PromptError::from(err);
                    if hooks.observes(StepEventKind::RunSettled) {
                        let reason = err.to_string();
                        hooks
                            .on_run_settled(
                                &hook_ctx,
                                RunSettled {
                                    outcome: SettledOutcome::Error(&reason),
                                    messages: None,
                                },
                            )
                            .await;
                    }
                    yield Err(err);
                };
                // Instrument under the agent span like the success path so
                // a load failure stays tied to invoke_agent.
                return Box::pin(stream.instrument(agent_span));
            }
        };

        let run = self.build_run(history_override);
        let source = StreamingTurnSource::new(
            &self.config.hooks,
            self.agent_name_or_default().to_string(),
            created_agent_span,
            self.config.record_telemetry_content,
        );

        let driver = drive_agent(
            self,
            source,
            run,
            agent_span.clone(),
            created_agent_span,
            memory_handle,
            hook_ctx,
        )
        .filter_map(|item| {
            std::future::ready(match item {
                Ok(DriveItem::Item(item)) => Some(Ok(item)),
                Ok(DriveItem::Done(_)) => None,
                Err(err) => Some(Err(err)),
            })
        });
        // The consumer of this stream drives the agent's bus: every poll that
        // leaves the run pending polls the driver.
        let driver = bus.drive(Box::pin(driver));

        Box::pin(driver.instrument(agent_span))
    }
}

/// Capacity of the event queue behind [`AgentRunner::run_channel`]: the number
/// of [`MultiTurnStreamItem`]s the run may buffer ahead of the consumer before
/// it parks on back-pressure.
pub const RUN_EVENTS_CAPACITY: usize = 32;

/// Event feed of an agent run started with [`AgentRunner::run_channel`].
///
/// Every [`MultiTurnStreamItem`] the run would have streamed is delivered here
/// in order, ending with [`MultiTurnStreamItem::FinalResponse`]. The feed is a
/// bounded queue ([`RUN_EVENTS_CAPACITY`]): a slow consumer applies
/// back-pressure to the run instead of losing events. Poll it as a
/// [`Stream`] from async code, or drain it with the non-blocking
/// [`try_next`](RunEvents::try_next) from a synchronous tick, such as a game loop,
/// UI frame, an ECS system.
///
/// Dropping the feed does not cancel the run; it simply stops receiving events
/// and the run future still resolves with the final
/// [`PromptResponse`].
#[derive(Debug)]
pub struct RunEvents {
    receiver: mpsc::Receiver<MultiTurnStreamItem>,
}

impl RunEvents {
    /// Take the next buffered event without waiting.
    ///
    /// Returns `None` both when no event is queued yet and once the run has
    /// finished and the feed is drained; use [`is_done`](RunEvents::is_done)
    /// to tell the two apart.
    pub fn try_next(&mut self) -> Option<MultiTurnStreamItem> {
        self.receiver.try_recv().ok()
    }

    /// Whether the run has finished and every event has been taken.
    pub fn is_done(&self) -> bool {
        self.receiver.is_terminated()
    }
}

impl Stream for RunEvents {
    type Item = MultiTurnStreamItem;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        Pin::new(&mut self.receiver).poll_next(cx)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.receiver.size_hint()
    }
}

impl FusedStream for RunEvents {
    fn is_terminated(&self) -> bool {
        self.receiver.is_terminated()
    }
}

impl AgentRunner {
    /// Split the run into a driving future and a [`RunEvents`] feed.
    ///
    /// The future performs the same agent loop as
    /// [`run`](AgentRunner::run) and [`stream`](AgentRunner::stream), and
    /// resolves with the final [`PromptResponse`]; the
    /// feed receives each intermediate [`MultiTurnStreamItem`] as it happens.
    /// Spawn the future on any executor and poll the feed from wherever the
    /// events are consumed; neither side assumes a runtime.
    ///
    /// The feed is bounded ([`RUN_EVENTS_CAPACITY`]); when it is full the run
    /// waits for the consumer rather than dropping events. Dropping the feed
    /// lets the run continue to completion unobserved. Like
    /// [`stream`](Self::stream), the run belongs to the span this method was
    /// called in, wherever the future is later polled.
    #[must_use = "the run does nothing until the future is driven"]
    pub fn run_channel(
        self,
    ) -> (
        impl Future<Output = Result<PromptResponse, PromptError>> + WasmCompatSend,
        RunEvents,
    ) {
        let (mut sender, receiver) = mpsc::channel(RUN_EVENTS_CAPACITY);
        // Captured here, not when the future is first polled: the doc above
        // says to spawn the future, and the run must still belong to the span
        // that split it.
        let ambient = tracing::Span::current();
        let future = async move {
            let mut stream = self.stream_under(ambient);
            let mut response = None;
            let mut forward = true;
            while let Some(item) = stream.next().await {
                let item = match item {
                    Ok(item) => item,
                    Err(err) => {
                        // Drain after the terminal error so span and lineage teardown
                        // completes instead of being dropped at the yield.
                        while stream.next().await.is_some() {}
                        return Err(err);
                    }
                };
                match item {
                    MultiTurnStreamItem::FinalResponse(done) => {
                        if forward {
                            let _ = sender
                                .send(MultiTurnStreamItem::FinalResponse(done.clone()))
                                .await;
                        }
                        response = Some(done);
                    }
                    item => {
                        if forward && sender.send(item).await.is_err() {
                            forward = false;
                        }
                    }
                }
            }
            response.ok_or_else(|| {
                PromptError::Provider(ProviderError::Response(
                    "agent run ended without producing a final response".to_string(),
                ))
            })
        };
        (future, RunEvents { receiver })
    }
}

/// Why [`stream_to_stdout`] returned no response.
#[derive(Debug, thiserror::Error)]
pub enum StreamToStdoutError {
    /// The run failed.
    #[error(transparent)]
    Run(#[from] PromptError),
    /// Writing to stdout failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// The stream ended without a final response.
    #[error("the stream ended without a final response")]
    Incomplete,
}

/// Print a streamed run's assistant text and reasoning to stdout and return
/// its final response.
///
/// Streaming metadata events, such as `MultiTurnStreamItem::CompletionCall`,
/// are not printed; metadata is returned on the [`PromptResponse`] via
/// accessors such as [`PromptResponse::completion_calls`]. A model-turn retry
/// prints a visible boundary because text already written to stdout cannot be
/// retracted. Fails with the run's error as soon as the stream yields it, a
/// stdout write failure, or [`StreamToStdoutError::Incomplete`] when the
/// stream ends without a final response.
pub async fn stream_to_stdout(
    stream: &mut StreamingResult,
) -> Result<PromptResponse, StreamToStdoutError> {
    let mut stdout = std::io::stdout();
    write!(stdout, "Response: ")?;
    while let Some(content) = stream.next().await {
        match content? {
            MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::Text {
                text,
                ..
            })) => {
                write!(stdout, "{text}")?;
                stdout.flush()?;
            }
            MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::End {
                content: AssistantContent::Reasoning(reasoning),
                ..
            })) => {
                write!(stdout, "{}", reasoning.text)?;
                stdout.flush()?;
            }
            MultiTurnStreamItem::FinalResponse(response) => return Ok(response),
            MultiTurnStreamItem::ModelTurnRetried { turn } => {
                write!(
                    stdout,
                    "\n[model turn {turn} rejected; retry requested]\nResponse: "
                )?;
                stdout.flush()?;
            }
            _ => {}
        }
    }
    Err(StreamToStdoutError::Incomplete)
}

#[cfg(test)]
mod malformed_tool_args_tests;
#[cfg(test)]
#[allow(irrefutable_let_patterns, unreachable_patterns)]
mod tests;
