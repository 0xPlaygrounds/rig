use rig_core::streaming::BlockId;
use rig_core::{message::AssistantContent, wasm_compat::WasmCompatSend};

use crate::{
    agent::engine::{DriveItem, StreamingTurnSource, drive_agent, streaming_error_into_prompt},
    agent::runner::AgentRunner,
    streaming::{BlockClose, Delta, StreamEvent, StreamedUserContent},
};
use futures::{SinkExt, Stream, StreamExt, channel::mpsc, stream::FusedStream};
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use tracing_futures::Instrument;

use crate::run::response::{CompletionCall, PromptResponse};
use crate::run::transcript::assistant_text_from_choice;
use crate::{
    agent::Agent,
    completion::{CompletionError, PromptError},
};
use rig_core::message::Message;

// The `Send` bound is dropped exactly where `rig-core`'s `WasmCompat*` markers
// go no-op — browser wasm. `rig-core` keys those markers on this same
// predicate, so keep the two in step: a bare `target_arch = "wasm32"` would
// also drop `Send` on WASI, where `rig-core` still requires it.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub type StreamingResult =
    Pin<Box<dyn Stream<Item = Result<MultiTurnStreamItem, StreamingError>> + Send>>;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub type StreamingResult = Pin<Box<dyn Stream<Item = Result<MultiTurnStreamItem, StreamingError>>>>;

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "camelCase")]
#[allow(
    clippy::large_enum_variant,
    reason = "the terminal items are one per run and are moved, not copied; boxing them would put an allocation on every consumer's match"
)]
pub enum MultiTurnStreamItem {
    /// A provider stream event — the content the **model emitted**: block
    /// starts and ends, text/reasoning deltas, tool-call deltas, the
    /// terminal record, unmodeled passthrough items. Tool-call block ends
    /// are not forwarded; the model's completed calls are reported as
    /// [`ToolCall`](Self::ToolCall) when the turn commits.
    StreamAssistantItem(StreamEvent),
    /// A tool call the **model emitted**, reported when the model turn is
    /// committed, for each call Rig routes to execution. Such a call is
    /// reported whether or not the tool body ultimately runs (a hook skip
    /// still reports it); it is **not** an execution-lifecycle event (see
    /// [`ToolExecutionCommitted`](Self::ToolExecutionCommitted)).
    ///
    /// Two kinds of model tool call are **not** reported here (their
    /// arguments still stream as tool-call deltas): a call rejected and
    /// handled by invalid-tool-call recovery (surfaced via that recovery
    /// path), and a structured-output Tool-mode output-tool call, which
    /// finalizes the run directly — its structured result is surfaced in
    /// the [`FinalResponse`](Self::FinalResponse) rather than as a completed
    /// call.
    ToolCall {
        /// The call as the model emitted it.
        tool_call: rig_core::message::ToolCall,
        /// The block this call streamed under (a buffered turn's call is
        /// keyed by its durable id): equal on its deltas, its execution
        /// commit and its result.
        block_id: BlockId,
    },
    /// Confirmation that Rig **executed and committed** a tool call. This is not
    /// a real-time start notification: it is surfaced together with its
    /// `ToolResult` only after the whole batch settles successfully. Use tool
    /// hooks for live host-side start/result observation.
    ///
    /// This item is emitted only for a tool whose body actually ran (it passed
    /// its `ToolCall` hook checks), never for a call dropped by a sibling's
    /// termination, skipped by a hook, or resolved by invalid-call recovery.
    /// Correlate it with the model call and result through `block_id`.
    ToolExecutionCommitted {
        /// The tool call as **executed**: the model's call with any
        /// [`DispatchAction::Patch`](crate::agent::DispatchAction::Patch) hook rewrite
        /// applied (so a redaction rewrite is reflected here, not leaked). The
        /// model's *original* call is reported via
        /// [`StreamAssistantItem`](Self::StreamAssistantItem).
        tool_call: rig_core::message::ToolCall,
        /// The block id correlating this execution with the model tool call
        /// ([`ToolCall::block_id`](Self::ToolCall)) and the resulting
        /// [`StreamedUserContent::ToolResult`].
        block_id: BlockId,
    },
    /// A streamed user content item: the **result** of an executed (or
    /// hook-skipped) tool call. The tool batch commits and surfaces atomically at
    /// every `tool_concurrency` (including the sequential default): results are
    /// surfaced (in call order) only after the whole batch settles successfully —
    /// a run that terminates mid-batch surfaces no successful tool results.
    StreamUserItem(StreamedUserContent),
    /// Details for one successfully completed completion request made by this agent stream.
    ///
    /// This is emitted when a provider call finishes. Usage is the provider's
    /// final usage for that completion request when available; it is not
    /// incremental per streamed token.
    ///
    /// ```rust,ignore
    /// match item {
    ///     MultiTurnStreamItem::CompletionCall(completion_call) => {
    ///         // Zero-valued usage means the provider reported no metrics.
    ///         if completion_call.usage.has_values() {
    ///             let context_tokens = completion_call.usage.input_tokens;
    ///         }
    ///     }
    ///     _ => {}
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
    /// Terminal for the run: nothing follows it automatically — no retry,
    /// further turn, or tool execution — so this item is the stream-side
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
    history: Option<Vec<Message>>,
) -> PromptResponse {
    let mut response = PromptResponse::new(assistant_text_from_choice(&content), aggregated_usage)
        .with_content(content)
        .with_completion_calls(completion_calls);
    response.messages = history;
    response
}

impl MultiTurnStreamItem {
    pub(crate) fn stream_item(item: StreamEvent) -> Self {
        Self::StreamAssistantItem(item)
    }

    /// Build a `FinalResponse` item from final-turn content, applying the
    /// run-finalization shaping of `final_response_from_content` (#1928).
    /// The one public entry point to that shaping, for mocks and adapters
    /// that synthesize final items outside the drive loop.
    pub fn final_response(
        content: Vec<AssistantContent>,
        aggregated_usage: crate::completion::Usage,
    ) -> Self {
        Self::FinalResponse(final_response_from_content(
            content,
            aggregated_usage,
            Vec::new(),
            None,
        ))
    }

    pub(crate) fn final_response_with_completion_calls(
        content: Vec<AssistantContent>,
        aggregated_usage: crate::completion::Usage,
        completion_calls: Vec<CompletionCall>,
        history: Option<Vec<Message>>,
    ) -> Self {
        Self::FinalResponse(final_response_from_content(
            content,
            aggregated_usage,
            completion_calls,
            history,
        ))
    }
}

/// Drain a provider stream abandoned by invalid tool-call recovery so the
/// reported usage for the recovered completion call is not lost.
pub(crate) async fn drain_stream_usage(
    stream: &mut crate::streaming::StreamingCompletionResponse,
) -> Result<crate::completion::Usage, StreamingError> {
    while let Some(content) = stream.next().await {
        match content {
            Ok(StreamEvent::Final(final_resp)) => {
                return Ok(final_resp.usage);
            }
            Ok(_) => {}
            Err(err) => return Err(err.into()),
        }
    }

    Ok(crate::completion::Usage::new())
}

/// Build the final streamed content for a finished run (#1928).
///
/// When the finishing turn carries a tool call it is a Tool-mode output-tool
/// call (a real tool call would have routed to `CallTools`, not `Done`). In that
/// case the tool call AND the model's prose are dropped, any reasoning/image
/// content is kept, and `output` is appended as the final text — so the streamed
/// [`PromptResponse::output`] string is the structured output rather than the
/// prose, with no unanswered tool_use, matching the non-streaming `output`. Note
/// this shapes only the surfaced [`PromptResponse::content`]; the persisted
/// message history is built by the state machine (which keeps the prose, like the
/// blocking driver), so `content` and `messages` intentionally differ on prose in
/// this case.
/// Otherwise returns `None` and the caller surfaces the turn's content unchanged.
pub(crate) fn finalize_streamed_choice(
    last_final_choice: &[AssistantContent],
    output: &str,
) -> Option<Vec<AssistantContent>> {
    let finalized_via_output_tool = last_final_choice
        .iter()
        .any(|item| matches!(item, AssistantContent::ToolCall(_)));
    if !finalized_via_output_tool {
        return None;
    }
    let mut items: Vec<AssistantContent> = last_final_choice
        .iter()
        .filter(|item| {
            !matches!(
                item,
                AssistantContent::ToolCall(_) | AssistantContent::Text(_)
            )
        })
        .cloned()
        .collect();
    // `items` is non-empty: the output text was just pushed unconditionally.
    items.push(AssistantContent::text(output.to_string()));
    Some(items)
}

#[derive(Debug, thiserror::Error)]
pub enum StreamingError {
    #[error("CompletionError: {0}")]
    Completion(#[from] CompletionError),
    #[error("PromptError: {0}")]
    Prompt(#[from] Box<PromptError>),
}

impl From<rig_core::memory::MemoryError> for StreamingError {
    fn from(err: rig_core::memory::MemoryError) -> Self {
        Self::Prompt(Box::new(PromptError::MemoryError(err)))
    }
}

impl AgentRunner {
    /// Drive the agent loop, streaming assistant content, tool activity, and a
    /// final response. Hooks fire at every observable point, including streamed
    /// text and tool-call deltas. Returns the stream after loading any
    /// configured conversation memory.
    ///
    /// Shares the drive loop, run construction, tool execution and fail-closed
    /// hook handling with the blocking [`run`](AgentRunner::run) via
    /// `drive_agent`, so the two behave identically apart from the streamed
    /// delta events.
    pub async fn stream(self) -> StreamingResult {
        let (agent_span, created_agent_span) = self.open_agent_span();

        let bus = self.config.bus.clone();
        let hook_ctx = self.hook_context(true);
        let resolved = {
            let resolve = self.resolve_history_and_memory(&hook_ctx);
            futures::pin_mut!(resolve);
            let mut driven = bus.drive(futures::stream::once(resolve));
            driven.next().await.unwrap_or(Ok((None, None)))
        };
        let (history_override, memory_handle) = match resolved {
            Ok(resolved) => resolved,
            Err(err) => {
                let stream = async_stream::stream! {
                    yield Err(StreamingError::from(err));
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

        // The blocking surface folds this same engine; the streaming surface
        // forwards intermediate items (the final response item is the last one)
        // and ends on `Done`.
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

/// Event feed of an agent run started with [`AgentRunner::run_channel`] or
/// [`Agent::run_channel`].
///
/// Every [`MultiTurnStreamItem`] the run would have streamed is delivered here
/// in order, ending with [`MultiTurnStreamItem::FinalResponse`]. The feed is a
/// bounded queue ([`RUN_EVENTS_CAPACITY`]): a slow consumer applies
/// back-pressure to the run instead of losing events. Poll it as a
/// [`Stream`] from async code, or drain it with the non-blocking
/// [`try_next`](RunEvents::try_next) from a synchronous tick — a game loop, a
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
    /// The future performs the whole agent loop — the same engine as
    /// [`run`](AgentRunner::run) and [`stream`](AgentRunner::stream) — and
    /// resolves with the final [`PromptResponse`]; the
    /// feed receives each intermediate [`MultiTurnStreamItem`] as it happens.
    /// Spawn the future on any executor and poll the feed from wherever the
    /// events are consumed; neither side assumes a runtime.
    ///
    /// The feed is bounded ([`RUN_EVENTS_CAPACITY`]); when it is full the run
    /// waits for the consumer rather than dropping events. Dropping the feed
    /// lets the run continue to completion unobserved.
    pub fn run_channel(
        self,
    ) -> (
        impl Future<Output = Result<PromptResponse, PromptError>> + WasmCompatSend,
        RunEvents,
    ) {
        let (mut sender, receiver) = mpsc::channel(RUN_EVENTS_CAPACITY);
        let future = async move {
            let mut stream = self.stream().await;
            let mut response = None;
            let mut forward = true;
            while let Some(item) = stream.next().await {
                let item = item.map_err(streaming_error_into_prompt)?;
                match item {
                    MultiTurnStreamItem::FinalResponse(done) => {
                        if forward {
                            // The consumer is gone; nothing left to forward.
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
                PromptError::CompletionError(CompletionError::ResponseError(
                    "agent run ended without producing a final response".to_string(),
                ))
            })
        };
        (future, RunEvents { receiver })
    }
}

impl Agent {
    /// Run `prompt` with the agent's defaults, returning the driving future and
    /// a [`RunEvents`] feed. See [`AgentRunner::run_channel`]; to configure the
    /// run first (history, turn budget, tool context, …), configure the runner
    /// from [`Agent::stream_prompt`] and call its
    /// [`run_channel`](AgentRunner::run_channel).
    pub fn run_channel<P: Into<Message> + WasmCompatSend>(
        &self,
        prompt: P,
    ) -> (
        impl Future<Output = Result<PromptResponse, PromptError>> + WasmCompatSend + use<P>,
        RunEvents,
    ) {
        AgentRunner::from_agent(self, prompt).run_channel()
    }
}

/// Helper function to stream assistant-visible completion output to stdout.
///
/// This helper prints streamed assistant text and reasoning. Streaming metadata
/// events, such as `MultiTurnStreamItem::CompletionCall`, are not printed;
/// metadata is returned on the [`PromptResponse`] via accessors such as
/// [`PromptResponse::completion_calls`]. A model-turn retry prints a visible
/// boundary because text already written to stdout cannot be retracted.
pub async fn stream_to_stdout(
    stream: &mut StreamingResult,
) -> Result<PromptResponse, std::io::Error> {
    let mut final_res = PromptResponse::empty();
    print!("Response: ");
    while let Some(content) = stream.next().await {
        match content {
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamEvent::BlockDelta {
                delta: Delta::Text { text },
                ..
            })) => {
                print!("{text}");
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamEvent::BlockEnd {
                end: BlockClose::Reasoning { .. },
                block: Some(AssistantContent::Reasoning(reasoning)),
                ..
            })) => {
                let reasoning = reasoning.display_text();
                print!("{reasoning}");
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                final_res = res;
            }
            Ok(MultiTurnStreamItem::ModelTurnRetried { turn }) => {
                print!("\n[model turn {turn} rejected; retry requested]\nResponse: ");
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            Err(err) => {
                eprintln!("Error: {err}");
            }
            _ => {}
        }
    }

    Ok(final_res)
}

#[cfg(test)]
#[allow(irrefutable_let_patterns, unreachable_patterns)]
mod tests;
