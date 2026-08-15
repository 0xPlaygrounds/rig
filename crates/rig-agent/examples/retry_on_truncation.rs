//! Retry a turn the provider cut short, without credentials or provider types.
//!
//! `ModelTurnFinished` reports why the provider stopped (`finish_reason`) and
//! the output-token cap that exact attempt ran under (`max_tokens`). Together
//! they make "the provider truncated me, go again with more room" a portable
//! policy: the hook below names no provider and never touches a raw response
//! type, so the same code works against every model Rig supports.
//!
//! The scripted model here truncates whenever its cap is below what the answer
//! needs, so the escalation is *causal* rather than staged — the loop ends
//! because the cap finally became large enough, not because a script said so.
//!
//! ```not_rust
//! cargo run -p rig-agent --example retry_on_truncation
//! ```

use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Result;
use futures::{StreamExt, stream};
use rig_agent::{
    AgentBuilder,
    agent::{
        AgentHook, CompletionCallAction, CompletionCallEvent, HookContext, ModelTurnAction,
        ModelTurnFinished, MultiTurnStreamItem, RequestPatch,
    },
    completion::{
        CompletionError, CompletionModel, CompletionRequest, CompletionResponse, FinishReason,
        Prompt, Usage,
    },
    streaming::StreamingPrompt,
    streaming::{RawStreamingChoice, StreamFinal, StreamingCompletionResponse},
};
use rig_core::message::AssistantContent;

/// The full answer costs this many output tokens; anything less is truncated.
const ANSWER_COST: u64 = 40;

const ANSWER: &str = "Rig normalizes every provider's stop reason into one vocabulary.";

/// A model that behaves like a real one under an output-token cap: it emits
/// what fits and reports `Length` when the cap cut it short.
#[derive(Clone)]
struct BudgetedModel;

impl BudgetedModel {
    /// What the model can say under `cap`, and how it stopped.
    fn answer_under(cap: Option<u64>) -> (String, FinishReason) {
        match cap {
            Some(cap) if cap < ANSWER_COST => {
                // Roughly proportional truncation — the point is only that the
                // text is cut and the reason says so.
                let kept = (ANSWER.len() as u64 * cap / ANSWER_COST) as usize;
                (ANSWER[..kept].to_owned(), FinishReason::Length)
            }
            _ => (ANSWER.to_owned(), FinishReason::Stop),
        }
    }
}

impl CompletionModel for BudgetedModel {
    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, CompletionError> {
        let (text, reason) = Self::answer_under(request.max_tokens);
        Ok(
            CompletionResponse::new(vec![AssistantContent::text(text)], Usage::new(), "budgeted")
                .with_finish_reason(reason),
        )
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse, CompletionError> {
        // Identical semantics on the streaming surface: the hook sees the same
        // reason and the same cap either way.
        let (text, reason) = Self::answer_under(request.max_tokens);
        Ok(StreamingCompletionResponse::stream(
            "budgeted",
            Box::pin(stream::iter([
                Ok(RawStreamingChoice::Message(text)),
                Ok(RawStreamingChoice::FinalResponse(
                    StreamFinal::new("budgeted", Usage::new()).with_finish_reason(reason),
                )),
            ])),
        ))
    }
}

/// Doubles the output cap each time the provider truncates a turn, up to a
/// ceiling. Provider-neutral: it reads only `FinishReason` and `max_tokens`.
struct GrowCapOnTruncation {
    cap: AtomicU64,
    ceiling: u64,
}

impl GrowCapOnTruncation {
    fn new(start: u64, ceiling: u64) -> Self {
        Self {
            cap: AtomicU64::new(start),
            ceiling,
        }
    }
}

impl AgentHook for GrowCapOnTruncation {
    /// Every attempt — including a retry — is prepared afresh, so the current
    /// cap is applied here and reported back on that attempt's
    /// `ModelTurnFinished`.
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        _event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        CompletionCallAction::patch(
            RequestPatch::new().max_tokens(self.cap.load(Ordering::Relaxed)),
        )
    }

    async fn on_model_turn_finished(
        &self,
        _ctx: &HookContext,
        event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        let reason = event.finish_reason;
        println!(
            "  turn {} · cap {:?} · stopped: {:?}",
            event.turn, event.max_tokens, reason
        );

        // `truncated_output()` is `Length | ContentFilter` — the reasons that
        // mean "cut short" rather than "finished". It is the same predicate
        // rig-agent uses internally, so the two cannot drift.
        let truncated = reason.is_some_and(FinishReason::truncated_output);
        // Retrying a turn that carries tool calls is rejected, so a policy that
        // might encounter one has to check before asking.
        let has_tool_call = event
            .content
            .iter()
            .any(|content| matches!(content, AssistantContent::ToolCall(_)));
        // `max_tokens` is *this* attempt's cap, patch included — growing past
        // the ceiling would be retrying a limit we already know we can't raise.
        let room = event.max_tokens.is_none_or(|cap| cap < self.ceiling);

        if truncated && !has_tool_call && room {
            let grown = event
                .max_tokens
                .map_or(self.ceiling, |cap| cap.saturating_mul(2).min(self.ceiling));
            self.cap.store(grown, Ordering::Relaxed);
            println!("  ↳ truncated with room to grow; retrying at {grown}");
            return ModelTurnAction::repeat();
        }
        ModelTurnAction::continue_run()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Starts far below what the answer costs, so the first attempts truncate.
    let agent = AgentBuilder::new(BudgetedModel)
        .add_hook(GrowCapOnTruncation::new(8, 256))
        .build();

    println!("blocking:");
    // Retries share the run's model-call budget, so it must allow all attempts.
    let answer = agent
        .prompt("Explain Rig's finish reasons.")
        .max_turns(8)
        .await?;
    println!("  => {answer}\n");

    // The same hook, unchanged, on the streaming surface: `finish_reason` and
    // `max_tokens` are read from the same per-attempt carrier either way, so
    // the escalation below is identical to the one above.
    println!("streaming:");
    let streaming_agent = AgentBuilder::new(BudgetedModel)
        .add_hook(GrowCapOnTruncation::new(8, 256))
        .build();
    let mut stream = streaming_agent
        .stream_prompt("Explain Rig's finish reasons.")
        .max_turns(8)
        .await;
    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::FinalResponse(final_response) = item? {
            println!("  => {}", final_response.output);
        }
    }

    Ok(())
}
