//! Bridging Rig's per-run conversation onto an A2A thread.
//!
//! A [`CompletionModel`](rig_core::completion::CompletionModel) receives a
//! [`CompletionRequest`](rig_core::completion::CompletionRequest) and nothing
//! else, so an [`A2AModel`](crate::A2AModel) cannot see which conversation a
//! run belongs to. A hook can: the runner reports it as
//! [`HookContext::conversation_id`], and a hook may patch the request the model
//! is about to receive.
//!
//! [`A2AThreadHook`] joins those two halves, so one A2A-backed agent serves
//! every conversation instead of one agent per conversation:
//!
//! ```no_run
//! use rig_agent::completion::Prompt;
//!
//! # async fn run(client: rig_a2a::A2AClient) -> anyhow::Result<()> {
//! // `A2AClient::agent` registers the hook for you.
//! let agent = client.agent().build();
//!
//! agent.prompt("what did we decide?").conversation("user-42").await?;
//! // A different conversation, same agent, separate remote thread.
//! agent.prompt("and here?").conversation("user-99").await?;
//! # Ok(()) }
//! ```
//!
//! The remote's own identifiers stay host-side, as they do for the tool
//! bridge: the hook passes Rig's conversation id, and the client resolves it
//! against the `contextId` and `taskId` the server issued. A model can neither
//! forge one nor lose one.

use rig_agent::agent::{
    AgentHook, CompletionCallAction, CompletionCallEvent, CompletionResponseEvent, HookContext,
    ObservationAction, RequestPatch, StreamResponseFinish,
};

use crate::model::THREAD_PARAMS_KEY;

/// Threads each run into the A2A conversation the run names.
///
/// Registered automatically by [`A2AClient::agent`](crate::A2AClient::agent)
/// and [`A2AClient::agent_for_conversation`](crate::A2AClient::agent_for_conversation).
/// Add it by hand only when building an [`AgentBuilder`] around an
/// [`A2AModel`](crate::A2AModel) yourself.
///
/// A run that names no conversation is left alone and stays single-turn — the
/// hook never invents a key, because a guessed conversation would silently
/// merge unrelated exchanges into one remote thread.
///
/// [`AgentBuilder`]: rig_agent::agent::AgentBuilder
#[derive(Debug, Clone, Copy, Default)]
pub struct A2AThreadHook;

impl AgentHook for A2AThreadHook {
    async fn on_completion_call(
        &self,
        ctx: &HookContext,
        _event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        let Some(conversation) = ctx.conversation_id() else {
            return CompletionCallAction::continue_run();
        };
        CompletionCallAction::patch(RequestPatch::new().additional_params(serde_json::json!({
            THREAD_PARAMS_KEY: { "conversation": conversation }
        })))
    }

    async fn on_completion_response(
        &self,
        ctx: &HookContext,
        event: CompletionResponseEvent<'_>,
    ) -> ObservationAction {
        log_turn(ctx, event.response_id);
        ObservationAction::Continue
    }

    async fn on_stream_response_finish(
        &self,
        ctx: &HookContext,
        event: StreamResponseFinish<'_>,
    ) -> ObservationAction {
        log_turn(ctx, event.response_id);
        ObservationAction::Continue
    }
}

/// Trace one model-surface turn by its remote task.
///
/// [`A2AModel`](crate::A2AModel) reports the A2A `taskId` as the response id,
/// which is the only per-turn handle into the remote's own logs. The tool
/// bridge publishes the equivalent as
/// [`A2AThreadInfo`](crate::A2AThreadInfo) result metadata; this is the model
/// surface's counterpart.
fn log_turn(ctx: &HookContext, task_id: Option<&str>) {
    let Some(task_id) = task_id else {
        return;
    };
    tracing::debug!(
        target: "rig_a2a",
        conversation = ctx.conversation_id(),
        turn = ctx.turn(),
        task_id,
        "a2a model turn"
    );
}
