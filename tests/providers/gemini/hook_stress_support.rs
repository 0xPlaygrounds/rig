//! Shared fixtures for the hook-system stress cassette suites
//! (`cassette::hook_stress*`): a comprehensive event-tap observer, flexible
//! request-patch / arg-rewrite / result-rewrite / terminate hooks, scratchpad
//! probes, and a third counting arithmetic tool.
//!
//! Every hook here is **deterministic** (no clocks/RNG), so the outbound
//! requests it produces stay byte-identical across replay. See
//! `tools_support`'s note on loose assertions: only rig-synthesized values are
//! pinned to exact equality.
#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use rig::agent::{
    AgentHook, CompletionCallAction, CompletionCallEvent, CompletionResponseEvent, HookContext,
    InvalidToolCallAction, ModelTurnFinished, ObservationAction, RequestPatch,
    StreamResponseFinish, TextDelta, ToolCall as ToolCallEvent, ToolCallAction, ToolCallDelta,
    ToolResultAction, ToolResultEvent,
};
use rig::completion::{CompletionModel, Document};
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

/// Forces tool use and a dependent chain so the model takes >= 2 turns.
pub(crate) const CHAIN_PREAMBLE: &str = "You are a calculator assistant. You MUST use the provided \
     tools for every arithmetic operation instead of computing results yourself. Perform the steps \
     in order, using the result of each step as an input to the next. Once you have the final tool \
     result, reply with the final numeric answer in plain text.";

/// Forces independent tool use (no dependency) so the model may batch calls.
pub(crate) const INDEPENDENT_TOOLS_PREAMBLE: &str = "You are a calculator assistant. You MUST use \
     the provided tools for every arithmetic operation instead of computing results yourself. Once \
     you have the tool results you need, reply with the requested numbers in plain text.";

// ---------------------------------------------------------------------------
// A third counting tool (add/subtract live in tools_support).
// ---------------------------------------------------------------------------

#[derive(Deserialize, Serialize)]
pub(crate) struct OperationArgs {
    pub(crate) x: i64,
    pub(crate) y: i64,
}

#[derive(Clone, Default)]
pub(crate) struct CallCounter(Arc<std::sync::atomic::AtomicUsize>);

impl CallCounter {
    pub(crate) fn count(&self) -> usize {
        self.0.load(std::sync::atomic::Ordering::SeqCst)
    }
    fn bump(&self) {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

/// `multiply` tool that counts its real executions.
#[derive(Clone, Default)]
pub(crate) struct CountingMultiply {
    pub(crate) counter: CallCounter,
}

impl Tool for CountingMultiply {
    const NAME: &'static str = "multiply";
    type Args = OperationArgs;
    type Output = i64;

    fn description(&self) -> String {
        "Multiply x and y together".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "x": { "type": "number", "description": "The first operand" },
                "y": { "type": "number", "description": "The second operand" }
            },
            "required": ["x", "y"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, rig::tool::ToolExecutionError> {
        self.counter.bump();
        Ok(args.x * args.y)
    }
}

// ---------------------------------------------------------------------------
// Observation: one comprehensive event tap.
// ---------------------------------------------------------------------------

/// One observed hook event: its variant tag and the one-based turn it fired on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Breadcrumb {
    pub(crate) tag: &'static str,
    pub(crate) turn: usize,
}

/// Cross-hook, cross-turn scratchpad value: how many `ToolCall`s have been seen.
#[derive(Clone, Default)]
pub(crate) struct ToolCallTally(pub(crate) usize);

/// A comprehensive observe-only hook: counts every event kind, records the
/// `HookContext` identity (run id / streaming flag / agent name), captures each
/// tool call's and result's `internal_call_id` for correlation, records an
/// ordered `(tag, turn)` breadcrumb, and bumps a shared `Scratchpad` tally on
/// every `ToolCall`.
#[derive(Clone, Default)]
pub(crate) struct EventTap {
    breadcrumbs: Arc<Mutex<Vec<Breadcrumb>>>,
    run_ids: Arc<Mutex<BTreeSet<String>>>,
    streaming: Arc<Mutex<Option<bool>>>,
    agent_name: Arc<Mutex<Option<String>>>,
    call_ids: Arc<Mutex<Vec<String>>>,
    result_ids: Arc<Mutex<Vec<String>>>,
}

impl EventTap {
    pub(crate) fn breadcrumbs(&self) -> Vec<Breadcrumb> {
        self.breadcrumbs.lock().expect("breadcrumbs").clone()
    }
    pub(crate) fn distinct_run_ids(&self) -> usize {
        self.run_ids.lock().expect("run_ids").len()
    }
    pub(crate) fn is_streaming(&self) -> Option<bool> {
        *self.streaming.lock().expect("streaming")
    }
    pub(crate) fn agent_name(&self) -> Option<String> {
        self.agent_name.lock().expect("agent_name").clone()
    }
    pub(crate) fn count(&self, tag: &str) -> usize {
        self.breadcrumbs()
            .iter()
            .filter(|crumb| crumb.tag == tag)
            .count()
    }
    pub(crate) fn call_ids(&self) -> Vec<String> {
        self.call_ids.lock().expect("call_ids").clone()
    }
    pub(crate) fn result_ids(&self) -> Vec<String> {
        self.result_ids.lock().expect("result_ids").clone()
    }
    /// Distinct turn indices observed, in ascending order.
    pub(crate) fn distinct_turns(&self) -> Vec<usize> {
        let set: BTreeSet<usize> = self.breadcrumbs().iter().map(|c| c.turn).collect();
        set.into_iter().collect()
    }
    /// Per-turn `(ToolCall count, ToolResult count)`.
    pub(crate) fn per_turn_tool_pairs(&self) -> BTreeMap<usize, (usize, usize)> {
        let mut map: BTreeMap<usize, (usize, usize)> = BTreeMap::new();
        for crumb in self.breadcrumbs() {
            let entry = map.entry(crumb.turn).or_default();
            match crumb.tag {
                "ToolCall" => entry.0 += 1,
                "ToolResult" => entry.1 += 1,
                _ => {}
            }
        }
        map
    }
}

impl EventTap {
    fn record(&self, ctx: &HookContext, tag: &'static str) {
        self.run_ids
            .lock()
            .expect("run_ids")
            .insert(ctx.run_id().as_str().to_string());
        *self.streaming.lock().expect("streaming") = Some(ctx.is_streaming());
        *self.agent_name.lock().expect("agent_name") = ctx.agent_name().map(str::to_string);
        self.breadcrumbs
            .lock()
            .expect("breadcrumbs")
            .push(Breadcrumb {
                tag,
                turn: ctx.turn(),
            });
    }
}

impl<M: CompletionModel> AgentHook<M> for EventTap {
    async fn on_completion_call(
        &self,
        ctx: &HookContext,
        _event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        self.record(ctx, "CompletionCall");
        CompletionCallAction::continue_run()
    }
    async fn on_completion_response(
        &self,
        ctx: &HookContext,
        _event: CompletionResponseEvent<'_, M>,
    ) -> ObservationAction {
        self.record(ctx, "CompletionResponse");
        ObservationAction::continue_run()
    }
    async fn on_model_turn_finished(
        &self,
        ctx: &HookContext,
        _event: ModelTurnFinished<'_>,
    ) -> ObservationAction {
        self.record(ctx, "ModelTurnFinished");
        ObservationAction::continue_run()
    }
    async fn on_invalid_tool_call(
        &self,
        ctx: &HookContext,
        _event: &rig::agent::InvalidToolCallContext,
    ) -> InvalidToolCallAction {
        self.record(ctx, "InvalidToolCall");
        InvalidToolCallAction::fail()
    }
    async fn on_tool_call(&self, ctx: &HookContext, event: ToolCallEvent<'_>) -> ToolCallAction {
        self.record(ctx, "ToolCall");
        self.call_ids
            .lock()
            .expect("call_ids")
            .push(event.internal_call_id.to_string());
        ctx.scratchpad()
            .update(|tally: &mut ToolCallTally| tally.0 += 1);
        ToolCallAction::run()
    }
    async fn on_tool_result(
        &self,
        ctx: &HookContext,
        event: ToolResultEvent<'_>,
    ) -> ToolResultAction {
        self.record(ctx, "ToolResult");
        self.result_ids
            .lock()
            .expect("result_ids")
            .push(event.internal_call_id.to_string());
        ToolResultAction::keep()
    }
    async fn on_text_delta(&self, ctx: &HookContext, _event: TextDelta<'_>) -> ObservationAction {
        self.record(ctx, "TextDelta");
        ObservationAction::continue_run()
    }
    async fn on_tool_call_delta(
        &self,
        ctx: &HookContext,
        _event: ToolCallDelta<'_>,
    ) -> ObservationAction {
        self.record(ctx, "ToolCallDelta");
        ObservationAction::continue_run()
    }
    async fn on_stream_response_finish(
        &self,
        ctx: &HookContext,
        _event: StreamResponseFinish<'_, M>,
    ) -> ObservationAction {
        self.record(ctx, "StreamResponseFinish");
        ObservationAction::continue_run()
    }
}

/// Registered *after* an [`EventTap`]: reads the shared `Scratchpad` tally on each
/// `ModelTurnFinished` and appends it — proving cross-hook, cross-turn shared
/// state.
#[derive(Clone, Default)]
pub(crate) struct ScratchpadReader {
    tallies: Arc<Mutex<Vec<usize>>>,
}

impl ScratchpadReader {
    pub(crate) fn tallies(&self) -> Vec<usize> {
        self.tallies.lock().expect("tallies").clone()
    }
}

impl<M: CompletionModel> AgentHook<M> for ScratchpadReader {
    async fn on_model_turn_finished(
        &self,
        ctx: &HookContext,
        _event: ModelTurnFinished<'_>,
    ) -> ObservationAction {
        let tally = ctx
            .scratchpad()
            .get::<ToolCallTally>()
            .map(|t| t.0)
            .unwrap_or(0);
        self.tallies.lock().expect("tallies").push(tally);
        ObservationAction::continue_run()
    }
}

// ---------------------------------------------------------------------------
// Steering: flexible request-patch / arg-rewrite / result-rewrite / terminate.
// ---------------------------------------------------------------------------

/// Applies a fixed [`RequestPatch`] on every `CompletionCall` — the workhorse for
/// the `RequestPatch` suite (preamble / temperature / max_tokens / tool_choice /
/// active_tools / additional_params / extra_context / history).
#[derive(Clone)]
pub(crate) struct ApplyPatch(pub(crate) RequestPatch);

impl<M: CompletionModel> AgentHook<M> for ApplyPatch {
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        _event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        CompletionCallAction::patch(self.0.clone())
    }
}

/// Applies a fixed [`RequestPatch`] only on the **first** turn. Use this for
/// per-turn steering that must not repeat forever — e.g. a forced
/// `tool_choice = Required`, which, if re-applied every turn, would force a tool
/// call on every turn and loop until `max_turns`.
#[derive(Clone)]
pub(crate) struct FirstTurnPatch(pub(crate) RequestPatch);

impl<M: CompletionModel> AgentHook<M> for FirstTurnPatch {
    async fn on_completion_call(
        &self,
        ctx: &HookContext,
        _event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        if ctx.turn() == 1 {
            CompletionCallAction::patch(self.0.clone())
        } else {
            CompletionCallAction::continue_run()
        }
    }
}

/// Convenience: a `Document` for `RequestPatch::context`.
pub(crate) fn fact_doc(id: &str, text: &str) -> Document {
    Document {
        id: id.to_string(),
        text: text.to_string(),
        additional_props: Default::default(),
    }
}

/// Sets one key of a named tool's arguments, preserving the rest — chains with
/// other `SetArg`s so several hooks compose.
#[derive(Clone)]
pub(crate) struct SetArg {
    pub(crate) tool: &'static str,
    pub(crate) key: &'static str,
    pub(crate) value: serde_json::Value,
}

impl<M: CompletionModel> AgentHook<M> for SetArg {
    async fn on_tool_call(&self, _ctx: &HookContext, event: ToolCallEvent<'_>) -> ToolCallAction {
        if event.tool_name == self.tool {
            let mut parsed: serde_json::Value =
                serde_json::from_str(event.args).unwrap_or_else(|_| json!({}));
            parsed[self.key] = self.value.clone();
            ToolCallAction::rewrite(parsed)
        } else {
            ToolCallAction::run()
        }
    }
}

/// How a [`RewriteToolResult`] transforms a named tool's output.
#[derive(Clone)]
pub(crate) enum ResultRewrite {
    /// Replace the whole result with a fixed marker.
    Replace(&'static str),
    /// Wrap the (possibly already-rewritten) result — chains with a prior rewrite.
    Wrap {
        prefix: &'static str,
        suffix: &'static str,
    },
    /// Keep only the first `n` characters.
    Truncate(usize),
}

/// Rewrites a named tool's result. Chains with other `RewriteToolResult`s.
#[derive(Clone)]
pub(crate) struct RewriteToolResult {
    pub(crate) tool: &'static str,
    pub(crate) rewrite: ResultRewrite,
}

impl<M: CompletionModel> AgentHook<M> for RewriteToolResult {
    async fn on_tool_result(
        &self,
        _ctx: &HookContext,
        event: ToolResultEvent<'_>,
    ) -> ToolResultAction {
        if event.tool_name != self.tool {
            return ToolResultAction::keep();
        }
        let new = match &self.rewrite {
            ResultRewrite::Replace(marker) => (*marker).to_string(),
            ResultRewrite::Wrap { prefix, suffix } => format!("{prefix}{}{suffix}", event.result),
            ResultRewrite::Truncate(n) => event.result.chars().take(*n).collect(),
        };
        ToolResultAction::rewrite(new)
    }
}

/// Terminates the run when a named tool *produces a result* (post-execution).
#[derive(Clone)]
pub(crate) struct TerminateOnResult {
    pub(crate) tool: &'static str,
    pub(crate) reason: &'static str,
}

impl<M: CompletionModel> AgentHook<M> for TerminateOnResult {
    async fn on_tool_result(
        &self,
        _ctx: &HookContext,
        event: ToolResultEvent<'_>,
    ) -> ToolResultAction {
        if event.tool_name == self.tool {
            ToolResultAction::stop(self.reason)
        } else {
            ToolResultAction::keep()
        }
    }
}
