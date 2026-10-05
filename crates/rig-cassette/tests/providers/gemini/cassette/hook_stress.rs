//! Long, multi-turn hook-system stress workflows recorded against real Gemini.
//!
//! These tests drive rich multi-turn workflows and assert *structural
//! invariants* of the merged hook system: `HookContext` identity/turn/streaming,
//! a shared `Scratchpad` threaded across hooks and turns, `RequestPatch`
//! context injection + `active_tools` narrowing, chained
//! `DispatchAction::Patch` -> observe -> `OutcomeAction::Replace` redaction,
//! and streaming lifecycle ordering / blocking-vs-streaming parity.
//!
//! ## On loose assertions
//!
//! Following `tools_support`'s convention: only values Rig synthesizes with **no
//! model input** (a hook-rewritten arg, a verbatim redaction marker, a
//! `HookContext` field, a scratchpad tally, an event *shape*) are pinned to exact
//! equality. Everything shaped by Gemini's generated text or its chosen call
//! count/ordering uses loose assertions (`contains`, `>=`, "mentions"), so these
//! cassettes survive re-recording. Deterministic hooks (no clocks/RNG) keep the
//! outbound requests byte-identical for replay.

use rig_cassette::agent::AgentReplayExt;
use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::Mutex;

use futures::StreamExt;
use rig::agent::{
    AgentHook, CompletionCallAction, CompletionCallEvent, DispatchAction, DispatchEvent,
    HookContext, ModelTurnAction, ModelTurnFinished, MultiTurnStreamItem, OutcomeAction,
    OutcomeEvent, RequestPatch,
};
use rig::completion::Document;
use rig::providers::gemini;

use super::super::support::with_gemini_cassette;
use super::super::tools_support::{CountingAdd, CountingSubtract};

/// Preamble that forces tool use and a dependent two-step chain so the model
/// takes at least two turns (compute A, then use A to compute B).
const CHAIN_PREAMBLE: &str = "You are a calculator assistant. You MUST use the provided tools for \
     every arithmetic operation instead of computing results yourself. Perform the steps in order, \
     using the result of each step as an input to the next. Once you have the final tool result, \
     reply with the final numeric answer in plain text.";

// ---------------------------------------------------------------------------
// Fixtures: hooks that observe HookContext identity, thread the Scratchpad, and
// steer requests/tools. All deterministic.
// ---------------------------------------------------------------------------

/// One observed hook event: its variant tag and the one-based turn it fired on.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Breadcrumb {
    tag: &'static str,
    turn: usize,
}

/// Cross-hook, cross-turn scratchpad value: how many `ToolCall`s the writer hook
/// has seen so far this run.
#[derive(Clone, Default)]
struct ToolCallTally(usize);

/// Records, for the whole run: the ordered lifecycle breadcrumb, the set of
/// `run_id`s seen, the `is_streaming` flag, and the `agent_name` — proving
/// `HookContext` identity is stable and correct. Also bumps a shared
/// `Scratchpad` tally on each `ToolCall`.
#[derive(Clone, Default)]
struct LifecycleRecorder {
    breadcrumbs: Arc<Mutex<Vec<Breadcrumb>>>,
    run_ids: Arc<Mutex<BTreeSet<String>>>,
    streaming: Arc<Mutex<Option<bool>>>,
    agent_name: Arc<Mutex<Option<String>>>,
}

impl LifecycleRecorder {}

impl LifecycleRecorder {
    fn record(&self, ctx: &HookContext, tag: &'static str) {
        self.run_ids
            .lock()
            .expect("run_ids")
            .insert(ctx.run_id().to_string());
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
impl AgentHook for LifecycleRecorder {
    async fn on_completion_call(
        &self,
        ctx: &HookContext,
        _event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        self.record(ctx, "CompletionCall");
        CompletionCallAction::continue_run()
    }
    async fn on_model_turn_finished(
        &self,
        ctx: &HookContext,
        _event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        self.record(ctx, "ModelTurnFinished");
        ModelTurnAction::continue_run()
    }
    async fn on_dispatch(&self, ctx: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        if event.tool_name().is_none() {
            return DispatchAction::proceed();
        }
        self.record(ctx, "ToolCall");
        ctx.scratchpad()
            .update(|tally: &mut ToolCallTally| tally.0 += 1);
        DispatchAction::proceed()
    }
    async fn on_outcome(&self, ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        if event.completion().is_some() {
            self.record(ctx, "CompletionResponse");
        } else if event.tool_name().is_some() {
            self.record(ctx, "ToolResult");
        }
        OutcomeAction::proceed()
    }
}

/// Registered *after* [`LifecycleRecorder`]: on each `ModelTurnFinished` it reads
/// the shared `Scratchpad` tally the recorder wrote and appends it to an external
/// log — proving the two hooks share run-scoped state that accumulates across
/// turns.
#[derive(Clone, Default)]
struct ScratchpadReader {
    tallies: Arc<Mutex<Vec<usize>>>,
}

impl ScratchpadReader {}

impl AgentHook for ScratchpadReader {
    async fn on_model_turn_finished(
        &self,
        ctx: &HookContext,
        _event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        let tally = ctx.scratchpad().get::<ToolCallTally>().map_or(0, |t| t.0);
        self.tallies.lock().expect("tallies").push(tally);
        ModelTurnAction::continue_run()
    }
}

/// `CompletionCall` hook that injects a run-state fact via `extra_context`,
/// narrows `active_tools`, and pins temperature — one merged `RequestPatch`.
#[derive(Clone)]
struct InjectContextAndNarrowTools {
    fact_id: &'static str,
    fact_text: &'static str,
    allow: &'static [&'static str],
}

impl AgentHook for InjectContextAndNarrowTools {
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        _event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        let doc = Document {
            id: self.fact_id.to_string(),
            text: self.fact_text.to_string(),
            additional_props: Default::default(),
        };
        CompletionCallAction::patch(
            RequestPatch::new()
                .context(doc)
                .active_tools(self.allow.iter().copied())
                .temperature(0.0),
        )
    }
}

/// `ToolCall` hook that rewrites a named tool's arguments to a fixed object,
/// regardless of what the model emitted (execution-args rewrite).
#[derive(Clone)]
struct ForceArgs {
    tool_name: &'static str,
    args: serde_json::Value,
}

impl AgentHook for ForceArgs {
    async fn on_dispatch(&self, _ctx: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        if event.tool_name() == Some(self.tool_name) {
            DispatchAction::rewrite_tool_args(event.kind, self.args.clone())
        } else {
            DispatchAction::proceed()
        }
    }
}

/// `ToolResult` hook that redacts a named tool's output with a fixed marker.
#[derive(Clone)]
struct RedactResult {
    tool_name: &'static str,
    marker: &'static str,
}

impl AgentHook for RedactResult {
    async fn on_outcome(&self, _ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        if event.tool_name() == Some(self.tool_name) {
            OutcomeAction::rewrite_tool_result(&event, self.marker)
        } else {
            OutcomeAction::proceed()
        }
    }
}

// ---------------------------------------------------------------------------
// 1. HookContext identity + Scratchpad threaded across a long multi-turn run.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 2. RequestPatch: extra_context injection + active_tools narrowing.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 3. Chained tool lifecycle: DispatchAction::Patch -> observe -> OutcomeAction::Replace.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 4. Streaming lifecycle ordering + is_streaming parity vs the blocking surface.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 5. Multi-tool workflow: per-turn atomic call/result pairing (batch surfacing).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 6. Hook Skip in a multi-tool workflow: the skipped tool never executes, yet
//    the run continues to a real answer (skip's zero-execution invariant).
// ---------------------------------------------------------------------------

// Compile-time proof the fixtures implement the hook trait for the Gemini model.
#[allow(unused)]
fn assert_hook_impls() {
    fn requires_hook<H: AgentHook>(_hook: H) {}
    requires_hook(LifecycleRecorder::default());
    requires_hook(ScratchpadReader::default());
    requires_hook(InjectContextAndNarrowTools {
        fact_id: "",
        fact_text: "",
        allow: &[],
    });
    requires_hook(ForceArgs {
        tool_name: "add",
        args: serde_json::Value::Null,
    });
    requires_hook(RedactResult {
        tool_name: "add",
        marker: "",
    });
}

/// Golden `gemini_tool_call_turns`: two tool turns on a wire that carries
/// no tool-call ids. Every id in the log is minted from the block that
/// assembled the call, so the record is the same on every run — the proof
/// that nothing the engine mints is random.
#[tokio::test]
async fn tool_call_turns_effect_log_is_the_golden_fixture() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    with_gemini_cassette(
        "hook_stress/streaming_lifecycle_ordering_and_context_streaming_flag",
        |client| async move {
            let recorder = rig_cassette::effect_log::EffectLogRecorder::new();
            let agent =
                rig::AgentBuilder::new(client.completion(gemini::completion::GEMINI_2_5_FLASH))
                    .name("stress-agent")
                    .preamble(CHAIN_PREAMBLE)
                    .temperature(0.0)
                    .tool(add)
                    .tool(subtract)
                    .record_to(recorder.clone())
                    .build();
            let mut stream = agent
                .prompt(
                    "First add 20 and 5 with the add tool. Then subtract 4 from that sum with the \
                     subtract tool. Report the final number.",
                )
                .max_turns(6)
                .stream();
            let mut saw_final = false;
            while let Some(item) = stream.next().await {
                if let Ok(MultiTurnStreamItem::FinalResponse(_)) = item {
                    saw_final = true;
                }
            }
            assert!(saw_final, "the stream must yield a FinalResponse");
            let log = agent.stamp(recorder.take());
            let tool_ids: Vec<&rig::message::CallId> = log
                .records
                .iter()
                .filter_map(|record| match &record.outcome {
                    Ok(rig::effect::Outcome::Completion(response)) => Some(response),
                    _ => None,
                })
                .flat_map(|response| response.choice.iter())
                .filter_map(|content| match content {
                    rig::message::AssistantContent::ToolCall(call) => Some(&call.id),
                    _ => None,
                })
                .collect();
            assert!(!tool_ids.is_empty(), "the program calls tools");
            assert!(
                tool_ids.iter().all(|id| id.is_local()),
                "every id-less wire call is named by its block: {tool_ids:?}"
            );
            crate::goldens::golden_effects("gemini_tool_call_turns", &log);
        },
    )
    .await;
}
