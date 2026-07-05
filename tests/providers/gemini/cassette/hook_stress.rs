//! Long, multi-turn hook-system stress workflows recorded against real Gemini.
//!
//! Where the small `tool_hooks` suite pins one hook decision each, these tests
//! drive rich multi-turn workflows and assert *structural invariants* of the
//! merged hook system: `HookContext` identity/turn/streaming, a shared
//! `Scratchpad` threaded across hooks and turns, `RequestPatch` context
//! injection + `active_tools` narrowing, chained `RewriteArgs` -> observe ->
//! `RewriteResult` redaction, and streaming lifecycle ordering / blocking-vs-
//! streaming parity.
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

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::Mutex;

use futures::StreamExt;
use rig::agent::{
    AgentHook, Flow, HookContext, MultiTurnStreamItem, RequestPatch, StepEvent, StreamingError,
};
use rig::client::CompletionClient;
use rig::completion::{Document, Prompt};
use rig::providers::gemini;
use rig::streaming::{StreamedAssistantContent, StreamedUserContent, StreamingPrompt};
use rig::tool::Tool;

use super::super::support::with_gemini_cassette;
use super::super::tools_support::{CountingAdd, CountingSubtract, SkipToolHook, ToolEventRecorder};
use crate::support::assert_nonempty_response;

type GeminiModel = gemini::completion::CompletionModel;

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

impl LifecycleRecorder {
    fn breadcrumbs(&self) -> Vec<Breadcrumb> {
        self.breadcrumbs.lock().expect("breadcrumbs").clone()
    }
    fn distinct_run_ids(&self) -> usize {
        self.run_ids.lock().expect("run_ids").len()
    }
    fn is_streaming(&self) -> Option<bool> {
        *self.streaming.lock().expect("streaming")
    }
    fn agent_name(&self) -> Option<String> {
        self.agent_name.lock().expect("agent_name").clone()
    }
    fn count(&self, tag: &str) -> usize {
        self.breadcrumbs()
            .iter()
            .filter(|crumb| crumb.tag == tag)
            .count()
    }
}

impl AgentHook<GeminiModel> for LifecycleRecorder {
    async fn on_event(&self, ctx: &HookContext, event: StepEvent<'_, GeminiModel>) -> Flow {
        let tag = match event {
            StepEvent::CompletionCall { .. } => Some("CompletionCall"),
            StepEvent::CompletionResponse { .. } => Some("CompletionResponse"),
            StepEvent::ModelTurnFinished { .. } => Some("ModelTurnFinished"),
            StepEvent::ToolCall { .. } => Some("ToolCall"),
            StepEvent::ToolResult { .. } => Some("ToolResult"),
            _ => None,
        };
        // Record identity on every event so it is proven stable across the run.
        self.run_ids
            .lock()
            .expect("run_ids")
            .insert(ctx.run_id().as_str().to_string());
        *self.streaming.lock().expect("streaming") = Some(ctx.is_streaming());
        *self.agent_name.lock().expect("agent_name") = ctx.agent_name().map(str::to_string);

        if matches!(event, StepEvent::ToolCall { .. }) {
            ctx.scratchpad()
                .update(|tally: &mut ToolCallTally| tally.0 += 1);
        }

        if let Some(tag) = tag {
            self.breadcrumbs
                .lock()
                .expect("breadcrumbs")
                .push(Breadcrumb {
                    tag,
                    turn: ctx.turn(),
                });
        }
        Flow::cont()
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

impl ScratchpadReader {
    fn tallies(&self) -> Vec<usize> {
        self.tallies.lock().expect("tallies").clone()
    }
}

impl AgentHook<GeminiModel> for ScratchpadReader {
    async fn on_event(&self, ctx: &HookContext, event: StepEvent<'_, GeminiModel>) -> Flow {
        if matches!(event, StepEvent::ModelTurnFinished { .. }) {
            let tally = ctx
                .scratchpad()
                .get::<ToolCallTally>()
                .map(|t| t.0)
                .unwrap_or(0);
            self.tallies.lock().expect("tallies").push(tally);
        }
        Flow::cont()
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

impl AgentHook<GeminiModel> for InjectContextAndNarrowTools {
    async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, GeminiModel>) -> Flow {
        if matches!(event, StepEvent::CompletionCall { .. }) {
            let doc = Document {
                id: self.fact_id.to_string(),
                text: self.fact_text.to_string(),
                additional_props: Default::default(),
            };
            Flow::patch_request(
                RequestPatch::new()
                    .context(doc)
                    .active_tools(self.allow.iter().copied())
                    .temperature(0.0),
            )
        } else {
            Flow::cont()
        }
    }
}

/// `ToolCall` hook that rewrites a named tool's arguments to a fixed object,
/// regardless of what the model emitted (execution-args rewrite).
#[derive(Clone)]
struct ForceArgs {
    tool_name: &'static str,
    args: serde_json::Value,
}

impl AgentHook<GeminiModel> for ForceArgs {
    async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, GeminiModel>) -> Flow {
        match event {
            StepEvent::ToolCall { tool_name, .. } if tool_name == self.tool_name => {
                Flow::rewrite_args(self.args.clone())
            }
            _ => Flow::cont(),
        }
    }
}

/// `ToolResult` hook that redacts a named tool's output with a fixed marker.
#[derive(Clone)]
struct RedactResult {
    tool_name: &'static str,
    marker: &'static str,
}

impl AgentHook<GeminiModel> for RedactResult {
    async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, GeminiModel>) -> Flow {
        match event {
            StepEvent::ToolResult { tool_name, .. } if tool_name == self.tool_name => {
                Flow::rewrite_result(self.marker)
            }
            _ => Flow::cont(),
        }
    }
}

// ---------------------------------------------------------------------------
// 1. HookContext identity + Scratchpad threaded across a long multi-turn run.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn lifecycle_and_scratchpad_thread_across_multi_turn_blocking() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let add_calls = add.counter.clone();
    let subtract_calls = subtract.counter.clone();
    let recorder = LifecycleRecorder::default();
    let reader = ScratchpadReader::default();
    let recorder_probe = recorder.clone();
    let reader_probe = reader.clone();

    with_gemini_cassette(
        "hook_stress/lifecycle_and_scratchpad_thread_across_multi_turn_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(CHAIN_PREAMBLE)
                .temperature(0.0)
                .tool(add)
                .tool(subtract)
                .build();

            let response = agent
                .prompt(
                    "First add 10 and 5 with the add tool. Then subtract 3 from that sum with the \
                     subtract tool. Report the final number.",
                )
                .max_turns(6)
                .add_hook(recorder)
                .add_hook(reader)
                .await
                .expect("dependent multi-turn tool run should succeed");

            assert_nonempty_response(&response);

            // --- HookContext identity is stable and correct across the run ---
            assert_eq!(
                recorder_probe.distinct_run_ids(),
                1,
                "run_id must be stable across every event of one run"
            );
            assert_eq!(
                recorder_probe.is_streaming(),
                Some(false),
                "blocking surface must report is_streaming() == false"
            );
            assert_eq!(
                recorder_probe.agent_name().as_deref(),
                Some("stress-agent"),
                "the configured agent name must reach the hook"
            );

            // --- turn() advances; the workflow really is multi-turn ---
            let crumbs = recorder_probe.breadcrumbs();
            let max_turn = crumbs.iter().map(|c| c.turn).max().unwrap_or(0);
            assert!(
                max_turn >= 2,
                "a dependent add-then-subtract chain must span >= 2 model turns, saw {crumbs:?}"
            );
            let turns: Vec<usize> = crumbs.iter().map(|c| c.turn).collect();
            assert!(
                turns.windows(2).all(|w| w[0] <= w[1]),
                "turn() must be non-decreasing across the run, saw {turns:?}"
            );

            // --- each tool call is paired with a result, and the shared
            //     Scratchpad tally tracks them across hooks and turns ---
            let tool_calls = recorder_probe.count("ToolCall");
            let tool_results = recorder_probe.count("ToolResult");
            assert_eq!(
                tool_calls, tool_results,
                "every observed ToolCall must have a paired ToolResult"
            );
            assert_eq!(
                add_calls.count() + subtract_calls.count(),
                tool_calls,
                "observed ToolCall events must equal real tool executions"
            );
            assert!(
                add_calls.count() >= 1 && subtract_calls.count() >= 1,
                "the chain must exercise both add and subtract"
            );

            // ScratchpadReader (a *different* hook) saw the writer's tally grow to
            // the final ToolCall count — cross-hook, cross-turn shared state.
            let tallies = reader_probe.tallies();
            assert!(
                !tallies.is_empty(),
                "ModelTurnFinished should fire, so the reader should see tallies"
            );
            assert!(
                tallies.windows(2).all(|w| w[0] <= w[1]),
                "the shared scratchpad tally must be non-decreasing, saw {tallies:?}"
            );
            assert_eq!(
                *tallies.last().expect("at least one tally"),
                tool_calls,
                "the final scratchpad tally must equal the total ToolCall count"
            );
        },
    )
    .await;
}

// ---------------------------------------------------------------------------
// 2. RequestPatch: extra_context injection + active_tools narrowing.
// ---------------------------------------------------------------------------

const VAULT_FACT_ID: &str = "vault-note";
const VAULT_FACT: &str = "Operational note: the vault access code is CINNABAR-42.";
const VAULT_CODE: &str = "CINNABAR-42";

#[tokio::test]
async fn request_patch_injects_context_and_narrows_active_tools_blocking() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let add_calls = add.counter.clone();
    let subtract_calls = subtract.counter.clone();

    with_gemini_cassette(
        "hook_stress/request_patch_injects_context_and_narrows_active_tools_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(
                    "You are a helpful assistant. Use a tool for any arithmetic. Consult the \
                     provided context for any facts you are asked about.",
                )
                .tool(add)
                .tool(subtract)
                .build();

            let response = agent
                .prompt(
                    "Two things: (1) tell me the vault access code, and (2) use a tool to compute \
                     41 + 1.",
                )
                .max_turns(5)
                // Inject the secret via extra_context and narrow the advertised
                // tools to `add` only (subtract is filtered out this run).
                .add_hook(InjectContextAndNarrowTools {
                    fact_id: VAULT_FACT_ID,
                    fact_text: VAULT_FACT,
                    allow: &["add"],
                })
                .await
                .expect("context-injecting, tool-narrowing run should succeed");

            // extra_context injection reached the model: the answer uses the fact
            // that appears only in the injected document (no model input).
            assert!(
                response.contains(VAULT_CODE),
                "the extra_context fact must reach the model; answer: {response:?}"
            );
            // active_tools narrowing is proven by the downstream negative: the
            // filtered-out tool never executes, while the advertised one does.
            assert_eq!(
                subtract_calls.count(),
                0,
                "subtract was filtered out of active_tools and must never execute"
            );
            assert!(
                add_calls.count() >= 1,
                "the advertised add tool should still run for 41 + 1"
            );
        },
    )
    .await;
}

// ---------------------------------------------------------------------------
// 3. Chained tool lifecycle: RewriteArgs -> observe -> RewriteResult redaction.
// ---------------------------------------------------------------------------

const REDACTION_MARKER: &str = "REDACTED-SUM-ZK7";

#[tokio::test]
async fn chained_arg_rewrite_then_result_redaction_blocking() {
    let add = CountingAdd::default();
    let recorder = ToolEventRecorder::default();
    let recorder_probe = recorder.clone();

    with_gemini_cassette(
        "hook_stress/chained_arg_rewrite_then_result_redaction_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(
                    "You are a calculator assistant. You MUST use the add tool for the addition. \
                     After the tool result is available, report the exact tool result text \
                     verbatim as your final answer.",
                )
                .temperature(0.0)
                .tool(add)
                .build();

            let response = agent
                .prompt("Use the add tool to add 2 and 2, then report the exact tool result.")
                .max_turns(4)
                // Hook order matters: rewrite args -> observe -> redact result.
                .add_hook(ForceArgs {
                    tool_name: CountingAdd::NAME,
                    args: serde_json::json!({ "x": 7, "y": 8 }),
                })
                .add_hook(recorder)
                .add_hook(RedactResult {
                    tool_name: CountingAdd::NAME,
                    marker: REDACTION_MARKER,
                })
                .await
                .expect("chained rewrite + redaction run should succeed");

            // The observer (registered after the rewriter) saw the *rewritten*
            // args — the tool executed against them, not the model's `2 + 2`.
            let calls = recorder_probe.recorded_calls();
            assert_eq!(calls.len(), 1, "exactly one add call, saw {calls:?}");
            let observed_args: serde_json::Value =
                serde_json::from_str(&calls[0].1).expect("observed args are JSON");
            assert_eq!(
                observed_args,
                serde_json::json!({ "x": 7, "y": 8 }),
                "the observer must see the hook-rewritten args"
            );

            // The observer saw the raw tool output (7 + 8 = 15), *before* the
            // downstream redaction hook replaced it.
            let results = recorder_probe.recorded_results();
            assert_eq!(results.len(), 1, "exactly one add result");
            assert_eq!(
                results[0].2, "15",
                "the observer must see the raw tool output before redaction"
            );

            // Paired positive + negative: the redacted marker reached the model,
            // and the raw executed result (15) did not.
            assert!(
                response.contains(REDACTION_MARKER),
                "the redaction marker must reach the model; answer: {response:?}"
            );
            assert!(
                !response.contains("15"),
                "the raw tool result must not reach the model; answer: {response:?}"
            );
        },
    )
    .await;
}

// ---------------------------------------------------------------------------
// 4. Streaming lifecycle ordering + is_streaming parity vs the blocking surface.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn streaming_lifecycle_ordering_and_context_streaming_flag() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let add_calls = add.counter.clone();
    let subtract_calls = subtract.counter.clone();
    let recorder = LifecycleRecorder::default();
    let recorder_probe = recorder.clone();

    with_gemini_cassette(
        "hook_stress/streaming_lifecycle_ordering_and_context_streaming_flag",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(CHAIN_PREAMBLE)
                .temperature(0.0)
                .tool(add)
                .tool(subtract)
                .build();

            let mut stream = agent
                .stream_prompt(
                    "First add 20 and 5 with the add tool. Then subtract 4 from that sum with the \
                     subtract tool. Report the final number.",
                )
                .add_hook(recorder)
                .multi_turn(6)
                .await;

            // Ordered stream-item taxonomy tags, so we can assert lifecycle order.
            let mut events: Vec<&'static str> = Vec::new();
            let mut saw_final = false;
            let mut final_text = String::new();
            while let Some(item) = stream.next().await {
                match item {
                    Ok(MultiTurnStreamItem::StreamAssistantItem(content)) => match content {
                        StreamedAssistantContent::Text(_) => events.push("text"),
                        StreamedAssistantContent::ToolCall { .. } => events.push("tool_call"),
                        StreamedAssistantContent::ToolCallDelta { .. } => {
                            events.push("tool_call_delta")
                        }
                        _ => {}
                    },
                    Ok(MultiTurnStreamItem::ToolExecutionStart { .. }) => {
                        events.push("tool_execution_start")
                    }
                    Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                        ..
                    })) => events.push("tool_result"),
                    Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                        saw_final = true;
                        final_text = response.response().to_owned();
                        events.push("final_response");
                    }
                    Ok(_) => {}
                    Err(StreamingError::Prompt(error)) => panic!("stream errored: {error:?}"),
                    Err(other) => panic!("stream errored: {other:?}"),
                }
            }

            assert!(saw_final, "the stream must yield a FinalResponse");
            assert_nonempty_response(&final_text);

            // Lifecycle ordering: a tool call precedes its execution start, which
            // precedes its result, which precedes the final response.
            let first = |tag: &str| events.iter().position(|e| *e == tag);
            let tool_call_at = first("tool_call").expect("a complete tool call is surfaced");
            let exec_start_at = first("tool_execution_start").expect("execution start is surfaced");
            let tool_result_at = first("tool_result").expect("a tool result is surfaced");
            let final_at = first("final_response").expect("a final response is surfaced");
            assert!(
                tool_call_at < exec_start_at,
                "the model-emitted tool call must precede its execution start: {events:?}"
            );
            assert!(
                exec_start_at <= tool_result_at,
                "execution start must precede its tool result: {events:?}"
            );
            assert!(
                tool_result_at < final_at,
                "tool results must precede the final response: {events:?}"
            );

            // Same medium-independent lifecycle as the blocking run, plus the
            // streaming flag flips.
            assert_eq!(
                recorder_probe.is_streaming(),
                Some(true),
                "the streaming surface must report is_streaming() == true"
            );
            assert_eq!(
                recorder_probe.distinct_run_ids(),
                1,
                "run_id must be stable across the streamed run too"
            );
            assert_eq!(recorder_probe.agent_name().as_deref(), Some("stress-agent"));
            assert!(
                recorder_probe.count("ModelTurnFinished") >= 2,
                "ModelTurnFinished must fire per accepted turn on the streaming surface"
            );
            assert!(
                add_calls.count() >= 1 && subtract_calls.count() >= 1,
                "the streamed chain must exercise both tools"
            );
        },
    )
    .await;
}

// ---------------------------------------------------------------------------
// 5. Multi-tool workflow: per-turn atomic call/result pairing (batch surfacing).
// ---------------------------------------------------------------------------

#[tokio::test]
async fn multi_tool_workflow_pairs_calls_and_results_per_turn_blocking() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let add_calls = add.counter.clone();
    let subtract_calls = subtract.counter.clone();
    let recorder = LifecycleRecorder::default();
    let recorder_probe = recorder.clone();

    with_gemini_cassette(
        "hook_stress/multi_tool_workflow_pairs_calls_and_results_per_turn_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(
                    "You are a calculator assistant. You MUST use the provided tools for every \
                     arithmetic operation. These two computations are independent — you may request \
                     them together. Once you have both results, report both numbers.",
                )
                .temperature(0.0)
                .tool(add)
                .tool(subtract)
                .build();

            let response = agent
                .prompt(
                    "Independently compute 12 + 8 using the add tool and 30 - 7 using the subtract \
                     tool, then report both results.",
                )
                .max_turns(5)
                .add_hook(recorder)
                .await
                .expect("independent multi-tool run should succeed");

            assert_nonempty_response(&response);
            assert!(
                add_calls.count() >= 1 && subtract_calls.count() >= 1,
                "both independent tools should run"
            );

            // Whether Gemini batches the two calls into one turn or splits them,
            // the atomic tool batch must pair every ToolCall with a ToolResult
            // *within the same turn* — no orphan call, no orphan result.
            let mut per_turn: BTreeMap<usize, (usize, usize)> = BTreeMap::new();
            for crumb in recorder_probe.breadcrumbs() {
                let entry = per_turn.entry(crumb.turn).or_default();
                match crumb.tag {
                    "ToolCall" => entry.0 += 1,
                    "ToolResult" => entry.1 += 1,
                    _ => {}
                }
            }
            for (turn, (calls, results)) in &per_turn {
                assert_eq!(
                    calls, results,
                    "turn {turn} must pair every ToolCall with a ToolResult (atomic batch)"
                );
            }
            assert_eq!(
                recorder_probe.count("ToolCall"),
                add_calls.count() + subtract_calls.count(),
                "observed ToolCall events must equal real tool executions"
            );
        },
    )
    .await;
}

// ---------------------------------------------------------------------------
// 6. Hook Skip in a multi-tool workflow: the skipped tool never executes, yet
//    the run continues to a real answer (skip's zero-execution invariant).
// ---------------------------------------------------------------------------

const SUBTRACT_SKIP_REASON: &str =
    "the subtract tool is offline; treat its result as unavailable and continue";

#[tokio::test]
async fn skip_in_multi_tool_workflow_leaves_tool_unexecuted_blocking() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let add_calls = add.counter.clone();
    let subtract_calls = subtract.counter.clone();

    with_gemini_cassette(
        "hook_stress/skip_in_multi_tool_workflow_leaves_tool_unexecuted_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(
                    "You are a calculator assistant. You MUST use the provided tools for every \
                     arithmetic operation. If a tool reports it is unavailable, acknowledge that in \
                     your answer and still report any results you do have.",
                )
                .temperature(0.0)
                .tool(add)
                .tool(subtract)
                .build();

            let response = agent
                .prompt(
                    "Use the add tool to compute 14 + 6, and use the subtract tool to compute \
                     40 - 9. Report what you can.",
                )
                .max_turns(5)
                // Skip every `subtract` call: its body must never run, but the run
                // continues with the skip reason surfaced as that tool's result.
                .add_hook(SkipToolHook {
                    tool_name: CountingSubtract::NAME,
                    reason: SUBTRACT_SKIP_REASON,
                })
                .await
                .expect("a skipped tool must not fail the run");

            assert_nonempty_response(&response);
            // Zero-execution invariant: the skipped tool's body never ran.
            assert_eq!(
                subtract_calls.count(),
                0,
                "the skipped subtract tool must never execute"
            );
            // The other tool still ran, so the run made real progress.
            assert!(
                add_calls.count() >= 1,
                "the non-skipped add tool should still execute"
            );
        },
    )
    .await;
}

// Compile-time proof the fixtures implement the hook trait for the Gemini model.
#[allow(unused)]
fn assert_hook_impls() {
    fn requires_hook<H: AgentHook<GeminiModel>>(_hook: H) {}
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
