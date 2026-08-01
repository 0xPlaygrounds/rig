//! Long, multi-turn hook-system stress workflows recorded against real Gemini.
//!
//! Where the small `tool_hooks` suite pins one hook decision each, these tests
//! drive rich multi-turn workflows and assert *structural invariants* of the
//! merged hook system: turn advancement, host state shared across hooks and
//! turns, `RequestPatch` context
//! injection + `active_tools` narrowing, chained `ToolCallAction::Rewrite` -> observe ->
//! `ToolResultAction::Rewrite` redaction, and streaming lifecycle ordering / blocking-vs-
//! streaming parity.
//!
//! ## On loose assertions
//!
//! Following `tools_support`'s convention: only values Rig synthesizes with **no
//! model input** (a hook-rewritten arg, a verbatim redaction marker, a turn
//! index, a shared tally, an event *shape*) are pinned to exact
//! equality. Everything shaped by Gemini's generated text or its chosen call
//! count/ordering uses loose assertions (`contains`, `>=`, "mentions"), so these
//! cassettes survive re-recording. Deterministic hooks (no clocks/RNG) keep the
//! outbound requests byte-identical for replay.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Mutex;

use futures::StreamExt;
use rig::agent::{
    CompletionCallAction, ModelTurnAction, ObservationAction, RequestPatch, ToolCallAction,
    ToolResultAction,
};
use rig::completion::Document;
use rig::hooks::{HookDecision, HookEntry, HookEvent};
use rig::providers::gemini;
use rig::stream::AgentStreamItem;
use rig::streaming::{StreamedAssistantContent, StreamedUserContent};
use rig::tool::Tool;

use super::super::support::with_gemini_cassette;
use super::super::tools_support::{
    CountingAdd, CountingSubtract, ToolEventRecorder, skip_tool_hook,
};
use crate::support::assert_nonempty_response;

/// Preamble that forces tool use and a dependent two-step chain so the model
/// takes at least two turns (compute A, then use A to compute B).
const CHAIN_PREAMBLE: &str = "You are a calculator assistant. You MUST use the provided tools for \
     every arithmetic operation instead of computing results yourself. Perform the steps in order, \
     using the result of each step as an input to the next. Once you have the final tool result, \
     reply with the final numeric answer in plain text.";

// ---------------------------------------------------------------------------
// Fixtures: hooks that observe the lifecycle, thread host-owned shared state,
// and steer requests/tools. All deterministic.
// ---------------------------------------------------------------------------

/// One observed hook event: its variant tag and the one-based turn it fired on.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Breadcrumb {
    tag: &'static str,
    turn: usize,
}

/// Cross-hook, cross-turn shared state: how many `ToolCall`s the writer hook
/// has seen so far this run. Hooks are attach-and-forget closures, so the host
/// owns this state and each hook captures the handle it needs.
#[derive(Clone, Default)]
struct ToolCallTally(Arc<Mutex<usize>>);

impl ToolCallTally {
    fn get(&self) -> usize {
        *self.0.lock().expect("tally")
    }
    fn bump(&self) {
        *self.0.lock().expect("tally") += 1;
    }
}

/// Records the ordered lifecycle breadcrumb `(tag, turn)` for the whole run and
/// bumps a shared [`ToolCallTally`] on each `ToolCall`.
///
/// Tool events carry no turn index of their own, so the recorder tracks the
/// current turn from the turn-bearing events (each turn opens with
/// `BeforeModelCall`).
#[derive(Clone, Default)]
struct LifecycleRecorder {
    breadcrumbs: Arc<Mutex<Vec<Breadcrumb>>>,
    current_turn: Arc<Mutex<usize>>,
    tally: ToolCallTally,
}

impl LifecycleRecorder {
    fn breadcrumbs(&self) -> Vec<Breadcrumb> {
        self.breadcrumbs.lock().expect("breadcrumbs").clone()
    }
    fn count(&self, tag: &str) -> usize {
        self.breadcrumbs()
            .iter()
            .filter(|crumb| crumb.tag == tag)
            .count()
    }
    fn tally(&self) -> ToolCallTally {
        self.tally.clone()
    }

    fn record(&self, tag: &'static str, turn: Option<usize>) {
        let turn = match turn {
            Some(turn) => {
                *self.current_turn.lock().expect("current_turn") = turn;
                turn
            }
            None => *self.current_turn.lock().expect("current_turn"),
        };
        self.breadcrumbs
            .lock()
            .expect("breadcrumbs")
            .push(Breadcrumb { tag, turn });
    }

    fn entry(&self) -> HookEntry {
        let recorder = self.clone();
        HookEntry::sync("lifecycle-recorder", move |event| match event {
            HookEvent::BeforeModelCall { turn, .. } => {
                recorder.record("CompletionCall", Some(turn));
                HookDecision::CompletionCall(CompletionCallAction::continue_run())
            }
            HookEvent::CompletionResponse { turn, .. } => {
                recorder.record("CompletionResponse", Some(turn));
                HookDecision::Observation(ObservationAction::continue_run())
            }
            HookEvent::ModelTurnFinished { turn, .. } => {
                recorder.record("ModelTurnFinished", Some(turn));
                HookDecision::ModelTurn(ModelTurnAction::continue_run())
            }
            HookEvent::StreamResponseFinish { turn, .. } => {
                recorder.record("StreamResponseFinish", Some(turn));
                HookDecision::Observation(ObservationAction::continue_run())
            }
            HookEvent::ToolCall { .. } => {
                recorder.record("ToolCall", None);
                recorder.tally.bump();
                HookDecision::ToolCall(ToolCallAction::run())
            }
            HookEvent::ToolResult { .. } => {
                recorder.record("ToolResult", None);
                HookDecision::ToolResult(ToolResultAction::keep())
            }
            _ => HookDecision::Continue,
        })
    }
}

/// Registered *after* [`LifecycleRecorder`]: on each `ModelTurnFinished` it
/// reads the shared tally the recorder wrote and appends it to an external log
/// — proving the two hooks share state that accumulates across turns.
#[derive(Clone, Default)]
struct TallyReader {
    tallies: Arc<Mutex<Vec<usize>>>,
}

impl TallyReader {
    fn tallies(&self) -> Vec<usize> {
        self.tallies.lock().expect("tallies").clone()
    }

    fn entry(&self, tally: ToolCallTally) -> HookEntry {
        let tallies = self.tallies.clone();
        HookEntry::new("tally-reader", move |event| {
            if matches!(event, HookEvent::ModelTurnFinished { .. }) {
                tallies.lock().expect("tallies").push(tally.get());
                return Box::pin(async {
                    HookDecision::ModelTurn(ModelTurnAction::continue_run())
                });
            }
            Box::pin(async { HookDecision::Continue })
        })
    }
}

/// `CompletionCall` hook that injects a run-state fact via `extra_context`,
/// narrows `active_tools`, and pins temperature — one merged `RequestPatch`.
fn inject_context_and_narrow_tools(
    fact_id: &'static str,
    fact_text: &'static str,
    allow: &'static [&'static str],
) -> HookEntry {
    HookEntry::sync("inject-context-narrow-tools", move |event| match event {
        HookEvent::BeforeModelCall { .. } => {
            let doc = Document {
                id: fact_id.to_string(),
                text: fact_text.to_string(),
                additional_props: Default::default(),
            };
            HookDecision::CompletionCall(CompletionCallAction::patch(
                RequestPatch::new()
                    .context(doc)
                    .active_tools(allow.iter().copied())
                    .temperature(0.0),
            ))
        }
        _ => HookDecision::Continue,
    })
}

/// `ToolCall` hook that rewrites a named tool's arguments to a fixed object,
/// regardless of what the model emitted (execution-args rewrite).
fn force_args(tool_name: &'static str, args: serde_json::Value) -> HookEntry {
    HookEntry::sync("force-args", move |event| match event {
        HookEvent::ToolCall { call, .. } if call.function.name == tool_name => {
            HookDecision::ToolCall(ToolCallAction::rewrite(args.clone()))
        }
        HookEvent::ToolCall { .. } => HookDecision::ToolCall(ToolCallAction::run()),
        _ => HookDecision::Continue,
    })
}

/// `ToolResult` hook that redacts a named tool's output with a fixed marker.
fn redact_result(tool_name: &'static str, marker: &'static str) -> HookEntry {
    HookEntry::sync("redact-result", move |event| match event {
        HookEvent::ToolResult { call, .. } if call.function.name == tool_name => {
            HookDecision::ToolResult(ToolResultAction::rewrite(marker))
        }
        HookEvent::ToolResult { .. } => HookDecision::ToolResult(ToolResultAction::keep()),
        _ => HookDecision::Continue,
    })
}

// ---------------------------------------------------------------------------
// 1. Turn advancement + shared host state threaded across a long multi-turn run.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn lifecycle_and_shared_state_thread_across_multi_turn_blocking() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let add_calls = add.counter.clone();
    let subtract_calls = subtract.counter.clone();
    let recorder = LifecycleRecorder::default();
    let reader = TallyReader::default();
    let recorder_probe = recorder.clone();
    let reader_probe = reader.clone();
    let recorder_entry = recorder.entry();
    let reader_entry = reader.entry(recorder.tally());

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
                .runner(
                    "First add 10 and 5 with the add tool. Then subtract 3 from that sum with the \
                     subtract tool. Report the final number.",
                )
                .max_turns(6)
                .add_hook(recorder_entry)
                .add_hook(reader_entry)
                .run()
                .await
                .map(|response| response.output)
                .expect("dependent multi-turn tool run should succeed");

            assert_nonempty_response(&response);

            // --- the blocking surface's medium-specific event set: it fires
            //     CompletionResponse and never the streamed response-finish ---
            assert!(
                recorder_probe.count("CompletionResponse") >= 1,
                "the blocking surface must fire CompletionResponse"
            );
            assert_eq!(
                recorder_probe.count("StreamResponseFinish"),
                0,
                "the blocking surface must never fire StreamResponseFinish"
            );

            // --- the turn index advances; the workflow really is multi-turn ---
            let crumbs = recorder_probe.breadcrumbs();
            let max_turn = crumbs.iter().map(|c| c.turn).max().unwrap_or(0);
            assert!(
                max_turn >= 2,
                "a dependent add-then-subtract chain must span >= 2 model turns, saw {crumbs:?}"
            );
            let turns: Vec<usize> = crumbs.iter().map(|c| c.turn).collect();
            assert!(
                turns.windows(2).all(|w| w[0] <= w[1]),
                "the turn index must be non-decreasing across the run, saw {turns:?}"
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

            // TallyReader (a *different* hook) saw the writer's tally grow to
            // the final ToolCall count — cross-hook, cross-turn shared state.
            let tallies = reader_probe.tallies();
            assert!(
                !tallies.is_empty(),
                "ModelTurnFinished should fire, so the reader should see tallies"
            );
            assert!(
                tallies.windows(2).all(|w| w[0] <= w[1]),
                "the shared tally must be non-decreasing, saw {tallies:?}"
            );
            assert_eq!(
                *tallies.last().expect("at least one tally"),
                tool_calls,
                "the final shared tally must equal the total ToolCall count"
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
                .runner(
                    "Two things: (1) tell me the vault access code, and (2) use a tool to compute \
                     41 + 1.",
                )
                .max_turns(5)
                // Inject the secret via extra_context and narrow the advertised
                // tools to `add` only (subtract is filtered out this run).
                .add_hook(inject_context_and_narrow_tools(
                    VAULT_FACT_ID,
                    VAULT_FACT,
                    &["add"],
                ))
                .run()
                .await
                .map(|response| response.output)
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
// 3. Chained tool lifecycle: ToolCallAction::Rewrite -> observe -> ToolResultAction::Rewrite.
// ---------------------------------------------------------------------------

const REDACTION_MARKER: &str = "REDACTED-SUM-ZK7";

#[tokio::test]
async fn chained_arg_rewrite_then_result_redaction_blocking() {
    let add = CountingAdd::default();
    let recorder = ToolEventRecorder::default();
    let recorder_probe = recorder.clone();
    let recorder_entry = recorder.entry();

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
                .runner("Use the add tool to add 2 and 2, then report the exact tool result.")
                .max_turns(4)
                // Hook order matters: rewrite args -> observe -> redact result.
                .add_hook(force_args(
                    CountingAdd::NAME,
                    serde_json::json!({ "x": 7, "y": 8 }),
                ))
                .add_hook(recorder_entry)
                .add_hook(redact_result(CountingAdd::NAME, REDACTION_MARKER))
                .run()
                .await
                .map(|response| response.output)
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
// 4. Streaming lifecycle ordering + medium parity vs the blocking surface.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn streaming_lifecycle_ordering_and_medium_specific_events() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let add_calls = add.counter.clone();
    let subtract_calls = subtract.counter.clone();
    let recorder = LifecycleRecorder::default();
    let recorder_probe = recorder.clone();
    let recorder_entry = recorder.entry();

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
                .runner(
                    "First add 20 and 5 with the add tool. Then subtract 4 from that sum with the \
                     subtract tool. Report the final number.",
                )
                .add_hook(recorder_entry)
                .max_turns(6)
                .stream_run();

            // Ordered stream-item taxonomy tags, so we can assert lifecycle order.
            let mut events: Vec<&'static str> = Vec::new();
            let mut saw_final = false;
            let mut final_text = String::new();
            while let Some(item) = stream.next().await {
                match item {
                    Ok(AgentStreamItem::Assistant(content)) => match content {
                        StreamedAssistantContent::Text(_) => events.push("text"),
                        StreamedAssistantContent::ToolCall { .. } => events.push("tool_call"),
                        StreamedAssistantContent::ToolCallDelta { .. } => {
                            events.push("tool_call_delta")
                        }
                        _ => {}
                    },
                    Ok(AgentStreamItem::ToolExecutionCommitted { .. }) => {
                        events.push("tool_execution_committed")
                    }
                    Ok(AgentStreamItem::User(StreamedUserContent::ToolResult { .. })) => {
                        events.push("tool_result")
                    }
                    Ok(AgentStreamItem::Final(response)) => {
                        saw_final = true;
                        final_text = response.output().to_owned();
                        events.push("final_response");
                    }
                    Ok(_) => {}
                    Err(error) => panic!("stream errored: {error:?}"),
                }
            }

            assert!(saw_final, "the stream must yield a FinalResponse");
            assert_nonempty_response(&final_text);

            // Lifecycle ordering: a tool call precedes its execution commit, which
            // precedes its result, which precedes the final response.
            let first = |tag: &str| events.iter().position(|e| *e == tag);
            let tool_call_at = first("tool_call").expect("a complete tool call is surfaced");
            let exec_commit_at =
                first("tool_execution_committed").expect("execution commit is surfaced");
            let tool_result_at = first("tool_result").expect("a tool result is surfaced");
            let final_at = first("final_response").expect("a final response is surfaced");
            assert!(
                tool_call_at < exec_commit_at,
                "the model-emitted tool call must precede its execution commit: {events:?}"
            );
            assert!(
                exec_commit_at <= tool_result_at,
                "execution commit must precede its tool result: {events:?}"
            );
            assert!(
                tool_result_at < final_at,
                "tool results must precede the final response: {events:?}"
            );

            // Same medium-independent lifecycle as the blocking run, plus the
            // streaming surface's own response-finish event (and never the
            // blocking surface's CompletionResponse).
            assert!(
                recorder_probe.count("StreamResponseFinish") >= 1,
                "the streaming surface must fire StreamResponseFinish"
            );
            assert_eq!(
                recorder_probe.count("CompletionResponse"),
                0,
                "the streaming surface must never fire the blocking CompletionResponse"
            );
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
    let recorder_entry = recorder.entry();

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
                .runner(
                    "Independently compute 12 + 8 using the add tool and 30 - 7 using the subtract \
                     tool, then report both results.",
                )
                .max_turns(5)
                .add_hook(recorder_entry).run()
                .await.map(|response| response.output)
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
                .runner(
                    "Use the add tool to compute 14 + 6, and use the subtract tool to compute \
                     40 - 9. Report what you can.",
                )
                .max_turns(5)
                // Skip every `subtract` call: its body must never run, but the run
                // continues with the skip reason surfaced as that tool's result.
                .add_hook(skip_tool_hook(
                    CountingSubtract::NAME,
                    SUBTRACT_SKIP_REASON,
                )).run()
                .await.map(|response| response.output)
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

// Compile-time proof every fixture yields a hook record.
#[allow(unused)]
fn assert_hook_records() {
    fn requires_entry(_entry: HookEntry) {}
    let recorder = LifecycleRecorder::default();
    requires_entry(recorder.entry());
    requires_entry(TallyReader::default().entry(recorder.tally()));
    requires_entry(inject_context_and_narrow_tools("", "", &[]));
    requires_entry(force_args("add", serde_json::Value::Null));
    requires_entry(redact_result("add", ""));
}
use rig::prelude::*;
