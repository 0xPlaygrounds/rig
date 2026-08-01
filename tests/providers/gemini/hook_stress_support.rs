//! Shared fixtures for the hook-system stress cassette suites
//! (`cassette::hook_stress*`): a comprehensive event-tap observer, flexible
//! request-patch / arg-rewrite / result-rewrite / terminate hooks, shared
//! cross-hook state probes, and a third counting arithmetic tool.
//!
//! Every hook here is **deterministic** (no clocks/RNG), so the outbound
//! requests it produces stay byte-identical across replay. See
//! `tools_support`'s note on loose assertions: only rig-synthesized values are
//! pinned to exact equality.
#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};

use std::sync::{Arc, Mutex};

use rig::agent::{
    CompletionCallAction, ModelTurnAction, ObservationAction, RequestPatch, ToolCallAction,
    ToolResultAction,
};
use rig::completion::Document;
use rig::hooks::{HookDecision, HookEntry, HookEvent};
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

/// Forces tool use and a dependent chain so the model takes >= 2 turns.
pub(crate) const CHAIN_PREAMBLE: &str = "You are a calculator assistant. You MUST use the provided \
     tools for every arithmetic operation instead of computing results yourself. Perform the steps \
     in order, using the result of each step as an input to the next. Once you have the final tool \
     result, reply with the final numeric answer in plain text.";

#[derive(Debug, thiserror::Error)]
#[error("math error")]
pub(crate) struct MathError;

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
    type Error = MathError;
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

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
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

/// Cross-hook, cross-turn shared state: how many `ToolCall`s have been seen.
/// The host owns it now (hooks are attach-and-forget closures that capture
/// whatever state they share), replacing the old run-scoped scratchpad.
#[derive(Clone, Default)]
pub(crate) struct ToolCallTally(Arc<Mutex<usize>>);

impl ToolCallTally {
    pub(crate) fn get(&self) -> usize {
        *self.0.lock().expect("tally")
    }
    fn bump(&self) {
        *self.0.lock().expect("tally") += 1;
    }
}

/// A comprehensive observe-only hook: records an ordered `(tag, turn)`
/// breadcrumb for every event kind, captures each tool call's and result's
/// `internal_call_id` for correlation, and bumps a shared [`ToolCallTally`] on
/// every `ToolCall`.
///
/// Tool events carry no turn index of their own, so the tap tracks the current
/// turn from the turn-bearing events (a turn opens with `BeforeModelCall`),
/// exactly reproducing what the old `HookContext::turn()` reported.
#[derive(Clone, Default)]
pub(crate) struct EventTap {
    breadcrumbs: Arc<Mutex<Vec<Breadcrumb>>>,
    call_ids: Arc<Mutex<Vec<String>>>,
    result_ids: Arc<Mutex<Vec<String>>>,
    current_turn: Arc<Mutex<usize>>,
    tally: ToolCallTally,
}

impl EventTap {
    pub(crate) fn breadcrumbs(&self) -> Vec<Breadcrumb> {
        self.breadcrumbs.lock().expect("breadcrumbs").clone()
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
    /// The shared tally this tap writes; hand it to [`tally_reader`] to prove
    /// cross-hook, cross-turn shared state.
    pub(crate) fn tally(&self) -> ToolCallTally {
        self.tally.clone()
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

    /// The observe-only hook record. Opted into delta observation so the
    /// streaming suites still see `TextDelta` / `ToolCallDelta`.
    pub(crate) fn entry(&self) -> HookEntry {
        let tap = self.clone();
        HookEntry::sync("event-tap", move |event| tap.observe(event)).observing_deltas()
    }

    fn observe(&self, event: HookEvent) -> HookDecision {
        match event {
            HookEvent::BeforeModelCall { turn, .. } => {
                self.record("CompletionCall", Some(turn));
                HookDecision::CompletionCall(CompletionCallAction::continue_run())
            }
            HookEvent::CompletionResponse { turn, .. } => {
                self.record("CompletionResponse", Some(turn));
                HookDecision::Observation(ObservationAction::continue_run())
            }
            HookEvent::ModelTurnFinished { turn, .. } => {
                self.record("ModelTurnFinished", Some(turn));
                HookDecision::ModelTurn(ModelTurnAction::continue_run())
            }
            HookEvent::InvalidToolCall(_) => {
                self.record("InvalidToolCall", None);
                // `None` in the classic hook: defer to a later entry.
                HookDecision::Continue
            }
            HookEvent::ToolCall {
                internal_call_id, ..
            } => {
                self.record("ToolCall", None);
                self.call_ids
                    .lock()
                    .expect("call_ids")
                    .push(internal_call_id);
                self.tally.bump();
                HookDecision::ToolCall(ToolCallAction::run())
            }
            HookEvent::ToolResult {
                internal_call_id, ..
            } => {
                self.record("ToolResult", None);
                self.result_ids
                    .lock()
                    .expect("result_ids")
                    .push(internal_call_id);
                HookDecision::ToolResult(ToolResultAction::keep())
            }
            HookEvent::TextDelta { turn, .. } => {
                self.record("TextDelta", Some(turn));
                HookDecision::Observation(ObservationAction::continue_run())
            }
            HookEvent::ToolCallDelta { turn, .. } => {
                self.record("ToolCallDelta", Some(turn));
                HookDecision::Observation(ObservationAction::continue_run())
            }
            HookEvent::StreamResponseFinish { turn, .. } => {
                self.record("StreamResponseFinish", Some(turn));
                HookDecision::Observation(ObservationAction::continue_run())
            }
            _ => HookDecision::Continue,
        }
    }
}

/// Registered *after* an [`EventTap`]: reads the shared [`ToolCallTally`] on
/// each `ModelTurnFinished` and appends it — proving cross-hook, cross-turn
/// shared state (the old `ScratchpadReader`, with the state owned by the host).
#[derive(Clone, Default)]
pub(crate) struct TallyReader {
    tallies: Arc<Mutex<Vec<usize>>>,
}

impl TallyReader {
    pub(crate) fn tallies(&self) -> Vec<usize> {
        self.tallies.lock().expect("tallies").clone()
    }
}

/// The reader's hook record over `tally`.
pub(crate) fn tally_reader(reader: &TallyReader, tally: ToolCallTally) -> HookEntry {
    let tallies = reader.tallies.clone();
    HookEntry::sync("tally-reader", move |event| {
        if matches!(event, HookEvent::ModelTurnFinished { .. }) {
            tallies.lock().expect("tallies").push(tally.get());
            return HookDecision::ModelTurn(ModelTurnAction::continue_run());
        }
        HookDecision::Continue
    })
}

// ---------------------------------------------------------------------------
// Steering: flexible request-patch / arg-rewrite / result-rewrite / terminate.
// ---------------------------------------------------------------------------

/// Applies a fixed [`RequestPatch`] on every `CompletionCall` — the workhorse for
/// the `RequestPatch` suite (preamble / temperature / max_tokens / tool_choice /
/// active_tools / additional_params / extra_context / history).
pub(crate) fn apply_patch(patch: RequestPatch) -> HookEntry {
    HookEntry::sync("apply-patch", move |event| match event {
        HookEvent::BeforeModelCall { .. } => {
            HookDecision::CompletionCall(CompletionCallAction::patch(patch.clone()))
        }
        _ => HookDecision::Continue,
    })
}

/// Applies a fixed [`RequestPatch`] only on the **first** turn. Use this for
/// per-turn steering that must not repeat forever — e.g. a forced
/// `tool_choice = Required`, which, if re-applied every turn, would force a tool
/// call on every turn and loop until `max_turns`.
pub(crate) fn first_turn_patch(patch: RequestPatch) -> HookEntry {
    HookEntry::sync("first-turn-patch", move |event| match event {
        HookEvent::BeforeModelCall { turn: 1, .. } => {
            HookDecision::CompletionCall(CompletionCallAction::patch(patch.clone()))
        }
        HookEvent::BeforeModelCall { .. } => {
            HookDecision::CompletionCall(CompletionCallAction::continue_run())
        }
        _ => HookDecision::Continue,
    })
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
/// other `set_arg`s so several hooks compose.
pub(crate) fn set_arg(
    tool: impl Into<String>,
    key: impl Into<String>,
    value: serde_json::Value,
) -> HookEntry {
    let tool = tool.into();
    let key = key.into();
    HookEntry::sync("set-arg", move |event| match event {
        HookEvent::ToolCall { call, .. } if call.function.name == tool => {
            let mut parsed = call.function.arguments;
            if !parsed.is_object() {
                parsed = json!({});
            }
            parsed[&key] = value.clone();
            HookDecision::ToolCall(ToolCallAction::rewrite(parsed))
        }
        HookEvent::ToolCall { .. } => HookDecision::ToolCall(ToolCallAction::run()),
        _ => HookDecision::Continue,
    })
}

/// How a [`rewrite_tool_result`] transforms a named tool's output.
#[derive(Clone)]
pub(crate) enum ResultRewrite {
    /// Replace the whole result with a fixed marker.
    Replace(String),
    /// Wrap the (possibly already-rewritten) result — chains with a prior rewrite.
    Wrap { prefix: String, suffix: String },
    /// Keep only the first `n` characters.
    Truncate(usize),
}

/// Rewrites a named tool's result. Chains with other `rewrite_tool_result`s:
/// each entry sees the running presentation, not the raw output.
pub(crate) fn rewrite_tool_result(tool: impl Into<String>, rewrite: ResultRewrite) -> HookEntry {
    let tool = tool.into();
    HookEntry::sync("rewrite-tool-result", move |event| match event {
        HookEvent::ToolResult {
            call, presentation, ..
        } if call.function.name == tool => {
            let new = match &rewrite {
                ResultRewrite::Replace(marker) => marker.clone(),
                ResultRewrite::Wrap { prefix, suffix } => {
                    format!("{prefix}{}{suffix}", presentation.render())
                }
                ResultRewrite::Truncate(n) => presentation.render().chars().take(*n).collect(),
            };
            HookDecision::ToolResult(ToolResultAction::rewrite(new))
        }
        HookEvent::ToolResult { .. } => HookDecision::ToolResult(ToolResultAction::keep()),
        _ => HookDecision::Continue,
    })
}

/// Terminates the run when a named tool *produces a result* (post-execution).
pub(crate) fn terminate_on_result(tool: impl Into<String>, reason: impl Into<String>) -> HookEntry {
    let tool = tool.into();
    let reason = reason.into();
    HookEntry::sync("terminate-on-result", move |event| match event {
        HookEvent::ToolResult { call, .. } if call.function.name == tool => {
            HookDecision::ToolResult(ToolResultAction::stop(reason.clone()))
        }
        HookEvent::ToolResult { .. } => HookDecision::ToolResult(ToolResultAction::keep()),
        _ => HookDecision::Continue,
    })
}
