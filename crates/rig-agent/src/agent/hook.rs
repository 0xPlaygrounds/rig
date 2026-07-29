//! The decision vocabulary for observing and steering an agent run.
//!
//! Hooks themselves are plain data: an ordered list of named callback records
//! ([`Hooks`](crate::hooks::Hooks) / [`HookEntry`](crate::hooks::HookEntry))
//! that answer owned [`HookEvent`](crate::hooks::HookEvent) values. This
//! module owns the *decisions* those callbacks return — one action type per
//! event, so unsupported combinations are rejected by the compiler instead of
//! interpreted at runtime — plus the composition helpers
//! ([`fold_completion_actions`], [`fold_observation_actions`],
//! [`fold_invalid_resolutions`], [`ToolCallResolution`],
//! [`ToolResultResolution`]) that define how several decisions for one event
//! combine.
//!
//! Decisions are independent of the agent's
//! [`CompletionModel`](crate::completion::CompletionModel): managed response
//! events carry canonical Rig messages, content, usage, and message IDs. Use
//! the direct completion or streaming APIs when an integration needs the
//! provider's typed raw response.
//!
//! Hooks run in registration order. Completion-call [`RequestPatch`] values
//! accumulate and merge; tool-call argument rewrites and tool-result
//! presentation rewrites chain into later entries, and a terminal `Skip`/`Stop`
//! preserves the rewrite accumulated before it. A [`ModelTurnAction::Retry`]
//! or stop action short-circuits the remaining entries for that event.
//!
//! Register observe-only entries before steering entries when every
//! observation is required: a steering stop intentionally prevents later
//! observers from running. Tool-result rewrites change the effective
//! presentation sent to the model and recorded as result-content telemetry;
//! the raw result remains unchanged for policy decisions and execution-outcome
//! metadata. A tool-result stop omits result content from telemetry.
//!
//! Blocking and streaming agents share model-turn, request, tool-call, and
//! tool-result resolution. Streaming adds delta-specific observations (opt in
//! per entry with [`HookEntry::observing_deltas`](crate::hooks::HookEntry::observing_deltas)),
//! but shared lifecycle actions have identical semantics on both surfaces.
//! Streamed deltas are provisional until the model turn is accepted; a retry is
//! surfaced as
//! [`MultiTurnStreamItem::ModelTurnRetried`](crate::agent::MultiTurnStreamItem::ModelTurnRetried)
//! so consumers can discard the rejected turn's deltas.
//!
//! # Example
//!
//! ```
//! use rig_agent::agent::ObservationAction;
//! use rig_agent::hooks::{HookDecision, HookEntry, HookEvent};
//!
//! let logger = HookEntry::new("response-logger", |event| {
//!     if let HookEvent::CompletionResponse { response, .. } = &event {
//!         println!(
//!             "message {:?}: {:?} ({:?})",
//!             response.message_id, response.choice, response.usage
//!         );
//!     }
//!     Box::pin(async { HookDecision::Observation(ObservationAction::continue_run()) })
//! });
//! # let _ = logger;
//! ```
//!
//! # Retrying a completed model turn
//!
//! A hook can reject a tool-free turn and either reuse the same prompt and
//! preceding history with fresh request preparation, or preserve the rejected
//! response and append corrective feedback. Retries use the run's existing
//! total model-call budget. A narrower policy limit belongs to the hook, which
//! owns its own state by capturing it:
//!
//! ```
//! use std::sync::Arc;
//! use std::sync::atomic::{AtomicUsize, Ordering};
//! use rig_agent::agent::ModelTurnAction;
//! use rig_agent::hooks::{HookDecision, HookEntry, HookEvent};
//! use rig_core::message::AssistantContent;
//!
//! fn retry_on_marker(max_retries: usize) -> HookEntry {
//!     let attempts = Arc::new(AtomicUsize::new(0));
//!     HookEntry::new("retry-on-marker", move |event| {
//!         let HookEvent::ModelTurnFinished { content, .. } = &event else {
//!             return Box::pin(async { HookDecision::Continue });
//!         };
//!         let rejected = content.iter().any(|content| {
//!             matches!(content, AssistantContent::Text(text) if text.text.contains("RETRY"))
//!         });
//!         if !rejected {
//!             return Box::pin(async { HookDecision::ModelTurn(ModelTurnAction::continue_run()) });
//!         }
//!         let attempt = attempts.fetch_add(1, Ordering::Relaxed) + 1;
//!         let action = if attempt <= max_retries {
//!             ModelTurnAction::retry_with_feedback("Return a complete answer.")
//!         } else {
//!             ModelTurnAction::stop("response retry limit exceeded")
//!         };
//!         Box::pin(async move { HookDecision::ModelTurn(action) })
//!     })
//! }
//! # let _hook = retry_on_marker(2);
//! ```

use rig_core::json_utils;
use rig_core::message::{Message, ToolChoice};

use crate::{completion::Document, tool::ToolOutput};

/// Opaque process-scoped identifier for one agent run.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RunId(String);

impl RunId {
    /// Mint a fresh run identifier. Hosts that correlate their own hook
    /// state per run can use this as the key.
    pub fn generate() -> Self {
        Self(rig_core::id::generate())
    }

    /// Identifier as text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for RunId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Diagnostics for an invalid model-emitted tool call.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct InvalidToolCallContext {
    /// Name emitted by the model.
    pub tool_name: String,
    /// Provider tool-call id, when present.
    pub tool_call_id: Option<String>,
    /// Rig correlation id, when present.
    pub internal_call_id: Option<String>,
    /// Emitted JSON arguments, when present.
    pub args: Option<String>,
    /// Executable tools advertised for the turn.
    pub available_tools: Vec<String>,
    /// Tools permitted by the active tool choice.
    pub allowed_tools: Vec<String>,
    /// Active tool choice.
    pub tool_choice: Option<ToolChoice>,
    /// Diagnostic history including the rejected output.
    pub chat_history: Vec<Message>,
    /// Whether the call came from the streaming path.
    pub is_streaming: bool,
}

/// How an accepted, tool-free model turn should be retried.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RetryRequest {
    /// Discard the rejected response and reuse the same prompt and preceding
    /// history with fresh request preparation.
    ///
    /// Completion-call hooks, retrieval, and dynamic tool resolution run again,
    /// so the resulting provider request may differ from the rejected attempt.
    Repeat,
    /// Preserve the rejected assistant response and append corrective feedback.
    Feedback(String),
}

/// Action for the medium-neutral [`ModelTurnFinished`] event.
///
/// Every retry consumes the run's existing total model-call budget. Rig does
/// not impose a separate response-retry limit; hooks that need one should keep
/// their own captured state. Retrying a turn containing
/// tool calls is rejected so provider-visible history never contains unanswered
/// calls. Use tool-call hooks to steer those turns instead.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ModelTurnAction {
    /// Accept the turn and continue the run.
    Continue,
    /// Reject the turn and request another model call.
    Retry(RetryRequest),
    /// Stop the run with a reason.
    Stop(String),
}

impl ModelTurnAction {
    /// Accepts the completed model turn.
    pub fn continue_run() -> Self {
        Self::Continue
    }

    /// Discards the response and reuses the same prompt and preceding history
    /// with fresh request preparation.
    pub fn repeat() -> Self {
        Self::Retry(RetryRequest::Repeat)
    }

    /// Preserves the response, appends corrective feedback, and retries.
    pub fn retry_with_feedback(feedback: impl Into<String>) -> Self {
        Self::Retry(RetryRequest::Feedback(feedback.into()))
    }

    /// Stops the run with the supplied reason.
    pub fn stop(reason: impl Into<String>) -> Self {
        Self::Stop(reason.into())
    }
}

/// A non-sticky patch applied only to the current turn's completion request.
///
/// [`Hooks`](crate::hooks::Hooks) merges patches in registration order
/// according to these rules:
///
/// - `extra_context` documents are appended in order.
/// - JSON-object `additional_params` values are shallow-merged, with later
///   top-level keys winning; a later non-object value replaces an earlier value.
/// - `active_tools` allow-lists are intersected.
/// - Scalar fields and `history` use last-writer-wins semantics, with a warning
///   when multiple hooks set the same field.
///
/// The merged patch does not mutate the agent's configured baseline and is not
/// carried into subsequent turns.
///
/// A patch carries **request-shaped** data only: fields that change what one
/// completion request looks like (model, preamble, sampling, tools, schema,
/// context, history). Run *policy* — `max_turns`, tool concurrency,
/// invalid-tool-call retry budgets, memory/conversation wiring, telemetry
/// recording — is configuration scoped to the whole run and deliberately has
/// no patch field.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct RequestPatch {
    /// Provider model identifier to use instead of the configured model for
    /// this turn (honored by providers that accept a per-request model
    /// override).
    #[serde(default)]
    pub model: Option<String>,
    /// Preamble to use instead of the agent's configured preamble for this turn.
    pub preamble: Option<String>,
    /// Sampling temperature to use for this turn.
    pub temperature: Option<f64>,
    /// Maximum output-token count to use for this turn.
    pub max_tokens: Option<u64>,
    /// Tool-choice policy to use for this turn.
    pub tool_choice: Option<ToolChoice>,
    /// Allow-list used to narrow the tools advertised for this turn.
    pub active_tools: Option<Vec<String>>,
    /// Provider-specific request parameters to apply for this turn.
    pub additional_params: Option<serde_json::Value>,
    /// Context documents appended to the request for this turn.
    pub extra_context: Vec<Document>,
    /// Conversation history to use instead of the current history for this turn.
    pub history: Option<Vec<Message>>,
    /// Structured-output JSON Schema to use instead of the configured
    /// `output_schema` for this turn.
    #[serde(default)]
    pub output_schema: Option<rig_core::schemars::Schema>,
}

fn merge_last_wins<T>(earlier: Option<T>, later: Option<T>, field: &str) -> Option<T> {
    match (earlier, later) {
        (Some(_), Some(later)) => {
            tracing::warn!(
                patch_field = field,
                "two hooks set the same request field; later wins"
            );
            Some(later)
        }
        (earlier, later) => later.or(earlier),
    }
}

impl RequestPatch {
    /// Creates an empty request patch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the configured provider model for this turn.
    pub fn model(mut self, value: impl Into<String>) -> Self {
        self.model = Some(value.into());
        self
    }

    /// Replaces the configured structured-output schema for this turn.
    pub fn output_schema(mut self, value: rig_core::schemars::Schema) -> Self {
        self.output_schema = Some(value);
        self
    }

    /// Replaces the agent's configured preamble for this turn.
    pub fn preamble(mut self, value: impl Into<String>) -> Self {
        self.preamble = Some(value.into());
        self
    }

    /// Sets the sampling temperature for this turn.
    pub fn temperature(mut self, value: f64) -> Self {
        self.temperature = Some(value);
        self
    }

    /// Sets the maximum output-token count for this turn.
    pub fn max_tokens(mut self, value: u64) -> Self {
        self.max_tokens = Some(value);
        self
    }

    /// Sets the tool-choice policy for this turn.
    pub fn tool_choice(mut self, value: ToolChoice) -> Self {
        self.tool_choice = Some(value);
        self
    }

    /// Sets the allow-list used to narrow the tools advertised for this turn.
    pub fn active_tools<I, S>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.active_tools = Some(values.into_iter().map(Into::into).collect());
        self
    }

    /// Sets provider-specific request parameters for this turn.
    ///
    /// When multiple patches provide JSON objects, their top-level keys are
    /// shallow-merged and values from later hooks win.
    pub fn additional_params(mut self, value: serde_json::Value) -> Self {
        self.additional_params = Some(value);
        self
    }

    /// Appends context documents to the request for this turn.
    pub fn extra_context<I>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = Document>,
    {
        self.extra_context.extend(values);
        self
    }

    /// Appends one context document to the request for this turn.
    pub fn context(mut self, value: Document) -> Self {
        self.extra_context.push(value);
        self
    }

    /// Replaces the conversation history for this turn.
    pub fn history<I>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = Message>,
    {
        self.history = Some(values.into_iter().collect());
        self
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.model.is_none()
            && self.output_schema.is_none()
            && self.preamble.is_none()
            && self.temperature.is_none()
            && self.max_tokens.is_none()
            && self.tool_choice.is_none()
            && self.active_tools.is_none()
            && self.additional_params.is_none()
            && self.extra_context.is_empty()
            && self.history.is_none()
    }

    /// Merge `later` into `self` with the documented per-field rules —
    /// the combine step of [`fold_completion_actions`] and of any host
    /// composing patches itself.
    pub fn merge(mut self, later: Self) -> Self {
        self.extra_context.extend(later.extra_context);
        self.additional_params = match (self.additional_params.take(), later.additional_params) {
            (Some(base), Some(patch)) if base.is_object() && patch.is_object() => {
                Some(json_utils::merge(base, patch))
            }
            (base, patch) => patch.or(base),
        };
        self.model = merge_last_wins(self.model, later.model, "model");
        self.output_schema =
            merge_last_wins(self.output_schema, later.output_schema, "output_schema");
        self.preamble = merge_last_wins(self.preamble, later.preamble, "preamble");
        self.temperature = merge_last_wins(self.temperature, later.temperature, "temperature");
        self.max_tokens = merge_last_wins(self.max_tokens, later.max_tokens, "max_tokens");
        self.tool_choice = merge_last_wins(self.tool_choice, later.tool_choice, "tool_choice");
        self.history = merge_last_wins(self.history, later.history, "history");
        self.active_tools = match (self.active_tools.take(), later.active_tools) {
            (Some(earlier), Some(later)) => {
                let later: std::collections::BTreeSet<_> = later.iter().collect();
                Some(
                    earlier
                        .into_iter()
                        .filter(|name| later.contains(name))
                        .collect(),
                )
            }
            (earlier, later) => earlier.or(later),
        };
        self
    }
}

/// Action for completion-call hooks.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
// `RequestPatch` grew an output-schema payload; these are transient
// decision values, never stored in bulk, so the size skew is fine.
#[allow(clippy::large_enum_variant)]
pub enum CompletionCallAction {
    /// Send the baseline request.
    Continue,
    /// Merge this per-turn patch into the request.
    Patch(RequestPatch),
    /// Stop the run with a reason.
    Stop(String),
}

impl CompletionCallAction {
    /// Creates an action that sends the request without adding a patch.
    pub fn continue_run() -> Self {
        Self::Continue
    }

    /// Creates an action that applies a per-turn request patch.
    pub fn patch(patch: RequestPatch) -> Self {
        Self::Patch(patch)
    }

    /// Creates an action that stops the run with the supplied reason.
    pub fn stop(reason: impl Into<String>) -> Self {
        Self::Stop(reason.into())
    }
}

/// Action for pre-tool hooks.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ToolCallAction {
    /// Execute with the current arguments.
    Run,
    /// Execute with replacement arguments.
    Rewrite(serde_json::Value),
    /// Do not execute; return this feedback to the model.
    Skip(String),
    /// Stop the run.
    Stop(String),
}

impl ToolCallAction {
    /// Creates an action that executes the tool with the current arguments.
    pub fn run() -> Self {
        Self::Run
    }

    /// Creates an action that replaces the arguments passed to the tool.
    pub fn rewrite(args: impl Into<serde_json::Value>) -> Self {
        Self::Rewrite(args.into())
    }

    /// Serializes replacement arguments and creates a rewrite action.
    ///
    /// Returns an error when `args` cannot be represented as JSON.
    pub fn try_rewrite<T: serde::Serialize>(args: &T) -> Result<Self, serde_json::Error> {
        Ok(Self::Rewrite(serde_json::to_value(args)?))
    }

    /// Creates an action that skips execution and returns feedback to the model.
    pub fn skip(reason: impl Into<String>) -> Self {
        Self::Skip(reason.into())
    }

    /// Creates an action that stops the run before executing the tool.
    pub fn stop(reason: impl Into<String>) -> Self {
        Self::Stop(reason.into())
    }
}

/// Action for post-tool hooks.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ToolResultAction {
    /// Keep the current presentation.
    Keep,
    /// Replace the effective presentation sent to the model and result-content
    /// telemetry.
    Rewrite(ToolOutput),
    /// Stop the run.
    Stop(String),
}

impl ToolResultAction {
    /// Creates an action that preserves the current model-visible presentation.
    pub fn keep() -> Self {
        Self::Keep
    }

    /// Creates an action that replaces the effective presentation sent to the
    /// model and result-content telemetry.
    ///
    /// The tool's raw structured result remains unchanged.
    pub fn rewrite(result: impl Into<String>) -> Self {
        Self::Rewrite(ToolOutput::text(result))
    }

    /// Creates an action that replaces the effective model and telemetry
    /// presentation with explicit structured or multimodal output.
    pub fn rewrite_output(output: ToolOutput) -> Self {
        Self::Rewrite(output)
    }

    /// Creates an action that stops the run after result handling.
    pub fn stop(reason: impl Into<String>) -> Self {
        Self::Stop(reason.into())
    }
}

/// Action for invalid-tool-call hooks and manual invalid-call resolution.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum InvalidToolCallAction {
    /// Preserve fail-fast behavior.
    Fail,
    /// Retry the model with corrective feedback.
    Retry {
        /// Feedback appended for the retry.
        feedback: String,
    },
    /// Repair the emitted tool name.
    Repair {
        /// Replacement registered tool name.
        tool_name: String,
    },
    /// Treat the invalid call as skipped.
    Skip {
        /// Synthetic model feedback.
        reason: String,
    },
    /// Stop the run.
    Stop {
        /// Stop reason.
        reason: String,
    },
}

impl InvalidToolCallAction {
    /// Creates an action that preserves fail-fast invalid-call handling.
    pub fn fail() -> Self {
        Self::Fail
    }

    /// Creates an action that retries the model with corrective feedback.
    pub fn retry(feedback: impl Into<String>) -> Self {
        Self::Retry {
            feedback: feedback.into(),
        }
    }

    /// Creates an action that replaces the invalid tool name.
    pub fn repair(tool_name: impl Into<String>) -> Self {
        Self::Repair {
            tool_name: tool_name.into(),
        }
    }

    /// Creates an action that treats the invalid call as skipped.
    pub fn skip(reason: impl Into<String>) -> Self {
        Self::Skip {
            reason: reason.into(),
        }
    }

    /// Creates an action that stops the run with the supplied reason.
    pub fn stop(reason: impl Into<String>) -> Self {
        Self::Stop {
            reason: reason.into(),
        }
    }
}

/// Action for observe-only lifecycle events.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ObservationAction {
    /// Continue the run.
    Continue,
    /// Stop the run.
    Stop(String),
}

impl ObservationAction {
    /// Creates an action that continues the run.
    pub fn continue_run() -> Self {
        Self::Continue
    }

    /// Creates an action that stops the run with the supplied reason.
    pub fn stop(reason: impl Into<String>) -> Self {
        Self::Stop(reason.into())
    }
}

// ── Data-protocol composition helpers ────────────────────────────────────
//
// Hosts that drive [`AgentRun`](super::run::AgentRun) directly (no hook list)
// express the same composition semantics over plain decision values. Two kinds
// of event exist:
//
// - *Independent* events (completion-call patches, observations,
//   invalid-call resolutions): the input each decider sees is the same, so
//   pre-collected decisions fold faithfully.
// - *Chained* events (tool-call rewrites, tool-result presentation): each
//   later decider's input carries the earlier rewrites, so a fold over
//   pre-collected decisions cannot reproduce the ordered-dispatch behavior. The
//   accumulators below are driven decision-by-decision instead: compute the
//   next decision against the accumulator's current value, then `apply` it.

/// Fold completion-call decisions in registration order: patches accumulate
/// via [`RequestPatch::merge`]'s rules, the first `Stop` wins.
///
/// Faithful to ordered hook dispatch: the event each decider sees
/// is identical, so decisions may be pre-collected. Note the stack also
/// stops *invoking* later hooks after a `Stop`; a host folding
/// pre-collected decisions should likewise stop computing them once one
/// stops, if its decision sources have side effects.
pub fn fold_completion_actions(actions: Vec<CompletionCallAction>) -> CompletionCallAction {
    let mut merged: Option<RequestPatch> = None;
    for action in actions {
        match action {
            CompletionCallAction::Continue => {}
            CompletionCallAction::Patch(patch) => {
                merged = Some(merged.map_or(patch.clone(), |value| value.merge(patch)));
            }
            stop @ CompletionCallAction::Stop(_) => return stop,
        }
    }
    match merged {
        Some(patch) if !patch.is_empty() => CompletionCallAction::Patch(patch),
        _ => CompletionCallAction::Continue,
    }
}

/// Fold observation decisions: the first non-`Continue` wins.
pub fn fold_observation_actions(actions: Vec<ObservationAction>) -> ObservationAction {
    actions
        .into_iter()
        .find(|action| !matches!(action, ObservationAction::Continue))
        .unwrap_or(ObservationAction::Continue)
}

/// Fold invalid-tool-call resolutions: the first `Some` wins (`None` from
/// every source preserves fail-fast behavior).
pub fn fold_invalid_resolutions(
    resolutions: Vec<Option<InvalidToolCallAction>>,
) -> Option<InvalidToolCallAction> {
    resolutions.into_iter().flatten().next()
}

/// Decision-by-decision accumulator for pre-execution tool-call steering,
/// carrying ordered hook dispatch's exact semantics: rewrites
/// chain (each later decision is computed against the current effective
/// arguments), and a terminal `Skip`/`Stop` short-circuits while *keeping*
/// the accumulated rewrite so the driver can report effective arguments.
#[derive(Debug, Clone)]
pub struct ToolCallResolution {
    original: serde_json::Value,
    effective: Option<serde_json::Value>,
    terminal: Option<ToolCallAction>,
}

impl ToolCallResolution {
    /// Start resolving a call with the model-emitted arguments.
    pub fn new(original_args: serde_json::Value) -> Self {
        Self {
            original: original_args,
            effective: None,
            terminal: None,
        }
    }

    /// The arguments the NEXT decision should be computed against
    /// (original, or the latest rewrite).
    pub fn args(&self) -> &serde_json::Value {
        self.effective.as_ref().unwrap_or(&self.original)
    }

    /// Apply one decision. Returns `false` once a terminal `Skip`/`Stop`
    /// has been applied — later sources must not be consulted, matching the
    /// stack's short-circuit.
    pub fn apply(&mut self, action: ToolCallAction) -> bool {
        if self.terminal.is_some() {
            return false;
        }
        match action {
            ToolCallAction::Run => true,
            ToolCallAction::Rewrite(value) => {
                self.effective = Some(value);
                true
            }
            terminal => {
                self.terminal = Some(terminal);
                false
            }
        }
    }

    /// Finish: the effective action plus, for a terminal action, any
    /// rewrite accumulated before it (the stack's "salvage" path).
    pub fn finish(self) -> (ToolCallAction, Option<serde_json::Value>) {
        match self.terminal {
            Some(terminal) => (terminal, self.effective),
            None => match self.effective {
                Some(value) => (ToolCallAction::Rewrite(value), None),
                None => (ToolCallAction::Run, None),
            },
        }
    }
}

/// Decision-by-decision accumulator for post-execution presentation
/// rewrites, carrying ordered hook dispatch's semantics: rewrites
/// chain, `Stop` short-circuits.
#[derive(Debug, Clone, Default)]
pub struct ToolResultResolution {
    effective: Option<ToolOutput>,
    stopped: Option<String>,
}

impl ToolResultResolution {
    /// Start resolving a tool result.
    pub fn new() -> Self {
        Self::default()
    }

    /// The presentation the NEXT decision should be computed against, when
    /// a rewrite has occurred (`None` means the raw presentation).
    pub fn presentation(&self) -> Option<&ToolOutput> {
        self.effective.as_ref()
    }

    /// Apply one decision. Returns `false` once `Stop` has been applied.
    pub fn apply(&mut self, action: ToolResultAction) -> bool {
        if self.stopped.is_some() {
            return false;
        }
        match action {
            ToolResultAction::Keep => true,
            ToolResultAction::Rewrite(output) => {
                self.effective = Some(output);
                true
            }
            ToolResultAction::Stop(reason) => {
                self.stopped = Some(reason);
                false
            }
        }
    }

    /// Finish: `Stop` if any source stopped, else the accumulated rewrite,
    /// else `Keep`.
    pub fn finish(self) -> ToolResultAction {
        match self.stopped {
            Some(reason) => ToolResultAction::Stop(reason),
            None => self
                .effective
                .map_or(ToolResultAction::Keep, ToolResultAction::Rewrite),
        }
    }
}

#[cfg(test)]
mod protocol_helper_tests {
    use super::*;

    #[test]
    fn completion_fold_merges_patches_in_order_and_stops_first() {
        let folded = fold_completion_actions(vec![
            CompletionCallAction::patch(RequestPatch::new().temperature(0.1)),
            CompletionCallAction::Continue,
            CompletionCallAction::patch(RequestPatch::new().temperature(0.2).max_tokens(9)),
        ]);
        assert_eq!(
            folded,
            CompletionCallAction::Patch(RequestPatch::new().temperature(0.2).max_tokens(9))
        );

        let stopped = fold_completion_actions(vec![
            CompletionCallAction::patch(RequestPatch::new().temperature(0.1)),
            CompletionCallAction::stop("halt"),
            CompletionCallAction::patch(RequestPatch::new().temperature(0.9)),
        ]);
        assert_eq!(stopped, CompletionCallAction::stop("halt"));
    }

    #[test]
    fn tool_call_resolution_chains_rewrites_and_salvages_on_terminal() {
        // Mirrors hook.rs `tool_call_rewrites_chain_in_registration_order`.
        let mut resolution = ToolCallResolution::new(serde_json::json!({"step": 0}));
        assert_eq!(resolution.args(), &serde_json::json!({"step": 0}));
        assert!(resolution.apply(ToolCallAction::rewrite(serde_json::json!({"step": 1}))));
        assert_eq!(resolution.args(), &serde_json::json!({"step": 1}));
        assert!(resolution.apply(ToolCallAction::rewrite(serde_json::json!({"step": 2}))));
        let (action, salvage) = resolution.finish();
        assert_eq!(
            action,
            ToolCallAction::rewrite(serde_json::json!({"step": 2}))
        );
        assert!(salvage.is_none());

        // Terminal skip keeps the accumulated rewrite (the salvage path).
        let mut resolution = ToolCallResolution::new(serde_json::json!({"step": 0}));
        assert!(resolution.apply(ToolCallAction::rewrite(serde_json::json!({"step": 1}))));
        assert!(!resolution.apply(ToolCallAction::skip("policy")));
        assert!(!resolution.apply(ToolCallAction::rewrite(serde_json::json!({"step": 9}))));
        let (action, salvage) = resolution.finish();
        assert_eq!(action, ToolCallAction::skip("policy"));
        assert_eq!(salvage, Some(serde_json::json!({"step": 1})));
    }

    #[test]
    fn tool_result_resolution_chains_and_stops() {
        let mut resolution = ToolResultResolution::new();
        assert!(resolution.apply(ToolResultAction::rewrite("redacted")));
        assert_eq!(
            resolution.presentation().map(ToolOutput::render),
            Some("redacted".to_string())
        );
        assert!(!resolution.apply(ToolResultAction::stop("terminal")));
        assert_eq!(resolution.finish(), ToolResultAction::stop("terminal"));
    }

    #[test]
    fn observation_and_invalid_folds_take_first_decisive() {
        assert_eq!(
            fold_observation_actions(vec![
                ObservationAction::Continue,
                ObservationAction::stop("late"),
                ObservationAction::Continue,
            ]),
            ObservationAction::stop("late")
        );
        assert_eq!(
            fold_invalid_resolutions(vec![None, Some(InvalidToolCallAction::fail()), None]),
            Some(InvalidToolCallAction::fail())
        );
        assert_eq!(fold_invalid_resolutions(vec![None, None]), None);
    }

    #[test]
    fn hook_vocabulary_round_trips_through_serde() {
        let patch = RequestPatch::new().temperature(0.5).max_tokens(64);
        let json = serde_json::to_string(&patch).expect("serialize patch");
        let back: RequestPatch = serde_json::from_str(&json).expect("deserialize patch");
        assert_eq!(patch, back);

        let action = InvalidToolCallAction::repair("real_tool");
        let json = serde_json::to_string(&action).expect("serialize action");
        let back: InvalidToolCallAction = serde_json::from_str(&json).expect("deserialize action");
        assert_eq!(action, back);

        let action = ToolResultAction::rewrite_output(ToolOutput::json(
            serde_json::json!({"status": "redacted"}),
        ));
        let json = serde_json::to_string(&action).expect("serialize tool-result action");
        let back: ToolResultAction =
            serde_json::from_str(&json).expect("deserialize tool-result action");
        assert_eq!(action, back);

        for action in [
            ToolResultAction::keep(),
            ToolResultAction::rewrite("plain text"),
            ToolResultAction::stop("halt"),
        ] {
            let json = serde_json::to_string(&action).expect("serialize tool-result action");
            let back: ToolResultAction =
                serde_json::from_str(&json).expect("deserialize tool-result action");
            assert_eq!(action, back);
        }
    }
}

#[cfg(test)]
mod migrated_tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use serde_json::json;

    fn doc(id: &str) -> crate::completion::Document {
        crate::completion::Document {
            id: id.into(),
            text: String::new(),
            additional_props: Default::default(),
        }
    }

    #[test]
    fn merge_appends_extra_context_in_order() {
        let merged = RequestPatch::new()
            .context(doc("a"))
            .merge(RequestPatch::new().context(doc("b")));
        assert_eq!(
            merged
                .extra_context
                .iter()
                .map(|d| d.id.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b"]
        );
    }

    #[test]
    fn merge_shallow_merges_additional_params_later_wins() {
        let merged = RequestPatch::new()
            .additional_params(json!({"x":1,"y":2}))
            .merge(RequestPatch::new().additional_params(json!({"y":3,"z":4})));
        assert_eq!(merged.additional_params, Some(json!({"x":1,"y":3,"z":4})));
    }

    #[test]
    fn merge_scalar_last_writer_wins() {
        assert_eq!(
            RequestPatch::new()
                .temperature(0.1)
                .merge(RequestPatch::new().temperature(0.9))
                .temperature,
            Some(0.9)
        );
    }

    #[test]
    fn merge_model_last_writer_wins() {
        assert_eq!(
            RequestPatch::new()
                .model("gpt-4o")
                .merge(RequestPatch::new().model("gpt-4o-mini"))
                .model,
            Some("gpt-4o-mini".to_string())
        );
        // An unset later field inherits the earlier value.
        assert_eq!(
            RequestPatch::new()
                .model("gpt-4o")
                .merge(RequestPatch::new())
                .model,
            Some("gpt-4o".to_string())
        );
    }

    #[test]
    fn merge_output_schema_last_writer_wins() {
        let earlier = rig_core::schemars::json_schema!({"type": "string"});
        let later = rig_core::schemars::json_schema!({"type": "number"});
        assert_eq!(
            RequestPatch::new()
                .output_schema(earlier.clone())
                .merge(RequestPatch::new().output_schema(later.clone()))
                .output_schema,
            Some(later)
        );
        assert_eq!(
            RequestPatch::new()
                .output_schema(earlier.clone())
                .merge(RequestPatch::new())
                .output_schema,
            Some(earlier)
        );
    }

    #[test]
    fn new_patch_fields_participate_in_is_empty() {
        assert!(RequestPatch::new().is_empty());
        assert!(!RequestPatch::new().model("gpt-4o").is_empty());
        assert!(
            !RequestPatch::new()
                .output_schema(rig_core::schemars::json_schema!({"type": "string"}))
                .is_empty()
        );
    }

    #[test]
    fn merge_active_tools_intersects() {
        let merged = RequestPatch::new()
            .active_tools(["add", "sub"])
            .merge(RequestPatch::new().active_tools(["sub", "mul"]));
        assert_eq!(merged.active_tools, Some(vec!["sub".into()]));
    }

    #[test]
    fn merge_active_tools_empty_intersection_yields_empty() {
        assert_eq!(
            RequestPatch::new()
                .active_tools(["a"])
                .merge(RequestPatch::new().active_tools(["b"]))
                .active_tools,
            Some(vec![])
        );
    }

    #[test]
    fn action_types_are_event_specific() {
        fn completion(_: CompletionCallAction) {}
        fn model_turn(_: ModelTurnAction) {}
        fn retry_request(_: RetryRequest) {}
        fn call(_: ToolCallAction) {}
        fn result(_: ToolResultAction) {}
        fn invalid(_: InvalidToolCallAction) {}
        fn observation(_: ObservationAction) {}
        completion(CompletionCallAction::continue_run());
        model_turn(ModelTurnAction::retry_with_feedback("try again"));
        retry_request(RetryRequest::Repeat);
        call(ToolCallAction::run());
        result(ToolResultAction::keep());
        invalid(InvalidToolCallAction::fail());
        observation(ObservationAction::continue_run());
        let calls = AtomicUsize::new(0);
        calls.fetch_add(1, Ordering::Relaxed);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }
}
