//! Structured tool-execution results.
//!
//! Rig separates three things a tool execution produces, so hooks, tracing,
//! telemetry, and policies never have to parse the model-visible string to
//! reason about what happened:
//!
//! 1. **model-visible output** — the text the LLM receives ([`ToolExecutionResult::model_output`]);
//! 2. **a structured outcome** — success, a classified failure, skipped, or
//!    denied ([`ToolOutcome`]);
//! 3. **type-erased extensions** — provider/application metadata that is *never*
//!    sent to the model ([`ToolResultExtensions`](crate::tool::ToolResultExtensions)).
//!
//! A tool author returns a [`ToolReturn`] to attach an outcome or extensions to a
//! successful output; the [`Tool::classify_error`](crate::tool::Tool::classify_error)
//! hook maps a tool's own error type to a [`ToolFailure`] without string parsing.
//! The dynamic tool boundary ([`ToolDyn`](crate::tool::ToolDyn)) carries the
//! resulting [`ToolExecutionResult`] all the way through to the
//! [`StepEvent::ToolResult`](crate::agent::StepEvent::ToolResult) hook event.

use crate::tool::ToolResultExtensions;
use serde::Serialize;

/// How a tool execution failed, as a closed set of standard kinds.
///
/// A hook, policy, or telemetry pipeline matches on this to make control-flow
/// decisions (e.g. "terminate after repeated timeouts, but keep going on a 404")
/// without parsing the model-visible error string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ToolFailureKind {
    /// The arguments could not be parsed or were rejected as invalid.
    InvalidArgs,
    /// The tool did not complete within its allotted time.
    Timeout,
    /// The tool call was cancelled (e.g. by an abort signal or a dropped future).
    Cancelled,
    /// The requested resource was not found (e.g. an HTTP 404).
    NotFound,
    /// The caller was not permitted to perform the action (e.g. an HTTP 401/403).
    PermissionDenied,
    /// The tool was rate limited (e.g. an HTTP 429).
    RateLimited,
    /// The upstream provider/service returned an error (e.g. an HTTP 5xx).
    Provider,
    /// A network/transport failure occurred before a response was received.
    Network,
    /// Any failure that does not fit a more specific kind.
    Other,
}

impl ToolFailureKind {
    /// A stable, machine-friendly identifier for the kind, suitable for tracing
    /// spans, metrics labels, and structured logs.
    pub const fn as_str(self) -> &'static str {
        match self {
            ToolFailureKind::InvalidArgs => "invalid_args",
            ToolFailureKind::Timeout => "timeout",
            ToolFailureKind::Cancelled => "cancelled",
            ToolFailureKind::NotFound => "not_found",
            ToolFailureKind::PermissionDenied => "permission_denied",
            ToolFailureKind::RateLimited => "rate_limited",
            ToolFailureKind::Provider => "provider",
            ToolFailureKind::Network => "network",
            ToolFailureKind::Other => "other",
        }
    }

    /// A sensible default for whether a failure of this kind is worth retrying,
    /// used by the per-kind constructors on [`ToolFailure`]. Transient kinds
    /// (timeout, rate-limited, network) default to retryable; kinds that will
    /// fail again identically (invalid args, not found, permission denied,
    /// cancelled) default to not retryable; ambiguous kinds default to unknown.
    const fn default_retryable(self) -> Option<bool> {
        match self {
            ToolFailureKind::Timeout | ToolFailureKind::RateLimited | ToolFailureKind::Network => {
                Some(true)
            }
            ToolFailureKind::InvalidArgs
            | ToolFailureKind::NotFound
            | ToolFailureKind::PermissionDenied
            | ToolFailureKind::Cancelled => Some(false),
            ToolFailureKind::Provider | ToolFailureKind::Other => None,
        }
    }
}

impl std::fmt::Display for ToolFailureKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A structured, model-independent description of a tool execution failure.
///
/// This is the "machine-visible" half of a failed tool call. The model-visible
/// text lives separately on [`ToolExecutionResult::model_output`]; this carries
/// the classification a hook or policy acts on. Build one with the per-kind
/// constructors (which pre-fill a sensible [`retryable`](Self::retryable) default)
/// and refine it with the builder setters:
///
/// ```
/// use rig_core::tool::{ToolFailure, ToolFailureKind};
///
/// let failure = ToolFailure::not_found("user 42 does not exist")
///     .with_http_status(404)
///     .with_code("USER_NOT_FOUND");
/// assert_eq!(failure.kind, ToolFailureKind::NotFound);
/// assert_eq!(failure.retryable, Some(false));
/// assert_eq!(failure.http_status, Some(404));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ToolFailure {
    /// The standard classification of the failure.
    pub kind: ToolFailureKind,
    /// A human-readable description of what went wrong. This is *not*
    /// automatically shown to the model — it is metadata for logs and policies.
    pub message: String,
    /// Whether retrying the call could plausibly succeed. `None` means unknown.
    pub retryable: Option<bool>,
    /// An optional machine-readable error code (e.g. a provider error code).
    pub code: Option<String>,
    /// An optional HTTP status code, when the failure originated from an HTTP call.
    pub http_status: Option<u16>,
}

impl ToolFailure {
    /// Construct a failure of the given `kind` with `message`. `retryable`,
    /// `code`, and `http_status` start unset — use the builder setters or a
    /// per-kind constructor to fill them.
    pub fn new(kind: ToolFailureKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            retryable: None,
            code: None,
            http_status: None,
        }
    }

    /// Construct a failure of `kind` with `message` and the kind's default
    /// [`retryable`](Self::retryable) hint. The per-kind constructors delegate here.
    pub(crate) fn of_kind(kind: ToolFailureKind, message: impl Into<String>) -> Self {
        Self {
            retryable: kind.default_retryable(),
            ..Self::new(kind, message)
        }
    }

    /// An [`InvalidArgs`](ToolFailureKind::InvalidArgs) failure.
    pub fn invalid_args(message: impl Into<String>) -> Self {
        Self::of_kind(ToolFailureKind::InvalidArgs, message)
    }

    /// A [`Timeout`](ToolFailureKind::Timeout) failure.
    pub fn timeout(message: impl Into<String>) -> Self {
        Self::of_kind(ToolFailureKind::Timeout, message)
    }

    /// A [`Cancelled`](ToolFailureKind::Cancelled) failure.
    pub fn cancelled(message: impl Into<String>) -> Self {
        Self::of_kind(ToolFailureKind::Cancelled, message)
    }

    /// A [`NotFound`](ToolFailureKind::NotFound) failure.
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::of_kind(ToolFailureKind::NotFound, message)
    }

    /// A [`PermissionDenied`](ToolFailureKind::PermissionDenied) failure.
    pub fn permission_denied(message: impl Into<String>) -> Self {
        Self::of_kind(ToolFailureKind::PermissionDenied, message)
    }

    /// A [`RateLimited`](ToolFailureKind::RateLimited) failure.
    pub fn rate_limited(message: impl Into<String>) -> Self {
        Self::of_kind(ToolFailureKind::RateLimited, message)
    }

    /// A [`Provider`](ToolFailureKind::Provider) failure.
    pub fn provider(message: impl Into<String>) -> Self {
        Self::of_kind(ToolFailureKind::Provider, message)
    }

    /// A [`Network`](ToolFailureKind::Network) failure.
    pub fn network(message: impl Into<String>) -> Self {
        Self::of_kind(ToolFailureKind::Network, message)
    }

    /// An [`Other`](ToolFailureKind::Other) failure — the catch-all.
    pub fn other(message: impl Into<String>) -> Self {
        Self::of_kind(ToolFailureKind::Other, message)
    }

    /// Set whether the failure is retryable.
    pub fn with_retryable(mut self, retryable: bool) -> Self {
        self.retryable = Some(retryable);
        self
    }

    /// Set a machine-readable error code.
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    /// Set the originating HTTP status code.
    pub fn with_http_status(mut self, status: u16) -> Self {
        self.http_status = Some(status);
        self
    }
}

impl std::fmt::Display for ToolFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)
    }
}

impl std::error::Error for ToolFailure {}

/// The structured outcome of a tool call: what happened, independent of the
/// model-visible text.
///
/// This is Rig's answer to "was that a success or an error?" without inspecting
/// the result string. It mirrors Pydantic AI's `outcome: 'success' | 'failed' |
/// 'denied'` and the Vercel AI SDK's `tool-result` vs `tool-error` distinction.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToolOutcome {
    /// The tool ran and produced output normally.
    Success,
    /// The tool failed. Carries the structured [`ToolFailure`]; the model still
    /// receives [`ToolExecutionResult::model_output`] as feedback.
    Error(ToolFailure),
    /// A [`ToolCall`](crate::agent::StepEvent::ToolCall) hook returned
    /// [`Flow::Skip`](crate::agent::Flow::Skip): the tool body did not run.
    Skipped,
    /// The call was denied (e.g. by an approval policy). The tool body did not run.
    Denied,
}

impl ToolOutcome {
    /// A stable, machine-friendly identifier for the outcome, for tracing/metrics.
    pub const fn as_str(&self) -> &'static str {
        match self {
            ToolOutcome::Success => "success",
            ToolOutcome::Error(_) => "error",
            ToolOutcome::Skipped => "skipped",
            ToolOutcome::Denied => "denied",
        }
    }

    /// Whether the tool ran successfully.
    pub const fn is_success(&self) -> bool {
        matches!(self, ToolOutcome::Success)
    }

    /// Whether the tool failed.
    pub const fn is_error(&self) -> bool {
        matches!(self, ToolOutcome::Error(_))
    }

    /// Whether the call was skipped by a hook before execution.
    pub const fn is_skipped(&self) -> bool {
        matches!(self, ToolOutcome::Skipped)
    }

    /// Whether the call was denied before execution.
    pub const fn is_denied(&self) -> bool {
        matches!(self, ToolOutcome::Denied)
    }

    /// The [`ToolFailure`] if this is an [`Error`](ToolOutcome::Error), else `None`.
    pub const fn failure(&self) -> Option<&ToolFailure> {
        match self {
            ToolOutcome::Error(failure) => Some(failure),
            _ => None,
        }
    }

    /// The [`ToolFailureKind`] if this is an [`Error`](ToolOutcome::Error), else `None`.
    pub const fn error_kind(&self) -> Option<ToolFailureKind> {
        match self {
            ToolOutcome::Error(failure) => Some(failure.kind),
            _ => None,
        }
    }

    /// Whether this is an [`Error`](ToolOutcome::Error) of exactly `kind`.
    ///
    /// The predicate a hook uses to react to a specific failure class:
    ///
    /// ```
    /// use rig_core::tool::{ToolFailure, ToolFailureKind, ToolOutcome};
    ///
    /// let outcome = ToolOutcome::Error(ToolFailure::timeout("slow upstream"));
    /// assert!(outcome.is_error_kind(ToolFailureKind::Timeout));
    /// assert!(!outcome.is_error_kind(ToolFailureKind::NotFound));
    /// ```
    pub fn is_error_kind(&self, kind: ToolFailureKind) -> bool {
        self.error_kind() == Some(kind)
    }
}

/// The full structured result of a single tool execution.
///
/// This is what the dynamic tool boundary ([`ToolDyn`](crate::tool::ToolDyn))
/// produces and what flows through to the
/// [`StepEvent::ToolResult`](crate::agent::StepEvent::ToolResult) hook event. It
/// keeps the three concerns separate:
///
/// - [`model_output`](Self::model_output): the text delivered to the model;
/// - [`outcome`](Self::outcome): the structured [`ToolOutcome`];
/// - [`extensions`](Self::extensions): metadata never sent to the model.
///
/// Tool authors rarely build this directly — they return a [`ToolReturn`] and the
/// boundary assembles it. Construct one explicitly only in a manual
/// [`ToolDyn`](crate::tool::ToolDyn) implementation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ToolExecutionResult {
    /// The text delivered to the model as the tool result. Present even for a
    /// failure, so the model gets useful feedback (a handled error message).
    pub model_output: String,
    /// The structured outcome of the call.
    pub outcome: ToolOutcome,
    /// Metadata attached by the tool, surfaced to hooks/tracing but never sent
    /// to the model.
    pub extensions: ToolResultExtensions,
}

impl ToolExecutionResult {
    /// Construct a result with the given model output and outcome, and no
    /// extensions.
    pub fn new(model_output: impl Into<String>, outcome: ToolOutcome) -> Self {
        Self {
            model_output: model_output.into(),
            outcome,
            extensions: ToolResultExtensions::new(),
        }
    }

    /// A successful result whose model output is `model_output` verbatim.
    pub fn success(model_output: impl Into<String>) -> Self {
        Self::new(model_output, ToolOutcome::Success)
    }

    /// A failed result: `model_output` is the model-visible feedback, `failure`
    /// the structured classification.
    pub fn failed(model_output: impl Into<String>, failure: ToolFailure) -> Self {
        Self::new(model_output, ToolOutcome::Error(failure))
    }

    /// A [`Skipped`](ToolOutcome::Skipped) result (the body did not run).
    pub fn skipped(model_output: impl Into<String>) -> Self {
        Self::new(model_output, ToolOutcome::Skipped)
    }

    /// A [`Denied`](ToolOutcome::Denied) result (the body did not run).
    pub fn denied(model_output: impl Into<String>) -> Self {
        Self::new(model_output, ToolOutcome::Denied)
    }

    /// Attach result extensions, replacing any already set.
    pub fn with_extensions(mut self, extensions: ToolResultExtensions) -> Self {
        self.extensions = extensions;
        self
    }
}

/// A tool's return value carrying an optional structured outcome and metadata.
///
/// The ergonomic path for a tool author who wants more than a plain success. A
/// tool's [`call`](crate::tool::Tool::call) still returns a bare
/// `T: Serialize` for the common case; override
/// [`Tool::call_structured`](crate::tool::Tool::call_structured) to return a
/// `ToolReturn<T>` instead when you need to:
///
/// - attach [`extensions`](Self::extensions) (provider/application metadata) to a
///   success;
/// - report a *handled failure* that still shows structured output to the model
///   ([`failed`](Self::failed));
/// - mark the call [`denied`](Self::denied) or [`skipped`](Self::skipped) from
///   inside the tool.
///
/// The [`output`](Self::output) is serialized to the model exactly as a normal
/// tool output would be (a `String` output stays verbatim; anything else becomes
/// JSON), so switching a tool from `T` to `ToolReturn<T>` never changes what the
/// model sees for the success case.
///
/// # Example
/// ```
/// use rig_core::tool::{ToolFailure, ToolReturn};
///
/// #[derive(Clone, Debug, PartialEq)]
/// struct RequestId(String);
///
/// // A success that also records the upstream request id for telemetry.
/// let ok: ToolReturn<String> =
///     ToolReturn::success("42 results".to_string()).with_extension(RequestId("req-9".into()));
///
/// // A handled failure that still gives the model a structured message.
/// let err: ToolReturn<String> =
///     ToolReturn::failed("no such city".to_string(), ToolFailure::not_found("city=atlantis"));
/// ```
#[derive(Debug, Clone)]
pub struct ToolReturn<T> {
    /// The value serialized as the model-visible tool output.
    pub output: T,
    /// The structured outcome. Defaults to [`ToolOutcome::Success`].
    pub outcome: ToolOutcome,
    /// Metadata surfaced to hooks/tracing but never sent to the model.
    pub extensions: ToolResultExtensions,
}

impl<T> ToolReturn<T> {
    /// A plain successful return wrapping `output`, with no extra metadata. This
    /// is what the default [`Tool::call_structured`](crate::tool::Tool::call_structured)
    /// produces from a bare [`Tool::call`](crate::tool::Tool::call) output.
    pub fn success(output: T) -> Self {
        Self {
            output,
            outcome: ToolOutcome::Success,
            extensions: ToolResultExtensions::new(),
        }
    }

    /// A return wrapping `output` with an explicit `outcome`.
    pub fn new(output: T, outcome: ToolOutcome) -> Self {
        Self {
            output,
            outcome,
            extensions: ToolResultExtensions::new(),
        }
    }

    /// A handled-failure return: `output` is still serialized to the model as
    /// feedback, but the outcome is [`ToolOutcome::Error`] carrying `failure`.
    pub fn failed(output: T, failure: ToolFailure) -> Self {
        Self::new(output, ToolOutcome::Error(failure))
    }

    /// A [`denied`](ToolOutcome::Denied) return (the model still sees `output`).
    pub fn denied(output: T) -> Self {
        Self::new(output, ToolOutcome::Denied)
    }

    /// A [`skipped`](ToolOutcome::Skipped) return (the model still sees `output`).
    pub fn skipped(output: T) -> Self {
        Self::new(output, ToolOutcome::Skipped)
    }

    /// Replace the outcome.
    pub fn with_outcome(mut self, outcome: ToolOutcome) -> Self {
        self.outcome = outcome;
        self
    }

    /// Replace the extensions wholesale.
    pub fn with_extensions(mut self, extensions: ToolResultExtensions) -> Self {
        self.extensions = extensions;
        self
    }

    /// Insert a single value into the extensions, returning the updated return.
    pub fn with_extension<
        E: Clone + crate::wasm_compat::WasmCompatSend + crate::wasm_compat::WasmCompatSync + 'static,
    >(
        mut self,
        extension: E,
    ) -> Self {
        self.extensions.insert(extension);
        self
    }
}

impl<T: Serialize> ToolReturn<T> {
    /// Serialize `output` to the model-visible string and assemble a
    /// [`ToolExecutionResult`], preserving the outcome and extensions.
    ///
    /// A `String` output is delivered verbatim; anything else is JSON-encoded —
    /// the same shaping a bare tool output receives. If serialization fails the
    /// result is an [`Other`](ToolFailureKind::Other) failure whose model output
    /// explains the serialization error.
    pub(crate) fn into_execution_result(self) -> ToolExecutionResult {
        match super::serialize_tool_output(&self.output) {
            Ok(model_output) => ToolExecutionResult {
                model_output,
                outcome: self.outcome,
                extensions: self.extensions,
            },
            Err(err) => ToolExecutionResult::failed(
                format!("failed to serialize tool output: {err}"),
                ToolFailure::other(err.to_string()),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_kind_constructors_prefill_retryable() {
        assert_eq!(ToolFailure::timeout("t").retryable, Some(true));
        assert_eq!(ToolFailure::rate_limited("r").retryable, Some(true));
        assert_eq!(ToolFailure::network("n").retryable, Some(true));
        assert_eq!(ToolFailure::not_found("nf").retryable, Some(false));
        assert_eq!(ToolFailure::permission_denied("p").retryable, Some(false));
        assert_eq!(ToolFailure::invalid_args("i").retryable, Some(false));
        assert_eq!(ToolFailure::cancelled("c").retryable, Some(false));
        assert_eq!(ToolFailure::provider("p").retryable, None);
        assert_eq!(ToolFailure::other("o").retryable, None);
        // `new` leaves it unset regardless of kind.
        assert_eq!(
            ToolFailure::new(ToolFailureKind::Timeout, "t").retryable,
            None
        );
    }

    #[test]
    fn failure_builders_compose() {
        let failure = ToolFailure::rate_limited("slow down")
            .with_http_status(429)
            .with_code("RATE_LIMIT")
            .with_retryable(false);
        assert_eq!(failure.kind, ToolFailureKind::RateLimited);
        assert_eq!(failure.http_status, Some(429));
        assert_eq!(failure.code.as_deref(), Some("RATE_LIMIT"));
        assert_eq!(failure.retryable, Some(false));
        assert_eq!(failure.to_string(), "rate_limited: slow down");
    }

    #[test]
    fn outcome_predicates() {
        let ok = ToolOutcome::Success;
        assert!(ok.is_success());
        assert!(!ok.is_error());
        assert_eq!(ok.error_kind(), None);
        assert_eq!(ok.as_str(), "success");

        let err = ToolOutcome::Error(ToolFailure::not_found("x"));
        assert!(err.is_error());
        assert!(err.is_error_kind(ToolFailureKind::NotFound));
        assert!(!err.is_error_kind(ToolFailureKind::Timeout));
        assert_eq!(err.error_kind(), Some(ToolFailureKind::NotFound));
        assert_eq!(
            err.failure().map(|f| f.kind),
            Some(ToolFailureKind::NotFound)
        );
        assert_eq!(err.as_str(), "error");

        assert!(ToolOutcome::Skipped.is_skipped());
        assert!(ToolOutcome::Denied.is_denied());
    }

    #[test]
    fn tool_return_success_serializes_verbatim_string() {
        let result = ToolReturn::success("hello\nworld".to_string()).into_execution_result();
        assert_eq!(result.model_output, "hello\nworld");
        assert!(result.outcome.is_success());
    }

    #[test]
    fn tool_return_object_serializes_as_json_and_keeps_outcome() {
        let result = ToolReturn::failed(
            serde_json::json!({ "status": "missing", "id": 7 }),
            ToolFailure::not_found("id 7"),
        )
        .into_execution_result();
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&result.model_output).unwrap(),
            serde_json::json!({ "status": "missing", "id": 7 })
        );
        assert!(result.outcome.is_error_kind(ToolFailureKind::NotFound));
    }

    #[test]
    fn tool_return_extensions_flow_into_execution_result() {
        #[derive(Clone, Debug, PartialEq)]
        struct ReqId(String);

        let result = ToolReturn::success(1u32)
            .with_extension(ReqId("abc".into()))
            .into_execution_result();
        assert_eq!(result.extensions.get::<ReqId>(), Some(&ReqId("abc".into())));
    }
}
