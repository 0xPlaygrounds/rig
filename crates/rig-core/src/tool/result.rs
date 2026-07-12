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

use crate::{OneOrMany, message::ToolResultContent, tool::ToolResultExtensions};
use serde::Serialize;

/// Model-facing tool output without magic string envelopes.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ToolOutput {
    /// Compatibility path for existing string-returning tools. The runner keeps
    /// supporting Rig's historical multimodal JSON envelope parser.
    LegacyText(String),
    /// Explicit rich content that is never guessed by parsing reserved JSON keys.
    Content(OneOrMany<ToolResultContent>),
}

impl ToolOutput {
    fn text_projection(&self) -> String {
        match self {
            Self::LegacyText(text) => text.clone(),
            Self::Content(content) => content
                .iter()
                .map(|part| match part {
                    ToolResultContent::Text(text) => text.text.clone(),
                    ToolResultContent::Image(_) => "[image]".to_owned(),
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

/// Structured classification and host-only metadata for a tool error.
#[derive(Debug, Clone)]
pub struct ToolErrorReport {
    /// Machine-readable failure classification.
    pub failure: ToolFailure,
    /// Metadata surfaced to hooks but never sent to the model.
    pub extensions: ToolResultExtensions,
}

impl ToolErrorReport {
    /// Create a report without extensions.
    pub fn new(failure: ToolFailure) -> Self {
        Self {
            failure,
            extensions: ToolResultExtensions::new(),
        }
    }

    /// Attach error metadata.
    pub fn with_extensions(mut self, extensions: ToolResultExtensions) -> Self {
        self.extensions = extensions;
        self
    }
}

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
///
/// # `Skipped` vs `Denied`
///
/// Both mean the tool body did not run, but they come from opposite sides — the
/// framework vs. the tool — and are not synonyms:
///
/// - [`Skipped`](Self::Skipped) is produced **by the framework** when a
///   [`ToolCall`](crate::agent::StepEvent::ToolCall) hook returns
///   [`Flow::Skip`](crate::agent::Flow::Skip). **Approval-policy denials use
///   `Flow::Skip`, so they surface as `Skipped`, not `Denied`** — there is no
///   `Flow::Deny`. A policy that wants to distinguish its denials from other
///   skips can key off the skip reason it supplied, or attach its own metadata.
/// - [`Denied`](Self::Denied) is authored **by the tool**, via
///   [`ToolReturn::denied`] / [`ToolExecutionResult::denied`] — the tool ran its
///   own check and refused the call (e.g. an internal authorization check). This
///   is the tool-side counterpart to a hook skip: **tools express refusal as
///   `Denied`, not `Skipped`** (there is no tool-authored skip constructor).
///
/// So [`is_skipped`](Self::is_skipped) means "a hook skipped the call" and
/// [`is_denied`](Self::is_denied) means "the tool refused it" — unambiguously,
/// because the split is enforced by the type system: a tool authors a
/// [`ToolReturnOutcome`], which has no `Skipped` variant, and the observed
/// `Skipped` outcome can be produced only inside the crate (the framework). A
/// tool cannot construct a return or a [`ToolExecutionResult`] that claims to
/// have been skipped.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToolOutcome {
    /// The tool ran and produced output normally.
    Success,
    /// The tool failed. Carries the structured [`ToolFailure`]; the model still
    /// receives [`ToolExecutionResult::model_output`] as feedback.
    Error(ToolFailure),
    /// The tool body did not run because a
    /// [`ToolCall`](crate::agent::StepEvent::ToolCall) hook returned
    /// [`Flow::Skip`](crate::agent::Flow::Skip). This is a **framework** outcome
    /// and includes approval-policy denials, which are expressed as `Flow::Skip`
    /// (see the [type-level note](ToolOutcome#skipped-vs-denied)).
    Skipped,
    /// A **tool** declared the call denied by returning [`ToolReturn::denied`] /
    /// [`ToolExecutionResult::denied`] — it ran its own check and refused. This is
    /// **not** produced by a hook `Flow::Skip` (those are
    /// [`Skipped`](Self::Skipped)); it is the tool-side counterpart to a skip (see
    /// the [type-level note](ToolOutcome#skipped-vs-denied)).
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

    /// Whether a hook skipped the call (`Flow::Skip`) before execution — a
    /// **framework** outcome that includes approval-policy denials (see the
    /// [type-level note](ToolOutcome#skipped-vs-denied)).
    pub const fn is_skipped(&self) -> bool {
        matches!(self, ToolOutcome::Skipped)
    }

    /// Whether a **tool** refused the call (via [`ToolReturn::denied`]). Hook /
    /// approval-policy `Flow::Skip` denials are [`Skipped`](Self::Skipped), not
    /// `Denied` (see the [type-level note](ToolOutcome#skipped-vs-denied)).
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

/// The outcome a *tool* declares for a call it executed.
///
/// A strict subset of [`ToolOutcome`]: a tool that ran can report
/// [`Success`](Self::Success), a handled [`Error`](Self::Error), or a
/// [`Denied`](Self::Denied) refusal — but it **cannot** be
/// [`Skipped`](ToolOutcome::Skipped). A skip is a *framework* decision made
/// before the tool runs (a [`ToolCall`](crate::agent::StepEvent::ToolCall) hook
/// returning [`Flow::Skip`](crate::agent::Flow::Skip)), so it has no
/// tool-authored representation. Because [`ToolReturn`] carries this type — not
/// [`ToolOutcome`] — it is *impossible* to construct a tool return (or a
/// tool-built [`ToolExecutionResult`]) that claims to have been skipped while
/// having actually run. Converts into the observed [`ToolOutcome`] via [`From`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToolReturnOutcome {
    /// The tool ran and produced output normally. Maps to [`ToolOutcome::Success`].
    Success,
    /// The tool ran and failed; carries the structured [`ToolFailure`]. The model
    /// still receives the return's output as feedback. Maps to [`ToolOutcome::Error`].
    Error(ToolFailure),
    /// The tool ran its own check and refused the call (e.g. an internal
    /// authorization check). Maps to [`ToolOutcome::Denied`].
    Denied,
}

impl ToolReturnOutcome {
    /// A stable, machine-friendly identifier, matching the [`ToolOutcome`] this
    /// maps to (`"success"` / `"error"` / `"denied"`).
    pub const fn as_str(&self) -> &'static str {
        match self {
            ToolReturnOutcome::Success => "success",
            ToolReturnOutcome::Error(_) => "error",
            ToolReturnOutcome::Denied => "denied",
        }
    }

    /// The [`ToolFailure`] if this is an [`Error`](Self::Error), else `None`.
    pub const fn failure(&self) -> Option<&ToolFailure> {
        match self {
            ToolReturnOutcome::Error(failure) => Some(failure),
            _ => None,
        }
    }
}

impl From<ToolReturnOutcome> for ToolOutcome {
    fn from(outcome: ToolReturnOutcome) -> Self {
        match outcome {
            ToolReturnOutcome::Success => ToolOutcome::Success,
            ToolReturnOutcome::Error(failure) => ToolOutcome::Error(failure),
            ToolReturnOutcome::Denied => ToolOutcome::Denied,
        }
    }
}

/// The full structured result of a single tool execution.
///
/// This is what the dynamic tool boundary ([`ToolDyn`](crate::tool::ToolDyn))
/// produces and what flows through to the
/// [`StepEvent::ToolResult`](crate::agent::StepEvent::ToolResult) hook event. It
/// keeps the three concerns separate:
///
/// - [`model_output`](Self::model_output()): the text delivered to the model;
/// - [`outcome`](Self::outcome()): the structured [`ToolOutcome`];
/// - [`extensions`](Self::extensions()): metadata never sent to the model.
///
/// Tool authors rarely build this directly — they return a [`ToolReturn`] and the
/// boundary assembles it. In a manual [`ToolDyn`](crate::tool::ToolDyn)
/// implementation construct one with [`success`](Self::success) /
/// [`failed`](Self::failed) / [`denied`](Self::denied); read it back with the
/// [`model_output`](Self::model_output()) / [`outcome`](Self::outcome()) /
/// [`extensions`](Self::extensions()) accessors.
///
/// The fields are crate-private on purpose: [`Skipped`](ToolOutcome::Skipped) is a
/// framework-only outcome (a hook [`Flow::Skip`](crate::agent::Flow::Skip)), and
/// there is no public constructor or setter that yields it — a tool that ran was
/// not skipped. See the [`ToolOutcome` note](ToolOutcome#skipped-vs-denied).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ToolExecutionResult {
    pub(crate) model_output: String,
    pub(crate) output: ToolOutput,
    pub(crate) outcome: ToolOutcome,
    pub(crate) extensions: ToolResultExtensions,
}

impl ToolExecutionResult {
    /// Construct a result with the given model output and outcome, and no
    /// extensions.
    ///
    /// Crate-internal because `outcome` is the full [`ToolOutcome`], including the
    /// framework-only [`Skipped`](ToolOutcome::Skipped). Tool authors use
    /// [`success`](Self::success) / [`failed`](Self::failed) /
    /// [`denied`](Self::denied), which cannot produce `Skipped`.
    pub(crate) fn new(model_output: impl Into<String>, outcome: ToolOutcome) -> Self {
        let model_output = model_output.into();
        Self {
            output: ToolOutput::LegacyText(model_output.clone()),
            model_output,
            outcome,
            extensions: ToolResultExtensions::new(),
        }
    }

    /// A successful result whose model output is `model_output` verbatim.
    pub fn success(model_output: impl Into<String>) -> Self {
        Self::new(model_output, ToolOutcome::Success)
    }

    /// A successful explicit rich result, bypassing legacy JSON-envelope parsing.
    pub fn success_content(content: OneOrMany<ToolResultContent>) -> Self {
        let output = ToolOutput::Content(content);
        Self {
            model_output: output.text_projection(),
            output,
            outcome: ToolOutcome::Success,
            extensions: ToolResultExtensions::new(),
        }
    }

    /// A failed result: `model_output` is the model-visible feedback, `failure`
    /// the structured classification.
    pub fn failed(model_output: impl Into<String>, failure: ToolFailure) -> Self {
        Self::new(model_output, ToolOutcome::Error(failure))
    }

    /// A [`Skipped`](ToolOutcome::Skipped) result (the body did not run).
    ///
    /// Framework-internal: `Skipped` is produced only when a `ToolCall` hook
    /// returns [`Flow::Skip`](crate::agent::Flow::Skip). A tool author expresses
    /// refusal with [`denied`](Self::denied) instead — see the
    /// [`ToolOutcome` note](ToolOutcome#skipped-vs-denied).
    pub(crate) fn skipped(model_output: impl Into<String>) -> Self {
        Self::new(model_output, ToolOutcome::Skipped)
    }

    /// A [`Denied`](ToolOutcome::Denied) result: the tool refused the call (the
    /// body did not run to completion). The tool-authored counterpart to a hook
    /// [`Flow::Skip`](crate::agent::Flow::Skip); see the
    /// [`ToolOutcome` note](ToolOutcome#skipped-vs-denied).
    pub fn denied(model_output: impl Into<String>) -> Self {
        Self::new(model_output, ToolOutcome::Denied)
    }

    /// Attach result extensions, replacing any already set.
    pub fn with_extensions(mut self, extensions: ToolResultExtensions) -> Self {
        self.extensions = extensions;
        self
    }

    /// Insert a single value into the result extensions, returning the updated
    /// result. The single-value counterpart to [`with_extensions`](Self::with_extensions),
    /// mirroring [`ToolReturn::with_extension`] for manual
    /// [`ToolDyn`](crate::tool::ToolDyn) implementations.
    pub fn with_extension<
        E: Clone + crate::wasm_compat::WasmCompatSend + crate::wasm_compat::WasmCompatSync + 'static,
    >(
        mut self,
        extension: E,
    ) -> Self {
        self.extensions.insert(extension);
        self
    }

    /// The text delivered to the model as the tool result. Present even for a
    /// failure, so the model gets useful feedback (a handled error message).
    pub fn model_output(&self) -> &str {
        &self.model_output
    }

    /// Structured model-facing output.
    pub fn output(&self) -> &ToolOutput {
        &self.output
    }

    /// The structured [`ToolOutcome`] of the call.
    pub fn outcome(&self) -> &ToolOutcome {
        &self.outcome
    }

    /// Metadata attached by the tool, surfaced to hooks/tracing but never sent
    /// to the model.
    pub fn extensions(&self) -> &ToolResultExtensions {
        &self.extensions
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
/// - mark the call [`denied`](Self::denied) — the tool ran its own check and
///   refused (the tool-side counterpart to a hook `Flow::Skip`; there is no
///   tool-authored *skipped* — that outcome is the framework's, see
///   [`ToolOutcome`]).
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
    /// The structured outcome the tool declares. A [`ToolReturnOutcome`] (not a
    /// [`ToolOutcome`]), so it can never be the framework-only
    /// [`Skipped`](ToolOutcome::Skipped). Defaults to
    /// [`ToolReturnOutcome::Success`].
    pub outcome: ToolReturnOutcome,
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
            outcome: ToolReturnOutcome::Success,
            extensions: ToolResultExtensions::new(),
        }
    }

    /// A return wrapping `output` with an explicit [`ToolReturnOutcome`].
    pub fn new(output: T, outcome: ToolReturnOutcome) -> Self {
        Self {
            output,
            outcome,
            extensions: ToolResultExtensions::new(),
        }
    }

    /// A handled-failure return: `output` is still serialized to the model as
    /// feedback, but the outcome is [`ToolReturnOutcome::Error`] carrying `failure`.
    pub fn failed(output: T, failure: ToolFailure) -> Self {
        Self::new(output, ToolReturnOutcome::Error(failure))
    }

    /// A [`denied`](ToolReturnOutcome::Denied) return: the tool ran its own check
    /// and refused (the model still sees `output`). The tool-side counterpart to
    /// a hook [`Flow::Skip`](crate::agent::Flow::Skip); there is no tool-authored
    /// *skipped* — `Skipped` is a framework outcome (see
    /// [`ToolReturnOutcome`](ToolReturnOutcome)).
    pub fn denied(output: T) -> Self {
        Self::new(output, ToolReturnOutcome::Denied)
    }

    /// Replace the outcome.
    pub fn with_outcome(mut self, outcome: ToolReturnOutcome) -> Self {
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
    /// [`ToolExecutionResult`], preserving the declared outcome and extensions.
    ///
    /// A `String` output is delivered verbatim; anything else is JSON-encoded —
    /// the same shaping a bare tool output receives.
    ///
    /// If serialization fails, the tool's [`extensions`](Self::extensions) and its
    /// declared *classification* are still preserved — a serialization failure is
    /// a rendering problem, independent of whether the tool succeeded, failed, or
    /// [`denied`](ToolReturnOutcome::Denied) the call — and only the `model_output`
    /// falls back to a string explaining the error. The one exception is a declared
    /// [`Success`](ToolReturnOutcome::Success): a success whose output cannot be
    /// rendered *is* an internal fault, so it becomes an
    /// [`Other`](ToolFailureKind::Other) failure.
    pub(crate) fn into_execution_result(self) -> ToolExecutionResult {
        let ToolReturn {
            output,
            outcome,
            extensions,
        } = self;
        match super::serialize_tool_output(&output) {
            Ok(model_output) => ToolExecutionResult {
                output: ToolOutput::LegacyText(model_output.clone()),
                model_output,
                outcome: outcome.into(),
                extensions,
            },
            Err(err) => {
                let outcome = match outcome {
                    // A success we cannot render is an internal serialization fault.
                    ToolReturnOutcome::Success => {
                        ToolOutcome::Error(ToolFailure::other(err.to_string()))
                    }
                    // A declared failure/denial keeps its classification.
                    other => other.into(),
                };
                let model_output = format!("failed to serialize tool output: {err}");
                ToolExecutionResult {
                    output: ToolOutput::LegacyText(model_output.clone()),
                    model_output,
                    outcome,
                    extensions,
                }
            }
        }
    }
}

// The structured result crosses `.await` points and is the output of the
// `WasmBoxedFuture` returned by `ToolDyn::call_structured`, so on native targets
// it must stay `Send + Sync`. This fails to compile if a future change (e.g. a
// non-`Send` field on `ToolResultExtensions`) drops the property.
#[cfg(not(target_family = "wasm"))]
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ToolExecutionResult>();
    assert_send_sync::<ToolOutcome>();
    assert_send_sync::<ToolFailure>();
};

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

    #[test]
    fn tool_return_outcome_maps_to_observed_outcome() {
        // The tool-authorable set is exactly { Success, Error, Denied } — there is
        // no `ToolReturnOutcome::Skipped` variant, so a tool return can never
        // claim it was skipped (that guarantee is enforced at compile time; there
        // is nothing to test at runtime). Each authorable outcome maps to its
        // observed `ToolOutcome`:
        assert_eq!(
            ToolOutcome::from(ToolReturnOutcome::Success),
            ToolOutcome::Success
        );
        assert_eq!(
            ToolOutcome::from(ToolReturnOutcome::Denied),
            ToolOutcome::Denied
        );
        let failure = ToolFailure::not_found("x");
        assert_eq!(
            ToolOutcome::from(ToolReturnOutcome::Error(failure.clone())),
            ToolOutcome::Error(failure)
        );

        assert_eq!(ToolReturnOutcome::Success.as_str(), "success");
        assert_eq!(ToolReturnOutcome::Denied.as_str(), "denied");
        assert_eq!(
            ToolReturnOutcome::Error(ToolFailure::timeout("t")).as_str(),
            "error"
        );
        assert_eq!(
            ToolReturnOutcome::Error(ToolFailure::timeout("t"))
                .failure()
                .map(|f| f.kind),
            Some(ToolFailureKind::Timeout)
        );
        assert_eq!(ToolReturnOutcome::Denied.failure(), None);
    }

    #[test]
    fn tool_return_denied_surfaces_as_denied_observed_outcome() {
        let result = ToolReturn::denied("refused".to_string()).into_execution_result();
        assert_eq!(*result.outcome(), ToolOutcome::Denied);
        assert_eq!(result.model_output(), "refused");
        assert!(result.outcome().is_denied());
        assert!(!result.outcome().is_skipped());
    }

    #[test]
    fn serialize_failure_preserves_declared_outcome_and_extensions() {
        // An output whose `Serialize` impl always errors, so `into_execution_result`
        // hits the fallback path.
        struct Unserializable;
        impl serde::Serialize for Unserializable {
            fn serialize<S: serde::Serializer>(&self, _s: S) -> Result<S::Ok, S::Error> {
                Err(serde::ser::Error::custom("cannot serialize"))
            }
        }

        #[derive(Clone, Debug, PartialEq)]
        struct ReqId(String);

        // A declared *handled failure* keeps its classification and extensions;
        // only `model_output` falls back to the serialization-error string. It must
        // NOT be silently reclassified to `Error(Other)`.
        let failed = ToolReturn::failed(Unserializable, ToolFailure::rate_limited("slow"))
            .with_extension(ReqId("req-1".into()))
            .into_execution_result();
        assert!(
            failed.outcome().is_error_kind(ToolFailureKind::RateLimited),
            "declared failure classification must survive serialize failure; got {:?}",
            failed.outcome()
        );
        assert_eq!(
            failed.outcome().failure().and_then(|f| f.retryable),
            Some(true),
            "the declared failure's retryable hint must survive"
        );
        assert_eq!(
            failed.extensions().get::<ReqId>(),
            Some(&ReqId("req-1".into())),
            "extensions must survive serialize failure"
        );
        assert!(failed.model_output().contains("failed to serialize"));

        // A declared *denial* is preserved (not turned into an error).
        let denied = ToolReturn::denied(Unserializable).into_execution_result();
        assert!(
            denied.outcome().is_denied(),
            "declared denial must survive serialize failure; got {:?}",
            denied.outcome()
        );

        // A declared *success* that cannot be rendered IS an internal fault.
        let ok = ToolReturn::success(Unserializable).into_execution_result();
        assert!(
            ok.outcome().is_error_kind(ToolFailureKind::Other),
            "a success whose output cannot serialize becomes Other; got {:?}",
            ok.outcome()
        );
    }
}
