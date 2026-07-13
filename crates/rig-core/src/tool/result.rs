//! Canonical structured tool execution errors and runtime results.

use std::sync::Arc;

use crate::tool::ToolContext;
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};

#[cfg(not(target_family = "wasm"))]
type ErrorSource = dyn std::error::Error + Send + Sync + 'static;
#[cfg(target_family = "wasm")]
type ErrorSource = dyn std::error::Error + 'static;

/// Normalized classification for a [`ToolExecutionError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ToolErrorKind {
    /// Arguments could not be decoded or validated.
    InvalidArgs,
    /// Execution exceeded its deadline.
    Timeout,
    /// Execution was cancelled after it started.
    Cancelled,
    /// The requested tool or resource does not exist.
    NotFound,
    /// Execution was prohibited by authorization policy.
    PermissionDenied,
    /// A provider rejected the call because of a rate limit.
    RateLimited,
    /// An upstream provider failed outside transport handling.
    Provider,
    /// The transport or network failed.
    Network,
    /// The tool ran and explicitly refused the requested operation.
    Refused,
    /// A failure that does not fit a normalized category.
    Other,
}

impl ToolErrorKind {
    /// Stable telemetry spelling for this classification.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidArgs => "invalid_args",
            Self::Timeout => "timeout",
            Self::Cancelled => "cancelled",
            Self::NotFound => "not_found",
            Self::PermissionDenied => "permission_denied",
            Self::RateLimited => "rate_limited",
            Self::Provider => "provider",
            Self::Network => "network",
            Self::Refused => "refused",
            Self::Other => "other",
        }
    }

    const fn default_retryable(self) -> Option<bool> {
        match self {
            Self::Timeout | Self::RateLimited | Self::Network => Some(true),
            Self::InvalidArgs
            | Self::Cancelled
            | Self::NotFound
            | Self::PermissionDenied
            | Self::Refused => Some(false),
            Self::Provider | Self::Other => None,
        }
    }
}

impl std::fmt::Display for ToolErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The single public error envelope for tool execution.
///
/// It combines normalized machine-readable classification with an
/// operator-facing message, optional model-specific feedback, retry and HTTP
/// hints, and an optional concrete source. Sources remain downcastable through
/// [`downcast_source_ref`](Self::downcast_source_ref) without exposing a second
/// public source wrapper type.
#[derive(Clone)]
#[non_exhaustive]
pub struct ToolExecutionError {
    /// Normalized failure classification.
    pub kind: ToolErrorKind,
    /// Operator-facing diagnostic message.
    pub message: String,
    /// Optional feedback rendered to the model instead of `message`.
    pub model_feedback: Option<String>,
    /// Whether retrying the operation can reasonably succeed.
    pub retryable: Option<bool>,
    /// Provider- or application-defined machine-readable code.
    pub code: Option<String>,
    /// Associated HTTP status, when applicable.
    pub http_status: Option<u16>,
    source: Option<Arc<ErrorSource>>,
}

impl std::fmt::Debug for ToolExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolExecutionError")
            .field("kind", &self.kind)
            .field("message", &self.message)
            .field("model_feedback", &self.model_feedback)
            .field("retryable", &self.retryable)
            .field("code", &self.code)
            .field("http_status", &self.http_status)
            .field("has_source", &self.source.is_some())
            .finish()
    }
}

impl ToolExecutionError {
    /// Create an error and apply the classification's default retryability.
    pub fn new(kind: ToolErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            model_feedback: None,
            retryable: kind.default_retryable(),
            code: None,
            http_status: None,
            source: None,
        }
    }

    /// Invalid-argument error.
    pub fn invalid_args(message: impl Into<String>) -> Self {
        Self::new(ToolErrorKind::InvalidArgs, message)
    }

    /// Timeout error.
    pub fn timeout(message: impl Into<String>) -> Self {
        Self::new(ToolErrorKind::Timeout, message)
    }

    /// Cancellation error.
    pub fn cancelled(message: impl Into<String>) -> Self {
        Self::new(ToolErrorKind::Cancelled, message)
    }

    /// Missing-tool or missing-resource error.
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(ToolErrorKind::NotFound, message)
    }

    /// Permission error.
    pub fn permission_denied(message: impl Into<String>) -> Self {
        Self::new(ToolErrorKind::PermissionDenied, message)
    }

    /// Rate-limit error.
    pub fn rate_limited(message: impl Into<String>) -> Self {
        Self::new(ToolErrorKind::RateLimited, message)
    }

    /// Upstream-provider error.
    pub fn provider(message: impl Into<String>) -> Self {
        Self::new(ToolErrorKind::Provider, message)
    }

    /// Network or transport error.
    pub fn network(message: impl Into<String>) -> Self {
        Self::new(ToolErrorKind::Network, message)
    }

    /// Tool-authored refusal. This is distinct from a framework skip, where the
    /// tool body never runs.
    pub fn refused(message: impl Into<String>) -> Self {
        Self::new(ToolErrorKind::Refused, message)
    }

    /// Unclassified error.
    pub fn other(message: impl Into<String>) -> Self {
        Self::new(ToolErrorKind::Other, message)
    }

    /// Build an error around a concrete downcastable source.
    pub fn from_source<E>(kind: ToolErrorKind, source: E) -> Self
    where
        E: std::error::Error + WasmCompatSend + WasmCompatSync + 'static,
    {
        let message = source.to_string();
        Self::new(kind, message).with_source(source)
    }

    /// Override the text rendered to the model while retaining `message` for
    /// operators.
    pub fn with_model_feedback(mut self, feedback: impl Into<String>) -> Self {
        self.model_feedback = Some(feedback.into());
        self
    }

    /// Override the retryability hint.
    pub fn with_retryable(mut self, retryable: bool) -> Self {
        self.retryable = Some(retryable);
        self
    }

    /// Attach a provider- or application-defined code.
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    /// Attach an HTTP status.
    pub fn with_http_status(mut self, status: u16) -> Self {
        self.http_status = Some(status);
        self
    }

    /// Attach a concrete source that can later be downcast.
    pub fn with_source<E>(mut self, source: E) -> Self
    where
        E: std::error::Error + WasmCompatSend + WasmCompatSync + 'static,
    {
        self.source = Some(Arc::new(source));
        self
    }

    /// Downcast the concrete source by type.
    pub fn downcast_source_ref<E>(&self) -> Option<&E>
    where
        E: std::error::Error + 'static,
    {
        self.source
            .as_deref()
            .and_then(|source| source.downcast_ref::<E>())
    }

    /// Text rendered as the model-visible tool result.
    pub fn model_message(&self) -> &str {
        self.model_feedback.as_deref().unwrap_or(&self.message)
    }
}

impl std::fmt::Display for ToolExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)
    }
}

impl std::error::Error for ToolExecutionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_deref()
            .map(|source| source as &(dyn std::error::Error + 'static))
    }
}

/// Framework-observed status of one tool invocation.
///
/// Tool authors return ordinary `Result<Output, ToolExecutionError>` and do not
/// construct this type. [`Skipped`](Self::Skipped) is framework-only and means
/// the tool body did not run; a tool-authored refusal is
/// [`Error`](Self::Error) with [`ToolErrorKind::Refused`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ToolExecutionStatus {
    /// The tool returned and its output was rendered successfully.
    Success,
    /// The tool or dispatch boundary failed.
    Error(ToolExecutionError),
    /// A pre-execution hook skipped the call, so the tool body did not run.
    Skipped,
}

impl ToolExecutionStatus {
    /// Stable telemetry spelling for the status.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Error(error) if matches!(error.kind, ToolErrorKind::Refused) => "refused",
            Self::Error(_) => "error",
            Self::Skipped => "skipped",
        }
    }

    /// Whether execution succeeded.
    pub const fn is_success(&self) -> bool {
        matches!(self, Self::Success)
    }

    /// Whether execution failed, including a tool refusal.
    pub const fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }

    /// Whether the framework skipped the call before execution.
    pub const fn is_skipped(&self) -> bool {
        matches!(self, Self::Skipped)
    }

    /// Whether the tool ran and refused the operation.
    pub fn is_refused(&self) -> bool {
        matches!(self, Self::Error(error) if error.kind == ToolErrorKind::Refused)
    }

    /// Structured error, if execution failed.
    pub fn error(&self) -> Option<&ToolExecutionError> {
        match self {
            Self::Error(error) => Some(error),
            Self::Success | Self::Skipped => None,
        }
    }

    /// Normalized error kind, if execution failed.
    pub fn error_kind(&self) -> Option<ToolErrorKind> {
        self.error().map(|error| error.kind)
    }

    /// Whether execution failed with `kind`.
    pub fn is_error_kind(&self, kind: ToolErrorKind) -> bool {
        self.error_kind() == Some(kind)
    }
}

/// The one structured result returned by the tool runtime and observed by hooks.
///
/// `model_output` preserves Rig's existing rendering rules: strings remain
/// verbatim and structured or multimodal values are JSON-encoded. Result
/// metadata stays in the associated [`ToolContext`] and is never rendered.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ToolExecution {
    pub(crate) model_output: String,
    pub(crate) status: ToolExecutionStatus,
    pub(crate) context: ToolContext,
}

impl ToolExecution {
    pub(crate) fn success(model_output: String, context: ToolContext) -> Self {
        Self {
            model_output,
            status: ToolExecutionStatus::Success,
            context,
        }
    }

    pub(crate) fn failed(error: ToolExecutionError, context: ToolContext) -> Self {
        let model_output = error.model_message().to_string();
        Self {
            model_output,
            status: ToolExecutionStatus::Error(error),
            context,
        }
    }

    pub(crate) fn skipped(reason: impl Into<String>, context: ToolContext) -> Self {
        Self {
            model_output: reason.into(),
            status: ToolExecutionStatus::Skipped,
            context,
        }
    }

    /// Output delivered to the model unless a result hook rewrites it.
    pub fn model_output(&self) -> &str {
        &self.model_output
    }

    /// Structured execution status.
    pub fn status(&self) -> &ToolExecutionStatus {
        &self.status
    }

    /// Read typed result metadata attached by the tool.
    pub fn metadata<T>(&self) -> Option<&T>
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.context.metadata::<T>()
    }

    /// Access the invocation context, including inbound values and result
    /// metadata. The context itself is never sent to the model.
    pub fn context(&self) -> &ToolContext {
        &self.context
    }
}

#[cfg(not(target_family = "wasm"))]
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ToolExecution>();
    assert_send_sync::<ToolExecutionError>();
};

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, thiserror::Error)]
    #[error("concrete source")]
    struct ConcreteSource;

    #[test]
    fn constructors_set_normalized_retryability() {
        assert_eq!(ToolExecutionError::timeout("t").retryable, Some(true));
        assert_eq!(ToolExecutionError::rate_limited("r").retryable, Some(true));
        assert_eq!(ToolExecutionError::network("n").retryable, Some(true));
        assert_eq!(ToolExecutionError::not_found("n").retryable, Some(false));
        assert_eq!(ToolExecutionError::invalid_args("i").retryable, Some(false));
        assert_eq!(ToolExecutionError::refused("r").retryable, Some(false));
        assert_eq!(ToolExecutionError::provider("p").retryable, None);
        assert_eq!(ToolExecutionError::other("o").retryable, None);
    }

    #[test]
    fn envelope_preserves_feedback_hints_and_concrete_source() {
        let error = ToolExecutionError::from_source(ToolErrorKind::Timeout, ConcreteSource)
            .with_model_feedback("try later")
            .with_retryable(false)
            .with_code("UPSTREAM_TIMEOUT")
            .with_http_status(504);

        assert_eq!(error.kind, ToolErrorKind::Timeout);
        assert_eq!(error.message, "concrete source");
        assert_eq!(error.model_message(), "try later");
        assert_eq!(error.retryable, Some(false));
        assert_eq!(error.code.as_deref(), Some("UPSTREAM_TIMEOUT"));
        assert_eq!(error.http_status, Some(504));
        assert!(error.downcast_source_ref::<ConcreteSource>().is_some());
        assert!(std::error::Error::source(&error).is_some());
    }

    #[test]
    fn status_distinguishes_skip_from_tool_refusal() {
        let skipped = ToolExecutionStatus::Skipped;
        let refused = ToolExecutionStatus::Error(ToolExecutionError::refused("no"));

        assert!(skipped.is_skipped());
        assert!(!skipped.is_refused());
        assert!(refused.is_refused());
        assert!(!refused.is_skipped());
        assert_eq!(skipped.as_str(), "skipped");
        assert_eq!(refused.as_str(), "refused");
    }

    #[test]
    fn execution_keeps_model_feedback_and_metadata() {
        #[derive(Clone, Debug, PartialEq)]
        struct RequestId(&'static str);

        let mut context = ToolContext::new();
        context.insert_metadata(RequestId("request-7"));
        let execution = ToolExecution::failed(
            ToolExecutionError::provider("operator detail")
                .with_model_feedback("please try another tool"),
            context,
        );

        assert_eq!(execution.model_output(), "please try another tool");
        assert!(execution.status().is_error_kind(ToolErrorKind::Provider));
        assert_eq!(
            execution.metadata::<RequestId>(),
            Some(&RequestId("request-7"))
        );
    }
}
