//! Canonical structured tool execution errors and the internal dispatch result.

use std::fmt;

use super::ToolContext;

/// Normalized classification for a tool execution error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ToolExecutionErrorKind {
    /// Tool arguments could not be parsed or were rejected.
    InvalidArgs,
    /// The tool did not complete before its deadline.
    Timeout,
    /// The call was cancelled.
    Cancelled,
    /// A requested resource was not found.
    NotFound,
    /// The operation was not permitted by an external system.
    PermissionDenied,
    /// The tool itself ran an authorization or policy check and refused the call.
    Denied,
    /// An upstream service rate-limited the call.
    RateLimited,
    /// An upstream provider returned an error.
    Provider,
    /// A network or transport operation failed.
    Network,
    /// Any failure without a more specific classification.
    Other,
}

impl ToolExecutionErrorKind {
    /// Stable machine-readable name for telemetry and policy matching.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidArgs => "invalid_args",
            Self::Timeout => "timeout",
            Self::Cancelled => "cancelled",
            Self::NotFound => "not_found",
            Self::PermissionDenied => "permission_denied",
            Self::Denied => "denied",
            Self::RateLimited => "rate_limited",
            Self::Provider => "provider",
            Self::Network => "network",
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
            | Self::Denied => Some(false),
            Self::Provider | Self::Other => None,
        }
    }
}

impl fmt::Display for ToolExecutionErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The single public error envelope for tool execution.
///
/// It separates the operator-facing [`message`](Self::message) from optional
/// [`model_feedback`](Self::model_feedback), carries stable classification and
/// retry hints, and may retain a concrete source for downcasting. When model
/// feedback is absent, the operator message remains the model-visible fallback,
/// preserving ordinary `Result` error behavior; use [`with_model_feedback`](Self::with_model_feedback)
/// when those audiences need different wording.
#[derive(Debug)]
#[non_exhaustive]
pub struct ToolExecutionError {
    kind: ToolExecutionErrorKind,
    message: String,
    model_feedback: Option<String>,
    retryable: Option<bool>,
    code: Option<String>,
    http_status: Option<u16>,
    #[cfg(not(target_family = "wasm"))]
    source: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
    #[cfg(target_family = "wasm")]
    source: Option<Box<dyn std::error::Error + 'static>>,
}

impl ToolExecutionError {
    /// Construct an error with explicit classification and operator message.
    pub fn new(kind: ToolExecutionErrorKind, message: impl Into<String>) -> Self {
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

    /// Invalid argument error.
    pub fn invalid_args(message: impl Into<String>) -> Self {
        Self::new(ToolExecutionErrorKind::InvalidArgs, message)
    }

    /// Timeout error.
    pub fn timeout(message: impl Into<String>) -> Self {
        Self::new(ToolExecutionErrorKind::Timeout, message)
    }

    /// Cancellation error.
    pub fn cancelled(message: impl Into<String>) -> Self {
        Self::new(ToolExecutionErrorKind::Cancelled, message)
    }

    /// Not-found error.
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(ToolExecutionErrorKind::NotFound, message)
    }

    /// External permission error.
    pub fn permission_denied(message: impl Into<String>) -> Self {
        Self::new(ToolExecutionErrorKind::PermissionDenied, message)
    }

    /// Tool-authored refusal. Framework hook skips use a distinct execution view.
    pub fn denied(message: impl Into<String>) -> Self {
        Self::new(ToolExecutionErrorKind::Denied, message)
    }

    /// Rate-limit error.
    pub fn rate_limited(message: impl Into<String>) -> Self {
        Self::new(ToolExecutionErrorKind::RateLimited, message)
    }

    /// Upstream provider error.
    pub fn provider(message: impl Into<String>) -> Self {
        Self::new(ToolExecutionErrorKind::Provider, message)
    }

    /// Network or transport error.
    pub fn network(message: impl Into<String>) -> Self {
        Self::new(ToolExecutionErrorKind::Network, message)
    }

    /// Unclassified error.
    pub fn other(message: impl Into<String>) -> Self {
        Self::new(ToolExecutionErrorKind::Other, message)
    }

    /// Attach feedback that is safe and useful to show to the model.
    pub fn with_model_feedback(mut self, feedback: impl Into<String>) -> Self {
        self.model_feedback = Some(feedback.into());
        self
    }

    /// Override the retryability hint.
    pub fn with_retryable(mut self, retryable: bool) -> Self {
        self.retryable = Some(retryable);
        self
    }

    /// Attach a machine-readable error code.
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    /// Attach the originating HTTP status.
    pub fn with_http_status(mut self, status: u16) -> Self {
        self.http_status = Some(status);
        self
    }

    /// Retain a concrete source error for inspection and downcasting.
    #[cfg(not(target_family = "wasm"))]
    pub fn with_source<E>(mut self, source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        self.source = Some(Box::new(source));
        self
    }

    /// Retain a concrete source error for inspection and downcasting.
    #[cfg(target_family = "wasm")]
    pub fn with_source<E>(mut self, source: E) -> Self
    where
        E: std::error::Error + 'static,
    {
        self.source = Some(Box::new(source));
        self
    }

    /// Normalized error kind.
    pub const fn kind(&self) -> ToolExecutionErrorKind {
        self.kind
    }

    /// Operator-facing diagnostic message.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Optional feedback intended for the model.
    pub fn model_feedback(&self) -> Option<&str> {
        self.model_feedback.as_deref()
    }

    /// Whether retrying may succeed, when known.
    pub const fn retryable(&self) -> Option<bool> {
        self.retryable
    }

    /// Machine-readable code, when available.
    pub fn code(&self) -> Option<&str> {
        self.code.as_deref()
    }

    /// Originating HTTP status, when available.
    pub const fn http_status(&self) -> Option<u16> {
        self.http_status
    }

    /// Downcast the retained source to a concrete error type.
    #[cfg(not(target_family = "wasm"))]
    pub fn downcast_source_ref<E>(&self) -> Option<&E>
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        self.source.as_deref()?.downcast_ref::<E>()
    }

    /// Downcast the retained source to a concrete error type.
    #[cfg(target_family = "wasm")]
    pub fn downcast_source_ref<E>(&self) -> Option<&E>
    where
        E: std::error::Error + 'static,
    {
        self.source.as_deref()?.downcast_ref::<E>()
    }

    pub(crate) fn model_output(&self) -> &str {
        self.model_feedback.as_deref().unwrap_or(&self.message)
    }
}

impl fmt::Display for ToolExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

/// Borrowed execution state exposed to hooks and telemetry.
///
/// This is the only public outcome view. Tool refusal is represented by an
/// [`Error`](Self::Error) whose kind is [`ToolExecutionErrorKind::Denied`]; a
/// framework pre-execution skip is [`Skipped`](Self::Skipped).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum ToolExecutionView<'a> {
    /// The tool completed successfully.
    Success,
    /// The tool returned a structured error or refusal.
    Error(&'a ToolExecutionError),
    /// Framework policy skipped the call before tool execution.
    Skipped,
}

impl<'a> ToolExecutionView<'a> {
    /// Stable execution state name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Error(error) if matches!(error.kind(), ToolExecutionErrorKind::Denied) => {
                "denied"
            }
            Self::Error(_) => "error",
            Self::Skipped => "skipped",
        }
    }

    /// Whether the tool completed successfully.
    pub const fn is_success(self) -> bool {
        matches!(self, Self::Success)
    }

    /// Whether execution produced an error or tool-authored refusal.
    pub const fn is_error(self) -> bool {
        matches!(self, Self::Error(_))
    }

    /// Whether framework policy skipped the call before execution.
    pub const fn is_skipped(self) -> bool {
        matches!(self, Self::Skipped)
    }

    /// Whether the tool itself refused the call.
    pub const fn is_denied(self) -> bool {
        matches!(self, Self::Error(error) if matches!(error.kind(), ToolExecutionErrorKind::Denied))
    }

    /// Structured error, if execution failed or was refused.
    pub const fn error(self) -> Option<&'a ToolExecutionError> {
        match self {
            Self::Error(error) => Some(error),
            Self::Success | Self::Skipped => None,
        }
    }

    /// Structured error kind, if execution failed or was refused.
    pub const fn error_kind(self) -> Option<ToolExecutionErrorKind> {
        match self {
            Self::Error(error) => Some(error.kind()),
            Self::Success | Self::Skipped => None,
        }
    }

    /// Whether the error has a specific kind.
    pub fn is_error_kind(self, kind: ToolExecutionErrorKind) -> bool {
        self.error_kind() == Some(kind)
    }
}

/// Crate-private transport from erased dispatch to the agent runtime.
pub(crate) struct ToolDispatchResult {
    pub(crate) model_output: String,
    pub(crate) error: Option<ToolExecutionError>,
    pub(crate) skipped: bool,
    pub(crate) context: ToolContext,
}

impl ToolDispatchResult {
    pub(crate) fn success(model_output: impl Into<String>, context: ToolContext) -> Self {
        Self {
            model_output: model_output.into(),
            error: None,
            skipped: false,
            context,
        }
    }

    pub(crate) fn failed(error: ToolExecutionError, context: ToolContext) -> Self {
        let model_output = error.model_output().to_string();
        Self {
            model_output,
            error: Some(error),
            skipped: false,
            context,
        }
    }

    pub(crate) fn skipped(model_output: impl Into<String>, context: ToolContext) -> Self {
        Self {
            model_output: model_output.into(),
            error: None,
            skipped: true,
            context,
        }
    }

    pub(crate) fn view(&self) -> ToolExecutionView<'_> {
        if self.skipped {
            ToolExecutionView::Skipped
        } else if let Some(error) = self.error.as_ref() {
            ToolExecutionView::Error(error)
        } else {
            ToolExecutionView::Success
        }
    }
}

#[cfg(not(target_family = "wasm"))]
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ToolExecutionError>();
    assert_send_sync::<ToolDispatchResult>();
};

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, thiserror::Error)]
    #[error("concrete")]
    struct Concrete;

    #[test]
    fn envelope_preserves_fields_and_source() {
        let error = ToolExecutionError::rate_limited("slow down")
            .with_model_feedback("please retry later")
            .with_code("RATE_LIMIT")
            .with_http_status(429)
            .with_source(Concrete);
        assert_eq!(error.kind(), ToolExecutionErrorKind::RateLimited);
        assert_eq!(error.retryable(), Some(true));
        assert_eq!(error.model_feedback(), Some("please retry later"));
        assert!(error.downcast_source_ref::<Concrete>().is_some());
    }

    #[test]
    fn refusal_and_skip_remain_distinct() {
        let context = ToolContext::new();
        let denied = ToolDispatchResult::failed(ToolExecutionError::denied("no"), context.clone());
        let skipped = ToolDispatchResult::skipped("policy", context);
        assert_eq!(denied.view().as_str(), "denied");
        assert_eq!(skipped.view().as_str(), "skipped");
    }
}
