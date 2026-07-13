//! Canonical structured tool errors and execution results.

use std::{error::Error, sync::Arc};

use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};

/// Normalized classification for a tool execution error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ToolErrorKind {
    /// Arguments could not be decoded or validated.
    InvalidArgs,
    /// Execution exceeded its deadline.
    Timeout,
    /// Execution was cancelled.
    Cancelled,
    /// The requested tool or resource was not found.
    NotFound,
    /// The tool refused the operation or authorization failed.
    PermissionDenied,
    /// A rate limit was reached.
    RateLimited,
    /// An upstream provider failed.
    Provider,
    /// A network operation failed.
    Network,
    /// Any other failure.
    Other,
}

impl ToolErrorKind {
    /// Stable machine-readable name.
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
            Self::Other => "other",
        }
    }

    const fn default_retryable(self) -> Option<bool> {
        match self {
            Self::Timeout | Self::RateLimited | Self::Network => Some(true),
            Self::InvalidArgs | Self::Cancelled | Self::NotFound | Self::PermissionDenied => {
                Some(false)
            }
            Self::Provider | Self::Other => None,
        }
    }
}

impl std::fmt::Display for ToolErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One public envelope for every tool execution failure.
///
/// It carries normalized policy fields, separate operator-facing and optional
/// model-facing messages, and an optional concrete source that can be downcast.
#[derive(Clone)]
pub struct ToolExecutionError {
    kind: ToolErrorKind,
    message: String,
    model_feedback: Option<String>,
    retryable: Option<bool>,
    code: Option<String>,
    http_status: Option<u16>,
    #[cfg(not(target_family = "wasm"))]
    source: Option<Arc<dyn Error + Send + Sync + 'static>>,
    #[cfg(target_family = "wasm")]
    source: Option<Arc<dyn Error + 'static>>,
}

impl ToolExecutionError {
    /// Construct an error with an explicit normalized kind.
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

    /// Invalid arguments.
    pub fn invalid_args(message: impl Into<String>) -> Self {
        Self::new(ToolErrorKind::InvalidArgs, message)
    }

    /// Timeout.
    pub fn timeout(message: impl Into<String>) -> Self {
        Self::new(ToolErrorKind::Timeout, message)
    }

    /// Cancellation.
    pub fn cancelled(message: impl Into<String>) -> Self {
        Self::new(ToolErrorKind::Cancelled, message)
    }

    /// Missing tool or resource.
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(ToolErrorKind::NotFound, message)
    }

    /// Tool refusal or authorization failure.
    pub fn permission_denied(message: impl Into<String>) -> Self {
        Self::new(ToolErrorKind::PermissionDenied, message)
    }

    /// Rate limit.
    pub fn rate_limited(message: impl Into<String>) -> Self {
        Self::new(ToolErrorKind::RateLimited, message)
    }

    /// Upstream provider failure.
    pub fn provider(message: impl Into<String>) -> Self {
        Self::new(ToolErrorKind::Provider, message)
    }

    /// Network failure.
    pub fn network(message: impl Into<String>) -> Self {
        Self::new(ToolErrorKind::Network, message)
    }

    /// Catch-all failure.
    pub fn other(message: impl Into<String>) -> Self {
        Self::new(ToolErrorKind::Other, message)
    }

    /// Build an `Other` error from a concrete source, preserving it for downcast.
    pub fn from_error<E>(error: E) -> Self
    where
        E: Error + WasmCompatSend + WasmCompatSync + 'static,
    {
        #[cfg(not(target_family = "wasm"))]
        {
            let source: Box<dyn Error + Send + Sync + 'static> = Box::new(error);
            return match source.downcast::<Self>() {
                Ok(error) => *error,
                Err(source) => {
                    let message = source.to_string();
                    let mut error = Self::other(message);
                    error.source = Some(Arc::from(source));
                    error
                }
            };
        }
        #[cfg(target_family = "wasm")]
        {
            let source: Box<dyn Error + 'static> = Box::new(error);
            match source.downcast::<Self>() {
                Ok(error) => *error,
                Err(source) => {
                    let message = source.to_string();
                    let mut error = Self::other(message);
                    error.source = Some(Arc::from(source));
                    error
                }
            }
        }
    }

    /// Attach model-visible feedback. Without it, the operator message is used.
    pub fn with_model_feedback(mut self, feedback: impl Into<String>) -> Self {
        self.model_feedback = Some(feedback.into());
        self
    }

    /// Override the retryability hint.
    pub fn with_retryable(mut self, retryable: bool) -> Self {
        self.retryable = Some(retryable);
        self
    }

    /// Attach an application/provider code.
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    /// Attach an HTTP status.
    pub fn with_http_status(mut self, status: u16) -> Self {
        self.http_status = Some(status);
        self
    }

    /// Preserve a concrete source for later downcasting.
    pub fn with_source<E>(mut self, source: E) -> Self
    where
        E: Error + WasmCompatSend + WasmCompatSync + 'static,
    {
        self.source = Some(Arc::new(source));
        self
    }

    /// Normalized kind.
    pub const fn kind(&self) -> ToolErrorKind {
        self.kind
    }

    /// Operator-facing message.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Model feedback, falling back to the operator-facing message.
    pub fn model_feedback(&self) -> &str {
        self.model_feedback.as_deref().unwrap_or(&self.message)
    }

    /// Explicit model feedback, if configured.
    pub fn explicit_model_feedback(&self) -> Option<&str> {
        self.model_feedback.as_deref()
    }

    /// Retryability hint.
    pub const fn retryable(&self) -> Option<bool> {
        self.retryable
    }

    /// Application/provider code.
    pub fn code(&self) -> Option<&str> {
        self.code.as_deref()
    }

    /// HTTP status.
    pub const fn http_status(&self) -> Option<u16> {
        self.http_status
    }

    /// Downcast the concrete source to `E`.
    pub fn downcast_ref<E>(&self) -> Option<&E>
    where
        E: Error + WasmCompatSend + WasmCompatSync + 'static,
    {
        self.source.as_ref()?.downcast_ref::<E>()
    }

    /// Whether the concrete source has type `E`.
    pub fn is<E>(&self) -> bool
    where
        E: Error + WasmCompatSend + WasmCompatSync + 'static,
    {
        self.downcast_ref::<E>().is_some()
    }
}

impl std::fmt::Display for ToolExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
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
            .field("source", &self.source.as_ref().map(|_| "<redacted>"))
            .finish()
    }
}

impl Error for ToolExecutionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source
            .as_deref()
            .map(|source| source as &(dyn Error + 'static))
    }
}

/// The single structured execution view used by dispatch, hooks, and telemetry.
///
/// Tools never construct this type: their ordinary `Result` is converted at the
/// crate-private erased boundary. Framework skips are represented separately
/// from failures, so a policy skip cannot be confused with a tool refusal.
#[derive(Clone, Debug)]
pub struct ToolResult {
    model_output: String,
    error: Option<ToolExecutionError>,
    skipped: bool,
}

impl ToolResult {
    pub(crate) fn success(model_output: impl Into<String>) -> Self {
        Self {
            model_output: model_output.into(),
            error: None,
            skipped: false,
        }
    }

    pub(crate) fn failed(error: ToolExecutionError) -> Self {
        Self {
            model_output: error.model_feedback().to_string(),
            error: Some(error),
            skipped: false,
        }
    }

    pub(crate) fn skipped(reason: impl Into<String>) -> Self {
        Self {
            model_output: reason.into(),
            error: None,
            skipped: true,
        }
    }

    /// Text delivered to the model before any presentation-only hook rewrite.
    pub fn model_output(&self) -> &str {
        &self.model_output
    }

    /// Structured execution error, if execution failed.
    pub fn error(&self) -> Option<&ToolExecutionError> {
        self.error.as_ref()
    }

    /// Whether the tool completed successfully.
    pub fn is_success(&self) -> bool {
        self.error.is_none() && !self.skipped
    }

    /// Whether execution failed.
    pub fn is_error(&self) -> bool {
        self.error.is_some()
    }

    /// Whether the framework skipped execution before the tool body ran.
    pub const fn is_skipped(&self) -> bool {
        self.skipped
    }

    /// Whether a tool refused execution.
    pub fn is_refused(&self) -> bool {
        self.error
            .as_ref()
            .is_some_and(|error| error.kind == ToolErrorKind::PermissionDenied)
    }

    /// Whether this is an error of exactly `kind`.
    pub fn is_error_kind(&self, kind: ToolErrorKind) -> bool {
        self.error.as_ref().is_some_and(|error| error.kind == kind)
    }

    pub(crate) fn status_name(&self) -> &'static str {
        if self.skipped {
            "skipped"
        } else if self.error.is_some() {
            "error"
        } else {
            "success"
        }
    }
}

#[cfg(not(target_family = "wasm"))]
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ToolExecutionError>();
    assert_send_sync::<ToolResult>();
};

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, thiserror::Error)]
    #[error("secret detail")]
    struct Concrete;

    #[test]
    fn envelope_is_classified_cloneable_downcastable_and_redacted() {
        let error = ToolExecutionError::provider("operator message")
            .with_model_feedback("safe feedback")
            .with_http_status(503)
            .with_source(Concrete);
        let cloned = error.clone();
        assert_eq!(error.kind(), ToolErrorKind::Provider);
        assert_eq!(error.model_feedback(), "safe feedback");
        assert_eq!(error.http_status(), Some(503));
        assert!(cloned.is::<Concrete>());
        assert!(!format!("{error:?}").contains("secret detail"));
    }

    #[test]
    fn converting_an_existing_envelope_preserves_classification() {
        let error = ToolExecutionError::from_error(ToolExecutionError::timeout("slow"));
        assert_eq!(error.kind(), ToolErrorKind::Timeout);
        assert_eq!(error.retryable(), Some(true));
    }

    #[test]
    fn skip_and_refusal_are_distinct() {
        let skipped = ToolResult::skipped("policy");
        let refused = ToolResult::failed(ToolExecutionError::permission_denied("tool refused"));
        assert!(skipped.is_skipped());
        assert!(!skipped.is_refused());
        assert!(refused.is_refused());
        assert!(!refused.is_skipped());
    }
}

#[cfg(test)]
mod migrated_tests {
    use super::*;

    #[test]
    fn per_kind_constructors_set_default_retryability() {
        for (error, retryable) in [
            (ToolExecutionError::timeout("t"), Some(true)),
            (ToolExecutionError::rate_limited("r"), Some(true)),
            (ToolExecutionError::network("n"), Some(true)),
            (ToolExecutionError::not_found("nf"), Some(false)),
            (ToolExecutionError::permission_denied("p"), Some(false)),
            (ToolExecutionError::invalid_args("i"), Some(false)),
            (ToolExecutionError::cancelled("c"), Some(false)),
            (ToolExecutionError::provider("p"), None),
            (ToolExecutionError::other("o"), None),
        ] {
            assert_eq!(error.retryable(), retryable);
        }
    }

    #[test]
    fn error_builder_preserves_policy_fields_and_feedback() {
        let error = ToolExecutionError::rate_limited("operator")
            .with_model_feedback("slow down")
            .with_retryable(false)
            .with_code("RATE_42")
            .with_http_status(429);
        assert_eq!(error.kind(), ToolErrorKind::RateLimited);
        assert_eq!(error.message(), "operator");
        assert_eq!(error.model_feedback(), "slow down");
        assert_eq!(error.retryable(), Some(false));
        assert_eq!(error.code(), Some("RATE_42"));
        assert_eq!(error.http_status(), Some(429));
        let result = ToolResult::failed(error);
        assert_eq!(result.model_output(), "slow down");
        assert!(result.is_error_kind(ToolErrorKind::RateLimited));
    }

    #[test]
    fn success_preserves_multiline_output_verbatim() {
        let result = ToolResult::success("hello\nworld");
        assert!(result.is_success());
        assert_eq!(result.model_output(), "hello\nworld");
        assert!(result.error().is_none());
    }

    #[test]
    fn result_states_are_mutually_distinguishable() {
        let success = ToolResult::success("ok");
        let failure = ToolResult::failed(ToolExecutionError::not_found("missing"));
        let skipped = ToolResult::skipped("policy");
        let refused = ToolResult::failed(ToolExecutionError::permission_denied("denied"));
        assert!(success.is_success());
        assert!(failure.is_error());
        assert!(skipped.is_skipped());
        assert!(refused.is_refused());
        assert!(!skipped.is_refused());
        assert!(!refused.is_skipped());
        assert_eq!(success.status_name(), "success");
        assert_eq!(failure.status_name(), "error");
        assert_eq!(skipped.status_name(), "skipped");
    }

    #[test]
    fn from_error_keeps_existing_envelope_and_wraps_other_sources() {
        #[derive(Debug, thiserror::Error)]
        #[error("boom")]
        struct Boom;
        let existing = ToolExecutionError::timeout("slow").with_code("T");
        let kept = ToolExecutionError::from_error(existing);
        assert_eq!(kept.kind(), ToolErrorKind::Timeout);
        assert_eq!(kept.code(), Some("T"));
        let wrapped = ToolExecutionError::from_error(Boom);
        assert_eq!(wrapped.kind(), ToolErrorKind::Other);
        assert!(wrapped.is::<Boom>());
    }
}
