//! Canonical structured tool errors and execution results.

use std::{error::Error, sync::Arc};

use crate::{
    tool::ToolOutput,
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

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
    /// An authorization or permission check failed. Intentional tool refusals
    /// use this normalized kind with a separate refusal disposition.
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

    const fn default_model_feedback(self) -> &'static str {
        match self {
            Self::InvalidArgs => "tool arguments were invalid",
            Self::Timeout => "tool execution timed out",
            Self::Cancelled => "tool execution was cancelled",
            Self::NotFound => "the requested tool or resource was not found",
            Self::PermissionDenied => "the tool denied the request",
            Self::RateLimited => "the tool was rate limited; try again later",
            Self::Provider => "the tool provider failed",
            Self::Network => "the tool could not reach its upstream service",
            Self::Other => "the tool failed",
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
/// It carries normalized policy fields, separate operator-facing diagnostics
/// and model-visible output, and an optional concrete source that can be
/// downcast. Explicit constructors use the diagnostic message as model-visible
/// output so deliberately authored validation failures remain actionable.
/// [`Self::from_error`] instead treats an arbitrary source as operator-only and
/// exposes safe kind-level feedback. Use [`Self::with_model_feedback`] or
/// [`Self::with_model_output`] to provide a purpose-built presentation.
#[derive(Clone)]
pub struct ToolExecutionError {
    kind: ToolErrorKind,
    message: String,
    model_output: Box<ToolOutput>,
    retryable: Option<bool>,
    code: Option<String>,
    http_status: Option<u16>,
    refusal: bool,
    #[cfg(not(target_family = "wasm"))]
    source: Option<Arc<dyn Error + Send + Sync + 'static>>,
    #[cfg(target_family = "wasm")]
    source: Option<Arc<dyn Error + 'static>>,
}

impl ToolExecutionError {
    /// Construct an error with an explicit normalized kind.
    pub fn new(kind: ToolErrorKind, message: impl Into<String>) -> Self {
        let message = message.into();
        Self {
            kind,
            model_output: Box::new(ToolOutput::text(message.clone())),
            message,
            retryable: kind.default_retryable(),
            code: None,
            http_status: None,
            refusal: false,
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

    /// An authorization or permission failure.
    ///
    /// This is an ordinary execution error. Use [`Self::refused`] when the tool
    /// intentionally declines the operation so hooks and telemetry can preserve
    /// the refusal as a distinct disposition.
    pub fn permission_denied(message: impl Into<String>) -> Self {
        Self::new(ToolErrorKind::PermissionDenied, message)
    }

    /// An intentional, tool-authored refusal.
    ///
    /// Refusals use the normalized [`ToolErrorKind::PermissionDenied`] kind but
    /// remain distinct from permission failures in [`ToolResult`].
    pub fn refused(message: impl Into<String>) -> Self {
        let mut error = Self::new(ToolErrorKind::PermissionDenied, message);
        error.refusal = true;
        error
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

    /// Build a safely presented `Other` error from a concrete source.
    ///
    /// The source's display string remains available as the operator-facing
    /// [`Self::message`] and the source remains downcastable, but the model sees
    /// only the stable feedback for [`ToolErrorKind::Other`]. Passing an existing
    /// `ToolExecutionError` preserves its classification and presentation.
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
                    let mut error = Self::other(message).redact_model_feedback();
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
                    let mut error = Self::other(message).redact_model_feedback();
                    error.source = Some(Arc::from(source));
                    error
                }
            }
        }
    }

    /// Replace the model-visible output with literal text feedback.
    pub fn with_model_feedback(mut self, feedback: impl Into<String>) -> Self {
        self.model_output = Box::new(ToolOutput::text(feedback));
        self
    }

    /// Replace the model-visible output with canonical JSON or multimodal
    /// content.
    pub fn with_model_output(mut self, output: ToolOutput) -> Self {
        self.model_output = Box::new(output);
        self
    }

    /// Replace potentially sensitive diagnostics with stable, kind-specific
    /// model feedback.
    ///
    /// Explicit error constructors make messages model-visible by default to
    /// keep failures actionable. Call this when an explicitly constructed
    /// error's operator diagnostic may contain secrets; [`Self::from_error`]
    /// already uses this safe presentation for arbitrary source errors.
    pub fn redact_model_feedback(mut self) -> Self {
        self.model_output = Box::new(ToolOutput::text(self.kind.default_model_feedback()));
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

    /// Literal model feedback, when the presentation is exactly one plain text
    /// block.
    ///
    /// Use [`Self::model_output`] for JSON or multimodal feedback.
    pub fn model_feedback(&self) -> Option<&str> {
        self.model_output.as_text()
    }

    /// Canonical model-visible presentation for this error.
    pub fn model_output(&self) -> &ToolOutput {
        &self.model_output
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

    /// Whether the tool intentionally refused the operation.
    pub const fn is_refusal(&self) -> bool {
        self.refusal
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
            .field("model_output", &self.model_output)
            .field("retryable", &self.retryable)
            .field("code", &self.code)
            .field("http_status", &self.http_status)
            .field("refusal", &self.refusal)
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

/// Private mutually exclusive state behind [`ToolResult`].
#[derive(Clone, Debug)]
enum ToolDisposition {
    Success(ToolOutput),
    Error(ToolExecutionError),
    Refused(ToolExecutionError),
    Skipped(ToolOutput),
}

/// The single structured execution view used by dispatch, hooks, and telemetry.
///
/// Each result has exactly one disposition. The tagged state is private so tool
/// authors keep returning ordinary `Result` values while runtime callers use
/// the stable query methods on this type.
#[derive(Clone, Debug)]
pub struct ToolResult {
    disposition: ToolDisposition,
}

impl ToolResult {
    /// Construct a successful canonical result at a runtime boundary.
    pub fn success(output: ToolOutput) -> Self {
        Self {
            disposition: ToolDisposition::Success(output),
        }
    }

    /// Construct a failed or refused canonical result.
    pub fn failed(error: ToolExecutionError) -> Self {
        let disposition = if error.is_refusal() {
            ToolDisposition::Refused(error)
        } else {
            ToolDisposition::Error(error)
        };
        Self { disposition }
    }

    /// Construct a result for a call suppressed before execution.
    pub fn skipped(reason: impl Into<String>) -> Self {
        Self {
            disposition: ToolDisposition::Skipped(ToolOutput::text(reason)),
        }
    }

    /// Canonical model-visible output before any presentation-only hook rewrite.
    pub fn output(&self) -> &ToolOutput {
        match &self.disposition {
            ToolDisposition::Success(output) | ToolDisposition::Skipped(output) => output,
            ToolDisposition::Error(error) | ToolDisposition::Refused(error) => error.model_output(),
        }
    }

    /// Structured execution error, if execution failed.
    ///
    /// Intentional refusals are available through [`Self::refusal`] instead.
    pub fn error(&self) -> Option<&ToolExecutionError> {
        match &self.disposition {
            ToolDisposition::Error(error) => Some(error),
            ToolDisposition::Success(_)
            | ToolDisposition::Refused(_)
            | ToolDisposition::Skipped(_) => None,
        }
    }

    /// Structured refusal details, if the tool intentionally declined the call.
    ///
    /// This is mutually exclusive with [`Self::error`].
    pub fn refusal(&self) -> Option<&ToolExecutionError> {
        match &self.disposition {
            ToolDisposition::Refused(error) => Some(error),
            ToolDisposition::Success(_)
            | ToolDisposition::Error(_)
            | ToolDisposition::Skipped(_) => None,
        }
    }

    /// Whether the tool completed successfully.
    pub fn is_success(&self) -> bool {
        matches!(&self.disposition, ToolDisposition::Success(_))
    }

    /// Whether execution failed.
    ///
    /// An intentional refusal is not an execution error; inspect
    /// [`Self::is_refused`] instead.
    pub fn is_error(&self) -> bool {
        matches!(&self.disposition, ToolDisposition::Error(_))
    }

    /// Whether the framework skipped execution before the tool body ran.
    pub fn is_skipped(&self) -> bool {
        matches!(&self.disposition, ToolDisposition::Skipped(_))
    }

    /// Whether a tool refused execution.
    pub fn is_refused(&self) -> bool {
        matches!(&self.disposition, ToolDisposition::Refused(_))
    }

    /// Whether this is an error of exactly `kind`.
    ///
    /// A refusal does not match, even though its envelope uses the normalized
    /// [`ToolErrorKind::PermissionDenied`] kind.
    pub fn is_error_kind(&self, kind: ToolErrorKind) -> bool {
        self.error().is_some_and(|error| error.kind == kind)
    }

    /// Stable telemetry name for this result's disposition.
    pub fn status_name(&self) -> &'static str {
        match &self.disposition {
            ToolDisposition::Success(_) => "success",
            ToolDisposition::Error(_) => "error",
            ToolDisposition::Refused(_) => "denied",
            ToolDisposition::Skipped(_) => "skipped",
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
        assert_eq!(error.model_feedback(), Some("safe feedback"));
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
    fn detailed_diagnostics_are_model_visible_by_default() {
        let error = ToolExecutionError::provider("upstream rejected field `region`");
        let result = ToolResult::failed(error.clone());

        assert_eq!(error.message(), "upstream rejected field `region`");
        assert_eq!(
            error.model_feedback(),
            Some("upstream rejected field `region`")
        );
        assert_eq!(
            result.output().as_text(),
            Some("upstream rejected field `region`")
        );
    }

    #[test]
    fn sensitive_diagnostics_can_be_explicitly_redacted() {
        let error = ToolExecutionError::provider("authorization header Bearer secret-token")
            .redact_model_feedback();
        let result = ToolResult::failed(error.clone());

        assert_eq!(error.message(), "authorization header Bearer secret-token");
        assert_eq!(error.model_feedback(), Some("the tool provider failed"));
        assert_eq!(result.output().as_text(), Some("the tool provider failed"));
        assert!(!result.output().render().contains("secret-token"));
    }

    #[test]
    fn errors_can_expose_structured_model_output() {
        let output = ToolOutput::json(serde_json::json!({
            "error": "invalid region",
            "allowed": ["us", "eu"]
        }));
        let result = ToolResult::failed(
            ToolExecutionError::invalid_args("region was invalid")
                .with_model_output(output.clone()),
        );

        assert_eq!(result.output(), &output);
        assert_eq!(result.error().unwrap().model_output(), &output);
        assert_eq!(result.error().unwrap().model_feedback(), None);
    }

    #[test]
    fn skip_refusal_and_permission_failure_are_distinct() {
        let skipped = ToolResult::skipped("policy");
        let refused = ToolResult::failed(ToolExecutionError::refused("tool refused"));
        let permission_failure = ToolResult::failed(ToolExecutionError::permission_denied(
            "authorization failed",
        ));
        assert!(skipped.is_skipped());
        assert!(!skipped.is_refused());
        assert!(refused.is_refused());
        assert!(!refused.is_skipped());
        assert!(!refused.is_error());
        assert!(refused.error().is_none());
        assert!(refused.refusal().is_some_and(|error| error.is_refusal()));
        assert!(permission_failure.is_error());
        assert!(!permission_failure.is_refused());
        assert!(permission_failure.refusal().is_none());
        assert!(permission_failure.is_error_kind(ToolErrorKind::PermissionDenied));
        assert!(!refused.is_error_kind(ToolErrorKind::PermissionDenied));
        assert_eq!(refused.status_name(), "denied");
        assert_eq!(permission_failure.status_name(), "error");
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
        assert_eq!(error.model_feedback(), Some("slow down"));
        assert_eq!(error.retryable(), Some(false));
        assert_eq!(error.code(), Some("RATE_42"));
        assert_eq!(error.http_status(), Some(429));
        let result = ToolResult::failed(error);
        assert_eq!(result.output().as_text(), Some("slow down"));
        assert!(result.is_error_kind(ToolErrorKind::RateLimited));
    }

    #[test]
    fn success_preserves_multiline_output_verbatim() {
        let result = ToolResult::success(ToolOutput::text("hello\nworld"));
        assert!(result.is_success());
        assert_eq!(result.output().as_text(), Some("hello\nworld"));
        assert!(result.error().is_none());
    }

    #[test]
    fn result_states_are_mutually_distinguishable() {
        let success = ToolResult::success(ToolOutput::text("ok"));
        let failure = ToolResult::failed(ToolExecutionError::not_found("missing"));
        let skipped = ToolResult::skipped("policy");
        let refused = ToolResult::failed(ToolExecutionError::refused("denied"));
        assert!(success.is_success());
        assert!(failure.is_error());
        assert!(skipped.is_skipped());
        assert!(refused.is_refused());
        assert!(!refused.is_error());
        assert!(!skipped.is_refused());
        assert!(!refused.is_skipped());
        assert_eq!(success.status_name(), "success");
        assert_eq!(failure.status_name(), "error");
        assert_eq!(skipped.status_name(), "skipped");
        assert_eq!(refused.status_name(), "denied");
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
        assert_eq!(wrapped.message(), "boom");
        assert_eq!(wrapped.model_feedback(), Some("the tool failed"));
    }

    #[test]
    fn from_error_preserves_refusal_disposition() {
        let refused = ToolExecutionError::from_error(
            ToolExecutionError::refused("declined").with_code("POLICY"),
        );
        assert!(refused.is_refusal());
        assert_eq!(refused.kind(), ToolErrorKind::PermissionDenied);
        assert_eq!(refused.code(), Some("POLICY"));
    }
}
