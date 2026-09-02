//! Canonical structured tool errors and execution results.

use std::error::Error;
#[cfg(not(target_family = "wasm"))]
use std::sync::Arc;

use crate::{
    tool::ToolOutput,
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

/// Normalized classification for a tool execution error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
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

// One row per kind: (stable name, default retryability, default model
// feedback). The macro emits the three parallel accessors from the single
// table so a new kind cannot update one and miss another.
macro_rules! kind_defaults {
    ($($variant:ident => ($name:literal, $retryable:expr, $feedback:literal)),+ $(,)?) => {
        impl ToolErrorKind {
            /// Stable machine-readable name.
            pub const fn as_str(self) -> &'static str {
                match self { $(Self::$variant => $name,)+ }
            }

            const fn default_retryable(self) -> Option<bool> {
                match self { $(Self::$variant => $retryable,)+ }
            }

            const fn default_model_feedback(self) -> &'static str {
                match self { $(Self::$variant => $feedback,)+ }
            }
        }
    };
}

kind_defaults! {
    InvalidArgs => ("invalid_args", Some(false), "tool arguments were invalid"),
    Timeout => ("timeout", Some(true), "tool execution timed out"),
    Cancelled => ("cancelled", Some(false), "tool execution was cancelled"),
    NotFound => ("not_found", Some(false), "the requested tool or resource was not found"),
    PermissionDenied => ("permission_denied", Some(false), "the tool denied the request"),
    RateLimited => ("rate_limited", Some(true), "the tool was rate limited; try again later"),
    Provider => ("provider", None, "the tool provider failed"),
    Network => ("network", Some(true), "the tool could not reach its upstream service"),
    Other => ("other", None, "the tool failed"),
}

// One `ToolExecutionError` constructor per kind, from a single table so a new
// kind cannot miss its shorthand. `refused` stays hand-written because it also
// sets the refusal disposition.
macro_rules! kind_ctors {
    ($($(#[$doc:meta])* $ctor:ident => $variant:ident),+ $(,)?) => {
        impl ToolExecutionError {
            $($(#[$doc])*
            pub fn $ctor(message: impl Into<String>) -> Self {
                Self::new(ToolErrorKind::$variant, message)
            })+
        }
    };
}

kind_ctors! {
    /// Invalid arguments.
    invalid_args => InvalidArgs,
    /// Timeout.
    timeout => Timeout,
    /// Cancellation.
    cancelled => Cancelled,
    /// Missing tool or resource.
    not_found => NotFound,
    /// An authorization or permission failure.
    ///
    /// This is an ordinary execution error. Use [`Self::refused`] when the tool
    /// intentionally declines the operation so hooks and telemetry can preserve
    /// the refusal as a distinct disposition.
    permission_denied => PermissionDenied,
    /// Rate limit.
    rate_limited => RateLimited,
    /// Upstream provider failure.
    provider => Provider,
    /// Network failure.
    network => Network,
    /// Catch-all failure.
    other => Other,
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
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(from = "ToolExecutionErrorRepr", into = "ToolExecutionErrorRepr")]
pub struct ToolExecutionError {
    kind: ToolErrorKind,
    message: String,
    model_output: Box<ToolOutput>,
    retryable: Option<bool>,
    code: Option<String>,
    http_status: Option<u16>,
    refusal: bool,
    /// The concrete source, kept for downcasting on native. Browser wasm
    /// retains none: its errors are not `Send + Sync`, and this type must be
    /// (it crosses the effect bus and the wire as part of an `Outcome`).
    #[cfg(not(target_family = "wasm"))]
    source: Option<Arc<dyn Error + Send + Sync + 'static>>,
}

/// The wire form of [`ToolExecutionError`]: every field but `source`. A
/// source is live process state (a boxed error); it does not cross a wire
/// and a report reconstructed from one has none.
#[derive(serde::Serialize, serde::Deserialize)]
struct ToolExecutionErrorRepr {
    kind: ToolErrorKind,
    message: String,
    model_output: ToolOutput,
    retryable: Option<bool>,
    code: Option<String>,
    http_status: Option<u16>,
    refusal: bool,
}

impl From<ToolExecutionError> for ToolExecutionErrorRepr {
    fn from(error: ToolExecutionError) -> Self {
        Self {
            kind: error.kind,
            message: error.message,
            model_output: *error.model_output,
            retryable: error.retryable,
            code: error.code,
            http_status: error.http_status,
            refusal: error.refusal,
        }
    }
}

impl From<ToolExecutionErrorRepr> for ToolExecutionError {
    fn from(repr: ToolExecutionErrorRepr) -> Self {
        Self {
            kind: repr.kind,
            message: repr.message,
            model_output: Box::new(repr.model_output),
            retryable: repr.retryable,
            code: repr.code,
            http_status: repr.http_status,
            refusal: repr.refusal,
            #[cfg(not(target_family = "wasm"))]
            source: None,
        }
    }
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
            #[cfg(not(target_family = "wasm"))]
            source: None,
        }
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
                Err(source) => Self::other(source.to_string()).redact_model_feedback(),
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
    ///
    /// Browser wasm keeps no source (see the field): the call is accepted
    /// and the source dropped, so tool code stays target-independent.
    #[cfg_attr(target_family = "wasm", allow(unused_mut))]
    pub fn with_source<E>(mut self, source: E) -> Self
    where
        E: Error + WasmCompatSend + WasmCompatSync + 'static,
    {
        #[cfg(not(target_family = "wasm"))]
        {
            self.source = Some(Arc::new(source));
        }
        #[cfg(target_family = "wasm")]
        {
            let _ = source;
        }
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
        #[cfg(not(target_family = "wasm"))]
        {
            self.source.as_ref()?.downcast_ref::<E>()
        }
        #[cfg(target_family = "wasm")]
        {
            None
        }
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
            .field("retryable", &self.retryable)
            .field("code", &self.code)
            .field("http_status", &self.http_status)
            .field("refusal", &self.refusal)
            .field("model_output", &"<redacted>")
            .field("source_configured", &self.has_source())
            .finish()
    }
}

impl ToolExecutionError {
    fn has_source(&self) -> bool {
        #[cfg(not(target_family = "wasm"))]
        {
            self.source.is_some()
        }
        #[cfg(target_family = "wasm")]
        {
            false
        }
    }
}

impl Error for ToolExecutionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        #[cfg(not(target_family = "wasm"))]
        {
            self.source
                .as_deref()
                .map(|source| source as &(dyn Error + 'static))
        }
        #[cfg(target_family = "wasm")]
        {
            None
        }
    }
}

/// Private mutually exclusive state behind [`ToolResult`].
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "status", content = "value", rename_all = "snake_case")]
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
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct ToolResult {
    disposition: ToolDisposition,
}

impl std::fmt::Debug for ToolResult {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let error = match &self.disposition {
            ToolDisposition::Error(error) | ToolDisposition::Refused(error) => Some(error),
            ToolDisposition::Success(_) | ToolDisposition::Skipped(_) => None,
        };
        formatter
            .debug_struct("ToolResult")
            .field("status", &self.status_name())
            .field("error_kind", &error.map(ToolExecutionError::kind))
            .field("retryable", &error.and_then(ToolExecutionError::retryable))
            .field("code", &error.and_then(ToolExecutionError::code))
            .field(
                "http_status",
                &error.and_then(ToolExecutionError::http_status),
            )
            .finish()
    }
}

impl ToolResult {
    /// Creates a successful canonical tool result.
    pub fn success(output: ToolOutput) -> Self {
        Self {
            disposition: ToolDisposition::Success(output),
        }
    }

    /// Creates a failed or refused canonical tool result.
    pub fn failed(error: ToolExecutionError) -> Self {
        let disposition = if error.is_refusal() {
            ToolDisposition::Refused(error)
        } else {
            ToolDisposition::Error(error)
        };
        Self { disposition }
    }

    /// Creates a result for a call skipped by runtime policy.
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

    /// Returns the stable telemetry name for this result disposition.
    /// The result as the tool author's `Result`: a success or a skip is
    /// `Ok(output)`, an error or a refusal is `Err(error)`.
    pub fn into_result(self) -> Result<ToolOutput, ToolExecutionError> {
        match self.disposition {
            ToolDisposition::Success(output) | ToolDisposition::Skipped(output) => Ok(output),
            ToolDisposition::Error(error) | ToolDisposition::Refused(error) => Err(error),
        }
    }

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
mod tests;

#[cfg(test)]
mod migrated_tests;
