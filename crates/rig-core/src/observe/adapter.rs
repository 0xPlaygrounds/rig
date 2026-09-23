//! Execution-local correlation and facts from provider request boundaries.
//!
//! ```
//! use rig_core::observe::scrub_diagnostic;
//!
//! assert_eq!(scrub_diagnostic("Bearer secret", &[]), "[redacted]");
//! ```

use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use super::{Action, Emitter, Observation, Stage, Subject, Witness};

/// Bounds diagnostic text and removes control characters. Known credentials
/// or credential markers redact the whole message; oversized messages are
/// replaced rather than truncated to a potentially sensitive prefix.
pub fn scrub_diagnostic(value: &str, secrets: &[String]) -> String {
    scrub::text(value, secrets)
}

/// The secrets a URL carries for diagnostic redaction: userinfo and known
/// credential query parameters, origin-form request URIs included. The
/// returned values are secrets: keep them runtime-only and never write
/// them into a diagnostic or an artifact.
pub fn diagnostic_url_secrets(url: &str) -> Vec<String> {
    scrub::url_secrets(url)
}

/// One fact about a provider operation, independently of its effect record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterObservation {
    /// Caller-owned logical operation identity within this execution.
    pub operation: String,
    /// One-based HTTP send ordinal; absent only when correlation is exhausted.
    pub attempt: Option<u64>,
    /// One-based host dispatch attempt, when explicitly supplied by the host.
    /// Independent of the HTTP send ordinal: a dispatch may send more than once.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_attempt: Option<std::num::NonZeroU64>,
    /// What the owning boundary observed.
    pub event: AdapterEvent,
    /// Scrubbed volatile diagnostics, excluded from semantic comparison.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub analysis: Option<AdapterAnalysis>,
}

/// Metadata from the provider boundary. Request/response bodies are not included.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum AdapterEvent {
    /// Fields actually present in a provider payload; absent fields are not updates.
    Provider {
        /// The provider's verdict and selected model, independent of transport closure.
        verdict: AdapterVerdict,
    },
    /// A provider error envelope, including envelopes carried under HTTP 200.
    ErrorEnvelope {
        /// Scrubbed original envelope fields, not an inferred HTTP status.
        error: AdapterErrorEnvelope,
    },
    /// Provider-reported usage for this attempt, including rejected responses.
    Usage {
        /// A cumulative snapshot: replace earlier snapshots, never sum them.
        usage: AdapterUsage,
    },
    /// A request is about to be sent.
    Started {
        /// HTTP method.
        method: String,
        /// Provider-declared route template, never a credential-bearing URI.
        route: String,
    },
    /// The transport returned response headers.
    Response {
        /// Actual HTTP response status, including successful responses.
        status: u16,
    },
    /// The body reached EOF, independently of any provider terminal verdict.
    TransportEof {
        /// Number of complete frames, excluding provider-recognized analysis-only frames.
        after: usize,
        /// Undelimited SSE bytes remaining at EOF; zero means no partial frame.
        partial_bytes: usize,
    },
    /// The owning request boundary closed this attempt.
    Finished {
        /// Whether the attempt completed, failed, or was dropped.
        ending: AdapterEnding,
    },
    /// A complete transport frame could not be decoded; the driver may continue.
    Corrupt {
        /// One-based ordinal counting known, unknown and corrupt frames,
        /// excluding provider-recognized analysis-only frames.
        frame: usize,
    },
    /// The context cannot assign another unique send ordinal.
    IdentityExhausted,
}

/// Sparse provider metadata: a present field supersedes its previous value.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterVerdict {
    /// Provider finish code, including unknown codes after scrubbing.
    pub finish_reason: Option<String>,
    /// Provider prompt-block/refusal reason, when reported.
    pub block_reason: Option<String>,
    /// Provider detail accompanying a finish/block reason.
    pub detail: Option<String>,
    /// Actual provider model/version, not the requested alias.
    pub model: Option<String>,
}

/// Analysis-only response metadata. Never use identifiers as operation join keys.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterAnalysis {
    /// Provider response identifier, scrubbed and bounded.
    pub response_id: Option<String>,
    /// Allowlisted response headers. None means the transport supplied no map;
    /// an empty map means it supplied no allowlisted values.
    pub headers: Option<std::collections::BTreeMap<String, String>>,
}

/// Original provider error-envelope fields, separate from Rig retry classification.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterErrorEnvelope {
    /// Provider code in its scalar wire spelling; unknown codes stay observable.
    pub code: Option<String>,
    /// Provider status name, when present.
    pub status: Option<String>,
    /// Bounded, scrubbed provider message, when present.
    pub message: Option<String>,
}

/// Error-envelope projection with optional code, message, and `type`/`status`.
/// Codes remain JSON values until emission validates their scalar shape.
#[derive(Default, Deserialize)]
pub struct ObservedError {
    pub code: Option<serde_json::Value>,
    #[serde(rename = "type", alias = "status")]
    pub kind: Option<String>,
    pub message: Option<String>,
}

impl ObservedError {
    /// Emit this envelope as the attempt's error-envelope fact, scrubbed.
    pub fn emit(self, sink: &mut dyn crate::wire::ObservationSink) {
        let code = self.code.map(|code| match code {
            serde_json::Value::String(code) => sink.scrub(&code),
            serde_json::Value::Number(code) => code.to_string(),
            _ => "[invalid]".to_owned(),
        });
        sink.emit(AdapterEvent::ErrorEnvelope {
            error: AdapterErrorEnvelope {
                code,
                status: self.kind.map(|value| sink.scrub(&value)),
                message: self.message.map(|value| sink.scrub(&value)),
            },
        });
    }
}

/// Deserialize one [`AdapterUsage`] counter a provider may send as a number,
/// a string or `null`: anything that is not a `u64` stays unknown rather
/// than failing the payload that carried it.
pub(crate) fn lenient_count<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<u64>, D::Error> {
    Ok(serde_json::Value::deserialize(deserializer)?.as_u64())
}

/// A provider's cumulative usage snapshot for one HTTP attempt.
///
/// Missing, invalid or negative counts remain unknown. A present zero is a
/// reported zero. These fields may overlap (for example cached tokens are
/// included in input tokens); never sum fields to invent a total. A later
/// snapshot replaces the earlier snapshot, including its unknown fields.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterUsage {
    /// Input tokens, including cached input where the provider includes it.
    pub input_tokens: Option<u64>,
    /// Output candidate tokens, as reported by the provider.
    pub output_tokens: Option<u64>,
    /// Provider-reported total, not a sum of the other fields.
    pub total_tokens: Option<u64>,
    /// Cached input tokens, when reported.
    pub cached_input_tokens: Option<u64>,
    /// Reasoning tokens, when reported separately.
    pub reasoning_tokens: Option<u64>,
    /// Provider tool-use input tokens, when reported separately.
    pub tool_input_tokens: Option<u64>,
}

/// Boundary identified by the adapter's original typed error, not its message.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdapterErrorBoundary {
    /// Request construction or validation failed.
    Request,
    /// The provider returned a failure response or error envelope.
    ProviderResponse,
    /// Response decoding or validation failed.
    Decode,
    /// A typed transport termination was reported.
    Transport,
    /// An erased client error does not establish a more specific boundary.
    #[default]
    Unknown,
}

impl AdapterErrorBoundary {
    pub(crate) fn from_http(error: &crate::http_client::Error) -> Self {
        use crate::http_client::Error as H;
        match error {
            H::Protocol(_) | H::InvalidHeaderValue(_) | H::NoHeaders => Self::Request,
            H::InvalidContentType(_) => Self::Decode,
            H::StreamEnded => Self::Transport,
            H::InvalidStatusCodeWithDetails { .. } => Self::ProviderResponse,
            H::Instance(_) => Self::Unknown,
        }
    }
}

/// How a provider attempt closed, distinct from the run's ending.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "ending", rename_all = "snake_case")]
pub enum AdapterEnding {
    /// The request produced a decoded response.
    Decoded,
    /// The request failed; provider payloads remain in the existing error report.
    Error {
        /// Known boundary before error conversion erases transport subtypes.
        #[serde(default)]
        boundary: AdapterErrorBoundary,
        /// Stable Rig error classification.
        kind: String,
        /// HTTP status reported by the failure, when available.
        status: Option<u16>,
        /// Retryability under the existing policy, not a decision to retry.
        retryable: bool,
    },
    /// A provider terminal record was decoded from the stream.
    Terminal,
    /// Transport EOF without a provider terminal record.
    Eof {
        /// Number of complete frames, excluding provider-recognized analysis-only frames.
        after: usize,
    },
    /// EOF left an undelimited SSE event; bytes are never retained in the fact.
    PartialFrame {
        /// Raw bytes since the last blank event delimiter, saturated at usize::MAX.
        byte_count: usize,
        /// Complete frames previously delivered, excluding analysis-only frames.
        after: usize,
    },
    /// The future or stream was dropped before its boundary closed.
    Dropped,
}

/// A caller-supplied witness and logical operation identity.
///
/// Clone this context for retries of the same operation. Create a new context
/// with a distinct identity for a different call, including identical parallel
/// requests. Identity is execution-local; it does not assert replay equivalence.
/// This handle is runtime-only and must never be serialized into provider data.
#[derive(Clone)]
pub struct AdapterContext {
    inner: Arc<AdapterContextInner>,
}

struct AdapterContextInner {
    sink: Arc<dyn Witness + Send + Sync>,
    subject: Subject,
    operation: String,
    next: Arc<Mutex<Option<u64>>>,
    host_attempt: Option<std::num::NonZeroU64>,
}

impl std::fmt::Debug for AdapterContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Caller-supplied identity can contain sensitive data; do not log it.
        f.debug_struct("AdapterContext").finish_non_exhaustive()
    }
}

impl AdapterContext {
    /// Caller-owned identity of this logical operation within the execution.
    pub fn operation(&self) -> &str {
        &self.inner.operation
    }

    /// Bind a logical operation to its witness and existing subject.
    /// The caller must use a non-sensitive, execution-unique operation identity.
    pub fn new(
        sink: Arc<dyn Witness + Send + Sync>,
        subject: Subject,
        operation: impl Into<String>,
    ) -> Self {
        Self {
            inner: Arc::new(AdapterContextInner {
                sink,
                subject,
                operation: operation.into(),
                next: Arc::new(Mutex::new(Some(1))),
                host_attempt: None,
            }),
        }
    }

    /// Bind an explicit host attempt to its current dispatch subject.
    ///
    /// The logical operation, witness and HTTP send counter remain shared.
    /// Existing clones retain their own subject and host ordinal, so an older
    /// in-flight attempt cannot be relabeled by a concurrent retry. The host
    /// owns ordinal allocation and must start a new context for a changed
    /// logical request rather than reusing this operation identity.
    pub fn for_host_attempt(&self, subject: Subject, attempt: std::num::NonZeroU64) -> Self {
        Self {
            inner: Arc::new(AdapterContextInner {
                sink: self.inner.sink.clone(),
                subject,
                operation: self.inner.operation.clone(),
                next: self.inner.next.clone(),
                host_attempt: Some(attempt),
            }),
        }
    }

    /// Begins a send and captures request credentials for diagnostic scrubbing.
    /// `route` must be a credential-free provider template without base-URL
    /// prefixes or query data. Returns `None` when attempt IDs are exhausted.
    pub(crate) fn attempt_for<B>(
        &self,
        request: &http::Request<B>,
        route: &str,
    ) -> Option<AdapterAttempt> {
        let mut attempt = self.begin(request.method(), route)?;
        attempt.secrets = scrub::request_secrets(request);
        Some(attempt)
    }

    fn emit(&self, attempt: Option<u64>, event: AdapterEvent) {
        self.emit_with_analysis(attempt, event, None);
    }

    fn emit_with_analysis(
        &self,
        attempt: Option<u64>,
        event: AdapterEvent,
        analysis: Option<AdapterAnalysis>,
    ) {
        self.inner.sink.observe(Observation::new(
            self.inner.subject.clone(),
            Stage::Handler,
            Emitter::named("rig-core/adapter"),
            Action::Adapter {
                observation: AdapterObservation {
                    operation: self.inner.operation.clone(),
                    attempt,
                    host_attempt: self.inner.host_attempt,
                    event,
                    analysis,
                },
            },
        ));
    }

    /// Allocates a send ordinal and emits its start under a credential-free
    /// route template. Emits exhaustion at the last ordinal; later calls return
    /// `None` without reusing identities or affecting provider execution.
    pub(crate) fn begin(&self, method: &http::Method, route: &str) -> Option<AdapterAttempt> {
        let number = {
            let mut next = self
                .inner
                .next
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let number = (*next)?;
            *next = number.checked_add(1);
            number
        };
        self.emit(
            Some(number),
            AdapterEvent::Started {
                method: method.to_string(),
                route: route.to_owned(),
            },
        );
        if number == u64::MAX {
            self.emit(None, AdapterEvent::IdentityExhausted);
        }
        Some(AdapterAttempt {
            context: self.clone(),
            number,
            closed: false,
            response_seen: false,
            sse_tail: crate::http_client::tail::SseTail::default(),
            secrets: Vec::new(),
            pending_response_id: None,
            error_boundary: None,
        })
    }
}

/// The request owns this guard until completion or cancellation.
pub(crate) struct AdapterAttempt {
    context: AdapterContext,
    number: u64,
    closed: bool,
    response_seen: bool,
    sse_tail: crate::http_client::tail::SseTail,
    secrets: Vec<String>,
    pending_response_id: Option<String>,
    error_boundary: Option<AdapterErrorBoundary>,
}

impl AdapterAttempt {
    pub(crate) fn text(&self, text: &str) -> String {
        scrub::text(text, &self.secrets)
    }

    pub(crate) fn emit_with_analysis(&self, event: AdapterEvent, analysis: AdapterAnalysis) {
        let analysis = (analysis != AdapterAnalysis::default()).then_some(analysis);
        self.context
            .emit_with_analysis(Some(self.number), event, analysis);
    }

    pub(crate) fn emit(&self, event: AdapterEvent) {
        self.context.emit(Some(self.number), event);
    }

    /// Project a payload's facts through the decoder that understands it.
    pub(crate) fn project(&mut self, project: impl FnOnce(&mut dyn crate::wire::ObservationSink)) {
        project(self);
    }

    pub(crate) fn provider(&mut self, verdict: AdapterVerdict, response_id: Option<String>) {
        if response_id.is_some() {
            self.pending_response_id = response_id;
        }
        // An ID-only payload must not manufacture a semantic provider event.
        // Retain at most one ID until the next verdict or the attempt closure.
        if verdict != AdapterVerdict::default() {
            let response_id = self.pending_response_id.take();
            self.emit_with_analysis(
                AdapterEvent::Provider { verdict },
                AdapterAnalysis {
                    response_id,
                    ..AdapterAnalysis::default()
                },
            );
        }
    }

    pub(crate) fn response_with_headers(
        &mut self,
        status: http::StatusCode,
        headers: Option<&http::HeaderMap>,
    ) {
        if !self.response_seen {
            self.response_seen = true;
            self.emit_with_analysis(
                AdapterEvent::Response {
                    status: status.as_u16(),
                },
                AdapterAnalysis {
                    headers: headers.map(|h| scrub::headers(h, &self.secrets)),
                    ..AdapterAnalysis::default()
                },
            );
        }
    }

    pub(crate) fn finish(&mut self, ending: AdapterEnding) {
        if !self.closed {
            self.closed = true;
            let response_id = self.pending_response_id.take();
            self.emit_with_analysis(
                AdapterEvent::Finished { ending },
                AdapterAnalysis {
                    response_id,
                    ..AdapterAnalysis::default()
                },
            );
        }
    }
}

impl Drop for AdapterAttempt {
    fn drop(&mut self) {
        self.finish(AdapterEnding::Dropped);
    }
}

/// The projection surface a decoder writes its observation facts through.
impl crate::wire::ObservationSink for AdapterAttempt {
    fn emit(&mut self, event: AdapterEvent) {
        AdapterAttempt::emit(self, event);
    }

    fn provider(&mut self, verdict: AdapterVerdict, response_id: Option<String>) {
        AdapterAttempt::provider(self, verdict, response_id);
    }

    fn scrub(&self, value: &str) -> String {
        self.text(value)
    }
}

/// Shared by the SSE transport and frame driver, which own different boundaries.
#[derive(Clone, Default)]
pub(crate) struct AdapterSlot(Arc<Mutex<Option<AdapterAttempt>>>);

impl AdapterSlot {
    pub(crate) fn transport_eof(&self, after: usize) {
        if let Some(attempt) = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
        {
            attempt.emit(AdapterEvent::TransportEof {
                after,
                partial_bytes: attempt.sse_tail.pending(),
            });
        }
    }

    /// Project a reply payload's facts through the decoder that understands
    /// it.
    pub(crate) fn project(&self, project: impl FnOnce(&mut dyn crate::wire::ObservationSink)) {
        if let Some(attempt) = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_mut()
        {
            attempt.project(project);
        }
    }

    /// Install the attempt this send's facts belong to.
    pub(crate) fn install(&self, attempt: Option<AdapterAttempt>) {
        *self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = attempt;
    }

    pub(crate) fn response(&self, status: http::StatusCode) {
        self.response_with_headers(status, None);
    }

    pub(crate) fn response_with_headers(
        &self,
        status: http::StatusCode,
        headers: Option<&http::HeaderMap>,
    ) {
        if let Some(attempt) = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_mut()
        {
            attempt.response_with_headers(status, headers);
        }
    }

    pub(crate) fn finish(&self, ending: AdapterEnding) {
        if let Some(attempt) = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_mut()
        {
            attempt.finish(ending);
        }
    }

    /// Preserve information before the SSE transport converts the owned error.
    pub(crate) fn error_boundary(&self, boundary: AdapterErrorBoundary) {
        if let Some(attempt) = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_mut()
        {
            attempt.error_boundary = Some(boundary);
        }
    }

    pub(crate) fn fail(&self, error: &crate::error::ProviderError) {
        if let Some(status) = error.provider_response_status() {
            self.response(status);
        }
        let report = error.report();
        let boundary = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .and_then(|attempt| attempt.error_boundary)
            .unwrap_or_else(|| error.boundary());
        self.finish(AdapterEnding::Error {
            boundary,
            kind: report.kind.code().to_owned(),
            status: report.http_status,
            retryable: report.is_retryable(),
        });
    }

    pub(crate) fn bytes(&self, bytes: &[u8]) {
        if let Some(attempt) = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_mut()
        {
            attempt.sse_tail.feed(bytes);
        }
    }

    pub(crate) fn eof(&self, after: usize) {
        if let Some(attempt) = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_mut()
        {
            let byte_count = attempt.sse_tail.pending();
            let ending = if byte_count == 0 {
                AdapterEnding::Eof { after }
            } else {
                AdapterEnding::PartialFrame { byte_count, after }
            };
            attempt.finish(ending);
        }
    }

    pub(crate) fn corrupt(&self, frame: usize) {
        if let Some(attempt) = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
        {
            attempt
                .context
                .emit(Some(attempt.number), AdapterEvent::Corrupt { frame });
        }
    }
}

mod scrub;
#[cfg(test)]
mod tests;
