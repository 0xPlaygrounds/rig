//! Execution-local correlation and facts from a provider's request boundary.

use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use super::{Action, Emitter, Observation, Stage, Subject, Witness};

/// Bound and scrub diagnostic text before persisting it.
///
/// Supply the credentials used by the owning request or connection. Known
/// credential values and common credential markers redact the whole message;
/// oversized messages are replaced rather than retaining a secret prefix.
pub fn scrub_diagnostic(value: &str, secrets: &[String]) -> String {
    scrub::text(value, secrets)
}

/// Collect URL userinfo and known credential query parameters for diagnostic
/// redaction, including origin-form request URIs. Returned values are secrets:
/// keep them runtime-only and never include them in diagnostics or artifacts.
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
    /// Host-clock intervals, attached to attempt closure. Never semantic data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timing: Option<AdapterTiming>,
}

/// Attempt intervals sampled from the witness's injected monotonic clock.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterTiming {
    /// Actual send to attempt closure, including decoding and stream consumption.
    /// None means the clock could not establish a nonnegative interval.
    pub request_duration: Option<std::time::Duration>,
    /// Send to the first observed nonempty response-body chunk, before decoding.
    /// None when no byte boundary was observed, including buffered unary
    /// transports without the hook. Never substitute headers or decoded frames.
    pub time_to_first_byte: Option<std::time::Duration>,
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

/// Provider-local projection, invoked only when observations are installed.
#[derive(Clone, Copy)]
pub(crate) struct PayloadObserver(pub fn(&[u8], &mut AdapterAttempt));

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
    pub(crate) fn from_completion(error: &crate::completion::CompletionError) -> Self {
        use crate::completion::CompletionError as C;
        match error {
            C::HttpError(error) => Self::from_http(error),
            C::JsonError(_) | C::ResponseError(_) => Self::Decode,
            C::UrlError(_) | C::RequestError(_) => Self::Request,
            C::ProviderError(_) | C::ProviderResponse(_) => Self::ProviderResponse,
        }
    }

    pub(crate) fn from_http(error: &crate::http_client::Error) -> Self {
        use crate::http_client::Error as H;
        match error {
            H::Protocol(_) | H::InvalidHeaderValue(_) | H::NoHeaders => Self::Request,
            H::InvalidContentType(_) => Self::Decode,
            H::StreamEnded => Self::Transport,
            H::InvalidStatusCode(_)
            | H::InvalidStatusCodeWithMessage(..)
            | H::InvalidStatusCodeWithDetails { .. } => Self::ProviderResponse,
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

    /// Attach observation context to a transport request without touching its payload.
    pub(crate) fn attach<B>(&self, request: &mut http::Request<B>, route: &'static str) {
        request.extensions_mut().insert((self.clone(), route));
    }

    pub(crate) fn slot_for_request<B>(request: &http::Request<B>) -> Option<AdapterSlot> {
        request.extensions().get::<(Self, &'static str)>()?;
        Some(AdapterSlot::default())
    }

    pub(crate) fn from_request<B>(request: &http::Request<B>) -> Option<AdapterAttempt> {
        let (context, route) = request.extensions().get::<(Self, &'static str)>()?;
        let mut attempt = context.begin(request.method(), route)?;
        attempt.payload_observer = request.extensions().get::<PayloadObserver>().copied();
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

    /// Begin an actual send with a static route template, excluding query data.
    /// Exhaustion is observed and disables further sends' correlation without
    /// changing the provider operation or reusing an attempt identity.
    pub(crate) fn begin(
        &self,
        method: &http::Method,
        route: &'static str,
    ) -> Option<AdapterAttempt> {
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
        let started = self.inner.sink.elapsed();
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
            sse_tail: crate::http_client::sse::tail::SseTail::default(),
            payload_observer: None,
            secrets: Vec::new(),
            pending_response_id: None,
            error_boundary: None,
            started,
            body_observer: ResponseBodyObserver {
                context: self.clone(),
                started,
                state: Arc::new(Mutex::new(ResponseBodyState::default())),
            },
        })
    }
}

/// The request owns this guard until completion or cancellation.
pub(crate) struct AdapterAttempt {
    context: AdapterContext,
    number: u64,
    closed: bool,
    response_seen: bool,
    sse_tail: crate::http_client::sse::tail::SseTail,
    payload_observer: Option<PayloadObserver>,
    secrets: Vec<String>,
    pending_response_id: Option<String>,
    error_boundary: Option<AdapterErrorBoundary>,
    started: Option<std::time::Duration>,
    body_observer: ResponseBodyObserver,
}

/// Optional request extension for transports that observe response-body chunks.
///
/// Call [`Self::observe`] at body arrival, before buffering or decoding. Clones
/// belong to one adapter attempt; they neither retain body bytes nor emit facts.
/// Transports without this extension need no observation work.
#[derive(Clone)]
pub struct ResponseBodyObserver {
    context: AdapterContext,
    started: Option<std::time::Duration>,
    state: Arc<Mutex<ResponseBodyState>>,
}

#[derive(Default)]
struct ResponseBodyState {
    seen: bool,
    closed: bool,
    first_byte: Option<std::time::Duration>,
}

impl ResponseBodyObserver {
    /// Sample the first nonempty body chunk once using the injected clock.
    /// Empty chunks and notifications after attempt closure are ignored.
    pub fn observe(&self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !state.closed && !state.seen {
            state.seen = true;
            state.first_byte = self
                .started
                .and_then(|start| self.context.inner.sink.elapsed()?.checked_sub(start));
        }
    }

    fn close(&self) -> Option<std::time::Duration> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.closed = true;
        state.first_byte
    }
}

impl AdapterAttempt {
    fn body_bytes(&mut self, bytes: &[u8]) {
        self.body_observer.observe(bytes);
    }

    pub(crate) fn body_observer(&self) -> Option<ResponseBodyObserver> {
        self.started.map(|_| self.body_observer.clone())
    }

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

    pub(crate) fn payload(&mut self, bytes: &[u8]) {
        if let Some(observer) = self.payload_observer {
            (observer.0)(bytes, self);
        }
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

    pub(crate) fn response(&mut self, status: http::StatusCode) {
        self.response_with_headers(status, None);
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
            let first_byte = self.body_observer.close();
            let timing = self.started.map(|start| AdapterTiming {
                request_duration: self
                    .context
                    .inner
                    .sink
                    .elapsed()
                    .and_then(|end| end.checked_sub(start)),
                time_to_first_byte: first_byte,
            });
            let response_id = self.pending_response_id.take();
            self.emit_with_analysis(
                AdapterEvent::Finished { ending },
                AdapterAnalysis {
                    response_id,
                    timing,
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

/// Shared by the SSE transport and frame driver, which own different boundaries.
#[derive(Clone, Default)]
pub(crate) struct AdapterSlot(Arc<Mutex<Option<AdapterAttempt>>>);

impl AdapterSlot {
    pub(crate) fn body_observer(&self) -> Option<ResponseBodyObserver> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .and_then(AdapterAttempt::body_observer)
    }

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

    pub(crate) fn payload(&self, bytes: &[u8]) {
        if let Some(attempt) = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_mut()
        {
            attempt.payload(bytes);
        }
    }

    pub(crate) fn start<B>(&self, request: &http::Request<B>) {
        *self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) =
            AdapterContext::from_request(request);
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

    pub(crate) fn fail(&self, error: &crate::completion::CompletionError) {
        if let Some(status) = error.provider_response_status() {
            self.response(status);
        }
        let report = crate::error::ErrorReport::from(error);
        let boundary = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .and_then(|attempt| attempt.error_boundary)
            .unwrap_or_else(|| AdapterErrorBoundary::from_completion(error));
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
            attempt.body_bytes(bytes);
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
