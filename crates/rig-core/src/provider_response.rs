//! Preserved provider error responses and their machine codes.
//!
//! ```
//! use rig_core::ProviderResponseError;
//!
//! let reply = ProviderResponseError::new(http::StatusCode::TOO_MANY_REQUESTS, "slow down");
//! assert!(reply.is_retryable());
//! ```
use http::StatusCode;

/// A raw provider error body with captured transport metadata.
/// Callers must supply the provider's actual payload, not a generated diagnostic.
/// Serialization omits headers and route; deserialization restores them as `None`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderResponseError {
    /// HTTP status of the provider response, when it was captured alongside the body.
    pub status: Option<StatusCode>,
    /// Raw response body as returned by the provider.
    pub body: String,
    /// Transport request ID from response headers or SDK metadata, when captured.
    pub provider_request_id: Option<String>,
    /// Captured response headers, including rate-limit metadata. `None` means
    /// not captured, rather than an empty header set. Omitted during serialization.
    pub headers: Option<http::HeaderMap>,
    /// The provider's own machine-readable code for the failure, when the
    /// transport reported one apart from the body: a gRPC status code name
    /// (`UNAVAILABLE`), an AWS exception type (`ThrottlingException`).
    /// `None` when the reply carried only a status and a body.
    pub code: Option<String>,
    /// Transport retry hint, used when neither a non-success status nor a
    /// known provider code decides; refusals are never retryable.
    pub transient: Option<bool>,
    /// Whether this error represents a content refusal. Refusals are never
    /// retryable; refusal content in a successful model answer is separate.
    pub refusal: bool,
    /// The provider and request path the reply answered, when the operation
    /// records them for diagnostics (model listing does). Omitted during
    /// serialization.
    pub route: Option<String>,
}

impl ProviderResponseError {
    /// Preserve a provider error response captured with its HTTP status.
    pub fn new(status: StatusCode, body: impl Into<String>) -> Self {
        let status = Some(status);
        Self {
            status,
            ..Self::without_status(body)
        }
    }

    /// Preserve a provider error body that has no HTTP status (gRPC / SDK
    /// transports).
    pub fn without_status(body: impl Into<String>) -> Self {
        Self {
            status: None,
            body: body.into(),
            provider_request_id: None,
            headers: None,
            code: None,
            transient: None,
            refusal: false,
            route: None,
        }
    }

    /// Preserve a provider error body delivered with no HTTP status of its
    /// own: an error event inside a stream, or a non-HTTP transport's reply.
    /// An error envelope that nests an HTTP status as its numeric code
    /// (`{"error":{"code":503}}`) records it as the status, as the same
    /// envelope on a non-success response would carry it.
    pub fn from_body(body: impl Into<String>) -> Self {
        let mut reply = Self::without_status(body);
        let status = reply.located(|envelope| envelope.http_status().filter(|_| envelope.nested));
        reply.status = status.and_then(|status| StatusCode::from_u16(status).ok());
        reply
    }

    /// Mark the reply as the provider's verdict on the content: a refusal,
    /// final, never retried.
    pub fn with_refusal(mut self, refusal: bool) -> Self {
        self.refusal = refusal;
        self
    }

    /// Attach the HTTP status a transport reported beside a reply that was
    /// first preserved without one (an SDK that hands back the raw HTTP
    /// response next to its typed exception). A status already set is kept.
    pub fn with_status(mut self, status: Option<StatusCode>) -> Self {
        if self.status.is_none() {
            self.status = status;
        }
        self
    }

    /// Attach the provider's own machine-readable code for the failure.
    pub fn with_code(mut self, code: Option<String>) -> Self {
        self.code = code.filter(|code| !code.is_empty());
        self
    }

    /// Replaces the transport retry hint. A refusal, a non-success HTTP
    /// status, and a provider code the table knows each outrank it.
    pub fn with_transient(mut self, transient: Option<bool>) -> Self {
        self.transient = transient;
        self
    }

    /// Whether the same request may reasonably be sent again, decided in
    /// this order: a refusal never is; a non-success HTTP status decides
    /// through [`crate::error::retryable_status`]; then a provider code the
    /// one table knows (the explicit [`Self::code`], else the code the body's
    /// error envelope names); then the `transient` hint; otherwise not.
    ///
    /// The body is read on every call, so the same envelope gets the same
    /// verdict from every decoder and transport, streamed or unary.
    pub fn is_retryable(&self) -> bool {
        if self.refusal {
            return false;
        }
        if let Some(status) = self.status.filter(|status| !status.is_success()) {
            return crate::error::retryable_status(Some(status.as_u16()));
        }
        let known = match &self.code {
            Some(code) => Envelope::default().named(code).verdict(),
            None => self.located(|envelope| envelope.verdict()),
        };
        known.or(self.transient).unwrap_or(false)
    }

    /// Returns the explicit code, else the code the body's error envelope
    /// names: its first nonempty string `code`, `status` or `type`, or else
    /// its numeric code in decimal (`"429"`).
    pub fn machine_code(&self) -> Option<String> {
        self.code
            .clone()
            .or_else(|| self.located(|envelope| envelope.code()))
    }

    /// The error the body's envelope holds, as `{"error": ...}`, or the
    /// error event itself; `None` when the body carries no envelope.
    pub fn envelope_json(&self) -> Option<serde_json::Value> {
        let doc: serde_json::Value = serde_json::from_str(&self.body).ok()?;
        let envelope = envelope(&doc)?;
        let error = envelope.error?.clone();
        Some(match envelope.nested {
            true => serde_json::json!({ "error": error }),
            false => error,
        })
    }

    /// `f` applied to the error the body carries, as [`delivered`] finds it.
    fn located<T>(&self, f: impl FnOnce(Envelope<'_>) -> Option<T>) -> Option<T> {
        let doc: serde_json::Value = serde_json::from_str(&self.body).ok()?;
        delivered(&doc).and_then(f)
    }

    /// Attach the transport request id the failed response reported.
    pub fn with_provider_request_id(mut self, request_id: Option<String>) -> Self {
        self.provider_request_id = request_id.filter(|id| !id.is_empty());
        self
    }

    /// Replaces captured response headers, including rate-limit metadata.
    pub fn with_headers(mut self, headers: Option<http::HeaderMap>) -> Self {
        self.headers = headers;
        self
    }
}

impl std::fmt::Display for ProviderResponseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.status {
            Some(status) => write!(f, "status {status}: {}", self.body)?,
            None => write!(f, "{}", self.body)?,
        }
        // The id support asks for belongs in the message a caller logs.
        if let Some(request_id) = &self.provider_request_id {
            write!(f, " (request id: {request_id})")?;
        }
        if let Some(route) = &self.route {
            write!(f, " [{route}]")?;
        }
        Ok(())
    }
}

impl std::error::Error for ProviderResponseError {}

/// Serialized error metadata with numeric HTTP status and no headers.
/// Volatile transport headers are excluded to keep replay records stable.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderResponseErrorWire {
    status: Option<u16>,
    body: String,
    provider_request_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    transient: Option<bool>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    refusal: bool,
}

impl serde::Serialize for ProviderResponseError {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        ProviderResponseErrorWire {
            status: self.status.map(|status| status.as_u16()),
            body: self.body.clone(),
            provider_request_id: self.provider_request_id.clone(),
            code: self.code.clone(),
            transient: self.transient,
            refusal: self.refusal,
        }
        .serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for ProviderResponseError {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error as _;
        let wire = ProviderResponseErrorWire::deserialize(deserializer)?;
        let status = wire
            .status
            .map(StatusCode::from_u16)
            .transpose()
            .map_err(D::Error::custom)?;
        Ok(Self {
            status,
            body: wire.body,
            provider_request_id: wire.provider_request_id,
            headers: None,
            code: wire.code,
            transient: wire.transient,
            refusal: wire.refusal,
            route: None,
        })
    }
}

/// Where a provider put its error, and the code it gave it: the first
/// nonempty string and the first integer among its code fields.
#[derive(Default)]
struct Envelope<'a> {
    /// The error value itself: an object, a message string, or the event.
    error: Option<&'a serde_json::Value>,
    name: Option<&'a str>,
    number: Option<i64>,
    /// The error nests in the document (`error`, `response.error`) rather
    /// than being the document itself, an `error` event.
    nested: bool,
}

/// The error `doc` carries, where [`envelope`] locates it, and whether it
/// nests in the document rather than being the document itself.
pub(crate) fn located_error(doc: &serde_json::Value) -> Option<(&serde_json::Value, bool)> {
    envelope(doc).and_then(|envelope| Some((envelope.error?, envelope.nested)))
}

/// Codes naming a transient condition, in any case, from every transport:
/// HTTP envelopes, gRPC status names, AWS exception types.
const TRANSIENT: &[&str] = &[
    "overloaded",
    "overloaded_error",
    "api_error",
    "rate_limit_error",
    "rate_limit_exceeded",
    "too_many_requests",
    "server_error",
    "server_is_overloaded",
    "slow_down",
    "unavailable",
    "resource_exhausted",
    "deadline_exceeded",
    "aborted",
    "ThrottlingException",
    "ServiceUnavailableException",
    "InternalServerException",
    "ModelNotReadyException",
    "ModelTimeoutException",
    "ModelStreamErrorException",
];

/// Codes naming a condition the same request meets again.
const FINAL: &[&str] = &[
    "insufficient_quota",
    "quota_exceeded",
    "billing_hard_limit_reached",
    "context_length_exceeded",
    "invalid_prompt",
];

/// The one reader of where a provider puts its error. In order: the
/// Responses `response.failed` nesting (`response.error`); a top-level
/// `error` object, or a nonempty `error` string (a message, no code); a
/// document that is itself an `error` event, whose code is its `code`
/// alone, since its `type` is the event tag. `null`, `{}` and `""` are no
/// envelope. Whether a frame is an error at all stays its decoder's call.
fn envelope(doc: &serde_json::Value) -> Option<Envelope<'_>> {
    use serde_json::Value;
    const FIELDS: &[&str] = &["code", "status", "type"];
    let object = |value: &&Value| value.as_object().is_some_and(|fields| !fields.is_empty());
    if let Some(error) = doc.pointer("/response/error").filter(object) {
        return Some(Envelope::of(error, FIELDS, true));
    }
    match doc.get("error") {
        Some(error) if object(&error) => Some(Envelope::of(error, FIELDS, true)),
        Some(error @ Value::String(message)) if !message.is_empty() => {
            Some(Envelope::of(error, &[], true))
        }
        _ => (doc.get("type").and_then(Value::as_str) == Some("error"))
            .then(|| Envelope::of(doc, &["code"], false)),
    }
}

/// The error a body already known to be one carries, for its verdict and
/// code: where [`envelope`] locates it; in a JSON array (Gemini's non-SSE
/// stream), the first element that carries one; else the document read as
/// a bare error object, as a decoder that handed over the inner error
/// (`{"type":"overloaded_error","message":".."}`) leaves it. A bare error
/// has a nonempty string `message` and no `error` or `response` of its own,
/// and its code is its `code`, `status` or `type`.
fn delivered(doc: &serde_json::Value) -> Option<Envelope<'_>> {
    if let Some(items) = doc.as_array() {
        return items.iter().find_map(delivered);
    }
    let message = doc.get("message").and_then(serde_json::Value::as_str);
    let bare = message.is_some_and(|message| !message.is_empty())
        && doc.get("error").is_none()
        && doc.get("response").is_none();
    envelope(doc).or_else(|| bare.then(|| Envelope::of(doc, &["code", "status", "type"], false)))
}

impl<'a> Envelope<'a> {
    fn named(mut self, name: &'a str) -> Self {
        self.name = Some(name);
        self
    }

    fn of(error: &'a serde_json::Value, fields: &[&str], nested: bool) -> Self {
        let values = || fields.iter().filter_map(|field| error.get(field));
        Self {
            error: Some(error),
            name: values().find_map(|value| value.as_str().filter(|name| !name.is_empty())),
            number: values().find_map(serde_json::Value::as_i64),
            nested,
        }
    }

    /// The name when there is one, as it says more than a number.
    fn code(&self) -> Option<String> {
        let number = || self.number.map(|number| number.to_string());
        self.name.map(str::to_owned).or_else(number)
    }

    /// The one provider code → retry verdict table, beside
    /// [`crate::error::retryable_status`]: what the name says, else what
    /// the number says read as an HTTP status, else nothing. gRPC
    /// `INTERNAL` stays out, as its transports have always left it.
    fn verdict(&self) -> Option<bool> {
        let named = |codes: &[&str]| {
            let name = self.name.unwrap_or_default();
            codes.iter().any(|code| code.eq_ignore_ascii_case(name))
        };
        let status = || {
            self.http_status()
                .map(|s| crate::error::retryable_status(Some(s)))
        };
        (named(TRANSIENT).then_some(true))
            .or_else(|| named(FINAL).then_some(false))
            .or_else(status)
    }

    fn http_status(&self) -> Option<u16> {
        let number = u16::try_from(self.number?).ok();
        number.filter(|number| (400..=599).contains(number))
    }
}

/// An id or model name as a response carries it: an empty one is none.
pub(crate) fn reported(id: Option<String>) -> Option<String> {
    id.filter(|id| !id.is_empty())
}

/// Parses an optional response body as JSON.
///
/// Returns:
/// - `Ok(Some(value))` when a body is present and valid JSON.
/// - `Ok(None)` when the body is absent or empty.
/// - `Err(error)` when a body is present but isn't valid JSON.
pub(crate) fn json(body: Option<&str>) -> Result<Option<serde_json::Value>, serde_json::Error> {
    body.filter(|body| !body.is_empty())
        .map(serde_json::from_str)
        .transpose()
}

#[cfg(test)]
mod tests;
