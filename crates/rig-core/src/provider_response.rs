//! Preserved provider error responses and shared capability-error inspection.
//!
//! ```
//! use rig_core::completion::CompletionError;
//!
//! let error = CompletionError::from_http_response(http::StatusCode::TOO_MANY_REQUESTS, "slow down");
//! assert!(error.is_retryable());
//! ```
use http::StatusCode;

/// A raw provider error body with captured transport metadata.
/// Callers must supply the provider's actual payload, not a generated diagnostic.
/// Serialization omits headers; deserialization restores them as `None`.
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
    /// Transport retry verdict used when status is absent or successful.
    /// A non-success HTTP status takes precedence; refusals are never retryable.
    pub transient: Option<bool>,
    /// Whether this error represents a content refusal. Refusals are never
    /// retryable; refusal content in a successful model answer is separate.
    pub refusal: bool,
}

impl ProviderResponseError {
    /// Preserve a provider error response captured with its HTTP status.
    pub fn new(status: StatusCode, body: impl Into<String>) -> Self {
        Self {
            status: Some(status),
            body: body.into(),
            provider_request_id: None,
            headers: None,
            code: None,
            transient: None,
            refusal: false,
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
        }
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

    /// Replaces the transport retry verdict. Used for absent or successful
    /// HTTP statuses unless the response is a refusal.
    pub fn with_transient(mut self, transient: Option<bool>) -> Self {
        self.transient = transient;
        self
    }

    /// Returns false for refusals. Otherwise classifies non-success HTTP statuses
    /// through [`crate::error::retryable_status`], or uses `transient` for absent
    /// or successful statuses. Missing verdicts default to false.
    pub fn is_retryable(&self) -> bool {
        if self.refusal {
            return false;
        }
        match self.status {
            Some(status) if !status.is_success() => {
                crate::error::retryable_status(Some(status.as_u16()))
            }
            _ => self.transient.unwrap_or(false),
        }
    }

    /// Returns the explicit code, falling back to nonempty string fields
    /// `error.code`, `error.status`, then `error.type` in the JSON body.
    pub fn machine_code(&self) -> Option<String> {
        self.code.clone().or_else(|| body_code(&self.body))
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
        })
    }
}

/// Returns the first nonempty string at `error.code`, `error.status`, or
/// `error.type`, in that order. Invalid JSON, non-string fields, and missing
/// envelopes yield no code.
pub fn body_code(body: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    let error = value.get("error")?;
    ["code", "status", "type"].iter().find_map(|field| {
        error
            .get(field)
            .and_then(serde_json::Value::as_str)
            .filter(|code| !code.is_empty())
            .map(str::to_owned)
    })
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

/// Preserves a decoded error envelope without assigning an HTTP status.
/// Supply the original body verbatim, not a reserialized typed event, to retain
/// unmodeled fields and original formatting. Drivers may attach transport metadata.
pub(crate) fn completion_error_from_body(
    body: impl Into<String>,
) -> crate::completion::CompletionError {
    crate::completion::CompletionError::ProviderResponse(ProviderResponseError::without_status(
        body,
    ))
}

/// Implements capability-error inspection and conversion helpers. Requires
/// `ProviderResponse(ProviderResponseError)` and `HttpError(http_client::Error)`
/// variants. Response accessors inspect only `ProviderResponse`.
macro_rules! impl_provider_response_helpers {
    ($error:ty) => {
        impl $error {
            /// Preserves the supplied status and verbatim body as [`Self::ProviderResponse`],
            /// including error envelopes returned with 2xx statuses. Attach request IDs
            /// and headers with the corresponding setters.
            pub fn from_http_response(status: http::StatusCode, body: impl Into<String>) -> Self {
                Self::ProviderResponse($crate::provider_response::ProviderResponseError::new(
                    status, body,
                ))
            }

            /// Converts captured HTTP failure responses to [`Self::ProviderResponse`],
            /// retaining status, body, and headers. Other transport errors become
            /// [`Self::HttpError`].
            pub fn from_transport_error(error: $crate::http_client::Error) -> Self {
                match error {
                    $crate::http_client::Error::InvalidStatusCodeWithDetails {
                        status,
                        body,
                        headers,
                    } => {
                        Self::from_http_response(status, body).with_response_headers(Some(headers))
                    }
                    other => Self::HttpError(other),
                }
            }

            /// Fills an absent provider response request ID, ignoring empty strings.
            /// Existing IDs and other error variants remain unchanged.
            pub fn with_provider_request_id(self, provider_request_id: Option<String>) -> Self {
                match self {
                    Self::ProviderResponse(response) if response.provider_request_id.is_none() => {
                        Self::ProviderResponse(
                            response.with_provider_request_id(provider_request_id),
                        )
                    }
                    other => other,
                }
            }

            /// Fills absent provider response headers. `None`, existing headers,
            /// and error variants without a response remain unchanged.
            pub fn with_response_headers(self, headers: Option<http::HeaderMap>) -> Self {
                let Some(headers) = headers else {
                    return self;
                };
                match self {
                    Self::ProviderResponse(response) if response.headers.is_none() => {
                        Self::ProviderResponse(response.with_headers(Some(headers)))
                    }
                    other => other,
                }
            }

            /// Attaches the HTTP status a transport reported beside a reply
            /// preserved without one (an SDK that carries the raw response
            /// next to its typed exception), so the reply classifies by its
            /// status like any HTTP reply; other variants pass through, and
            /// a status already captured is kept.
            pub fn with_provider_status(self, status: Option<http::StatusCode>) -> Self {
                match self {
                    Self::ProviderResponse(response) => {
                        Self::ProviderResponse(response.with_status(status))
                    }
                    other => other,
                }
            }

            /// Attaches the provider's own machine-readable code for the
            /// failure (a gRPC status code name, an AWS exception type) to
            /// a preserved provider response; other variants pass through.
            pub fn with_provider_code(self, code: Option<String>) -> Self {
                match self {
                    Self::ProviderResponse(response) => {
                        Self::ProviderResponse(response.with_code(code))
                    }
                    other => other,
                }
            }

            /// Replaces a preserved response's transport retry verdict, used for
            /// absent or successful HTTP statuses. Other variants are unchanged.
            ///
            /// [`ProviderResponseError::is_retryable`]: $crate::provider_response::ProviderResponseError::is_retryable
            pub fn with_transient(self, transient: Option<bool>) -> Self {
                match self {
                    Self::ProviderResponse(response) => {
                        Self::ProviderResponse(response.with_transient(transient))
                    }
                    other => other,
                }
            }

            /// Classifies transport failures with [`transient_transport`] and
            /// preserved responses with [`ProviderResponseError::is_retryable`].
            /// All other variants return false.
            ///
            /// [`transient_transport`]: $crate::error::transient_transport
            /// [`ProviderResponseError::is_retryable`]: $crate::provider_response::ProviderResponseError::is_retryable
            pub fn is_retryable(&self) -> bool {
                match self {
                    Self::HttpError(error) => $crate::error::transient_transport(error),
                    Self::ProviderResponse(response) => response.is_retryable(),
                    _ => false,
                }
            }

            /// Preserves a verbatim provider error body with no HTTP status as
            /// [`Self::ProviderResponse`].
            pub fn from_provider_body(body: impl Into<String>) -> Self {
                Self::ProviderResponse(
                    $crate::provider_response::ProviderResponseError::without_status(body),
                )
            }

            /// Returns the preserved body for [`Self::ProviderResponse`], or `None`
            /// for other variants. An empty body returns `Some("")`, while
            /// [`Self::provider_response_json`] maps it to `Ok(None)`.
            pub fn provider_response_body(&self) -> Option<&str> {
                match self {
                    Self::ProviderResponse(response) => Some(response.body.as_str()),
                    _ => None,
                }
            }

            /// Parses the provider response body as JSON.
            ///
            /// Returns:
            /// - `Ok(Some(value))` when a body is present and valid JSON.
            /// - `Ok(None)` when the body is absent or empty.
            /// - `Err(error)` when a body is present but isn't valid JSON.
            pub fn provider_response_json(
                &self,
            ) -> Result<Option<serde_json::Value>, serde_json::Error> {
                $crate::provider_response::json(self.provider_response_body())
            }

            /// Returns the preserved provider response status, if any. This may
            /// be 2xx for an error envelope; a successful HTTP status does not mean
            /// the operation succeeded. Other error variants return `None`.
            pub fn provider_response_status(&self) -> Option<http::StatusCode> {
                match self {
                    Self::ProviderResponse(response) => response.status,
                    _ => None,
                }
            }

            /// Returns the captured provider request ID, or `None` if absent or
            /// this variant carries no provider response.
            pub fn provider_request_id(&self) -> Option<&str> {
                match self {
                    Self::ProviderResponse(response) => response.provider_request_id.as_deref(),
                    _ => None,
                }
            }

            /// Returns captured response headers. This example reads the seconds
            /// form of `Retry-After`:
            ///
            /// ```no_run
            /// # use rig_core::completion::CompletionError;
            /// # use std::time::Duration;
            /// fn backoff(error: &CompletionError) -> Option<Duration> {
            ///     let seconds = error
            ///         .provider_response_headers()?
            ///         .get(http::header::RETRY_AFTER)?
            ///         .to_str()
            ///         .ok()?
            ///         .parse()
            ///         .ok()?;
            ///     Some(Duration::from_secs(seconds))
            /// }
            /// ```
            ///
            /// Returns `None` when no headers were captured: non-HTTP
            /// transports (gRPC / SDK clients), Rig-generated diagnostics,
            /// and errors funnelled from only a status and body (e.g. via
            /// [`Self::from_http_response`]). `None` therefore means "not
            /// captured", never "the response had no headers".
            pub fn provider_response_headers(&self) -> Option<&http::HeaderMap> {
                match self {
                    Self::ProviderResponse(response) => response.headers.as_ref(),
                    _ => None,
                }
            }
        }
    };
}

pub(crate) use impl_provider_response_helpers;

/// Implements setters for `message_id`, `response_id`, `provider_request_id`,
/// and `model` fields of type `Option<String>`. Empty strings become `None`.
macro_rules! response_metadata_setters {
    ($ty:ty) => {
        impl $ty {
            /// Replaces the provider message ID, treating an empty string as absent.
            pub fn with_message_id(self, message_id: impl Into<String>) -> Self {
                self.with_optional_message_id(Some(message_id.into()))
            }

            /// Attach the provider-assigned message ID when the provider
            /// reported one.
            pub fn with_optional_message_id(
                mut self,
                message_id: Option<impl Into<String>>,
            ) -> Self {
                self.message_id = message_id.map(Into::into).filter(|id| !id.is_empty());
                self
            }

            /// Attach the provider-assigned response-scoped ID.
            pub fn with_response_id(self, response_id: impl Into<String>) -> Self {
                self.with_optional_response_id(Some(response_id.into()))
            }

            /// Attach the provider-assigned response-scoped ID when the
            /// provider reported one.
            pub fn with_optional_response_id(
                mut self,
                response_id: Option<impl Into<String>>,
            ) -> Self {
                self.response_id = response_id.map(Into::into).filter(|id| !id.is_empty());
                self
            }

            /// Attach the provider's transport-level request identifier.
            pub fn with_provider_request_id(self, request_id: impl Into<String>) -> Self {
                self.with_optional_provider_request_id(Some(request_id.into()))
            }

            /// Attach the provider's transport-level request identifier when
            /// the provider reported one.
            pub fn with_optional_provider_request_id(
                mut self,
                request_id: Option<impl Into<String>>,
            ) -> Self {
                self.provider_request_id = request_id.map(Into::into).filter(|id| !id.is_empty());
                self
            }

            /// Attach the provider-reported model identifier.
            ///
            /// An empty string is treated as absent, matching the identifier
            /// setters.
            pub fn with_model(self, model: impl Into<String>) -> Self {
                self.with_optional_model(Some(model.into()))
            }

            /// Attach the provider-reported model identifier when the
            /// response carried one.
            pub fn with_optional_model(mut self, model: Option<impl Into<String>>) -> Self {
                self.model = model.map(Into::into).filter(|model| !model.is_empty());
                self
            }
        }
    };
}

pub(crate) use response_metadata_setters;
/// Metadata setters for the normalized non-completion modality responses
/// (transcription, image generation, audio generation). Same empty-string
/// filtering rule as [`response_metadata_setters`]; these responses carry no
/// message-scoped ID because nothing they produce is ever replayed as an
/// assistant message.
macro_rules! modality_response_metadata_setters {
    ($ty:ty) => {
        impl $ty {
            /// Attach the provider-assigned response-scoped ID.
            pub fn with_response_id(self, response_id: impl Into<String>) -> Self {
                self.with_optional_response_id(Some(response_id.into()))
            }

            /// Attach the provider-assigned response-scoped ID when the
            /// provider reported one. An empty string is treated as absent.
            pub fn with_optional_response_id(
                mut self,
                response_id: Option<impl Into<String>>,
            ) -> Self {
                self.response_id = response_id.map(Into::into).filter(|id| !id.is_empty());
                self
            }

            /// Attach the provider's transport-level request identifier.
            pub fn with_provider_request_id(self, request_id: impl Into<String>) -> Self {
                self.with_optional_provider_request_id(Some(request_id.into()))
            }

            /// Attach the provider's transport-level request identifier when
            /// the provider reported one. An empty string is treated as absent.
            pub fn with_optional_provider_request_id(
                mut self,
                request_id: Option<impl Into<String>>,
            ) -> Self {
                self.provider_request_id = request_id.map(Into::into).filter(|id| !id.is_empty());
                self
            }

            /// Attach the provider-reported model identifier.
            pub fn with_model(self, model: impl Into<String>) -> Self {
                self.with_optional_model(Some(model.into()))
            }

            /// Attach the provider-reported model identifier when the
            /// response carried one. An empty string is treated as absent.
            pub fn with_optional_model(mut self, model: Option<impl Into<String>>) -> Self {
                self.model = model.map(Into::into).filter(|model| !model.is_empty());
                self
            }

            /// Attach the usage the provider reported.
            pub fn with_usage(mut self, usage: $crate::completion::Usage) -> Self {
                self.usage = usage;
                self
            }

            /// Replaces the preserved provider response payload.
            pub fn with_raw(mut self, raw: impl Into<serde_json::Value>) -> Self {
                self.raw = raw.into();
                self
            }
        }
    };
}
pub(crate) use modality_response_metadata_setters;

/// Declares a capability error enum with the shared core variants
/// (`HttpError`, `JsonError`, `ResponseError`, `ProviderError`,
/// `ProviderResponse`) and wires up [`impl_provider_response_helpers!`] for
/// it, so the five modality errors stay structurally identical.
///
/// `$noun` names the capability in the generated docs (e.g. `"transcription"`
/// → "Error returned by the transcription model provider"). The first brace
/// block is spliced between `JsonError` and `ResponseError` (request-building
/// and URL errors live there); the optional second block is spliced before
/// `ProviderError` for capability-specific variants.
macro_rules! provider_error_enum {
    (
        $(#[$extra_doc:meta])*
        $name:ident, $noun:literal {
            $($mid_variants:tt)*
        }
        $({ $($late_variants:tt)* })?
    ) => {
        #[doc = concat!("Errors returned by ", $noun, " models.")]
        ///
        /// Inspect provider failures with [`Self::provider_response_body`],
        /// [`Self::provider_response_json`], and [`Self::provider_response_status`].
        $(#[$extra_doc])*
        #[derive(Debug, thiserror::Error)]
        pub enum $name {
            /// A transport failure that produced no provider reply (a
            /// connection error, a timeout, a status reported without a
            /// body); a reply with a body is [`Self::ProviderResponse`].
            #[error("HttpError: {0}")]
            HttpError($crate::http_client::Error),

            /// Json error (e.g.: serialization, deserialization)
            #[error("JsonError: {0}")]
            JsonError(#[from] serde_json::Error),

            $($mid_variants)*

            #[doc = concat!("Error parsing the ", $noun, " response")]
            #[error("ResponseError: {0}")]
            ResponseError(String),

            $($($late_variants)*)?

            #[doc = concat!("Error returned by the ", $noun, " model provider")]
            #[error("ProviderError: {0}")]
            ProviderError(String),

            #[doc = concat!("Raw error response preserved from the ", $noun, " model provider")]
            #[error("ProviderResponseError: {0}")]
            ProviderResponse($crate::provider_response::ProviderResponseError),
        }

        $crate::provider_response::impl_provider_response_helpers!($name);
        $crate::wire::impl_wire_error!($name);

        impl From<$crate::http_client::Error> for $name {
            fn from(error: $crate::http_client::Error) -> Self {
                Self::from_transport_error(error)
            }
        }
    };
}

pub(crate) use provider_error_enum;

#[cfg(test)]
mod tests;
