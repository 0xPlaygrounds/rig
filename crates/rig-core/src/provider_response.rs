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
    /// Transport retry verdict used when status is absent or successful.
    /// A non-success HTTP status takes precedence; refusals are never retryable.
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
        Self {
            status: Some(status),
            body: body.into(),
            provider_request_id: None,
            headers: None,
            code: None,
            transient: None,
            refusal: false,
            route: None,
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

#[cfg(test)]
mod tests;
