//! Shared logic for inspecting provider error response bodies across capability errors.
use http::StatusCode;

/// A raw error response preserved from a provider.
///
/// Capability errors store this in their `ProviderResponse` variants when Rig
/// has the provider's response body in hand. Unlike `ProviderError(String)`,
/// which may carry Rig-generated diagnostics, this type always represents the
/// payload the provider actually returned.
///
/// Prefer [`Self::new`] / [`Self::without_status`] and the `with_*` setters
/// over a struct literal: a literal has to be revisited every time transport
/// metadata grows, and the constructors do not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderResponseError {
    /// HTTP status of the provider response, when it was captured alongside the body.
    pub status: Option<StatusCode>,
    /// Raw response body as returned by the provider.
    pub body: String,
    /// The provider's transport request id for the failed call (HTTP response
    /// header such as Anthropic `request-id` / OpenAI `x-request-id`, or SDK
    /// response metadata) — the id provider support asks for when
    /// investigating a request, which matters most on exactly these failed
    /// calls. `None` means the provider did not report one — a documented
    /// outcome, never a secondary error (rig#2314).
    pub provider_request_id: Option<String>,
    /// The response's headers, verbatim, when the capture path had them in
    /// hand — the rate-limit metadata (`Retry-After`, `x-ratelimit-*`) a
    /// caller needs to back off correctly after a 429 (rig#2210). Boxed to
    /// keep this error small enough for `clippy::result_large_err`. `None`
    /// means "not captured", never "the response had no headers".
    pub headers: Option<Box<http::HeaderMap>>,
}

impl ProviderResponseError {
    /// Preserve a provider error response captured with its HTTP status.
    pub fn new(status: StatusCode, body: impl Into<String>) -> Self {
        Self {
            status: Some(status),
            body: body.into(),
            provider_request_id: None,
            headers: None,
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
        }
    }

    /// Attach the transport request id the failed response reported.
    pub fn with_provider_request_id(mut self, request_id: Option<String>) -> Self {
        self.provider_request_id = request_id.filter(|id| !id.is_empty());
        self
    }

    /// Attach the response's headers, so rate-limit metadata survives onto the
    /// error (rig#2210).
    pub fn with_headers(mut self, headers: Option<Box<http::HeaderMap>>) -> Self {
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

/// The wire shape of [`ProviderResponseError`]: the status as its number,
/// the body and the provider's request id — the error's identity. The
/// headers are not on the wire: they are the transport's (`date`, the
/// rate-limit counters), differ on every response to the same request,
/// and a record that carried them was never the same twice — an effect
/// log's error record replayed from one cassette diverged from itself a
/// second later. A deserialized error has `headers: None`, "not captured",
/// which is what a replayed error is.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderResponseErrorWire {
    status: Option<u16>,
    body: String,
    provider_request_id: Option<String>,
}

impl serde::Serialize for ProviderResponseError {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        ProviderResponseErrorWire {
            status: self.status.map(|status| status.as_u16()),
            body: self.body.clone(),
            provider_request_id: self.provider_request_id.clone(),
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
        })
    }
}

/// Parses an optional response body as JSON.
///
/// Returns:
/// - `Ok(Some(value))` when a body is present and valid JSON.
/// - `Ok(None)` when no body is present.
/// - `Err(error)` when a body is present but isn't valid JSON.
pub(crate) fn json(body: Option<&str>) -> Result<Option<serde_json::Value>, serde_json::Error> {
    body.filter(|body| !body.is_empty())
        .map(serde_json::from_str)
        .transpose()
}

pub(crate) fn completion_error_from_body(
    body: impl Into<String>,
) -> crate::completion::CompletionError {
    crate::completion::CompletionError::ProviderResponse(ProviderResponseError::without_status(
        body,
    ))
}

/// Implements the `provider_response_*` inspection helpers on a capability error
/// enum.
///
/// The enum must have a `ProviderResponse(`[`ProviderResponseError`]`)` variant
/// and an `HttpError(`[`http_client::Error`](crate::http_client::Error)`)`
/// variant; the generated helpers read from those two sources only, since they
/// are the only ones that genuinely represent a provider's response.
macro_rules! impl_provider_response_helpers {
    ($error:ty $(, $report:ident)?) => {
        impl $error {
            /// Builds an error from a captured HTTP status and raw response body,
            /// routing it so the `provider_response_*` helpers stay useful.
            ///
            /// This is the single funnel every HTTP-error path should use instead
            /// of flattening a status and body into a `ProviderError(String)`:
            /// - A **success (2xx)** status carries a provider-authored error
            ///   envelope, so it is preserved as [`Self::ProviderResponse`]
            ///   together with the status.
            /// - A **non-success** status is preserved as
            ///   [`Self::HttpError`]`(`[`http_client::Error::InvalidStatusCodeWithMessage`](crate::http_client::Error::InvalidStatusCodeWithMessage)`)`.
            ///
            /// Either way the raw `body` is kept verbatim and the status stays
            /// recoverable through [`Self::provider_response_status`]. Read the
            /// response body exactly once and hand it here for both branches.
            pub fn from_http_response(status: http::StatusCode, body: impl Into<String>) -> Self {
                if status.is_success() {
                    Self::ProviderResponse($crate::provider_response::ProviderResponseError::new(
                        status, body,
                    ))
                } else {
                    Self::HttpError($crate::http_client::Error::InvalidStatusCodeWithMessage(
                        status,
                        body.into(),
                    ))
                }
            }

            /// [`Self::from_http_response`] for paths that captured the
            /// provider's transport request id alongside the response
            /// (rig#2314).
            ///
            /// Unlike the metadata-less funnel, a **non-success** status is
            /// preserved as [`Self::ProviderResponse`] too — `http_client`'s
            /// error type has no slot for provider metadata, and the id the
            /// provider reported on a failed call is exactly what support
            /// asks for. Classification therefore follows the *code path*
            /// (did this call site capture transport metadata?), never the
            /// presence of the header on a particular response, so a given
            /// provider's errors classify consistently. The status stays
            /// recoverable through [`Self::provider_response_status`] and the
            /// id through [`Self::provider_request_id`].
            pub fn from_http_response_with_request_id(
                status: http::StatusCode,
                body: impl Into<String>,
                provider_request_id: Option<String>,
            ) -> Self {
                Self::ProviderResponse(
                    $crate::provider_response::ProviderResponseError::new(status, body)
                        .with_provider_request_id(provider_request_id),
                )
            }

            /// Attaches the response's headers to an error just built by one of
            /// the `from_http_response*` funnels, so rate-limit metadata
            /// (`Retry-After`, `x-ratelimit-*`) survives onto it (rig#2210).
            ///
            /// This is a separate step rather than a funnel parameter because
            /// the funnels' classification is fixed by the *call path* (does
            /// this provider have a request-id contract?), while header capture
            /// depends only on whether the transport handed the response back.
            /// Both routes can therefore carry headers:
            /// [`Self::ProviderResponse`] stores them alongside the request id,
            /// and a non-success [`Self::HttpError`] is upgraded in place to
            /// [`http_client::Error::InvalidStatusCodeWithDetails`](crate::http_client::Error::InvalidStatusCodeWithDetails),
            /// which displays identically to the header-less variant.
            ///
            /// Passing `None` leaves the error untouched, as does calling this
            /// on a variant with no response to annotate. An error that already
            /// captured headers keeps the ones it has: the first capture is the
            /// one that saw the response, so this never overwrites.
            pub fn with_response_headers(self, headers: Option<Box<http::HeaderMap>>) -> Self {
                let Some(headers) = headers else {
                    return self;
                };
                match self {
                    // The first capture is the one that saw the response; a
                    // later caller only fills the gap, mirroring how the
                    // request-id slot is stamped (rig#2314).
                    Self::ProviderResponse(response) if response.headers.is_none() => {
                        Self::ProviderResponse(response.with_headers(Some(headers)))
                    }
                    Self::HttpError($crate::http_client::Error::InvalidStatusCodeWithMessage(
                        status,
                        body,
                    )) => {
                        Self::HttpError($crate::http_client::Error::InvalidStatusCodeWithDetails {
                            status,
                            body,
                            headers,
                        })
                    }
                    other => other,
                }
            }

            /// Preserves a raw provider error body that has **no HTTP status**.
            ///
            /// Use this for non-HTTP transports (gRPC / SDK clients such as AWS
            /// Bedrock, Vertex AI, or the gRPC Gemini client) where the provider
            /// returns an error payload but no [`http::StatusCode`] is available.
            /// The body is preserved as [`Self::ProviderResponse`] with
            /// `status == None`, so [`Self::provider_response_body`] still surfaces
            /// it while [`Self::provider_response_status`] returns `None`.
            pub fn from_provider_body(body: impl Into<String>) -> Self {
                Self::ProviderResponse(
                    $crate::provider_response::ProviderResponseError::without_status(body),
                )
            }

            /// Returns the raw provider response body when available.
            ///
            /// This is available for:
            /// - `Self::ProviderResponse` using its preserved body.
            /// - `Self::HttpError` when it wraps an HTTP non-success response that
            ///   carries a body.
            ///
            /// Returns `None` for any other variant — for example a Rig-generated
            /// `ProviderError` diagnostic, or a failure from a transport with no
            /// provider response body to preserve. An empty preserved body is
            /// reported as `Some("")` (the provider returned no payload), which is
            /// distinct from `None`; note that [`Self::provider_response_json`]
            /// maps that same empty body to `Ok(None)`.
            pub fn provider_response_body(&self) -> Option<&str> {
                match self {
                    Self::ProviderResponse(response) => Some(response.body.as_str()),
                    Self::HttpError(error) => error.non_success_body(),
                    _ => None,
                }
            }

            /// Parses the provider response body as JSON.
            ///
            /// Returns:
            /// - `Ok(Some(value))` when a body is present and valid JSON.
            /// - `Ok(None)` when no provider response body is available.
            /// - `Err(error)` when a body is present but isn't valid JSON.
            pub fn provider_response_json(
                &self,
            ) -> Result<Option<serde_json::Value>, serde_json::Error> {
                $crate::provider_response::json(self.provider_response_body())
            }

            /// Returns the HTTP status code when this error preserves one, either
            /// from a non-success HTTP response, from a preserved provider
            /// response, or from a 2xx error envelope.
            ///
            /// **Warning:** this can return a **2xx** status. Some providers send
            /// an error envelope alongside a success status, which Rig preserves
            /// via [`Self::ProviderResponse`]. Callers must not infer failure from
            /// the status code alone — the existence of this error already means
            /// the call failed. Returns `None` for non-HTTP transports (gRPC / SDK
            /// clients) and for variants that carry no provider response.
            pub fn provider_response_status(&self) -> Option<http::StatusCode> {
                match self {
                    Self::ProviderResponse(response) => response.status,
                    Self::HttpError(error) => error.non_success_status(),
                    $(Self::$report(report) => report
                        .http_status
                        .and_then(|status| http::StatusCode::from_u16(status).ok()),)?
                    _ => None,
                }
            }

            /// Returns the provider's transport request id for the failed
            /// call, when the capture path preserved one (rig#2314) — the id
            /// provider support asks for. `None` for providers that report
            /// none, for paths that captured no transport metadata, and for
            /// errors with no provider response at all.
            pub fn provider_request_id(&self) -> Option<&str> {
                match self {
                    Self::ProviderResponse(response) => response.provider_request_id.as_deref(),
                    $(Self::$report(report) => report.request_id.as_deref(),)?
                    _ => None,
                }
            }

            /// Returns the response's headers when the capture path preserved
            /// them (rig#2210) — the rate-limit metadata (`Retry-After`,
            /// `x-ratelimit-*`) a caller needs to back off correctly:
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
            /// errors funnelled from only a status and body (e.g. via
            /// [`Self::from_http_response`]), and transports that report a
            /// non-success status without preserving them. `None` therefore
            /// means "not captured", never "the response had no headers".
            pub fn provider_response_headers(&self) -> Option<&http::HeaderMap> {
                match self {
                    Self::ProviderResponse(response) => response.headers.as_deref(),
                    Self::HttpError(error) => error.non_success_headers(),
                    _ => None,
                }
            }
        }
    };
}

pub(crate) use impl_provider_response_helpers;

/// Implements the shared response-metadata setters (`with_message_id`,
/// `with_response_id`, `with_provider_request_id`, `with_model`, `with_raw`
/// and their `_optional` forms) on a response type with `message_id`,
/// `response_id`, `provider_request_id`, and `model` fields of type
/// `Option<String>` and a `raw` field of type `serde_json::Value`.
///
/// An empty string is treated as absent: gateways that echo `""` for fields
/// they don't populate must not produce a `Some("")` that differs between the
/// buffered and streaming paths. The invariant lives in these generated
/// setters so no provider call site can diverge. `finish_reason` handling is
/// intentionally left to each type, since reconciliation rules differ.
///
/// `raw` is not an identifier, but it belongs here for the same reason the
/// identifiers do: it is per-attempt metadata that both surfaces observe —
/// the unary response and the streaming terminal record carry the same field
/// with the same meaning, populated at the provider seams from one setter, so
/// neither surface can grow a variant the other lacks.
macro_rules! response_metadata_setters {
    ($ty:ty) => {
        impl $ty {
            /// Attach the provider-assigned message ID.
            ///
            /// An empty string is treated as absent: gateways that echo `""`
            /// for fields they don't populate must not produce a `Some("")`
            /// that differs between the buffered and streaming paths. All
            /// identifier and model setters share this rule so the invariant
            /// lives here rather than at every provider call site.
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

            /// Attach the provider's own response, serialized — the value the
            /// model's inherent raw method would have returned. Every provider
            /// seam calls this; see the `raw` field for the exact meaning of
            /// the payload (and of `Value::Null`).
            pub fn with_raw(mut self, raw: impl Into<serde_json::Value>) -> Self {
                self.raw = raw.into();
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

            /// Attach the provider's own response, serialized — the value the
            /// model's inherent raw method would have returned.
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
            /// Http error (e.g.: connection error, timeout, etc.)
            #[error("HttpError: {0}")]
            HttpError(#[from] $crate::http_client::Error),

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
    };
}

pub(crate) use provider_error_enum;

#[cfg(test)]
mod tests;
