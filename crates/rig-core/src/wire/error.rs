//! Error construction and classification for operation drivers.
//!
//! ```
//! use rig_core::{completion::CompletionError, wire::WireError};
//!
//! let error = CompletionError::decode("missing response content".to_owned());
//! assert!(error.provider_response_status().is_none());
//! ```

use crate::error::ErrorReport;
use crate::observe::AdapterErrorBoundary;
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};

/// Constructs and classifies transport, provider-response, and decode errors
/// while preserving available response metadata.
pub trait WireError: std::error::Error + WasmCompatSend + WasmCompatSync + Sized + 'static {
    /// Route a transport error: a non-success reply the transport reported
    /// as an error is the provider's reply; a response-less failure is not.
    fn transport(error: crate::http_client::Error) -> Self;

    /// Preserves a provider error body and status, including error envelopes
    /// returned with successful HTTP statuses.
    fn http_response(status: http::StatusCode, body: &str) -> Self;

    /// An undecodable body.
    fn json(error: serde_json::Error) -> Self;

    /// A reply that decoded but does not answer the request.
    fn decode(message: String) -> Self;

    /// The provider's error envelope, preserved without a status: the
    /// decoder read it off a 2xx body, and the driver stamps the status it
    /// arrived with.
    fn provider_body(body: &str) -> Self;

    /// Attach the status a preserved reply arrived with, when it has none.
    fn with_provider_status(self, status: Option<http::StatusCode>) -> Self;

    /// Attaches the provider's transport request ID when supported.
    fn with_provider_request_id(self, request_id: Option<String>) -> Self;

    /// Attaches response headers, including rate-limit metadata.
    fn with_response_headers(self, headers: Option<http::HeaderMap>) -> Self;

    /// The HTTP status of a preserved provider reply, when this error has
    /// one.
    fn provider_response_status(&self) -> Option<http::StatusCode>;

    /// The provider's reply preserved on this error, when it has one: the
    /// payload the decoder's projection still reads facts off.
    fn provider_response_body(&self) -> Option<&str>;

    /// Adds provider and path context when supported. The default returns the
    /// error unchanged, preserving provider bodies.
    fn with_route(self, _provider: &str, _path: &str) -> Self {
        self
    }

    /// The wire form of this error, for observation and the bus.
    fn report(&self) -> ErrorReport;

    /// Which boundary produced it, by the typed error rather than its text.
    fn boundary(&self) -> AdapterErrorBoundary;
}

/// Implements [`WireError`] for enums with the standard provider-error variants
/// and constructors. Additional variants are classified as request faults.
macro_rules! impl_wire_error {
    ($error:ty) => {
        impl $crate::wire::WireError for $error {
            fn transport(error: $crate::http_client::Error) -> Self {
                Self::from_transport_error(error)
            }

            fn http_response(status: http::StatusCode, body: &str) -> Self {
                Self::from_http_response(status, body)
            }

            fn json(error: serde_json::Error) -> Self {
                Self::JsonError(error)
            }

            fn decode(message: String) -> Self {
                Self::ResponseError(message)
            }

            fn provider_body(body: &str) -> Self {
                Self::from_provider_body(body)
            }

            fn with_provider_status(self, status: Option<http::StatusCode>) -> Self {
                Self::with_provider_status(self, status)
            }

            fn with_provider_request_id(self, request_id: Option<String>) -> Self {
                Self::with_provider_request_id(self, request_id)
            }

            fn with_response_headers(self, headers: Option<http::HeaderMap>) -> Self {
                Self::with_response_headers(self, headers)
            }

            fn provider_response_status(&self) -> Option<http::StatusCode> {
                Self::provider_response_status(self)
            }

            fn provider_response_body(&self) -> Option<&str> {
                Self::provider_response_body(self)
            }

            fn report(&self) -> $crate::error::ErrorReport {
                $crate::error::ErrorReport::from(self)
            }

            fn boundary(&self) -> $crate::observe::AdapterErrorBoundary {
                use $crate::observe::AdapterErrorBoundary as B;
                match self {
                    Self::HttpError(error) => B::from_http(error),
                    Self::JsonError(_) | Self::ResponseError(_) => B::Decode,
                    Self::ProviderError(_) | Self::ProviderResponse(_) => B::ProviderResponse,
                    // Every remaining variant an operation adds is a fault
                    // in the request it was asked to build.
                    _ => B::Request,
                }
            }
        }
    };
}

pub(crate) use impl_wire_error;
