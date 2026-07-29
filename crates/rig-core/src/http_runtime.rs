//! The shared, concrete HTTP executor for provider free functions.
//!
//! [`HttpRuntime`] is one plain struct owning the HTTP transport — the
//! replacement for the `H` type parameter threaded through provider client
//! aliases. Transport variation is an enum, not a generic: the default arm
//! is `reqwest`, and the `test-utils` arm replays scripted/recorded
//! exchanges so the pure provider functions are exercisable by cassette-style
//! tests without a network.

use http::StatusCode;

use crate::completion::CompletionError;
use crate::http_client::{self, HttpClientExt};

/// The concrete HTTP executor. Cheap to clone.
#[derive(Clone)]
pub struct HttpRuntime {
    transport: Transport,
}

#[derive(Clone)]
pub(crate) enum Transport {
    Reqwest(reqwest::Client),
    #[cfg(feature = "test-utils")]
    Recording(crate::test_utils::RecordingHttpClient),
    #[cfg(feature = "test-utils")]
    Sequenced(crate::test_utils::SequencedHttpClient),
}

impl Default for HttpRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl HttpRuntime {
    /// A runtime over a default `reqwest` client.
    pub fn new() -> Self {
        Self {
            transport: Transport::Reqwest(reqwest::Client::new()),
        }
    }

    /// A runtime over an existing `reqwest` client.
    pub fn from_reqwest(client: reqwest::Client) -> Self {
        Self {
            transport: Transport::Reqwest(client),
        }
    }

    /// A runtime that replays through a recording/replaying test client.
    #[cfg(feature = "test-utils")]
    pub fn recording(client: crate::test_utils::RecordingHttpClient) -> Self {
        Self {
            transport: Transport::Recording(client),
        }
    }

    /// A runtime that replays a scripted sequence of test responses.
    #[cfg(feature = "test-utils")]
    pub fn sequenced(client: crate::test_utils::SequencedHttpClient) -> Self {
        Self {
            transport: Transport::Sequenced(client),
        }
    }

    /// The underlying transport, for provider modules that drive streaming
    /// helpers generic over `HttpClientExt` (transitional until the sans-IO
    /// stream parser lands).
    pub(crate) fn transport(&self) -> &Transport {
        &self.transport
    }

    /// Send a request and return `(status, body)`.
    ///
    /// Non-success statuses are returned as values (with their body) rather
    /// than errors, so callers can hand them to a provider's
    /// `parse_response` for uniform error shaping; only transport-level
    /// failures error.
    pub async fn send(
        &self,
        request: http::Request<Vec<u8>>,
    ) -> Result<(StatusCode, String), CompletionError> {
        let sent = match &self.transport {
            Transport::Reqwest(client) => client.send::<Vec<u8>, Vec<u8>>(request).await,
            #[cfg(feature = "test-utils")]
            Transport::Recording(client) => client.send::<Vec<u8>, Vec<u8>>(request).await,
            #[cfg(feature = "test-utils")]
            Transport::Sequenced(client) => client.send::<Vec<u8>, Vec<u8>>(request).await,
        };
        match sent {
            Ok(response) => {
                let status = response.status();
                let body = http_client::text(response).await?;
                Ok((status, body))
            }
            // The transport layer folds non-success statuses into this error
            // variant with the body preserved; surface them as values.
            Err(http_client::Error::InvalidStatusCodeWithMessage(status, body)) => {
                Ok((status, body))
            }
            Err(http_client::Error::InvalidStatusCode(status)) => Ok((status, String::new())),
            Err(error) => Err(CompletionError::HttpError(error)),
        }
    }
}

impl std::fmt::Debug for HttpRuntime {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let transport = match &self.transport {
            Transport::Reqwest(_) => "reqwest",
            #[cfg(feature = "test-utils")]
            Transport::Recording(_) => "recording",
            #[cfg(feature = "test-utils")]
            Transport::Sequenced(_) => "sequenced",
        };
        formatter
            .debug_struct("HttpRuntime")
            .field("transport", &transport)
            .finish()
    }
}
