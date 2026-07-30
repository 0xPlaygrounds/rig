//! The shared, concrete HTTP executor for provider free functions.
//!
//! [`HttpRuntime`] is one plain struct owning the HTTP transport — the
//! replacement for the deleted `H: HttpClientExt` type parameter that used to
//! be threaded through every provider client alias and model. Transport variation is an enum, not a generic: the default arm
//! is `reqwest`, and the `test-utils` arm replays scripted/recorded
//! exchanges so the pure provider functions are exercisable by cassette-style
//! tests without a network.

use http::StatusCode;

use crate::completion::CompletionError;
use crate::http_client::{self, Backend};

/// The concrete HTTP executor. Cheap to clone.
#[derive(Clone)]
pub struct HttpRuntime {
    transport: Transport,
}

#[derive(Clone)]
pub(crate) enum Transport {
    Reqwest(reqwest::Client),
    #[cfg(any(test, feature = "test-utils"))]
    Recording(crate::test_utils::RecordingHttpClient),
    #[cfg(any(test, feature = "test-utils"))]
    Sequenced(crate::test_utils::SequencedHttpClient),
    #[cfg(any(test, feature = "test-utils"))]
    MockStreaming(crate::test_utils::MockStreamingClient),
    #[cfg(any(test, feature = "test-utils"))]
    StreamingError(crate::test_utils::HttpErrorStreamingClient),
    #[cfg(any(test, feature = "test-utils"))]
    SequencedStreaming(crate::test_utils::SequencedStreamingHttpClient),
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
    #[cfg(any(test, feature = "test-utils"))]
    pub fn recording(client: crate::test_utils::RecordingHttpClient) -> Self {
        Self {
            transport: Transport::Recording(client),
        }
    }

    /// A runtime that replays a scripted sequence of test responses.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn sequenced(client: crate::test_utils::SequencedHttpClient) -> Self {
        Self {
            transport: Transport::Sequenced(client),
        }
    }

    /// A runtime whose streaming responses replay canned SSE bytes.
    ///
    /// The streaming counterpart of [`recording`](Self::recording): lets a test
    /// drive a provider's `open_stream` end to end without a network.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn mock_streaming(client: crate::test_utils::MockStreamingClient) -> Self {
        Self {
            transport: Transport::MockStreaming(client),
        }
    }

    /// A runtime whose streaming responses fail with a non-success status and
    /// body, for asserting that providers preserve provider errors on the
    /// streaming path.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn streaming_error(client: crate::test_utils::HttpErrorStreamingClient) -> Self {
        Self {
            transport: Transport::StreamingError(client),
        }
    }

    /// A runtime replaying a scripted sequence of streaming responses.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn sequenced_streaming(client: crate::test_utils::SequencedStreamingHttpClient) -> Self {
        Self {
            transport: Transport::SequencedStreaming(client),
        }
    }

    /// Open an SSE event stream for `request`.
    ///
    /// This is the transport edge of the sans-IO stream parsers: backend
    /// genericity ends here, and provider stream state machines consume the
    /// concrete
    /// [`BoxedEventSource`](crate::http_client::sse::BoxedEventSource).
    ///
    /// `allow_missing_content_type` relaxes the `text/event-stream`
    /// content-type check for backends that omit it (ChatGPT's backend API).
    pub(crate) fn sse_events(
        &self,
        request: http::Request<Vec<u8>>,
        allow_missing_content_type: bool,
    ) -> crate::http_client::sse::BoxedEventSource {
        use crate::http_client::sse::boxed_event_source;
        match &self.transport {
            Transport::Reqwest(client) => {
                boxed_event_source(client.clone(), request, allow_missing_content_type)
            }
            #[cfg(any(test, feature = "test-utils"))]
            Transport::Recording(client) => {
                boxed_event_source(client.clone(), request, allow_missing_content_type)
            }
            #[cfg(any(test, feature = "test-utils"))]
            Transport::Sequenced(client) => {
                boxed_event_source(client.clone(), request, allow_missing_content_type)
            }
            #[cfg(any(test, feature = "test-utils"))]
            Transport::MockStreaming(client) => {
                boxed_event_source(client.clone(), request, allow_missing_content_type)
            }
            #[cfg(any(test, feature = "test-utils"))]
            Transport::StreamingError(client) => {
                boxed_event_source(client.clone(), request, allow_missing_content_type)
            }
            #[cfg(any(test, feature = "test-utils"))]
            Transport::SequencedStreaming(client) => {
                boxed_event_source(client.clone(), request, allow_missing_content_type)
            }
        }
    }

    /// Send a request and return the raw byte-stream response.
    ///
    /// The non-SSE transport edge, for providers whose streams are not
    /// server-sent events (Ollama's native NDJSON `/api/chat` stream). The
    /// returned [`StreamingResponse`](crate::http_client::StreamingResponse) is
    /// already type-erased, so no HTTP-client genericity leaks to the caller.
    pub(crate) async fn send_streaming(
        &self,
        request: http::Request<Vec<u8>>,
    ) -> Result<crate::http_client::StreamingResponse, http_client::Error> {
        match &self.transport {
            Transport::Reqwest(client) => client.send_streaming(request).await,
            #[cfg(any(test, feature = "test-utils"))]
            Transport::Recording(client) => client.send_streaming(request).await,
            #[cfg(any(test, feature = "test-utils"))]
            Transport::Sequenced(client) => client.send_streaming(request).await,
            #[cfg(any(test, feature = "test-utils"))]
            Transport::MockStreaming(client) => client.send_streaming(request).await,
            #[cfg(any(test, feature = "test-utils"))]
            Transport::StreamingError(client) => client.send_streaming(request).await,
            #[cfg(any(test, feature = "test-utils"))]
            Transport::SequencedStreaming(client) => client.send_streaming(request).await,
        }
    }

    /// Send a request and return `(status, body bytes)`.
    ///
    /// The binary sibling of [`send`](Self::send) for modality endpoints
    /// whose success bodies are not text (e.g. audio). Non-success statuses
    /// are returned as values with their body preserved; only
    /// transport-level failures error.
    pub(crate) async fn send_bytes(
        &self,
        request: http::Request<Vec<u8>>,
    ) -> Result<(StatusCode, Vec<u8>), http_client::Error> {
        let sent = match &self.transport {
            Transport::Reqwest(client) => client.send(request).await,
            #[cfg(any(test, feature = "test-utils"))]
            Transport::Recording(client) => client.send(request).await,
            #[cfg(any(test, feature = "test-utils"))]
            Transport::Sequenced(client) => client.send(request).await,
            #[cfg(any(test, feature = "test-utils"))]
            Transport::MockStreaming(client) => client.send(request).await,
            #[cfg(any(test, feature = "test-utils"))]
            Transport::StreamingError(client) => client.send(request).await,
            #[cfg(any(test, feature = "test-utils"))]
            Transport::SequencedStreaming(client) => client.send(request).await,
        };
        Self::flatten_bytes_response(sent)
    }

    /// Send a multipart request and return `(status, body bytes)`.
    ///
    /// Same value-vs-error contract as [`send_bytes`](Self::send_bytes).
    pub(crate) async fn send_multipart(
        &self,
        request: http::Request<http_client::MultipartForm>,
    ) -> Result<(StatusCode, Vec<u8>), http_client::Error> {
        let sent = match &self.transport {
            Transport::Reqwest(client) => client.send_multipart(request).await,
            #[cfg(any(test, feature = "test-utils"))]
            Transport::Recording(client) => client.send_multipart(request).await,
            #[cfg(any(test, feature = "test-utils"))]
            Transport::Sequenced(client) => client.send_multipart(request).await,
            #[cfg(any(test, feature = "test-utils"))]
            Transport::MockStreaming(client) => client.send_multipart(request).await,
            #[cfg(any(test, feature = "test-utils"))]
            Transport::StreamingError(client) => client.send_multipart(request).await,
            #[cfg(any(test, feature = "test-utils"))]
            Transport::SequencedStreaming(client) => client.send_multipart(request).await,
        };
        Self::flatten_bytes_response(sent)
    }

    fn flatten_bytes_response(
        sent: http_client::Result<http::Response<bytes::Bytes>>,
    ) -> Result<(StatusCode, Vec<u8>), http_client::Error> {
        match sent {
            Ok(response) => {
                let status = response.status();
                Ok((status, response.into_body().to_vec()))
            }
            // The transport layer folds non-success statuses into this error
            // variant with the body preserved; surface them as values.
            Err(http_client::Error::InvalidStatusCodeWithMessage(status, body)) => {
                Ok((status, body.into_bytes()))
            }
            Err(http_client::Error::InvalidStatusCode(status)) => Ok((status, Vec::new())),
            Err(error) => Err(error),
        }
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
            Transport::Reqwest(client) => client.send(request).await,
            #[cfg(any(test, feature = "test-utils"))]
            Transport::Recording(client) => client.send(request).await,
            #[cfg(any(test, feature = "test-utils"))]
            Transport::Sequenced(client) => client.send(request).await,
            #[cfg(any(test, feature = "test-utils"))]
            Transport::MockStreaming(client) => client.send(request).await,
            #[cfg(any(test, feature = "test-utils"))]
            Transport::StreamingError(client) => client.send(request).await,
            #[cfg(any(test, feature = "test-utils"))]
            Transport::SequencedStreaming(client) => client.send(request).await,
        };
        match sent {
            Ok(response) => {
                let status = response.status();
                let body = String::from_utf8_lossy(response.body()).into_owned();
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
            #[cfg(any(test, feature = "test-utils"))]
            Transport::Recording(_) => "recording",
            #[cfg(any(test, feature = "test-utils"))]
            Transport::Sequenced(_) => "sequenced",
            #[cfg(any(test, feature = "test-utils"))]
            Transport::MockStreaming(_) => "mock-streaming",
            #[cfg(any(test, feature = "test-utils"))]
            Transport::StreamingError(_) => "streaming-error",
            #[cfg(any(test, feature = "test-utils"))]
            Transport::SequencedStreaming(_) => "sequenced-streaming",
        };
        formatter
            .debug_struct("HttpRuntime")
            .field("transport", &transport)
            .finish()
    }
}
