//! A module that details a generic implementation of SSE using a generic HTTP client.

use http::Request;

#[derive(Clone)]
struct StreamingRequest<T> {
    client: T,
    req: Request<Vec<u8>>,
    last_event_id: Option<String>,
}
