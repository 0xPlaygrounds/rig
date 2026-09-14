use super::*;

/// `None` means "no response" and must not be confused with an empty map:
/// a response-less failure has no headers, a reply always reports its own.
#[test]
fn non_success_headers_absent_when_not_captured() {
    for error in [
        Error::StreamEnded,
        Error::NoHeaders,
        Error::instance(std::io::Error::other("connection reset")),
    ] {
        assert!(error.non_success_headers().is_none());
        assert!(error.non_success_status().is_none());
    }

    // A captured-but-empty map is `Some`, not `None`.
    let error = Error::InvalidStatusCodeWithDetails {
        status: StatusCode::TOO_MANY_REQUESTS,
        body: "rate limited".to_string(),
        headers: Box::new(HeaderMap::new()),
    };
    assert!(error.non_success_headers().is_some_and(HeaderMap::is_empty));
}
