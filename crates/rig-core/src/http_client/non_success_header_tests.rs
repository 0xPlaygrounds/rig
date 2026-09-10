use super::*;

/// `None` means "not captured" and must not be confused with an empty map:
/// every other shape of this error reports it.
#[test]
fn non_success_headers_absent_when_not_captured() {
    for error in [
        Error::InvalidStatusCodeWithMessage(
            StatusCode::TOO_MANY_REQUESTS,
            "rate limited".to_string(),
        ),
        Error::InvalidStatusCode(StatusCode::TOO_MANY_REQUESTS),
        Error::StreamEnded,
    ] {
        assert!(error.non_success_headers().is_none());
    }

    // A captured-but-empty map is `Some`, not `None`.
    let error = Error::InvalidStatusCodeWithDetails {
        status: StatusCode::TOO_MANY_REQUESTS,
        body: "rate limited".to_string(),
        headers: Box::new(HeaderMap::new()),
    };
    assert!(error.non_success_headers().is_some_and(HeaderMap::is_empty));
}
