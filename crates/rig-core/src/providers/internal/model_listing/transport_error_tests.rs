use super::*;

/// Regression (rig#2314 review): the header-preserving transport variant
/// must classify as an ApiError with provider/path context exactly like
/// the header-less one — the reqwest transport now emits it for every
/// non-2xx.
#[test]
fn details_variant_maps_to_api_error_with_context() {
    let error = map_transport_error(
        "test-provider",
        "/models",
        http_client::Error::InvalidStatusCodeWithDetails {
            status: http::StatusCode::UNAUTHORIZED,
            body: r#"{"error":"no"}"#.to_string(),
            headers: Box::new(http::HeaderMap::new()),
        },
    );
    let with_message = map_transport_error(
        "test-provider",
        "/models",
        http_client::Error::InvalidStatusCodeWithMessage(
            http::StatusCode::UNAUTHORIZED,
            r#"{"error":"no"}"#.to_string(),
        ),
    );
    assert_eq!(format!("{error}"), format!("{with_message}"));
    assert!(
        matches!(error, ModelListingError::ApiError { .. }),
        "got {error:?}"
    );
}
