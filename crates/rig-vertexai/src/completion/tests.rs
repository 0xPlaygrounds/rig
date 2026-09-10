use super::*;

// The `send()` RPC error type comes from the `google-cloud-aiplatform-v1`
// SDK and is not trivially constructible, so `rpc_error` is generic over
// `impl Display` and we pin it here with a representative error string of
// its parameter type. This guards against a revert to `ProviderError`,
// which would surface the body as `None`.
#[test]
fn rpc_error_preserves_raw_text_without_http_status() {
    let raw = "status: Unavailable, message: \"the service is currently unavailable\"";

    let err = rpc_error(raw);

    assert_eq!(err.provider_response_body(), Some(raw));
    assert_eq!(err.provider_response_status(), None);
}
