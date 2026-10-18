use super::{GeminiGrpc, GeminiGrpcError};

/// A key that cannot travel as gRPC metadata is rejected before any
/// connection is attempted, on a current-thread runtime.
#[tokio::test]
async fn new_rejects_unsendable_api_key_without_connecting() {
    let result = GeminiGrpc::new("key\n").await;
    assert!(
        matches!(result, Err(GeminiGrpcError::InvalidApiKey(_))),
        "{result:?}"
    );
}
