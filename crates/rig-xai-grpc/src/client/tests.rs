use super::{XaiGrpc, XaiGrpcError};

/// A key that cannot travel as gRPC metadata is rejected before any
/// connection is attempted.
#[tokio::test]
async fn new_rejects_unsendable_api_key_without_connecting() {
    let result = XaiGrpc::new("key\n").await;
    assert!(
        matches!(result, Err(XaiGrpcError::InvalidApiKey(_))),
        "{result:?}"
    );
}
