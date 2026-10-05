use crate::completion::rpc_error;

/// The embedding wire classifies its RPC failures by the same rule as the
/// completion wire: the code is kept, and only the transient codes retry.
#[test]
fn embedding_rpc_errors_carry_the_code_and_its_verdict() {
    let busy = rpc_error(&tonic::Status::resource_exhausted("quota"));
    assert!(busy.is_retryable());
    assert_eq!(
        rig_core::error::ErrorReport::from(&busy).code.as_deref(),
        Some("RESOURCE_EXHAUSTED")
    );
    let bad = rpc_error(&tonic::Status::invalid_argument("dims"));
    assert!(!bad.is_retryable());
    assert_eq!(
        rig_core::error::ErrorReport::from(&bad).code.as_deref(),
        Some("INVALID_ARGUMENT")
    );
    assert_eq!(bad.provider_response_status(), None);
}
