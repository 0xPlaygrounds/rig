use super::*;
use rig_core::ProviderResponseError;

/// rig#2314: the AWS request id attaches to preserved provider bodies and
/// leaves Rig-authored diagnostics untouched.
#[test]
fn attach_request_id_targets_provider_responses_only() {
    let attached = attach_request_id(
        CompletionError::ProviderResponse(ProviderResponseError::without_status("aws said no")),
        Some("aws-req-1".to_string()),
    );
    assert_eq!(attached.provider_request_id(), Some("aws-req-1"));
    assert!(
        attached.to_string().contains("request id: aws-req-1"),
        "the id appears in the logged message: {attached}"
    );

    let untouched = attach_request_id(
        CompletionError::ProviderError("rig diagnostic".to_string()),
        Some("aws-req-1".to_string()),
    );
    assert_eq!(untouched.provider_request_id(), None);
}
