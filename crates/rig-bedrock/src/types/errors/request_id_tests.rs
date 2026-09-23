use super::*;
use rig_core::ProviderResponseError;

/// rig#2314: the AWS request id attaches to preserved provider bodies and
/// leaves Rig-authored diagnostics untouched.
#[test]
fn a_request_id_attaches_to_provider_responses_only() {
    let attached =
        ProviderError::ProviderResponse(ProviderResponseError::without_status("aws said no"))
            .with_provider_request_id(Some("aws-req-1".to_string()));
    assert_eq!(attached.provider_request_id(), Some("aws-req-1"));
    assert!(
        attached.to_string().contains("request id: aws-req-1"),
        "the id appears in the logged message: {attached}"
    );

    let untouched = ProviderError::Provider("rig diagnostic".to_string())
        .with_provider_request_id(Some("aws-req-1".to_string()));
    assert_eq!(untouched.provider_request_id(), None);
}
