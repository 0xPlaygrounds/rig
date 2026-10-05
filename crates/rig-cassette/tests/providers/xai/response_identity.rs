//! Response identity metadata (rig#2265): xAI reports `x-request-id` on its
//! Responses-shaped API; blocking and streaming turns carry it identically.

use rig::providers::xai;

use rig::completion::CompletionRequest;

/// 401 auth rejection (rig#2314 error matrix): contract classification holds
/// on the auth tier; the recording documents whether xAI's auth tier sends
/// the id it omits on 4xx.
#[tokio::test]
async fn auth_rejection_classifies_with_contract() {
    use super::support::with_xai_cassette_bogus_key;

    with_xai_cassette_bogus_key(
        "response_identity/auth_rejection_classifies_with_contract",
        |client| async move {
            let model = client.completion(xai::GROK_3_MINI);
            let error = model
                .call(CompletionRequest::new("Never authenticated"))
                .await
                .expect_err("a bogus key must be rejected");
            assert!(
                matches!(error, rig::error::ProviderError::ProviderResponse(_)),
                "got {error:?}"
            );
            // Derived from the recording.
            let _ = error.provider_request_id();
        },
    )
    .await;
}
