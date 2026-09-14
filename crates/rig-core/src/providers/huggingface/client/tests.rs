use super::SubProvider;

#[test]
fn test_client_initialization() {
    let _client = crate::providers::huggingface::Client::new_with(
        "dummy-key",
        crate::test_utils::RecordingHttpClient::new(""),
    )
    .expect("Client::new() failed");
    let _client_from_builder = crate::providers::huggingface::Client::builder()
        .api_key("dummy-key")
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("Client::builder() failed");
}

#[test]
fn fireworks_model_identifier_is_idempotent() {
    // A bare id is qualified once...
    assert_eq!(
        SubProvider::Fireworks.model_identifier("deepseek-v3"),
        "accounts/fireworks/models/deepseek-v3"
    );
    // ...and an already-qualified id (e.g. a per-request model override)
    // is left untouched rather than double-prefixed.
    assert_eq!(
        SubProvider::Fireworks.model_identifier("accounts/fireworks/models/deepseek-v3"),
        "accounts/fireworks/models/deepseek-v3"
    );
    // Other sub-providers pass the id through verbatim.
    assert_eq!(
        SubProvider::HFInference.model_identifier("meta-llama/Llama-3.1-8B"),
        "meta-llama/Llama-3.1-8B"
    );
}
