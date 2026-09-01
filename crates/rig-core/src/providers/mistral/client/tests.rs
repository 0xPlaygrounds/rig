use super::Usage;

#[test]
fn test_client_initialization() {
    let _client = crate::providers::mistral::Client::new_with(
        "dummy-key",
        crate::test_utils::RecordingHttpClient::new(""),
    )
    .expect("Client::new() failed");
    let builder: crate::providers::mistral::ClientBuilder =
        crate::providers::mistral::Client::builder().api_key("dummy-key");
    let _client_from_builder = builder
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("Client::builder() failed");
}

#[test]
fn usage_retains_live_service_tier() {
    let usage: Usage = serde_json::from_value(serde_json::json!({
        "completion_tokens": 4,
        "prompt_tokens": 20,
        "total_tokens": 24,
        "prompt_tokens_details": { "cached_tokens": 0 },
        "service_tier": "standard"
    }))
    .expect("live Mistral usage should deserialize");

    assert_eq!(usage.service_tier.as_deref(), Some("standard"));
    assert_eq!(
        serde_json::to_value(usage).expect("Mistral usage should serialize")["service_tier"],
        "standard"
    );
}
