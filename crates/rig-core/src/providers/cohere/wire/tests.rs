use super::*;
use crate::test_utils::json_body;
use crate::wire::secret::tests::a_config_reloads_without_its_credential;
use crate::wire::{Mode, Wire};

fn cohere() -> CohereConfig {
    CohereConfig::new("cohere-test-key")
}

/// Chat goes to the Compatibility API under the configured root, with the
/// key as a bearer token.
#[test]
fn chat_addresses_the_compatibility_api() {
    let wire = cohere()
        .with_base_url("http://127.0.0.1:9/")
        .completion("command-a-03-2025");
    let encoded = wire
        .encode(crate::completion::CompletionRequest::new("hi"), Mode::Unary)
        .expect("the request encodes");
    assert_eq!(
        encoded.request.uri().to_string(),
        "http://127.0.0.1:9/compatibility/v1/chat/completions"
    );
    assert_eq!(
        encoded.request.headers()[http::header::AUTHORIZATION],
        "Bearer cohere-test-key"
    );
    assert_eq!(json_body(&encoded.request)["model"], "command-a-03-2025");
}

#[test]
fn a_serialized_config_carries_no_key_material() {
    a_config_reloads_without_its_credential(&cohere(), "cohere-test-key", |cohere| &cohere.api_key);

    let wire = cohere().completion("command-a-03-2025");
    let serialized = serde_json::to_string(&wire).expect("the wire serializes");
    assert!(
        !serialized.contains("cohere-test-key"),
        "a wire a host may persist must not carry the credential: {serialized}"
    );
}
