//! Doubleword request encoding without provider I/O. No Doubleword Responses
//! recording or live credential is available for a flex-tier cassette.

use crate::completion::{CompletionRequest, GenerationOptions, OnUnsupported, ServiceTier};
use crate::error::ProviderError;
use crate::operation::Completion;
use crate::providers::doubleword;
use crate::test_utils::json_body;
use crate::wire::{Encoded, Mode, Operation, Wire};
use serde_json::{Value, json};

const PROMPT: &str = "Summarize the history of artificial intelligence.";

fn request(options: GenerationOptions) -> CompletionRequest {
    CompletionRequest::new(PROMPT).options(options)
}

fn encoded(
    wire: &impl Wire<Op = Completion, Payload = Encoded>,
    options: GenerationOptions,
) -> Result<Encoded, ProviderError> {
    let request = Completion::prepare(request(options), &wire.describe())?;
    Ok(wire.encode(request, Mode::Unary)?)
}

fn assert_refused(result: Result<Encoded, ProviderError>, option: &str) {
    assert!(
        matches!(
            &result,
            Err(ProviderError::UnsupportedOption(refusal))
                if refusal.option.as_ref() == option
                    && refusal.provider == doubleword::PROVIDER_NAME
                    && refusal.model == doubleword::QWEN3_5_9B
        ),
        "expected Doubleword to refuse {option}, got {result:?}"
    );
}

/// Checks local endpoint and typed-option encoding because no live Doubleword
/// credential or Responses cassette is available.
#[test]
fn responses_flex_uses_doubleword_endpoint_without_background() {
    let client = doubleword::new("test-key");
    let wire = client.responses(doubleword::QWEN3_5_9B).wire;
    let encoded = encoded(
        &wire,
        GenerationOptions::default().service_tier(ServiceTier::Flex),
    )
    .expect("Doubleword Responses accepts the documented flex tier");
    let body = json_body(&encoded.request);

    assert_eq!(encoded.request.method(), http::Method::POST);
    assert_eq!(
        encoded.request.uri().to_string(),
        "https://api.doubleword.ai/v1/responses"
    );
    assert_eq!(
        body.get("service_tier").and_then(Value::as_str),
        Some("flex")
    );
    assert_eq!(
        body.get("model").and_then(Value::as_str),
        Some(doubleword::QWEN3_5_9B)
    );
    assert_eq!(
        body.pointer("/input/0/content/0/text")
            .and_then(Value::as_str),
        Some(PROMPT)
    );
    assert!(body.get("background").is_none(), "{body}");
}

/// Compares local request encodings because no live Doubleword credential or
/// Responses cassette is available.
#[test]
fn responses_flex_adds_only_the_tier_to_the_default_request() {
    let wire = doubleword::new("test-key")
        .responses(doubleword::QWEN3_5_9B)
        .wire;
    let baseline = encoded(&wire, GenerationOptions::default())
        .expect("default Doubleword Responses request encodes");
    let mut expected = json_body(&baseline.request);
    assert!(expected.get("service_tier").is_none(), "{expected}");
    assert!(expected.get("background").is_none(), "{expected}");
    expected
        .as_object_mut()
        .expect("a Responses request is an object")
        .insert("service_tier".to_owned(), json!("flex"));
    let flex = encoded(
        &wire,
        GenerationOptions::default().service_tier(ServiceTier::Flex),
    )
    .expect("flex Doubleword Responses request encodes");

    assert_eq!(json_body(&flex.request), expected);
}

/// Unsupported-tier refusals happen before provider I/O and require no cassette.
#[test]
fn responses_refuses_undocumented_tiers() {
    let wire = doubleword::new("test-key")
        .responses(doubleword::QWEN3_5_9B)
        .wire;
    for tier in [
        ServiceTier::Auto,
        ServiceTier::Default,
        ServiceTier::Priority,
    ] {
        assert_refused(
            encoded(&wire, GenerationOptions::default().service_tier(tier)),
            "service_tier",
        );
    }
}

/// Unsupported-option refusals happen before provider I/O and require no cassette.
#[test]
fn responses_flex_does_not_enable_unverified_options() {
    let wire = doubleword::new("test-key")
        .responses(doubleword::QWEN3_5_9B)
        .wire;
    assert_refused(
        encoded(
            &wire,
            GenerationOptions::default()
                .service_tier(ServiceTier::Flex)
                .seed(42),
        ),
        "seed",
    );
}

/// Checks the local unsupported-option policy because no live Doubleword
/// credential or Responses cassette is available.
#[test]
fn responses_ignore_keeps_flex_while_dropping_unverified_options() {
    let wire = doubleword::new("test-key")
        .responses(doubleword::QWEN3_5_9B)
        .wire;
    let encoded = encoded(
        &wire,
        GenerationOptions::default()
            .service_tier(ServiceTier::Flex)
            .seed(42)
            .on_unsupported(OnUnsupported::Ignore),
    )
    .expect("ignoring an unsupported seed keeps the supported flex tier");
    let body = json_body(&encoded.request);

    assert_eq!(
        body.get("service_tier").and_then(Value::as_str),
        Some("flex")
    );
    assert!(body.get("seed").is_none(), "{body}");
}

/// Checks local route selection and refusal before provider I/O; no cassette is
/// needed to establish the existing Chat Completions contract.
#[test]
fn chat_keeps_default_route_and_refuses_flex() {
    let client = doubleword::new("test-key");
    let default = client.completion(doubleword::QWEN3_5_9B).wire;
    let baseline = encoded(&default, GenerationOptions::default())
        .expect("default Doubleword completion request encodes");

    assert_eq!(
        baseline.request.uri().to_string(),
        "https://api.doubleword.ai/v1/chat/completions"
    );
    assert!(json_body(&baseline.request).get("service_tier").is_none());
    assert_refused(
        encoded(
            &client.chat(doubleword::QWEN3_5_9B).wire,
            GenerationOptions::default().service_tier(ServiceTier::Flex),
        ),
        "service_tier",
    );
}
