//! The Gemini config's own tests: what it reads, and what it never writes.

use super::{API_KEY_ENV, BASE_URL, EMBEDDING_001, Gemini};
use crate::completion::{CompletionError, CompletionRequest};
use crate::driver::HasVerify;
use crate::wire::{Mode, Wire};

/// A wire is data a host may serialize into a scene, a component or a config
/// file, and Gemini's key is the one credential in it — including on the
/// GenerateContent family, whose `encode` puts the key in the request URI.
/// Nothing serialized may carry it.
#[test]
fn a_serialized_config_carries_no_key_material() {
    let gemini = Gemini::new("AIzaSyNOTAREALKEY-0123456789");
    crate::wire::secret::tests::a_config_reloads_without_its_credential(
        &gemini,
        "AIzaSyNOTAREALKEY-0123456789",
        |gemini| &gemini.api_key,
    );
    assert_eq!(
        serde_json::to_string(&gemini).expect("the config serializes"),
        format!(r#"{{"api_key":"[redacted]","base_url":"{BASE_URL}"}}"#)
    );

    // The wires built from it are data too, and the completion wire is the
    // one a host is most likely to store.
    let wire = gemini.generate_content("gemini-2.5-flash");
    let json = serde_json::to_string(&wire).expect("the wire serializes");
    assert!(
        !json.contains("AIzaSyNOTAREALKEY"),
        "the serialized wire leaked the key: {json}"
    );
    assert!(!format!("{wire:?}").contains("AIzaSyNOTAREALKEY"));

    // …and it is still the key that goes out on the request, so the
    // redaction above is a serialization property, not a lost credential.
    let encoded = wire
        .encode(probe_request(), Mode::Unary)
        .expect("the request encodes");
    let uri = match encoded.requests.as_slice() {
        [request] => request.uri().clone(),
        requests => panic!("expected one request, got {}", requests.len()),
    };
    assert!(
        uri.query()
            .is_some_and(|query| query.contains("key=AIzaSyNOTAREALKEY-0123456789")),
        "the encoded request lost the key: {uri}"
    );
}

/// The one variable the client layer read, unchanged: a host's existing
/// environment keeps working.
#[test]
fn from_env_reads_the_documented_variable() {
    assert_eq!(API_KEY_ENV, "GEMINI_API_KEY");
}

/// A minimal request, for a test that reads only the request envelope.
fn probe_request() -> CompletionRequest {
    CompletionRequest {
        model: None,
        chat_history: vec!["probe".into()],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    }
}

/// Gemini has no anonymous mode — every endpoint spends the key, in the
/// query or in a header — and a config reloaded from a scene or a config
/// file holds none by contract. So each surface refuses in `encode`, naming
/// the variable that supplies one, and no request carrying an empty `key=`
/// is ever built.
#[test]
fn a_config_without_a_key_refuses_to_encode_wherever_the_key_is_spent() {
    let gemini = Gemini::new("");

    let completion = gemini
        .generate_content("gemini-2.5-flash")
        .encode(probe_request(), Mode::Unary)
        .expect_err("a GenerateContent request with no key is refused");
    assert!(
        matches!(&completion, CompletionError::MissingCredential { env_var } if *env_var == API_KEY_ENV),
        "{completion:?}"
    );
    assert!(completion.to_string().contains(API_KEY_ENV), "{completion}");

    // The Interactions family authenticates by header rather than by query
    // and goes through the same seam, so it refuses the same way.
    let interactions = gemini
        .interactions("gemini-2.5-flash")
        .encode(probe_request(), Mode::Streaming)
        .expect_err("an Interactions request with no key is refused");
    assert!(
        matches!(&interactions, CompletionError::MissingCredential { env_var } if *env_var == API_KEY_ENV),
        "{interactions:?}"
    );

    // Every other surface refuses too, each through its own operation's
    // error type, and each message names the variable.
    let embedding = gemini
        .embeddings(EMBEDDING_001, None)
        .encode(vec!["probe".to_owned()], Mode::Unary)
        .expect_err("an embedding request with no key is refused");
    assert!(embedding.to_string().contains(API_KEY_ENV), "{embedding}");

    let listing = gemini
        .models()
        .encode((), Mode::Unary)
        .expect_err("a listing with no key is refused");
    assert!(listing.to_string().contains(API_KEY_ENV), "{listing}");

    let interactions_listing = gemini
        .interactions_models()
        .encode((), Mode::Unary)
        .expect_err("an Interactions listing with no key is refused");
    assert!(
        interactions_listing.to_string().contains(API_KEY_ENV),
        "{interactions_listing}"
    );

    let verify = gemini
        .verify()
        .encode((), Mode::Unary)
        .expect_err("a key check with no key is refused");
    assert!(verify.to_string().contains(API_KEY_ENV), "{verify}");
}

/// The seam only *checks*: a configured key lands byte for byte where the
/// recorded traffic has it — last in the query for the GenerateContent
/// family, in `x-goog-api-key` with no query credential at all for the
/// Interactions family.
#[test]
fn a_configured_key_reaches_the_request_exactly_where_it_did_before() {
    let gemini = Gemini::new("test-key");
    let wire = gemini.generate_content("gemini-2.5-flash");

    let unary = wire
        .encode(probe_request(), Mode::Unary)
        .expect("the request encodes");
    assert_eq!(
        unary
            .requests
            .first()
            .expect("an encode produces a request")
            .uri()
            .to_string(),
        format!("{BASE_URL}/v1beta/models/gemini-2.5-flash:generateContent?key=test-key")
    );

    let streamed = wire
        .encode(probe_request(), Mode::Streaming)
        .expect("the streamed request encodes");
    assert_eq!(
        streamed
            .requests
            .first()
            .expect("an encode produces a request")
            .uri()
            .to_string(),
        format!(
            "{BASE_URL}/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse&key=test-key"
        )
    );

    let interactions = gemini
        .interactions("gemini-2.5-flash")
        .encode(probe_request(), Mode::Unary)
        .expect("the Interactions request encodes");
    let request = interactions
        .requests
        .first()
        .expect("an encode produces a request");
    assert_eq!(
        request.uri().to_string(),
        format!("{BASE_URL}/v1beta/interactions")
    );
    assert_eq!(
        request
            .headers()
            .get(Gemini::INTERACTIONS_KEY_HEADER)
            .and_then(|value| value.to_str().ok()),
        Some("test-key")
    );
}
