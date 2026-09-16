//! The Gemini config's own tests: what it reads, and what it never writes.

use super::{API_KEY_ENV, BASE_URL, Gemini};
use crate::wire::Wire;

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
    let request = crate::completion::CompletionRequest {
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
    };
    let encoded = wire
        .encode(request, crate::wire::Mode::Unary)
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
