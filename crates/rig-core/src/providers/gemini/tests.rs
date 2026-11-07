//! The Gemini config's own tests: what it reads, and what it never writes.

use super::{API_KEY_ENV, BASE_URL, GeminiConfig};
use crate::wire::Wire;

/// A wire is data a host may serialize into a scene, a component or a config
/// file, and Gemini's key is the one credential in it. Nothing serialized may
/// carry it.
#[test]
fn a_serialized_config_carries_no_key_material() {
    let gemini = GeminiConfig::new("AIzaSyNOTAREALKEY-0123456789");
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
    let wire = gemini.completion("gemini-2.5-flash");
    let json = serde_json::to_string(&wire).expect("the wire serializes");
    assert!(
        !json.contains("AIzaSyNOTAREALKEY"),
        "the serialized wire leaked the key: {json}"
    );
    assert!(!format!("{wire:?}").contains("AIzaSyNOTAREALKEY"));

    // …and it is still the key that goes out on the request, so the
    // redaction above is a serialization property, not a lost credential.
    let request = crate::completion::CompletionRequest::new("probe");
    let encoded = wire
        .encode(request, crate::wire::Mode::Unary)
        .expect("the request encodes");
    assert_eq!(
        encoded.request.headers()[GeminiConfig::API_KEY_HEADER],
        "AIzaSyNOTAREALKEY-0123456789",
        "the encoded request lost the key"
    );
}

/// A built request is what a host logs when it debugs a call, so its `Debug`
/// must hide the key the header still sends. An unusable key stays an error
/// from the builder.
#[test]
fn a_debug_printed_request_hides_the_key_it_sends() {
    const KEY: &str = "AIzaSyNOTAREALKEY-0123456789";
    let gemini = GeminiConfig::new(KEY);
    let probe = || crate::completion::CompletionRequest::new("probe");
    let requests = [
        gemini
            .completion("gemini-3.8-flash")
            .encode(probe(), crate::wire::Mode::Unary)
            .expect("the request encodes")
            .request,
        gemini
            .interactions("gemini-3.8-flash")
            .encode(probe(), crate::wire::Mode::Unary)
            .expect("the request encodes")
            .request,
    ];
    for request in requests {
        let sent = request
            .headers()
            .get(GeminiConfig::API_KEY_HEADER)
            .expect("the request carries the key header");
        assert_eq!(sent.as_bytes(), KEY.as_bytes(), "the header lost the key");
        assert!(
            !format!("{request:?}").contains(KEY),
            "the debug-printed request leaked the key"
        );
    }

    let unusable = GeminiConfig::new("not\na header value")
        .request(http::Method::GET, "/v1beta/models")
        .body(());
    assert!(unusable.is_err(), "an unusable key must not be dropped");
}

/// The one variable the client layer read, unchanged: a host's existing
/// environment keeps working.
#[test]
fn from_env_reads_the_documented_variable() {
    assert_eq!(API_KEY_ENV, "GEMINI_API_KEY");
}
