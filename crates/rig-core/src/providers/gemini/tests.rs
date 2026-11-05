//! The Gemini config's own tests: what it reads, and what it never writes.

use super::{API_KEY_ENV, Auth, BASE_URL, GeminiConfig};
use crate::wire::Wire;

/// A wire is data a host may serialize into a scene, a component or a config
/// file, and Gemini's key is the one credential in it — including on the
/// GenerateContent family, whose `encode` puts the key in the request URI.
/// Nothing serialized may carry it.
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
    let uri = encoded.request.uri().clone();
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

/// The official SDKs put the key in a header. That placement keeps it out of
/// every URL a proxy or log may retain, and a gateway in front of Gemini may
/// accept only it.
#[test]
fn header_auth_sends_the_key_out_of_the_uri() {
    let gemini = GeminiConfig::new("AIzaSyNOTAREALKEY-0123456789").with_auth(Auth::Header);
    let wire = gemini.completion("gemini-2.5-flash");
    let encoded = wire
        .encode(
            crate::completion::CompletionRequest::new("probe"),
            crate::wire::Mode::Unary,
        )
        .expect("the request encodes");
    let request = &encoded.request;

    assert_eq!(
        request.uri().path(),
        "/v1beta/models/gemini-2.5-flash:generateContent"
    );
    assert_eq!(request.uri().query(), None);
    assert_eq!(
        request
            .headers()
            .get(GeminiConfig::API_KEY_HEADER)
            .and_then(|value| value.to_str().ok()),
        Some("AIzaSyNOTAREALKEY-0123456789"),
    );
}

/// A query-auth request carries the key once, in the URI, and never also in
/// the header.
#[test]
fn query_auth_keeps_the_key_out_of_the_header() {
    let gemini = GeminiConfig::new("AIzaSyNOTAREALKEY-0123456789");
    assert_eq!(gemini.auth, Auth::Query);
    let encoded = gemini
        .completion("gemini-2.5-flash")
        .encode(
            crate::completion::CompletionRequest::new("probe"),
            crate::wire::Mode::Unary,
        )
        .expect("the request encodes");

    assert!(
        encoded
            .request
            .uri()
            .query()
            .is_some_and(|query| query.contains("key=AIzaSyNOTAREALKEY-0123456789"))
    );
    assert!(
        encoded
            .request
            .headers()
            .get(GeminiConfig::API_KEY_HEADER)
            .is_none()
    );
}

/// The default placement stays absent from serialized configs so existing
/// configs round-trip byte for byte; a changed placement is named, and still
/// carries no key.
#[test]
fn auth_placement_serializes_without_key_material() {
    let default_json = serde_json::to_string(&GeminiConfig::new("AIzaSyNOTAREALKEY-0123456789"))
        .expect("the config serializes");
    assert!(!default_json.contains("auth"));

    let header_json = serde_json::to_string(
        &GeminiConfig::new("AIzaSyNOTAREALKEY-0123456789").with_auth(Auth::Header),
    )
    .expect("the config serializes");
    assert!(header_json.contains(r#""auth":"header""#));
    assert!(!header_json.contains("AIzaSyNOTAREALKEY"));

    let reloaded: GeminiConfig =
        serde_json::from_str(&header_json).expect("the config deserializes");
    assert_eq!(reloaded.auth, Auth::Header);
}
