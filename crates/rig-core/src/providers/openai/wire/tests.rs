//! The provider configuration and the dialect table.

use super::*;

/// A recorded request (`"when"`) or reply (`"then"`) body from a cassette
/// under `tests/cassettes/openai/`.
///
/// Hand-rolled rather than YAML-parsed because `serde_yaml` is not a
/// dev-dependency of this crate, and a cassette is never edited: the two
/// scalar forms the recorder emits — a single-quoted one-liner for a JSON
/// body, a `|+` literal block for an SSE body — are the whole grammar.
pub(super) fn recorded(section: &str, relative: &str) -> String {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/cassettes/openai/");
    let text = std::fs::read_to_string(format!("{root}{relative}"))
        .unwrap_or_else(|error| panic!("cassette {relative} is readable: {error}"));
    let (when, then) = text
        .split_once("\nthen:\n")
        .unwrap_or_else(|| panic!("cassette {relative} has a `then:` section"));
    let scope = if section == "when" { when } else { then };
    let mut lines = scope.lines();
    let body = lines
        .by_ref()
        .find(|line| line.trim_start().starts_with("body:"))
        .and_then(|line| line.split_once("body:"))
        .map(|(_, rest)| rest.trim())
        .unwrap_or_else(|| panic!("cassette {relative} names a {section} body"));
    if let Some(quoted) = body.strip_prefix('\'').and_then(|b| b.strip_suffix('\'')) {
        return quoted.replace("''", "'");
    }
    // A `|`/`|+` literal block: every following line is indented by four
    // spaces, and the block ends at the first line indented less than that.
    let mut block = String::new();
    for line in lines {
        if !line.is_empty() && !line.starts_with("    ") {
            break;
        }
        block.push_str(line.get(4..).unwrap_or_default());
        block.push('\n');
    }
    block
}

/// The recorded body, parsed as JSON.
pub(super) fn recorded_json(section: &str, relative: &str) -> serde_json::Value {
    serde_json::from_str(&recorded(section, relative))
        .unwrap_or_else(|error| panic!("cassette {relative} {section} body is JSON: {error}"))
}

/// A wire is data a host may store in a config file or a scene, so it must
/// be serializable — and serializing it must never write the credential.
#[test]
fn a_serialized_configuration_carries_no_key_material() {
    let json = serde_json::to_string(&OpenAI::new("sk-secret")).expect("the config serializes");
    assert!(!json.contains("sk-secret"), "the key leaked: {json}");
    assert!(json.contains("[redacted]"), "{json}");

    // And neither does a wire built from it, which is what a host actually
    // stores.
    let chat = serde_json::to_string(&OpenAI::new("sk-secret").chat("gpt-5.2"))
        .expect("the wire serializes");
    assert!(!chat.contains("sk-secret"), "the key leaked: {chat}");

    // `Debug` is the other way a key escapes — into a log line.
    assert!(!format!("{:?}", OpenAI::new("sk-secret")).contains("sk-secret"));
}

/// A dialect is an identity, so its wire format is its name.
#[test]
fn a_dialect_round_trips_through_its_name() {
    let json = serde_json::to_string(&GROQ).expect("a dialect serializes");
    assert_eq!(json, "\"groq\"");
    assert_eq!(
        serde_json::from_str::<Dialect>(&json).expect("a dialect deserializes"),
        GROQ
    );

    // Through a whole configuration, which is how it actually travels.
    let config = OpenAI::new("k").with_dialect(&MISTRAL);
    let restored: OpenAI =
        serde_json::from_str(&serde_json::to_string(&config).expect("serializes"))
            .expect("deserializes");
    assert_eq!(restored.dialect, MISTRAL);
    assert_eq!(restored.base_url, MISTRAL.base_url);
}

/// A name this build does not know is an error, not a half-constructed
/// provider pointed at nothing.
#[test]
fn an_unknown_dialect_name_is_rejected() {
    let error = serde_json::from_str::<Dialect>("\"not-a-provider\"")
        .expect_err("an unknown dialect is rejected");
    assert!(
        error.to_string().contains("not-a-provider"),
        "the error names the dialect: {error}"
    );
}

/// The table `Deserialize` looks names up in must contain every dialect, or
/// a stored wire would fail to load for a provider this build supports.
#[test]
fn every_dialect_is_reachable_by_name() {
    for dialect in all() {
        assert_eq!(
            by_name(dialect.name),
            Some(dialect),
            "{} is missing from the lookup table",
            dialect.name
        );
    }
}

/// Azure addresses a deployment in the URL and versions the API with a query
/// parameter; every other dialect resolves a path against its base URL.
#[test]
fn azure_routes_the_model_through_the_url() {
    let azure = OpenAI::with_key(&AZURE, "k")
        .with_base_url("https://example.openai.azure.com")
        .with_api_version("2024-10-21");
    assert_eq!(
        azure.uri("/chat/completions", Some("my-deployment")),
        "https://example.openai.azure.com/openai/deployments/my-deployment/chat/completions?api-version=2024-10-21"
    );

    let openai = OpenAI::new("k");
    assert_eq!(
        openai.uri("/chat/completions", None),
        "https://api.openai.com/v1/chat/completions"
    );
}

/// The credential goes in the header the dialect uses, and a local server
/// started without a key gets no `Authorization` header at all.
#[test]
fn the_dialect_decides_the_credential_header() {
    fn headers(provider: &OpenAI) -> http::HeaderMap {
        provider
            .authenticate(http::Request::get("https://example.invalid/"))
            .body(())
            .expect("builds")
            .headers()
            .clone()
    }

    let openai = headers(&OpenAI::new("sk-test"));
    assert_eq!(openai["authorization"], "Bearer sk-test");

    let azure = headers(&OpenAI::with_key(&AZURE, "azure-key"));
    assert_eq!(azure["api-key"], "azure-key");
    assert!(!azure.contains_key("authorization"));

    let keyless = headers(&OpenAI::with_key(&LLAMACPP, ""));
    assert!(
        !keyless.contains_key("authorization"),
        "`llama-server` rejects a request carrying a key it was not started with"
    );
    let keyed = headers(&OpenAI::with_key(&LLAMACPP, "local"));
    assert_eq!(keyed["authorization"], "Bearer local");
}

/// A dialect with no token-free credential check says so, instead of
/// verifying against an endpoint that bills the caller.
#[test]
fn a_dialect_without_a_verify_endpoint_refuses_to_invent_one() {
    use crate::wire::{Mode, Wire};

    assert!(
        OpenAI::with_key(&PERPLEXITY, "k")
            .verify_wire()
            .encode((), Mode::Unary)
            .is_err()
    );
    assert!(OpenAI::new("k").verify_wire().encode((), Mode::Unary).is_ok());
}
