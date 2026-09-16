//! [`Secret`]'s own tests, and the round-trip legs every provider's
//! redaction test shares.
//!
//! The round-trip contract is one property restated per provider
//! configuration, so it is written once here and *called* with a config
//! rather than copied into each provider's `tests.rs`.

use super::{REDACTED, Secret};
use crate::completion::CompletionRequest;
use crate::operation::Completion;
use crate::providers::anthropic::wire::Anthropic;
use crate::providers::cohere::wire::Cohere;
use crate::providers::gemini::Gemini;
use crate::providers::openai::wire::OpenAI;
use crate::wire::{Mode, Wire};

#[test]
fn a_secret_never_renders_or_serializes_its_value() {
    let secret = Secret::from("sk-live-do-not-print");
    assert_eq!(format!("{secret:?}"), "[redacted]");
    assert_eq!(
        serde_json::to_string(&secret).expect("a string serializes"),
        "\"[redacted]\""
    );
    assert_eq!(secret.expose(), "sk-live-do-not-print");
}

#[test]
fn secrets_compare_by_value() {
    assert_eq!(Secret::from("k"), Secret::from("k"));
    assert_ne!(Secret::from("k"), Secret::from("j"));
}

#[test]
fn a_secret_deserializes_from_a_bare_string() {
    let secret: Secret = serde_json::from_str("\"k\"").expect("a string deserializes");
    assert_eq!(secret.expose(), "k");
}

/// The sentinel is not a credential, so it must not reload as one: a wire a
/// host stored and loaded back carries no key, which is the state
/// `is_empty` reports and `from_env` re-hydrates.
#[test]
fn the_redaction_sentinel_reloads_as_no_credential() {
    let stored =
        serde_json::to_string(&Secret::from("sk-live-do-not-print")).expect("a secret serializes");
    let reloaded: Secret = serde_json::from_str(&stored).expect("the sentinel deserializes");
    assert!(reloaded.is_empty());
    assert_eq!(reloaded, Secret::default());
}

/// The one property behind every provider's redaction test: whatever a
/// reloaded wire puts in its request envelope, it is never the sentinel
/// wearing a credential's clothes — as a bearer token, as `x-api-key`, or
/// in a `?key=` query.
#[test]
fn a_reloaded_wire_sends_no_credential_sentinel() {
    sends_no_sentinel(
        &OpenAI::new("sk-bearer-key").chat("gpt-5.2"),
        "sk-bearer-key",
    );
    sends_no_sentinel(
        &OpenAI::new("sk-bearer-key").responses("gpt-5.2"),
        "sk-bearer-key",
    );
    sends_no_sentinel(
        &Anthropic::new("sk-header-key").messages("claude-haiku-4-5"),
        "sk-header-key",
    );
    sends_no_sentinel(
        &Gemini::new("AIzaSyQUERY-KEY").generate_content("gemini-2.5-flash"),
        "AIzaSyQUERY-KEY",
    );
    sends_no_sentinel(
        &Cohere::new("cohere-bearer-key").chat("command-a-03-2025"),
        "cohere-bearer-key",
    );

    // Not vacuous: a wire whose credential *is* the sentinel does send it,
    // which is exactly what a transparent `Deserialize` reloaded.
    assert!(request_envelope(&OpenAI::new(REDACTED).chat("gpt-5.2")).contains("Bearer [redacted]"));
}

/// A minimal request, for a test that only reads the request envelope.
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

/// Every URI and header one encode produced, as one searchable string.
pub(crate) fn request_envelope<W: Wire<Op = Completion>>(wire: &W) -> String {
    let encoded = wire
        .encode(probe_request(), Mode::Unary)
        .expect("the request encodes");
    encoded
        .requests
        .iter()
        .map(|request| {
            let headers: String = request
                .headers()
                .iter()
                .map(|(name, value)| format!("{name}: {}\n", value.to_str().unwrap_or_default()))
                .collect();
            format!("{}\n{headers}", request.uri())
        })
        .collect()
}

/// Assert `config` writes `key` nowhere, and that reloading what it *does*
/// write leaves no credential at all — not the sentinel, which is
/// indistinguishable from a key on the way back out.
pub(crate) fn a_config_reloads_without_its_credential<C>(
    config: &C,
    key: &str,
    credential: impl Fn(&C) -> &Secret,
) where
    C: std::fmt::Debug + serde::Serialize + serde::de::DeserializeOwned,
{
    assert!(
        !credential(config).is_empty(),
        "the fixture must carry a credential for this to prove anything"
    );
    let json = serde_json::to_string(config).expect("a config serializes");
    assert!(
        !json.contains(key),
        "a config a host may persist must not carry the credential: {json}"
    );
    assert!(json.contains(REDACTED), "{json}");
    assert!(
        !format!("{config:?}").contains(key),
        "`Debug` leaked the credential"
    );

    let reloaded: C = serde_json::from_str(&json).expect("a config reloads");
    assert!(
        credential(&reloaded).is_empty(),
        "a reloaded config must report no credential, not the sentinel: {reloaded:?}"
    );
}

/// Assert the wire sends `key`, and that the same wire reloaded from its own
/// serialized form sends the sentinel nowhere.
fn sends_no_sentinel<W>(wire: &W, key: &str)
where
    W: Wire<Op = Completion> + serde::Serialize + serde::de::DeserializeOwned,
{
    let sent = request_envelope(wire);
    assert!(
        sent.contains(key),
        "the wire never sent the credential, so reloading it proves nothing: {sent}"
    );

    let json = serde_json::to_string(wire).expect("a wire serializes");
    let reloaded: W = serde_json::from_str(&json).expect("a wire reloads");
    let sent = request_envelope(&reloaded);
    // `[redacted]` percent-encodes its brackets in a query, never its body.
    assert!(
        !sent.contains("redacted"),
        "a reloaded wire sent the redaction sentinel as a credential: {sent}"
    );
}
