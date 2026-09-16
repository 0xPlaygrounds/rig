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

/// Azure accepts an account key *or* an Entra bearer token, and they are not
/// two spellings of one credential: the key goes out as `api-key`, the token
/// as `Authorization: Bearer`.
#[test]
fn azure_accepts_either_credential_under_its_own_header() {
    fn headers(provider: &OpenAI) -> http::HeaderMap {
        provider
            .authenticate(http::Request::get("https://example.invalid/"))
            .body(())
            .expect("builds")
            .headers()
            .clone()
    }

    // The dialect names the alternative, which is what `from_env_with` reads
    // when the primary variable is unset.
    let alternative = AZURE
        .alternate_auth
        .expect("azure accepts a second credential");
    assert_eq!(alternative.api_key_env, "AZURE_TOKEN");
    assert_eq!(alternative.auth, Auth::Bearer);
    assert_eq!(AZURE.api_key_env, "AZURE_API_KEY");

    let keyed = headers(&OpenAI::with_key(&AZURE, "account-key"));
    assert_eq!(keyed["api-key"], "account-key");
    assert!(!keyed.contains_key("authorization"));

    let token = headers(&OpenAI::with_alternate_key(&AZURE, "entra-token"));
    assert_eq!(token["authorization"], "Bearer entra-token");
    assert!(
        !token.contains_key("api-key"),
        "an Entra token is not an account key"
    );
}

/// Hugging Face's router picks a sub-provider, and the choice is observable:
/// Fireworks addresses models by a qualified id, and only the default
/// sub-provider serves the endpoints that put the model in the URL.
#[test]
fn the_huggingface_sub_route_decides_the_model_and_the_routes() {
    assert_eq!(
        SubRoute::Fireworks.model_identifier("llama-3.3-70b"),
        "accounts/fireworks/models/llama-3.3-70b"
    );
    // Idempotent: the rewrite runs on the resolved request model, so an
    // already-qualified per-request override must not be prefixed twice.
    assert_eq!(
        SubRoute::Fireworks.model_identifier("accounts/fireworks/models/llama-3.3-70b"),
        "accounts/fireworks/models/llama-3.3-70b"
    );
    assert_eq!(
        SubRoute::Together.model_identifier("llama-3.3-70b"),
        "llama-3.3-70b"
    );

    // The slugs the router routes by.
    assert_eq!(SubRoute::HFInference.slug(), "hf-inference/models");
    assert_eq!(SubRoute::Fireworks.slug(), "fireworks-ai");
    assert_eq!(SubRoute::from("my-route").slug(), "my-route");

    // `None` behaves as the router's own default, which is the only
    // sub-provider that serves the model-routed endpoints.
    let default = OpenAI::with_key(&HUGGINGFACE, "hf");
    assert_eq!(
        default
            .modality_uri("transcription", "/audio/transcriptions", "openai/whisper-large-v3")
            .expect("hf-inference serves transcription"),
        "https://router.huggingface.co/openai/whisper-large-v3"
    );

    let routed = default.clone().with_sub_route(SubRoute::Together);
    let error = routed
        .modality_uri("transcription", "/audio/transcriptions", "whisper")
        .expect_err("only hf-inference serves transcription");
    assert_eq!(error, "transcription endpoint is not supported yet for together");
    assert!(
        routed
            .modality_uri("image generation", "/images/generations", "sd")
            .is_err()
    );

    // A dialect that does not route keeps its fixed path.
    assert_eq!(
        OpenAI::new("k")
            .modality_uri("transcription", "/audio/transcriptions", "whisper-1")
            .expect("openai serves transcription"),
        "https://api.openai.com/v1/audio/transcriptions"
    );
}

/// A dialect that offers no reranking says so, rather than posting to a path
/// its server never served.
#[test]
fn only_a_dialect_with_a_rerank_path_reranks() {
    use crate::operation::RerankRequest;
    use crate::wire::{Mode, Wire};

    let request = || RerankRequest {
        query: "q".to_owned(),
        documents: vec!["a".to_owned(), "b".to_owned()],
    };
    assert!(
        OpenAI::new("k")
            .reranker("any")
            .encode(request(), Mode::Unary)
            .is_err(),
        "OpenAI has no reranking endpoint"
    );
    assert!(
        OpenAI::with_key(&LLAMACPP, "")
            .reranker("bge-reranker-v2-m3")
            .encode(request(), Mode::Unary)
            .is_ok()
    );
}
