//! The provider configuration and the dialect table.

use super::*;
use crate::wire::secret::tests::a_config_reloads_without_its_credential;

/// A recorded request (`"when"`) or reply (`"then"`) body from a cassette
/// under `crates/rig-cassette/fixtures/cassettes/openai/`.
///
/// Hand-rolled rather than YAML-parsed because `serde_yaml` is not a
/// dev-dependency of this crate, and a cassette is never edited: the two
/// scalar forms the recorder emits — a single-quoted one-liner for a JSON
/// body, a `|+` literal block for an SSE body — are the whole grammar.
pub(super) fn recorded(section: &str, relative: &str) -> String {
    let root = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../rig-cassette/fixtures/cassettes/openai/"
    );
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
    a_config_reloads_without_its_credential(&OpenAI::new("sk-secret"), "sk-secret", |openai| {
        &openai.api_key
    });

    // And neither does a wire built from it, which is what a host actually
    // stores — on either completion endpoint.
    for wire in [
        serde_json::to_string(&OpenAI::new("sk-secret").chat("gpt-5.2")),
        serde_json::to_string(&OpenAI::new("sk-secret").responses("gpt-5.2")),
    ] {
        let json = wire.expect("the wire serializes");
        assert!(!json.contains("sk-secret"), "the key leaked: {json}");
    }
}

/// The gateway-specific fields of the one configuration — the account, the
/// instructions, the caller identity a Responses gateway asks for — travel
/// with it: a stored ChatGPT wire loads back as the same wire, minus the
/// credential.
#[test]
fn a_gateway_configuration_round_trips_without_its_credential() {
    let chatgpt = crate::providers::chatgpt::DIALECT;
    let config = OpenAI::with_key(&chatgpt, "tok-secret").with_account_id("acct-1");
    assert_eq!(
        config.instructions.as_deref(),
        chatgpt.quirks.default_instructions
    );
    assert!(config.identity.is_some());

    let json = serde_json::to_string(&config).expect("serializes");
    assert!(!json.contains("tok-secret"), "the token leaked: {json}");
    let restored: OpenAI = serde_json::from_str(&json).expect("deserializes");
    assert_eq!(restored.dialect, chatgpt);
    assert_eq!(restored.account_id.as_deref(), Some("acct-1"));
    assert_eq!(restored.instructions, config.instructions);
    assert_eq!(restored.identity, config.identity);
    assert!(restored.api_key.is_empty());

    let wire = config.responses("gpt-5.4");
    let restored: super::super::responses_api::wire::Responses =
        serde_json::from_str(&serde_json::to_string(&wire).expect("serializes"))
            .expect("deserializes");
    assert_eq!(restored.system_instructions, wire.system_instructions);
    assert_eq!(restored.provider.dialect, chatgpt);
}

/// The default completion wire is the dialect's flagship route, and it is
/// data like the two wires it chooses between: it stores without the
/// credential and loads back onto the same route.
#[test]
fn the_default_completion_wire_is_the_dialects_route_and_round_trips() {
    use crate::wire::Wire as _;
    let openai = OpenAI::new("sk-secret").completion("gpt-5.2");
    let groq = OpenAI::with_key(&GROQ, "gsk-secret").completion("llama-3.3-70b-versatile");
    assert!(matches!(openai, OpenAiWire::Responses(_)), "{openai:?}");
    assert!(matches!(groq, OpenAiWire::Chat(_)), "{groq:?}");
    assert_eq!(openai.route(), Some("/responses"));
    assert_eq!(groq.route(), Some("/chat/completions"));

    for (wire, secret) in [(openai, "sk-secret"), (groq, "gsk-secret")] {
        let json = serde_json::to_string(&wire).expect("the wire serializes");
        assert!(!json.contains(secret), "the key leaked: {json}");
        let restored: OpenAiWire = serde_json::from_str(&json).expect("the wire deserializes");
        assert_eq!(restored.model(), wire.model());
        assert_eq!(restored.provider().dialect, wire.provider().dialect);
        assert_eq!(
            std::mem::discriminant(&restored),
            std::mem::discriminant(&wire)
        );
    }
}

/// The endpoint is configuration: a route chosen once on the configuration
/// overrides the dialect's flagship for every completion it builds, and it
/// is stored beside the dialect rather than inside it — a dialect
/// serializes as its name alone, so an override written into its quirks
/// would not survive a round trip.
#[test]
fn a_configured_route_overrides_the_dialects_and_round_trips() {
    use crate::wire::Wire as _;
    let on_chat = OpenAI::new("sk-secret").with_route(Route::Chat);
    let on_responses = OpenAI::with_key(&GROQ, "gsk-secret").with_route(Route::Responses);
    assert_eq!(on_chat.completion_route(), Route::Chat);
    assert_eq!(on_responses.completion_route(), Route::Responses);
    assert_eq!(
        on_chat.completion("gpt-5.2").route(),
        Some("/chat/completions")
    );
    assert_eq!(on_responses.completion("llama").route(), Some("/responses"));

    let restored: OpenAI =
        serde_json::from_str(&serde_json::to_string(&on_chat).expect("serializes"))
            .expect("deserializes");
    assert_eq!(restored.completion_route(), Route::Chat);
    assert!(
        matches!(restored.completion("gpt-5.2"), OpenAiWire::Chat(_)),
        "the configured route survives storage"
    );
    // Without an override, the stored form names no route at all.
    let json = serde_json::to_string(&OpenAI::new("sk-secret")).expect("serializes");
    assert!(!json.contains("route"), "{json}");
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
    assert!(
        OpenAI::new("k")
            .verify_wire()
            .encode((), Mode::Unary)
            .is_ok()
    );
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
            .modality_uri(
                "transcription",
                "/audio/transcriptions",
                "openai/whisper-large-v3"
            )
            .expect("hf-inference serves transcription"),
        "https://router.huggingface.co/openai/whisper-large-v3"
    );

    let routed = default.clone().with_sub_route(SubRoute::Together);
    let error = routed
        .modality_uri("transcription", "/audio/transcriptions", "whisper")
        .expect_err("only hf-inference serves transcription");
    assert_eq!(
        error,
        "transcription endpoint is not supported yet for together"
    );
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

/// Every dialect's listing and credential-check URL, against the paths the
/// recorded fixtures under `crates/rig-cassette/fixtures/cassettes/<provider>/**` actually show.
///
/// This is the audit in executable form. The deleted client did things to a
/// request that each wire's `encode` must now state — `llama-server`'s
/// unversioned operational routes are the case that got through review once
/// (`verify_path: "/props"` resolved to `/v1/props`, which that server
/// answers 404) — so the paths are pinned rather than re-derived.
#[test]
fn every_dialects_listing_and_verify_urls_match_the_recorded_paths() {
    // (dialect, models URL, verify URL or None when the dialect offers no
    // token-free check)
    let expected: &[(&Dialect, &str, Option<&str>)] = &[
        (
            &OPENAI,
            "https://api.openai.com/v1/models",
            Some("https://api.openai.com/v1/models"),
        ),
        (
            &DEEPSEEK,
            "https://api.deepseek.com/models",
            Some("https://api.deepseek.com/user/balance"),
        ),
        (
            &GROQ,
            "https://api.groq.com/openai/v1/models",
            Some("https://api.groq.com/openai/v1/models"),
        ),
        (
            &MISTRAL,
            "https://api.mistral.ai/v1/models",
            Some("https://api.mistral.ai/v1/models"),
        ),
        (
            &OPENROUTER,
            "https://openrouter.ai/api/v1/models",
            Some("https://openrouter.ai/api/v1/key"),
        ),
        (
            &VENICE,
            "https://api.venice.ai/api/v1/models",
            Some("https://api.venice.ai/api/v1/models"),
        ),
        (
            // The one that was wrong: `/props` is a root route on
            // `llama-server`, while `/models` is versioned.
            &LLAMACPP,
            "http://localhost:8080/v1/models",
            Some("http://localhost:8080/props"),
        ),
        (
            &MIRA,
            "https://api.mira.network/v1/models",
            Some("https://api.mira.network/user-credits"),
        ),
        (
            &HUGGINGFACE,
            "https://router.huggingface.co/models",
            Some("https://router.huggingface.co/api/whoami-v2"),
        ),
        (
            &TOGETHER,
            "https://api.together.xyz/v1/models",
            Some("https://api.together.xyz/models"),
        ),
        (&PERPLEXITY, "https://api.perplexity.ai/models", None),
        (
            &crate::providers::xai::DIALECT,
            "https://api.x.ai/v1/models",
            Some("https://api.x.ai/v1/api-key"),
        ),
    ];

    for (dialect, models, verify) in expected {
        let provider = OpenAI::with_key(dialect, "k");
        assert_eq!(
            provider.uri(dialect.quirks.models_path, None),
            *models,
            "{} model-listing URL",
            dialect.name
        );
        match verify {
            Some(verify) => assert_eq!(
                provider.uri(dialect.quirks.verify_path, None),
                *verify,
                "{} verify URL",
                dialect.name
            ),
            None => assert!(
                dialect.quirks.verify_path.is_empty(),
                "{} has no token-free credential check",
                dialect.name
            ),
        }
    }
}

/// `llama-server` serves its operational routes at the root; the versioned
/// base URL applies to everything else.
#[test]
fn llamacpp_serves_its_operational_routes_unversioned() {
    let provider = OpenAI::with_key(&LLAMACPP, "");
    for route in LLAMACPP.quirks.root_relative_routes {
        assert_eq!(
            provider.uri(route, None),
            format!("http://localhost:8080{route}"),
            "{route} is a root route"
        );
    }
    // Everything else keeps the `/v1` the base URL carries.
    assert_eq!(
        provider.uri("/chat/completions", None),
        "http://localhost:8080/v1/chat/completions"
    );
    assert_eq!(
        provider.uri("/rerank", None),
        "http://localhost:8080/v1/rerank"
    );
    // And no other dialect has any.
    for dialect in all() {
        if dialect.name != "llamacpp" {
            assert!(
                dialect.quirks.root_relative_routes.is_empty(),
                "{} declares root routes it never had",
                dialect.name
            );
        }
    }
}
