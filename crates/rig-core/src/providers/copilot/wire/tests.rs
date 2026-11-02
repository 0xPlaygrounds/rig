//! Copilot's wires, driven from recorded bytes.
//!
//! The bodies are read out of the cassettes rather than copied into this
//! file, so a recorded turn and the assertion about it cannot drift. The
//! cassettes are read-only here.

use super::*;
use crate::completion::CompletionRequest;
use crate::message::Message;
use crate::operation::Completion;
use crate::test_utils::{RecordingHttpClient, json_body};
use crate::wire::Mode;
use crate::wire::Wire;
use crate::wire::secret::tests::a_config_reloads_without_its_credential;
use bytes::Bytes;
use serde_json::Value;

/// One recorded interaction's request or reply body, read out of a cassette.
///
/// The format is a `when:`/`then:` document whose bodies are single-quoted
/// scalars (`''` for a quote). Parsed here rather than with a YAML
/// dependency, and never written.
fn cassette_body(path: &str, section: &str) -> String {
    let file = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../rig-cassette/fixtures/cassettes/copilot")
        .join(path);
    let text = std::fs::read_to_string(&file)
        .unwrap_or_else(|error| panic!("{} should be readable: {error}", file.display()));
    let section = text
        .split_once(&format!("\n{section}:"))
        .unwrap_or_else(|| panic!("{} should record a {section} section", file.display()))
        .1;
    section
        .split_once("  body: ")
        .unwrap_or_else(|| panic!("{} should record a {section} body", file.display()))
        .1
        .lines()
        .next()
        .unwrap_or_default()
        .trim()
        .trim_start_matches('\'')
        .trim_end_matches('\'')
        .replace("''", "'")
}

/// A Copilot addressed with a token that carries no `proxy-ep=` segment, so
/// the base URL is the default one.
fn copilot() -> OpenAIConfig {
    OpenAIConfig::with_key(&DIALECT, "tid=copilot-session-token")
}

fn prompt() -> CompletionRequest {
    CompletionRequest::new("say hi")
}

/// The first text block of a folded turn.
fn text_of(response: &crate::completion::CompletionResponse) -> Option<String> {
    response.choice.iter().find_map(|content| match content {
        crate::message::AssistantContent::Text(text) => Some(text.text.clone()),
        _ => None,
    })
}

/// The request one `encode` produced, for the envelope assertions.
fn encoded(
    wire: &impl Wire<Op = Completion, Payload = crate::wire::Encoded>,
) -> http::Request<Body> {
    wire.encode(prompt(), Mode::Unary)
        .expect("the request encodes")
        .request
}

// ── routing ─────────────────────────────────────────────────────────────

/// A session token naming its own endpoint.
const TOKEN: &str = "tid=1;proxy-ep=proxy.individual.githubcopilot.com;exp=2";

/// Exercise the registry's erased handler against recorded Copilot replies,
/// including the actual outbound request; no live credentials are needed.
async fn registry_request(
    config: crate::providers::registry::ProviderConfig,
    model: &str,
    cassette: &str,
) -> crate::test_utils::CapturedHttpRequest {
    use crate::effect::{EffectId, EffectKind};
    use crate::http_client::DynHttpClient;
    use crate::serve::Dispatch;

    let transport = RecordingHttpClient::new(Bytes::from(cassette_body(cassette, "then")));
    let handler = config.completion_handler(
        "copilot",
        model,
        DynHttpClient::new(transport.clone()),
        crate::catalog::ModelFacts::default(),
    );
    let mut request = prompt();
    request
        .chat_history
        .insert(0, Message::system("be brief").into());
    handler
        .handle(
            EffectKind::Completion {
                request: request.into(),
                stream: false,
            },
            Dispatch::new(EffectId::from_raw(1), false),
        )
        .await
        .into_outcome()
        .await
        .expect("the registry handler folds the recorded Copilot reply");
    let mut requests = transport.requests();
    assert_eq!(requests.len(), 1);
    requests.remove(0)
}

/// The registry must select the same routes and editor envelope as the
/// dedicated provider. Recorded replies catch a wrong decoder as well.
#[tokio::test]
async fn registry_copilot_preserves_model_routing_and_editor_envelope() {
    use crate::providers::registry::ProviderId;

    for (model, path, cassette) in [
        (
            super::super::GPT_4O,
            "/chat/completions",
            "agent/completion_smoke.yaml",
        ),
        (
            super::super::GPT_5_3_CODEX,
            "/responses",
            "routing/codex_models_route_through_responses.yaml",
        ),
    ] {
        let config = ProviderId::resolve("copilot")
            .ok()
            .and_then(|id| id.config(TOKEN))
            .expect("a registered Copilot selection");
        let request = registry_request(config, model, cassette).await;
        assert_eq!(
            request.uri,
            format!("https://api.individual.githubcopilot.com{path}"),
            "the session endpoint and model route must both survive materialization"
        );
        assert_eq!(request.headers["copilot-integration-id"], "vscode-chat");
        assert_eq!(
            request.headers["editor-version"],
            super::super::EDITOR_VERSION
        );
        assert_eq!(request.headers["openai-intent"], "conversation-panel");
        assert_eq!(request.headers["x-initiator"], "user");
        assert_eq!(
            request.headers[http::header::AUTHORIZATION],
            format!("Bearer {TOKEN}")
        );
        if path == "/responses" {
            let body: serde_json::Value =
                serde_json::from_slice(&request.body).expect("JSON request");
            assert!(body.get("instructions").is_none());
            assert_eq!(body["input"][0]["role"], "system");
        }
    }
}

/// Explicit registry configuration overrides the model's default route and
/// instruction placement, while keeping the gateway's editor envelope.
#[tokio::test]
async fn registry_copilot_preserves_explicit_configuration_after_reload() {
    use crate::providers::openai::Route;
    use crate::providers::registry::ProviderConfig;

    let config = ProviderConfig::OpenAi(
        OpenAIConfig::with_key(&DIALECT, "")
            .with_base_url("https://gateway.invalid/copilot")
            .with_route(Route::Responses)
            .with_system_instructions_placement(SystemInstructionsPlacement::Instructions),
    );
    let saved = serde_json::to_string(&config).expect("configuration serializes");
    let loaded: ProviderConfig = serde_json::from_str(&saved).expect("configuration reloads");
    let request = registry_request(
        loaded.with_credential(TOKEN),
        super::super::GPT_4O,
        "routing/codex_models_route_through_responses.yaml",
    )
    .await;
    assert_eq!(request.uri, "https://gateway.invalid/copilot/responses");
    assert_eq!(request.headers["copilot-integration-id"], "vscode-chat");
    assert_eq!(request.headers["openai-intent"], "conversation-panel");
    let body: serde_json::Value = serde_json::from_slice(&request.body).expect("JSON request");
    assert_eq!(body["instructions"], "be brief");
    assert_eq!(body["model"], super::super::GPT_4O);
}

// ── the two completion routes, folded from recorded replies ─────────────

/// The chat route's recorded turn folds to the normalized response.
#[tokio::test]
async fn the_chat_route_folds_its_recorded_turn() {
    let body = cassette_body("agent/completion_smoke.yaml", "then");
    let response = crate::driver::Model::new(
        copilot().completion(super::super::GPT_4O),
        RecordingHttpClient::new(Bytes::from(body)),
    )
    .call(prompt())
    .await
    .expect("the recorded chat body folds");

    assert_eq!(response.provider(), PROVIDER_NAME);
    assert!(
        text_of(&response)
            .is_some_and(|text| text.contains("Rust is a systems programming language")),
        "{:?}",
        response.choice
    );
    assert_eq!(response.usage.input_tokens, Some(38));
}

/// The Responses route's recorded turn folds to the normalized response —
/// same operation, same fold, a different wire underneath.
#[tokio::test]
async fn the_responses_route_folds_its_recorded_turn() {
    let body = cassette_body("routing/codex_models_route_through_responses.yaml", "then");
    let response = crate::driver::Model::new(
        copilot().completion(super::super::GPT_5_3_CODEX),
        RecordingHttpClient::new(Bytes::from(body)),
    )
    .call(prompt())
    .await
    .expect("the recorded responses body folds");

    assert_eq!(response.provider(), PROVIDER_NAME);
    assert_eq!(response.model(), Some(super::super::GPT_5_3_CODEX));
    assert!(
        text_of(&response).is_some_and(|text| text.contains("Refactoring is the process")),
        "{:?}",
        response.choice
    );
    assert_eq!(response.response_id(), Some("resp_REDACTED_1"));
}

/// Copilot's Responses route answers a tool-calling turn with a
/// *contentless* reasoning item — empty `content`, empty `summary`, no
/// encrypted payload, just an id. The next turn has to replay it verbatim:
/// `crates/rig-cassette/fixtures/cassettes/copilot/typed_prompt_tools/prompt_typed_with_tool_call_roundtrip.yaml`
/// records `{"id":"id_REDACTED_1","summary":[],"type":"reasoning"}` ahead of
/// the tool result in its second request, so the block has to survive the
/// fold or the replayed history is missing an item the provider sent.
#[tokio::test]
async fn a_contentless_reasoning_item_survives_the_fold() {
    let body = cassette_body(
        "typed_prompt_tools/prompt_typed_with_tool_call_roundtrip.yaml",
        "then",
    );
    let response = crate::driver::Model::new(
        copilot().completion(super::super::GPT_5_3_CODEX),
        RecordingHttpClient::new(Bytes::from(body)),
    )
    .call(prompt())
    .await
    .expect("the recorded responses body folds");

    let reasoning = response
        .choice
        .iter()
        .find(|content| matches!(content, crate::message::AssistantContent::Reasoning(_)))
        .unwrap_or_else(|| panic!("the turn's reasoning item survives: {:?}", response.choice));
    assert_eq!(
        reasoning.native_item(),
        Some(&serde_json::json!({
            "content": [],
            "id": "id_REDACTED_1",
            "summary": [],
            "type": "reasoning"
        }))
    );
}

// ── the modality wires ──────────────────────────────────────────────────

/// The embeddings reply folds with the request's inputs joined back on, and
/// a reply without a usage block reports no counter rather than failing:
/// Copilot's multi-vendor route omits it.
#[tokio::test]
async fn the_embeddings_wire_folds_its_recorded_reply() {
    let body = cassette_body("embeddings/embeddings_smoke.yaml", "then");
    let documents = vec![
        "Rust values memory safety and predictable performance.".to_owned(),
        "Streaming responses arrive incrementally instead of all at once.".to_owned(),
        "Embeddings turn text into numeric vectors for similarity search.".to_owned(),
    ];
    let bound = crate::driver::Model::new(
        copilot().embedding(super::super::TEXT_EMBEDDING_3_SMALL, None),
        RecordingHttpClient::new(Bytes::from(body)),
    );
    assert_eq!(
        bound.capabilities().ndims,
        1536,
        "the width defaults from the model"
    );
    assert_eq!(bound.capabilities().max_documents, 1024);

    let response = bound
        .call(documents.clone())
        .await
        .expect("the recorded embeddings body folds");
    assert_eq!(response.provider, PROVIDER_NAME);
    assert_eq!(response.embeddings.len(), documents.len());
    assert_eq!(response.embeddings[0].document, documents[0]);
    assert!(!response.embeddings[0].vec.is_empty());
}

/// The recorded embeddings request carries the width resolved from the model
/// identifier: the client layer defaulted it, and the cassette records
/// `"dimensions":1536` for a caller who named none.
#[test]
fn the_embeddings_request_sends_the_resolved_width() {
    let wire = copilot().embedding(super::super::TEXT_EMBEDDING_3_SMALL, None);
    let encoded = wire
        .encode(vec!["one".to_owned()], Mode::Unary)
        .expect("the request encodes");
    let body = json_body(&encoded.request);
    assert_eq!(body["dimensions"], serde_json::json!(1536));
    assert_eq!(body["model"], serde_json::json!("text-embedding-3-small"));

    // The legacy Ada model accepts no width at all.
    let ada = copilot().embedding(super::super::TEXT_EMBEDDING_ADA_002, None);
    let encoded = ada
        .encode(vec!["one".to_owned()], Mode::Unary)
        .expect("the request encodes");
    let body = json_body(&encoded.request);
    assert!(body.get("dimensions").is_none(), "{body}");
}

/// Copilot's catalogue names the vendor behind each model and nests the
/// modality under `capabilities.type`, which is what distinguishes it from
/// the OpenAI-shaped listing.
#[tokio::test]
async fn the_model_listing_folds_its_recorded_catalogue() {
    let body = cassette_body("models/list_models_smoke.yaml", "then");
    let recorded: Value = serde_json::from_str(&body).expect("the recorded catalogue is JSON");
    let models = crate::driver::Model::new(
        super::super::model_listing::Models::new(copilot()),
        RecordingHttpClient::new(Bytes::from(body)),
    )
    .list()
    .await
    .expect("the recorded catalogue folds");

    let entries = recorded["data"].as_array().expect("recorded model entries");
    assert_eq!(models.len(), entries.len());
    for (model, entry) in models.iter().zip(entries) {
        assert_eq!(Some(model.id.as_str()), entry["id"].as_str());
        assert_eq!(
            model.supports_tool_calls,
            entry
                .pointer("/capabilities/supports/tool_calls")
                .and_then(Value::as_bool),
            "tool-calling support for {}",
            model.id
        );
    }

    let opus = models
        .iter()
        .find(|model| model.id == "claude-opus-4.7")
        .expect("the recorded catalogue names claude-opus-4.7");
    assert_eq!(opus.name.as_deref(), Some("Claude Opus 4.7"));
    assert_eq!(opus.r#type.as_deref(), Some("chat"));
    assert_eq!(opus.supports_tool_calls, Some(true));
}

// ── the configuration is storable data ──────────────────────────────────

/// A wire is data a host may store in a scene, a component or a config
/// file, so the credential must not travel with it.
#[test]
fn a_serialized_config_carries_no_credential() {
    a_config_reloads_without_its_credential(
        &OpenAIConfig::with_key(&DIALECT, "tid=super-secret-session-token"),
        "super-secret",
        |copilot| &copilot.api_key,
    );

    // And the wires built from it, whichever route.
    for model in [super::super::GPT_4O, super::super::GPT_5_3_CODEX] {
        let json = serde_json::to_string(
            &OpenAIConfig::with_key(&DIALECT, "tid=super-secret-session-token").completion(model),
        )
        .expect("the wire serializes");
        assert!(!json.contains("super-secret"), "{model}: {json}");
    }
}

/// Persistence must refuse callbacks it cannot restore; this is a local data contract.
#[test]
fn named_dialect_persistence_rejects_replaced_hook_definitions() {
    // Even a copied callback set is not the registered static definition.
    // Function addresses cannot establish whether a custom definition is reloadable.
    static REPLACEMENT: DialectHooks = DialectHooks {
        default_endpoint: HOOKS.default_endpoint,
        model_route: HOOKS.model_route,
        completion_envelope: HOOKS.completion_envelope,
        modality_envelope: None,
    };
    let changed = Dialect {
        quirks: Quirks {
            hooks: Some(&REPLACEMENT),
            ..DIALECT.quirks
        },
        ..DIALECT
    };
    assert_ne!(changed, DIALECT);
    assert!(serde_json::to_value(changed).is_err());
    let restored: Dialect = serde_json::from_value(serde_json::to_value(DIALECT).unwrap()).unwrap();
    assert_eq!(restored, DIALECT);
    assert!(std::ptr::eq(restored.quirks.hooks.unwrap(), &HOOKS));
}

/// Synthetic credentials and explicit hosts exercise construction precedence, not server behavior.
/// Recorded reply normalization is covered by the registry and direct-route tests above.
#[test]
fn configured_outbound_endpoints_remain_explicit_after_rotation() {
    use crate::providers::registry::{ProviderConfig, ProviderId};

    let keys = [
        "tid=1;proxy-ep=proxy.individual.githubcopilot.com;",
        "tid=2;proxy-ep=proxy.business.githubcopilot.com;",
    ];
    for model in [super::super::GPT_4O, super::super::GPT_5_3_CODEX] {
        for base in [
            "https://api.githubcopilot.com",
            "https://api.individual.githubcopilot.com",
            "https://explicit.invalid/copilot",
        ] {
            for route in [Route::Chat, Route::Responses] {
                let config = ProviderConfig::OpenAi(
                    OpenAIConfig::with_key(&DIALECT, keys[0])
                        .with_base_url(base)
                        .with_route(route),
                );
                let restored: ProviderConfig =
                    serde_json::from_value(serde_json::to_value(config).unwrap()).unwrap();
                for key in keys {
                    let ProviderConfig::OpenAi(provider) = restored.clone().with_credential(key)
                    else {
                        panic!("OpenAI family")
                    };
                    let request = encoded(&provider.completion(model));
                    let path = match route {
                        Route::Chat => "/chat/completions",
                        Route::Responses => "/responses",
                    };
                    assert_eq!(request.uri().to_string(), format!("{base}{path}"));
                    assert_eq!(
                        request.headers()[http::header::AUTHORIZATION],
                        format!("Bearer {key}")
                    );
                    assert_eq!(request.headers()["copilot-integration-id"], "vscode-chat");
                }
            }
        }
        let registered = ProviderId::resolve("copilot").unwrap();
        for (key, host) in keys.into_iter().zip([
            "api.individual.githubcopilot.com",
            "api.business.githubcopilot.com",
        ]) {
            let ProviderConfig::OpenAi(provider) = registered.config(key).unwrap() else {
                panic!("OpenAI family")
            };
            let request = encoded(&provider.completion(model));
            assert_eq!(request.uri().host(), Some(host));
        }
    }
}

/// The local envelope depends on input history before either codec consumes it; no reply is needed.
#[test]
fn the_completion_envelope_sees_the_original_vision_and_assistant_history() {
    use crate::message::{DocumentSourceKind, Image, UserContent};
    let mut request = prompt();
    request.chat_history = vec![
        Message::assistant("send an image").into(),
        Message::User {
            content: vec![UserContent::Image(Image {
                data: DocumentSourceKind::Url("https://image.invalid/example.png".into()),
                ..Image::default()
            })],
        }
        .into(),
    ];
    for model in [super::super::GPT_4O, super::super::GPT_5_3_CODEX] {
        let wire = copilot().completion(model);
        for mode in [Mode::Unary, Mode::Streaming] {
            let encoded = wire.encode(request.clone(), mode).unwrap();
            let request = &encoded.request;
            assert_eq!(request.headers()["x-initiator"], "agent");
            assert_eq!(request.headers()["copilot-vision-request"], "true");
            assert_eq!(request.headers()["openai-intent"], "conversation-panel");
        }
    }
}

/// The facts the Copilot wire is given are the facts its descriptor
/// answers with, which `DynModel::spec` returns.
#[test]
fn the_wire_answers_from_the_facts_it_is_given() {
    use crate::catalog::{ModelFacts, ModelSpec};
    use crate::wire::Wire as _;

    let vendor = crate::providers::registry::ProviderId::catalog("copilot").expect("a vendor");
    let spec = ModelSpec::new(vendor, "gpt-4.1").with_max_output_tokens(1_234);
    let wire = copilot()
        .completion("gpt-4.1")
        .with_facts(ModelFacts::new(spec));
    assert_eq!(
        wire.describe()
            .spec()
            .and_then(|spec| spec.max_output_tokens),
        Some(1_234)
    );
}
