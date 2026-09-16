//! Copilot's wires, driven from recorded bytes.
//!
//! The bodies are read out of the cassettes rather than copied into this
//! file, so a recorded turn and the assertion about it cannot drift. The
//! cassettes are read-only here.

use super::*;
use crate::completion::{CompletionModel, CompletionRequest};
use crate::driver::Bound;
use crate::embeddings::EmbeddingModel as _;
use crate::message::Message;
use crate::model::ModelLister as _;
use crate::test_utils::RecordingHttpClient;
use bytes::Bytes;

/// One recorded interaction's request or reply body, read out of a cassette.
///
/// The format is a `when:`/`then:` document whose bodies are single-quoted
/// scalars (`''` for a quote). Parsed here rather than with a YAML
/// dependency, and never written.
fn cassette_body(path: &str, section: &str) -> String {
    let file = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/cassettes/copilot")
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
fn copilot() -> Copilot {
    Copilot::new("tid=copilot-session-token")
}

fn prompt() -> CompletionRequest {
    CompletionRequest {
        model: None,
        chat_history: vec![Message::user("say hi")],
        documents: Vec::new(),
        tools: Vec::new(),
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    }
}

/// The first text block of a folded turn.
fn text_of(response: &crate::completion::CompletionResponse) -> Option<String> {
    response.choice.iter().find_map(|content| match content {
        crate::message::AssistantContent::Text(text) => Some(text.text.clone()),
        _ => None,
    })
}

/// The request one `encode` produced, for the envelope assertions.
fn encoded(wire: &CopilotWire) -> http::Request<Body> {
    let mut encoded = wire
        .encode(prompt(), Mode::Unary)
        .expect("the request encodes");
    assert_eq!(encoded.requests.len(), 1, "one route, one request");
    encoded.requests.remove(0)
}

// ── routing ─────────────────────────────────────────────────────────────

/// The route is a property of the model, and the choice is made in one
/// place. `tests/cassettes/copilot/routing/` records both halves: a Codex
/// model answered by `/responses`, every other model by `/chat/completions`.
#[test]
fn the_model_chooses_the_route() {
    let copilot = copilot();
    for model in [
        super::super::GPT_5_3_CODEX,
        super::super::GPT_5_1_CODEX,
        // The predicate is the identifier, not a table: an unannounced
        // Codex model still routes correctly, in any casing.
        "GPT-6-CODEX-PREVIEW",
    ] {
        assert!(
            matches!(copilot.completion(model).wire, OpenAiWire::Responses(_)),
            "{model} is served by /responses"
        );
    }
    for model in [
        super::super::GPT_4O,
        super::super::CLAUDE_SONNET_4_6,
        super::super::GEMINI_3_FLASH,
        super::super::O3_MINI,
    ] {
        assert!(
            matches!(copilot.completion(model).wire, OpenAiWire::Chat(_)),
            "{model} is served by /chat/completions"
        );
    }
}

/// Each route posts to its own path, and both carry Copilot's editor
/// envelope: without it the API answers 400 regardless of the body.
#[test]
fn both_routes_carry_copilots_editor_envelope() {
    let copilot = copilot();
    for (wire, path) in [
        (
            copilot.completion(super::super::GPT_4O),
            "/chat/completions",
        ),
        (
            copilot.completion(super::super::GPT_5_3_CODEX),
            "/responses",
        ),
    ] {
        let request = encoded(&wire);
        assert_eq!(
            request.uri().path(),
            path,
            "{:?} posts to {path}",
            wire.model()
        );
        let headers = request.headers();
        assert_eq!(
            headers.get_all(http::header::AUTHORIZATION).iter().count(),
            1,
            "the delegated wire's credential header is replaced, not duplicated"
        );
        assert_eq!(
            headers
                .get(http::header::AUTHORIZATION)
                .and_then(|value| value.to_str().ok()),
            Some("Bearer tid=copilot-session-token")
        );
        assert_eq!(
            headers.get("copilot-integration-id").map(|v| v.as_bytes()),
            Some(&b"vscode-chat"[..])
        );
        assert_eq!(
            headers.get("editor-version").map(|v| v.as_bytes()),
            Some(super::super::EDITOR_VERSION.as_bytes())
        );
        assert_eq!(
            headers.get("openai-intent").map(|v| v.as_bytes()),
            Some(&b"conversation-panel"[..])
        );
        // A first user turn is the user's; `X-Initiator` is lowercased by
        // `HeaderName`, as it is on the wire.
        assert_eq!(
            headers.get("x-initiator").map(|v| v.as_bytes()),
            Some(&b"user"[..])
        );
        assert!(
            headers.get("copilot-vision-request").is_none(),
            "a text-only turn does not claim vision"
        );
    }
}

/// The intent is a per-turn header, so it lives on the wire and survives
/// whichever route was chosen.
#[test]
fn the_intent_is_a_wire_option_on_both_routes() {
    for model in [super::super::GPT_4O, super::super::GPT_5_3_CODEX] {
        let wire = copilot().completion(model).with_edits_intent();
        assert_eq!(wire.intent(), CopilotIntent::Edits);
        assert_eq!(
            encoded(&wire)
                .headers()
                .get("openai-intent")
                .map(|value| value.as_bytes()),
            Some(&b"conversation-edits"[..]),
            "{model}"
        );
    }
}

// ── the two completion routes, folded from recorded replies ─────────────

/// The chat route's recorded turn folds to the normalized response.
#[tokio::test]
async fn the_chat_route_folds_its_recorded_turn() {
    let body = cassette_body("agent/completion_smoke.yaml", "then");
    let response = Bound::new(
        copilot().completion(super::super::GPT_4O),
        RecordingHttpClient::new(Bytes::from(body)),
    )
    .completion(prompt())
    .await
    .expect("the recorded chat body folds");

    assert_eq!(response.provider, PROVIDER_NAME);
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
    let response = Bound::new(
        copilot().completion(super::super::GPT_5_3_CODEX),
        RecordingHttpClient::new(Bytes::from(body)),
    )
    .completion(prompt())
    .await
    .expect("the recorded responses body folds");

    assert_eq!(response.provider, PROVIDER_NAME);
    assert_eq!(response.model.as_deref(), Some(super::super::GPT_5_3_CODEX));
    assert!(
        text_of(&response).is_some_and(|text| text.contains("Refactoring is the process")),
        "{:?}",
        response.choice
    );
    assert_eq!(response.response_id.as_deref(), Some("resp_REDACTED_1"));
}

/// Copilot's Responses route answers a tool-calling turn with a
/// *contentless* reasoning item — empty `content`, empty `summary`, no
/// encrypted payload, just an id. The next turn has to replay it verbatim:
/// `tests/cassettes/copilot/typed_prompt_tools/prompt_typed_with_tool_call_roundtrip.yaml`
/// records `{"id":"id_REDACTED_1","summary":[],"type":"reasoning"}` ahead of
/// the tool result in its second request, so the block has to survive the
/// fold or the replayed history is missing an item the provider sent.
#[tokio::test]
async fn a_contentless_reasoning_item_survives_the_fold() {
    let body = cassette_body(
        "typed_prompt_tools/prompt_typed_with_tool_call_roundtrip.yaml",
        "then",
    );
    let response = Bound::new(
        copilot().completion(super::super::GPT_5_3_CODEX),
        RecordingHttpClient::new(Bytes::from(body)),
    )
    .completion(prompt())
    .await
    .expect("the recorded responses body folds");

    let reasoning = response
        .choice
        .iter()
        .find_map(|content| match content {
            crate::message::AssistantContent::Reasoning(reasoning) => Some(reasoning),
            _ => None,
        })
        .unwrap_or_else(|| panic!("the turn's reasoning item survives: {:?}", response.choice));
    assert_eq!(reasoning.id.as_deref(), Some("id_REDACTED_1"));
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
    let bound = Bound::new(
        copilot().embeddings(super::super::TEXT_EMBEDDING_3_SMALL, None),
        RecordingHttpClient::new(Bytes::from(body)),
    );
    assert_eq!(bound.ndims(), 1536, "the width defaults from the model");
    assert_eq!(bound.max_documents(), 1024);

    let response = bound
        .embed_texts_response(documents.clone())
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
    let wire = copilot().embeddings(super::super::TEXT_EMBEDDING_3_SMALL, None);
    let mut encoded = wire
        .encode(vec!["one".to_owned()], Mode::Unary)
        .expect("the request encodes");
    let Body::Bytes(bytes) = encoded.requests.remove(0).into_body() else {
        panic!("the embeddings body is bytes");
    };
    let body: serde_json::Value = serde_json::from_slice(&bytes).expect("the body is JSON");
    assert_eq!(body["dimensions"], serde_json::json!(1536));
    assert_eq!(body["model"], serde_json::json!("text-embedding-3-small"));

    // The legacy Ada model accepts no width at all.
    let ada = copilot().embeddings(super::super::TEXT_EMBEDDING_ADA_002, None);
    let mut encoded = ada
        .encode(vec!["one".to_owned()], Mode::Unary)
        .expect("the request encodes");
    let Body::Bytes(bytes) = encoded.requests.remove(0).into_body() else {
        panic!("the embeddings body is bytes");
    };
    let body: serde_json::Value = serde_json::from_slice(&bytes).expect("the body is JSON");
    assert!(body.get("dimensions").is_none(), "{body}");
}

/// Copilot's catalogue names the vendor behind each model and nests the
/// modality under `capabilities.type`, which is what distinguishes it from
/// the OpenAI-shaped listing.
#[tokio::test]
async fn the_model_listing_folds_its_recorded_catalogue() {
    let body = cassette_body("models/list_models_smoke.yaml", "then");
    let models = Bound::new(
        copilot().models(),
        RecordingHttpClient::new(Bytes::from(body)),
    )
    .list_all()
    .await
    .expect("the recorded catalogue folds");

    let opus = models
        .iter()
        .find(|model| model.id == "claude-opus-4.7")
        .expect("the recorded catalogue names claude-opus-4.7");
    assert_eq!(opus.name.as_deref(), Some("Claude Opus 4.7"));
    assert_eq!(opus.r#type.as_deref(), Some("chat"));
}

// ── the configuration is storable data ──────────────────────────────────

/// A wire is data a host may store in a scene, a component or a config
/// file, so the credential must not travel with it.
#[test]
fn a_serialized_config_carries_no_credential() {
    let json = serde_json::to_string(&Copilot::new("tid=super-secret-session-token"))
        .expect("the config serializes");
    assert!(!json.contains("super-secret"), "{json}");
    assert!(json.contains("[redacted]"), "{json}");

    // And the wires built from it, whichever route.
    for model in [super::super::GPT_4O, super::super::GPT_5_3_CODEX] {
        let json = serde_json::to_string(
            &Copilot::new("tid=super-secret-session-token").completion(model),
        )
        .expect("the wire serializes");
        assert!(!json.contains("super-secret"), "{model}: {json}");
    }
}

/// A session token names the REST endpoint it was minted for; nothing else
/// knows it, so the configuration reads it off the credential — and an
/// explicit base URL still wins.
#[test]
fn the_base_url_comes_from_the_token_unless_overridden() {
    assert_eq!(
        Copilot::new("tid=abc;proxy-ep=proxy.individual.githubcopilot.com;").base_url,
        "https://api.individual.githubcopilot.com"
    );
    assert_eq!(
        Copilot::new("tid=abc").base_url,
        "https://api.githubcopilot.com"
    );
    assert_eq!(
        Copilot::new("tid=abc;proxy-ep=proxy.individual.githubcopilot.com;")
            .with_base_url("https://gateway.invalid")
            .base_url,
        "https://gateway.invalid"
    );
    // A non-GitHub host in a credential is not a routing instruction.
    assert_eq!(
        Copilot::new("tid=abc;proxy-ep=evil.invalid;").base_url,
        "https://api.githubcopilot.com"
    );
}
