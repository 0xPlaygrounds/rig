use crate::completion::{CompletionRequest, ToolDefinition};
use crate::providers::registry::{ProviderId, ProviderRef};
use crate::providers::{anthropic, openai};
use crate::wire::{Body, Encoded, Framing, Mode, Wire, WireFrame};
use serde_json::{Value, json};

fn request() -> CompletionRequest {
    let mut request = CompletionRequest::new("Find the Rust files");
    request.max_tokens = Some(128);
    request.tools = vec![ToolDefinition {
        name: "find_files".to_owned(),
        description: "Find files by extension".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": {"extension": {"type": "string"}},
            "required": ["extension"]
        }),
    }];
    request
}

fn body(encoded: &Encoded) -> Value {
    match encoded.request.body() {
        Body::Bytes(bytes) => serde_json::from_slice(bytes).expect("a JSON request"),
        Body::Multipart(_) => panic!("completion requests must be JSON"),
    }
}

/// These request-boundary tests need no provider reply. No OpenCode credentials
/// are available to record cassettes; the existing protocol decoders are reused.
#[test]
fn chat_uses_each_plans_endpoint_with_authentication_and_tools() {
    for (dialect, root) in [
        (&openai::wire::OPENCODE_ZEN, "https://opencode.ai/zen/v1"),
        (&openai::wire::OPENCODE_GO, "https://opencode.ai/zen/go/v1"),
    ] {
        let wire = openai::OpenAIConfig::with_key(dialect, "test-key").chat("kimi-k2.6");
        for mode in [Mode::Unary, Mode::Streaming] {
            let encoded = wire.encode(request(), mode).expect("chat request encodes");
            assert_eq!(
                encoded.request.uri().to_string(),
                format!("{root}/chat/completions")
            );
            assert_eq!(
                encoded.request.headers().get("authorization").unwrap(),
                "Bearer test-key"
            );
            assert_eq!(wire.describe().name, dialect.name);
            let body = body(&encoded);
            assert_eq!(body["model"], "kimi-k2.6");
            assert_eq!(body["messages"][0]["content"], "Find the Rust files");
            assert_eq!(body["max_tokens"], 128);
            assert_eq!(body["tools"][0]["function"]["name"], "find_files");
            if mode == Mode::Streaming {
                assert_eq!(encoded.framing, Framing::Sse);
                assert_eq!(body["stream"], true);
            } else {
                assert_eq!(encoded.framing, Framing::Whole);
            }
        }
    }
}

/// The route and request grammar are local choices, tested without live traffic.
#[test]
fn responses_uses_each_plans_endpoint_and_responses_tool_shape() {
    for (dialect, root) in [
        (&openai::wire::OPENCODE_ZEN, "https://opencode.ai/zen/v1"),
        (&openai::wire::OPENCODE_GO, "https://opencode.ai/zen/go/v1"),
    ] {
        let wire = openai::OpenAIConfig::with_key(dialect, "test-key").responses("gpt-6-luna");
        for mode in [Mode::Unary, Mode::Streaming] {
            let encoded = wire
                .encode(request(), mode)
                .expect("Responses request encodes");
            assert_eq!(
                encoded.request.uri().to_string(),
                format!("{root}/responses")
            );
            assert_eq!(
                encoded.request.headers().get("authorization").unwrap(),
                "Bearer test-key"
            );
            let body = body(&encoded);
            assert_eq!(body["model"], "gpt-6-luna");
            assert!(body["input"].is_array());
            assert_eq!(body["max_output_tokens"], 128);
            assert_eq!(body["tools"][0]["type"], "function");
            assert_eq!(body["tools"][0]["name"], "find_files");
            if mode == Mode::Streaming {
                assert_eq!(encoded.framing, Framing::Sse);
                assert_eq!(body["stream"], true);
            } else {
                assert_eq!(encoded.framing, Framing::Whole);
                assert!(body.get("stream").is_none());
            }
        }
    }
}

/// The Messages presets share the existing decoder; this tests outbound routing.
#[test]
fn messages_uses_each_plans_endpoint_and_anthropic_authentication() {
    for (dialect, url, model) in [
        (
            &anthropic::wire::OPENCODE_ZEN,
            "https://opencode.ai/zen/v1/messages",
            "claude-sonnet-4-6",
        ),
        (
            &anthropic::wire::OPENCODE_GO,
            "https://opencode.ai/zen/go/v1/messages",
            "minimax-m2.7",
        ),
    ] {
        let wire = anthropic::AnthropicConfig::with_dialect("test-key", dialect).completion(model);
        for mode in [Mode::Unary, Mode::Streaming] {
            let encoded = wire
                .encode(request(), mode)
                .expect("Messages request encodes");
            assert_eq!(encoded.request.uri().to_string(), url);
            assert_eq!(
                encoded.request.headers().get("x-api-key").unwrap(),
                "test-key"
            );
            assert_eq!(
                encoded.request.headers().get("anthropic-version").unwrap(),
                "2023-06-01"
            );
            assert_eq!(wire.describe().name, dialect.name);
            let body = body(&encoded);
            assert_eq!(body["model"], model);
            assert_eq!(body["max_tokens"], 128);
            assert_eq!(body["tools"][0]["name"], "find_files");
            assert_eq!(body["tools"][0]["input_schema"]["type"], "object");
            if mode == Mode::Streaming {
                assert_eq!(encoded.framing, Framing::Sse);
                assert_eq!(body["stream"], true);
            }
        }
    }
}

/// Model-list URLs are preset data and can be checked without contacting a provider.
#[test]
fn model_listing_stays_on_the_selected_plan() {
    for (dialect, url) in [
        (
            &openai::wire::OPENCODE_ZEN,
            "https://opencode.ai/zen/v1/models",
        ),
        (
            &openai::wire::OPENCODE_GO,
            "https://opencode.ai/zen/go/v1/models",
        ),
    ] {
        let encoded = openai::OpenAIConfig::with_key(dialect, "test-key")
            .models()
            .encode(Default::default(), Mode::Unary)
            .expect("model-list request encodes");
        assert_eq!(encoded.request.method(), http::Method::GET);
        assert_eq!(encoded.request.uri().to_string(), url);
    }
}

#[test]
fn registered_plans_and_protocols_round_trip_without_credentials() {
    for vendor in ["opencode", "opencode-go"] {
        for format in ["openai", "anthropic"] {
            let selection = format!("{vendor}/{format}");
            let id = ProviderId::resolve(&selection).expect("registered OpenCode preset");
            let reference =
                ProviderRef::configured(id.config("test-secret"), "model").expect("nonempty model");
            let encoded = serde_json::to_string(&reference).expect("reference serializes");
            assert!(!encoded.contains("test-secret"));
            let restored: ProviderRef = serde_json::from_str(&encoded).expect("reference reloads");
            assert_eq!(restored, reference);
            assert_eq!(restored.id(), Some(id));
        }
    }
}

/// OpenCode exposes its model catalogs publicly, so requesting them cannot
/// validate a key. This local policy must reject verification before any HTTP call.
#[test]
fn public_model_catalogs_are_not_used_to_verify_credentials() {
    for dialect in [&openai::wire::OPENCODE_ZEN, &openai::wire::OPENCODE_GO] {
        let error = openai::OpenAIConfig::with_key(dialect, "invalid-key")
            .verify()
            .encode((), Mode::Unary)
            .expect_err("a public catalog cannot verify credentials");
        assert!(error.to_string().contains("no endpoint"));
    }
    for dialect in [
        &anthropic::wire::OPENCODE_ZEN,
        &anthropic::wire::OPENCODE_GO,
    ] {
        let error = anthropic::AnthropicConfig::with_dialect("invalid-key", dialect)
            .verify()
            .encode((), Mode::Unary)
            .expect_err("a public catalog cannot verify credentials");
        assert!(error.to_string().contains("no endpoint"));
    }
}

/// These entries were captured from the public Zen and Go catalogs. A full
/// completion cassette is unnecessary for the model-list metadata fallback.
#[test]
fn both_protocols_decode_catalog_entries_without_display_names() {
    for (openai, anthropic, model) in [
        (
            &openai::wire::OPENCODE_ZEN,
            &anthropic::wire::OPENCODE_ZEN,
            "claude-fable-5",
        ),
        (
            &openai::wire::OPENCODE_GO,
            &anthropic::wire::OPENCODE_GO,
            "minimax-m3",
        ),
    ] {
        let catalog = json!({
            "object": "list",
            "data": [{
                "id": model,
                "object": "model",
                "created": 1790724002_u64,
                "owned_by": "opencode"
            }]
        });
        let frames = [WireFrame::Text(catalog.to_string())];
        let chat = crate::test_utils::decode_reply(
            &openai::OpenAIConfig::with_key(openai, "test-key").models(),
            &None,
            Mode::Unary,
            frames.clone(),
            catalog.clone(),
        )
        .expect("OpenAI-format catalog decodes");
        let messages = crate::test_utils::decode_reply(
            &anthropic::AnthropicConfig::with_dialect("test-key", anthropic).models(),
            &None,
            Mode::Unary,
            frames,
            catalog,
        )
        .expect("Messages-format catalog decodes");
        for page in [chat, messages] {
            assert!(page.next.is_none());
            assert_eq!(page.models.len(), 1);
            let entry = page.models.iter().next().expect("the model is preserved");
            assert_eq!(entry.id, model);
            assert_eq!(entry.display_name(), model);
        }
    }
}
