//! The Messages wire, driven from recorded bytes and no socket.
//!
//! The unary and streamed bodies below are one recorded Messages turn, taken
//! both ways. Folding them through the same decoder is the property
//! this port exists for, and it is checked here without a transport so a
//! failure names the decoder rather than the harness.

use super::*;
use crate::wire::Operation;
use crate::wire::secret::tests::a_config_reloads_without_its_credential;

#[test]
fn a_serialized_provider_never_carries_its_key() {
    let provider =
        AnthropicConfig::new("sk-live-do-not-leak").with_beta("prompt-caching-2024-07-31");
    a_config_reloads_without_its_credential(&provider, "sk-live-do-not-leak", |provider| {
        &provider.api_key
    });

    let wire = provider.completion("claude-haiku-4-5");
    let json = serde_json::to_string(&wire).expect("the wire serializes");
    assert!(!json.contains("sk-live-do-not-leak"));
    let restored: Messages = serde_json::from_str(&json).expect("the wire round-trips");
    assert_eq!(restored.model, "claude-haiku-4-5");
    assert_eq!(restored.provider.dialect, ANTHROPIC);
    assert!(restored.provider.api_key.is_empty());
}

#[test]
fn a_dialect_round_trips_by_name_and_rejects_an_unknown_one() {
    for dialect in [ANTHROPIC, ZAI, MINIMAX, MOONSHOT, XIAOMIMIMO] {
        let json = serde_json::to_string(&dialect).expect("a dialect serializes");
        assert_eq!(json, format!("\"{}\"", dialect.name));
        let restored: Dialect = serde_json::from_str(&json).expect("a dialect round-trips");
        assert_eq!(restored, dialect);
    }
    assert!(serde_json::from_str::<Dialect>("\"not-a-provider\"").is_err());
}

#[test]
fn a_gateway_defaults_max_tokens_to_its_one_documented_ceiling() {
    assert_eq!(
        ANTHROPIC.default_max_tokens("claude-haiku-4-5"),
        Some(64_000)
    );
    assert_eq!(ANTHROPIC.default_max_tokens("some-unknown-model"), None);
    // A gateway documents one ceiling rather than per-model limits, so an
    // unrecognized model still gets a usable default.
    assert_eq!(ZAI.default_max_tokens("some-unknown-model"), Some(4096));
    // `strict_tool_schemas` is a const quirk of a const dialect, so the
    // gateway's disagreement with Anthropic is a compile-time fact, not a
    // runtime one.
    const _: () = assert!(!ZAI.quirks.strict_tool_schemas);
    const _: () = assert!(ANTHROPIC.quirks.strict_tool_schemas);
}

/// Each dialect reads images on its documented vision models only, in user
/// turns and tool results; no Messages model reads assistant images.
#[test]
fn each_dialect_reads_images_on_its_vision_models() {
    use crate::completion::ReplayTarget;

    for (dialect, model, images) in [
        (&ANTHROPIC, "claude-sonnet-4-6", true),
        (&ZAI, "glm-4.6", false),
        (&ZAI, "glm-4.5-air", false),
        (&ZAI, "glm-4.5v", true),
        (&ZAI, "glm-4.6v-flash", true),
        (&ZAI, "glm-5v-turbo", true),
        (&MOONSHOT, "kimi-k2-thinking", false),
        (&MOONSHOT, "kimi-k2-0905-preview", false),
        (&MOONSHOT, "moonshot-v1-8k", false),
        (&MOONSHOT, "moonshot-v1-8k-vision-preview", true),
        (&MOONSHOT, "kimi-k2.6", true),
        (&MOONSHOT, "kimi-k3", true),
        (&MINIMAX, "MiniMax-M2.7", false),
        (&MINIMAX, "MiniMax-M3", true),
        (&XIAOMIMIMO, "mimo-v2-flash", false),
        (&XIAOMIMIMO, "mimo-v2-pro", false),
        (&XIAOMIMIMO, "mimo-v2.5-pro", false),
        (&XIAOMIMIMO, "mimo-v2-omni", true),
        (&XIAOMIMIMO, "mimo-v2.5", true),
        (&XIAOMIMIMO, "mimo-v2.6-flash", true),
    ] {
        // The model the request addresses decides, not the wire's own.
        let accepts = AnthropicConfig::with_key(dialect, "sk-test")
            .completion("another-model")
            .accepts(model);
        assert_eq!(accepts.user_images, images, "{model}");
        assert_eq!(accepts.tool_result_images, images, "{model}");
        assert!(!accepts.assistant_images, "{model}");
        assert!(accepts.tools, "{model}");
    }
}

/// Every Messages-format dialect asks for tool input as it is written by
/// default, as pi does for every provider it does not know to reject it.
#[test]
fn every_dialect_defaults_to_eager_tool_input() {
    for quirks in [Quirks::anthropic(), Quirks::gateway()] {
        assert_eq!(quirks.tool_input_streaming, ToolInputStreaming::Eager);
    }
    for dialect in all() {
        assert_eq!(
            dialect.quirks.tool_input_streaming,
            ToolInputStreaming::Eager,
            "{}",
            dialect.name
        );
    }
    assert_eq!(Quirks::gateway().max_tokens, MaxTokens::Fixed(4096));
}

/// The path `config` posts a completion to.
fn messages_path(config: AnthropicConfig) -> String {
    let wire = config.completion("a-model");
    let mut request = CompletionRequest::new("hello");
    request.max_tokens = Some(64);
    let request = Completion::prepare(request, &wire.describe()).expect("the request prepares");
    let encoded = wire
        .encode(request, Mode::Unary)
        .expect("the request encodes");
    encoded.request.uri().path().to_owned()
}

/// Every preset posts to exactly one `/v1/messages`.
#[test]
fn every_dialect_posts_to_one_v1_messages() {
    for dialect in all() {
        let path = messages_path(AnthropicConfig::with_key(dialect, "sk-test"));
        assert!(path.ends_with("/v1/messages"), "{}: {path}", dialect.name);
        assert_eq!(path.matches("/v1").count(), 1, "{}: {path}", dialect.name);
    }
}

/// A dialect whose default base URL names the endpoint is normalized like
/// a caller-supplied one.
#[test]
fn a_dialect_base_url_is_normalized() {
    let dialect = compatible(
        "custom",
        "https://gateway.example/anthropic/v1/",
        "CUSTOM_API_KEY",
        None,
    );
    let config = AnthropicConfig::with_key(&dialect, "sk-test");
    assert_eq!(config.base_url, "https://gateway.example/anthropic");
    assert_eq!(messages_path(config), "/anthropic/v1/messages");
}
