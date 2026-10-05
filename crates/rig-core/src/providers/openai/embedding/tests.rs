use super::*;
use serde_json::{Value, json};

/// OpenAI's own embeddings contract always reports usage; the permissive
/// [`CompatibleEmbeddingResponse`] is the shape for compatible servers that
/// omit it. A body without `usage` must not decode as the strict type.
#[test]
fn public_openai_embedding_response_requires_usage() {
    let body = r#"{
            "object": "list",
            "model": "text-embedding-3-small",
            "data": [{ "object": "embedding", "index": 0, "embedding": [0.1] }]
        }"#;

    assert!(serde_json::from_str::<EmbeddingResponse>(body).is_err());
    let compatible: CompatibleEmbeddingResponse =
        serde_json::from_str(body).expect("the compatible shape tolerates a missing usage");
    assert!(compatible.usage.is_none());
}

/// The width table the embeddings wire falls back to when a caller states no
/// dimension. `ada-002` shares `3-small`'s width but is excluded from the
/// request field by the wire, which is a separate rule.
#[test]
fn known_openai_models_resolve_their_default_width() {
    assert_eq!(
        model_dimensions_from_identifier(TEXT_EMBEDDING_3_LARGE),
        Some(3_072)
    );
    assert_eq!(
        model_dimensions_from_identifier(TEXT_EMBEDDING_3_SMALL),
        Some(1_536)
    );
    assert_eq!(
        model_dimensions_from_identifier(TEXT_EMBEDDING_ADA_002),
        Some(1_536)
    );
    assert_eq!(model_dimensions_from_identifier("some-other-model"), None);
}

fn usage(body: Value) -> crate::completion::Usage {
    serde_json::from_value::<Usage>(body)
        .expect("chat-completions usage should deserialize")
        .to_normalized()
}

/// Mistral's Voxtral models report audio beside `prompt_tokens`, so the
/// input count is the sum; a text turn's detail block is unaffected. The
/// numbers are a live Voxtral turn's, quoted verbatim.
#[test]
fn usage_counts_audio_tokens_reported_beside_the_prompt_as_input() {
    let voxtral = usage(json!({
        "prompt_audio_seconds": 0,
        "prompt_tokens": 6,
        "completion_tokens": 2,
        "total_tokens": 383,
        "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 375}
    }));
    assert_eq!(voxtral.input_tokens, Some(381));
    assert_eq!(voxtral.output_tokens, Some(2));
    assert_eq!(voxtral.total_tokens, Some(381 + 2));

    let text = usage(json!({
        "prompt_tokens": 19,
        "completion_tokens": 2,
        "total_tokens": 21,
        "prompt_tokens_details": {"cached_tokens": 0}
    }));
    assert_eq!(text.input_tokens, Some(19));
}
