use super::*;

/// [`CompatibleEmbeddingResponse`] decodes a compatible server's reply that
/// omits usage.
#[test]
fn compatible_embedding_response_tolerates_a_missing_usage() {
    let body = r#"{
            "object": "list",
            "model": "text-embedding-3-small",
            "data": [{ "object": "embedding", "index": 0, "embedding": [0.1] }]
        }"#;

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

/// An embeddings reply's cached prompt tokens reach the normalized usage.
#[test]
fn embedding_usage_reports_cached_prompt_tokens() {
    let usage: EmbeddingUsage = serde_json::from_str(
        r#"{"prompt_tokens": 8, "total_tokens": 8, "prompt_tokens_details": {"cached_tokens": 5}}"#,
    )
    .expect("embeddings usage should deserialize");
    let normalized = crate::completion::Usage::from(&usage);
    assert_eq!(normalized.input_tokens, Some(8));
    assert_eq!(normalized.total_tokens, Some(8));
    assert_eq!(normalized.cached_input_tokens, Some(5));
    assert_eq!(normalized.output_tokens, Some(0));
}

/// Without a details block, Mistral's top-level `num_cached_tokens` is the
/// cached prompt count.
#[test]
fn embedding_usage_falls_back_to_num_cached_tokens() {
    let usage: EmbeddingUsage =
        serde_json::from_str(r#"{"prompt_tokens": 8, "total_tokens": 8, "num_cached_tokens": 3}"#)
            .expect("embeddings usage should deserialize");
    let normalized = crate::completion::Usage::from(&usage);
    assert_eq!(normalized.cached_input_tokens, Some(3));
}
