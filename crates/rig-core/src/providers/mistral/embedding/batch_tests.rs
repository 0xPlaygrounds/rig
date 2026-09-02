use super::*;

/// The chunk size `EmbeddingsBuilder` uses. Recording the 256-input
/// success it guards would commit ~5 MB of returned vectors to a fixture;
/// the cap it must stay under is pinned live in
/// `tests/providers/mistral/capability_edges.rs`.
#[test]
fn builder_chunks_at_mistrals_cap_not_openais() {
    use crate::client::EmbeddingsClient;
    use crate::embeddings::EmbeddingModel as _;
    assert_eq!(MAX_DOCUMENTS, 256);
    let client =
        super::super::Client::new_with("key", crate::test_utils::RecordingHttpClient::new(""))
            .expect("client");
    assert_eq!(
        client.embedding_model(super::MISTRAL_EMBED).max_documents(),
        256,
        "the generic model must take the provider's cap; the shared default is OpenAI's 1024, \
             which Mistral rejects"
    );
}

/// `mistral-embed` is fixed-width, and its width is what `ndims()` must
/// report — a model declaring 0 cannot size a vector store.
#[test]
fn mistral_embed_declares_its_width_without_requesting_it() {
    assert_eq!(Mistral::default_ndims(MISTRAL_EMBED), Some(1024));
    assert_eq!(Mistral::default_ndims(CODESTRAL_EMBED), None);

    // The declared width must not become a `dimensions` request field:
    // Mistral rejects that parameter for every model but Codestral.
    assert!(matches!(
        Mistral.embedding_dimensions(MISTRAL_EMBED, Some(1024)),
        Ok(None)
    ));
    // Any other value is still a genuine request for the parameter.
    assert!(
        Mistral
            .embedding_dimensions(MISTRAL_EMBED, Some(512))
            .is_err()
    );
}
