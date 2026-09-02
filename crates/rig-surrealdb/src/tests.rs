use super::{Mem, SurrealSearchFilter, SurrealVectorStore};
use rig_core::{
    embeddings::{Embedding, EmbeddingError, EmbeddingModel, EmbeddingResponse},
    vector_store::{VectorStoreIndex, request::Filter},
};
use serde_json::json;
use surrealdb::Surreal;

#[derive(Clone)]
struct MockEmbeddingModel;

impl EmbeddingModel for MockEmbeddingModel {
    fn max_documents(&self) -> usize {
        4
    }

    fn ndims(&self) -> usize {
        3
    }

    async fn embed_texts_response(
        &self,
        texts: impl IntoIterator<Item = String> + Send,
    ) -> Result<EmbeddingResponse, EmbeddingError> {
        Ok(EmbeddingResponse::new(
            texts
                .into_iter()
                .map(|text| Embedding {
                    document: text,
                    vec: vec![0.0, 0.0, 0.0],
                })
                .collect(),
            "mock",
        ))
    }
}

#[allow(clippy::panic)]
#[test]
fn filter_from_json_preserves_nested_values() {
    let filter = match SurrealSearchFilter::try_from(Filter::Eq(
        "metadata".to_string(),
        json!({
            "name": "rig",
            "flags": { "native": true },
            "tags": ["surreal", "json"]
        }),
    )) {
        Ok(filter) => filter,
        Err(err) => panic!("unexpected surreal filter conversion failure: {err}"),
    };

    let sql = filter.to_string();

    assert!(sql.starts_with("metadata = {"));
    assert!(sql.contains("name: 'rig'"));
    assert!(sql.contains("flags: { native: true }"));
    assert!(sql.contains("tags: ['surreal', 'json']"));
}

#[allow(clippy::panic)]
#[tokio::test]
async fn surreal_vector_store_supports_type_erased_queries() {
    fn assert_dyn<T: VectorStoreIndex + Send + Sync + 'static>(_: T) {}

    let surreal = match Surreal::new::<Mem>(()).await {
        Ok(surreal) => surreal,
        Err(err) => panic!("failed to create in-memory surreal client: {err}"),
    };
    let vector_store = SurrealVectorStore::with_defaults(MockEmbeddingModel, surreal);

    assert_dyn(vector_store);
}
