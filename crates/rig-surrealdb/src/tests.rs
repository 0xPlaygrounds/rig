use super::{Mem, SurrealSearchFilter, SurrealVectorStore};
use rig_core::{
    embeddings::Embedding,
    vector_store::{VectorStoreIndex, request::Filter},
};
use serde_json::json;
use surrealdb::Surreal;

/// A deterministic embedding runtime: every text embeds to three zeros.
#[derive(Clone)]
struct MockEmbeddingModel;

impl MockEmbeddingModel {
    fn model() -> rig_core::Model<rig_core::driver::Local<rig_core::operation::Embedding>, Self> {
        rig_core::Model::new(
            rig_core::driver::Local::new("mock")
                .with_capabilities(rig_core::wire::Capabilities::embedding(4, 3)),
            Self,
        )
    }
}

impl rig_core::driver::Transport<rig_core::driver::Local<rig_core::operation::Embedding>>
    for MockEmbeddingModel
{
    fn send(
        &self,
        texts: Vec<String>,
        _exchange: rig_core::driver::Exchange,
    ) -> rig_core::driver::Opening<rig_core::driver::Step<rig_core::operation::Embedding>> {
        let response = rig_core::embeddings::EmbeddingResponse::new(
            texts
                .into_iter()
                .map(|text| Embedding {
                    document: text,
                    vec: vec![0.0, 0.0, 0.0],
                })
                .collect(),
            "mock",
        );
        rig_core::driver::Opening::ready(rig_core::driver::Opened::new(futures::stream::iter([
            Ok(rig_core::driver::Step::End(response)),
        ])))
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
    let vector_store = SurrealVectorStore::with_defaults(MockEmbeddingModel::model(), surreal);

    assert_dyn(vector_store);
}
