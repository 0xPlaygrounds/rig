use super::{Mem, SurrealSearchFilter, SurrealVectorStore};
use rig_core::{
    embeddings::Embedding,
    vector_store::{VectorStoreIndex, request::Filter},
};
use serde_json::json;
use surrealdb::Surreal;

#[derive(Clone)]
struct MockEmbeddingModel;

impl rig_core::wire::Wire for MockEmbeddingModel {
    type Op = rig_core::operation::Embedding;
    type Payload = Vec<String>;
    type Frame = Vec<String>;
    type Decoder = Self;

    fn name(&self) -> &str {
        "mock"
    }

    fn encode(
        &self,
        texts: Vec<String>,
        _mode: rig_core::wire::Mode,
    ) -> Result<Vec<String>, rig_core::error::EncodeError> {
        Ok(texts)
    }

    fn decoder(&self, _mode: rig_core::wire::Mode) -> Self {
        self.clone()
    }

    fn capabilities(&self) -> rig_core::operation::EmbeddingCapabilities {
        rig_core::operation::EmbeddingCapabilities::new(4, 3)
    }
}

impl rig_core::driver::Transport<MockEmbeddingModel> for MockEmbeddingModel {
    fn send(
        &self,
        texts: Vec<String>,
        _mode: rig_core::wire::Mode,
        _observation: Option<rig_core::driver::Observation>,
    ) -> Result<
        impl Future<Output = rig_core::driver::Opened<Vec<String>, Vec<String>>>
        + Send
        + 'static
        + use<>,
        rig_core::error::ProviderError,
    > {
        Ok(std::future::ready(rig_core::driver::Opened::new(
            futures::stream::iter([Ok(texts)]),
        )))
    }
}

impl rig_core::wire::Decoder<rig_core::operation::Embedding, Vec<String>> for MockEmbeddingModel {
    type Event = Vec<String>;

    fn classify(&self, texts: Vec<String>) -> rig_core::wire::WireEvent<Vec<String>> {
        rig_core::wire::WireEvent::Known(texts)
    }

    fn interpret(
        &mut self,
        texts: Vec<String>,
        out: &mut rig_core::wire::Output<rig_core::operation::Embedding>,
    ) {
        use rig_core::wire::Sink as _;
        out.push(Ok(rig_core::embeddings::EmbeddingResponse::new(
            texts
                .into_iter()
                .map(|text| Embedding {
                    document: text,
                    vec: vec![0.0, 0.0, 0.0],
                })
                .collect(),
            "mock",
        )));
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
    let vector_store = SurrealVectorStore::with_defaults(
        rig_core::Model::new(MockEmbeddingModel, MockEmbeddingModel),
        surreal,
    );

    assert_dyn(vector_store);
}
