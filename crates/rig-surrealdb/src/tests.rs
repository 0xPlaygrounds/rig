use super::{Mem, SurrealSearchFilter, SurrealVectorStore};
use rig_core::{
    embeddings::Embedding,
    vector_store::{
        InsertDocuments, VectorStoreIndex,
        request::{DynamicSearchFilter, Filter, SearchFilter, VectorSearchRequest},
    },
};
use serde_json::json;
use surrealdb::Surreal;

/// A deterministic embedding runtime: every text embeds to the first unit axis.
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
        let response = rig_core::embeddings::EmbeddingResponse {
            provider: "mock".into(),
            ..rig_core::embeddings::EmbeddingResponse::new(
                texts
                    .into_iter()
                    .map(|text| Embedding {
                        document: text,
                        vec: vec![1.0, 0.0, 0.0],
                    })
                    .collect(),
            )
        };
        rig_core::driver::Opening::ready(rig_core::driver::Opened::new(futures::stream::iter([
            Ok(rig_core::driver::Step::End(response)),
        ])))
    }
}

#[allow(clippy::panic)]
#[test]
fn filter_from_json_preserves_nested_values() {
    let filter = match SurrealSearchFilter::from_dynamic_filter(Filter::Eq(
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

#[tokio::test]
async fn filters_apply_before_ranking_and_preserve_boolean_grouping() -> anyhow::Result<()> {
    use surrealdb::types::Value;

    let db = Surreal::new::<Mem>(()).await?;
    db.use_ns("filters").use_db("filters").await?;
    let store = SurrealVectorStore::with_defaults(MockEmbeddingModel::model(), db);
    let quoted = r"O'Brien\notes";
    store
        .insert_documents(
            [
                ("closest", vec![1.0, 0.0, 0.0]),
                ("keep", vec![0.8, 0.6, 0.0]),
                (quoted, vec![0.0, 1.0, 0.0]),
            ]
            .into_iter()
            .map(|(name, vec)| {
                (
                    json!({"name": name}),
                    vec![Embedding {
                        document: name.to_owned(),
                        vec,
                    }],
                )
            })
            .collect(),
        )
        .await?;
    let eq = |text| SurrealSearchFilter::eq("embedded_text", Value::from_t(text));
    let cases = [
        (None, 3, None, vec!["closest", "keep", quoted]),
        (Some(eq("keep")), 1, None, vec!["keep"]),
        (Some(eq("missing")), 3, None, vec![]),
        (Some(eq(quoted)), 3, None, vec![quoted]),
        (Some(eq("closest").not()), 1, None, vec!["keep"]),
        (
            Some(eq("closest").or(eq("keep")).and(eq("closest").not())),
            3,
            None,
            vec!["keep"],
        ),
        (Some(eq("keep").or(eq(quoted))), 3, Some(0.9), vec![]),
        (
            Some(SurrealSearchFilter::from_dynamic_filter(Filter::Eq(
                "embedded_text".into(),
                json!("keep"),
            ))?),
            3,
            None,
            vec!["keep"],
        ),
    ];
    for (filter, samples, threshold, mut expected) in cases {
        let mut builder = VectorSearchRequest::builder()
            .query("query")
            .samples(samples);
        if let Some(filter) = filter {
            builder = builder.filter(filter);
        }
        if let Some(threshold) = threshold {
            builder = builder.threshold(threshold);
        }
        let req = builder.build();
        let documents = store.top_n::<serde_json::Value>(req.clone()).await?;
        let ids = store.top_n_ids(req).await?;
        let mut names = documents
            .iter()
            .filter_map(|row| row.document.get("name").and_then(serde_json::Value::as_str))
            .collect::<Vec<_>>();
        names.sort_unstable();
        expected.sort_unstable();
        anyhow::ensure!(
            names == expected && documents.len() == expected.len(),
            "expected {expected:?}, got {documents:?}"
        );
        let mut document_ids = documents.iter().map(|row| &row.id).collect::<Vec<_>>();
        let mut actual_ids = ids.iter().map(|row| &row.id).collect::<Vec<_>>();
        document_ids.sort_unstable();
        actual_ids.sort_unstable();
        anyhow::ensure!(
            document_ids == actual_ids,
            "document and ID searches disagree: {documents:?}, {ids:?}"
        );
    }
    Ok(())
}
