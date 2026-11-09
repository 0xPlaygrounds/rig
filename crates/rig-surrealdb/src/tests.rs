use super::{Mem, SurrealDistanceFunction, SurrealSearchFilter, SurrealVectorStore};
use rig_core::{
    embeddings::Embedding,
    vector_store::{
        InsertDocuments, VectorStoreIndex,
        request::{DynamicSearchFilter, Filter, VectorSearchRequest},
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
async fn distance_searches_rank_nearest_and_threshold_scores() -> anyhow::Result<()> {
    for (metric, scores) in [
        (
            SurrealDistanceFunction::Euclidean,
            [0.0, -0.5, -41.0_f64.sqrt()],
        ),
        (SurrealDistanceFunction::Hamming, [0.0, -1.0, -3.0]),
    ] {
        let db = Surreal::new::<Mem>(()).await?;
        db.use_ns("distances").use_db("distances").await?;
        let store = SurrealVectorStore::new(MockEmbeddingModel::model(), db, None, metric);
        store
            .insert_documents(
                [
                    ("exact", vec![1.0, 0.0, 0.0]),
                    ("near", vec![1.0, 0.5, 0.0]),
                    ("far", vec![4.0, 4.0, 4.0]),
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
        let [exact, near, far] = scores;
        for (samples, threshold, expected) in [
            (1, None, vec![("exact", exact)]),
            (2, None, vec![("exact", exact), ("near", near)]),
            (
                3,
                None,
                vec![("exact", exact), ("near", near), ("far", far)],
            ),
            (3, Some(near), vec![("exact", exact), ("near", near)]),
            (3, Some(0.0), vec![("exact", exact)]),
            (3, Some(0.1), vec![]),
        ] {
            assert_scored_search(&store, samples, threshold, &expected).await?;
        }
    }
    Ok(())
}

#[tokio::test]
async fn similarity_searches_keep_negative_scores_without_a_threshold() -> anyhow::Result<()> {
    let db = Surreal::new::<Mem>(()).await?;
    db.use_ns("similarities").use_db("similarities").await?;
    let store = SurrealVectorStore::with_defaults(MockEmbeddingModel::model(), db);
    store
        .insert_documents(
            [
                ("same", vec![1.0, 0.0, 0.0]),
                ("orthogonal", vec![0.0, 1.0, 0.0]),
                ("opposite", vec![-1.0, 0.0, 0.0]),
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
    assert_scored_search(
        &store,
        3,
        None,
        &[("same", 1.0), ("orthogonal", 0.0), ("opposite", -1.0)],
    )
    .await?;
    assert_scored_search(&store, 3, Some(0.0), &[("same", 1.0), ("orthogonal", 0.0)]).await?;
    Ok(())
}

async fn assert_scored_search<C: surrealdb::Connection>(
    store: &SurrealVectorStore<C>,
    samples: u64,
    threshold: Option<f64>,
    expected: &[(&str, f64)],
) -> anyhow::Result<()> {
    let mut builder = VectorSearchRequest::builder()
        .query("query")
        .samples(samples);
    if let Some(threshold) = threshold {
        builder = builder.threshold(threshold);
    }
    let req = builder.build();
    let documents = store.top_n::<serde_json::Value>(req.clone()).await?;
    let ids = store.top_n_ids(req).await?;
    anyhow::ensure!(
        documents.len() == expected.len() && ids.len() == expected.len(),
        "expected {expected:?}, got {documents:?}, {ids:?}"
    );
    for ((document, id), &(name, score)) in documents.iter().zip(&ids).zip(expected) {
        anyhow::ensure!(
            document
                .document
                .get("name")
                .and_then(serde_json::Value::as_str)
                == Some(name)
                && document.id == id.id
                && (document.score - score).abs() < 1e-10
                && (id.score - score).abs() < 1e-10,
            "expected {name} at {score}, got {document:?}, {id:?}"
        );
    }
    Ok(())
}
