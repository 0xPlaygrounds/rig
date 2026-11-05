use rig::{
    Embed,
    embeddings::{Embedding, EmbeddingsBuilder},
    falkordb::{
        FalkorDBClient, FalkorDBSearchFilter,
        vector_index::{IndexConfig, VectorSimilarityFunction},
    },
    providers::openai,
    vector_store::{
        InsertDocuments, VectorStoreError, VectorStoreIndex,
        request::{SearchFilter, VectorSearchRequest},
    },
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use testcontainers::{
    GenericImage,
    core::{IntoContainerPort, WaitFor},
    runners::AsyncRunner,
};

use crate::common::{WORD_DEFINITIONS, axis_embedding, mock_embeddings, openai_client};

#[derive(Clone, Debug, Embed, Serialize, Deserialize)]
struct Word {
    id: String,
    #[embed]
    definition: String,
    category: String,
    rating: i64,
}

#[tokio::test]
async fn vector_search_test() -> anyhow::Result<()> {
    let container = GenericImage::new("falkordb/falkordb", "v4.22.3")
        .with_exposed_port(6379.tcp())
        .with_wait_for(WaitFor::message_on_stdout("Ready to accept connections"))
        .start()
        .await?;
    let uri = format!(
        "falkor://{}:{}",
        container.get_host().await?,
        container.get_host_port_ipv4(6379).await?
    );
    let server = httpmock::MockServer::start_async().await;
    mock_embeddings(
        &server,
        json!({"input": WORD_DEFINITIONS, "model": "text-embedding-ada-002"}),
        [
            axis_embedding(0),
            axis_embedding(1),
            axis_embedding(0).into_iter().map(|value| -value).collect(),
        ],
    );
    mock_embeddings(
        &server,
        json!({"input": ["find"], "model": "text-embedding-ada-002"}),
        [axis_embedding(0)],
    );
    mock_embeddings(
        &server,
        json!({"input": ["bad dimensions"], "model": "text-embedding-ada-002"}),
        [vec![1.0, 0.0]],
    );
    let openai = openai_client(&server);
    let model = openai.embedding(openai::TEXT_EMBEDDING_ADA_002, None);
    for metric in [
        VectorSimilarityFunction::Cosine,
        VectorSimilarityFunction::Euclidean,
    ] {
        let client = FalkorDBClient::connect(&uri, &format!("rig_{metric:?}")).await?;
        let config = IndexConfig::new("Rig Documents")
            .embedding_property("vector value")
            .similarity_function(metric);
        client
            .create_vector_index(config.clone(), model.capabilities().ndims)
            .await?;
        let index = client
            .get_index(
                model.clone(),
                &config.node_label,
                &config.embedding_property,
            )
            .await?;
        anyhow::ensure!(index.index_config().similarity_function == metric);

        let words = WORD_DEFINITIONS
            .iter()
            .enumerate()
            .map(|(i, definition)| Word {
                id: format!("doc{i}"),
                definition: (*definition).into(),
                category: if i == 0 {
                    "a\\' OR true //".into()
                } else {
                    "other".into()
                },
                rating: i as i64,
            });
        let documents = EmbeddingsBuilder::new(model.clone())
            .documents(words)?
            .build()
            .await?;
        index.insert_documents(documents).await?;
        index.insert_documents::<Word>(vec![]).await?;

        let request = VectorSearchRequest::builder()
            .query("find")
            .samples(3)
            .build();
        let results = index.top_n::<Word>(request.clone()).await?;
        anyhow::ensure!(results.len() == 3);
        anyhow::ensure!(
            results
                .iter()
                .map(|result| result.document.id.as_str())
                .collect::<Vec<_>>()
                == ["doc0", "doc1", "doc2"]
        );
        anyhow::ensure!(
            results
                .first()
                .is_some_and(|result| (result.score - 1.0).abs() < 1e-6)
        );
        anyhow::ensure!(
            results
                .windows(2)
                .all(|pair| matches!(pair, [left, right] if left.score > right.score))
        );
        let ids = index.top_n_ids(request.clone()).await?;
        anyhow::ensure!(ids.len() == results.len());
        for (id, result) in ids.iter().zip(&results) {
            anyhow::ensure!(id.id == result.id && (id.score - result.score).abs() < 1e-6);
            id.id.parse::<i64>()?;
        }
        let raw = index.top_n::<Value>(request).await?;
        anyhow::ensure!(
            raw.first()
                .and_then(|result| result.document.get("vector value"))
                == Some(&Value::Null)
        );
        let invalid_shape = VectorSearchRequest::builder()
            .query("find")
            .samples(1)
            .build();
        anyhow::ensure!(index.top_n::<String>(invalid_shape).await.is_err());
        if metric == VectorSimilarityFunction::Cosine {
            let inclusive = VectorSearchRequest::builder()
                .query("find")
                .samples(3)
                .threshold(0.0)
                .build();
            anyhow::ensure!(index.top_n_ids(inclusive).await?.len() == 2);
        }

        let threshold = VectorSearchRequest::builder()
            .query("find")
            .samples(3)
            .threshold(0.9)
            .build();
        anyhow::ensure!(index.top_n_ids(threshold).await?.len() == 1);
        let filtered = VectorSearchRequest::builder()
            .query("find")
            .samples(3)
            .filter(
                FalkorDBSearchFilter::eq("category", json!("a\\' OR true //"))
                    .or(FalkorDBSearchFilter::eq("category", json!("other"))
                        .and(FalkorDBSearchFilter::gt("rating", json!(1)))),
            )
            .build();
        let filtered = index.top_n::<Word>(filtered).await?;
        anyhow::ensure!(
            filtered
                .iter()
                .map(|result| result.document.id.as_str())
                .collect::<Vec<_>>()
                == ["doc0", "doc2"]
        );
        let null = VectorSearchRequest::builder()
            .query("find")
            .samples(3)
            .filter(FalkorDBSearchFilter::eq("missing", Value::Null))
            .build();
        anyhow::ensure!(index.top_n_ids(null).await?.len() == 3);

        let empty = VectorSearchRequest::builder()
            .query("not mocked")
            .samples(0)
            .build();
        anyhow::ensure!(index.top_n_ids(empty.clone()).await?.is_empty());
        anyhow::ensure!(index.top_n::<Value>(empty).await?.is_empty());
        let overflow = VectorSearchRequest::builder()
            .query("not mocked")
            .samples(u64::MAX)
            .build();
        anyhow::ensure!(matches!(
            index.top_n_ids(overflow).await,
            Err(VectorStoreError::SamplesOutOfRange { .. })
        ));
        let invalid = VectorSearchRequest::builder()
            .query("bad dimensions")
            .samples(1)
            .build();
        anyhow::ensure!(index.top_n_ids(invalid).await.is_err());
        anyhow::ensure!(
            client
                .get_index(model.clone(), "Missing", "embedding")
                .await
                .is_err()
        );
        anyhow::ensure!(client.create_vector_index(config, 0).await.is_err());

        let word = Word {
            id: "multi".into(),
            definition: "multiple vectors".into(),
            category: "other".into(),
            rating: 3,
        };
        let wrong_width = vec![(
            word.clone(),
            vec![Embedding {
                document: word.definition.clone(),
                vec: vec![1.0],
            }],
        )];
        anyhow::ensure!(index.insert_documents(wrong_width).await.is_err());
        index
            .insert_documents(vec![(
                word.clone(),
                vec![
                    Embedding {
                        document: word.definition.clone(),
                        vec: axis_embedding(0),
                    },
                    Embedding {
                        document: word.definition.clone(),
                        vec: axis_embedding(1),
                    },
                ],
            )])
            .await?;
        let all = VectorSearchRequest::builder()
            .query("find")
            .samples(10)
            .build();
        anyhow::ensure!(index.top_n_ids(all).await?.len() == 5);
    }
    Ok(())
}
