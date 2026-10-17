use rig::http_client::{Bytes, HttpClientExt, Request};
use rig::providers::openai;
use rig::vector_store::request::{SearchFilter, VectorSearchRequest};
use rig::vector_store::{InsertDocuments, VectorStoreIndex};
use rig::weaviate::{WeaviateFilter, WeaviateVectorStore};
use rig::{Embed, embeddings::EmbeddingsBuilder};
use serde::{Deserialize, Serialize};
use serde_json::json;
use testcontainers::{
    ContainerAsync, GenericImage, ImageExt,
    core::{IntoContainerPort, WaitFor},
    runners::AsyncRunner,
};

use crate::common::{
    WORD_DEFINITIONS, axis_embedding, mock_embeddings, openai_client, skip_if_docker_unavailable,
};

const WEAVIATE_PORT: u16 = 8080;
const QUERY: &str = "What is a glarb-glarb?";

#[derive(Embed, Clone, Serialize, Deserialize, Debug, PartialEq)]
struct Word {
    id: String,
    #[embed]
    definition: String,
}

#[tokio::test]
async fn vector_search_test() {
    if skip_if_docker_unavailable("vector_search_test") {
        return;
    }

    let container = start_container().await;
    let host = container.get_host().await.unwrap();
    let port = container.get_host_port_ipv4(WEAVIATE_PORT).await.unwrap();
    let url = format!("http://{host}:{port}");

    // Each definition lies on its own axis and the query on glarb-glarb's, so
    // cosine similarities are exactly 1 for glarb-glarb and 0 for the others.
    let server = httpmock::MockServer::start();
    mock_embeddings(
        &server,
        json!({ "input": WORD_DEFINITIONS, "model": "text-embedding-ada-002" }),
        (0..3).map(axis_embedding),
    );
    mock_embeddings(
        &server,
        json!({ "input": [QUERY], "model": "text-embedding-ada-002" }),
        [axis_embedding(1)],
    );
    let model = openai_client(&server).embedding(openai::TEXT_EMBEDDING_ADA_002, None);

    let words = WORD_DEFINITIONS
        .iter()
        .enumerate()
        .map(|(i, definition)| Word {
            id: format!("doc{i}"),
            definition: definition.to_string(),
        });
    let documents = EmbeddingsBuilder::new(model.clone())
        .documents(words)
        .unwrap()
        .build()
        .await
        .unwrap();

    // A collection the server's auto-schema creates on first insert, with
    // one unnamed vector.
    let store = WeaviateVectorStore::new(model.clone(), url.clone(), "word").unwrap();
    store.insert_documents(documents.clone()).await.unwrap();
    assert_searches(&store).await;

    // A collection declared up front with a named vector.
    create_collection_with_named_vector(&url, "NamedWord", "definition").await;
    let named = WeaviateVectorStore::new(model, url, "NamedWord")
        .unwrap()
        .target_vector("definition");
    named.insert_documents(documents).await.unwrap();
    assert_searches(&named).await;
}

async fn assert_searches(store: &WeaviateVectorStore) {
    let best = store
        .top_n::<Word>(
            VectorSearchRequest::builder()
                .query(QUERY)
                .samples(1)
                .build(),
        )
        .await
        .unwrap();
    assert_eq!(best.len(), 1);
    let best = best.first().unwrap();
    assert_eq!(best.document.id, "doc1");
    assert!((best.score - 1.0).abs() < 1e-4, "score {}", best.score);

    let ids = store
        .top_n_ids(
            VectorSearchRequest::builder()
                .query(QUERY)
                .samples(3)
                .build(),
        )
        .await
        .unwrap();
    assert_eq!(ids.len(), 3);
    assert_eq!(
        ids.first().map(|hit| hit.id.as_str()),
        Some(best.id.as_str())
    );

    // A threshold is a minimum similarity: only glarb-glarb scores above 0.5.
    let thresholded = store
        .top_n_ids(
            VectorSearchRequest::builder()
                .query(QUERY)
                .samples(3)
                .threshold(0.5)
                .build(),
        )
        .await
        .unwrap();
    assert_eq!(thresholded.len(), 1, "{thresholded:?}");

    let filtered = store
        .top_n::<Word>(
            VectorSearchRequest::builder()
                .query(QUERY)
                .samples(3)
                .filter(
                    WeaviateFilter::like("document", "*flurbo*")
                        .or(WeaviateFilter::like("document", "*linglingdong*"))
                        .and(WeaviateFilter::like("document", "*linglingdong*").not()),
                )
                .build(),
        )
        .await
        .unwrap();
    let ids: Vec<_> = filtered
        .iter()
        .map(|hit| hit.document.id.as_str())
        .collect();
    assert_eq!(ids, ["doc0"]);
}

async fn create_collection_with_named_vector(url: &str, collection: &str, vector: &str) {
    let body = json!({
        "class": collection,
        "vectorConfig": { vector: { "vectorizer": { "none": {} }, "vectorIndexType": "hnsw" } },
    });
    let request = Request::post(format!("{url}/v1/schema"))
        .header("content-type", "application/json")
        .body(serde_json::to_vec(&body).unwrap())
        .unwrap();
    rig::rig_reqwest::shared()
        .send::<_, Bytes>(request)
        .await
        .expect("Weaviate creates the collection");
}

async fn start_container() -> ContainerAsync<GenericImage> {
    // Pinned so a rerun tests the same server version.
    GenericImage::new("semitechnologies/weaviate", "1.40.1")
        .with_wait_for(WaitFor::message_on_either_std("Serving weaviate"))
        .with_exposed_port(WEAVIATE_PORT.tcp())
        .with_env_var("AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED", "true")
        .with_env_var("PERSISTENCE_DATA_PATH", "/var/lib/weaviate")
        .with_env_var("DEFAULT_VECTORIZER_MODULE", "none")
        .with_env_var("CLUSTER_HOSTNAME", "node1")
        // CI runners' disks are often over Weaviate's default 90% read-only
        // watermark, which would reject every insert.
        .with_env_var("DISK_USE_WARNING_PERCENTAGE", "100")
        .with_env_var("DISK_USE_READONLY_PERCENTAGE", "100")
        .start()
        .await
        .expect("Failed to start Weaviate container")
}
