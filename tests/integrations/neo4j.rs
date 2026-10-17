use serde_json::json;
use testcontainers::{
    GenericImage, ImageExt,
    core::{IntoContainerPort, WaitFor},
    runners::AsyncRunner,
};

use crate::common::{
    WORD_DEFINITIONS, axis_embedding, mock_embeddings, openai_client, skip_if_docker_unavailable,
};
use futures::{StreamExt, TryStreamExt};
use rig::graph_store::{Edge, GraphIndex, GraphStore, KnowledgeGraph, Node};
use rig::neo4j::{Neo4jClient, ToBoltType, vector_index::IndexConfig};
use rig::vector_store::VectorStoreIndex;
use rig::vector_store::request::VectorSearchRequest;
use rig::{
    Embed,
    driver::Model,
    embeddings::{Embedding, EmbeddingsBuilder},
    providers::openai,
};

const BOLT_PORT: u16 = 7687;
const HTTP_PORT: u16 = 7474;

#[derive(Embed, Clone, serde::Deserialize, Debug)]
struct Word {
    id: String,
    #[embed]
    definition: String,
}

/// Starts a Neo4j container without auth and connects to it. Pinned like
/// `pgvector:pg17` / `scylla:5.4`: a floating `latest` defeats layer caching
/// and lets a rerun silently test a different database version.
async fn start_neo4j() -> (testcontainers::ContainerAsync<GenericImage>, Neo4jClient) {
    let container = GenericImage::new("neo4j", "5.26.29")
        .with_wait_for(WaitFor::Duration {
            length: std::time::Duration::from_secs(5),
        })
        .with_exposed_port(BOLT_PORT.tcp())
        .with_exposed_port(HTTP_PORT.tcp())
        .with_env_var("NEO4J_AUTH", "none")
        .start()
        .await
        .expect("Failed to start Neo 4J container");

    let port = container.get_host_port_ipv4(BOLT_PORT).await.expect("");
    let host = container.get_host().await.expect("").to_string();

    let neo4j_client = Neo4jClient::connect(&format!("neo4j://{host}:{port}"), "", "")
        .await
        .expect("");
    (container, neo4j_client)
}

#[tokio::test]
async fn vector_search_test() {
    if skip_if_docker_unavailable("vector_search_test") {
        return;
    }

    let (_container, neo4j_client) = start_neo4j().await;

    // Setup mock openai API
    let server = httpmock::MockServer::start();

    mock_embeddings(
        &server,
        json!({ "input": WORD_DEFINITIONS, "model": "text-embedding-ada-002" }),
        [
            vec![-0.001; 1536],
            vec![0.0023064255; 1536],
            vec![-0.001; 1536],
        ],
    );
    mock_embeddings(
        &server,
        json!({ "input": ["What is a glarb?"], "model": "text-embedding-ada-002" }),
        [vec![0.0024064254; 1536]],
    );

    let openai_client = openai_client(&server);

    // Select the embedding model and generate our embeddings
    let model = openai_client.embedding(openai::TEXT_EMBEDDING_ADA_002, None);

    let embeddings = create_embeddings(model.clone()).await;

    futures::stream::iter(embeddings)
        .map(|(doc, embeddings)| {
            let embedding = embeddings.first().expect("expected at least one embedding");
            neo4j_client.graph.run(
                neo4rs::query(
                    "
                        CREATE
                            (document:DocumentEmbeddings {
                                id: $id,
                                document: $document,
                                embedding: $embedding})
                        RETURN document",
                )
                .param("id", doc.id)
                // Here we use the first embedding but we could use any of them.
                // Neo4j only takes primitive types or arrays as properties.
                .param("embedding", embedding.vec.clone())
                .param("document", doc.definition.to_bolt_type()),
            )
        })
        .buffer_unordered(3)
        .try_collect::<Vec<_>>()
        .await
        .expect("");

    // Create a vector index on our vector store
    println!("Creating vector index...");
    neo4j_client
        .graph
        .run(neo4rs::query(
            "CREATE VECTOR INDEX vector_index IF NOT EXISTS
                FOR (m:DocumentEmbeddings)
                ON m.embedding
                OPTIONS { indexConfig: {
                    `vector.dimensions`: 1536,
                    `vector.similarity_function`: 'cosine'
                    }}",
        ))
        .await
        .expect("");

    // ℹ️ The index name must be unique among both indexes and constraints.
    // A newly created index is not immediately available but is created in the background.

    // Check if the index exists with db.awaitIndex(), the call timeouts if the index is not ready
    let index_exists = neo4j_client
        .graph
        .run(neo4rs::query("CALL db.awaitIndex('vector_index')"))
        .await;
    if index_exists.is_err() {
        println!("Index not ready, waiting for index...");
        std::thread::sleep(std::time::Duration::from_secs(5));
    }

    println!("Index exists: {index_exists:?}");

    // Create a vector index on our vector store
    // IMPORTANT: Reuse the same model that was used to generate the embeddings
    let index = neo4j_client
        .get_index(model, "vector_index")
        .await
        .expect("");

    let query = "What is a glarb?";
    let req = VectorSearchRequest::builder()
        .query(query)
        .samples(1)
        .build();

    // Query the index
    let results = index.top_n::<serde_json::Value>(req).await.expect("");

    let value = &results.first().expect("").document;

    assert_eq!(
        value,
        &serde_json::json!({
            "id": "doc1",
            "document": "Definition of a *glarb-glarb*: A glarb-glarb is an ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.",
            "embedding": serde_json::Value::Null
        })
    );
}

async fn create_embeddings(model: Model<openai::wire::Embeddings>) -> Vec<(Word, Vec<Embedding>)> {
    let words = WORD_DEFINITIONS
        .iter()
        .enumerate()
        .map(|(i, definition)| Word {
            id: format!("doc{i}"),
            definition: definition.to_string(),
        });

    EmbeddingsBuilder::new(model)
        .documents(words)
        .expect("")
        .build()
        .await
        .expect("")
}

#[tokio::test]
async fn graph_retrieval_test() {
    if skip_if_docker_unavailable("graph_retrieval_test") {
        return;
    }
    let (_container, client) = start_neo4j().await;
    let store = client.graph_store().await.expect("graph store");

    let lumen = Node::new("Lumen cell", "product", "A solid-state battery.");
    let mira = Node::new("Mira Okonkwo", "person", "A materials scientist.");
    store
        .insert_graph(KnowledgeGraph {
            nodes: vec![lumen.clone(), mira.clone()],
            edges: vec![
                Edge::new("Mira Okonkwo", "invented", "Lumen cell", ""),
                Edge::new("Mira Okonkwo", "works at", "Halvard Dynamics", ""),
                Edge::new("Halvard Dynamics", "headquartered in", "Tromso", ""),
            ],
        })
        .await
        .expect("insert graph");
    // Merging fills the bare Halvard node and the empty edge description.
    store
        .insert_graph(KnowledgeGraph {
            nodes: vec![Node::new("Halvard Dynamics", "company", "An energy firm.")],
            edges: vec![Edge::new(
                "Mira Okonkwo",
                "invented",
                "Lumen cell",
                "In 2019.",
            )],
        })
        .await
        .expect("merge graph");
    let embedded = [(lumen, 0), (mira, 1)]
        .into_iter()
        .map(|(node, axis)| {
            let embedding = Embedding {
                document: node.description.clone(),
                vec: axis_embedding(axis),
            };
            (node, vec![embedding])
        })
        .collect();
    store
        .insert_embedded_nodes(embedded)
        .await
        .expect("insert embeddings");
    client
        .create_vector_index(IndexConfig::new("entities"), "Entity", 1536)
        .await
        .expect("create index");

    let server = httpmock::MockServer::start();
    mock_embeddings(
        &server,
        json!({ "input": ["What does the Lumen cell's inventor do?"], "model": "text-embedding-ada-002" }),
        [axis_embedding(0)],
    );
    let model = openai_client(&server).embedding(openai::TEXT_EMBEDDING_ADA_002, None);
    let seeds = client.get_index(model, "entities").await.expect("index");
    let index = GraphIndex::new(seeds, store).hops(2);
    let req = VectorSearchRequest::builder()
        .query("What does the Lumen cell's inventor do?")
        .samples(1)
        .build();

    let results = index.top_n::<KnowledgeGraph>(req).await.expect("search");

    let result = results.first().expect("one result");
    assert_eq!(result.id, "lumen cell");
    assert_eq!(
        result
            .document
            .nodes
            .iter()
            .map(|node| (node.id.as_str(), node.kind.as_str()))
            .collect::<Vec<_>>(),
        [
            ("lumen cell", "product"),
            ("mira okonkwo", "person"),
            ("halvard dynamics", "company"),
        ]
    );
    assert_eq!(
        result
            .document
            .edges
            .iter()
            .map(|edge| (edge.relation.as_str(), edge.description.as_str()))
            .collect::<Vec<_>>(),
        [("invented", "In 2019."), ("works at", "")]
    );
}
