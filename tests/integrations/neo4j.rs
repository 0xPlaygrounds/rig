use serde_json::json;
use testcontainers::{GenericImage, ImageExt, core::IntoContainerPort, runners::AsyncRunner};

use crate::common::{WORD_DEFINITIONS, mock_embeddings, openai_client};
use futures::{StreamExt, TryStreamExt};
use rig::neo4j::{Neo4jClient, Neo4jSearchFilter, ToBoltType};
use rig::vector_store::VectorStoreIndex;
use rig::vector_store::request::{SearchFilter, VectorSearchRequest};
use rig::{
    Embed,
    driver::Model,
    embeddings::{Embedding, EmbeddingsBuilder},
    providers::openai,
};

const BOLT_PORT: u16 = 7687;
const HTTP_PORT: u16 = 7474;
/// The longest the server or the vector index may take to be ready: a
/// bound on a hang, as each is waited for.
const READY_WITHIN: std::time::Duration = std::time::Duration::from_secs(180);
/// How often the server, or the vector index, is asked again until it is
/// ready.
const POLL: std::time::Duration = std::time::Duration::from_millis(500);

#[derive(Embed, Clone, serde::Deserialize, Debug)]
struct Word {
    id: String,
    #[embed]
    definition: String,
}

#[tokio::test]
async fn vector_search_test() {
    // Setup a local Neo 4J container for testing. NOTE: docker service must be running.
    // Pinned like `pgvector:pg17` / `scylla:5.4`: a floating `latest` defeats
    // layer caching and lets a rerun silently test a different database version.
    let container = GenericImage::new("neo4j", "5.26.29")
        .with_exposed_port(BOLT_PORT.tcp())
        .with_exposed_port(HTTP_PORT.tcp())
        .with_env_var("NEO4J_AUTH", "none")
        .start()
        .await
        .expect("Failed to start Neo 4J container");

    let port = container.get_host_port_ipv4(BOLT_PORT).await.expect("");
    let host = container.get_host().await.expect("").to_string();

    // The server is ready once it answers a query over Bolt.
    let deadline = tokio::time::Instant::now() + READY_WITHIN;
    let neo4j_client = loop {
        let ready = match Neo4jClient::connect(&format!("neo4j://{host}:{port}"), "", "").await {
            Ok(client) => match client.graph.run(neo4rs::query("RETURN 1")).await {
                Ok(()) => Ok(client),
                Err(error) => Err(error.to_string()),
            },
            Err(error) => Err(error.to_string()),
        };
        match ready {
            Ok(client) => break client,
            Err(error) => assert!(
                tokio::time::Instant::now() < deadline,
                "Neo4j never answered a query: {error}"
            ),
        }
        tokio::time::sleep(POLL).await;
    };

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

    // db.awaitIndex() returns once the index is online, or fails when it
    // times out; ask again until the deadline.
    let deadline = tokio::time::Instant::now() + READY_WITHIN;
    while let Err(error) = neo4j_client
        .graph
        .run(neo4rs::query("CALL db.awaitIndex('vector_index')"))
        .await
    {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the vector index never came online: {error}"
        );
        tokio::time::sleep(POLL).await;
    }

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

    let literals = string_filters_preserve_values(&neo4j_client.graph).await;
    assert!(literals.is_ok(), "{literals:?}");
}

async fn string_filters_preserve_values(
    graph: &neo4rs::Graph,
) -> Result<(), Box<dyn std::error::Error>> {
    for value in [
        r"C:\notes\temp.txt",
        r"literal\n\t\b\r\f\u0027\U00000027",
        r"trailing\",
        "quote' and \"double\"",
        r"backslash\'quote",
        "line\nbreak\ttab\rcarriage\u{0008}\u{000c}",
        "Unicode 😺 and backslash \\",
    ] {
        for predicate in [
            Neo4jSearchFilter::eq("value", json!(value)),
            Neo4jSearchFilter::member("value", vec![json!("absent"), json!(value)]),
            Neo4jSearchFilter::contains("value", value),
            Neo4jSearchFilter::starts_with("value", value),
            Neo4jSearchFilter::ends_with("value", value),
        ] {
            let query = format!(
                "WITH {{value: $value}} AS n {} RETURN n.value AS value",
                predicate.render()
            );
            let mut rows = graph
                .execute(neo4rs::query(&query).param("value", value))
                .await?;
            let row = rows
                .next()
                .await?
                .ok_or_else(|| std::io::Error::other(format!("filter dropped {value:?}")))?;
            let actual: String = row.get("value")?;
            if actual != value {
                return Err(std::io::Error::other(format!(
                    "filter changed {value:?} to {actual:?}"
                ))
                .into());
            }
        }
    }
    let predicate = Neo4jSearchFilter::matches("value", r"\d+\.\d+").render();
    let mut rows = graph
        .execute(neo4rs::query(&format!(
            "WITH {{value: '12.34'}} AS n {predicate} RETURN n.value AS value"
        )))
        .await?;
    if rows.next().await?.is_none() {
        return Err(std::io::Error::other("regex filter lost its backslashes").into());
    }
    Ok(())
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
