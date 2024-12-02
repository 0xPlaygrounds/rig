use testcontainers::{
    core::{IntoContainerPort, Mount, WaitFor},
    runners::AsyncRunner,
    GenericImage, ImageExt,
};

use futures::StreamExt;
use rig::vector_store::VectorStoreIndex;
use rig::{
    embeddings::{Embedding, EmbeddingsBuilder},
    providers::openai,
    Embed, OneOrMany,
};
use rig_neo4j::{vector_index::SearchParams, Neo4jClient, ToBoltType};

const BOLT_PORT: u16 = 7687;
const HTTP_PORT: u16 = 7474;

#[derive(Embed, Clone, serde::Deserialize, Debug)]
struct FakeDefinition {
    id: String,
    #[embed]
    definition: String,
}

#[tokio::test]
async fn vector_search_test() {
    // Setup a local qdrant container for testing. NOTE: docker service must be running.
    let container = GenericImage::new("neo4j", "latest")
        .with_wait_for(WaitFor::Duration {
            length: std::time::Duration::from_secs(5),
        })
        .with_exposed_port(BOLT_PORT.tcp())
        .with_exposed_port(HTTP_PORT.tcp())
        .with_mount(mount)
        .with_env_var("NEO4J_AUTH", "none")
        .start()
        .await
        .expect("Failed to start qdrant container");
}