use testcontainers::{
    core::{IntoContainerPort, Mount, WaitFor},
    runners::AsyncRunner,
    GenericImage, ImageExt,
};

use neo4rs::{ConfigBuilder, Graph};
use rig_neo4j::{
    vector_index::{IndexConfig, SearchParams},
    Neo4jClient, ToBoltType,
};
use rig::embeddings::EmbeddingsBuilder;

const BOLT_PORT: u16 = 7687;
const HTTP_PORT: u16 = 7474;

#[derive(Embed, Clone, serde::Deserialize, Debug, PartialEq)]
struct FakeDefinition {
    id: String,
    #[embed]
    definition: String,
}

#[tokio::test]
async fn vector_search_test() {
    let mount = Mount::volume_mount("data", "./data");
    // Setup a local MongoDB Atlas container for testing. NOTE: docker service must be running.
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
        .expect("Failed to start MongoDB Atlas container");

    let port = container.get_host_port_ipv4(BOLT_PORT).await.unwrap();

    let config = ConfigBuilder::default()
        .uri(format!("neo4j://localhost:{port}"))
        .build()
        .unwrap();

    let neo4j_client = Neo4jClient {
        graph: Graph::connect(config).await.unwrap(),
    };

    // Initialize OpenAI client
    let openai_client = openai::Client::from_env();

    // Select the embedding model and generate our embeddings
    let model = openai_client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);

    let embeddings = EmbeddingsBuilder::new(model.clone())
            .document(WordDefinition {
                id: "doc0".to_string(),
                definition: "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets".to_string(),
            })?
            .document(WordDefinition {
                id: "doc1".to_string(),
                definition: "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
            })?
            .document(WordDefinition {
                id: "doc2".to_string(),
                definition: "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.".to_string(),
            })?
            .build()
            .await?;
}
