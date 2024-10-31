//! A Rig vector store for Neo4j.
//!
//! This crate is a companion crate to the [rig-core crate](https://github.com/rig-ai/rig-core).
//! It provides a vector store implementation that uses Neo4j as the underlying datastore.
//!
//! See the [README](https://github.com/0xPlaygrounds/rig/tree/main/rig-neo4j) for more information.
//!
//! ## Prerequisites
//!
//! ### GenAI Plugin
//! The GenAI plugin is enabled by default in Neo4j Aura.
//!
//! The plugin needs to be installed on self-managed instances. This is done by moving the neo4j-genai.jar
//! file from /products to /plugins in the Neo4j home directory, or, if you are using Docker, by starting
//! the Docker container with the extra parameter `--env NEO4J_PLUGINS='["genai"]'`.
//!
//! For more information, see [Operations Manual → Configure plugins](https://neo4j.com/docs/operations-manual/current/plugins/configure/).
//!
//! ### Pre-existing Vector Index
//!
//! The [Neo4jVectorStoreIndex](Neo4jVectorIndex) struct is designed to work with a pre-existing
//! Neo4j vector index. You can create the index using the Neo4j browser or the Neo4j language.
//! See the [Neo4j documentation](https://neo4j.com/docs/genai/tutorials/embeddings-vector-indexes/setup/vector-index/)
//! for more information.
//!
//! The index name must be unique among both indexes and constraints.
//! A newly created index is not immediately available but is created in the background.
//!
//! ```cypher
//! CREATE VECTOR INDEX moviePlots
//!     FOR (m:Movie)
//!     ON m.embedding
//!     OPTIONS {indexConfig: {
//!         `vector.dimensions`: 1536,
//!         `vector.similarity_function`: 'cosine'
//!     }}
//! ```
//!
//! ## Simple example:
//!
//! More examples can be found in the [/examples](https://github.com/0xPlaygrounds/rig/tree/main/rig-neo4j/examples) folder.
//! ```
//! use rig::{providers::openai, vector_store::VectorStoreIndex};
//! use serde::Deserialize;
//! use std::env;
//!
//! #[tokio::main]
//! async fn main() {
//!     let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
//!     let openai_client = Client::new(&openai_api_key);
//!     let model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);
//!
//!     let client = Neo4jClient::from_config(
//!         ConfigBuilder::default()
//!             .uri(NEO4J_URI)
//!             .db(NEO4J_DB)
//!             .user(NEO4J_USERNAME)
//!             .password(NEO4J_PASSWORD)
//!             .build()
//!             .unwrap(),
//!     )
//!    .await
//!    .unwrap();
//!
//!     let index = client.index(
//!         model,
//!         IndexConfig::new("moviePlotsEmbedding"),
//!         SearchParams::default(),
//!     );
//!
//!     #[derive(Debug, Deserialize)]
//!     struct Movie {
//!         title: String,
//!         plot: String,
//!     }
//!     let results = index.top_n::<Movie>("Batman", 3).await?;
//!     println!("{:#?}", results);
//! }
//! ```
//!
//!
pub mod vector_index;
use neo4rs::*;
use rig::{embeddings::EmbeddingModel, vector_store::VectorStoreError};
use vector_index::{IndexConfig, Neo4jVectorIndex, SearchParams};

pub struct Neo4jClient {
    pub graph: Graph,
}

fn neo4j_to_rig_error(e: neo4rs::Error) -> VectorStoreError {
    VectorStoreError::DatastoreError(Box::new(e))
}

pub trait ToBoltType {
    fn to_bolt_type(&self) -> BoltType;
}

impl<T> ToBoltType for T
where
    T: serde::Serialize,
{
    fn to_bolt_type(&self) -> BoltType {
        match serde_json::to_value(self) {
            Ok(json_value) => match json_value {
                serde_json::Value::Null => BoltType::Null(BoltNull),
                serde_json::Value::Bool(b) => BoltType::Boolean(BoltBoolean::new(b)),
                serde_json::Value::Number(num) => {
                    if let Some(i) = num.as_i64() {
                        BoltType::Integer(BoltInteger::new(i))
                    } else if let Some(f) = num.as_f64() {
                        BoltType::Float(BoltFloat::new(f))
                    } else {
                        println!("Couldn't map to BoltType, will ignore.");
                        BoltType::Null(BoltNull) // Handle unexpected number type
                    }
                }
                serde_json::Value::String(s) => BoltType::String(BoltString::new(&s)),
                serde_json::Value::Array(arr) => BoltType::List(
                    arr.iter()
                        .map(|v| v.to_bolt_type())
                        .collect::<Vec<BoltType>>()
                        .into(),
                ),
                serde_json::Value::Object(obj) => {
                    let mut bolt_map = BoltMap::new();
                    for (k, v) in obj {
                        bolt_map.put(BoltString::new(&k), v.to_bolt_type());
                    }
                    BoltType::Map(bolt_map)
                }
            },
            Err(_) => {
                println!("Couldn't serialize to JSON, will ignore.");
                BoltType::Null(BoltNull) // Handle serialization error
            }
        }
    }
}

impl Neo4jClient {
    pub fn new(graph: Graph) -> Self {
        Self { graph }
    }

    pub async fn connect(uri: &str, user: &str, password: &str) -> Result<Self, VectorStoreError> {
        let graph = Graph::new(uri, user, password)
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;
        Ok(Self { graph })
    }

    pub async fn from_config(config: Config) -> Result<Self, VectorStoreError> {
        let graph = Graph::connect(config)
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;
        Ok(Self { graph })
    }

    /// Returns a `Neo4jVectorIndex` that mirrors an existing Neo4j Vector Index.
    ///
    /// An index (of type "vector") of the same name as `index_name` must already exist for the Neo4j database.
    /// See the Neo4j [documentation (Create vector index)](https://neo4j.com/docs/genai/tutorials/embeddings-vector-indexes/setup/vector-index/) for more information on creating indexes.
    ///
    /// ❗IMPORTANT: The index must be created with the same embedding model that will be used to query the index.
    pub fn index<M: EmbeddingModel>(
        &self,
        model: M,
        index_config: IndexConfig,
        search_params: SearchParams,
    ) -> Neo4jVectorIndex<M> {
        Neo4jVectorIndex::new(self.graph.clone(), model, index_config, search_params)
    }
}

#[allow(dead_code)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector_index::display::SearchResult;
    use neo4rs::ConfigBuilder;
    use rig::{
        providers::openai::{Client, TEXT_EMBEDDING_ADA_002},
        vector_store::VectorStoreIndex,
    };
    use serde::Deserialize;
    use std::env;

    const NEO4J_URI: &str = "neo4j+s://demo.neo4jlabs.com:7687";
    const NEO4J_DB: &str = "recommendations";
    const NEO4J_USERNAME: &str = "recommendations";
    const NEO4J_PASSWORD: &str = "recommendations";

    #[derive(Debug, Deserialize)]
    struct Movie {
        title: String,
        plot: String,
    }

    #[tokio::test]
    async fn test_connect() {
        let result = Neo4jClient::from_config(
            ConfigBuilder::default()
                .uri(NEO4J_URI)
                .db(NEO4J_DB)
                .user(NEO4J_USERNAME)
                .password(NEO4J_PASSWORD)
                .build()
                .unwrap(),
        )
        .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_vector_search_no_display() {
        let results = vector_search().await.unwrap();
        assert!(results.len() > 0);
    }

    #[cfg(feature = "display")]
    #[allow(dead_code)]
    #[tokio::test]
    async fn test_vector_search_display() {
        let results = vector_search().await.unwrap();
        let search_results: Vec<SearchResult> = results
            .into_iter()
            .map(|(score, id, doc)| vector_index::display::SearchResult {
                title: doc.title,
                id,
                description: doc.plot,
                score,
            })
            .collect();
        println!(
            "{:#}",
            vector_index::display::SearchResults(&search_results)
        );
        assert!(search_results.len() > 0);
    }

    async fn vector_search() -> Result<Vec<(f64, String, Movie)>, VectorStoreError> {
        let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
        let openai_client = Client::new(&openai_api_key);
        let model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);

        let client = Neo4jClient::from_config(
            ConfigBuilder::default()
                .uri(NEO4J_URI)
                .db(NEO4J_DB)
                .user(NEO4J_USERNAME)
                .password(NEO4J_PASSWORD)
                .build()
                .unwrap(),
        )
        .await
        .unwrap();

        let index = client.index(
            model,
            IndexConfig::new("moviePlotsEmbedding"),
            SearchParams::default(),
        );
        Ok(index.top_n::<Movie>("Batman", 3).await?)
    }
}
