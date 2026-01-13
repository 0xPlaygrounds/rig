//! A Rig vector store for Neo4j.
//!
//! This crate is a companion crate to the [rig-core crate](https://github.com/0xPlaygrounds/rig).
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
//! For more information, see [Operations Manual → Configure plugins](https://neo4j.com/docs/upgrade-migration-guide/current/version-5/migration/install-and-configure/#_plugins).
//!
//! ### Pre-existing Vector Index
//!
//! The [Neo4jVectorStoreIndex](Neo4jVectorIndex) struct is designed to work with a pre-existing
//! Neo4j vector index. You can create the index using the Neo4j browser, a raw Cypher query, or the
//! [Neo4jClient::create_vector_index] method.
//! See the [Neo4j documentation](https://neo4j.com/docs/genai/tutorials/embeddings-vector-indexes/setup/vector-index/)
//! for more information.
//!
//! The index name must be unique among both indexes and constraints.
//! ❗A newly created index is not immediately available but is created in the background.
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
//! More examples can be found in the [/examples](https://github.com/0xPlaygrounds/rig/tree/main/rig-neo4j/examples) folder.
//! ```
//! use rig_neo4j::{vector_index::*, Neo4jClient};
//! use neo4rs::ConfigBuilder;
//! use rig::{providers::openai::*, vector_store::VectorStoreIndex};
//! use serde::Deserialize;
//! use std::env;
//!
//! #[tokio::main]
//! async fn main() {
//!     let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
//!     let openai_client = Client::new(&openai_api_key);
//!     let model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);
//!
//!
//!     const NEO4J_URI: &str = "neo4j+s://demo.neo4jlabs.com:7687";
//!     const NEO4J_DB: &str = "recommendations";
//!     const NEO4J_USERNAME: &str = "recommendations";
//!     const NEO4J_PASSWORD: &str = "recommendations";
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
//!     let index = client.get_index(
//!         model,
//!         "moviePlotsEmbedding"
//!     ).await.unwrap();
//!
//!     #[derive(Debug, Deserialize)]
//!     struct Movie {
//!         title: String,
//!         plot: String,
//!     }
//!     let results = index.top_n::<Movie>("Batman", 3).await.unwrap();
//!     println!("{:#?}", results);
//! }
//! ```
pub mod vector_index;
use std::str::FromStr;

use futures::TryStreamExt;
use neo4rs::*;
use rig::{
    embeddings::EmbeddingModel,
    vector_store::{VectorStoreError, request::SearchFilter},
};
use serde::{Deserialize, Serialize};
use vector_index::{IndexConfig, Neo4jVectorIndex, VectorSimilarityFunction};

pub struct Neo4jClient {
    pub graph: Graph,
}

fn neo4j_to_rig_error(e: neo4rs::Error) -> VectorStoreError {
    VectorStoreError::DatastoreError(Box::new(e))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Neo4jSearchFilter(String);

impl SearchFilter for Neo4jSearchFilter {
    type Value = serde_json::Value;

    fn eq(key: String, value: Self::Value) -> Self {
        Self(format!("n.{} = {}", key, serialize_cypher(value)))
    }

    fn gt(key: String, value: Self::Value) -> Self {
        Self(format!("n.{key} > {}", serialize_cypher(value)))
    }

    fn lt(key: String, value: Self::Value) -> Self {
        Self(format!("n.{key} < {}", serialize_cypher(value)))
    }

    fn and(self, rhs: Self) -> Self {
        Self(format!("({}) AND ({})", self.0, rhs.0))
    }

    fn or(self, rhs: Self) -> Self {
        Self(format!("({}) OR ({})", self.0, rhs.0))
    }
}

impl Neo4jSearchFilter {
    pub fn render(self) -> String {
        format!("WHERE {}", self.0)
    }

    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Self {
        Self(format!("NOT ({})", self.0))
    }

    pub fn gte(key: String, value: <Self as SearchFilter>::Value) -> Self {
        Self(format!("n.{key} >= {}", serialize_cypher(value)))
    }

    pub fn lte(key: String, value: <Self as SearchFilter>::Value) -> Self {
        Self(format!("n.{key} <= {}", serialize_cypher(value)))
    }

    pub fn member(key: String, values: Vec<<Self as SearchFilter>::Value>) -> Self {
        Self(format!(
            "n.{key} IN {}",
            serialize_cypher(serde_json::Value::Array(values))
        ))
    }

    // String matching

    /// Tests whether the value at `key` contains the pattern
    pub fn contains<S>(key: String, pattern: S) -> Self
    where
        S: AsRef<str>,
    {
        Self(format!(
            "n.{key} CONTAINS {}",
            serialize_cypher(serde_json::Value::String(pattern.as_ref().into()))
        ))
    }

    /// Tests whether the value at `key` starts with the pattern
    pub fn starts_with<S>(key: String, pattern: S) -> Self
    where
        S: AsRef<str>,
    {
        Self(format!(
            "n.{key} STARTS WITH {}",
            serialize_cypher(serde_json::Value::String(pattern.as_ref().into()))
        ))
    }

    /// Tests whether the value at `key` ends with the pattern
    pub fn ends_with<S>(key: String, pattern: S) -> Self
    where
        S: AsRef<str>,
    {
        Self(format!(
            "n.{key} ENDS WITH {}",
            serialize_cypher(serde_json::Value::String(pattern.as_ref().into()))
        ))
    }

    pub fn matches<S>(key: String, pattern: S) -> Self
    where
        S: AsRef<str>,
    {
        Self(format!(
            "n.{key} =~ {}",
            serialize_cypher(serde_json::Value::String(pattern.as_ref().into()))
        ))
    }
}

fn serialize_cypher(value: serde_json::Value) -> String {
    use serde_json::Value::*;
    match value {
        Null => "null".into(),
        Bool(b) => b.to_string(),
        Number(n) => n.to_string(),
        String(s) => format!("'{}'", s.replace('\'', "\\'")),
        Array(arr) => {
            format!(
                "[{}]",
                arr.into_iter()
                    .map(serialize_cypher)
                    .collect::<Vec<std::string::String>>()
                    .join(", ")
            )
        }
        Object(obj) => {
            format!(
                "{{{}}}",
                obj.into_iter()
                    .map(|(k, v)| format!("{k}: {}", serialize_cypher(v)))
                    .collect::<Vec<std::string::String>>()
                    .join(", ")
            )
        }
    }
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
    const GET_INDEX_QUERY: &'static str = "
    SHOW VECTOR INDEXES
    YIELD name, properties, options
    WHERE name=$index_name
    RETURN name, properties, options
    ";

    const SHOW_INDEXES_QUERY: &'static str = "SHOW VECTOR INDEXES YIELD name RETURN name";

    pub fn new(graph: Graph) -> Self {
        Self { graph }
    }

    pub async fn connect(uri: &str, user: &str, password: &str) -> Result<Self, VectorStoreError> {
        tracing::info!("Connecting to Neo4j DB at {} ...", uri);
        let graph = Graph::new(uri, user, password)
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;
        tracing::info!("Connected to Neo4j");
        Ok(Self { graph })
    }

    pub async fn from_config(config: Config) -> Result<Self, VectorStoreError> {
        let graph = Graph::connect(config)
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;
        Ok(Self { graph })
    }

    pub async fn execute_and_collect<T: for<'a> Deserialize<'a>>(
        graph: &Graph,
        query: Query,
    ) -> Result<Vec<T>, VectorStoreError> {
        graph
            .execute(query)
            .await
            .map_err(neo4j_to_rig_error)?
            .into_stream_as::<T>()
            .try_collect::<Vec<T>>()
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))
    }

    /// Returns a `Neo4jVectorIndex` that mirrors an existing Neo4j Vector Index.
    ///
    /// An index (of type "vector") of the same name as `index_name` must already exist for the Neo4j database.
    /// See the Neo4j [documentation (Create vector index)](https://neo4j.com/docs/genai/tutorials/embeddings-vector-indexes/setup/vector-index/) for more information on creating indexes.
    ///
    /// ❗IMPORTANT: The index must be created with the same embedding model that will be used to query the index.
    pub async fn get_index<M: EmbeddingModel>(
        &self,
        model: M,
        index_name: &str,
    ) -> Result<Neo4jVectorIndex<M>, VectorStoreError> {
        #[derive(Deserialize)]
        struct IndexInfo {
            name: String,
            properties: Vec<String>,
            options: IndexOptions,
        }

        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct IndexOptions {
            _index_provider: String,
            index_config: IndexConfigDetails,
        }

        #[derive(Deserialize)]
        struct IndexConfigDetails {
            #[serde(rename = "vector.dimensions")]
            vector_dimensions: i64,
            #[serde(rename = "vector.similarity_function")]
            vector_similarity_function: String,
        }

        let index_info = Self::execute_and_collect::<IndexInfo>(
            &self.graph,
            neo4rs::query(Self::GET_INDEX_QUERY).param("index_name", index_name),
        )
        .await?;

        let index_config = if let Some(index) = index_info.first() {
            if index.options.index_config.vector_dimensions != model.ndims() as i64 {
                tracing::warn!(
                    "The embedding vector dimensions of the existing Neo4j DB index ({}) do not match the provided model dimensions ({}). This may affect search performance.",
                    index.options.index_config.vector_dimensions,
                    model.ndims()
                );
            }
            IndexConfig::new(index.name.clone())
                .embedding_property(index.properties.first().unwrap())
                .similarity_function(VectorSimilarityFunction::from_str(
                    &index.options.index_config.vector_similarity_function,
                )?)
        } else {
            let indexes = Self::execute_and_collect::<String>(
                &self.graph,
                neo4rs::query(Self::SHOW_INDEXES_QUERY),
            )
            .await?;
            return Err(VectorStoreError::DatastoreError(Box::new(
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!(
                        "Index `{index_name}` not found in database. Available indexes: {indexes:?}"
                    ),
                ),
            )));
        };
        Ok(Neo4jVectorIndex::new(
            self.graph.clone(),
            model,
            index_config,
        ))
    }

    /// Calls the `CREATE VECTOR INDEX` Neo4j query and waits for the index to be created.
    /// A newly created index is not immediately fully available but is created (i.e. data is indexed) in the background.
    ///
    /// ❗ If there is already an index targeting the same node label and property, the new index creation will fail.
    ///
    /// ### Arguments
    /// * `index_name` - The name of the index to create.
    /// * `node_label` - The label of the nodes to which the index will be applied. For example, if your nodes have
    ///   the label `:Movie`, pass "Movie" as the `node_label` parameter.
    /// * `embedding_prop_name` (optional) - The name of the property that contains the embedding vectors. Defaults to "embedding".
    ///
    pub async fn create_vector_index(
        &self,
        index_config: IndexConfig,
        node_label: &str,
        model: &impl EmbeddingModel,
    ) -> Result<(), VectorStoreError> {
        // Create a vector index on our vector store
        tracing::info!("Creating vector index {} ...", index_config.index_name);

        let create_vector_index_query = format!(
            "
            CREATE VECTOR INDEX $index_name IF NOT EXISTS
            FOR (m:{})
            ON m.{}
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: $dimensions,
                    `vector.similarity_function`: $similarity_function
                }}
            }}",
            node_label, index_config.embedding_property
        );

        self.graph
            .run(
                neo4rs::query(&create_vector_index_query)
                    .param("index_name", index_config.index_name.clone())
                    .param(
                        "similarity_function",
                        index_config.similarity_function.clone().to_bolt_type(),
                    )
                    .param("dimensions", model.ndims() as i64),
            )
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        // Check if the index exists with db.awaitIndex(), the call timeouts if the index is not ready
        let index_exists = self
            .graph
            .run(
                neo4rs::query("CALL db.awaitIndex($index_name, 10000)")
                    .param("index_name", index_config.index_name.clone()),
            )
            .await;

        if index_exists.is_err() {
            tracing::warn!(
                "Index with name `{}` is not ready or could not be created.",
                index_config.index_name.clone()
            );
        }

        tracing::info!(
            "Index created successfully with name: {}",
            index_config.index_name
        );
        Ok(())
    }
}
