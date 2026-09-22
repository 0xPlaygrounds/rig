//! Neo4j vector store for Rig.
//!
//! [`Neo4jVectorIndex`] queries a vector index that must already exist, created
//! externally or through [`Neo4jClient::create_vector_index`]. Neo4j builds new
//! indexes in the background, so they are not queryable immediately. Self-managed
//! instances need the GenAI plugin installed; Neo4j Aura enables it by default.
//! The crate [README](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-neo4j)
//! covers setup and further examples.
//!
//! ```no_run
//! use neo4rs::ConfigBuilder;
//! use rig_core::providers::openai::{self, wire::OpenAI};
//! use rig_core::vector_store::VectorStoreIndex;
//! use rig_core::vector_store::request::VectorSearchRequest;
//! use rig_neo4j::Neo4jClient;
//! use rig_reqwest::prelude::*;
//! use serde::Deserialize;
//!
//! #[derive(Debug, Deserialize)]
//! struct Movie {
//!     title: String,
//!     plot: String,
//! }
//!
//! #[tokio::main]
//! async fn main() -> Result<(), anyhow::Error> {
//!     let openai = OpenAI::from_env()?.bound()?;
//!     let model = openai.embedding(openai::TEXT_EMBEDDING_ADA_002, None);
//!
//!     let client = Neo4jClient::from_config(
//!         ConfigBuilder::default()
//!             .uri("neo4j+s://demo.neo4jlabs.com:7687")
//!             .db("recommendations")
//!             .user("recommendations")
//!             .password("recommendations")
//!             .build()?,
//!     )
//!     .await?;
//!
//!     // ❗IMPORTANT: reuse the model the stored embeddings were generated with.
//!     let index = client.get_index(model, "moviePlotsEmbedding").await?;
//!
//!     let req = VectorSearchRequest::builder()
//!         .query("Batman")
//!         .samples(3)
//!         .build();
//!
//!     let results = index.top_n::<Movie>(req).await?;
//!     println!("{results:#?}");
//!
//!     Ok(())
//! }
//! ```
pub mod vector_index;
use std::str::FromStr;

use futures::TryStreamExt;
use neo4rs::*;
use rig_core::{
    embeddings::EmbeddingModel,
    vector_store::{VectorStoreError, request::SearchFilter},
};
use serde::{Deserialize, Serialize};
use vector_index::{IndexConfig, Neo4jVectorIndex, VectorSimilarityFunction};

pub struct Neo4jClient {
    pub graph: Graph,
}

/// Cypher predicate over the matched node `n`.
///
/// Property keys are spliced into the query verbatim and string values are only
/// single-quote escaped, so neither should carry untrusted input.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Neo4jSearchFilter(String);

impl SearchFilter for Neo4jSearchFilter {
    type Value = serde_json::Value;

    fn eq(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self(format!("n.{} = {}", key.as_ref(), serialize_cypher(value)))
    }

    fn gt(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self(format!("n.{} > {}", key.as_ref(), serialize_cypher(value)))
    }

    fn lt(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self(format!("n.{} < {}", key.as_ref(), serialize_cypher(value)))
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

    pub fn not(self) -> Self {
        Self(format!("NOT ({})", self.0))
    }

    pub fn gte(key: &str, value: <Self as SearchFilter>::Value) -> Self {
        Self(format!("n.{key} >= {}", serialize_cypher(value)))
    }

    pub fn lte(key: &str, value: <Self as SearchFilter>::Value) -> Self {
        Self(format!("n.{key} <= {}", serialize_cypher(value)))
    }

    pub fn member(key: &str, values: Vec<<Self as SearchFilter>::Value>) -> Self {
        Self(format!(
            "n.{key} IN {}",
            serialize_cypher(serde_json::Value::Array(values))
        ))
    }

    /// Matches property values containing `pattern`.
    pub fn contains<S>(key: &str, pattern: S) -> Self
    where
        S: AsRef<str>,
    {
        Self(format!(
            "n.{key} CONTAINS {}",
            serialize_cypher(serde_json::Value::String(pattern.as_ref().into()))
        ))
    }

    /// Matches property values starting with `pattern`.
    pub fn starts_with<S>(key: &str, pattern: S) -> Self
    where
        S: AsRef<str>,
    {
        Self(format!(
            "n.{key} STARTS WITH {}",
            serialize_cypher(serde_json::Value::String(pattern.as_ref().into()))
        ))
    }

    /// Matches property values ending with `pattern`.
    pub fn ends_with<S>(key: &str, pattern: S) -> Self
    where
        S: AsRef<str>,
    {
        Self(format!(
            "n.{key} ENDS WITH {}",
            serialize_cypher(serde_json::Value::String(pattern.as_ref().into()))
        ))
    }

    /// Matches property values against the Cypher regular expression `pattern`.
    pub fn matches<S>(key: &str, pattern: S) -> Self
    where
        S: AsRef<str>,
    {
        Self(format!(
            "n.{key} =~ {}",
            serialize_cypher(serde_json::Value::String(pattern.as_ref().into()))
        ))
    }
}

/// Renders a JSON value as a Cypher literal, escaping single quotes in strings.
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

/// Conversion into a Bolt parameter value.
pub trait ToBoltType {
    /// Converts through JSON, yielding `BoltType::Null` for values that fail to
    /// serialize or fall outside Bolt's numeric range.
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
                        BoltType::Null(BoltNull)
                    }
                }
                serde_json::Value::String(s) => BoltType::String(BoltString::new(&s)),
                serde_json::Value::Array(arr) => BoltType::List(
                    arr.iter()
                        .map(ToBoltType::to_bolt_type)
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
                BoltType::Null(BoltNull)
            }
        }
    }
}

impl Neo4jClient {
    const GET_INDEX_QUERY: &'static str = "
    SHOW VECTOR INDEXES
    YIELD name, labelsOrTypes, properties, options
    WHERE name=$index_name
    RETURN name, labelsOrTypes, properties, options
    ";

    const SHOW_INDEXES_QUERY: &'static str = "SHOW VECTOR INDEXES YIELD name RETURN name";

    pub fn new(graph: Graph) -> Self {
        Self { graph }
    }

    pub async fn connect(uri: &str, user: &str, password: &str) -> Result<Self, VectorStoreError> {
        tracing::info!("Connecting to Neo4j DB at {} ...", uri);
        let graph = Graph::new(uri, user, password)
            .await
            .map_err(VectorStoreError::datastore)?;
        tracing::info!("Connected to Neo4j");
        Ok(Self { graph })
    }

    pub async fn from_config(config: Config) -> Result<Self, VectorStoreError> {
        let graph = Graph::connect(config)
            .await
            .map_err(VectorStoreError::datastore)?;
        Ok(Self { graph })
    }

    pub async fn execute_and_collect<T: for<'a> Deserialize<'a>>(
        graph: &Graph,
        query: Query,
    ) -> Result<Vec<T>, VectorStoreError> {
        graph
            .execute(query)
            .await
            .map_err(VectorStoreError::datastore)?
            .into_stream_as::<T>()
            .try_collect::<Vec<T>>()
            .await
            .map_err(VectorStoreError::datastore)
    }

    /// Returns an index handle mirroring the existing vector index `index_name`,
    /// adopting its embedding property, similarity function, and node label.
    ///
    /// `model` must be the model whose embeddings populated the index; a
    /// dimension mismatch is only warned about. Errors when the index does not
    /// exist or defines no property.
    pub async fn get_index<M: EmbeddingModel>(
        &self,
        model: M,
        index_name: &str,
    ) -> Result<Neo4jVectorIndex<M>, VectorStoreError> {
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct IndexInfo {
            name: String,
            labels_or_types: Vec<String>,
            properties: Vec<String>,
            options: IndexOptions,
        }

        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct IndexOptions {
            #[allow(dead_code)]
            index_provider: Option<String>,
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
            let embedding_property = index.properties.first().ok_or_else(|| {
                VectorStoreError::DatastoreError(
                    "Neo4j index is missing an embedding property".into(),
                )
            })?;
            let mut config = IndexConfig::new(index.name.clone())
                .embedding_property(embedding_property)
                .similarity_function(VectorSimilarityFunction::from_str(
                    &index.options.index_config.vector_similarity_function,
                )?);
            // Inserts must target the label the index is attached to.
            if let Some(label) = index.labels_or_types.first() {
                config = config.node_label(label);
            }
            config
        } else {
            let indexes = Self::execute_and_collect::<String>(
                &self.graph,
                neo4rs::query(Self::SHOW_INDEXES_QUERY),
            )
            .await?;
            return Err(VectorStoreError::datastore(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "Index `{index_name}` not found in database. Available indexes: {indexes:?}"
                ),
            )));
        };
        Ok(Neo4jVectorIndex::new(
            self.graph.clone(),
            model,
            index_config,
        ))
    }

    /// Creates a vector index over `node_label` if one of that name does not
    /// already exist, sized to `model`'s dimensions.
    ///
    /// `node_label` and the configured embedding property are spliced into the
    /// Cypher statement verbatim. Waiting for the index to come online is
    /// best effort: a timeout is logged as a warning rather than returned.
    pub async fn create_vector_index(
        &self,
        index_config: IndexConfig,
        node_label: &str,
        model: &impl EmbeddingModel,
    ) -> Result<(), VectorStoreError> {
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
            .map_err(VectorStoreError::datastore)?;

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
