#![recursion_limit = "256"]
//! FalkorDB vector storage and search for Rig, using the asynchronous `falkordb-rs` driver.
//!
//! Vector indexes are identified by their node label and embedding property.
//! Reuse the embedding model that populated the index for queries.
//!
//! ```no_run
//! use rig_core::providers::openai::{self, OpenAI};
//! use rig_core::vector_store::{VectorStoreIndex, request::VectorSearchRequest};
//! use rig_falkordb::{FalkorDBClient, vector_index::IndexConfig};
//!
//! # #[tokio::main]
//! # async fn main() -> anyhow::Result<()> {
//! let model = OpenAI::from_env()?.embedding(openai::TEXT_EMBEDDING_3_SMALL, None);
//! let client = FalkorDBClient::connect("falkor://localhost:6379", "documents").await?;
//! client.create_vector_index(IndexConfig::new("Document"), model.capabilities().ndims).await?;
//! let index = client.get_index(model, "Document", "embedding").await?;
//! let request = VectorSearchRequest::builder().query("graph databases").samples(3).build();
//! let results = index.top_n::<serde_json::Value>(request).await?;
//! # Ok(())
//! # }
//! ```

mod filter;
pub mod vector_index;

use falkordb::{
    AsyncGraph, EntityType, FalkorClientBuilder, FalkorIndex, FalkorValue, IndexStatus, IndexType,
};
use rig_core::{DynModel, operation::Embedding, vector_store::VectorStoreError};
use serde_json::Value;
use vector_index::{FalkorDBVectorIndex, IndexConfig};

pub use filter::FalkorDBSearchFilter;

/// Errors validating FalkorDB index configuration and embeddings.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(
        "Invalid Cypher identifier {0:?}: names must be nonempty and contain no backticks or NUL bytes"
    )]
    InvalidIdentifier(String),
    #[error("Invalid vector dimensions: {0}; expected a positive u32")]
    InvalidDimensions(usize),
    #[error("Embedding has {actual} dimensions; the index requires {expected}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("No node vector index exists for {label}.{property}")]
    IndexNotFound { label: String, property: String },
    #[error("The node vector index for {label}.{property} is not operational")]
    IndexNotReady { label: String, property: String },
    #[error("Vector index metadata is missing options for property {0:?}")]
    MissingIndexOptions(String),
    #[error("Integer {0} is outside FalkorDB's signed 64-bit range")]
    IntegerOutOfRange(serde_json::Number),
}

/// An asynchronous FalkorDB graph connection.
#[derive(Clone)]
pub struct FalkorDBClient {
    /// Driver handle for application-specific Cypher queries.
    pub graph: AsyncGraph,
}

impl FalkorDBClient {
    /// Wraps an existing asynchronous graph handle, preserving its connection settings.
    pub fn new(graph: AsyncGraph) -> Self {
        Self { graph }
    }

    /// Connects to `uri` and selects `graph_name`. Authentication is configured in the URI.
    pub async fn connect(uri: &str, graph_name: &str) -> Result<Self, VectorStoreError> {
        let client = FalkorClientBuilder::new_async()
            .with_connection_info(uri.try_into().map_err(VectorStoreError::datastore)?)
            .build()
            .await
            .map_err(VectorStoreError::datastore)?;
        Ok(Self::new(client.select_graph(graph_name)))
    }

    /// Creates a node vector index and waits until it is operational.
    ///
    /// `ndims` must match the embedding model. Driver errors, including readiness
    /// timeouts, are returned as datastore errors.
    pub async fn create_vector_index(
        &self,
        config: IndexConfig,
        ndims: usize,
    ) -> Result<(), VectorStoreError> {
        config.validate()?;
        let dimensions = u32::try_from(ndims)
            .ok()
            .filter(|value| *value > 0)
            .ok_or_else(|| VectorStoreError::datastore(Error::InvalidDimensions(ndims)))?;
        self.graph
            .clone()
            .create_node_vector_index_op(
                &config.node_label,
                &[&config.embedding_property],
                dimensions,
                config.similarity_function.into(),
            )
            .wait()
            .await
            .map_err(VectorStoreError::datastore)
    }

    /// Opens an operational node vector index, adopting its dimensions and metric.
    ///
    /// `model` must be the model used to generate stored embeddings. A known
    /// model dimension that differs from the index is rejected.
    pub async fn get_index(
        &self,
        model: impl Into<DynModel<Embedding>>,
        node_label: &str,
        embedding_property: &str,
    ) -> Result<FalkorDBVectorIndex, VectorStoreError> {
        let model = model.into();
        let indexes = self
            .graph
            .clone()
            .list_indices()
            .await
            .map_err(VectorStoreError::datastore)?;
        let (config, dimensions) = read_index(
            &indexes.data,
            node_label,
            embedding_property,
            model.capabilities().ndims,
        )?;
        Ok(FalkorDBVectorIndex::new(
            self.graph.clone(),
            model,
            config,
            dimensions,
        ))
    }
}

fn read_index(
    indexes: &[FalkorIndex],
    label: &str,
    property: &str,
    model_dimensions: usize,
) -> Result<(IndexConfig, usize), VectorStoreError> {
    IndexConfig::new(label)
        .embedding_property(property)
        .validate()?;
    let index = indexes
        .iter()
        .find(|index| {
            index.entity_type == EntityType::Node
                && index.index_label == label
                && index
                    .field_types
                    .get(property)
                    .is_some_and(|types| types.contains(&IndexType::Vector))
        })
        .ok_or_else(|| {
            VectorStoreError::datastore(Error::IndexNotFound {
                label: label.into(),
                property: property.into(),
            })
        })?;
    if index.status != IndexStatus::Active {
        return Err(VectorStoreError::datastore(Error::IndexNotReady {
            label: label.into(),
            property: property.into(),
        }));
    }
    #[derive(serde::Deserialize)]
    struct Options {
        dimension: usize,
        #[serde(rename = "similarityFunction")]
        similarity_function: vector_index::VectorSimilarityFunction,
    }
    let options: Options = index
        .options
        .get(property)
        .cloned()
        .ok_or_else(|| VectorStoreError::datastore(Error::MissingIndexOptions(property.into())))?
        .deserialize_into()
        .map_err(VectorStoreError::datastore)?;
    if options.dimension == 0 || u32::try_from(options.dimension).is_err() {
        return Err(VectorStoreError::datastore(Error::InvalidDimensions(
            options.dimension,
        )));
    }
    if model_dimensions != 0 && model_dimensions != options.dimension {
        return Err(VectorStoreError::datastore(Error::DimensionMismatch {
            expected: options.dimension,
            actual: model_dimensions,
        }));
    }
    Ok((
        IndexConfig::new(label)
            .embedding_property(property)
            .similarity_function(options.similarity_function),
        options.dimension,
    ))
}

fn quote_identifier(name: &str) -> Result<String, VectorStoreError> {
    if name.is_empty() || name.contains(['`', '\0']) {
        return Err(VectorStoreError::datastore(Error::InvalidIdentifier(
            name.into(),
        )));
    }
    Ok(format!("`{name}`"))
}

fn json_to_falkor(value: Value) -> Result<FalkorValue, VectorStoreError> {
    Ok(match value {
        Value::Null => FalkorValue::None,
        Value::Bool(value) => FalkorValue::Bool(value),
        Value::String(value) => FalkorValue::String(value),
        Value::Number(value) => {
            if let Some(integer) = value.as_i64() {
                FalkorValue::I64(integer)
            } else if value.is_u64() {
                return Err(VectorStoreError::datastore(Error::IntegerOutOfRange(value)));
            } else {
                FalkorValue::F64(serde_json::from_value(Value::Number(value))?)
            }
        }
        Value::Array(values) => FalkorValue::Array(
            values
                .into_iter()
                .map(json_to_falkor)
                .collect::<Result<_, _>>()?,
        ),
        Value::Object(values) => FalkorValue::Map(
            values
                .into_iter()
                .map(|(key, value)| Ok((key, json_to_falkor(value)?)))
                .collect::<Result<_, VectorStoreError>>()?,
        ),
    })
}

#[cfg(test)]
mod tests;
