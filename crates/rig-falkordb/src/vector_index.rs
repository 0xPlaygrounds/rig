//! Configuration, document insertion, and nearest-neighbor search over FalkorDB node vector indexes.

use std::collections::HashMap;

use falkordb::{AsyncGraph, FalkorValue, Row, VectorSimilarity};
use futures::TryStreamExt;
use rig_core::{
    DynModel, Embed,
    embeddings::Embedding,
    vector_store::{
        InsertDocuments, VectorSearchIdResult, VectorSearchResult, VectorStoreError,
        VectorStoreIndex, flatten_embedded, request::VectorSearchRequest,
    },
    wasm_compat::WasmCompatSend,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::{Error, FalkorDBSearchFilter, json_to_falkor, quote_identifier};

/// Label, embedding property, and distance metric of a node vector index.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IndexConfig {
    pub node_label: String,
    pub embedding_property: String,
    pub similarity_function: VectorSimilarityFunction,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self::new("Document")
    }
}

impl IndexConfig {
    /// Selects `node_label`, the `embedding` property, and cosine similarity.
    pub fn new(node_label: impl Into<String>) -> Self {
        Self {
            node_label: node_label.into(),
            embedding_property: "embedding".into(),
            similarity_function: VectorSimilarityFunction::Cosine,
        }
    }

    pub fn embedding_property(mut self, property: impl Into<String>) -> Self {
        self.embedding_property = property.into();
        self
    }

    pub fn similarity_function(mut self, function: VectorSimilarityFunction) -> Self {
        self.similarity_function = function;
        self
    }

    pub(crate) fn validate(&self) -> Result<(), VectorStoreError> {
        quote_identifier(&self.node_label)?;
        quote_identifier(&self.embedding_property)?;
        Ok(())
    }
}

/// Index distance metric, converted to a higher-is-better similarity on retrieval.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VectorSimilarityFunction {
    /// Returns `1 - cosine_distance`.
    #[default]
    Cosine,
    /// Returns `1 / (1 + euclidean_distance)`.
    Euclidean,
}

impl From<VectorSimilarityFunction> for VectorSimilarity {
    fn from(value: VectorSimilarityFunction) -> Self {
        match value {
            VectorSimilarityFunction::Cosine => Self::Cosine,
            VectorSimilarityFunction::Euclidean => Self::Euclidean,
        }
    }
}

/// An operational FalkorDB node vector index and its embedding model.
///
/// Obtain a handle with [`crate::FalkorDBClient::get_index`]. Result IDs are
/// FalkorDB node IDs, not document properties. Filters and thresholds apply
/// after nearest-neighbor selection and can return fewer than `samples` rows.
#[derive(Clone)]
pub struct FalkorDBVectorIndex {
    graph: AsyncGraph,
    embedding_model: DynModel<rig_core::operation::Embedding>,
    index_config: IndexConfig,
    dimensions: usize,
}

impl FalkorDBVectorIndex {
    pub(crate) fn new(
        graph: AsyncGraph,
        embedding_model: DynModel<rig_core::operation::Embedding>,
        index_config: IndexConfig,
        dimensions: usize,
    ) -> Self {
        Self {
            graph,
            embedding_model,
            index_config,
            dimensions,
        }
    }

    /// Returns the configuration read from the database.
    pub fn index_config(&self) -> &IndexConfig {
        &self.index_config
    }

    async fn run_search(
        &self,
        req: VectorSearchRequest<FalkorDBSearchFilter>,
        return_node: bool,
    ) -> Result<Vec<Row>, VectorStoreError> {
        let query = search_query(&self.index_config, &req, return_node)?;
        if req.samples() == 0 {
            return Ok(Vec::new());
        }
        let embedding = self.embedding_model.embed_text(req.query()).await?;
        validate_dimensions(&embedding, self.dimensions)?;
        self.graph
            .clone()
            .ro_query(&query.text)
            .with_params(query.params)
            .with_param("node_label", self.index_config.node_label.as_str())
            .with_param(
                "embedding_property",
                self.index_config.embedding_property.as_str(),
            )
            .with_param("samples", req.samples())
            .with_param("query_vector", embedding.vec)
            .execute()
            .await
            .map_err(VectorStoreError::datastore)?
            .data
            .try_collect()
            .await
            .map_err(VectorStoreError::datastore)
    }
}

struct SearchQuery {
    text: String,
    params: Vec<(String, FalkorValue)>,
}

fn search_query(
    config: &IndexConfig,
    req: &VectorSearchRequest<FalkorDBSearchFilter>,
    return_node: bool,
) -> Result<SearchQuery, VectorStoreError> {
    if req.samples() > i64::MAX as u64 {
        return Err(VectorStoreError::SamplesOutOfRange {
            requested: req.samples(),
            max: i64::MAX as u64,
        });
    }
    let mut params = Vec::new();
    let mut conditions = Vec::new();
    if let Some(threshold) = req.threshold() {
        conditions.push("similarity >= $threshold".to_owned());
        params.push(("threshold".into(), FalkorValue::F64(threshold)));
    }
    if let Some(filter) = req.filter() {
        let (condition, filter_params) = filter.compile()?;
        conditions.push(format!("({condition})"));
        params.extend(filter_params);
    }
    let predicate = if conditions.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", conditions.join(" AND "))
    };
    let similarity = match config.similarity_function {
        VectorSimilarityFunction::Cosine => "1.0 - score",
        VectorSimilarityFunction::Euclidean => "1.0 / (1.0 + score)",
    };
    let document = if return_node {
        format!(
            ", node {{.*, {}: null}} AS document",
            quote_identifier(&config.embedding_property)?
        )
    } else {
        String::new()
    };
    Ok(SearchQuery {
        text: format!(
            "CALL db.idx.vector.queryNodes($node_label, $embedding_property, $samples, vecf32($query_vector))
             YIELD node, score
             WITH node, {similarity} AS similarity
             {predicate}
             RETURN similarity AS score, ID(node) AS element_id{document}
             ORDER BY score DESC, element_id ASC"
        ),
        params,
    })
}

impl VectorStoreIndex for FalkorDBVectorIndex {
    type Filter = FalkorDBSearchFilter;

    /// Returns node properties with the embedding property set to null.
    async fn top_n<T: DeserializeOwned + WasmCompatSend>(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<VectorSearchResult<T>>, VectorStoreError> {
        #[derive(Deserialize)]
        struct SearchResult<T> {
            score: f64,
            element_id: i64,
            document: T,
        }
        self.run_search(req, true)
            .await?
            .into_iter()
            .map(|row| {
                let row: SearchResult<T> =
                    row.deserialize().map_err(VectorStoreError::datastore)?;
                Ok(VectorSearchResult {
                    score: row.score,
                    id: row.element_id.to_string(),
                    document: row.document,
                })
            })
            .collect()
    }

    /// Returns scores and node IDs without fetching document properties.
    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<VectorSearchIdResult>, VectorStoreError> {
        #[derive(Deserialize)]
        struct SearchResult {
            score: f64,
            element_id: i64,
        }
        self.run_search(req, false)
            .await?
            .into_iter()
            .map(|row| {
                let row: SearchResult = row.deserialize().map_err(VectorStoreError::datastore)?;
                Ok(VectorSearchIdResult {
                    score: row.score,
                    id: row.element_id.to_string(),
                })
            })
            .collect()
    }
}

fn validate_dimensions(embedding: &Embedding, expected: usize) -> Result<(), VectorStoreError> {
    if embedding.vec.len() != expected {
        return Err(VectorStoreError::datastore(Error::DimensionMismatch {
            expected,
            actual: embedding.vec.len(),
        }));
    }
    Ok(())
}

fn insert_query(config: &IndexConfig) -> Result<String, VectorStoreError> {
    Ok(format!(
        "UNWIND $items AS item CREATE (node:{}) SET node = item.properties SET node.{} = vecf32(item.embedding)",
        quote_identifier(&config.node_label)?,
        quote_identifier(&config.embedding_property)?,
    ))
}

fn prepare_documents<Doc: Serialize>(
    documents: Vec<(Doc, Vec<Embedding>)>,
    dimensions: usize,
    embedding_property: &str,
) -> Result<Vec<FalkorValue>, VectorStoreError> {
    flatten_embedded(documents, |document, embedding| {
        validate_dimensions(&embedding, dimensions)?;
        let mut properties = match json_to_falkor(document.clone())? {
            FalkorValue::Map(properties) => properties,
            value => HashMap::from([("document".into(), value)]),
        };
        properties.remove(embedding_property);
        properties.insert(
            "embedded_text".into(),
            FalkorValue::String(embedding.document),
        );
        Ok(FalkorValue::Map(HashMap::from([
            ("properties".into(), FalkorValue::Map(properties)),
            (
                "embedding".into(),
                FalkorValue::Array(embedding.vec.into_iter().map(FalkorValue::F64).collect()),
            ),
        ])))
    })
}

impl InsertDocuments for FalkorDBVectorIndex {
    /// Inserts one node per embedding, flattening object fields into properties.
    ///
    /// Non-object documents use the `document` property. The configured embedding
    /// property and `embedded_text` are reserved. Fields must be valid FalkorDB
    /// property values; nested objects are not supported by the database.
    async fn insert_documents<Doc: Serialize + Embed + WasmCompatSend>(
        &self,
        documents: Vec<(Doc, Vec<Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        let items = prepare_documents(
            documents,
            self.dimensions,
            &self.index_config.embedding_property,
        )?;
        if items.is_empty() {
            return Ok(());
        }
        self.graph
            .clone()
            .query(insert_query(&self.index_config)?)
            .with_param("items", items)
            .execute()
            .await
            .map_err(VectorStoreError::datastore)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
