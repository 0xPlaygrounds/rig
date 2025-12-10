//! A vector index for a Neo4j graph DB.
//!
//! This module provides a way to perform vector searches on a Neo4j graph DB.
//! It uses the [Neo4j vector index](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/)
//! to search for similar nodes based on a query.

use neo4rs::{Graph, Query};
use rig::{
    embeddings::{Embedding, EmbeddingModel},
    vector_store::{
        VectorStoreError, VectorStoreIndex,
        request::{SearchFilter, VectorSearchRequest},
    },
};
use serde::{Deserialize, Serialize, de::Error};

use crate::{Neo4jClient, Neo4jSearchFilter};

pub struct Neo4jVectorIndex<M>
where
    M: EmbeddingModel,
{
    graph: Graph,
    embedding_model: M,
    index_config: IndexConfig,
}

/// The index name must be unique among both indexes and constraints.
/// A newly created index is not immediately available but is created in the background.
///
/// #### Default Values
/// - `index_name`: "vector_index"
/// - `embedding_property`: "embedding"
/// - `similarity_function`: VectorSimilarityFunction::Cosine
#[derive(Serialize, Deserialize, Clone)]
pub struct IndexConfig {
    pub index_name: String,
    pub embedding_property: String,
    pub similarity_function: VectorSimilarityFunction,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            index_name: "vector_index".to_string(),
            embedding_property: "embedding".to_string(),
            similarity_function: VectorSimilarityFunction::Cosine,
        }
    }
}

impl IndexConfig {
    pub fn new(index_name: impl Into<String>) -> Self {
        Self {
            index_name: index_name.into(),
            embedding_property: "embedding".to_string(),
            similarity_function: VectorSimilarityFunction::Cosine,
        }
    }

    pub fn index_name(mut self, index_name: &str) -> Self {
        self.index_name = index_name.to_string();
        self
    }

    pub fn similarity_function(mut self, similarity_function: VectorSimilarityFunction) -> Self {
        self.similarity_function = similarity_function;
        self
    }

    pub fn embedding_property(mut self, embedding_property: &str) -> Self {
        self.embedding_property = embedding_property.to_string();
        self
    }
}

/// Cosine is most commonly used, but Euclidean is also supported.
/// See [Neo4j vector similarity functions](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/#similarity-functions)
/// for more information.
#[derive(Default, Serialize, Deserialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum VectorSimilarityFunction {
    #[default]
    Cosine,
    Euclidean,
}

use std::str::FromStr;

impl FromStr for VectorSimilarityFunction {
    type Err = VectorStoreError;

    fn from_str(s: &str) -> Result<Self, VectorStoreError> {
        match s.to_lowercase().as_str() {
            "cosine" => Ok(VectorSimilarityFunction::Cosine),
            "euclidean" => Ok(VectorSimilarityFunction::Euclidean),
            _ => Err(VectorStoreError::JsonError(serde_json::Error::custom(
                format!("Invalid similarity function: {s}"),
            ))),
        }
    }
}

const BASE_VECTOR_SEARCH_QUERY: &str = "
    CALL db.index.vector.queryNodes($index_name, $num_candidates, $queryVector)
    YIELD node, score
";

impl<M> Neo4jVectorIndex<M>
where
    M: EmbeddingModel,
{
    pub fn new(graph: Graph, embedding_model: M, index_config: IndexConfig) -> Self {
        Self {
            graph,
            embedding_model,
            index_config,
        }
    }

    /// Build a Neo4j query that performs a vector search against an index.
    /// See [Query vector index](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/#query-vector-index) for more information.
    ///
    /// Query template:
    /// ```
    /// CALL db.index.vector.queryNodes($index_name, $num_candidates, $queryVector)
    /// YIELD node, score
    /// WHERE {where_clause}
    /// RETURN score, ID(node) as element_id, node {.*, embedding:null } as node
    /// ```
    pub fn build_vector_search_query(
        &self,
        prompt_embedding: Embedding,
        return_node: bool,
        req: &VectorSearchRequest<Neo4jSearchFilter>,
    ) -> Query {
        let where_clause = match (req.threshold(), req.filter()) {
            (Some(thresh), Some(filt)) => Neo4jSearchFilter::gt("distance".into(), thresh.into())
                .and(filt.clone())
                .render(),
            (Some(thresh), _) => Neo4jSearchFilter::gt("distance".into(), thresh.into()).render(),
            (_, Some(filt)) => filt.clone().render(),
            _ => String::new(),
        };

        // Propertiy containing the embedding vectors are excluded from the returned node
        let query = format!(
            "\
            {}\
            \t{}\n\
            \tRETURN score, ID(node) as element_id {}
            ",
            BASE_VECTOR_SEARCH_QUERY,
            where_clause,
            if return_node {
                format!(
                    ", node {{.*, {}:null }} as node",
                    self.index_config.embedding_property
                )
            } else {
                "".to_string()
            }
        );

        tracing::debug!("Query before params: {}", query);

        Query::new(query)
            .param("queryVector", prompt_embedding.vec)
            .param("num_candidates", req.samples() as i64)
            .param("index_name", self.index_config.index_name.clone())
    }
}

/// Search parameters for a vector search. Neo4j currently only supports post-vector-search filtering.
pub struct SearchParams {
    /// Sets the **post-filter** field of the search params. Uses a WHERE clause.
    /// See [Neo4j WHERE clause](https://neo4j.com/docs/cypher-manual/current/clauses/where/) for more information.
    post_vector_search_filter: Option<String>,
}

impl SearchParams {
    /// Initializes a new `SearchParams` with default values.
    pub fn new(filter: Option<String>) -> Self {
        Self {
            post_vector_search_filter: filter,
        }
    }

    pub fn filter(mut self, filter: String) -> Self {
        self.post_vector_search_filter = Some(filter);
        self
    }
}

impl Default for SearchParams {
    fn default() -> Self {
        Self::new(None)
    }
}

#[derive(Debug, Deserialize)]
pub struct RowResultNode<T> {
    score: f64,
    element_id: i64,
    node: T,
}

#[derive(Debug, Deserialize)]
struct RowResult {
    score: f64,
    element_id: i64,
}

impl<M> VectorStoreIndex for Neo4jVectorIndex<M>
where
    M: EmbeddingModel + std::marker::Sync + Send,
{
    type Filter = Neo4jSearchFilter;

    /// Get the top n nodes and scores matching the query.
    ///
    /// #### Generic Type Parameters
    ///
    /// - `T`: The type used to deserialize the result from the Neo4j query.
    ///   It must implement the `serde::Deserialize` trait.
    ///
    /// #### Returns
    ///
    /// Returns a `Result` containing a vector of tuples. Each tuple contains:
    /// - A `f64` representing the similarity score
    /// - A `String` representing the node ID
    /// - A value of type `T` representing the deserialized node data
    ///
    async fn top_n<T: for<'a> Deserialize<'a> + std::marker::Send>(
        &self,
        req: VectorSearchRequest<Neo4jSearchFilter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        let prompt_embedding = self.embedding_model.embed_text(req.query()).await?;
        let query = self.build_vector_search_query(prompt_embedding, true, &req);

        let rows = Neo4jClient::execute_and_collect::<RowResultNode<T>>(&self.graph, query).await?;

        let results = rows
            .into_iter()
            .map(|row| (row.score, row.element_id.to_string(), row.node))
            .collect::<Vec<_>>();

        Ok(results)
    }

    /// Get the top n ids and scores matching the query. Runs faster than top_n since it doesn't need to transfer and parse
    /// the full nodes and embeddings to the client.
    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<Neo4jSearchFilter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let prompt_embedding = self.embedding_model.embed_text(req.query()).await?;

        let query = self.build_vector_search_query(prompt_embedding, true, &req);

        let rows = Neo4jClient::execute_and_collect::<RowResult>(&self.graph, query).await?;

        let results = rows
            .into_iter()
            .map(|row| (row.score, row.element_id.to_string()))
            .collect::<Vec<_>>();

        Ok(results)
    }
}
