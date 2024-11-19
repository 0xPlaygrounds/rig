//! A vector index for a Neo4j graph DB.
//!
//! This module provides a way to perform vector searches on a Neo4j graph DB.
//! It uses the [Neo4j vector index](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/)
//! to search for similar nodes based on a query.

use futures::TryStreamExt;
use neo4rs::{Graph, Query};
use rig::{
    embeddings::{Embedding, EmbeddingModel},
    vector_store::{VectorStoreError, VectorStoreIndex},
};
use serde::{Deserialize, Serialize};

use crate::{neo4j_to_rig_error, ToBoltType};

pub struct Neo4jVectorIndex<M: EmbeddingModel> {
    graph: Graph,
    embedding_model: M,
    search_params: SearchParams,
    index_config: IndexConfig,
}

/// The index name must be unique among both indexes and constraints.
/// A newly created index is not immediately available but is created in the background.
#[derive(Serialize, Deserialize, Clone)]
pub struct IndexConfig {
    index_name: String,
    embedding_properties: Vec<String>,
    similarity_function: VectorSimilarityFunction,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            index_name: "vector_index".to_string(),
            embedding_properties: vec!["embedding".to_string()],
            similarity_function: VectorSimilarityFunction::Cosine,
        }
    }
}

impl IndexConfig {
    pub fn new(index_name: impl Into<String>) -> Self {
        Self {
            index_name: index_name.into(),
            embedding_properties: vec!["embedding".to_string()],
            similarity_function: VectorSimilarityFunction::Cosine,
        }
    }

    pub fn index_name(mut self, index_name: impl Into<String>) -> Self {
        self.index_name = index_name.into();
        self
    }

    pub fn similarity_function(mut self, similarity_function: VectorSimilarityFunction) -> Self {
        self.similarity_function = similarity_function;
        self
    }

    pub fn embedding_properties(mut self, embedding_properties: Vec<String>) -> Self {
        self.embedding_properties = embedding_properties;
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

const BASE_VECTOR_SEARCH_QUERY: &str = "
    CALL db.index.vector.queryNodes($index_name, $num_candidates, $queryVector)
    YIELD node, score
    RETURN score, ID(node) as element_id
";

const GET_INDEX_EMBEDDING_PROPERTY: &str = "
    SHOW VECTOR INDEXES
    YIELD name, properties
    WHERE name=$index_name
    RETURN properties[0]
";

impl<M: EmbeddingModel> Neo4jVectorIndex<M> {
    pub async fn new(
        graph: Graph,
        embedding_model: M,
        index_config: IndexConfig,
        search_params: SearchParams,
    ) -> Result<Self, VectorStoreError> {
        let mut index = Self {
            graph,
            embedding_model,
            index_config,
            search_params,
        };
        index.index_config.embedding_properties = index.get_index_embedding_properties().await?;
        Ok(index)
    }

    pub async fn get_index_embedding_properties(
        &mut self,
    ) -> Result<Vec<String>, VectorStoreError> {
        let property_name = self
            .execute_and_collect::<Vec<String>>(
                neo4rs::query(GET_INDEX_EMBEDDING_PROPERTY)
                    .param("index_name", self.index_config.index_name.clone()),
            )
            .await?;
        Ok(property_name[0].clone())
    }

    /// Calls the `CREATE VECTOR INDEX` Neo4j query and waits for the index to be created.
    /// A newly created index is not immediately fully available but is created (i.e. data is indexed) in the background.
    ///
    /// ❗ If there is already an index targetting the same node label and property, the new index creation will fail.
    ///
    /// ### Arguments
    /// * `node_label` - The label of the nodes to which the index will be applied. For example, if your nodes have
    ///                  the label `:Movie`, pass "Movie" as the `node_label` parameter.
    /// * `embedding_prop_name` (optional) - The name of the property that contains the embedding vectors. Defaults to "embedding".
    ///
    pub async fn create_and_await_vector_index(
        &self,
        node_label: String,
        embedding_prop_name: Option<String>,
    ) -> Result<(), VectorStoreError> {
        // Create a vector index on our vector store
        tracing::info!("Creating vector index {} ...", self.index_config.index_name);

        let property = embedding_prop_name.unwrap_or("embedding".to_string());
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
            node_label, property
        );

        self.graph
            .run(
                neo4rs::query(&create_vector_index_query)
                    .param("index_name", self.index_config.index_name.clone())
                    .param(
                        "similarity_function",
                        self.index_config.similarity_function.clone().to_bolt_type(),
                    )
                    .param("dimensions", self.embedding_model.ndims() as i64),
            )
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        // Check if the index exists with db.awaitIndex(), the call timeouts if the index is not ready
        let index_exists = self
            .graph
            .run(
                neo4rs::query("CALL db.awaitIndex($index_name, 10000)")
                    .param("index_name", self.index_config.index_name.clone()),
            )
            .await;

        if index_exists.is_err() {
            tracing::warn!(
                "Index with name `{}` is not ready or could not be created.",
                self.index_config.index_name
            );
        }

        tracing::info!(
            "Index created successfully with name: {}",
            self.index_config.index_name
        );
        Ok(())
    }

    /// Build a Neo4j query that performs a vector search against an index.
    /// See [Query vector index](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/#query-vector-index) for more information.
    pub fn build_vector_search_query(
        &self,
        prompt_embedding: Embedding,
        return_node: bool,
        n: usize,
    ) -> Query {
        let where_clause = match &self.search_params.post_vector_search_filter {
            Some(filter) => format!("WHERE {}", filter),
            None => "".to_string(),
        };

        // Properties containing the embedding vectors are excluded from the returned node
        let embedding_properties = self
            .index_config
            .embedding_properties
            .iter()
            .map(|p| format!("{}:null", p))
            .collect::<Vec<String>>()
            .join(", ");

        Query::new(format!(
            "
            {}
            {}
            {}
            ",
            BASE_VECTOR_SEARCH_QUERY,
            where_clause,
            if return_node {
                format!(", node {{.*, {} }} as node", embedding_properties)
            } else {
                "".to_string()
            }
        ))
        .param("queryVector", prompt_embedding.vec)
        .param("num_candidates", n as i64)
        .param("index_name", self.index_config.index_name.clone())
    }

    pub async fn execute_and_collect<T: for<'a> Deserialize<'a>>(
        &self,
        query: Query,
    ) -> Result<Vec<T>, VectorStoreError> {
        self.graph
            .execute(query)
            .await
            .map_err(neo4j_to_rig_error)?
            .into_stream_as::<T>()
            .try_collect::<Vec<T>>()
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))
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

impl<M: EmbeddingModel + std::marker::Sync + Send> VectorStoreIndex for Neo4jVectorIndex<M> {
    /// Get the top n nodes and scores matching the query.
    ///
    /// #### Generic Type Parameters
    ///
    /// - `T`: The type used to deserialize the result from the Neo4j query.
    ///        It must implement the `serde::Deserialize` trait.
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
        query: &str,
        n: usize,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        let prompt_embedding = self.embedding_model.embed_text(query).await?;
        let query = self.build_vector_search_query(prompt_embedding, true, n);

        let rows = self.execute_and_collect::<RowResultNode<T>>(query).await?;

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
        query: &str,
        n: usize,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let prompt_embedding = self.embedding_model.embed_text(query).await?;

        let query = self.build_vector_search_query(prompt_embedding, false, n);

        let rows = self.execute_and_collect::<RowResult>(query).await?;

        let results = rows
            .into_iter()
            .map(|row| (row.score, row.element_id.to_string()))
            .collect::<Vec<_>>();

        Ok(results)
    }
}
