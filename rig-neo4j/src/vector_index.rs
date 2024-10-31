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
use serde::Deserialize;

use crate::neo4j_to_rig_error;

pub struct Neo4jVectorIndex<M: EmbeddingModel> {
    graph: Graph,
    embedding_model: M,
    index_name: String,
    search_params: SearchParams,
}

impl<M: EmbeddingModel> Neo4jVectorIndex<M> {
    const BASE_VECTOR_SEARCH_QUERY: &str = "
    CALL db.index.vector.queryNodes($index_name, $num_candidates, $queryVector)
    YIELD node, score
    ";

    pub fn new(
        graph: Graph,
        embedding_model: M,
        index_name: &str,
        search_params: SearchParams,
    ) -> Self {
        Self {
            graph,
            embedding_model,
            index_name: index_name.to_string(),
            search_params,
        }
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

        Query::new(format!(
            "
            {}
            {}
            RETURN score, ID(node) as element_id {}
            ",
            Self::BASE_VECTOR_SEARCH_QUERY,
            where_clause,
            if return_node { ", node as node" } else { "" }
        ))
        .param("queryVector", prompt_embedding.vec)
        .param("num_candidates", n as i64)
        .param("index_name", self.index_name.clone())
    }

    pub async fn execute_and_collect<T: for<'a> Deserialize<'a>>(
        &self,
        query: Query,
    ) -> Result<Vec<T>, VectorStoreError> {
        Ok(self
            .graph
            .execute(query)
            .await
            .map_err(neo4j_to_rig_error)?
            .into_stream_as::<T>()
            .try_collect::<Vec<T>>()
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?)
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
        let prompt_embedding = self.embedding_model.embed_document(query).await?;
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
        let prompt_embedding = self.embedding_model.embed_document(query).await?;

        let query = self.build_vector_search_query(prompt_embedding, false, n);

        let rows = self.execute_and_collect::<RowResult>(query).await?;

        let results = rows
            .into_iter()
            .map(|row| (row.score, row.element_id.to_string()))
            .collect::<Vec<_>>();

        Ok(results)
    }
}

// ===============================
// Utilities to print results from a vector search
// ===============================


#[cfg(feature = "display")]
#[allow(dead_code)]
pub mod display {
    use std::fmt::Display;

    use term_size;
    #[derive(Debug)]
    pub struct SearchResult {
        pub title: String,
        pub id: String,
        pub description: String,
        pub score: f64,
    }

    pub struct SearchResults<'a>(pub &'a Vec<SearchResult>);

    impl<'a> Display for SearchResults<'a> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let width = term_size::dimensions().map(|(w, _)| w).unwrap_or(150);
            let title_width = 40;
            let id_width = 10;
            let description_width = width - title_width - id_width - 2; // 2 for spaces

            write!(
                f,
                "{:<title_width$} {:<id_width$} {:<description_width$}\n",
                "Title", "ID", "Description"
            )?;
            write!(f, "{}\n", "-".repeat(width))?;
            for result in self.0 {
                let wrapped_title = textwrap::fill(&result.title, title_width);
                let wrapped_description = textwrap::fill(&result.description, description_width);
                let title_lines: Vec<&str> = wrapped_title.lines().collect();
                let description_lines: Vec<&str> = wrapped_description.lines().collect();
                let max_lines = title_lines.len().max(description_lines.len());

                for i in 0..max_lines {
                    let title_line = title_lines.get(i).unwrap_or(&"");
                    let description_line = description_lines.get(i).unwrap_or(&"");
                    if i == 0 {
                        write!(
                            f,
                            "{:<title_width$} {:<id_width$} {:<description_width$}\n",
                            title_line, result.id, description_line
                        )?;
                    } else {
                        write!(
                            f,
                            "{:<title_width$} {:<id_width$} {:<description_width$}\n",
                            title_line, "", description_line
                        )?;
                    }
                }
            }
            Ok(())
        }
    }
}
